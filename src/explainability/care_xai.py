"""CARE-XAI: Cross-Modal/Multi-Task Attribution and Reliability Explanation Framework.

Implements:
1. Transformer Attention Rollout (Chefer et al. / Abnar & Zuidema)
2. Integrated Gradients attribution
3. Quantitative Faithfulness Benchmarks:
   - Pointing Game (Localization Precision against radiologist ground-truth masks)
   - Deletion Curves (Pixel-flipping test of explanation faithfulness)
4. Counterfactual Lesion Masking
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class CareXAI:
    """Explainability and Faithfulness evaluation suite for X-R-MiTNet."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def compute_attention_rollout(
        self, image: torch.Tensor, discard_ratio: float = 0.9, head_fusion: str = "mean"
    ) -> np.ndarray:
        """Computes Transformer Attention Rollout from [CLS] to all spatial patches.

        Args:
            image: (1, 1, 224, 224) input MRI scan tensor
            discard_ratio: fraction of lowest attention weights to prune for noise reduction
            head_fusion: 'mean', 'max', or 'min' across attention heads
        Returns:
            heatmap: (224, 224) normalized attribution map in [0, 1]
        """
        image = image.to(self.device)
        with torch.no_grad():
            out = self.model(image, return_attention=True)
            attn_list = out["attention_weights"]  # List of L tensors (1, num_heads, 197, 197)

        # Fuse heads across layers
        num_layers = len(attn_list)
        fused_attns = []

        for attn in attn_list:
            if head_fusion == "mean":
                fused = attn.mean(dim=1)  # (1, 197, 197)
            elif head_fusion == "max":
                fused = attn.max(dim=1)[0]
            else:
                fused = attn.min(dim=1)[0]

            # Discard lowest weights to eliminate background noise
            if discard_ratio > 0.0:
                flat = fused.view(fused.size(0), -1)
                num_discard = int(flat.size(-1) * discard_ratio)
                indices = flat.argsort(dim=-1)[:, :num_discard]
                flat.scatter_(1, indices, 0.0)
                fused = flat.view(fused.shape)

            # Add identity matrix to account for residual connections (skip connections)
            I = torch.eye(fused.size(-1), device=fused.device).unsqueeze(0)
            fused_aug = 0.5 * fused + 0.5 * I
            fused_aug = fused_aug / fused_aug.sum(dim=-1, keepdim=True)
            fused_attns.append(fused_aug)

        # Matrix multiply along depth
        rollout = fused_attns[0]
        for l in range(1, num_layers):
            rollout = torch.bmm(fused_attns[l], rollout)

        # Extract [CLS] token attention to all spatial patches (indices 1 to 196)
        cls_attn = rollout[0, 0, 1:]  # (196,)
        grid_size = int(np.sqrt(cls_attn.size(0)))
        heatmap_2d = cls_attn.view(1, 1, grid_size, grid_size)

        # Upsample 14x14 attention grid to full 224x224 image resolution
        heatmap_full = F.interpolate(heatmap_2d, size=(224, 224), mode="bicubic", align_corners=False)
        heatmap_np = heatmap_full.squeeze().cpu().numpy()

        # Min-max normalization
        h_min, h_max = heatmap_np.min(), heatmap_np.max()
        if h_max - h_min > 1e-8:
            heatmap_np = (heatmap_np - h_min) / (h_max - h_min)
        else:
            heatmap_np = np.zeros_like(heatmap_np)

        return heatmap_np

    def compute_integrated_gradients(
        self, image: torch.Tensor, target_class: int, steps: int = 20
    ) -> np.ndarray:
        """Calculates Integrated Gradients attribution along path from blank baseline to input.

        Args:
            image: (1, 1, 224, 224) input MRI tensor
            target_class: integer class index 0..3
            steps: number of Riemann approximation steps
        Returns:
            ig_heatmap: (224, 224) normalized gradient attribution in [0, 1]
        """
        image = image.to(self.device)
        baseline = torch.zeros_like(image).to(self.device)

        # Scaled images along the straight line path
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        total_grads = torch.zeros_like(image)

        for alpha in alphas:
            scaled_input = baseline + alpha * (image - baseline)
            scaled_input.requires_grad_(True)

            out = self.model(scaled_input, return_attention=False)
            logits = out["logits"]
            score = logits[0, target_class]

            self.model.zero_grad()
            score.backward()
            total_grads += scaled_input.grad

        # Average gradient * (input - baseline)
        avg_grads = total_grads / (steps + 1)
        ig = (image - baseline) * avg_grads
        ig_np = ig.abs().squeeze().detach().cpu().numpy()

        # Min-max normalization
        ig_min, ig_max = ig_np.min(), ig_np.max()
        if ig_max - ig_min > 1e-8:
            ig_norm = (ig_np - ig_min) / (ig_max - ig_min)
        else:
            ig_norm = np.zeros_like(ig_np)

        return ig_norm

    def evaluate_pointing_game(self, heatmap: np.ndarray, ground_truth_mask: np.ndarray) -> bool:
        """Pointing Game Localization Precision.

        Checks whether the single maximum-attribution coordinate falls inside the ground-truth mask.
        Args:
            heatmap: (224, 224) attribution map in [0, 1]
            ground_truth_mask: (224, 224) binary radiologist mask [0 or 1]
        Returns:
            is_hit: True if peak coordinate is inside tumor mask, False otherwise
        """
        peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        is_hit = bool(ground_truth_mask[peak_idx[0], peak_idx[1]] > 0.5)
        return is_hit

    def evaluate_deletion_curve(
        self, image: torch.Tensor, heatmap: np.ndarray, target_class: int, num_steps: int = 5
    ) -> List[float]:
        """Deletion Faithfulness Benchmark (Pixel-Flipping).

        Progressively masks out the top most-attributed patches and records the confidence drop.
        A steep drop proves that the model genuinely relies on those salient regions.
        """
        image = image.to(self.device)
        patch_size = 16
        grid_size = 224 // patch_size
        num_patches = grid_size * grid_size

        # Compute average attribution score per 16x16 patch
        patch_scores = []
        for r in range(grid_size):
            for c in range(grid_size):
                patch_area = heatmap[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size]
                patch_scores.append(float(patch_area.mean()))

        # Sort patch indices descending by importance
        sorted_patch_indices = np.argsort(patch_scores)[::-1]

        # Evaluate probabilities as patches are progressively deleted
        confidences = []
        perturbed_img = image.clone()

        step_size = num_patches // num_steps
        with torch.no_grad():
            initial_prob = self.model(perturbed_img)["probabilities"][0, target_class].item()
            confidences.append(initial_prob)

            for step in range(1, num_steps + 1):
                # Delete patches for this step (zero them out)
                patches_to_delete = sorted_patch_indices[: step * step_size]
                mask_grid = torch.ones_like(perturbed_img)
                for idx in patches_to_delete:
                    r = idx // grid_size
                    c = idx % grid_size
                    mask_grid[:, :, r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size] = 0.0

                masked_input = image * mask_grid
                prob = self.model(masked_input)["probabilities"][0, target_class].item()
                confidences.append(prob)

        return confidences
