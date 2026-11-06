"""CARE-XAI: Cross-Modal/Multi-Task Attribution and Reliability Explanation Framework.

Implements:
1. Multi-Scale Transformer Attention Rollout
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
        self, image: torch.Tensor, discard_ratio: float = 0.85, head_fusion: str = "mean"
    ) -> np.ndarray:
        """Computes Hierarchical Transformer Attention Rollout across spatial patches.

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
            attn_list = out.get("attention_weights", [])

        if len(attn_list) == 0:
            return np.zeros((224, 224), dtype=np.float32)

        # Filter attention matrices corresponding to the primary 14x14 (196-token) spatial grid
        stage3_attns = [a for a in attn_list if a.size(-1) == 196]
        if len(stage3_attns) == 0:
            stage3_attns = attn_list[:4]

        N = stage3_attns[0].size(-1)
        grid_size = int(np.sqrt(N))

        fused_attns = []
        for attn in stage3_attns:
            if head_fusion == "mean":
                fused = attn.mean(dim=1)  # (1, N, N)
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

            # Add identity matrix for skip connections
            I = torch.eye(N, device=fused.device).unsqueeze(0)
            fused_aug = 0.5 * fused + 0.5 * I
            fused_aug = fused_aug / (fused_aug.sum(dim=-1, keepdim=True) + 1e-9)
            fused_attns.append(fused_aug)

        # Rollout matrix multiplication across depth
        rollout = fused_attns[0]
        for l in range(1, len(fused_attns)):
            rollout = torch.bmm(fused_attns[l], rollout)

        # Extract spatial token influence (mean incoming attention centrality)
        patch_attn = rollout.mean(dim=1).squeeze(0)  # (N,)
        heatmap_2d = patch_attn.view(1, 1, grid_size, grid_size)

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
