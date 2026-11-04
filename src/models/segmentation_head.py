"""CNN-Free Auxiliary Segmentation Decoder for Spatial Localization and Multi-Task Supervision.

Decodes spatial patch tokens (14x14) into a full-resolution (224x224) pixel-wise tumor mask
using progressive linear projections and PixelShuffle expansion without any 2D spatial convolutions.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PureTransformerSegmentationDecoder(nn.Module):
    """Pure CNN-free patch-to-pixel upsampling decoder.

    Progressively expands 14x14 patch tokens into 224x224 mask resolution
    via 4 successive linear feature expansions and PixelShuffle operations.
    """

    def __init__(self, embed_dim: int = 256, grid_size: int = 14):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Stage 1: 14x14 -> 28x28 (upscale x2)
        # Pointwise 1x1 linear expansion: D -> 4 * (D // 2)
        dim1 = embed_dim // 2
        self.expand1 = nn.Conv2d(embed_dim, dim1 * 4, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(dim1)
        self.act1 = nn.GELU()

        # Stage 2: 28x28 -> 56x56 (upscale x2)
        dim2 = dim1 // 2
        self.expand2 = nn.Conv2d(dim1, dim2 * 4, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(dim2)
        self.act2 = nn.GELU()

        # Stage 3: 56x56 -> 112x112 (upscale x2)
        dim3 = dim2 // 2
        self.expand3 = nn.Conv2d(dim2, dim3 * 4, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(dim3)
        self.act3 = nn.GELU()

        # Stage 4: 112x112 -> 224x224 (upscale x2)
        # Maps to single-channel binary tumor mask
        self.expand4 = nn.Conv2d(dim3, 1 * 4, kernel_size=1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

        patch_tokens: (B, 196, D) spatial patch embeddings (without [CLS] token)
        Returns:
            mask_logits: (B, 1, 224, 224) unnormalized logits
            pred_mask: (B, 1, 224, 224) probabilities in [0, 1]
        """
        B, N, D = patch_tokens.shape
        H = W = self.grid_size

        # Reshape sequence back to spatial 2D grid: (B, D, 14, 14)
        x = patch_tokens.transpose(1, 2).reshape(B, D, H, W)

        # 14x14 -> 28x28
        x = self.pixel_shuffle(self.expand1(x))
        x = self.act1(self.norm1(x))

        # 28x28 -> 56x56
        x = self.pixel_shuffle(self.expand2(x))
        x = self.act2(self.norm2(x))

        # 56x56 -> 112x112
        x = self.pixel_shuffle(self.expand3(x))
        x = self.act3(self.norm3(x))

        # 112x112 -> 224x224
        mask_logits = self.pixel_shuffle(self.expand4(x))  # (B, 1, 224, 224)
        pred_mask = torch.sigmoid(mask_logits)

        return mask_logits, pred_mask


class DiceBCELoss(nn.Module):
    """Combined Soft Dice Loss + Binary Cross-Entropy Loss for tumor segmentation."""

    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:

        logits: (B, 1, H, W) raw prediction logits
        target: (B, 1, H, W) ground truth binary mask in [0, 1]
        """
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)

        # Soft Dice Loss
        intersection = (probs_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)

        return (1.0 - self.bce_weight) * dice_loss + self.bce_weight * bce_loss

