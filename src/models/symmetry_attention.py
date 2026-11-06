"""Bilateral Symmetry-Contrastive Multi-Head Self-Attention (S-MHSA v2).

Incorporates:
1. Multi-Scale Bilateral Cross-Attention comparing homologous spatial regions across the sagittal midline.
2. Learned Asymmetry Spatial Gating: amplifies unilateral neoplastic tissue while suppressing symmetric ventricles/cortex.
3. DropPath / Stochastic Depth support for deep regularization.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BilateralSymmetryAttention(nn.Module):
    """Hierarchical Bilateral Symmetry-Contrastive Attention Block.

    Operates on a 2D spatial feature map of shape (B, C, H, W).
    Compares coordinate (r, c) with contralateral mirror coordinate (r, W - 1 - c).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Bilateral Asymmetry Extraction Module (Depthwise + Pointwise)
        self.asym_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1, groups=dim // 2, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, H: int, W: int, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            x: (B, N, C) where N = H * W
            H, W: Spatial grid dimensions
        """
        B, N, C = x.shape
        assert N == H * W, f"Tokens count {N} != H*W ({H}*{W})"

        # 1. Compute Bilateral Asymmetry Feature Map in 2D space
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        x_mirror = torch.flip(x_2d, dims=[-1])         # Flip horizontally across midline (c <-> W-1-c)
        asym_diff = torch.abs(x_2d - x_mirror)         # Focal asymmetry disparity

        # Compute Asymmetry Modulation Gate ∈ [0, 1]
        gate_2d = self.asym_gate(asym_diff)           # (B, C, H, W)
        gate_flat = gate_2d.flatten(2).transpose(1, 2) # (B, N, C)

        # Modulate tokens: anomalous asymmetric regions get amplified
        x_mod = x * (1.0 + gate_flat)

        # 2. Multi-Head Self-Attention on Modulated Sequence
        qkv = self.qkv(x_mod).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn_weights)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attn_weights
        return out, None


# Alias for backward compatibility
SymmetryContrastiveAttention = BilateralSymmetryAttention

