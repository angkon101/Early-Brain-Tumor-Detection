"""Bilateral Symmetry-Contrastive Multi-Head Self-Attention (S-MHSA).

Neurological brain MRI features strong bilateral left-right anatomical symmetry.
S-MHSA incorporates a contralateral mirror-attention mechanism: each patch at spatial
coordinate (r, c) on a 14x14 grid explicitly compares itself against its contralateral
mirror patch (r, 13 - c) across the interhemispheric brain midline.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetryContrastiveAttention(nn.Module):
    """Bilateral Symmetry-Contrastive Multi-Head Self-Attention block.

    Operates on sequence of tokens [CLS, patch_1, ..., patch_N] where N = grid_size x grid_size.
    Computes standard MHSA plus a symmetry residual modulation that amplifies asymmetric lesions
    (tumors) while suppressing healthy symmetric brain tissue (normal ventricles, hemispheres).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        grid_size: int = 14,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size

        # Standard Q, K, V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Bilateral Asymmetry Projection & Gate
        # Projects feature difference |x(r, c) - x(r, W - 1 - c)| into an attention modulation gate
        self.asymmetry_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid(),
        )

        # Precompute contralateral mirror indices on the grid
        # Coordinate (r, c) -> index (r * grid_size + c). Mirror is (r, grid_size - 1 - c)
        mirror_indices = []
        for r in range(grid_size):
            for c in range(grid_size):
                mirror_idx = r * grid_size + (grid_size - 1 - c)
                mirror_indices.append(mirror_idx)
        self.register_buffer("mirror_indices", torch.tensor(mirror_indices, dtype=torch.long))

    def compute_symmetry_residual(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Computes bilateral asymmetry between each patch and its contralateral mirror.

        Args:
            patch_tokens: (B, N, D) where N = grid_size x grid_size
        Returns:
            asymmetry_gate: (B, N, D) values in [0, 1]
        """
        # Gather mirror patch tokens across the brain midline
        mirror_tokens = patch_tokens[:, self.mirror_indices, :]  # (B, N, D)
        # Compute absolute difference
        asymmetry_diff = torch.abs(patch_tokens - mirror_tokens)  # (B, N, D)
        # Gate modulation: close to 0 if perfectly symmetric, higher if anomalous mass is present
        gate = self.asymmetry_proj(asymmetry_diff)
        return gate

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:

        x: (B, 1 + N, D) tensor containing [CLS, patches]
        return_attention: whether to return the raw attention matrix for CARE-XAI
        Returns:
            out: (B, 1 + N, D)
            attn_weights: (B, num_heads, 1 + N, 1 + N) or None
        """
        B, S, D = x.shape
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # 1. Bilateral symmetry residual modulation on spatial patches
        asym_gate = self.compute_symmetry_residual(patch_tokens)  # (B, N, D)
        modulated_patches = patch_tokens * (1.0 + asym_gate)      # Asymmetric patches get amplified
        x_mod = torch.cat([cls_token, modulated_patches], dim=1)  # (B, 1 + N, D)

        # 2. Standard Multi-Head Self-Attention
        qkv = self.qkv(x_mod).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, S, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn_weights)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attn_weights
        return out, None
