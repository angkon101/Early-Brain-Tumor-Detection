"""X-R-MiTNet: Explainable Reliability-Aware Multimodal Imaging Transformer Network.

A CNN-free Vision Transformer architecture featuring:
1. Bilateral Symmetry-Contrastive Multi-Head Attention (S-MHSA)
2. Evidential Deep Learning for Calibrated Clinical Uncertainty & Reliability Estimation
3. Auxiliary CNN-Free Patch-to-Pixel Progressive Segmentation Decoder
4. Integrated hooks for CARE-XAI attention rollout and faithfulness benchmarking
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.symmetry_attention import SymmetryContrastiveAttention
from src.models.evidential import EvidentialHead, EvidentialLoss
from src.models.segmentation_head import PureTransformerSegmentationDecoder, DiceBCELoss


class LinearPatchEmbedding(nn.Module):
    """CNN-Free Linear Patch Embedding.

    Extracts non-overlapping 16x16 patches from a (1, 224, 224) MRI scan
    and projects them via a linear layer into embed_dim features.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size {img_size} must be divisible by patch_size {patch_size}"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = in_channels * patch_size * patch_size

        # Pure linear projection on flattened patch pixels
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: (B, 1, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        P = self.patch_size
        G = self.grid_size

        # Unfold 2D image into flattened patches: (B, G, G, C, P, P)
        patches = x.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # Flatten each patch to patch_dim: (B, G*G, C*P*P)
        patches = patches.view(B, G * G, -1)
        # Linear projection
        out = self.proj(patches)  # (B, 196, embed_dim)
        return out


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with S-MHSA, Pre-LayerNorm, and MLP."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, grid_size: int = 14, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SymmetryContrastiveAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            grid_size=grid_size,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x, return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class XRMiTNet(nn.Module):
    """Explainable Reliability-Aware Multimodal Imaging Transformer Network (X-R-MiTNet).

    Multi-task CNN-free architecture providing:
    - Early/Small Brain Tumor Detection
    - 4-Class Classification (Glioma, Meningioma, Pituitary, No Tumor)
    - Auxiliary 2D Tumor Segmentation
    - Calibrated Epistemic Uncertainty (u) and Reliability Score (omega)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 4,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 1. Patch Embedding & Position Encoding
        self.patch_embed = LinearPatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # 2. Transformer Encoder Blocks with S-MHSA
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                grid_size=self.grid_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 3. Evidential Classification Head (Primary Detection)
        self.evidential_head = EvidentialHead(
            in_features=embed_dim, num_classes=num_classes, hidden_dim=embed_dim // 2
        )

        # 4. Auxiliary CNN-Free Segmentation Decoder
        self.seg_decoder = PureTransformerSegmentationDecoder(
            embed_dim=embed_dim, grid_size=self.grid_size
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Args:

        x: (B, 1, 224, 224) normalized MRI scan
        return_attention: whether to return attention matrices for CARE-XAI
        Returns:
            dict with classification, uncertainty, reliability, and segmentation outputs
        """
        B = x.shape[0]

        # 1. Patchify & add CLS token + Positional Encoding
        patches = self.patch_embed(x)  # (B, 196, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat((cls_tokens, patches), dim=1)  # (B, 197, D)
        tokens = self.pos_drop(tokens + self.pos_embed)

        # 2. Forward through Transformer Encoder
        attention_matrices = []
        for block in self.blocks:
            tokens, attn = block(tokens, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_matrices.append(attn)

        tokens = self.norm(tokens)
        cls_feat = tokens[:, 0]        # (B, D)
        patch_feat = tokens[:, 1:]     # (B, 196, D)

        # 3. Evidential Detection Prediction
        evidential_out = self.evidential_head(cls_feat)

        # 4. Auxiliary Segmentation Prediction
        mask_logits, pred_mask = self.seg_decoder(patch_feat)

        out = {
            "logits": evidential_out["logits"],               # (B, 4)
            "probabilities": evidential_out["probabilities"], # (B, 4)
            "evidence": evidential_out["evidence"],           # (B, 4)
            "alpha": evidential_out["alpha"],                 # (B, 4)
            "uncertainty": evidential_out["uncertainty"],     # (B,) in (0, 1]
            "reliability": evidential_out["reliability"],     # (B,) in [0, 1]
            "mask_logits": mask_logits,                       # (B, 1, 224, 224)
            "pred_mask": pred_mask,                           # (B, 1, 224, 224) in [0, 1]
        }

        if return_attention:
            out["attention_weights"] = attention_matrices

        return out
