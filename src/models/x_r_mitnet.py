"""X-R-MiTNet (v2): Hierarchical Multi-Scale Asymmetry-Aware Transformer.

Advanced Multi-Task Architecture for Brain Tumor Detection and Segmentation:
1. Multi-Scale Overlapping Convolutional Patch Tokenizer (retains fine necrotic micro-textures).
2. 4-Stage Hierarchical Transformer Backbone:
   - Stage 1: 56x56 resolution, C=64 (Fine tissue boundaries)
   - Stage 2: 28x28 resolution, C=128 (Local lesion patterns)
   - Stage 3: 14x14 resolution, C=256 (Bilateral Symmetry S-MHSA & anatomical midline comparison)
   - Stage 4: 7x7 resolution, C=512 (Global semantic pathology representation)
3. Multi-Scale Feature Pyramid Network (FPN) Dense Decoder for 224x224 segmentation.
4. Dirichlet Evidential Classification Head with Multi-Scale Pooling for Calibrated Clinical Uncertainty.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.symmetry_attention import BilateralSymmetryAttention
from src.models.evidential import EvidentialHead


class OverlappingPatchStem(nn.Module):
    """Overlapping Gated Convolutional Stem.

    Extracts high-frequency radiomic edge gradients and safely downsamples
    (1, 224, 224) -> (64, 56, 56) without crude non-overlapping information loss.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        return x  # (B, 64, 56, 56)


class DownsampleBlock(nn.Module):
    """Spatial downsampling between transformer stages with channel expansion."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class TransformerStage(nn.Module):
    """Hierarchical Stage containing multiple S-MHSA blocks with residual connections."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        enable_symmetry: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": BilateralSymmetryAttention(
                    dim=dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop, enable_symmetry=enable_symmetry
                ),
                "norm2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(int(dim * mlp_ratio), dim),
                    nn.Dropout(drop),
                ),
            })
            for _ in range(depth)
        ])

    def forward(
        self, x: torch.Tensor, H: int, W: int, return_attention: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_weights = []
        for block in self.blocks:
            norm_x = block["norm1"](x)
            attn_out, attn = block["attn"](norm_x, H, W, return_attention=return_attention)
            if return_attention and attn is not None:
                attn_weights.append(attn)
            x = x + attn_out
            x = x + block["mlp"](block["norm2"](x))
        return x, attn_weights


class FeaturePyramidSegmentationDecoder(nn.Module):
    """Dense Feature Pyramid Decoder with multi-scale skip connections.

    Fuses features from Stage 4 (7x7), Stage 3 (14x14), Stage 2 (28x28), and Stage 1 (56x56)
    into a full-resolution (224x224) binary tumor segmentation mask.
    """

    def __init__(self, dims: List[int] = [64, 128, 256, 512]):
        super().__init__()
        # Lateral 1x1 convolutions
        self.lat4 = nn.Conv2d(dims[3], 128, kernel_size=1)
        self.lat3 = nn.Conv2d(dims[2], 128, kernel_size=1)
        self.lat2 = nn.Conv2d(dims[1], 128, kernel_size=1)
        self.lat1 = nn.Conv2d(dims[0], 128, kernel_size=1)

        # Smooth convolutions
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Final progressive expansion from 56x56 -> 112x112 -> 224x224
        self.expand1 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.expand2 = nn.Sequential(
            nn.Conv2d(64, 1 * 4, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
        )

    def forward(
        self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            f1: (B, 64, 56, 56)
            f2: (B, 128, 28, 28)
            f3: (B, 256, 14, 14)
            f4: (B, 512, 7, 7)
        """
        # Top-down FPN pathway
        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)

        p2 = self.lat2(f2) + F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)

        p1 = self.lat1(f1) + F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)
        p1 = self.smooth1(p1)  # (B, 128, 56, 56)

        # Expand 56x56 -> 112x112 -> 224x224
        x = self.expand1(p1)   # (B, 64, 112, 112)
        mask_logits = self.expand2(x)  # (B, 1, 224, 224)
        pred_mask = torch.sigmoid(mask_logits)

        return mask_logits, pred_mask


class XRMiTNet(nn.Module):
    """Hierarchical Multi-Scale Asymmetry-Aware Transformer Network (X-R-MiTNet v2).

    Combines:
    - Multi-scale overlapping stem
    - 4-stage progressive transformer encoder with bilateral symmetry contrast
    - Dense U-Net / FPN skip-connection decoder
    - Evidential Dirichlet classification head with multi-scale pooled features
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 1,
        num_classes: int = 4,
        dims: List[int] = [64, 128, 256, 512],
        depths: List[int] = [2, 2, 4, 2],
        num_heads: List[int] = [2, 4, 8, 16],
        drop_rate: float = 0.1,
        use_symmetry: bool = True,
        use_overlapping_stem: bool = True,
        use_evidential: bool = True,
        use_segmentation: bool = True,
    ):
        super().__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.use_symmetry = use_symmetry
        self.use_overlapping_stem = use_overlapping_stem
        self.use_evidential = use_evidential
        self.use_segmentation = use_segmentation

        # 1. Multi-Scale Overlapping Stem vs Non-Overlapping Patch Projection
        if self.use_overlapping_stem:
            self.stem = OverlappingPatchStem(in_channels=in_channels, out_channels=dims[0])
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.GELU(),
            )

        # 2. Stage 1: 56x56
        self.stage1 = TransformerStage(
            dim=dims[0], depth=depths[0], num_heads=num_heads[0], drop=drop_rate, enable_symmetry=use_symmetry
        )
        self.down1 = DownsampleBlock(dims[0], dims[1])

        # 3. Stage 2: 28x28
        self.stage2 = TransformerStage(
            dim=dims[1], depth=depths[1], num_heads=num_heads[1], drop=drop_rate, enable_symmetry=use_symmetry
        )
        self.down2 = DownsampleBlock(dims[1], dims[2])

        # 4. Stage 3: 14x14 (Primary Bilateral Symmetry Attention)
        self.stage3 = TransformerStage(
            dim=dims[2], depth=depths[2], num_heads=num_heads[2], drop=drop_rate, enable_symmetry=use_symmetry
        )
        self.down3 = DownsampleBlock(dims[2], dims[3])

        # 5. Stage 4: 7x7 (Global Semantic Context)
        self.stage4 = TransformerStage(
            dim=dims[3], depth=depths[3], num_heads=num_heads[3], drop=drop_rate, enable_symmetry=use_symmetry
        )

        # 6. Multi-Scale Feature Pooling for Classifier
        classifier_dim = dims[2] + dims[3]
        self.norm_cls = nn.LayerNorm(classifier_dim)

        if self.use_evidential:
            self.evidential_head = EvidentialHead(in_features=classifier_dim, num_classes=num_classes, hidden_dim=256)
            self.linear_cls = None
        else:
            self.evidential_head = None
            self.linear_cls = nn.Linear(classifier_dim, num_classes)

        # 7. Dense FPN Auxiliary Segmentation Decoder
        if self.use_segmentation:
            self.seg_decoder = FeaturePyramidSegmentationDecoder(dims=dims)
        else:
            self.seg_decoder = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Args:
            x: (B, 1, 224, 224) input MRI slice
            return_attention: whether to return attention matrices for CARE-XAI
        """
        B = x.shape[0]

        # Stem: (B, 64, 56, 56)
        f1_stem = self.stem(x)
        H1, W1 = f1_stem.shape[2], f1_stem.shape[3]
        tokens1 = f1_stem.flatten(2).transpose(1, 2)
        tokens1, attn1 = self.stage1(tokens1, H1, W1, return_attention=return_attention)
        f1 = tokens1.transpose(1, 2).reshape(B, self.dims[0], H1, W1)

        # Stage 2: (B, 128, 28, 28)
        f2_in = self.down1(f1)
        H2, W2 = f2_in.shape[2], f2_in.shape[3]
        tokens2 = f2_in.flatten(2).transpose(1, 2)
        tokens2, attn2 = self.stage2(tokens2, H2, W2, return_attention=return_attention)
        f2 = tokens2.transpose(1, 2).reshape(B, self.dims[1], H2, W2)

        # Stage 3: (B, 256, 14, 14)
        f3_in = self.down2(f2)
        H3, W3 = f3_in.shape[2], f3_in.shape[3]
        tokens3 = f3_in.flatten(2).transpose(1, 2)
        tokens3, attn3 = self.stage3(tokens3, H3, W3, return_attention=return_attention)
        f3 = tokens3.transpose(1, 2).reshape(B, self.dims[2], H3, W3)

        # Stage 4: (B, 512, 7, 7)
        f4_in = self.down3(f3)
        H4, W4 = f4_in.shape[2], f4_in.shape[3]
        tokens4 = f4_in.flatten(2).transpose(1, 2)
        tokens4, attn4 = self.stage4(tokens4, H4, W4, return_attention=return_attention)
        f4 = tokens4.transpose(1, 2).reshape(B, self.dims[3], H4, W4)

        # Multi-Scale Global Average Pooling (Stage 3 + Stage 4)
        pool3 = f3.mean(dim=[-2, -1])  # (B, 256)
        pool4 = f4.mean(dim=[-2, -1])  # (B, 512)
        f_cls = torch.cat([pool3, pool4], dim=1)  # (B, 768)
        f_cls = self.norm_cls(f_cls)

        # Classification Head (Evidential Dirichlet vs Standard Softmax)
        if self.use_evidential and self.evidential_head is not None:
            cls_out = self.evidential_head(f_cls)
        else:
            logits = self.linear_cls(f_cls)
            probs = F.softmax(logits, dim=-1)
            # Shannon entropy as uncertainty proxy
            ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) / math.log(self.num_classes)
            cls_out = {
                "logits": logits,
                "probabilities": probs,
                "uncertainty": ent,
                "alpha": probs * 10.0 + 1.0,
                "evidence": probs * 10.0,
            }

        # Auxiliary FPN Dense Segmentation (if enabled)
        if self.use_segmentation and self.seg_decoder is not None:
            mask_logits, pred_mask = self.seg_decoder(f1, f2, f3, f4)
        else:
            mask_logits = torch.zeros(B, 1, 224, 224, device=x.device)
            pred_mask = torch.zeros(B, 1, 224, 224, device=x.device)

        out = {
            **cls_out,
            "mask_logits": mask_logits,
            "pred_mask": pred_mask,
            "stage_features": [f1, f2, f3, f4],
        }

        if return_attention:
            out["attention_weights"] = attn3 + attn4

        return out
