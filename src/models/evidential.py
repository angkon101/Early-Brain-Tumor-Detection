"""Evidential Deep Learning for Clinical Uncertainty Quantification and Reliability Estimation.

Replaces standard overconfident Softmax with a Dirichlet distribution over class probabilities
(Sensoy et al., NeurIPS). Calculates calibrated epistemic uncertainty (u) and reliability score (omega).
Includes Focal Evidential Loss for handling class imbalance and focusing on borderline tumor margins.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """Evidential Classification Head mapping feature embedding to Dirichlet parameters.

    Outputs non-negative evidence e >= 0, Dirichlet concentration parameters alpha = e + 1,
    expected class probabilities p = alpha / S, and epistemic uncertainty u = K / S.
    """

    def __init__(self, in_features: int, num_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
            x: (B, D) pooled representation
        """
        logits = self.mlp(x)
        # Non-negative evidence via Softplus activation
        evidence = F.softplus(logits)
        alpha = evidence + 1.0  # Dirichlet parameters alpha_k >= 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength S = sum(alpha)

        # Expected class probabilities under Dirichlet posterior
        probabilities = alpha / S

        # Calibrated epistemic uncertainty u = K / S in (0, 1]
        uncertainty = (self.num_classes / S).squeeze(-1)

        # Reliability score omega = 1 - u
        reliability = torch.clamp(1.0 - uncertainty, min=0.0, max=1.0)

        return {
            "evidence": evidence,
            "alpha": alpha,
            "probabilities": probabilities,
            "uncertainty": uncertainty,
            "reliability": reliability,
            "logits": logits,
        }


class EvidentialLoss(nn.Module):
    """Focal Evidential Cross-Entropy Loss with KL divergence regularization.

    Incorporates focal modulation factor (1 - p_t)^gamma to prioritize
    difficult and early-stage small tumors over easily classified control scans.
    """

    def __init__(self, num_classes: int = 4, kl_coeff: float = 0.2, gamma: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.kl_coeff = kl_coeff
        self.gamma = gamma

    def forward(
        self, alpha: torch.Tensor, target: torch.Tensor, epoch_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            alpha: (B, K) Dirichlet concentration parameters >= 1.0
            target: (B,) integer ground-truth class labels 0..K-1
            epoch_ratio: Annealing factor min(1.0, current_epoch / max_anneal_epochs)
        """
        y_one_hot = F.one_hot(target, num_classes=self.num_classes).float()  # (B, K)
        S = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1)

        # 1. Expected probability for target class: p_t
        p = alpha / S
        pt = torch.sum(y_one_hot * p, dim=1, keepdim=True)
        focal_weight = torch.pow(torch.clamp(1.0 - pt, min=1e-6, max=1.0), self.gamma)

        # 2. Adjusted Cross-Entropy loss under Dirichlet: E[ -log(p_k) ]
        ace_sample = torch.sum(
            y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True
        )
        focal_ace_loss = (focal_weight * ace_sample).mean()

        # 3. KL Divergence regularizer: penalize false high evidence
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde.squeeze(-1))
            - torch.sum(torch.lgamma(alpha_tilde), dim=1)
            + torch.sum(
                (alpha_tilde - 1.0)
                * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
                dim=1,
            )
        ).mean()

        # Annealed total loss
        annealed_kl = min(1.0, epoch_ratio) * self.kl_coeff * kl
        total_loss = focal_ace_loss + annealed_kl

        return total_loss, focal_ace_loss, kl


# Alias for backwards compatibility and explicit naming
FocalEvidentialLoss = EvidentialLoss

