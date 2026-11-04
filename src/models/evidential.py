"""Evidential Deep Learning for Clinical Uncertainty Quantification and Reliability Estimation.

Replaces standard overconfident Softmax with a Dirichlet distribution over class probabilities
(Sensoy et al., NeurIPS). Calculates calibrated epistemic uncertainty (u) and reliability score (omega).
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
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:

        x: (B, D) pooled representation (e.g., from [CLS] token)
        Returns:
            dict containing:
                'evidence': (B, K) >= 0
                'alpha': (B, K) >= 1
                'probabilities': (B, K) in [0, 1], sum to 1
                'uncertainty': (B,) in (0, 1]
                'reliability': (B,) in [0, 1) where omega = 1 - u
                'logits': (B, K) raw unconstrained logits
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
    """Evidential Cross-Entropy Loss with Kullback-Leibler (KL) divergence regularization.

    Penalizes misclassifications while encouraging high uncertainty (uniform Dirichlet)
    on ambiguous or out-of-distribution inputs.
    """

    def __init__(self, num_classes: int = 4, kl_coeff: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.kl_coeff = kl_coeff

    def forward(
        self, alpha: torch.Tensor, target: torch.Tensor, epoch_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:

        alpha: (B, K) Dirichlet concentration parameters >= 1.0
        target: (B,) integer ground-truth class labels 0..K-1
        epoch_ratio: Annealing factor min(1.0, current_epoch / max_anneal_epochs)
        Returns:
            total_loss, ace_loss, kl_loss
        """
        # Convert target to one-hot encoding
        y_one_hot = F.one_hot(target, num_classes=self.num_classes).float()  # (B, K)
        S = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1)

        # 1. Adjusted Cross-Entropy loss under Dirichlet distribution: E[ -log(p_k) ]
        # E[ -log(p_k) ] = digamma(S) - digamma(alpha_k)
        ace_loss = torch.sum(
            y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1
        ).mean()

        # 2. KL Divergence regularizer: penalize high evidence on incorrect classes
        # Target Dirichlet: uniform prior Dir(1, 1, ..., 1)
        # Modify alpha to keep ground truth class evidence, regularize misleading evidence
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)

        # Analytical KL[ Dir(alpha_tilde) || Dir(1) ]
        ones = torch.ones_like(alpha)
        kl = (
            torch.lgamma(S_tilde.squeeze(-1))
            - torch.sum(torch.lgamma(alpha_tilde), dim=1)
            + torch.sum(
                (alpha_tilde - 1.0)
                * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
                dim=1,
            )
        ).mean()

        # Annealed total evidential loss
        annealed_kl = min(1.0, epoch_ratio) * self.kl_coeff * kl
        total_loss = ace_loss + annealed_kl

        return total_loss, ace_loss, kl
