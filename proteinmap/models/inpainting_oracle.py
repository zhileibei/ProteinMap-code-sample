"""
OracleClassifier — image-only CNN that maps a single-channel antibody stain
(128×128) to 29-class compartment probabilities.

Used as a semantic inpainting metric: if the predicted image carries the right
compartment signal, the oracle should predict the correct compartment.
"""

import torch
import torch.nn as nn


class OracleClassifier(nn.Module):
    """
    Small CNN classifier: (B, 1, 128, 128) → (B, 29) logits.

    Architecture:
        Conv(1→32) + BN + ReLU + MaxPool → 64×64
        Conv(32→64) + BN + ReLU + MaxPool → 32×32
        Conv(64→128) + BN + ReLU + MaxPool → 16×16
        Conv(128→256) + BN + ReLU
        AdaptiveAvgPool → (256, 1, 1)
        Flatten → Linear(256, 29)
    """

    def __init__(self, n_classes: int = 29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities. Input: (B, 1, 128, 128)."""
        return torch.sigmoid(self.forward(x))
