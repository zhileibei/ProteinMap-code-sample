"""
Classification and inpainting heads for LocalizationModel.

ClassificationHead : Linear(in_dim, n_classes) + optional dropout — wraps head from
                     ImageSeqClassifier, no logic change.
InpaintingDecoder  : delegates to SimpleDecoder from image_seq_adapter.py, no logic change.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Multi-label classification head.

    input:  (B, in_dim)
    output: (B, n_classes)   raw logits — caller applies sigmoid / BCEWithLogitsLoss
    """

    def __init__(self, in_dim: int, n_classes: int = 29, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InpaintingDecoder(nn.Module):
    """
    Antibody-stain reconstruction decoder.

    Delegates to SimpleDecoder from image_seq_adapter.py — no logic change.

    input:  (B, in_dim)
    output: (B, 1, 128, 128)   unbounded (no activation; MSE-friendly)
    """

    def __init__(self, in_dim: int = 512):
        super().__init__()
        from proteinmap.models.image_seq_adapter import SimpleDecoder
        self._decoder = SimpleDecoder(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder(x)
