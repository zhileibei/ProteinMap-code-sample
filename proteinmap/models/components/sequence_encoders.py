"""
Pluggable sequence encoders for LocalizationModel.

ESM2PooledEncoder : (B, 1280)     → (B, out_dim)   via Linear + ReLU
OneHotEncoder     : (B, L, 20)    → (B, out_dim)   via Conv1d stack + pool + Linear
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceEncoderBase(nn.Module, ABC):
    """Abstract base for all sequence encoders."""

    @property
    @abstractmethod
    def output_dim(self) -> int: ...


class ESM2PooledEncoder(SequenceEncoderBase):
    """
    Linear projection of mean-pooled ESM2 vectors.

    Wraps seq_proj from ImageSeqClassifier — no logic change.

    input:  (B, esm2_dim)
    output: (B, out_dim)
    """

    def __init__(self, esm2_dim: int = 1280, out_dim: int = 256):
        super().__init__()
        self._out_dim = out_dim
        self.proj = nn.Linear(esm2_dim, out_dim)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(x))


class OneHotEncoder(SequenceEncoderBase):
    """
    3-layer Conv1d + AdaptiveAvgPool1d(1) + Linear.
    Handles variable-length sequences; output is fixed-size.

    input:  (B, L, 20)   — permuted internally to (B, 20, L)
    output: (B, out_dim)

    Architecture: Conv1d(20→64→128→256, kernel=5, same-padding) + pool → Linear(256, out_dim)
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self._out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv1d(20,  64,  kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),    # (B, 256, 1)
        )
        self.proj = nn.Linear(256, out_dim)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 20) → (B, 20, L)
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)    # (B, 256)
        return F.relu(self.proj(x))     # (B, out_dim)
