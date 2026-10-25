"""
Pluggable vision encoders for LocalizationModel.

SmallCNNEncoder        : thin wrapper around SmallCNN from image_seq_adapter.py
PretrainedVisionEncoder: torchvision backbone + projection head, frozen by default
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoderBase(nn.Module, ABC):
    """Abstract base for all vision encoders."""

    @property
    @abstractmethod
    def output_dim(self) -> int: ...


class SmallCNNEncoder(VisionEncoderBase):
    """
    Thin wrapper around SmallCNN from image_seq_adapter.py.
    No logic change — delegates entirely to the existing implementation.

    input:  (B, 3, 128, 128)
    output: (B, out_dim)
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self._out_dim = out_dim
        from proteinmap.models.image_seq_adapter import SmallCNN
        self._cnn = SmallCNN(out_dim)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._cnn(x)


class PretrainedVisionEncoder(VisionEncoderBase):
    """
    torchvision backbone with the final FC replaced by a projection head.
    Backbone weights are frozen by default.

    input:  (B, 3, 128, 128)   — images are resized to backbone input inside torchvision
    output: (B, out_dim)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        out_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self._out_dim = out_dim
        import torchvision.models as tvm

        model_fn = getattr(tvm, backbone)
        base = model_fn(weights="IMAGENET1K_V1")
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self._backbone = base
        self._proj = nn.Linear(in_features, out_dim)

        if freeze_backbone:
            for p in self._backbone.parameters():
                p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._backbone(x)           # (B, in_features)
        return F.relu(self._proj(feat))    # (B, out_dim)
