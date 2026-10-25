"""
proteinmap.models.components — pluggable encoder/fusion/head subpackage.

Public surface:
    sequence_encoders: ESM2PooledEncoder, OneHotEncoder
    vision_encoders:   SmallCNNEncoder, PretrainedVisionEncoder
    fusion:            FusionModule
    heads:             ClassificationHead, InpaintingDecoder
    localization_model: LocalizationModel, ModelConfig (and sub-configs)
"""
from proteinmap.models.components.sequence_encoders import ESM2PooledEncoder, OneHotEncoder
from proteinmap.models.components.vision_encoders import SmallCNNEncoder, PretrainedVisionEncoder
from proteinmap.models.components.fusion import FusionModule
from proteinmap.models.components.heads import ClassificationHead, InpaintingDecoder
from proteinmap.models.components.localization_model import (
    LocalizationModel,
    ModelConfig,
    SequenceEncoderConfig,
    VisionEncoderConfig,
    FusionConfig,
    HeadsConfig,
)

__all__ = [
    "ESM2PooledEncoder",
    "OneHotEncoder",
    "SmallCNNEncoder",
    "PretrainedVisionEncoder",
    "FusionModule",
    "ClassificationHead",
    "InpaintingDecoder",
    "LocalizationModel",
    "ModelConfig",
    "SequenceEncoderConfig",
    "VisionEncoderConfig",
    "FusionConfig",
    "HeadsConfig",
]
