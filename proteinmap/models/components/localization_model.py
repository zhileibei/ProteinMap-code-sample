"""
LocalizationModel — composes pluggable encoders + fusion + heads.

ModelConfig dataclass supports YAML round-trip (PyYAML, no schema dependency).

Checkpoint format (produced by scripts/train.py):
{
    "model_config": dataclasses.asdict(cfg),
    "model_state":  {k: v.cpu() for k, v in model.state_dict().items()},
    "threshold":    0.5,
    "val_macro_auroc": float,
}
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from proteinmap.models.components.sequence_encoders import (
    ESM2PooledEncoder,
    OneHotEncoder,
)
from proteinmap.models.components.vision_encoders import (
    SmallCNNEncoder,
    PretrainedVisionEncoder,
)
from proteinmap.models.components.fusion import FusionModule
from proteinmap.models.components.heads import ClassificationHead, InpaintingDecoder


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SequenceEncoderConfig:
    type:     str = "esm2_pooled"   # 'esm2_pooled' | 'one_hot' | 'none'
    esm2_dim: int = 1280
    out_dim:  int = 256


@dataclass
class VisionEncoderConfig:
    type:              str  = "small_cnn"   # 'small_cnn' | 'pretrained' | 'subcell_vit' | 'none'
    backbone:          str  = "resnet18"
    out_dim:           int  = 256
    freeze_backbone:   bool = True
    subcell_vit_ckpt:  Optional[str] = None   # path injected at runtime for subcell_vit


@dataclass
class RNAEncoderConfig:
    type:         str           = "mlp"   # 'mlp' | 'none'
    n_genes:      Optional[int] = None    # injected at runtime from RNAExpressionLoader.n_genes
    hidden_dim:   int           = 512
    out_dim:      int           = 256


@dataclass
class FusionConfig:
    mode:    str           = "concat"   # 'concat' | 'single_seq' | 'single_vis' | 'vis_only'
    out_dim: Optional[int] = None       # None → input_dim (no extra projection)
    gate:    bool          = False


@dataclass
class HeadsConfig:
    n_classes:  int   = 29
    dropout:    float = 0.0
    inpainting: bool  = False


@dataclass
class ModelConfig:
    name:        str                    = "image_seq_classifier"
    seq_encoder: SequenceEncoderConfig  = field(default_factory=SequenceEncoderConfig)
    vis_encoder: VisionEncoderConfig    = field(default_factory=VisionEncoderConfig)
    rna_encoder: Optional[RNAEncoderConfig] = None
    fusion:      FusionConfig           = field(default_factory=FusionConfig)
    heads:       HeadsConfig            = field(default_factory=HeadsConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)

        rna_enc_cfg = None
        if "rna_encoder" in d and d["rna_encoder"] is not None:
            rna_raw = d["rna_encoder"]
            if isinstance(rna_raw, dict) and rna_raw.get("type", "none") != "none":
                rna_enc_cfg = RNAEncoderConfig(**rna_raw)

        return cls(
            name=d.get("name", "localization_model"),
            seq_encoder=SequenceEncoderConfig(**d.get("seq_encoder", {})),
            vis_encoder=VisionEncoderConfig(**d.get("vis_encoder", {})),
            rna_encoder=rna_enc_cfg,
            fusion=FusionConfig(**d.get("fusion", {})),
            heads=HeadsConfig(**d.get("heads", {})),
        )

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, default_flow_style=False)


# ---------------------------------------------------------------------------
# LocalizationModel
# ---------------------------------------------------------------------------

class LocalizationModel(nn.Module):
    """
    Composes: seq_encoder + vis_encoder + rna_encoder + FusionModule
              + ClassificationHead + optional InpaintingDecoder.

    forward(seq_input=None, vis_input=None, rna_input=None)
        seq_input: (B, 1280) for esm2_pooled  OR  (B, L, 20) for one_hot  OR  None
        vis_input: (B, 3, H, W)  OR  None
        rna_input: (B, n_genes)  OR  None
        → dict{cls_logits: (B, n_classes), inp_pred: (B, 1, 128, 128) or None}

    At least one encoder must be active.

    Factory:
        LocalizationModel.from_config(cfg: ModelConfig) → LocalizationModel
    """

    def __init__(
        self,
        seq_encoder,
        vis_encoder,
        fusion:      FusionModule,
        cls_head:    ClassificationHead,
        inp_decoder: Optional[InpaintingDecoder] = None,
        rna_encoder=None,
    ):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.vis_encoder = vis_encoder
        self.rna_encoder = rna_encoder
        self.fusion      = fusion
        self.cls_head    = cls_head
        self.inp_decoder = inp_decoder

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "LocalizationModel":
        # ---- sequence encoder ----------------------------------------
        if cfg.seq_encoder.type == "esm2_pooled":
            seq_enc = ESM2PooledEncoder(
                esm2_dim=cfg.seq_encoder.esm2_dim,
                out_dim=cfg.seq_encoder.out_dim,
            )
            seq_dim = cfg.seq_encoder.out_dim
        elif cfg.seq_encoder.type == "one_hot":
            seq_enc = OneHotEncoder(out_dim=cfg.seq_encoder.out_dim)
            seq_dim = cfg.seq_encoder.out_dim
        else:  # 'none'
            seq_enc = None
            seq_dim = 0

        # ---- vision encoder ------------------------------------------
        if cfg.vis_encoder.type == "small_cnn":
            vis_enc = SmallCNNEncoder(out_dim=cfg.vis_encoder.out_dim)
            vis_dim = cfg.vis_encoder.out_dim
        elif cfg.vis_encoder.type == "pretrained":
            vis_enc = PretrainedVisionEncoder(
                backbone=cfg.vis_encoder.backbone,
                out_dim=cfg.vis_encoder.out_dim,
                freeze_backbone=cfg.vis_encoder.freeze_backbone,
            )
            vis_dim = cfg.vis_encoder.out_dim
        elif cfg.vis_encoder.type == "subcell_vit":
            from proteinmap.models.components.vis_encoders import SubCellViTEncoder
            if cfg.vis_encoder.subcell_vit_ckpt is None:
                raise ValueError("subcell_vit_ckpt must be set in VisionEncoderConfig")
            vis_enc = SubCellViTEncoder(vit_ckpt=cfg.vis_encoder.subcell_vit_ckpt)
            vis_dim = cfg.vis_encoder.out_dim  # 1536
        else:  # 'none'
            vis_enc = None
            vis_dim = 0

        # ---- RNA encoder ---------------------------------------------
        rna_enc = None
        rna_dim = 0
        if cfg.rna_encoder is not None and cfg.rna_encoder.type != "none":
            from proteinmap.models.components.rna_encoders import RNAExpressionEncoder
            if cfg.rna_encoder.n_genes is None:
                raise ValueError("rna_encoder.n_genes must be set before building the model")
            rna_enc = RNAExpressionEncoder(
                n_genes=cfg.rna_encoder.n_genes,
                out_dim=cfg.rna_encoder.out_dim,
                hidden_dim=cfg.rna_encoder.hidden_dim,
                encoder_type=cfg.rna_encoder.type,
            )
            rna_dim = cfg.rna_encoder.out_dim

        # ---- fusion --------------------------------------------------
        # vis_only is an alias for single_vis; handled here so FusionModule
        # doesn't need to change
        fusion_mode = cfg.fusion.mode
        if fusion_mode == "vis_only":
            fusion_mode = "single_vis"

        # Build total input dim for concat (includes RNA if present)
        if fusion_mode == "concat":
            # Extend FusionModule to handle 3 inputs by concatenating manually
            # We'll use a simple approach: build fusion over (seq+vis+rna) dims
            total_in = seq_dim + vis_dim + rna_dim
            fused_dim = cfg.fusion.out_dim if cfg.fusion.out_dim is not None else total_in
            fusion_proj = nn.Linear(total_in, fused_dim) if total_in != fused_dim else None
            fusion = _ConcatFusion(
                seq_dim=seq_dim, vis_dim=vis_dim, rna_dim=rna_dim,
                out_dim=fused_dim, proj=fusion_proj,
            )
        else:
            fusion = FusionModule(
                seq_dim=seq_dim,
                vis_dim=vis_dim,
                out_dim=cfg.fusion.out_dim,
                mode=fusion_mode,
                gate=cfg.fusion.gate,
            )
            fused_dim = fusion.output_dim

        # ---- heads ---------------------------------------------------
        cls_head = ClassificationHead(
            in_dim=fused_dim,
            n_classes=cfg.heads.n_classes,
            dropout=cfg.heads.dropout,
        )
        inp_decoder = InpaintingDecoder(in_dim=fused_dim) if cfg.heads.inpainting else None

        return cls(seq_enc, vis_enc, fusion, cls_head, inp_decoder, rna_enc)

    @property
    def supports_inpainting(self) -> bool:
        return self.inp_decoder is not None

    def forward(
        self,
        seq_input: torch.Tensor = None,
        vis_input: torch.Tensor = None,
        rna_input: torch.Tensor = None,
    ) -> dict:
        seq_feat = (
            self.seq_encoder(seq_input)
            if (self.seq_encoder is not None and seq_input is not None)
            else None
        )
        vis_feat = (
            self.vis_encoder(vis_input)
            if (self.vis_encoder is not None and vis_input is not None)
            else None
        )
        rna_feat = (
            self.rna_encoder(rna_input)
            if (self.rna_encoder is not None and rna_input is not None)
            else None
        )

        if isinstance(self.fusion, _ConcatFusion):
            fused = self.fusion(seq_feat=seq_feat, vis_feat=vis_feat, rna_feat=rna_feat)
        else:
            fused = self.fusion(seq_feat=seq_feat, vis_feat=vis_feat)

        cls_logits = self.cls_head(fused)
        inp_pred   = self.inp_decoder(fused) if self.inp_decoder is not None else None

        return {"cls_logits": cls_logits, "inp_pred": inp_pred}


class _ConcatFusion(nn.Module):
    """
    Simple concat fusion supporting 2 or 3 encoder outputs (seq, vis, rna).
    Replaces FusionModule when an rna_encoder is present or for 3-way concat.
    """

    def __init__(
        self,
        seq_dim: int,
        vis_dim: int,
        rna_dim: int,
        out_dim: int,
        proj: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.seq_dim = seq_dim
        self.vis_dim = vis_dim
        self.rna_dim = rna_dim
        self._out_dim = out_dim
        self.proj = proj

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        seq_feat: Optional[torch.Tensor] = None,
        vis_feat: Optional[torch.Tensor] = None,
        rna_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [f for f in (seq_feat, vis_feat, rna_feat) if f is not None]
        if not parts:
            raise ValueError("_ConcatFusion: at least one feature must be non-None")
        x = torch.cat(parts, dim=-1)
        if self.proj is not None:
            x = self.proj(x)
        return x
