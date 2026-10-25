"""
LocalizationPredictor — unified BasePredictor that eliminates the ~120 duplicated lines
across ImageSeqPredictor and ImageSeqInpaintingPredictor.

Key insight: run_on_dataloader is identical across all neural predictors except for
(a) how sequence input is resolved and (b) whether inpainting outputs are collected —
both are now dispatched through LocalizationModel.forward() dict output.

Backward compatibility: result dict schema is identical to ImageSeqPredictor /
ImageSeqInpaintingPredictor, so UnifiedEvaluator and the old Evaluator both work.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from proteinmap.models.base import BasePredictor
from proteinmap.models.components.localization_model import (
    LocalizationModel,
    ModelConfig,
    SequenceEncoderConfig,
    VisionEncoderConfig,
    RNAEncoderConfig,
    FusionConfig,
    HeadsConfig,
)


class LocalizationPredictor(BasePredictor):
    """
    Universal predictor for LocalizationModel.

    Replaces: ImageSeqPredictor + ImageSeqInpaintingPredictor (existing classes kept intact).

    For evaluation with ESM2-pooled mode, pass pooled_cache so run_on_dataloader
    can avoid deserialising per-protein pkl files.
    For one_hot mode, pass seq_loader (a SequenceLoader) so ensembl_ids can be
    looked up in the TSV cache.
    For seq_rna_baseline, pass rna_loader (a RNAExpressionLoader) so RNA vectors
    are resolved from atlas_name in batch metadata.
    """

    def __init__(
        self,
        model: LocalizationModel,
        name: str,
        device: str = None,
        threshold: float = 0.5,
        pooled_cache: dict = None,
        seq_mode: str = "esm2_pooled",
        seq_loader=None,    # SequenceLoader for one_hot eval; optional
        rna_loader=None,    # RNAExpressionLoader; optional
    ):
        self.model        = model
        self._name        = name
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold    = threshold
        self._pooled_cache = pooled_cache
        self.seq_mode     = seq_mode
        self._seq_loader  = seq_loader
        self._rna_loader  = rna_loader

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = None,
        pooled_cache_path: str = None,
        tsv_path: str = None,
        rna_loader=None,    # RNAExpressionLoader; optional
    ) -> "LocalizationPredictor":
        """
        Reconstruct model architecture from the embedded 'model_config' dict
        and load weights from 'model_state'.

        For old checkpoints that lack 'model_config', use the --legacy_model flag
        in scripts/evaluate.py instead.
        """
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)

        cfg_dict = ckpt.get("model_config")
        if cfg_dict is None:
            raise ValueError(
                "Checkpoint does not contain 'model_config'. "
                "For old-format checkpoints use evaluate.py --legacy_model."
            )

        # Reconstruct RNAEncoderConfig if present
        rna_enc_cfg = None
        rna_raw = cfg_dict.get("rna_encoder")
        if rna_raw is not None and isinstance(rna_raw, dict) and rna_raw.get("type", "none") != "none":
            rna_enc_cfg = RNAEncoderConfig(**rna_raw)
            # If rna_loader provided and n_genes not set in ckpt, inject it
            if rna_enc_cfg.n_genes is None and rna_loader is not None:
                rna_enc_cfg.n_genes = rna_loader.n_genes

        cfg = ModelConfig(
            name=cfg_dict.get("name", "localization_model"),
            seq_encoder=SequenceEncoderConfig(**cfg_dict.get("seq_encoder", {})),
            vis_encoder=VisionEncoderConfig(**cfg_dict.get("vis_encoder", {})),
            rna_encoder=rna_enc_cfg,
            fusion=FusionConfig(**cfg_dict.get("fusion", {})),
            heads=HeadsConfig(**cfg_dict.get("heads", {})),
        )

        model = LocalizationModel.from_config(cfg)
        model.load_state_dict(ckpt["model_state"])

        pooled_cache = None
        if pooled_cache_path:
            print(f"  Loading pooled ESM2 cache from {pooled_cache_path} ...")
            if pooled_cache_path.endswith(".joblib"):
                import joblib
                raw = joblib.load(pooled_cache_path)
                pooled_cache = {}
                for k, v in raw.items():
                    if hasattr(v, "cpu"):
                        v = v.cpu()
                    if hasattr(v, "numpy"):
                        v = v.numpy()
                    pooled_cache[k] = np.asarray(v, dtype=np.float32)
            else:
                with open(pooled_cache_path, "rb") as f:
                    pooled_cache = pickle.load(f)

        seq_loader = None
        if tsv_path and cfg.seq_encoder.type == "one_hot":
            from proteinmap.data.sequence_loader import SequenceLoader
            seq_loader = SequenceLoader("one_hot", tsv_path=tsv_path)

        return cls(
            model=model,
            name=cfg.name,
            device=device,
            threshold=ckpt.get("threshold", 0.5),
            pooled_cache=pooled_cache,
            seq_mode=cfg.seq_encoder.type,
            seq_loader=seq_loader,
            rna_loader=rna_loader,
        )

    # ------------------------------------------------------------------
    # BasePredictor interface (single-sample; used by legacy Evaluator)
    # ------------------------------------------------------------------

    def predict_compartment(
        self,
        esm2_embedding: np.ndarray,   # (1, 2000, 1280)
        protein_length: int,
        landmark_stains: np.ndarray,  # (3, 128, 128)
    ) -> dict:
        pooled = esm2_embedding[0, :protein_length, :].mean(axis=0)   # (1280,)
        seq_t  = torch.from_numpy(pooled).unsqueeze(0).float().to(self.device)
        img_t  = torch.from_numpy(landmark_stains).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(seq_input=seq_t, vis_input=img_t)
        proba = torch.sigmoid(out["cls_logits"]).squeeze(0).cpu().numpy().astype(np.float32)
        binary = (proba >= self.threshold).astype(np.int32)
        if binary.sum() == 0:
            binary[np.argmax(proba)] = 1
        return {"proba": proba, "binary": binary}

    def predict_inpainting(
        self,
        esm2_embedding: np.ndarray,
        protein_length: int,
        landmark_stains: np.ndarray,
    ) -> Optional[np.ndarray]:
        if not self.supports_inpainting:
            return None
        pooled = esm2_embedding[0, :protein_length, :].mean(axis=0)
        seq_t  = torch.from_numpy(pooled).unsqueeze(0).float().to(self.device)
        img_t  = torch.from_numpy(landmark_stains).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(seq_input=seq_t, vis_input=img_t)
        return out["inp_pred"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    @property
    def supports_inpainting(self) -> bool:
        return self.model.supports_inpainting

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Batch inference (called by Evaluator / UnifiedEvaluator)
    # ------------------------------------------------------------------

    def run_on_dataloader(self, dataloader, limit_batches: int = None) -> list[dict]:
        """
        Single shared implementation that handles:
        - Pooled-cache lookup with per-sample ESM2 pooling fallback
        - One-hot encoding path (via seq_loader)
        - RNA expression vector path (via rna_loader)
        - Inpainting output collection (when model.supports_inpainting)
        - Same result dict schema as ImageSeqPredictor / ImageSeqInpaintingPredictor
        """
        rows = []
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{self.name} inference")):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            if batch is None:
                continue

            if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[1], list):
                datums, metas = batch
            else:
                datums = batch
                metas = [{}] * datums[0].shape[0]

            X_esm2, X_len, X_stains, y_multi, y_ab_true = datums
            B = X_esm2.shape[0]

            # Sequence input
            seq_t = self._resolve_seq_input(X_esm2, X_len, metas)   # (B, seq_dim) or None

            # Vision input
            img_t = X_stains.float().to(self.device)                 # (B, C, H, W)

            # RNA input
            rna_t = None
            if self._rna_loader is not None:
                rna_vecs = []
                for i in range(B):
                    atlas_name = metas[i].get("atlas_name", "") if i < len(metas) else ""
                    rna_vecs.append(self._rna_loader.get_expression(str(atlas_name)))
                rna_t = torch.from_numpy(np.stack(rna_vecs)).float().to(self.device)

            with torch.no_grad():
                out    = self.model(seq_input=seq_t, vis_input=img_t, rna_input=rna_t)
                probas = torch.sigmoid(out["cls_logits"]).cpu().numpy()  # (B, n_classes)
                if out["inp_pred"] is not None:
                    inp_pred = out["inp_pred"].squeeze(1).cpu().numpy()  # (B, H, W)
                else:
                    inp_pred = None

            y_ab_true_np = (
                y_ab_true.cpu().numpy()
                if (y_ab_true is not None and hasattr(y_ab_true, "cpu"))
                else None
            )

            for i in range(B):
                proba  = probas[i].astype(np.float32)
                binary = (proba >= self.threshold).astype(np.int32)
                if binary.sum() == 0:
                    binary[np.argmax(proba)] = 1
                meta = metas[i] if i < len(metas) else {}

                ab_pred = inp_pred[i].astype(np.float32) if inp_pred is not None else None
                ab_true = (
                    y_ab_true_np[i].astype(np.float32)
                    if y_ab_true_np is not None else None
                )
                mse = (
                    float(np.mean((ab_true - ab_pred) ** 2))
                    if (ab_true is not None and ab_pred is not None)
                    else np.nan
                )

                rows.append({
                    "model":             self.name,
                    "antibody":          meta.get("antibody"),
                    "ensembl_ids":       meta.get("ensembl_ids"),
                    "if_plate_id":       meta.get("if_plate_id"),
                    "position":          meta.get("position"),
                    "sample":            meta.get("sample"),
                    "cell_id":           meta.get("cell_id"),
                    "true_locations":    meta.get("locations"),
                    "atlas_name":        meta.get("atlas_name"),
                    "inpainting_mse":    mse,
                    "y_true_multilabel": y_multi[i].cpu().numpy().astype(np.int32),
                    "y_pred_proba":      proba,
                    "y_pred_binary":     binary,
                    "y_antibody_pred":   ab_pred,
                    "y_antibody_true":   ab_true,
                })
        return rows

    def _resolve_seq_input(
        self,
        X_esm2: torch.Tensor,   # (B, 1, 2000, 1280) — raw ESM2 from dataloader
        X_len:  torch.Tensor,   # (B,) integer protein lengths
        metas:  list[dict],
    ) -> Optional[torch.Tensor]:
        """
        Build (B, seq_dim) sequence tensor on self.device.
        Returns None if seq_encoder is 'none'.

        ESM2-pooled path: pooled_cache lookup → per-sample mean-pool fallback.
        One-hot path:     seq_loader TSV lookup; falls back to zero vector if missing.
        """
        if self.seq_mode == "none":
            return None

        B = X_esm2.shape[0]
        vecs = []

        if self.seq_mode == "one_hot":
            from proteinmap.data.sequence_loader import MAX_SEQ_LEN, ONEHOT_DIM
            for i in range(B):
                eid = (metas[i].get("ensembl_ids") or "").split(",")[0].strip()
                if self._seq_loader is not None:
                    rec = self._seq_loader[eid]
                    vec = rec.one_hot
                else:
                    vec = None
                if vec is None:
                    vec = np.zeros((MAX_SEQ_LEN, ONEHOT_DIM), dtype=np.float32)
                vecs.append(torch.from_numpy(vec))
        else:
            # ESM2-pooled (default)
            for i in range(B):
                vec = None
                if self._pooled_cache is not None:
                    eid = (metas[i].get("ensembl_ids") or "").split(",")[0].strip()
                    vec = self._pooled_cache.get(eid)

                if vec is None:
                    l = int(X_len[i].item())
                    vec = X_esm2[i, 0, :l, :].mean(0).numpy()

                vecs.append(torch.from_numpy(np.asarray(vec, dtype=np.float32)))

        return torch.stack(vecs).to(self.device)
