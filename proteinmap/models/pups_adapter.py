"""
PUPS adapter — wraps SubCellProtModel behind the BasePredictor interface.

The adapter supports two usage modes:
  1. Single-sample predict_compartment / predict_inpainting (used by MislocDataset)
  2. Batch inference via run_on_dataloader() (used by Evaluator for HPA splits)
     — replicates the 2-pass pattern from PUPS/src/utils/run_inference_czi.py

Checkpoint default:
  PUPS/checkpoints/splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt
"""

import pathlib
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make PUPS importable without installing it as a package
# ---------------------------------------------------------------------------
_PUPS_ROOT = pathlib.Path(__file__).parent.parent.parent.parent / "PUPS"
if str(_PUPS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PUPS_ROOT))

from src.model.full_model import SubCellProtModel  # noqa: E402

from proteinmap.data.collate import _collate_with_meta  # noqa: F401

from proteinmap.models.base import BasePredictor  # noqa: E402


class PUPSPredictor(BasePredictor):
    """Thin wrapper around SubCellProtModel."""

    DEFAULT_CKPT = (
        str(_PUPS_ROOT / "checkpoints" /
            "splice_isoform_dataset_cell_line_and_gene_split_full"
            "-epoch=01-val_combined_loss=0.18.ckpt")
    )

    def __init__(self, checkpoint_path: str = None, device: str = None):
        if checkpoint_path is None:
            checkpoint_path = self.DEFAULT_CKPT
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self._model = SubCellProtModel.load_from_checkpoint(checkpoint_path)
        self._model.eval()
        self._model.to(self.device)
        self._inpainting_loss = torch.nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------
    # BasePredictor interface (single-sample)
    # ------------------------------------------------------------------

    def predict_compartment(
        self,
        esm2_embedding: np.ndarray,
        protein_length: int,
        landmark_stains: np.ndarray,
    ) -> dict:
        X_esm2 = torch.from_numpy(esm2_embedding).float().unsqueeze(0).to(self.device)
        X_len = torch.tensor([protein_length]).to(self.device)
        X_stains = torch.from_numpy(landmark_stains).float().unsqueeze(0).to(self.device)
        dummy_y_multi = torch.zeros(1, 29).to(self.device)
        dummy_y_ab = torch.zeros(1, 128, 128).to(self.device)

        with torch.no_grad():
            _ab_pred, y_bin, y_proba = self._model.predict_step(
                (X_esm2, X_len, X_stains, dummy_y_multi, dummy_y_ab), batch_idx=0
            )
        return {
            "proba": y_proba[0].astype(np.float32),
            "binary": y_bin[0].astype(np.int32),
        }

    def predict_inpainting(
        self,
        esm2_embedding: np.ndarray,
        protein_length: int,
        landmark_stains: np.ndarray,
    ) -> Optional[np.ndarray]:
        X_esm2 = torch.from_numpy(esm2_embedding).float().unsqueeze(0).to(self.device)
        X_len = torch.tensor([protein_length]).to(self.device)
        X_stains = torch.from_numpy(landmark_stains).float().unsqueeze(0).to(self.device)
        dummy_y_multi = torch.zeros(1, 29).to(self.device)
        dummy_y_ab = torch.zeros(1, 128, 128).to(self.device)

        with torch.no_grad():
            y_ab_pred, _y_bin, _y_proba = self._model.predict_step(
                (X_esm2, X_len, X_stains, dummy_y_multi, dummy_y_ab), batch_idx=0
            )
        # y_ab_pred: (1, 1, H, W) → (H, W)
        return y_ab_pred.squeeze().cpu().numpy().astype(np.float32)

    @property
    def supports_inpainting(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "PUPS"

    # ------------------------------------------------------------------
    # Batch inference (used by Evaluator)
    # ------------------------------------------------------------------

    def run_on_dataloader(self, dataloader: DataLoader, limit_batches: int = None) -> list[dict]:
        """
        Run inference on an arbitrary DataLoader that yields
        ((X_esm2, X_len, X_stains, y_multi, y_ab), meta_dict) batches.

        Returns a list of per-sample dicts matching the schema used by
        run_inference_czi.py.
        """
        rows = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{self.name} inference")):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                if batch is None:
                    continue

                # Support dataloaders with or without metadata
                if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[1], list):
                    datums, metas = batch
                    has_meta = True
                else:
                    datums = batch
                    metas = [{}] * len(datums[0])
                    has_meta = False  # noqa: F841

                X_esm2, X_len, X_stains, y_multi, y_ab = datums

                X_esm2 = X_esm2.to(self.device)
                X_len = X_len.to(self.device)
                X_stains = X_stains.to(self.device)
                y_multi = y_multi.to(self.device)
                y_ab = y_ab.to(self.device)

                y_ab_pred, y_pred_bin, y_pred_proba = self._model.predict_step(
                    (X_esm2, X_len, X_stains, y_multi, y_ab), batch_idx=0
                )
                y_ab_pred_sq = y_ab_pred.squeeze(1)  # (N, H, W)

                y_ab_gpu = y_ab.to(y_ab_pred_sq.device).type_as(y_ab_pred_sq)
                mse_per_sample = (
                    self._inpainting_loss(y_ab_pred_sq, y_ab_gpu)
                    .mean(dim=(-2, -1))
                    .cpu()
                    .numpy()
                )

                N = X_esm2.shape[0]
                for i in range(N):
                    meta = metas[i] if i < len(metas) else {}
                    rows.append({
                        "model": self.name,
                        "gene": meta.get("gene"),
                        "mutation": meta.get("mutation"),
                        "antibody": meta.get("antibody"),
                        "ensembl_ids": meta.get("ensembl_ids"),
                        "if_plate_id": meta.get("if_plate_id"),
                        "position": meta.get("position"),
                        "sample": meta.get("sample"),
                        "cell_id": meta.get("cell_id"),
                        "true_locations": meta.get("locations"),
                        "atlas_name": meta.get("atlas_name"),
                        "inpainting_mse": float(mse_per_sample[i]),
                        "y_true_multilabel": y_multi[i].cpu().numpy().astype(np.int32),
                        "y_pred_proba": y_pred_proba[i].astype(np.float32),
                        "y_pred_binary": y_pred_bin[i].astype(np.int32),
                        "y_antibody_pred": y_ab_pred_sq[i].cpu().numpy().astype(np.float32),
                        "y_antibody_true": y_ab[i].cpu().numpy().astype(np.float32),
                    })

        return rows
