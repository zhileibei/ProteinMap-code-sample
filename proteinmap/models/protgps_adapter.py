"""
ProtGPS adapter — wraps the ProtGPS compartment classifier behind BasePredictor.

ProtGPS is cloned into a configurable local cache on first use.
It accepts ESM-2 embeddings directly, so we reuse the PUPS ESM2 cache.

supports_inpainting = False  (ProtGPS is classification-only)

References:
  - GitHub: https://github.com/HazyResearch/ProtGPS
  - Paper: "ProtGPS: Predicting protein subcellular localization" (Hazyresearch)
"""

import os
import pathlib
import subprocess
import sys
from typing import Optional

import numpy as np
import torch

from proteinmap.models.base import BasePredictor
from proteinmap.data.label_mapping import protgps_label_to_multilabel, PUPS_CLASSES

_PROTGPS_ROOT = pathlib.Path(
    os.environ.get(
        "PROTEINMAP_PROTGPS_ROOT",
        str(pathlib.Path.home() / ".cache" / "proteinmap" / "protgps"),
    )
).expanduser()
_PROTGPS_REPO = "https://github.com/pgmikhael/protgps.git"


def _ensure_protgps_cloned():
    """Clone ProtGPS repository if not already present."""
    if not _PROTGPS_ROOT.exists():
        _PROTGPS_ROOT.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning ProtGPS into {_PROTGPS_ROOT} ...")
        subprocess.run(
            ["git", "clone", _PROTGPS_REPO, str(_PROTGPS_ROOT)],
            check=True,
        )
    if str(_PROTGPS_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROTGPS_ROOT))


class ProtGPSPredictor(BasePredictor):
    """
    Wrapper around ProtGPS.

    ProtGPS architecture (from the repo):
      - Accepts pre-computed ESM-2 embeddings (mean-pooled or full)
      - Outputs a probability distribution over its own compartment vocabulary
      - We map that vocabulary → PUPS 29-class space via label_mapping.py

    Usage:
        predictor = ProtGPSPredictor(checkpoint_path="/path/to/protgps.ckpt")
        out = predictor.predict_compartment(esm2_emb, seq_len, landmarks)
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        _ensure_protgps_cloned()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Import ProtGPS model
        try:
            from protgps.models.classifier import Classifier  # noqa: F401
            _model_cls = Classifier
        except ImportError as exc:
            raise ImportError(
                f"Could not import ProtGPS model class from {_PROTGPS_ROOT}. "
                "Verify the repository structure and adjust the import path in "
                "proteinmap/models/protgps_adapter.py."
            ) from exc

        self._model = _model_cls.load_from_checkpoint(checkpoint_path)
        self._model.eval()
        self._model.to(self.device)

        # Retrieve ProtGPS class names for vocab mapping (repo-dependent attribute name)
        if hasattr(self._model, "class_names"):
            self._protgps_classes = list(self._model.class_names)
        elif hasattr(self._model, "classes"):
            self._protgps_classes = list(self._model.classes)
        else:
            # Fallback: standard ProtGPS 10-class vocabulary
            self._protgps_classes = [
                "nucleus", "cytoplasm", "mitochondria", "er", "golgi",
                "plasma_membrane", "secreted", "lysosome", "peroxisome",
                "nuclear_membrane",
            ]

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def predict_compartment(
        self,
        esm2_embedding: np.ndarray,
        protein_length: int,
        landmark_stains: np.ndarray,
    ) -> dict:
        """
        ProtGPS does not use landmark stains — only ESM-2 embeddings.

        The embedding is mean-pooled over the sequence dimension to produce
        a (1280,) vector, matching ProtGPS's expected input format.
        """
        # esm2_embedding: (1, 2000, 1280)  → mean-pool valid positions → (1280,)
        emb = esm2_embedding[0, :protein_length, :]  # (L, 1280)
        emb_pooled = emb.mean(axis=0)                # (1280,)
        x = torch.from_numpy(emb_pooled).float().unsqueeze(0).to(self.device)  # (1, 1280)

        with torch.no_grad():
            logits = self._model(x)  # (1, n_classes)
        proba_protgps = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (n_classes,)

        # Map ProtGPS probabilities to PUPS 29-class space
        proba_pups = np.zeros(len(PUPS_CLASSES), dtype=np.float32)
        from proteinmap.data.label_mapping import PROTGPS_TO_PUPS, PUPS_CLASS_INDEX
        for i, cls in enumerate(self._protgps_classes):
            pups_names = PROTGPS_TO_PUPS.get(cls.lower(), [])
            for pups_name in pups_names:
                if pups_name in PUPS_CLASS_INDEX:
                    proba_pups[PUPS_CLASS_INDEX[pups_name]] = max(
                        proba_pups[PUPS_CLASS_INDEX[pups_name]], proba_protgps[i]
                    )

        # Threshold at 0.5 for binary predictions
        binary_pups = (proba_pups >= 0.5).astype(np.int32)
        # Ensure at least the top class is predicted
        if binary_pups.sum() == 0 and len(proba_pups) > 0:
            binary_pups[np.argmax(proba_pups)] = 1

        return {"proba": proba_pups, "binary": binary_pups}

    def predict_inpainting(
        self,
        esm2_embedding: np.ndarray,
        protein_length: int,
        landmark_stains: np.ndarray,
    ) -> Optional[np.ndarray]:
        return None  # ProtGPS does not support inpainting

    @property
    def supports_inpainting(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ProtGPS"

    # ------------------------------------------------------------------
    # Batch inference (used by Evaluator)
    # ------------------------------------------------------------------

    def run_on_dataloader(self, dataloader, limit_batches: int = None) -> list[dict]:
        """Run batch inference; returns per-sample dicts like PUPSPredictor."""
        rows = []
        with torch.no_grad():
            from tqdm import tqdm
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

                X_esm2, X_len, _X_stains, y_multi, _y_ab = datums
                N = X_esm2.shape[0]

                for i in range(N):
                    emb_np = X_esm2[i].cpu().numpy()    # (1, 2000, 1280)
                    plen = int(X_len[i].cpu().item())
                    pred = self.predict_compartment(emb_np, plen, None)
                    meta = metas[i] if i < len(metas) else {}
                    rows.append({
                        "model": self.name,
                        "antibody": meta.get("antibody"),
                        "ensembl_ids": meta.get("ensembl_ids"),
                        "if_plate_id": meta.get("if_plate_id"),
                        "position": meta.get("position"),
                        "sample": meta.get("sample"),
                        "cell_id": meta.get("cell_id"),
                        "true_locations": meta.get("locations"),
                        "atlas_name": meta.get("atlas_name"),
                        "inpainting_mse": np.nan,
                        "y_true_multilabel": y_multi[i].cpu().numpy().astype(np.int32),
                        "y_pred_proba": pred["proba"],
                        "y_pred_binary": pred["binary"],
                        "y_antibody_pred": None,
                        "y_antibody_true": None,
                    })
        return rows
