"""
ESM2 baseline adapter — wraps a trained sklearn LogisticRegression classifier
behind BasePredictor.

The classifier is trained on mean-pooled ESM2 embeddings (1280-dim) using
script 07_train_esm2_baseline.py.  No image stains or GPU required.

supports_inpainting = False  (classification-only)
"""

import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from proteinmap.models.base import BasePredictor
from proteinmap.data.constants import CLASSES

SPLIT_FILE = {"train": "train", "val": "valid", "test": "test"}


class ESM2BaselinePredictor(BasePredictor):
    """
    Naive baseline: mean-pool ESM2 → sklearn OneVsRest LogisticRegression → 29 classes.

    When pooled_cache_path and data_root are supplied, run_on_dataloader bypasses
    the image-loading DataLoader entirely and reads directly from the pooled cache,
    making inference as fast as the training script.

    Usage:
        predictor = ESM2BaselinePredictor(
            model_path="results/esm2_baseline/esm2_baseline_classifier.pkl",
            pooled_cache_path="data/esm2_cache/pooled_esm2.pkl",
            data_root="data/czi_224",
        )
        out = predictor.predict_compartment(esm2_emb, seq_len, landmarks)
    """

    def __init__(
        self,
        model_path: str,
        pooled_cache_path: str = None,
        data_root: str = None,
        threshold: float = None,
    ):
        with open(model_path, "rb") as f:
            ckpt = pickle.load(f)
        self.clf = ckpt["classifier"]
        self.threshold = threshold if threshold is not None else ckpt.get("threshold", 0.5)
        self._pooled_cache_path = pooled_cache_path
        self._data_root = data_root
        self._ml_binarizer = MultiLabelBinarizer().fit(CLASSES)

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
        Landmark stains are ignored — only ESM-2 embeddings are used.

        esm2_embedding: (1, 2000, 1280) float32
        protein_length: int — actual sequence length (used for mean-pooling)
        """
        pooled = esm2_embedding[0, :protein_length, :].mean(axis=0)  # (1280,)
        proba = self.clf.predict_proba([pooled])[0].astype(np.float32)  # (29,)

        binary = (proba >= self.threshold).astype(np.int32)
        if binary.sum() == 0:
            binary[np.argmax(proba)] = 1

        return {"proba": proba, "binary": binary}

    def predict_from_pooled(self, pooled_vec: np.ndarray) -> dict:
        """Predict directly from an already mean-pooled (1280,) vector."""
        proba = self.clf.predict_proba([pooled_vec])[0].astype(np.float32)
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
        return None

    @property
    def supports_inpainting(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ESM2Baseline"

    # ------------------------------------------------------------------
    # Batch inference (used by Evaluator)
    # ------------------------------------------------------------------

    def run_on_dataloader(self, dataloader, limit_batches: int = None) -> list[dict]:
        """
        If pooled_cache_path and data_root were supplied at construction, bypass
        the DataLoader (which loads images we don't need) and run directly from
        the pooled embedding cache.  Otherwise fall back to the DataLoader path.
        """
        if self._pooled_cache_path and self._data_root:
            return self._run_from_pooled_cache(limit_batches)
        return self._run_from_dataloader(dataloader, limit_batches)

    def _run_from_pooled_cache(self, limit_batches: int = None) -> list[dict]:
        """Fast path: no image I/O — pure dict lookups into the pooled cache."""
        print(f"  Loading pooled cache from {self._pooled_cache_path} ...")
        with open(self._pooled_cache_path, "rb") as f:
            pooled = pickle.load(f)

        # Determine split from the dataloader's dataset — default to test
        # We infer split from which antibody files exist; use test by default.
        split_name = "test"
        antibody_file = f"{self._data_root}/annotations/{SPLIT_FILE[split_name]}_antibodies.txt"
        antibody_set = set(open(antibody_file).read().splitlines())

        df = pd.read_csv(f"{self._data_root}/cell_crops/metadata.csv")
        df = df[df["antibody"].isin(antibody_set)]
        df["_ensembl_primary"] = df["ensembl_ids"].str.strip().str.split(",").str[0].str.strip()
        df = df[df["locations"].apply(lambda v: isinstance(v, str))]

        def union_locations(series):
            labels = set()
            for loc_str in series:
                labels.update(c.strip() for c in loc_str.split(","))
            return list(labels)

        label_map = df.groupby("_ensembl_primary")["locations"].apply(union_locations)
        # keep one metadata row per protein for reporting fields
        meta_map = df.drop_duplicates("_ensembl_primary").set_index("_ensembl_primary")

        rows = []
        for i, (ensembl_id, labels) in enumerate(
            tqdm(label_map.items(), desc=f"{self.name} inference (pooled cache)")
        ):
            if limit_batches is not None and i >= limit_batches:
                break
            vec = pooled.get(ensembl_id)
            if vec is None:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                multilabel = self._ml_binarizer.transform([labels])[0].astype(np.int32)

            pred = self.predict_from_pooled(vec)
            meta = meta_map.loc[ensembl_id] if ensembl_id in meta_map.index else {}

            rows.append({
                "model": self.name,
                "antibody": meta.get("antibody") if hasattr(meta, "get") else None,
                "ensembl_ids": ensembl_id,
                "if_plate_id": meta.get("if_plate_id") if hasattr(meta, "get") else None,
                "position": meta.get("position") if hasattr(meta, "get") else None,
                "sample": meta.get("sample") if hasattr(meta, "get") else None,
                "cell_id": meta.get("cell_id") if hasattr(meta, "get") else None,
                "true_locations": ",".join(labels),
                "atlas_name": meta.get("atlas_name") if hasattr(meta, "get") else None,
                "inpainting_mse": np.nan,
                "y_true_multilabel": multilabel,
                "y_pred_proba": pred["proba"],
                "y_pred_binary": pred["binary"],
                "y_antibody_pred": None,
                "y_antibody_true": None,
            })
        return rows

    def _run_from_dataloader(self, dataloader, limit_batches: int = None) -> list[dict]:
        """Fallback: iterate a DataLoader (slow if it loads images)."""
        from tqdm import tqdm

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

            X_esm2, X_len, _X_stains, y_multi, _y_ab = datums
            N = X_esm2.shape[0]

            for i in range(N):
                emb_np = X_esm2[i].cpu().numpy()
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
