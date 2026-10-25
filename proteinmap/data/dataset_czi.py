"""
SubCellDatasetCZI — HPA CZI_224 dataset loader.

Reimplemented from PUPS/src/dataset/dataset_czi.py with imports updated to
reference proteinmap.data.constants instead of src.dataset.dataset.

No PUPS source tree required to import this module.
"""

import pickle
import os
import warnings
from collections import OrderedDict
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from proteinmap.data.channel_conventions import CHANNEL_CONVENTIONS, PNG_LOADER_INDEX
from proteinmap.data.constants import CLASSES, DatasetType

DATA_ROOT = os.environ.get("PROTEINMAP_DATA_ROOT", "data")
ESM2_CACHE = os.environ.get("PROTEINMAP_ESM2_CACHE", f"{DATA_ROOT}/esm2_cache")

SPLIT_FILE = {
    DatasetType.TRAIN: "train",
    DatasetType.EVAL: "valid",
    DatasetType.TEST: "test",
}

_ESM2_PAD_LEN = 2000


class _ESM2LRUCache:
    """
    Per-process LRU cache for full-sequence ESM2 embeddings.

    Stores trimmed (length, 1280) arrays to reduce memory; re-pads to
    (2000, 1280) on retrieval so callers see the same shape as before.

    Each DataLoader worker gets its own independent copy of this cache
    (fork semantics), so no locking is needed.

    Parameters
    ----------
    cache_dir : str
        Directory containing per-protein ``{ensembl_id}.pkl`` files.
    maxsize : int
        Maximum number of proteins held in memory at once.
        At ~400 AA average length, each entry ≈ 2 MB, so
        ``maxsize=512`` ≈ 1 GB per worker process.
    """

    def __init__(self, cache_dir: str, maxsize: int = 512) -> None:
        self._dir = cache_dir
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def get(self, ensembl_id: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Return ``(X_esm2_encoding, protein_length)`` or ``None`` on disk miss.

        ``X_esm2_encoding`` has shape ``(1, 2000, 1280)`` float32 — identical
        to the shape previously returned by a direct ``pickle.load``.
        """
        if ensembl_id in self._cache:
            self._cache.move_to_end(ensembl_id)
        else:
            try:
                with open(f"{self._dir}/{ensembl_id}.pkl", "rb") as f:
                    emb = pickle.load(f)
            except Exception:
                return None

            length = int(emb["length"])
            # Trim zero-padding; store only the meaningful rows.
            arr_trimmed = emb["embedding"][:length, :].copy()  # (length, 1280)

            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)  # evict least-recently-used
            self._cache[ensembl_id] = (arr_trimmed, length)

        arr_trimmed, length = self._cache[ensembl_id]

        # Re-pad to (2000, 1280) — upstream code expects this shape.
        padded = np.zeros((_ESM2_PAD_LEN, 1280), dtype=np.float32)
        padded[:length, :] = arr_trimmed
        return padded[None, ...], length  # (1, 2000, 1280), int


class SubCellDatasetCZI(Dataset):
    def __init__(
        self,
        split: DatasetType,
        data_root: str = DATA_ROOT,
        esm2_cache_dir: str = ESM2_CACHE,
        esm2_cache_size: int = 512,
        pairs_csv: Optional[str] = None,
    ):
        if pairs_csv is not None:
            pairs = pd.read_csv(pairs_csv, usecols=["antibody", "atlas_name"])
            pairs["_keep"] = True
            df = pd.read_csv(f"{data_root}/cell_crops/metadata.csv")
            df = df.merge(pairs, on=["antibody", "atlas_name"], how="left")
            self.df = df[df["_keep"] == True].drop(columns="_keep").reset_index(drop=True)
        else:
            split_name = SPLIT_FILE[split]
            antibody_set = set(
                open(f"{data_root}/annotations/{split_name}_antibodies.txt")
                .read()
                .splitlines()
            )
            df = pd.read_csv(f"{data_root}/cell_crops/metadata.csv")
            self.df = df[df["antibody"].isin(antibody_set)].reset_index(drop=True)

        self.data_root = data_root
        self.esm2_cache_dir = esm2_cache_dir
        self.ml_binarizer = MultiLabelBinarizer().fit(CLASSES)
        self._esm2 = _ESM2LRUCache(esm2_cache_dir, maxsize=esm2_cache_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        res = self.get_item_verbose(idx)
        if res is None:
            return None
        datum, _meta = res
        return datum

    def get_item_verbose(self, idx, filter_low_values=0.19):
        try:
            row = self.df.iloc[idx]
            ensembl_id = row["ensembl_ids"].strip().split(",")[0].strip()

            result = self._esm2.get(ensembl_id)
            if result is None:
                return None
            X_esm2_encoding, X_protein_len = result  # (1, 2000, 1280), int
        except Exception:
            return None

        plate = row["if_plate_id"]
        pos = row["position"]
        sample = row["sample"]
        cell_id = int(float(row["cell_id"]))
        img_path = f"{self.data_root}/cell_crops/{plate}/{plate}_{pos}_{sample}_{cell_id}_cell_image.png"

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # (H, W, 4) BGRA, uint16 or uint8
        if img is None:
            return None
        scale = 65535.0 if img.dtype == np.uint16 else 255.0
        # Resize to 128×128 to match the checkpoint's training resolution
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        _ch = CHANNEL_CONVENTIONS["hpa_czi_png"]
        _idx = PNG_LOADER_INDEX["cv2"]
        nuclei_ch      = img[:, :, _idx[_ch["dna"]]].astype(np.float32)
        microtubule_ch = img[:, :, _idx[_ch["microtubule"]]].astype(np.float32)
        er_ch          = img[:, :, _idx[_ch["er"]]].astype(np.float32)
        antibody_ch    = img[:, :, _idx[_ch["protein"]]].astype(np.float32)

        X_landmark_stains = np.stack([nuclei_ch, microtubule_ch, er_ch]) / scale
        y_antibody_stain = antibody_ch / scale

        if filter_low_values is not None:
            X_landmark_stains[X_landmark_stains < filter_low_values] = 0

        locations = row["locations"]
        if not isinstance(locations, str):
            return None  # skip rows with missing location labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_multilabel = self.ml_binarizer.transform(
                [locations.split(",")]
            )[0]

        return (
            (X_esm2_encoding, X_protein_len, X_landmark_stains, y_multilabel, y_antibody_stain),
            row.to_dict(),
        )
