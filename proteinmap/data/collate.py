"""
Collate helpers and metadata-exposing dataset wrapper for CZI_224 inference.

Reimplemented from PUPS/src/utils/run_inference_czi.py with imports updated
to reference proteinmap.data.dataset_czi instead of src.dataset.dataset_czi.

No PUPS source tree required to import this module.
"""

import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from proteinmap.data.channel_conventions import CHANNEL_CONVENTIONS, PNG_LOADER_INDEX
from proteinmap.data.constants import CLASSES, DatasetType
from proteinmap.data.dataset_czi import SubCellDatasetCZI, _ESM2LRUCache


class _CZIWithMeta(SubCellDatasetCZI):
    """Returns (datum, meta_dict) instead of datum alone."""

    def __getitem__(self, idx):
        res = self.get_item_verbose(idx)
        if res is None:
            return None
        datum, meta = res
        return datum, meta


class _CZIWithMetaV2(Dataset):
    """
    Like _CZIWithMeta but filters by (antibody, atlas_name) pairs CSV
    instead of an antibody-list file.  Accepts an `input_size` parameter
    for the image resize resolution (default 128; use 448 for SubCell).

    Used by the split_v2 evaluation splits:
        val_v2, protein_test_v2, cell_line_test_v2
    """

    def __init__(
        self,
        pairs_csv: str,
        data_root: str,
        esm2_cache_dir: str,
        input_size: int = 128,
        esm2_cache_size: int = 512,
        skip_image: bool = False,
    ):
        pairs = pd.read_csv(pairs_csv, usecols=["antibody", "atlas_name"])
        pairs["_keep"] = True
        df = pd.read_csv(f"{data_root}/cell_crops/metadata.csv")
        df = df.merge(pairs, on=["antibody", "atlas_name"], how="left")
        self.df = df[df["_keep"] == True].drop(columns="_keep").reset_index(drop=True)

        self.data_root = data_root
        self.esm2_cache_dir = esm2_cache_dir
        self.input_size = input_size
        self.skip_image = skip_image
        self.ml_binarizer = MultiLabelBinarizer().fit(CLASSES)
        self._esm2 = _ESM2LRUCache(esm2_cache_dir, maxsize=esm2_cache_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        res = self.get_item_verbose(idx)
        if res is None:
            return None
        datum, meta = res
        return datum, meta

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

        if self.skip_image:
            h = self.input_size
            X_landmark_stains = np.zeros((3, h, h), dtype=np.float32)
            y_antibody_stain  = np.zeros((h, h), dtype=np.float32)
        else:
            plate   = row["if_plate_id"]
            pos     = row["position"]
            sample  = row["sample"]
            cell_id = int(float(row["cell_id"]))
            img_path = (
                f"{self.data_root}/cell_crops/{plate}/"
                f"{plate}_{pos}_{sample}_{cell_id}_cell_image.png"
            )

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            scale = 65535.0 if img.dtype == np.uint16 else 255.0
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

            _ch  = CHANNEL_CONVENTIONS["hpa_czi_png"]
            _idx = PNG_LOADER_INDEX["cv2"]
            nuclei_ch      = img[:, :, _idx[_ch["dna"]]].astype(np.float32)
            microtubule_ch = img[:, :, _idx[_ch["microtubule"]]].astype(np.float32)
            er_ch          = img[:, :, _idx[_ch["er"]]].astype(np.float32)
            antibody_ch    = img[:, :, _idx[_ch["protein"]]].astype(np.float32)

            X_landmark_stains = np.stack([nuclei_ch, microtubule_ch, er_ch]) / scale
            y_antibody_stain  = antibody_ch / scale

            if filter_low_values is not None:
                X_landmark_stains[X_landmark_stains < filter_low_values] = 0

        locations = row["locations"]
        if not isinstance(locations, str):
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_multilabel = self.ml_binarizer.transform([locations.split(",")])[0]

        return (
            (X_esm2_encoding, X_protein_len, X_landmark_stains, y_multilabel, y_antibody_stain),
            row.to_dict(),
        )


def _collate_with_meta(batch):
    """Collate (datum, meta) pairs; skip None and blank-ESM2 samples."""
    batch = [b for b in batch if b is not None]
    # b[0][0] = X_esm2_encoding; skip all-zero embeddings
    batch = [b for b in batch if b[0][0].sum() != 0]
    if not batch:
        return None
    datums = torch.utils.data.default_collate([b[0] for b in batch])
    metas = [b[1] for b in batch]
    return datums, metas


def _ignore_bad_collate(batch):
    """Plain collate for loss-only pass (no metadata)."""
    batch = [b for b in batch if b is not None]
    batch = [b for b in batch if b[0].sum() != 0]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)
