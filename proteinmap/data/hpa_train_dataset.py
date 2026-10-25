"""
HPATrainDataset — unified per-cell training dataset for ProteinMap.

Replaces:
  ImageSeqDataset           (scripts/10_train_image_seq_baseline.py)
  ImageSeqInpaintingDataset (scripts/13_train_image_seq_inpainting.py)

The only difference between those two was whether the antibody channel is returned;
that is now controlled by `return_antibody=True/False`.

Image loading / normalisation reuses the 10-line block from
SubCellDatasetCZI.get_item_verbose().
"""

import warnings
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from proteinmap.data.channel_conventions import CHANNEL_CONVENTIONS, PNG_LOADER_INDEX
from proteinmap.data.constants import CLASSES, DatasetType
from proteinmap.data.sequence_loader import SequenceLoader


SPLIT_FILE = {
    DatasetType.TRAIN: "train",
    DatasetType.EVAL:  "valid",
    DatasetType.TEST:  "test",
}


class HPATrainDataset(Dataset):
    """
    Per-cell dataset for training / validation.

    Two loading modes:

    1. Legacy (antibody-list):
         HPATrainDataset(split=DatasetType.TRAIN, data_root=..., seq_loader=...)
       Filters metadata by the antibody IDs in {data_root}/annotations/{split}_antibodies.txt.

    2. Pairs CSV (protein × cell-line split):
         HPATrainDataset(split=None, data_root=..., seq_loader=...,
                         pairs_csv=".../split_v2/train_pairs.csv")
       Filters metadata by (antibody, atlas_name) pairs from the CSV produced by
       make_protein_cellline_split.py.  `split` is ignored when pairs_csv is given.

    __getitem__ returns one of:
        (landmark (3,H,W), seq_input, label (29,))
        (landmark (3,H,W), seq_input, label (29,), antibody (H,W))
        (landmark (3,H,W), seq_input, label (29,), rna_vec (n_genes,))
        (landmark (3,H,W), seq_input, label (29,), antibody (H,W), rna_vec (n_genes,))

    seq_input shape depends on SequenceLoader mode:
        esm2_pooled → (1280,) float32
        one_hot     → (MAX_SEQ_LEN, 20) float32
    seq_input is a zero vector when seq_loader is None (subcell / vis-only models).

    Returns None for unreadable samples; use a collate_fn that filters these.
    """

    def __init__(
        self,
        split: Optional[DatasetType],
        data_root: str,
        seq_loader: Optional[SequenceLoader],
        return_antibody: bool = False,
        filter_low_values: float = 0.19,
        pairs_csv: Optional[str] = None,
        rna_loader=None,      # Optional[RNAExpressionLoader]
        input_size: int = 128,
        skip_image: bool = False,
    ):
        df = pd.read_csv(f"{data_root}/cell_crops/metadata.csv")

        if pairs_csv is not None:
            # Protein × cell-line split: filter by (antibody, atlas_name) pairs
            pairs = pd.read_csv(pairs_csv, usecols=["antibody", "atlas_name"])
            pairs["_keep"] = True
            df = df.merge(pairs, on=["antibody", "atlas_name"], how="left")
            self.df = df[df["_keep"] == True].drop(columns="_keep").reset_index(drop=True)
        else:
            # Legacy: filter by antibody list
            split_name = SPLIT_FILE[split]
            antibody_set = set(
                open(f"{data_root}/annotations/{split_name}_antibodies.txt")
                .read()
                .splitlines()
            )
            self.df = df[df["antibody"].isin(antibody_set)].reset_index(drop=True)

        self.data_root = data_root
        self.seq_loader = seq_loader
        self.return_antibody = return_antibody
        self.filter_low_values = filter_low_values
        self.ml_binarizer = MultiLabelBinarizer().fit(CLASSES)
        self.rna_loader = rna_loader
        self.input_size = input_size
        self.skip_image = skip_image

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Optional[tuple]:
        try:
            row = self.df.iloc[idx]
        except IndexError:
            return None

        # ---- sequence feature ----------------------------------------
        if self.seq_loader is not None:
            ensembl_id = row["ensembl_ids"].strip().split(",")[0].strip()
            rec = self.seq_loader[ensembl_id]

            if self.seq_loader.mode == "esm2_pooled":
                if rec.esm2_pooled is None:
                    return None
                seq_input = rec.esm2_pooled       # (1280,) float32
            else:
                if rec.one_hot is None:
                    return None
                seq_input = rec.one_hot            # (MAX_SEQ_LEN, 20) float32
        else:
            # Vision-only model (e.g. subcell): return zero seq vector
            from proteinmap.data.sequence_loader import MAX_SEQ_LEN, ONEHOT_DIM
            seq_input = np.zeros((1280,), dtype=np.float32)  # won't be used by model

        # ---- image ---------------------------------------------------
        if self.skip_image:
            # Sequence-only models (esm2_baseline, seq_rna_baseline): skip disk read
            h = self.input_size
            landmark_stains = np.zeros((3, h, h), dtype=np.float32)
            antibody_stain  = np.zeros((h, h), dtype=np.float32)
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

            _ch = CHANNEL_CONVENTIONS["hpa_czi_png"]
            _idx = PNG_LOADER_INDEX["cv2"]
            nuclei_ch      = img[:, :, _idx[_ch["dna"]]].astype(np.float32)
            microtubule_ch = img[:, :, _idx[_ch["microtubule"]]].astype(np.float32)
            er_ch          = img[:, :, _idx[_ch["er"]]].astype(np.float32)
            antibody_ch    = img[:, :, _idx[_ch["protein"]]].astype(np.float32)

            landmark_stains = np.stack([nuclei_ch, microtubule_ch, er_ch]) / scale
            antibody_stain  = antibody_ch / scale

            if self.filter_low_values is not None:
                landmark_stains[landmark_stains < self.filter_low_values] = 0

        # ---- label ---------------------------------------------------
        locations = row.get("locations")
        if not isinstance(locations, str):
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label = self.ml_binarizer.transform(
                [locations.split(",")]
            )[0].astype(np.float32)              # (29,)

        # ---- RNA expression ------------------------------------------
        if self.rna_loader is not None:
            atlas_name = row.get("atlas_name", "")
            rna_vec = self.rna_loader.get_expression(str(atlas_name))  # (n_genes,) float32
        else:
            rna_vec = None

        # ---- assemble output -----------------------------------------
        if self.return_antibody and rna_vec is not None:
            return landmark_stains, seq_input, label, antibody_stain, rna_vec
        elif self.return_antibody:
            return landmark_stains, seq_input, label, antibody_stain
        elif rna_vec is not None:
            return landmark_stains, seq_input, label, rna_vec
        else:
            return landmark_stains, seq_input, label
