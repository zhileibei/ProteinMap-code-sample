"""
MislocDataset — wraps Cancer Mutations Screen TIFF crops into the PUPS input format.

Input layout (per .npz file saved by scripts/02_prep_misloc_crops.py):
    ch0: DNA/nuclei   (uint16 → float32 / 65535)
    ch1: Protein      (FLAG-tagged, uint16 → float32 / 65535)
    ch2: ER           (uint16 → float32 / 65535)
    ch3: Mito         (uint16 → float32 / 65535)

CAVEAT (landmark mismatch):
    PUPS expects landmark channels [nuclei, microtubule, ER].
    Misloc provides [DNA, Protein, ER, Mito].
    We substitute Mito (ch3) for microtubule — this is an approximation.

CAVEAT (modality shift):
    HPA data uses endogenous antibody staining; misloc uses FLAG-tagged
    overexpression in U2OS cells.  Performance numbers should be interpreted
    accordingly.
"""

import pathlib
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from proteinmap.data.channel_conventions import CHANNEL_CONVENTIONS
from proteinmap.data.label_mapping import misloc_location_to_multilabel


class MislocDataset(Dataset):
    """
    Dataset of single-cell crops from the Cancer Mutations Screen.

    Args:
        manifest_path: path to the parquet manifest produced by
            scripts/02_prep_misloc_crops.py.  Must contain columns:
            gene, mutation, wt_location, mt_location, npz_path,
            esm2_cache_dir.
        esm2_cache_dir: directory containing {gene}.pkl (WT) and
            {gene}_{mutation}.pkl (mutant) ESM2 embeddings.
        use_mutant_seq: if True, load mutant ESM2 embeddings; otherwise WT.
        landmark_mode: "mito_sub" (default) — substitute Mito for microtubule.
        filter_low_values: zero-out landmark pixels below this threshold
            (matches SubCellDatasetCZI.get_item_verbose behaviour).
    """

    def __init__(
        self,
        manifest_path: str,
        esm2_cache_dir: str,
        use_mutant_seq: bool = False,
        landmark_mode: str = "mito_sub",
        filter_low_values: float = 0.19,
        pairs_csv: str = None,
        keep_genes: set = None,
        keep_mutations: set = None,
    ):
        self.manifest = pd.read_csv(manifest_path)
        self.esm2_cache_dir = pathlib.Path(esm2_cache_dir)
        self.use_mutant_seq = use_mutant_seq
        self.landmark_mode = landmark_mode
        self.filter_low_values = filter_low_values

        # Join location labels from pairs CSV if the manifest columns are empty.
        # pairs_csv has columns: Gene, Mutation, WT_Location, MT_Location.
        # Manifest WT rows have mutation == gene; MT rows have mutation == "GENE MUTCODE".
        if pairs_csv is not None:
            pairs = pd.read_csv(pairs_csv)
            wt_loc_by_gene = (
                pairs.groupby("Gene")["WT_Location"].first().to_dict()
            )
            mt_loc_by_mut = {
                (str(r["Gene"]).upper(), str(r["Mutation"])): str(r["MT_Location"])
                for _, r in pairs.iterrows()
            }
            def _wt_loc(row):
                return wt_loc_by_gene.get(row["gene"].upper(), "")
            def _mt_loc(row):
                return mt_loc_by_mut.get((row["gene"].upper(), row["mutation"]), "")
            self.manifest["wt_location"] = self.manifest.apply(_wt_loc, axis=1)
            self.manifest["mt_location"] = self.manifest.apply(_mt_loc, axis=1)

        # Filter to evaluatable genes/mutations only.
        # For WT mode: keep rows whose gene is in keep_genes.
        # For MT mode: keep rows whose full mutation string is in keep_mutations.
        if keep_genes is not None:
            self.manifest = self.manifest[
                self.manifest["gene"].str.upper().isin(keep_genes)
            ].reset_index(drop=True)
        if keep_mutations is not None:
            self.manifest = self.manifest[
                self.manifest["mutation"].isin(keep_mutations)
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        gene = row["gene"]
        mutation = row["mutation"]
        location_col = "mt_location" if self.use_mutant_seq else "wt_location"
        location_str = row.get(location_col, row.get("wt_location", ""))

        # ------------------------------------------------------------------
        # ESM2 embedding
        # ------------------------------------------------------------------
        if self.use_mutant_seq:
            # manifest stores mutation as "<GENE> <CODE>" (e.g. "TIMM8A C66W");
            # cache files are named "<GENE>_<CODE>.pkl" (e.g. "TIMM8A_C66W.pkl")
            mut_code = mutation.split()[-1]
            emb_key = f"{gene}_{mut_code}"
        else:
            emb_key = gene

        emb_path = self.esm2_cache_dir / f"{emb_key}.pkl"
        try:
            with open(emb_path, "rb") as f:
                emb = pickle.load(f)
            X_esm2 = emb["embedding"][None, ...]  # (1, 2000, 1280)
            X_len = emb["length"]
        except Exception:
            return None

        # Skip if embedding is all-zero (failed or missing)
        if X_esm2.sum() == 0:
            return None

        # ------------------------------------------------------------------
        # Image channels
        # ------------------------------------------------------------------
        try:
            data = np.load(row["npz_path"])
            # channels: (4, 128, 128) float32  [DNA, Protein, ER, Mito]
            channels = data["channels"]
        except Exception:
            return None

        _ch = CHANNEL_CONVENTIONS["misloc_npz"]
        dna_ch     = channels[_ch["dna"]]
        protein_ch = channels[_ch["protein"]]
        er_ch      = channels[_ch["er"]]
        mito_ch    = channels[_ch["mito"]]

        if self.landmark_mode == "mito_sub":
            # [nuclei, mito (≈microtubule), ER]
            X_landmark_stains = np.stack([dna_ch, mito_ch, er_ch], axis=0)  # (3, 128, 128)
        else:
            raise ValueError(f"Unknown landmark_mode: {self.landmark_mode}")

        if self.filter_low_values is not None:
            X_landmark_stains[X_landmark_stains < self.filter_low_values] = 0

        y_antibody_stain = protein_ch  # (128, 128)

        # ------------------------------------------------------------------
        # Label
        # ------------------------------------------------------------------
        y_multilabel = misloc_location_to_multilabel(str(location_str))

        # ------------------------------------------------------------------
        # Metadata (mirroring SubCellDatasetCZI row.to_dict() format)
        # ------------------------------------------------------------------
        meta = {
            "gene": gene,
            "mutation": mutation,
            "wt_location": row.get("wt_location", ""),
            "mt_location": row.get("mt_location", ""),
            "locations": location_str,
            "antibody": f"{gene}_{mutation}",
            "ensembl_ids": row.get("ensembl_id", ""),
            "if_plate_id": row.get("plate", ""),
            "position": row.get("well", ""),
            "sample": row.get("sample", ""),
            "cell_id": row.get("cell_id", ""),
            "atlas_name": "misloc_U2OS",
            "use_mutant_seq": self.use_mutant_seq,
        }

        return (
            (X_esm2, X_len, X_landmark_stains, y_multilabel, y_antibody_stain),
            meta,
        )

    def __repr__(self):
        return (
            f"MislocDataset(n={len(self)}, "
            f"use_mutant={'mutant' if self.use_mutant_seq else 'WT'}, "
            f"landmark_mode={self.landmark_mode})"
        )
