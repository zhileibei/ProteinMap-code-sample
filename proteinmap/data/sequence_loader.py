"""
Unified sequence feature loader for ProteinMap.

Supports two modes:
  'esm2_pooled' — loads mean-pooled ESM2 vectors from a pre-built pickle cache
                  {ensembl_id: (1280,) np.ndarray}
  'one_hot'     — loads raw AA sequences from a SwissProt TSV, encodes on-the-fly

TSV join strategy: The SwissProt TSV 'Ensembl' column has semicolon-separated ENSG IDs.
We split on ';' and explode to one row per ENSG to maximise coverage.
HPA metadata uses the first comma-separated ENSG as primary ID.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"   # 20 standard AAs
AA_TO_IDX   = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ONEHOT_DIM  = 20
MAX_SEQ_LEN = 2000   # matches ESM2 cache padding


@dataclass
class SequenceRecord:
    ensembl_id:  str
    esm2_pooled: Optional[np.ndarray]   # (1280,) float32; None if not loaded/found
    one_hot:     Optional[np.ndarray]   # (MAX_SEQ_LEN, 20) float32; None if not loaded/found


def load_tsv_sequence_map(tsv_path: str) -> dict[str, str]:
    """
    Parse a SwissProt TSV and return {ensembl_id: aa_sequence}.

    The 'Ensembl' column may contain semicolon-separated ENSG IDs.
    Each ID gets its own entry pointing to the same sequence.
    """
    df = pd.read_csv(tsv_path, sep="\t")

    # Detect column names (case-insensitive)
    col_lower = {c.lower(): c for c in df.columns}
    ensembl_col = col_lower.get("ensembl")
    seq_col = col_lower.get("sequence") or col_lower.get("aa_sequence")

    if ensembl_col is None or seq_col is None:
        raise ValueError(
            f"TSV must have 'Ensembl' and 'Sequence' (or 'AA_sequence') columns; "
            f"found: {list(df.columns)}"
        )

    seq_map: dict[str, str] = {}
    for _, row in df.iterrows():
        raw_ids = str(row[ensembl_col])
        seq = str(row[seq_col]).strip()
        for eid in raw_ids.split(";"):
            eid = eid.strip()
            if eid and eid.lower() != "nan":
                seq_map[eid] = seq
    return seq_map


def one_hot_encode(sequence: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Encode an amino-acid string to a (max_len, 20) float32 matrix.
    Zero-padded at the end; unknown AAs produce an all-zero row.
    """
    out = np.zeros((max_len, ONEHOT_DIM), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_len]):
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            out[i, idx] = 1.0
    return out


class SequenceLoader:
    """
    Unified sequence feature loader.

    mode='esm2_pooled': loads from a pre-built pooled_esm2.pkl
                        {ensembl_id: (1280,) np.ndarray}
    mode='one_hot':     loads raw AA from SwissProt TSV, encodes on the fly

    Usage:
        loader = SequenceLoader('esm2_pooled', pooled_cache_path='/path/to/pooled_esm2.pkl')
        rec = loader['ENSG00000123456']   # → SequenceRecord
        stats = loader.coverage_stats(['ENSG00000123456', ...])
    """

    def __init__(
        self,
        mode: str,
        pooled_cache_path: str = None,
        tsv_path: str = None,
    ):
        if mode not in ("esm2_pooled", "one_hot"):
            raise ValueError(f"mode must be 'esm2_pooled' or 'one_hot', got {mode!r}")
        self.mode = mode

        self._pooled: Optional[dict] = None
        self._seq_map: Optional[dict] = None

        if mode == "esm2_pooled":
            if pooled_cache_path is None:
                raise ValueError("pooled_cache_path is required for mode='esm2_pooled'")
            print(f"[SequenceLoader] Loading pooled ESM2 cache from {pooled_cache_path} ...")
            if pooled_cache_path.endswith(".joblib"):
                import joblib
                raw = joblib.load(pooled_cache_path)
                # Values may be torch.Tensor — convert to np.ndarray
                self._pooled = {}
                for k, v in raw.items():
                    if hasattr(v, "cpu"):
                        v = v.cpu()
                    if hasattr(v, "numpy"):
                        v = v.numpy()
                    self._pooled[k] = np.asarray(v, dtype=np.float32)
            else:
                with open(pooled_cache_path, "rb") as f:
                    self._pooled = pickle.load(f)
            print(f"  {len(self._pooled)} entries loaded.")

        else:  # one_hot
            if tsv_path is None:
                raise ValueError("tsv_path is required for mode='one_hot'")
            print(f"[SequenceLoader] Loading TSV sequence map from {tsv_path} ...")
            self._seq_map = load_tsv_sequence_map(tsv_path)
            print(f"  {len(self._seq_map)} unique Ensembl IDs mapped.")

    def __getitem__(self, ensembl_id: str) -> SequenceRecord:
        if self.mode == "esm2_pooled":
            vec = self._pooled.get(ensembl_id)
            return SequenceRecord(
                ensembl_id=ensembl_id,
                esm2_pooled=np.asarray(vec, dtype=np.float32) if vec is not None else None,
                one_hot=None,
            )
        else:
            seq = self._seq_map.get(ensembl_id)
            return SequenceRecord(
                ensembl_id=ensembl_id,
                esm2_pooled=None,
                one_hot=one_hot_encode(seq) if seq is not None else None,
            )

    def coverage_stats(self, ensembl_ids: list[str]) -> dict:
        """
        Report hit/miss statistics for a list of Ensembl IDs.
        Returns {total, found, missing_fraction} — useful for sanity checks.
        """
        lookup = self._pooled if self.mode == "esm2_pooled" else self._seq_map
        total = len(ensembl_ids)
        found = sum(1 for eid in ensembl_ids if eid in lookup)
        return {
            "total": total,
            "found": found,
            "missing_fraction": (total - found) / total if total > 0 else 0.0,
        }
