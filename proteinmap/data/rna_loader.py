"""
RNAExpressionLoader — returns bulk RNA expression vector for a given cell line.

Reads rna_celline.tsv which is in long format:
  Gene | Gene name | Cell line | TPM | pTPM | nTPM | ...

Pivoted internally to: genes (rows) × cell lines (cols) for fast lookup.
Indexed by atlas_name (cell-line-specific, not protein-specific).
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd


def _normalize_cell_line(name: str) -> str:
    """Lower-case, strip whitespace, remove non-alphanumeric for fuzzy match."""
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())


class RNAExpressionLoader:
    """
    Return bulk RNA expression vector for a given cell line.

    The TSV is long-format (Gene × Cell line rows).  It is pivoted to a wide
    matrix on load so lookups are O(1).

    Parameters
    ----------
    tsv_path : str
        Path to rna_celline.tsv.
    value_col : str
        Which expression column to use (default 'nTPM'; falls back to 'TPM').
    log1p : bool
        If True, apply log1p transform to the expression values (default True).
    """

    def __init__(self, tsv_path: str, value_col: str = "nTPM", log1p: bool = True):
        print(f"[RNAExpressionLoader] Loading {tsv_path} ...")
        df = pd.read_csv(tsv_path, sep="\t")

        # Auto-detect value column
        if value_col not in df.columns:
            for fallback in ("TPM", "pTPM", "nTPM"):
                if fallback in df.columns:
                    value_col = fallback
                    break

        # Pivot: rows = Gene, cols = Cell line
        wide = df.pivot_table(
            index="Gene", columns="Cell line", values=value_col, aggfunc="mean"
        )
        wide = wide.fillna(0.0)

        if log1p:
            wide = np.log1p(wide)

        self.gene_ids: List[str] = list(wide.index.astype(str))
        self.n_genes: int = len(self.gene_ids)

        # Build normalised lookup: {norm_name: np.ndarray(n_genes,) float32}
        self._mat: dict[str, np.ndarray] = {}
        for col in wide.columns:
            self._mat[_normalize_cell_line(str(col))] = wide[col].values.astype(np.float32)

        print(f"  {self.n_genes} genes x {len(self._mat)} cell lines loaded.")

    def get_expression(self, atlas_name: str) -> np.ndarray:
        """
        Return (n_genes,) float32 vector for the given cell line atlas_name.
        Returns zero vector if the cell line is not found.
        """
        key = _normalize_cell_line(atlas_name)
        vec = self._mat.get(key)
        if vec is None:
            return np.zeros(self.n_genes, dtype=np.float32)
        return vec
