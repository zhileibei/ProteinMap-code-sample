"""
Cell-line holdout dataset.

Subclasses SubCellDatasetCZI to filter rows to a specific set of cell lines.

These cell lines match PUPS Holdout 1: they were held out from PUPS training
via the CELL_LINE_AND_GENE hash split, so this benchmark tests generalisation
to cell lines the model has never seen during training.
"""

from torch.utils.data import Dataset

from proteinmap.data.dataset_czi import SubCellDatasetCZI
from proteinmap.data.constants import DatasetType

# Held-out cell lines (matching PUPS Holdout 1 — excluded from PUPS training via CELL_LINE_AND_GENE split)
DEFAULT_HOLDOUT_CELL_LINES = [
    "HeLa",
    "MCF7",
    "A431",
    "RH30",
    "PC3",
    "SHSY5Y",
]


class SubCellDatasetCZICellLine(SubCellDatasetCZI):
    """
    SubCellDatasetCZI filtered to a given set of cell lines.

    Args:
        holdout_cell_lines: list of atlas_name values to keep.
            Defaults to DEFAULT_HOLDOUT_CELL_LINES.
        split: DatasetType (TRAIN / EVAL / TEST).  Use TEST for benchmarking.
        **kwargs: forwarded to SubCellDatasetCZI (data_root, esm2_cache_dir).

    Note:
        The parent class filters by antibody split file (train/valid/test_antibodies.txt).
        This subclass further restricts rows by cell-line name in the `atlas_name` column.
    """

    def __init__(
        self,
        holdout_cell_lines: list = None,
        split: DatasetType = DatasetType.TEST,
        **kwargs,
    ):
        super().__init__(split=split, **kwargs)
        if holdout_cell_lines is None:
            holdout_cell_lines = DEFAULT_HOLDOUT_CELL_LINES
        self.holdout_cell_lines = holdout_cell_lines
        self.df = self.df[self.df["atlas_name"].isin(holdout_cell_lines)].reset_index(drop=True)

    def __repr__(self):
        return (
            f"SubCellDatasetCZICellLine("
            f"n={len(self)}, cell_lines={self.holdout_cell_lines})"
        )
