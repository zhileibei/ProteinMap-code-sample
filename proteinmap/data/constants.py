"""
Canonical benchmark constants — class vocabulary and dataset split enum.

Extracted from PUPS/src/dataset/dataset.py (lines 15–53).
No external dependencies; safe to import without PUPS on path.
"""

from enum import Enum


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"


CLASSES = [
    [
        "Nucleoplasm",
        "Cytosol",
        "Vesicles",
        "Plasma membrane",
        "Mitochondria",
        "Golgi apparatus",
        "Endoplasmic reticulum",
        "Nucleoli",
        "Nuclear bodies",
        "Nuclear speckles",
        "Nuclear membrane",
        "Peroxisomes",
        "Microtubules",
        "Centrosome",
        "Cytokinetic bridge",
        "Mitotic chromosome",
        "Centriolar satellite",
        "Focal adhesion sites",
        "Cell Junctions",
        "Lipid droplets",
        "Nucleoli fibrillar center",
        "Actin filaments",
        "Mitotic spindle",
        "Midbody ring",
        "Cytoplasmic bodies",
        "Nucleoli rim",
        "Midbody",
        "Intermediate filaments",
        "Aggresome",
    ]
]
