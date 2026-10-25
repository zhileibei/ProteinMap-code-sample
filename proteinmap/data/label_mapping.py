"""
Vocabulary mapping: misloc location strings and ProtGPS class names → PUPS CLASSES.

PUPS CLASSES (index 0-28):
  0  Nucleoplasm          10 Nuclear membrane      20 Nucleoli fibrillar center
  1  Cytosol              11 Peroxisomes            21 Actin filaments
  2  Vesicles             12 Microtubules           22 Mitotic spindle
  3  Plasma membrane      13 Centrosome             23 Midbody ring
  4  Mitochondria         14 Cytokinetic bridge     24 Cytoplasmic bodies
  5  Golgi apparatus      15 Mitotic chromosome     25 Nucleoli rim
  6  Endoplasmic reticulum 16 Centriolar satellite  26 Midbody
  7  Nucleoli             17 Focal adhesion sites   27 Intermediate filaments
  8  Nuclear bodies       18 Cell Junctions         28 Aggresome
  9  Nuclear speckles     19 Lipid droplets
"""

import numpy as np

from proteinmap.data.constants import CLASSES

# Canonical PUPS class list (same as CLASSES[0] in constants.py)
PUPS_CLASSES = CLASSES[0]

PUPS_CLASS_INDEX = {c: i for i, c in enumerate(PUPS_CLASSES)}

# ---------------------------------------------------------------------------
# Misloc → PUPS mapping
# Misloc uses location names from the HPA/OpenCell ontology; most are identical
# to PUPS labels or require minor normalisation.
# ---------------------------------------------------------------------------
MISLOC_TO_PUPS = {
    # Direct matches (case variants)
    "nucleoplasm": "Nucleoplasm",
    "cytoplasm": "Cytosol",
    "cytosol": "Cytosol",
    "vesicles": "Vesicles",
    "plasma membrane": "Plasma membrane",
    "mitochondria": "Mitochondria",
    "golgi apparatus": "Golgi apparatus",
    "golgi": "Golgi apparatus",
    "endoplasmic reticulum": "Endoplasmic reticulum",
    "er": "Endoplasmic reticulum",
    "nucleoli": "Nucleoli",
    "nuclear bodies": "Nuclear bodies",
    "nuclear speckles": "Nuclear speckles",
    "nuclear membrane": "Nuclear membrane",
    "peroxisomes": "Peroxisomes",
    "microtubules": "Microtubules",
    "centrosome": "Centrosome",
    "cytokinetic bridge": "Cytokinetic bridge",
    "mitotic chromosome": "Mitotic chromosome",
    "centriolar satellite": "Centriolar satellite",
    "focal adhesion sites": "Focal adhesion sites",
    "focal adhesions": "Focal adhesion sites",
    "cell junctions": "Cell Junctions",
    "lipid droplets": "Lipid droplets",
    "nucleoli fibrillar center": "Nucleoli fibrillar center",
    "actin filaments": "Actin filaments",
    "mitotic spindle": "Mitotic spindle",
    "midbody ring": "Midbody ring",
    "cytoplasmic bodies": "Cytoplasmic bodies",
    "nucleoli rim": "Nucleoli rim",
    "midbody": "Midbody",
    "intermediate filaments": "Intermediate filaments",
    "aggresome": "Aggresome",
    # Common misloc-specific aliases
    "nucleus": "Nucleoplasm",
    "nuclear": "Nucleoplasm",
    "cytoplasmic": "Cytosol",
    "mito": "Mitochondria",
    "ER": "Endoplasmic reticulum",
}

# ---------------------------------------------------------------------------
# ProtGPS → PUPS mapping
# ProtGPS (https://github.com/HazyResearch/ProtGPS) uses a 10-compartment vocab
# ---------------------------------------------------------------------------
PROTGPS_TO_PUPS = {
    "nucleus": ["Nucleoplasm"],
    "cytoplasm": ["Cytosol"],
    "mitochondria": ["Mitochondria"],
    "er": ["Endoplasmic reticulum"],
    "golgi": ["Golgi apparatus"],
    "plasma_membrane": ["Plasma membrane"],
    "secreted": ["Vesicles"],          # nearest approximation
    "lysosome": ["Vesicles"],          # nearest approximation
    "peroxisome": ["Peroxisomes"],
    "nuclear_membrane": ["Nuclear membrane"],
}


def misloc_location_to_multilabel(location_str: str) -> np.ndarray:
    """
    Convert a misloc location string (comma-separated) to a 29-dim binary vector.

    Unknown tokens are silently ignored — callers should log if vector is all-zero.
    """
    vec = np.zeros(len(PUPS_CLASSES), dtype=np.int32)
    if not isinstance(location_str, str):
        return vec
    for token in location_str.split(","):
        key = token.strip().lower()
        pups_name = MISLOC_TO_PUPS.get(key)
        if pups_name is not None and pups_name in PUPS_CLASS_INDEX:
            vec[PUPS_CLASS_INDEX[pups_name]] = 1
    return vec


def protgps_label_to_multilabel(protgps_labels: list[str]) -> np.ndarray:
    """Convert a list of ProtGPS class names to a 29-dim binary PUPS vector."""
    vec = np.zeros(len(PUPS_CLASSES), dtype=np.int32)
    for label in protgps_labels:
        pups_names = PROTGPS_TO_PUPS.get(label.lower(), [])
        for pups_name in pups_names:
            if pups_name in PUPS_CLASS_INDEX:
                vec[PUPS_CLASS_INDEX[pups_name]] = 1
    return vec
