"""
Channel conventions for all image data sources used in ProteinMap.

Four conventions are documented. Always reference the appropriate sub-dict
when loading images to make the source explicit.

For PNG files, hpa_czi_png stores BGRA letters (B/G/R/A). Resolve to integer
indices using PNG_LOADER_INDEX[loader], where loader is "cv2" or "pil".
"""

# Maps each image-loading library to its BGRA letter → array index mapping.
# cv2.imread(IMREAD_UNCHANGED) returns BGRA; PIL.Image.open returns RGBA.
PNG_LOADER_INDEX = {
    "cv2": {"B": 0, "G": 1, "R": 2, "A": 3},
    "pil": {"R": 0, "G": 1, "B": 2, "A": 3},
}

CHANNEL_CONVENTIONS = {
    # Convention 1: how HPA renders staining colors for visualization.
    # Source: HPA image documentation — "The protein of interest is visualized
    # in green, while reference markers for microtubules (red), endoplasmic
    # reticulum (yellow) and nucleus (blue) are used to outline the cell."
    "hpa_display": {
        "protein":     "green",
        "microtubule": "red",
        "er":          "yellow",
        "dna":         "blue",
    },

    # Convention 2: BGRA channel letters for HPA and CZI PNG cell_crops files.
    # Resolve to integer indices via PNG_LOADER_INDEX[loader].
    # Source: CZI SubCell dataset — "microtubule fluorescence in the Blue (B)
    # channel, endoplasmic reticulum fluorescence in the Green (G) channel,
    # DNA fluorescence in the Red (R) channel, and the protein of interest
    # fluorescence in the Alpha (A) channel."
    "hpa_czi_png": {
        "microtubule": "B",
        "er":          "G",
        "dna":         "R",
        "protein":     "A",
    },

    # Convention 3: raw microscope channel numbers per load_data.csv
    # (URL_Orig* columns from cpg0026-lacoste_haghighi-rare-diseases).
    # Staining: Hoechst (nucleus), anti-FLAG (protein), MitoTracker, concanavalin A (ER).
    # NOTE: CSV column order is (DNA, ER, Mito, Protein) and does NOT match
    # the microscope channel numbers below.
    "misloc_csv": {
        "dna":     1,  # ch1, URL_OrigDNA,     DAPI/Hoechst
        "protein": 2,  # ch2, URL_OrigProtein, anti-FLAG antibody
        "mito":    3,  # ch3, URL_OrigMito,    MitoTracker
        "er":      4,  # ch4, URL_OrigER,      concanavalin A
    },

    # Convention 4: axis-0 indices in the 'channels' array of preprocessed
    # Misloc .npz files.
    # Source: scripts/archive/02_prep_misloc_crops.py:379 —
    # stacked as enumerate(["dna", "protein", "er", "mito"]);
    # CSV columns are matched by name keyword, not by column position.
    "misloc_npz": {
        "dna":     0,
        "protein": 1,
        "er":      2,
        "mito":    3,
    },

    # Convention 5: axis-0 indices in the 'x' array of PerturbMulti .npz files.
    # Source: xingjiepan/PerturbMulti HuggingFace dataset (CC-BY-4.0).
    # 18-channel multiplex imaging of HeLa/liver cells under CRISPR / diet perturbations.
    "perturb_multi_npz": {
        "alb":          0,
        "polyt":        1,
        "rrna":         2,
        "m6pr":         3,
        "cathb":        4,
        "perilipin":    5,
        "sqstm1":       6,
        "lc3b":         7,
        "tomm20":       8,
        "calreticulin": 9,
        "ps6rp":        10,
        "na_k_atpase":  11,
        "snap23":       12,
        "tom70":        13,
        "rab7":         14,
        "mtrna":        15,
        "vimentin":     16,
        "gapdh":        17,
    },

    # Model input order for the 3-channel landmark tensor passed to the vision encoder.
    # For Misloc data, "mito" substitutes for "microtubule" (no microtubule channel available).
    "landmark_stack": {
        "nuclei":      0,
        "microtubule": 1,
        "er":          2,
    },
}
