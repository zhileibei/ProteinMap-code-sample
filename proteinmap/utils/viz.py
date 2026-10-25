"""
Central cell-image visualization utilities.

Convention defaults by file type:
  PNG → "hpa_czi_png" (BGRA)
  NPZ → "misloc_npz" (axis-0 channels)

Display colors always follow the HPA display convention (hpa_display):
  dna=blue, er=yellow, microtubule=red, protein=green
"""

import numpy as np

from proteinmap.data.channel_conventions import CHANNEL_CONVENTIONS, PNG_LOADER_INDEX

def infer_convention(path: str) -> str:
    """Infer the image layout from its file extension."""
    suffix = str(path).lower().rsplit(".", 1)[-1]
    if suffix == "png":
        return "hpa_czi_png"
    if suffix == "npz":
        return "misloc_npz"
    raise ValueError(f"Unknown image format for {path}; pass convention explicitly")


def _normalise(arr: np.ndarray) -> np.ndarray:
    lo = arr.min()
    hi = np.percentile(arr, 99.9)
    return (arr - lo) / (hi - lo + 1e-8)


def _colorize(channel: np.ndarray, color: str) -> np.ndarray:
    """Map a (H, W) float array to a (H, W, 3) float32 RGB array."""
    h, w = channel.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    c = np.abs(channel)
    if color == "blue":
        rgb[:, :, 2] = c
    elif color == "green":
        rgb[:, :, 1] = c
    elif color == "yellow":
        rgb[:, :, 0] = c
        rgb[:, :, 1] = c
    elif color == "red":
        rgb[:, :, 0] = c
    return rgb


def render_cell_image(
    path: str,
    convention: str = None,
    channels: list = None,
) -> np.ndarray:
    """Load a 4-channel cell image and render it as an RGB composite.

    Channels are colored using the HPA display convention (hpa_display):
    DNA=blue, ER=yellow, microtubule=red, protein=green.

    Args:
        path:       Path to a PNG (HPA/CZI) or NPZ (Misloc) file.
        convention: CHANNEL_CONVENTIONS key for the source format.
                    Inferred from file extension if None.
        channels:   List of channel names to include in the composite
                    (e.g. ["dna", "er", "microtubule"] to exclude protein).
                    Defaults to all channels when None.

    Returns:
        (H, W, 3) float32 array in [0, 1].
    """
    if convention is None:
        convention = infer_convention(path)

    ch_map = CHANNEL_CONVENTIONS[convention]
    display = CHANNEL_CONVENTIONS["hpa_display"]

    # Load raw array
    if path.endswith(".npz"):
        arr = np.load(path)["channels"]  # (4, H, W)
        raw_channels = {name: arr[idx].astype(np.float32) for name, idx in ch_map.items()}
    else:
        import cv2
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # (H, W, 4) BGRA
        _idx = PNG_LOADER_INDEX["cv2"]
        raw_channels = {name: raw[:, :, _idx[letter]] for name, letter in ch_map.items()}

    # Filter to requested channels (all by default)
    if channels is not None:
        raw_channels = {k: v for k, v in raw_channels.items() if k in channels}

    # Composite: sum colorized channels, clip to [0, 1]
    sample = next(iter(raw_channels.values()))
    composite = np.zeros((*sample.shape, 3), dtype=np.float32)
    for signal, ch_arr in raw_channels.items():
        color = display[signal]
        composite += _colorize(_normalise(ch_arr), color)
    return np.clip(composite, 0, 1)
