"""
Image reconstruction metrics for the antibody-stain inpainting task.

Public API:
    compute_inpainting_metrics(y_true, y_pred)
        → dict: mse, ssim, pearson_r
"""

import numpy as np
from skimage.metrics import structural_similarity as _skimage_ssim
from scipy.stats import pearsonr as _pearsonr


def compute_inpainting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute image reconstruction metrics for predicted antibody stain images.

    Args:
        y_true: (N, H, W) float32  ground-truth antibody channel
        y_pred: (N, H, W) float32  model-predicted antibody channel

    Returns dict with:
        mse         — mean pixel-wise MSE across all N images
        ssim        — mean SSIM (structural similarity) across images
        pearson_r   — mean Pearson correlation across images
        per_image_mse    — (N,) array
        per_image_ssim   — (N,) array
        per_image_pearson — (N,) array
    """
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
    )
    assert y_true.ndim == 3, "Expected (N, H, W) arrays"

    n = y_true.shape[0]
    per_image_mse = np.zeros(n, dtype=np.float64)
    per_image_ssim = np.zeros(n, dtype=np.float64)
    per_image_pearson = np.zeros(n, dtype=np.float64)

    for i in range(n):
        gt = y_true[i].astype(np.float64)
        pr = y_pred[i].astype(np.float64)

        per_image_mse[i] = float(np.mean((gt - pr) ** 2))

        data_range = max(gt.max() - gt.min(), 1e-6)
        per_image_ssim[i] = float(
            _skimage_ssim(gt, pr, data_range=data_range)
        )

        gt_flat = gt.ravel()
        pr_flat = pr.ravel()
        if gt_flat.std() < 1e-8 or pr_flat.std() < 1e-8:
            per_image_pearson[i] = np.nan
        else:
            r, _ = _pearsonr(gt_flat, pr_flat)
            per_image_pearson[i] = float(r)

    return {
        "mse": float(np.mean(per_image_mse)),
        "ssim": float(np.mean(per_image_ssim)),
        "pearson_r": float(np.nanmean(per_image_pearson)),
        "per_image_mse": per_image_mse,
        "per_image_ssim": per_image_ssim,
        "per_image_pearson": per_image_pearson,
    }
