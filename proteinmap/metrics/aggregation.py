"""
Per-protein aggregation and bootstrap confidence intervals for ProteinMap metrics.

Formalises the ad-hoc logic in scripts/09_pups_agg_results.py and makes it reusable.

Public API:
    aggregate_per_protein(results_df)      → protein-level DataFrame
    bootstrap_ci(y_true, y_proba, fn, ...) → {mean, std, ci_low, ci_high, n_bootstrap}
    compute_metrics_with_ci(results_df)    → {per_cell: {...}, per_protein: {...}}
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from proteinmap.metrics.compartment_metrics import compute_compartment_metrics
from proteinmap.data.constants import CLASSES as _PUPS_CLASSES_NESTED

PUPS_CLASS_NAMES = _PUPS_CLASSES_NESTED[0]


# ---------------------------------------------------------------------------
# Per-protein aggregation
# ---------------------------------------------------------------------------

def aggregate_per_protein(
    results_df: pd.DataFrame,
    proba_col: str = "y_pred_proba",
    label_col: str = "y_true_multilabel",
    id_col:    str = "ensembl_ids",
) -> pd.DataFrame:
    """
    Aggregate per-cell predictions to one row per protein.

    Normalises ensembl_id: first item in comma-separated list (matches script 09 exactly).
    y_pred_proba:      mean over cells of the same protein.
    y_true_multilabel: union (elementwise max) over cells — matches ESM2 baseline
                       label construction.

    Returns DataFrame with one row per unique primary ensembl_id.
    """
    df = results_df.copy()
    df["_ensembl_primary"] = (
        df[id_col].str.strip().str.split(",").str[0].str.strip()
    )

    agg = (
        df.groupby("_ensembl_primary")
        .agg(
            **{
                proba_col: (proba_col, lambda x: np.stack(x.values).mean(0)),
                label_col: (label_col, lambda x: np.stack(x.values).max(0).astype(np.int32)),
            }
        )
        .reset_index()
        .rename(columns={"_ensembl_primary": "ensembl_ids"})
    )
    return agg


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true:      np.ndarray,
    y_proba:     np.ndarray,
    metric_fn:   Callable,
    n_bootstrap: int = 1000,
    confidence:  float = 0.95,
    seed:        int = 42,
    protein_ids: Optional[np.ndarray] = None,
) -> dict:
    """
    Bootstrap confidence interval for a scalar metric.

    IID unit: proteins (not cells).
    - protein_ids provided (per-cell data): clustered resampling — all cells of a
      sampled protein move together. Gives correctly wider CIs than naive cell-level.
    - protein_ids is None (per-protein data): row-level resampling.

    Args:
        y_true:      (N, C) int ground-truth labels
        y_proba:     (N, C) float predicted probabilities
        metric_fn:   (y_true_subset, y_proba_subset) → float
        n_bootstrap: number of bootstrap samples
        confidence:  width of CI, e.g. 0.95
        seed:        random seed for reproducibility
        protein_ids: (N,) array of protein/group IDs (for clustered resampling)

    Returns:
        {mean, std, ci_low, ci_high, n_bootstrap}
    """
    rng = np.random.default_rng(seed)
    scores: list[float] = []

    if protein_ids is not None:
        unique_proteins = np.unique(protein_ids)
        n_proteins = len(unique_proteins)
        prot_idx: dict = {uid: np.where(protein_ids == uid)[0] for uid in unique_proteins}

        for _ in range(n_bootstrap):
            sampled = rng.choice(unique_proteins, size=n_proteins, replace=True)
            idx = np.concatenate([prot_idx[uid] for uid in sampled])
            score = float(metric_fn(y_true[idx], y_proba[idx]))
            if np.isfinite(score):
                scores.append(score)
    else:
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            score = float(metric_fn(y_true[idx], y_proba[idx]))
            if np.isfinite(score):
                scores.append(score)

    if not scores:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_bootstrap": 0}

    arr = np.array(scores)
    alpha = 1.0 - confidence
    return {
        "mean":        float(np.mean(arr)),
        "std":         float(np.std(arr)),
        "ci_low":      float(np.percentile(arr, 100 * alpha / 2)),
        "ci_high":     float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "n_bootstrap": len(scores),
    }


# ---------------------------------------------------------------------------
# Metric functions for CI
# ---------------------------------------------------------------------------

def _macro_auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    positives = y_true.sum(axis=0)
    valid = (positives > 0) & (positives < len(y_true))
    if valid.sum() == 0:
        return np.nan
    return float(roc_auc_score(y_true[:, valid], y_proba[:, valid], average="macro"))


def _macro_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    y_pred = (y_proba >= 0.5).astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _micro_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    y_pred = (y_proba >= 0.5).astype(int)
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


_METRIC_FNS: dict[str, Callable] = {
    "macro_auroc": _macro_auroc,
    "macro_f1":    _macro_f1,
    "micro_f1":    _micro_f1,
}


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_metrics_with_ci(
    results_df:  pd.DataFrame,
    mode:        str = "per_cell",
    n_bootstrap: int = 1000,
    ci_metrics:  tuple = ("macro_auroc", "macro_f1", "micro_f1"),
) -> dict:
    """
    Single entry point for all metrics (compartment + inpainting + CI).

    Steps:
      1. Optional per-protein aggregation (aggregate_per_protein)
      2. compute_compartment_metrics() [reused from compartment_metrics.py]
      3. bootstrap_ci() for ci_metrics — protein-clustered for per-cell data,
         direct row-level for per-protein data
      4. compute_inpainting_metrics() if y_antibody_pred column is present

    mode: 'per_cell' | 'per_protein' | 'both'

    Returns:
    {
      "per_cell":    {macro_auroc, ..., ci: {metric: {mean, ci_low, ci_high}, ...}} or None
      "per_protein": same or None
    }
    """
    from proteinmap.metrics.inpainting_metrics import compute_inpainting_metrics

    def _compute_for_df(df: pd.DataFrame, clustered_ids: Optional[np.ndarray]) -> dict:
        y_true  = np.stack(df["y_true_multilabel"].values)   # (N, C)
        y_proba = np.stack(df["y_pred_proba"].values)         # (N, C)
        y_pred  = (y_proba >= 0.5).astype(int)

        metrics = compute_compartment_metrics(y_true, y_proba, y_pred, PUPS_CLASS_NAMES)

        # Bootstrap CIs
        ci_dict: dict = {}
        if n_bootstrap > 0:
            for name in ci_metrics:
                fn = _METRIC_FNS.get(name)
                if fn is None:
                    continue
                ci_dict[name] = bootstrap_ci(
                    y_true, y_proba, fn,
                    n_bootstrap=n_bootstrap,
                    protein_ids=clustered_ids,
                )
        metrics["ci"] = ci_dict

        # Inpainting metrics (if antibody predictions are present)
        if "y_antibody_pred" in df.columns:
            valid = df["y_antibody_pred"].apply(lambda v: v is not None)
            if valid.any():
                ab_true = np.stack(df.loc[valid, "y_antibody_true"].values)
                ab_pred = np.stack(df.loc[valid, "y_antibody_pred"].values)
                inp = compute_inpainting_metrics(ab_true, ab_pred)
                metrics["inpainting_mse"]       = inp["mse"]
                metrics["inpainting_ssim"]      = inp["ssim"]
                metrics["inpainting_pearson_r"] = inp["pearson_r"]

        return metrics

    output: dict = {"per_cell": None, "per_protein": None}

    if mode in ("per_cell", "both"):
        protein_ids = None
        if "ensembl_ids" in results_df.columns:
            protein_ids = (
                results_df["ensembl_ids"]
                .str.strip().str.split(",").str[0].str.strip()
                .values
            )
        output["per_cell"] = _compute_for_df(results_df, clustered_ids=protein_ids)

    if mode in ("per_protein", "both"):
        protein_df = aggregate_per_protein(results_df)
        # Per-protein data — each row is one protein, no clustering needed
        output["per_protein"] = _compute_for_df(protein_df, clustered_ids=None)

    return output
