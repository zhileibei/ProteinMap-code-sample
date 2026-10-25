"""
Compartment prediction metrics (multi-label classification).

Public API:
    compute_compartment_metrics(y_true, y_proba, y_pred, class_names)
        → dict of scalar metrics + per-class DataFrame

    compute_mutation_shift_detection(wt_preds_df, mt_preds_df, location_pairs_df)
        → dict with wt_localization_accuracy, mt_localization_accuracy,
               shift_detected_rate
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_compartment_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> dict:
    """
    Compute macro/micro AUROC, AUPRC, F1, and exact-match accuracy for a
    multi-label classification problem.

    Args:
        y_true:  (N, C) int32 binary ground-truth matrix
        y_proba: (N, C) float32 predicted probabilities
        y_pred:  (N, C) int32 binary predicted labels
        class_names: list of C class-name strings

    Returns dict with keys:
        macro_auroc, macro_auprc, macro_f1, micro_f1, sample_f1,
        macro_precision, macro_recall, exact_match_accuracy,
        per_class_df  (DataFrame: class | auroc | auprc | f1 | support)
    """
    assert y_true.shape == y_proba.shape == y_pred.shape
    n_classes = len(class_names)
    assert y_true.shape[1] == n_classes

    # AUROC requires both positive and negative examples in each class.
    positives = y_true.sum(axis=0)
    valid_mask = (positives > 0) & (positives < len(y_true))
    valid_idx = np.where(valid_mask)[0]

    per_class_auroc = np.full(n_classes, np.nan)
    per_class_auprc = np.full(n_classes, np.nan)
    per_class_f1 = np.full(n_classes, np.nan)

    for i in valid_idx:
        per_class_auroc[i] = roc_auc_score(y_true[:, i], y_proba[:, i])
        per_class_auprc[i] = average_precision_score(y_true[:, i], y_proba[:, i])
        per_class_f1[i] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

    macro_auroc = float(np.nanmean(per_class_auroc)) if len(valid_idx) else float("nan")
    macro_auprc = float(np.nanmean(per_class_auprc)) if len(valid_idx) else float("nan")

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    sample_f1 = float(f1_score(y_true, y_pred, average="samples", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Exact-match accuracy (all labels must match)
    exact_match_accuracy = float(np.all(y_true == y_pred, axis=1).mean())

    per_class_df = pd.DataFrame({
        "class": class_names,
        "auroc": per_class_auroc,
        "auprc": per_class_auprc,
        "f1": per_class_f1,
        "support": y_true.sum(axis=0).astype(int),
    })

    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "sample_f1": sample_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "exact_match_accuracy": exact_match_accuracy,
        "per_class_df": per_class_df,
    }


def compute_mutation_shift_detection(
    wt_preds_df: pd.DataFrame,
    mt_preds_df: pd.DataFrame,
    location_pairs_df: pd.DataFrame,
    class_names: list,
) -> dict:
    """
    Evaluate whether the model detects gain/loss of localization in mutants.

    Args:
        wt_preds_df:  DataFrame with columns [gene, mutation, y_pred_binary,
                       y_true_multilabel]  (WT sequences)
        mt_preds_df:  same schema but with mutant ESM2 embeddings
        location_pairs_df:  the 194-row ground-truth CSV with columns
                             [Gene, Mutation, WT_location, MT_location]
                             indicating the expected shift.
        class_names:  list of 29 PUPS class names

    Returns dict with keys:
        wt_localization_accuracy   — fraction of WT predictions that include
                                     the known WT location compartment
        mt_localization_accuracy   — fraction of MT predictions that include
                                     the known MT location compartment
        shift_detected_rate        — fraction of pairs where the model's top
                                     predicted class changed between WT and MT
        n_pairs                    — number of evaluated pairs
    """
    from proteinmap.data.label_mapping import PUPS_CLASS_INDEX, MISLOC_TO_PUPS

    def location_to_idx(loc_str: str) -> int:
        key = str(loc_str).strip().lower()
        pups_name = MISLOC_TO_PUPS.get(key)
        if pups_name is None:
            return -1
        return PUPS_CLASS_INDEX.get(pups_name, -1)

    n_total = 0
    wt_correct = 0
    mt_correct = 0
    shift_detected = 0

    # WT rows have mutation == gene (WT well label); index by gene only and
    # average proba / majority-vote binary across all WT cells for that gene.
    wt_by_gene = {}
    for _, r in wt_preds_df.iterrows():
        g = str(r["gene"]).upper()
        wt_by_gene.setdefault(g, []).append(r)

    def _agg_wt(rows):
        proba = np.stack([r["y_pred_proba"] for r in rows]).mean(axis=0)
        binary = (proba >= 0.5).astype(np.int32)
        return proba, binary

    # MT rows have mutation == "GENE MUTCODE"; index by (gene, mutation).
    mt_index = {
        (str(r["gene"]).upper(), str(r["mutation"])): r
        for _, r in mt_preds_df.iterrows()
    }

    for _, pair in location_pairs_df.iterrows():
        gene = str(pair.get("Gene", pair.get("gene", ""))).upper()
        mutation = str(pair.get("Mutation", pair.get("mutation", "")))
        wt_loc = str(pair.get("WT_Location", pair.get("WT_location", pair.get("wt_location", ""))))
        mt_loc = str(pair.get("MT_Location", pair.get("MT_location", pair.get("mt_location", ""))))

        if gene not in wt_by_gene or (gene, mutation) not in mt_index:
            continue

        n_total += 1
        wt_proba, wt_pred = _agg_wt(wt_by_gene[gene])
        mt_row = mt_index[(gene, mutation)]
        mt_pred = mt_row["y_pred_binary"]   # (29,) int
        mt_proba = mt_row["y_pred_proba"]   # (29,) float

        # WT localization accuracy: top-1 predicted class matches known WT compartment
        wt_idx = location_to_idx(wt_loc)
        if wt_idx >= 0 and np.argmax(wt_proba) == wt_idx:
            wt_correct += 1

        # MT localization accuracy: top-1 predicted class matches known MT compartment
        mt_idx = location_to_idx(mt_loc)
        if mt_idx >= 0 and np.argmax(mt_proba) == mt_idx:
            mt_correct += 1

        # Shift detected: top predicted class changes between WT and MT
        if np.argmax(wt_proba) != np.argmax(mt_proba):
            shift_detected += 1

    if n_total == 0:
        return {
            "wt_localization_accuracy": np.nan,
            "mt_localization_accuracy": np.nan,
            "shift_detected_rate": np.nan,
            "n_pairs": 0,
        }

    return {
        "wt_localization_accuracy": wt_correct / n_total,
        "mt_localization_accuracy": mt_correct / n_total,
        "shift_detected_rate": shift_detected / n_total,
        "n_pairs": n_total,
    }
