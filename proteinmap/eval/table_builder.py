"""
Table builder — aggregates JSON metrics from all eval runs into the two
comparison tables described in plan.md.

Table 1: Compartment prediction
    Model | Split | macro-AUROC | macro-AUPRC | macro-F1 | Exact-match

Table 2: Image inpainting (PUPS only)
    Model | Split | MSE | SSIM | Pearson r

Usage:
    python scripts/06_build_tables.py --results_dir results/ --output results/tables.md
"""

import json
import pathlib
from typing import Optional

import pandas as pd


def _load_metrics(results_dir: str) -> list[dict]:
    """Recursively load all *_metrics.json files under results_dir."""
    root = pathlib.Path(results_dir)
    metrics = []
    for path in sorted(root.rglob("*_metrics.json")):
        with open(path) as f:
            m = json.load(f)
        metrics.append(m)
    return metrics


def build_compartment_table(results_dir: str) -> pd.DataFrame:
    """
    Build Table 1: compartment prediction comparison.

    Returns a DataFrame ready for printing/saving.
    """
    metrics_list = _load_metrics(results_dir)
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m.get("model", "?"),
            "Split": _format_split(m.get("split", "?")),
            "N": m.get("n_samples", "?"),
            "macro-AUROC": _fmt(m.get("macro_auroc")),
            "macro-AUPRC": _fmt(m.get("macro_auprc")),
            "macro-F1": _fmt(m.get("macro_f1")),
            "micro-F1": _fmt(m.get("micro_f1")),
            "Exact-match": _fmt(m.get("exact_match_accuracy")),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort: model alphabetical, then split order
    split_order = ["hpa_proteins", "hpa_cell_lines", "misloc_wt", "misloc_mt"]
    df["_split_rank"] = df["Split"].map(
        lambda s: next((i for i, k in enumerate(split_order) if k in s.lower()), 99)
    )
    df = df.sort_values(["Model", "_split_rank"]).drop(columns=["_split_rank"])
    return df


def build_inpainting_table(results_dir: str) -> pd.DataFrame:
    """
    Build Table 2: inpainting metrics (PUPS only).

    Returns a DataFrame ready for printing/saving.
    """
    metrics_list = _load_metrics(results_dir)
    rows = []
    for m in metrics_list:
        if "inpainting_mse" not in m and "oracle_pred_auroc" not in m:
            continue
        rows.append({
            "Model": m.get("model", "?"),
            "Split": _format_split(m.get("split", "?")),
            "N": m.get("n_samples", "?"),
            "MSE": _fmt(m.get("inpainting_mse"), 6),
            "SSIM": _fmt(m.get("inpainting_ssim")),
            "Pearson r": _fmt(m.get("inpainting_pearson_r")),
            "Oracle AUROC (true)": _fmt(m.get("oracle_true_auroc")),
            "Oracle AUROC (pred)": _fmt(m.get("oracle_pred_auroc")),
            "Oracle proba L1": _fmt(m.get("oracle_proba_l1")),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def save_tables_markdown(
    results_dir: str,
    output_path: str,
    compartment_df: Optional[pd.DataFrame] = None,
    inpainting_df: Optional[pd.DataFrame] = None,
) -> None:
    """Write both tables to a markdown file."""
    if compartment_df is None:
        compartment_df = build_compartment_table(results_dir)
    if inpainting_df is None:
        inpainting_df = build_inpainting_table(results_dir)

    lines = ["# Benchmark Results\n"]

    lines.append("## Table 1: Compartment Prediction\n")
    if not compartment_df.empty:
        lines.append(compartment_df.to_markdown(index=False))
    else:
        lines.append("_No results found._")
    lines.append("\n")

    lines.append("## Table 2: Image Inpainting (PUPS)\n")
    if not inpainting_df.empty:
        lines.append(inpainting_df.to_markdown(index=False))
    else:
        lines.append("_No inpainting results found._")
    lines.append("\n")

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Tables saved to: {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.{decimals}f}"


def _format_split(split_name: str) -> str:
    mapping = {
        "hpa_proteins": "HPA held-out proteins",
        "hpa_cell_lines": "HPA held-out cell lines",
        "misloc_wt": "Misloc WT",
        "misloc_mt": "Misloc mutant",
    }
    return mapping.get(split_name, split_name)
