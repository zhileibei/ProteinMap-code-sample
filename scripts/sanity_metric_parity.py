"""
Sanity check 3 — verify new metrics match old Evaluator output.

Checks:
1. Load an existing per-cell results.pkl (e.g. results/image_seq_baseline/)
2. compute_metrics_with_ci(mode='per_cell', n_bootstrap=0) matches
   the existing hpa_proteins_metrics.json to within 1e-4
3. compute_metrics_with_ci(mode='per_protein') matches
   the 09_pups_agg_results.py output JSON (if available)
4. Assert CI ordering: ci_low < mean < ci_high for all metrics
5. Assert CI width shrinks with more proteins (50% subsample)

Usage:
    python scripts/sanity_metric_parity.py \\
        --results_pkl results/image_seq_baseline/hpa_proteins_results.pkl \\
        --existing_metrics results/image_seq_baseline/hpa_proteins_metrics.json \\
        [--protein_metrics results/image_seq_inpainting/hpa_proteins_aggregated_metrics.json]
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from proteinmap.metrics.aggregation import compute_metrics_with_ci


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_pkl",       required=True,
                   help="Per-cell results DataFrame (.pkl)")
    p.add_argument("--existing_metrics",  required=True,
                   help="Existing per-cell metrics JSON from Evaluator")
    p.add_argument("--protein_metrics",   default=None,
                   help="Existing per-protein metrics JSON from script 09 (optional)")
    p.add_argument("--n_bootstrap",       type=int, default=200,
                   help="Bootstrap samples for CI checks (low number for speed)")
    p.add_argument("--tol",              type=float, default=1e-4,
                   help="Tolerance for metric parity check")
    return p.parse_args()


def main():
    args = parse_args()
    passed = 0
    failed = 0

    def check(condition: bool, msg: str):
        nonlocal passed, failed
        if condition:
            print(f"  [OK]  {msg}")
            passed += 1
        else:
            print(f"  [FAIL] {msg}")
            failed += 1

    print("=" * 60)
    print("Sanity 3: Metric parity")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading {args.results_pkl} ...")
    results_df = pd.read_pickle(args.results_pkl)
    print(f"  {len(results_df)} rows loaded")
    check(len(results_df) > 0, "results_df is non-empty")

    with open(args.existing_metrics) as f:
        existing = json.load(f)
    print(f"  Existing metrics: macro_auroc={existing.get('macro_auroc', 'N/A'):.4f}")

    # ------------------------------------------------------------------
    # Step 1-2: Per-cell parity (no CI)
    # ------------------------------------------------------------------
    print("\n[1] Per-cell metric parity (n_bootstrap=0) ...")
    cell_out = compute_metrics_with_ci(results_df, mode="per_cell", n_bootstrap=0)
    cell_m = cell_out["per_cell"]
    check(cell_m is not None, "per_cell metrics computed")

    tol = args.tol
    for key in ("macro_auroc", "macro_auprc", "macro_f1", "micro_f1",
                "exact_match_accuracy"):
        if key not in existing or key not in cell_m:
            continue
        old_val = float(existing[key])
        new_val = float(cell_m[key])
        diff = abs(old_val - new_val)
        check(diff < tol, f"{key}: old={old_val:.6f}  new={new_val:.6f}  diff={diff:.2e} < {tol}")

    # ------------------------------------------------------------------
    # Step 3: Per-protein parity (if reference provided)
    # ------------------------------------------------------------------
    if args.protein_metrics:
        print("\n[3] Per-protein metric parity ...")
        with open(args.protein_metrics) as f:
            prot_ref = json.load(f)
        prot_out = compute_metrics_with_ci(results_df, mode="per_protein", n_bootstrap=0)
        prot_m = prot_out["per_protein"]
        check(prot_m is not None, "per_protein metrics computed")
        for key in ("macro_auroc", "macro_f1", "micro_f1"):
            if key not in prot_ref or key not in prot_m:
                continue
            old_val = float(prot_ref[key])
            new_val = float(prot_m[key])
            diff = abs(old_val - new_val)
            check(diff < tol, f"[protein] {key}: ref={old_val:.6f}  new={new_val:.6f}  diff={diff:.2e}")
    else:
        print("\n[3] Skipped per-protein parity (--protein_metrics not provided)")

    # ------------------------------------------------------------------
    # Step 4: CI ordering
    # ------------------------------------------------------------------
    print(f"\n[4] CI ordering checks (n_bootstrap={args.n_bootstrap}) ...")
    ci_out = compute_metrics_with_ci(results_df, mode="per_cell", n_bootstrap=args.n_bootstrap)
    ci = ci_out["per_cell"].get("ci", {})
    for metric_name, ci_dict in ci.items():
        lo, mean, hi = ci_dict["ci_low"], ci_dict["mean"], ci_dict["ci_high"]
        check(lo < hi, f"{metric_name}: ci_low ({lo:.4f}) < ci_high ({hi:.4f})")
        check(lo <= mean <= hi,
              f"{metric_name}: ci_low ≤ mean ≤ ci_high ({lo:.4f} ≤ {mean:.4f} ≤ {hi:.4f})")

    # ------------------------------------------------------------------
    # Step 5: CI shrinks with subsample
    # ------------------------------------------------------------------
    print("\n[5] CI width shrinks with fewer proteins ...")
    if "ensembl_ids" in results_df.columns:
        proteins = (
            results_df["ensembl_ids"].str.strip().str.split(",").str[0].str.strip().unique()
        )
        n_half = max(10, len(proteins) // 2)
        rng = np.random.default_rng(99)
        half_proteins = rng.choice(proteins, size=n_half, replace=False)
        half_df = results_df[
            results_df["ensembl_ids"].str.strip().str.split(",").str[0].str.strip()
            .isin(half_proteins)
        ]
        full_out  = compute_metrics_with_ci(results_df, mode="per_cell", n_bootstrap=args.n_bootstrap)
        half_out  = compute_metrics_with_ci(half_df,    mode="per_cell", n_bootstrap=args.n_bootstrap)
        full_ci = full_out["per_cell"].get("ci", {}).get("macro_auroc", {})
        half_ci = half_out["per_cell"].get("ci", {}).get("macro_auroc", {})
        if full_ci and half_ci:
            full_width = full_ci["ci_high"] - full_ci["ci_low"]
            half_width = half_ci["ci_high"] - half_ci["ci_low"]
            check(half_width >= full_width * 0.9,
                  f"Half-sample CI width ({half_width:.4f}) ≥ 90% of full-sample ({full_width:.4f})")
    else:
        print("  Skipped (no ensembl_ids column)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
