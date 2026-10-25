"""
UnifiedEvaluator — extends Evaluator with per-protein mode and bootstrap CIs.

Does not replace the existing Evaluator; both coexist.

Output files (in output_dir):
    {split_name}_results.pkl          — per-cell DataFrame (same schema as Evaluator)
    {split_name}_cell_metrics.json    — per-cell metrics + CI
    {split_name}_protein_metrics.json — per-protein metrics + CI (if enabled)
    {split_name}_per_class.csv        — per-class breakdown (from per-cell pass)
"""

import json
import pathlib
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from proteinmap.models.base import BasePredictor
from proteinmap.metrics.aggregation import compute_metrics_with_ci


class UnifiedEvaluator:
    """
    Drives a BasePredictor over a DataLoader and computes per-cell and/or
    per-protein metrics with bootstrap CIs.

    Args:
        predictor:               any BasePredictor subclass
        output_dir:              directory where results/metrics are saved
        compute_protein_metrics: if True, also run per-protein aggregation + metrics
        n_bootstrap:             bootstrap samples for CI (0 → skip CI)
        limit_batches:           cap number of inference batches (smoke-test mode)
    """

    def __init__(
        self,
        predictor: BasePredictor,
        output_dir: str,
        compute_protein_metrics: bool = True,
        n_bootstrap: int = 1000,
        limit_batches: Optional[int] = None,
    ):
        self.predictor               = predictor
        self.output_dir              = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compute_protein_metrics = compute_protein_metrics
        self.n_bootstrap             = n_bootstrap
        self.limit_batches           = limit_batches

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self, dataloader: DataLoader, split_name: str) -> dict:
        """
        Run inference + compute metrics for `split_name`.

        Returns the full metrics dict:
        {
          "per_cell":    {...metrics..., "ci": {...}},
          "per_protein": {...metrics..., "ci": {...}} or None
        }
        """
        print(f"\n[UnifiedEvaluator] {self.predictor.name} on {split_name}")

        rows = self.predictor.run_on_dataloader(
            dataloader, limit_batches=self.limit_batches
        )
        if not rows:
            raise RuntimeError("No samples returned from predictor — check DataLoader.")

        results_df = pd.DataFrame(rows)
        print(f"  Collected {len(results_df)} samples.")

        # Save raw per-cell results (same as Evaluator)
        results_pkl = self.output_dir / f"{split_name}_results.pkl"
        with open(results_pkl, "wb") as f:
            pickle.dump(results_df, f)

        mode = "both" if self.compute_protein_metrics else "per_cell"
        metrics_all = compute_metrics_with_ci(
            results_df,
            mode=mode,
            n_bootstrap=self.n_bootstrap,
        )

        # Save cell-level metrics
        if metrics_all["per_cell"] is not None:
            self._save_metrics(metrics_all["per_cell"], split_name, "cell")

        # Save protein-level metrics
        if metrics_all["per_protein"] is not None:
            self._save_metrics(metrics_all["per_protein"], split_name, "protein")

        self._print_summary(
            self.predictor.name, split_name,
            metrics_all["per_cell"] or metrics_all["per_protein"]
        )
        return metrics_all

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_metrics(self, metrics: dict, split_name: str, suffix: str):
        # Extract and save per-class CSV
        per_class_df = metrics.pop("per_class_df", None)
        if per_class_df is not None:
            per_class_df.to_csv(
                self.output_dir / f"{split_name}_per_class.csv", index=False
            )

        # JSON-serialisable metrics
        def _to_json(v):
            if isinstance(v, (bool, str)):
                return v
            if isinstance(v, (int, float, np.integer, np.floating)):
                f = float(v)
                return None if (f != f) else f   # drop NaN
            if isinstance(v, dict):
                result = {kk: _to_json(vv) for kk, vv in v.items()}
                return {kk: vv for kk, vv in result.items() if vv is not None}
            return None

        json_metrics = {k: _to_json(v) for k, v in metrics.items() if _to_json(v) is not None}

        out_path = self.output_dir / f"{split_name}_{suffix}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"  Saved {out_path.name}")

    def _print_summary(self, model_name: str, split_name: str, metrics: Optional[dict]):
        if metrics is None:
            return
        print(f"\n  {'='*50}")
        print(f"  Model: {model_name}  Split: {split_name}")
        print(f"  macro_AUROC: {metrics.get('macro_auroc', float('nan')):.4f}")
        print(f"  macro_AUPRC: {metrics.get('macro_auprc', float('nan')):.4f}")
        print(f"  macro_F1:    {metrics.get('macro_f1', float('nan')):.4f}")
        print(f"  micro_F1:    {metrics.get('micro_f1', float('nan')):.4f}")
        ci = metrics.get("ci", {})
        if ci.get("macro_auroc"):
            c = ci["macro_auroc"]
            print(f"  AUROC 95% CI: [{c['ci_low']:.4f}, {c['ci_high']:.4f}]")
        if "inpainting_mse" in metrics:
            print(f"  Inp. MSE:     {metrics['inpainting_mse']:.6f}")
            print(f"  Inp. SSIM:    {metrics.get('inpainting_ssim', float('nan')):.4f}")
            print(f"  Inp. Pearson: {metrics.get('inpainting_pearson_r', float('nan')):.4f}")
        print(f"  {'='*50}\n")
