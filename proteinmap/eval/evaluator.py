"""
Evaluator — orchestrates model inference + metric computation for one split.

Usage:
    evaluator = Evaluator(predictor=PUPSPredictor(ckpt), output_dir="results/hpa_proteins/")
    evaluator.evaluate(dataloader, split_name="hpa_proteins")

Outputs (in output_dir):
    {split_name}_results.pkl   — pandas DataFrame with per-sample predictions
    {split_name}_metrics.json  — scalar metrics dict
"""

import json
import pathlib
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from proteinmap.models.base import BasePredictor
from proteinmap.metrics.compartment_metrics import compute_compartment_metrics
from proteinmap.metrics.inpainting_metrics import compute_inpainting_metrics

from proteinmap.data.constants import CLASSES as _PUPS_CLASSES_NESTED

PUPS_CLASS_NAMES = _PUPS_CLASSES_NESTED[0]


class Evaluator:
    """
    Drives a BasePredictor over a DataLoader and computes all benchmark metrics.

    Args:
        predictor:   any BasePredictor subclass (PUPSPredictor, ProtGPSPredictor)
        output_dir:  directory where results and metrics are saved
        limit_batches: cap number of batches (smoke-test mode)
    """

    def __init__(
        self,
        predictor: BasePredictor,
        output_dir: str,
        limit_batches: Optional[int] = None,
    ):
        self.predictor = predictor
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit_batches = limit_batches

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self, dataloader: DataLoader, split_name: str) -> dict:
        """
        Run inference + compute metrics for `split_name`.

        Returns the metrics dict and saves results/metrics to output_dir.
        """
        print(f"\n[Evaluator] {self.predictor.name} on {split_name}")

        # ------------------------------------------------------------------
        # Pass 1 — collect per-sample predictions
        # ------------------------------------------------------------------
        rows = self.predictor.run_on_dataloader(
            dataloader, limit_batches=self.limit_batches
        )
        if not rows:
            raise RuntimeError("No samples returned from predictor — check DataLoader.")

        results_df = pd.DataFrame(rows)
        print(f"  Collected {len(results_df)} samples.")

        # ------------------------------------------------------------------
        # Compartment metrics
        # ------------------------------------------------------------------
        y_true = np.stack(results_df["y_true_multilabel"].values)   # (N, 29)
        y_proba = np.stack(results_df["y_pred_proba"].values)        # (N, 29)
        y_pred = np.stack(results_df["y_pred_binary"].values)        # (N, 29)

        comp_metrics = compute_compartment_metrics(
            y_true, y_proba, y_pred, PUPS_CLASS_NAMES
        )

        metrics = {
            "model": self.predictor.name,
            "split": split_name,
            "n_samples": len(results_df),
            "macro_auroc": comp_metrics["macro_auroc"],
            "macro_auprc": comp_metrics["macro_auprc"],
            "macro_f1": comp_metrics["macro_f1"],
            "micro_f1": comp_metrics["micro_f1"],
            "sample_f1": comp_metrics["sample_f1"],
            "macro_precision": comp_metrics["macro_precision"],
            "macro_recall": comp_metrics["macro_recall"],
            "exact_match_accuracy": comp_metrics["exact_match_accuracy"],
        }

        # ------------------------------------------------------------------
        # Inpainting metrics (PUPS only)
        # ------------------------------------------------------------------
        if self.predictor.supports_inpainting:
            valid = results_df["y_antibody_pred"].notna()
            if valid.any():
                y_ab_true = np.stack(results_df.loc[valid, "y_antibody_true"].values)
                y_ab_pred = np.stack(results_df.loc[valid, "y_antibody_pred"].values)
                inp_metrics = compute_inpainting_metrics(y_ab_true, y_ab_pred)
                metrics["inpainting_mse"] = inp_metrics["mse"]
                metrics["inpainting_ssim"] = inp_metrics["ssim"]
                metrics["inpainting_pearson_r"] = inp_metrics["pearson_r"]
                # Also store mean MSE from per-row column if available
                if "inpainting_mse" in results_df.columns:
                    metrics["inpainting_mse_rowwise"] = float(
                        results_df["inpainting_mse"].mean()
                    )

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        results_pkl = self.output_dir / f"{split_name}_results.pkl"
        metrics_json = self.output_dir / f"{split_name}_metrics.json"
        per_class_csv = self.output_dir / f"{split_name}_per_class.csv"

        with open(results_pkl, "wb") as f:
            pickle.dump(results_df, f)

        # Save per-class DataFrame separately
        comp_metrics["per_class_df"].to_csv(per_class_csv, index=False)

        # JSON-serialisable metrics (exclude DataFrames/arrays)
        json_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        with open(metrics_json, "w") as f:
            json.dump(json_metrics, f, indent=2)

        self._print_summary(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print_summary(self, metrics: dict):
        print(f"\n  {'='*50}")
        print(f"  Model: {metrics['model']}  Split: {metrics['split']}")
        print(f"  Samples:     {metrics['n_samples']}")
        print(f"  macro_AUROC: {metrics['macro_auroc']:.4f}")
        print(f"  macro_AUPRC: {metrics['macro_auprc']:.4f}")
        print(f"  macro_F1:    {metrics['macro_f1']:.4f}")
        print(f"  micro_F1:    {metrics['micro_f1']:.4f}")
        print(f"  ExactMatch:  {metrics['exact_match_accuracy']:.4f}")
        if "inpainting_mse" in metrics:
            print(f"  Inp. MSE:    {metrics['inpainting_mse']:.6f}")
            print(f"  Inp. SSIM:   {metrics.get('inpainting_ssim', float('nan')):.4f}")
            print(f"  Inp. Pearson:{metrics.get('inpainting_pearson_r', float('nan')):.4f}")
        print(f"  {'='*50}\n")
