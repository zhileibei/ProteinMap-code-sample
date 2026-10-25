"""Small synthetic checks for the public evaluation code sample."""

import unittest

import numpy as np
import pandas as pd

from proteinmap.metrics.aggregation import (
    _macro_auroc,
    aggregate_per_protein,
    bootstrap_ci,
)
from proteinmap.metrics.compartment_metrics import compute_compartment_metrics


class MetricsTests(unittest.TestCase):
    def test_aggregation_counts_each_protein_once(self):
        rows = pd.DataFrame(
            {
                "ensembl_ids": ["protein_a", "protein_a", "protein_b"],
                "y_pred_proba": [
                    np.array([0.2, 0.8]),
                    np.array([0.6, 0.4]),
                    np.array([0.9, 0.1]),
                ],
                "y_true_multilabel": [
                    np.array([1, 0]),
                    np.array([0, 1]),
                    np.array([1, 0]),
                ],
            }
        )
        result = aggregate_per_protein(rows).set_index("ensembl_ids")
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result.loc["protein_a", "y_pred_proba"], [0.4, 0.6])
        np.testing.assert_array_equal(result.loc["protein_a", "y_true_multilabel"], [1, 1])

    def test_auroc_skips_class_without_negative_examples(self):
        labels = np.array([[1, 0], [1, 0], [1, 1], [1, 1]])
        probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.8], [0.6, 0.9]])
        predictions = (probabilities >= 0.5).astype(int)
        result = compute_compartment_metrics(labels, probabilities, predictions, ["a", "b"])
        self.assertTrue(np.isnan(result["per_class_df"].loc[0, "auroc"]))
        self.assertAlmostEqual(result["macro_auroc"], 1.0)

    def test_cluster_bootstrap_uses_proteins_as_sampling_units(self):
        labels = np.array([[0], [0], [1], [1]])
        probabilities = np.array([[0.1], [0.2], [0.8], [0.9]])
        protein_ids = np.array(["a", "a", "b", "b"])
        result = bootstrap_ci(
            labels, probabilities, _macro_auroc,
            n_bootstrap=60, protein_ids=protein_ids,
        )
        self.assertGreater(result["n_bootstrap"], 0)
        self.assertLess(result["n_bootstrap"], 60)
        self.assertAlmostEqual(result["ci_low"], 1.0)
        self.assertAlmostEqual(result["ci_high"], 1.0)


if __name__ == "__main__":
    unittest.main()
