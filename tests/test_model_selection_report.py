"""Check paired reporting and preserve unavailable metric scores."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_model_selection import component_means, external_grid, paired, rouge_components, values


class ModelSelectionReportTests(unittest.TestCase):
    def test_rouge_components_use_one_best_f1_reference_and_preserve_primary(self):
        rows = [{"id": "x", "references": ["a", "a b c e"]}]
        generated = [{"id": "x", "response": "a b c d"}]
        primary = [{"id": "x", "rougeL": 75.}]
        actual = rouge_components(rows, generated, primary)[0]
        self.assertEqual(actual["reference_index_0based"], 1)
        self.assertEqual([actual[key] for key in ("rougeL_precision", "rougeL_recall", "rougeL_f1")], [75., 75., 75.])
        with self.assertRaisesRegex(AssertionError, "F1 differs"):
            rouge_components(rows, generated, [{"id": "x", "rougeL": 74.}])
        with self.assertRaisesRegex(AssertionError, "IDs"):
            rouge_components(rows, [{"id": "wrong", "response": "a"}], primary)
        empty = rouge_components(rows, [{"id": "x", "response": ""}], [{"id": "x", "rougeL": 0.}])[0]
        self.assertEqual(empty["reference_index_0based"], 0)
        self.assertEqual(empty["rougeL_f1"], 0.)

    def test_macro_f1_is_not_harmonic_mean_of_average_precision_recall(self):
        rows = [{"rougeL_precision": 100., "rougeL_recall": 50., "rougeL_f1": 200 / 3},
                {"rougeL_precision": 50., "rougeL_recall": 100., "rougeL_f1": 200 / 3}]
        actual = component_means(rows)
        self.assertEqual(actual["rougeL_precision"], 75.)
        self.assertEqual(actual["rougeL_recall"], 75.)
        self.assertAlmostEqual(actual["rougeL_f1"], 200 / 3)
        self.assertNotEqual(actual["rougeL_f1"], 75.)

    def test_external_grid_preserves_real_zero_and_unscored_cells(self):
        runs = {("sft", "gpt2_base", "self_inst"): {"primary": {"metrics": {"rougeL": 0.}}, "supplementary": None}}
        ordered = [("original", "gpt2_base"), ("sft", "gpt2_base")]
        primary = external_grid(runs, ordered, "rougeL")
        self.assertEqual(primary.shape, (2, 3))
        self.assertEqual(primary[1, 0], 0.)
        self.assertTrue(np.isnan(primary[0]).all())
        self.assertTrue(np.isnan(primary[1, 1:]).all())
        self.assertTrue(np.isnan(external_grid(runs, ordered, "meteor")).all())

    def test_paired_differences_and_missing_metric(self):
        left = [{"id": "a", "rougeL": 2}, {"id": "b", "rougeL": 4}, {"id": "c", "rougeL": 1}]
        right = [{"id": "a", "rougeL": 4}, {"id": "b", "rougeL": 1}, {"id": "c", "rougeL": 1}]
        self.assertEqual(paired(left, right, "rougeL"), {"n": 3, "difference": -1 / 3, "higher": 1, "tie": 1, "lower": 1})
        self.assertIsNone(paired(left, right, "meteor"))
        with self.assertRaisesRegex(AssertionError, "IDs"):
            paired(left, list(reversed(right)), "rougeL")
        actual = values({"primary": {"metrics": {"rougeL": 3}}, "supplementary": None})
        self.assertEqual(actual, {"rougeL": 3})
        self.assertNotIn("meteor", actual)


if __name__ == "__main__":
    unittest.main()
