"""Check paired reporting and preserve unavailable metric scores."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_model_selection import external_grid, paired, values


class ModelSelectionReportTests(unittest.TestCase):
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
