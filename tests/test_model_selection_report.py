"""Check paired reporting and preserve unavailable metric scores."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_model_selection import paired, values


class ModelSelectionReportTests(unittest.TestCase):
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
