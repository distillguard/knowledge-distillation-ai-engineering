"""Verify label/row identity and independently known calibration statistics."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_candidate_diagnostics as report


class DiagnosticTests(unittest.TestCase):
    def test_recompute_conditional_metrics_and_reject_wrong_source(self):
        records = [{"source_row": i, "label": y} for i, y in enumerate([0, 1, 1, 0])]
        prediction = {"source_rows": [0, 1, 2, 3], "labels": [0, 1, 1, 0], "predictions": [0, 0, 1, 1],
                      "positive_conditional_probability": [.1, .2, .8, .9]}
        expected = report.legacy.classification_metrics(prediction["labels"], prediction["predictions"], prediction["positive_conditional_probability"])
        actual = report.validate_predictions(prediction, records, expected)
        self.assertEqual(actual["accuracy"], .5)
        self.assertEqual(actual["macro_f1"], .5)
        self.assertAlmostEqual(actual["positive_class_conditional_label_brier"], .375)
        self.assertAlmostEqual(actual["calibration"]["ece"], .35)
        self.assertEqual(sum(b["n"] for b in actual["calibration"]["bins"]), 4)
        self.assertEqual(actual["positive_prediction_rate"], .5)
        for changed in ({**prediction, "source_rows": [3, 2, 1, 0]}, {**prediction, "labels": [1, 1, 1, 0]},
                        {**prediction, "positive_conditional_probability": [float("nan")] * 4}):
            with self.assertRaises(AssertionError):
                report.validate_predictions(changed, records, expected)
        with self.assertRaisesRegex(AssertionError, "disagree"):
            report.validate_predictions(prediction, records, {**expected, "accuracy": 1.})


if __name__ == "__main__":
    unittest.main()
