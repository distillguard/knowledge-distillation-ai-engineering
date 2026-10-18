"""Hand-computable metric cases; no WordNet download, evaluator weights or GPU."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import supplementary_metrics as metrics


class SupplementaryMetricTests(unittest.TestCase):
    def test_meteor_uses_best_reference_and_default_fragmentation_penalty(self):
        with patch.object(metrics, "wordnet", Mock(synsets=Mock(return_value=[]))):
            score = metrics.meteor("The correct answer", ["other words", "the correct answer"])
            self.assertAlmostEqual(score, 1 - .5 * (1 / 3) ** 3)
            self.assertEqual(metrics.meteor("", ["the correct answer"]), 0)
            self.assertEqual(metrics.meteor("unrelated", ["the correct answer"]), 0)
        with self.assertRaises(AssertionError):
            metrics.meteor("answer", [])

    def test_distinct4_counts_corpus_duplicates_and_preserves_response_boundaries(self):
        self.assertEqual(metrics.distinct4(["one two three four five"] * 2),
                         {"unique_4grams": 2, "total_4grams": 4, "distinct4": .5})
        self.assertEqual(metrics.distinct4(["one two three", "four five six"]),
                         {"unique_4grams": 0, "total_4grams": 0, "distinct4": None})
        self.assertEqual(metrics.distinct4(["One two three four", "one two three four"])["distinct4"], 1)

    def test_ece_retains_bins_and_calculates_confidence_accuracy_gap(self):
        result = metrics.calibration([0, 0, 1, 1], [0, 1, 0, 1], [.2, .8, .2, .8])
        self.assertAlmostEqual(result["ece"], .3)
        self.assertEqual(len(result["bins"]), 15)
        occupied = [row for row in result["bins"] if row["n"]]
        self.assertEqual(len(occupied), 1)
        self.assertEqual(occupied[0]["upper_inclusive"], .8)
        self.assertEqual(occupied[0]["n"], 4)
        self.assertEqual(occupied[0]["accuracy"], .5)
        self.assertIsNone(result["bins"][0]["mean_confidence"])

    def test_ece_handles_extreme_and_tied_probabilities_and_rejects_invalid_input(self):
        self.assertEqual(metrics.calibration([0, 1], [0, 1], [0, 1])["ece"], 0)
        tied = metrics.calibration([0, 1], [0, 0], [.5, .5])
        self.assertEqual(tied["ece"], 0)
        self.assertEqual(tied["bins"][7]["n"], 2)
        for labels, predictions, probabilities in (([], [], []), ([0], [0], [float("nan")]),
                                                    ([0], [0], [1.1]), ([0], [2], [.5]),
                                                    ([0, 1], [0], [.5])):
            with self.subTest(probabilities=probabilities), self.assertRaises(AssertionError):
                metrics.calibration(labels, predictions, probabilities)


if __name__ == "__main__":
    unittest.main()
