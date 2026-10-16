"""Check multiple references, safe generation resume, and aligned publication on CPU."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_generation_benchmarks as evaluation
from prepare_generation_benchmarks import group, normalize
import torch


class GenerationTests(unittest.TestCase):
    def test_preserves_multiple_references_and_full_text(self):
        row = normalize({"instruction": "Task\ntext", "output": ["first", "second"]}, "row:0")
        self.assertEqual(row["references"], ["first", "second"])
        self.assertEqual(row["instruction"], "Task\ntext")
        self.assertEqual(row["input"], "")
        self.assertEqual(group("  TASK\n Text "), group(row["instruction"]))
        with self.assertRaises(AssertionError):
            normalize({"instruction": "Task", "output": []}, "invalid")

    def test_best_reference_is_scored_instead_of_first_reference(self):
        scorer = evaluation.legacy.rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = evaluation.score_references(scorer, "The correct answer!", ["An unrelated response", "the correct answer"])
        self.assertEqual(scores["rougeL"], 100)
        self.assertEqual(scores["exact_match"], 100)

    def test_resume_restores_rng_and_rejects_wrong_identity_or_prefix(self):
        with tempfile.TemporaryDirectory(dir=evaluation.ROOT / "tmp") as folder, \
                patch.object(torch.cuda, "get_rng_state_all", return_value=[]), \
                patch.object(torch.cuda, "set_rng_state_all"):
            path = Path(folder) / "state.pt"
            records = [{"id": "a", "response": "saved", "generated_tokens": 1}]
            torch.manual_seed(123)
            torch.rand(3)
            evaluation.save_progress(path, "fingerprint:benchmark", records)
            expected = torch.rand(8)
            torch.manual_seed(999)
            restored = evaluation.restore_progress(path, "fingerprint:benchmark", records + [{"id": "b"}])
            self.assertEqual(restored, records)
            self.assertTrue(torch.equal(torch.rand(8), expected))
            before = torch.get_rng_state()
            for fingerprint, rows in (("other", records), ("fingerprint:benchmark", [{"id": "wrong"}])):
                with self.assertRaises(AssertionError):
                    evaluation.restore_progress(path, fingerprint, rows)
                self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_legacy_cap_rate_is_preserved_without_inventing_item_flags(self):
        with tempfile.TemporaryDirectory(dir=evaluation.ROOT / "tmp") as folder, patch.object(evaluation, "OUT", Path(folder)):
            rows = [{"id": "a", "category": "task", "references": ["first", "correct answer"]}]
            generated = [{"id": "a", "response": "correct answer", "generated_tokens": 256, "length_capped": None}]
            evaluation.publish("gpt2_base", "dolly", rows, generated, {"fingerprint": "test"}, .056)
            result = json.loads((Path(folder) / "gpt2_base/dolly.json").read_text())
            self.assertEqual(result["metrics"]["length_capped"], .056)
            self.assertEqual(result["metrics"]["rougeL"], 100)
            with (evaluation.ROOT / result["per_item_csv"]).open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["length_capped"], "")
            with self.assertRaises(AssertionError):
                evaluation.publish("gpt2_base", "dolly", rows, [{**generated[0], "id": "wrong"}], {"fingerprint": "test"}, .056)

    def test_completed_result_rejects_different_prepared_ids(self):
        with tempfile.TemporaryDirectory(dir=evaluation.ROOT / "tmp") as folder, \
                patch.object(evaluation, "OUT", Path(folder) / "results"), \
                patch.object(evaluation, "LOCAL", Path(folder)), \
                patch.object(evaluation, "BENCHMARKS", {"dolly": "Dolly"}), \
                patch.object(evaluation, "identity", return_value={"fingerprint": "test"}):
            rows = [{"id": "a", "category": "task", "references": ["answer"]}]
            generated = [{"id": "a", "response": "answer", "generated_tokens": 1, "length_capped": False}]
            evaluation.publish("gpt2_base", "dolly", rows, generated, {"fingerprint": "test"})
            prepared = Path(folder) / "prepared/dolly.jsonl"
            prepared.parent.mkdir()
            prepared.write_text('{"id":"a"}\n', encoding="utf-8")
            data = {"counts": {"dolly": {"included": 1}}}
            self.assertIn("dolly", evaluation.load_completed("gpt2_base", data))
            prepared.write_text('{"id":"different"}\n', encoding="utf-8")
            with self.assertRaises(AssertionError):
                evaluation.load_completed("gpt2_base", data)


if __name__ == "__main__":
    unittest.main()
