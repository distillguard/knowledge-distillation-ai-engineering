"""Check metric publication, native length accounting and phase isolation without BERT weights."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_supplementary_metrics as scoring
import run_supplementary_metrics as runner
from filelock import FileLock, Timeout


class Tokenizer:
    model_max_length = 512

    def encode(self, text, **kwargs):
        return [0, *range(len(text.split())), 2]

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]


class SupplementaryScoringTests(unittest.TestCase):
    def test_input_chunks_bound_embedding_cache_without_losing_final_rows(self):
        rows = [{"id": str(index), "references": ["one two three"]} for index in range(65)]
        generated = [{"id": row["id"], "response": "one two three"} for row in rows]
        batches = []

        def score(candidates, references, batch_size):
            batches.append((len(candidates), len(references), batch_size))
            return tuple(SimpleNamespace(tolist=lambda n=len(candidates): [1.] * n) for _ in range(3))

        scorer = SimpleNamespace(_tokenizer=Tokenizer(), score=score)
        with patch.object(scoring.metrics, "wordnet", Mock(synsets=Mock(return_value=[]))):
            scored = scoring.score_outputs(scorer, rows, generated)
        self.assertEqual([r["id"] for r in scored], [r["id"] for r in rows])
        self.assertEqual(batches, [(64, 64, 64), (1, 1, 64)])

    def test_runner_isolates_legacy_dependencies_and_preserves_parent_environment(self):
        with tempfile.TemporaryDirectory(dir=scoring.ROOT / "tmp") as folder, \
                patch.object(runner, "LOCAL", Path(folder)), \
                patch.dict(os.environ, {"PYTHONPATH": "caller-environment"}):
            (Path(folder) / "deps").mkdir()
            for stage in ("prepare", "score"):
                forwarded = ["--model", "gpt2_base", "--phase", "sft"] if stage == "score" else []
                with self.subTest(stage=stage), patch.object(runner.subprocess, "run") as run:
                    runner.main([stage, *forwarded])
                    call = run.call_args
                    self.assertEqual(call.args[0][6:], forwarded)
                    self.assertEqual(call.kwargs["env"]["PYTHONPATH"], str(Path(folder) / "deps"))
                    self.assertEqual(call.kwargs["env"]["USE_TF"], "0")
                    self.assertTrue(call.kwargs["check"])
                    self.assertEqual(os.environ["PYTHONPATH"], "caller-environment")

    def test_scores_preserve_full_meteor_references_and_audit_bert_truncation(self):
        rows = [{"id": "a", "references": ["other words", "the correct answer"]},
                {"id": "b", "references": ["word " * 600]}]
        generated = [{"id": "a", "response": "the correct answer"}, {"id": "b", "response": ""}]
        values = ([.9, 0], [.8, 0], [.85, 0])
        scorer = SimpleNamespace(_tokenizer=Tokenizer(), score=Mock(return_value=tuple(SimpleNamespace(tolist=lambda v=v: v) for v in values)))
        with patch.object(scoring.metrics, "wordnet", Mock(synsets=Mock(return_value=[]))):
            scored = scoring.score_outputs(scorer, rows, generated)
        self.assertAlmostEqual(scored[0]["meteor"], 100 * (1 - .5 / 27))
        self.assertEqual(scored[1]["meteor"], 0)
        self.assertEqual(scored[0]["bertscore_f1"], 85)
        self.assertEqual(json.loads(scored[1]["bert_reference_full_tokens"]), [602])
        self.assertEqual(json.loads(scored[1]["bert_reference_retained_tokens"]), [512])
        self.assertEqual(scored[1]["bert_reference_truncated_count"], 1)
        self.assertEqual(scored[1]["bert_response_full_tokens"], 2)
        self.assertEqual(scored[1]["bert_response_truncated"], 0)
        self.assertEqual(len(scorer.score.call_args.args[1][1][0].split()), 600)

        with tempfile.TemporaryDirectory(dir=scoring.ROOT / "tmp") as folder:
            folder = Path(folder)
            spec = {"fingerprint": "test", "model": "gemma2_9b_it", "phase": "sft", "benchmark": "dolly"}
            primary = {"metrics": {"rougeL": 10}}
            primary_items = [{"id": row["id"], "category": "task", "rougeL": 10} for row in rows]
            result = scoring.publish(folder, spec, primary, primary_items, scored, generated)
            self.assertEqual(result["metrics"]["bertscore_f1"], 42.5)
            self.assertIsNone(result["diversity"]["distinct4"])
            self.assertEqual(result["bert_length_audit"]["reference_count"], 3)
            self.assertEqual(result["bert_length_audit"]["truncated_reference_count"], 1)
            self.assertIsNone(result["sample_sd_across_generation_seeds"])
            self.assertEqual(scoring.load_completed(folder, spec, rows, generated), result)
            for bad_spec, bad_rows in (({**spec, "phase": "original"}, rows),
                                        ({**spec, "fingerprint": "different"}, rows),
                                        (spec, [{**rows[0], "id": "changed"}, rows[1]])):
                with self.assertRaises(AssertionError):
                    scoring.load_completed(folder, bad_spec, bad_rows, generated)
            (folder / "per_item.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(AssertionError):
                scoring.load_completed(folder, spec, rows, generated)

    def test_shared_lock_blocks_scoring_before_model_or_source_loading(self):
        with tempfile.TemporaryDirectory(dir=scoring.ROOT / "tmp") as folder:
            root = Path(folder)
            (root / "results/generation_benchmarks_v1").mkdir(parents=True)
            (root / "tmp/dolly_sft_v1").mkdir(parents=True)
            (root / "results/generation_benchmarks_v1/completion.json").write_text(json.dumps({"study_models": ["gpt2_base"]}), encoding="utf-8")
            with patch.object(scoring, "ROOT", root), patch.object(scoring, "source") as source, \
                    FileLock(str(root / "tmp/dolly_sft_v1/worker.lock")):
                with self.assertRaises(Timeout):
                    scoring.main(["--model", "gpt2_base"])
                source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
