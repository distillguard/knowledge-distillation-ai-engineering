import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kd_ecqa_v2 as v2


class FixedBudgetTests(unittest.TestCase):
    def test_only_training_seed_expressions_change_and_globals_are_isolated(self):
        source = v2.trainer_source()  # Also asserts whole-function AST equivalence after seed normalization.
        tree = ast.parse(source)
        args = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "TrainingArguments")
        kwargs = {k.arg: k.value for k in args.keywords}
        self.assertEqual(kwargs["num_train_epochs"].value, 3)
        self.assertEqual(kwargs["warmup_steps"].value, 71)
        for seed in v2.SEEDS:
            engine = v2.engine_for(seed)
            self.assertEqual(engine.EXPERIMENT_SEED, seed)
            self.assertEqual(engine.train_stage.__globals__["EXPERIMENT_SEED"], seed)
            self.assertEqual(engine.LOCAL, v2.LOCAL / f"seed{seed}")
            self.assertNotEqual(engine.OUT, v2.kd.OUT)
            self.assertEqual(engine.report, v2.report)
        self.assertEqual(v2.kd.LOCAL.name, "ecqa_kd_v1")

    def test_fixed_epoch_ignores_better_dev_and_test(self):
        epochs = [{"epoch": ep, "global_step": ep*473, "dev": {"n": 1079, "accuracy": 1/ep}, "test_accuracy": 1/ep} for ep in (1, 2, 3)]
        self.assertEqual(v2.fixed_selection(epochs, "test")["selected"]["epoch"], 3)
        with self.assertRaises(AssertionError):
            v2.fixed_selection(epochs[:2], "test")

    def test_reuse_matrix_cannot_reuse_changed_teacher_or_wrong_seed(self):
        inv = v2.base.read(v2.INVENTORY)
        self.assertEqual(sum(r["reuse_training"] for r in inv["rows"]), 5)
        self.assertEqual(sum(r["reuse_test"] for r in inv["rows"]), 3)
        self.assertFalse(any(r["reuse_training"] for r in inv["rows"] if r["seed"] != 10))
        via = next(r for r in inv["rows"] if r["seed"] == 10 and r["stage"] == "via_fkl")
        self.assertFalse(via["reuse_training"])
        with self.assertRaises(AssertionError):
            v2.reuse_training(v2.engine_for(10), via, {})

    def test_protocol_records_seed_fixed_endpoint_and_preserves_numerics(self):
        teacher = v2.kd.selected("qwen35_4b")[0]
        before = deepcopy(v2.base.read(v2.kd.OUT / "ta_fkl/training_protocol.json"))
        for seed in v2.SEEDS:
            spec = v2.protocol(seed, "ta_fkl", teacher, {"fingerprint": "test"})
            self.assertEqual(spec["recipe"]["seed"], seed)
            self.assertEqual(spec["data_seed"], seed)
            self.assertEqual(spec["teacher_initialization"], teacher)
            for key in ("loss", "learning_rate", "epochs", "maximum_updates", "warmup_updates", "temperature", "token_chunk", "student_logits_handling", "safety"):
                self.assertEqual(spec[key], before[key])
        self.assertEqual(before, v2.base.read(v2.kd.OUT / "ta_fkl/training_protocol.json"))

    def test_report_keeps_old_text_and_does_not_publish_unverified_test(self):
        folder = v2.base.ROOT / "tmp"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            root = Path(temp)
            original = "# 기존 결과\r\n\r\n보존할 결과입니다.\r\n".encode()
            for name in v2.MDS:
                (root / name).write_bytes(original)
            with patch.object(v2.base, "ROOT", root), patch.object(v2, "LOCAL", root / "local"), patch.object(v2, "OUT", root / "out"):
                path = v2.OUT / "seed10/direct_fkl/sft/qwen35_08b.json"
                v2.base.write(path, {"status": "completed", "overall": {"n": 2194}})
                v2.report()
                v2.report()
                for name in v2.MDS:
                    text = (root / name).read_bytes()
                    self.assertTrue(text.startswith(original))
                    self.assertEqual(text.count(v2.BEGIN.encode()), 1)
                    self.assertIn("전체 시험 0/18".encode(), text)

    def test_raw_parser_rejects_wrong_answer_or_id(self):
        folder = v2.base.ROOT / "tmp"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            root = Path(temp)
            item = {"id": "a", "gold": "A", "fingerprint": "test", "output_tokens": 12,
                "response": "ANSWER: B\nEXPLANATION: reason", "predicted": "A", "explanation_present": True, "format_valid": True}
            v2.base.write(root / "a.json", item)
            with self.assertRaises(AssertionError):
                v2.raw_records(root, [{"id": "a", "gold": "A"}], "test")


if __name__ == "__main__":
    unittest.main()
