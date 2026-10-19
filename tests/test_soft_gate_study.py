"""Verify the isolated gate intervention and official truncation semantics on CPU."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from filelock import FileLock, Timeout
import torch
import soft_gate_study as study


class GateStudyTests(unittest.TestCase):
    def test_equal_forward_and_exact_selection_gradient_decomposition(self):
        self.assertEqual(study.gradient_audit()["status"], "passed")

    def test_masking_and_equal_distributions(self):
        teacher = torch.tensor([[[2., 0., -2.], [0., 2., -2.], [0., -2., 2.]]], requires_grad=True)
        student = teacher.detach().flip(-1).clone().requires_grad_()
        loss, _ = study.soft_loss(teacher, student, torch.tensor([[-100, -100, 0]]))
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.all(student.grad[:, [0, 2]] == 0))
        self.assertGreater(float(student.grad[:, 1].norm()), 0)
        same = teacher.detach().clone().requires_grad_()
        equal, _ = study.soft_loss(teacher, same, torch.tensor([[-100, -100, 0]]))
        self.assertAlmostEqual(float(equal.detach()), 0, places=7)

    def test_concat_tokenization_and_response_truncation(self):
        class Tokenizer:
            eos_token_id = 999
            def encode(self, text, **kwargs):
                return list(range(len(text)))
        with patch.object(study.base, "render_prompt", return_value="p" * 200):
            row = study.encode(Tokenizer(), {"instruction":"i", "input":"", "output":"r" * 400})
        self.assertEqual(len(row["input_ids"]), 512)
        self.assertTrue(row["truncated"])
        self.assertEqual(row["labels"][:200], [-100] * 200)
        self.assertNotIn(999, row["input_ids"])
        with patch.object(study.base, "render_prompt", return_value="p" * 257):
            self.assertIsNone(study.encode(Tokenizer(), {"instruction":"i", "input":"", "output":"r"}))

    def test_occupied_shared_lock_prevents_gpu_model_loading(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as folder:
            root = Path(folder)
            study.base.write_json(root / "preparation.json", {"producer_sha256": study.base.sha(study.__file__), "data": {}})
            with patch.object(study, "OUT", root), patch.object(study, "LOCAL", root / "new"), \
                 patch.object(study.base, "LOCAL", root), patch.object(study.AutoModelForCausalLM, "from_pretrained") as loader:
                with FileLock(str(root / "worker.lock")):
                    with self.assertRaises(Timeout):
                        study.baseline(False)
                loader.assert_not_called()

    def test_generation_resume_restores_rng_and_keeps_completed_batches(self):
        class Tokenizer:
            eos_token_id = 999
            def decode(self, tokens, **kwargs):
                return " ".join(map(str, tokens))
        class Model:
            def __init__(self, interrupt=False):
                self.calls, self.interrupt = 0, interrupt
            def generate(self, input_ids, **kwargs):
                self.calls += 1
                if self.interrupt and self.calls == 2:
                    raise RuntimeError("interrupted")
                return torch.cat([input_ids, torch.randint(1, 100, (len(input_ids), 3))], dim=1)
        tensor, full = torch.tensor, torch.full
        def on_cpu(function, *args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return function(*args, **kwargs)
        records = [{"id": str(i), "prompt_ids": [i + 1]} for i in range(19)]
        spec = {"fingerprint": "fixture", "generation": {"batch_size": 8}}
        with tempfile.TemporaryDirectory(dir=study.ROOT) as folder, \
             patch.object(study, "status"), \
             patch.object(study.torch, "tensor", side_effect=lambda *a, **k: on_cpu(tensor, *a, **k)), \
             patch.object(study.torch, "full", side_effect=lambda *a, **k: on_cpu(full, *a, **k)), \
             patch.object(study.torch.cuda, "get_rng_state", side_effect=torch.get_rng_state), \
             patch.object(study.torch.cuda, "set_rng_state"):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                study.generate(Model(True), Tokenizer(), records, Path(folder) / "resumed", spec, 10)
            resumed = study.generate(Model(), Tokenizer(), records, Path(folder) / "resumed", spec, 10)
            complete = study.generate(Model(), Tokenizer(), records, Path(folder) / "complete", spec, 10)
        self.assertEqual(resumed["responses"], complete["responses"])


if __name__ == "__main__":
    unittest.main()
