from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kd_ecqa


class KlTests(unittest.TestCase):
    def test_omitting_fp32_output_copy_keeps_bf16_logits_loss_and_gradient(self):
        import torch
        accelerator = SimpleNamespace(num_processes=1, mixed_precision="bf16", autocast=nullcontext)
        inputs = {"input_ids": torch.zeros((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5)), "labels": torch.tensor([[-100, -100, -100, 3, 4]])}
        class Student:
            def __init__(self, logits):
                self.logits = logits
            def _original_forward(self, **kwargs):
                return SimpleNamespace(logits=self.logits[:, -kwargs["logits_to_keep"]:])
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=self._original_forward(**kwargs).logits.float())
        for direction in ("fkl", "rkl"):
            s = torch.randn((1, 5, 7), dtype=torch.bfloat16, requires_grad=True)
            t = torch.randn((1, 5, 7), dtype=torch.bfloat16, requires_grad=True)
            student, teacher = Student(s), Student(t)
            expected, _ = kd_ecqa.response_kl(student, teacher, inputs, direction)
            gradient = torch.autograd.grad(expected, s)[0]
            actual, output = kd_ecqa.training_response_kl(student, teacher, inputs, direction, accelerator)
            actual.backward()
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(s.grad, gradient))
            self.assertEqual(output.logits.dtype, torch.bfloat16)
            self.assertIsNone(t.grad)

    def test_paired_comparison_checks_alignment_and_counts_direction(self):
        a = [{"id": str(i), "gold": "A", "predicted": p} for i, p in enumerate("ABBC")]
        b = [{"id": str(i), "gold": "A", "predicted": p} for i, p in enumerate("AABB")]
        counts = kd_ecqa.paired_counts(a, b)
        self.assertEqual(counts["second_minus_first_accuracy_pp"], 25.)
        self.assertEqual(counts["same_wrong_answer"], 1)
        self.assertEqual(counts["both_wrong"], 2)
        b[0]["id"] = "different"
        with self.assertRaises(AssertionError):
            kd_ecqa.paired_counts(a, b)

    def test_kd_selection_uses_dev_and_equal_epoch_budget(self):
        candidates = [{"epoch": ep, "learning_rate": kd_ecqa.LR,
            "dev": {"accuracy": .7, "macro_f1": .69, "choice_nll": .5},
            "test": {"accuracy": ep/3}} for ep in (1, 2, 3)]
        self.assertEqual(kd_ecqa.choose(candidates)["epoch"], 1)
        candidates[1]["dev"]["choice_nll"] = .4
        self.assertEqual(kd_ecqa.choose(candidates)["epoch"], 2)
        candidates[2]["dev"]["accuracy"] = .71
        self.assertEqual(kd_ecqa.choose(candidates)["epoch"], 3)
        with self.assertRaises(AssertionError):
            kd_ecqa.choose(candidates[:2])

    def test_kd_evaluation_globals_do_not_modify_sft_destinations(self):
        original = (kd_ecqa.base.LOCAL, kd_ecqa.base.OUT)
        evaluator = kd_ecqa.isolated_helper(kd_ecqa.base, "test_kd_evaluation")
        evaluator.LOCAL = kd_ecqa.LOCAL / "ta_fkl"
        evaluator.OUT = kd_ecqa.OUT / "ta_fkl"
        self.assertEqual((kd_ecqa.base.LOCAL, kd_ecqa.base.OUT), original)
        self.assertNotEqual(evaluator.LOCAL, original[0])
        self.assertNotEqual(evaluator.OUT, original[1])

    def test_chunked_kl_matches_dense_loss_and_student_gradient(self):
        import torch
        torch.manual_seed(10)
        for direction in ("fkl", "rkl"):
            s = torch.randn(5, 7, requires_grad=True)
            t = torch.randn(5, 7, requires_grad=True)
            logq, logp = s.log_softmax(-1), t.detach().log_softmax(-1)
            expected = ((logp.exp()*(logp-logq)) if direction == "fkl" else (logq.exp()*(logq-logp))).sum(-1).mean()
            gradient = torch.autograd.grad(expected, s)[0]
            actual = kd_ecqa.token_kl(s, t, direction, chunk_size=2)
            actual.backward()
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
            self.assertTrue(torch.allclose(s.grad, gradient, atol=1e-6))
            self.assertIsNone(t.grad)
        with self.assertRaises(AssertionError):
            kd_ecqa.token_kl(s, t, "invalid")

    def test_response_positions_and_teacher_gradient_are_correct(self):
        import torch
        class Model:
            def __init__(self, logits):
                self.logits = logits
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(logits=self.logits[:, -kwargs["logits_to_keep"]:])
        for direction in ("fkl", "rkl"):
            s = torch.randn(1, 5, 7, requires_grad=True)
            t = torch.randn(1, 5, 7, requires_grad=True)
            student, teacher = Model(s), Model(t)
            inputs = {"input_ids": torch.zeros((1, 5), dtype=torch.long),
                "attention_mask": torch.ones((1, 5)), "labels": torch.tensor([[-100, -100, -100, 3, 4]])}
            loss, _ = kd_ecqa.response_kl(student, teacher, inputs, direction)
            expected = kd_ecqa.token_kl(s[:, 2:4].reshape(2, 7), t[:, 2:4].reshape(2, 7), direction)
            self.assertTrue(torch.allclose(loss, expected))
            loss.backward()
            self.assertIsNone(t.grad)
            self.assertEqual(int(s.grad[:, :2].count_nonzero()), 0)
            self.assertEqual(int(s.grad[:, 4:].count_nonzero()), 0)
            self.assertFalse(student.kwargs["use_cache"])
            self.assertEqual(student.kwargs["logits_to_keep"], 3)


if __name__ == "__main__":
    unittest.main()
