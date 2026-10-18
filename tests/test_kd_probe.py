"""Check probe loss gradients, dropout restoration and genuine MC variation on CPU."""
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_kd_pair as probe
import torch
from filelock import FileLock, Timeout


class OPTAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.
        self.other = torch.nn.Dropout(.05)


class ProbeTests(unittest.TestCase):
    def test_loss_masks_predictions_and_freezes_teacher(self):
        teacher = torch.tensor([[[1., 0., -1.], [2., 1., -1.], [0., 0., 0.]]], requires_grad=True)
        student = torch.tensor([[[0., 2., 0.], [0., 1., 2.], [1., 0., 1.]]], requires_grad=True)
        loss, stats = probe.soft_loss(teacher, student, torch.tensor([[-100, -100, 0]]))
        p, q = teacher.detach()[0, 1].softmax(-1), student.detach()[0, 1].softmax(-1)
        expected_forward = sum(float(a) * math.log(float(a / b)) for a, b in zip(p, q))
        expected_reverse = sum(float(b) * math.log(float(b / a)) for a, b in zip(p, q))
        self.assertAlmostEqual(stats["forward_kl"], expected_forward, places=6)
        self.assertAlmostEqual(stats["reverse_kl"], expected_reverse, places=6)
        self.assertEqual(stats["response_tokens"], 1)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(int(torch.count_nonzero(student.grad[0, [0, 2]])), 0)
        self.assertGreater(float(student.grad[0, 1].norm()), 0)
        equal = student.detach().clone().requires_grad_()
        zero, _ = probe.soft_loss(equal.detach(), equal, torch.tensor([[-100, -100, 0]]))
        self.assertAlmostEqual(float(zero.detach()), 0, places=6)

    def test_gpt2_attention_dropout_restores_its_original_probability(self):
        class GPT2Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_dropout = torch.nn.Dropout(0.)
        model = torch.nn.Module()
        model.config = SimpleNamespace(model_type="gpt2", num_hidden_layers=1)
        model.attention = GPT2Attention()
        model.eval()
        with probe.mc_dropout(model):
            self.assertTrue(model.attention.training)
            self.assertEqual(model.attention.attn_dropout.p, .1)
        self.assertFalse(model.attention.training)
        self.assertEqual(model.attention.attn_dropout.p, 0.)

    def test_mc_restores_scalar_dropout_and_mixed_training_flags_after_error(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(model_type="opt", num_hidden_layers=1, layerdrop=0)
        model.attention = OPTAttention()
        model.eval(); model.attention.other.train()
        before = [m.training for m in model.modules()]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with probe.mc_dropout(model) as count:
                self.assertEqual(count, 1)
                self.assertTrue(model.attention.training)
                self.assertEqual(model.attention.dropout, .1)
                self.assertEqual(model.attention.other.p, .05)
                raise RuntimeError("interrupted")
        self.assertEqual(model.attention.dropout, 0)
        self.assertEqual([m.training for m in model.modules()], before)

    def test_bald_distinguishes_distribution_changes_from_identical_passes(self):
        class Fixed:
            def __init__(self, values):
                self.values, self.index = values, 0
            def __call__(self, **kwargs):
                probabilities = torch.tensor(self.values[self.index])
                self.index += 1
                return SimpleNamespace(logits=probabilities.log().reshape(1, 1, 2).repeat(1, 2, 1))
        labels = torch.tensor([[-100, 0]])
        result = probe.bald(Fixed([[.8, .2], [.2, .8], [.8, .2]]), {}, labels)
        h = lambda p: -sum(x * math.log(x) for x in p)
        self.assertAlmostEqual(result["mean_bald"], h([.6, .4]) - h([.8, .2]), places=6)
        self.assertGreater(result["probability_entries_differing_from_first_pass"], 0)
        self.assertEqual(result["mean_top1_probability_range"], 0)
        same = probe.bald(Fixed([[.8, .2]] * 3), {}, labels)
        self.assertLess(abs(same["mean_bald"]), 1e-6)
        self.assertEqual(same["probability_entries_differing_from_first_pass"], 0)

    def test_shared_lock_blocks_before_model_or_cuda_work(self):
        with tempfile.TemporaryDirectory(dir=probe.ROOT / "tmp") as folder, patch.object(probe, "LOCAL", Path(folder)), patch.object(probe, "run") as run:
            with FileLock(str(Path(folder) / "worker.lock")):
                with self.assertRaises(Timeout):
                    probe.main(["--pair", "gpt2"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
