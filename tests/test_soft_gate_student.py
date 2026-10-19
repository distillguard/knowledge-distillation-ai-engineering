"""Check full-parameter SFT masking, accumulation and decay groups on CPU."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
import train_soft_gate_student as student


class StudentTests(unittest.TestCase):
    def test_collator_preserves_eos_targets_and_masks_prompt_and_padding(self):
        rows = [{"input_ids": [3, 4, 50256], "labels": [-100, 4, 50256]}]
        inputs, labels = student.collate(rows, "cpu")
        self.assertEqual(inputs["input_ids"].shape, (1, 512))
        self.assertEqual(labels[0, :4].tolist(), [-100, 4, 50256, -100])
        self.assertEqual(inputs["attention_mask"].sum().item(), 3)

    def test_accumulation_matches_effective_batch_token_mean_gradient(self):
        torch.manual_seed(10)
        model = torch.nn.Linear(3, 5)
        class Tiny:
            def __call__(self, input_ids, **kwargs):
                features = torch.nn.functional.one_hot(input_ids, 3).float()
                return SimpleNamespace(logits=model(features))
        rows = [{"input_ids": [0, 1, 2], "labels": [-100, 1, 2]},
                {"input_ids": [1, 2], "labels": [-100, 2]}]
        # The fake vocabulary uses 0 for padding; input padding remains excluded from loss.
        original = student.collate
        def small(records, device):
            inputs, labels = original(records, device)
            inputs["input_ids"][inputs["attention_mask"] == 0] = 0
            return inputs, labels
        from unittest.mock import patch
        with patch.object(student, "collate", side_effect=small):
            loss, count = student.response_loss(Tiny(), rows, "cpu")
            (loss/count).backward()
            expected = [p.grad.clone() for p in model.parameters()]
            model.zero_grad(set_to_none=True)
            for row in rows:
                loss, _ = student.response_loss(Tiny(), [row], "cpu")
                (loss/count).backward()
        for parameter, gradient in zip(model.parameters(), expected):
            self.assertTrue(torch.allclose(parameter.grad, gradient, atol=1e-7))

    def test_optimizer_groups_preserve_official_bias_and_layernorm_exclusions(self):
        model = torch.nn.Module()
        model.linear = torch.nn.Linear(3, 4)
        model.ln_f = torch.nn.LayerNorm(4)
        groups = student.parameter_groups(model)
        self.assertEqual({id(p) for p in groups[0]["params"]}, {id(model.linear.weight)})
        self.assertEqual(len(groups[1]["params"]), 3)
        self.assertEqual(groups[1]["weight_decay"], 0)


if __name__ == "__main__":
    unittest.main()
