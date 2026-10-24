from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_ecqa_ce as audit
from audit_ecqa_ce import summarize
import render_results


class ComparableCeTests(unittest.TestCase):
    def test_mean_uses_equal_example_weights_for_different_response_lengths(self):
        records = [{"id": "short", "ce": 1., "response_tokens": 2, "seconds": .1},
                   {"id": "long", "ce": 3., "response_tokens": 6, "seconds": .2}]
        result = summarize(records)
        self.assertEqual(result["response_ce"], 2.)
        self.assertEqual(result["supervised_tokens"], 8)
        self.assertEqual(result["n"], 2)
        with self.assertRaises(AssertionError):
            summarize([records[0], records[0]])
        with self.assertRaises(AssertionError):
            summarize([{**records[0], "ce": float("nan")}])

    def test_frozen_checkpoint_ce_masks_prompt_and_resumes_without_recomputing(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([0., 1., 2.]))
                self.calls = 0

            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
                self.calls += 1
                self.assert_no_cache = not use_cache
                return SimpleNamespace(logits=self.weight.view(1, 1, 3).expand(1, logits_to_keep, 3))

        model = Model()
        before = model.weight.detach().clone()
        features = [{"input_ids": [0, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                    {"input_ids": [0, 0, 1], "attention_mask": [1, 1, 1], "labels": [-100, 0, 1]}]
        tensor = torch.tensor
        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return tensor(*args, **kwargs)
        temporary_root = audit.base.ROOT / "local/test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temporary_root) as folder, patch.object(audit, "LOCAL", Path(folder)/"local"), \
                patch.object(audit, "OUT", Path(folder)/"out"), \
                patch.object(audit.gpu_safety, "stopping_guard", return_value=SimpleNamespace(check=lambda: None)), \
                patch("torch.tensor", side_effect=cpu_tensor):
            assert Path(folder).resolve().is_relative_to(audit.base.ROOT.resolve())
            rows = [{"id": "short"}, {"id": "long"}]
            result = audit.evaluate_split(model, features, rows, 1, "train", {"fingerprint": "test"})
            loss = torch.nn.functional.cross_entropy(before.expand(2, 3), tensor([0, 1]))
            expected = (torch.nn.functional.cross_entropy(before.view(1, 3), tensor([2]))+loss)/2
            self.assertAlmostEqual(result["response_ce"], float(expected), places=6)
            self.assertEqual(result["supervised_tokens"], 3)
            self.assertFalse(model.training)
            self.assertFalse(model.weight.requires_grad)
            self.assertIsNone(model.weight.grad)
            self.assertTrue(torch.equal(model.weight, before))
            self.assertTrue(model.assert_no_cache)
            resumed = audit.evaluate_split(model, features, rows, 1, "train", {"fingerprint": "test"})
            self.assertEqual(resumed, result)
            self.assertEqual(model.calls, 2)

    def test_ce_figure_section_does_not_replace_baseline_table(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot
        temporary_root = audit.base.ROOT / "local/test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        token_data = json.loads((render_results.OUT/"token_lengths.json").read_text(encoding="utf-8"))
        with TemporaryDirectory(dir=temporary_root) as folder:
            root = Path(folder).resolve()
            assert root.is_relative_to(audit.base.ROOT.resolve())
            out = root/"artifacts/ecqa"
            (out/"ce_audit").mkdir(parents=True)
            audit.base.write(out/"token_lengths.json", token_data)
            audit.base.write(out/"ce_audit/protocol.json", {"fingerprint": "unit-test"})
            audit.base.write(out/"ce_audit/summary.json", {"status": "completed", "fingerprint": "unit-test",
                "epochs": [{"epoch": ep, "train": {"n": 7566, "response_ce": 1./ep},
                            "dev": {"n": 1079, "response_ce": 1.1/ep}} for ep in (1, 2, 3)]})
            sections = "\n".join(f"<!-- {key}_START -->\nPENDING\n<!-- {key}_END -->"
                                 for key in ("BASELINE_STATUS", "CE_AUDIT"))
            for name in render_results.DOCS:
                (root/name).write_text(sections, encoding="utf-8")
            with patch.object(render_results, "ROOT", root), patch.object(render_results, "OUT", out), \
                    patch("matplotlib.figure.Figure.savefig"):
                render_results.render()
            for name in render_results.DOCS:
                text = (root/name).read_text(encoding="utf-8")
                baseline = text.split("<!-- BASELINE_STATUS_START -->")[1].split("<!-- BASELINE_STATUS_END -->")[0]
                self.assertIn("| Qwen3.5-4B | original |", baseline)
                self.assertNotIn("Train eval CE", baseline)
                self.assertIn("| Epoch | Train eval CE | Dev CE |", text)


if __name__ == "__main__":
    unittest.main()
