"""Protect original/SFT output isolation and disabled-adapter inference on CPU."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_original_models as original
import torch
from filelock import FileLock, Timeout


class OriginalEvaluationTests(unittest.TestCase):
    def test_shared_worker_lock_prevents_concurrent_inference(self):
        with tempfile.TemporaryDirectory(dir=original.ROOT / "tmp") as folder, \
                patch.object(original, "SFT_LOCAL", Path(folder)), patch.object(original, "evaluate") as run:
            with FileLock(str(Path(folder) / "worker.lock")):
                with self.assertRaises(Timeout):
                    original.main(["--model", "gpt2_base"])
            run.assert_not_called()

    def test_publish_is_atomic_and_cache_checks_identity_ids_and_raw_hash(self):
        with tempfile.TemporaryDirectory(dir=original.ROOT / "tmp") as folder:
            root = Path(folder)
            records = [{"id": "a", "category": "task", "references": ["correct answer"]}]
            spec = {"fingerprint": "original-test", "phase": "before_our_dolly_sft"}
            saved = original.evaluation.LOCAL, original.evaluation.OUT

            def fake_generate(name, key, model, tokenizer, rows, fingerprint):
                self.assertEqual(fingerprint, "original-test:vicuna")
                generated = [{"id": "a", "response": "correct answer", "generated_tokens": 2, "length_capped": False}]
                original.write_json(original.evaluation.LOCAL / "evaluation" / name / f"{key}_seed10.json", generated)
                return generated

            publish = original.evaluation.publish

            def interrupted_publish(*args):
                publish(*args)
                raise RuntimeError("interrupted before final publication")

            with patch.object(original, "OUT", root / "public"), patch.object(original, "LOCAL", root / "private"), \
                    patch.object(original.evaluation, "generate", side_effect=fake_generate):
                with patch.object(original.evaluation, "publish", side_effect=interrupted_publish):
                    with self.assertRaises(RuntimeError):
                        original.generate("gpt2_base", "vicuna", None, None, records, spec)
                self.assertFalse((original.OUT / "gpt2_base/vicuna.json").exists())
                self.assertEqual((original.evaluation.LOCAL, original.evaluation.OUT), saved)
                original.generate("gpt2_base", "vicuna", None, None, records, spec)
                result = original.load_completed("gpt2_base", "vicuna", spec, records)
                self.assertEqual(result["metrics"]["rougeL"], 100)
                self.assertTrue(result["per_item_csv"].endswith("public/gpt2_base/vicuna_per_item.csv"))
                self.assertEqual((original.evaluation.LOCAL, original.evaluation.OUT), saved)
                for bad_spec, bad_rows in (({**spec, "fingerprint": "sft"}, records),
                                            (spec, [{**records[0], "id": "wrong"}])):
                    with self.assertRaises(AssertionError):
                        original.load_completed("gpt2_base", "vicuna", bad_spec, bad_rows)
                (original.ROOT / result["raw_responses"]).write_text("[]", encoding="utf-8")
                with self.assertRaises(AssertionError):
                    original.load_completed("gpt2_base", "vicuna", spec, records)

    def test_dolly_redirection_is_restored_on_failure(self):
        legacy = original.evaluation.legacy
        saved = legacy.LOCAL, legacy.OUT
        records = [{"id": "a"}]
        with patch.object(original, "rows", return_value=records), \
                patch.object(legacy, "generate", side_effect=RuntimeError("generation failed")):
            with self.assertRaises(RuntimeError):
                original.generate("gpt2_base", "dolly", None, None, records, {"fingerprint": "test"})
        self.assertEqual((legacy.LOCAL, legacy.OUT), saved)

    def test_completed_original_generation_never_loads_model(self):
        with tempfile.TemporaryDirectory(dir=original.ROOT / "tmp") as folder, \
                patch.object(original, "OUT", Path(folder)), \
                patch.object(original.evaluation, "prepared_protocol", return_value={}), \
                patch.object(original, "identity", return_value={"fingerprint": "test"}), \
                patch.object(original, "rows", return_value=[{"id": "a"}]), \
                patch.object(original, "load_completed", return_value={"status": "completed"}), \
                patch.object(original.study, "load_model") as load:
            original.evaluate("gpt2_base", ["dolly"])
            load.assert_not_called()

    def test_classification_reuses_only_verified_private_predictions(self):
        legacy = original.evaluation.legacy
        records = [{"source_row": 0, "label": 0}, {"source_row": 1, "label": 1}]
        predictions = {"source_rows": [0, 1], "labels": [0, 1], "predictions": [0, 1],
                       "positive_conditional_probability": [.2, .8]}
        scores = legacy.classification_metrics([0, 1], [0, 1], [.2, .8])
        saved = legacy.LOCAL

        def classify(name, model, data):
            original.write_json(legacy.LOCAL / "evaluation" / f"{name}_sst2_predictions.json", predictions)
            return {"sst2": scores}

        with tempfile.TemporaryDirectory(dir=original.ROOT / "tmp") as folder, \
                patch.object(original, "OUT", Path(folder) / "public"), \
                patch.object(original, "LOCAL", Path(folder) / "private"), \
                patch.object(legacy, "classification_data", return_value=({"sst2": records}, {"sst2": {"rows": 2}})), \
                patch.object(legacy, "classify", side_effect=classify) as run:
            spec = {"fingerprint": "original", "phase": "before_our_dolly_sft"}
            original.classify("gpt2_base", None, None, spec)
            original.classify("gpt2_base", None, None, spec)
            run.assert_called_once()
            self.assertEqual(legacy.LOCAL, saved)
            path = original.LOCAL / "gpt2_base/evaluation/gpt2_base_sst2_predictions.json"
            original.write_json(path, {**predictions, "labels": [1, 0]})
            with self.assertRaises(AssertionError):
                original.classify("gpt2_base", None, None, spec)

    def test_original_inference_disables_a_nonzero_adapter(self):
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora import LoraLayer
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(123)
        base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16,
                                        n_positions=16, bos_token_id=1, eos_token_id=2,
                                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
        ids = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            expected = base(ids).logits.clone()
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"],
                                             fan_in_fan_out=True, task_type="CAUSAL_LM"))
        layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
        for layer in layers:
            torch.nn.init.normal_(layer.lora_B["default"].weight)
        with torch.inference_mode():
            self.assertFalse(torch.allclose(model(ids).logits, expected))

        def check(name, key, active_model, tokenizer, records, spec):
            self.assertTrue(all(layer.disable_adapters for layer in layers))
            self.assertFalse(any(parameter.requires_grad for parameter in active_model.parameters()))
            with torch.inference_mode():
                torch.testing.assert_close(active_model(ids).logits, expected)

        with tempfile.TemporaryDirectory(dir=original.ROOT / "tmp") as folder, \
                patch.object(original, "OUT", Path(folder)), \
                patch.object(original.evaluation, "prepared_protocol", return_value={}), \
                patch.object(original, "identity", return_value={"fingerprint": "test"}), \
                patch.object(original, "rows", return_value=[{"id": "a"}]), \
                patch.object(original, "load_completed", return_value=None), \
                patch.object(original.study, "load_model", return_value=model) as load, \
                patch.object(original.study, "tokenizer_for", return_value=None), \
                patch.object(original, "generate", side_effect=check) as generate, \
                patch.object(torch.cuda, "empty_cache"):
            original.evaluate("gpt2_base", ["dolly"])
            load.assert_called_once_with("gpt2_base")
            generate.assert_called_once()
        self.assertTrue(all(not layer.disable_adapters for layer in layers))


if __name__ == "__main__":
    unittest.main()
