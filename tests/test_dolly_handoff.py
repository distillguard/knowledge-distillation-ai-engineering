"""Exercise portable handoffs, completion checks, and unsafe imports without ML dependencies."""

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dolly_handoff as handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "source"
        self.run_id = "4090-gemma-01"
        self.result_prefix = f"results/external_nonquantized/{self.run_id}/"
        self.inputs = {split: (json.dumps({"id": f"raw:{index}", "instruction": split,
                                          "input": "", "output": "example"}) + "\n").encode()
                       for index, split in enumerate(handoff.SPLITS)}
        self.parent = {
            "dataset": {"id": "MiniLLM/dolly", "revision": "pinned-dataset"},
            "counts": dict.fromkeys(handoff.SPLITS, 1),
            "lora": {"r": 64, "lora_alpha": 16, "lora_dropout": 0.05,
                     "target_modules": "all-linear", "bias": "none", "task_type": "CAUSAL_LM"},
            "learning_rate_candidates": [1e-4, 2e-4], "epochs": 3, "effective_batch_examples": 16,
            "training_seed": 0, "optimizer": {"name": "torch.optim.AdamW"},
            "scheduler": {"name": "constant_with_warmup", "warmup_ratio": 0.03},
            "response_truncation": False, "selection": "development NLL",
            "checkpoint_and_validation_cadence": "every complete epoch"}
        self.parent["fingerprint"] = handoff._fingerprint(self.parent, parent=True)
        self.parent["prepared_files_sha256"] = {
            f"{split}.jsonl": handoff._sha(data) for split, data in self.inputs.items()}
        self.model = {"id": "google/gemma-2-9b-it", "revision": "pinned-gemma"}
        self.write_json("results/dolly_sft_v1/protocol.json", self.parent)
        self.write_json("results/dolly_sft_additional_v1/registry.json", {handoff.MODEL: self.model})
        self.reference = {"revision": self.model["revision"], "parent_protocol_fingerprint": self.parent["fingerprint"],
                          "prepared_files_sha256": {
                              f"tmp/dolly_sft_additional_v1/{handoff.MODEL}/{handoff.MODEL}/{split}.jsonl": handoff._sha(data)
                              for split, data in self.inputs.items()}}
        self.write_json(f"results/dolly_sft_additional_v1/{handoff.MODEL}/preparation.json", self.reference)
        for split, data in self.inputs.items():
            self.write(f"tmp/dolly_sft_v1/{split}.jsonl", data)
        self.make_completed_run()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def write_json(self, relative, value):
        return self.write(relative, handoff._json_bytes(value))

    def read_json(self, relative):
        return json.loads((self.root / relative).read_bytes())

    def make_completed_run(self):
        self.protocol = copy.deepcopy(self.parent)
        self.protocol.update(run_id=self.run_id, parent_protocol_fingerprint=self.parent["fingerprint"],
                             models={handoff.MODEL: self.model}, max_total_training_tokens=1024,
                             max_prompt_tokens=None, precision={handoff.MODEL: {
                                 "base_weights": "BF16, no weight quantization", "normalization": "FP32",
                                 "trainable_lora": "FP32", "compute": "BF16 autocast"}},
                             source_hash_policy="UTF-8 source bytes with CRLF normalized to LF")
        self.write("scripts/dolly_external.py", b"# pinned external source\r\n")
        self.protocol["source_script_sha256"] = handoff._sha(b"# pinned external source\n")
        for key, relative in (("engine_script_sha256", "scripts/dolly_sft.py"),
                              ("study_script_sha256", "scripts/dolly_study.py"),
                              ("handoff_script_sha256", "scripts/dolly_handoff.py"),
                              ("external_probe_script_sha256", "scripts/preflight_external_gemma.py")):
            self.write(relative, b"# pinned source\r\n")
            self.protocol[key] = handoff._sha(b"# pinned source\n")
        self.protocol["prepared_files_sha256"] = {
            f"tmp/external_nonquantized/{self.run_id}/{part}/{split}.jsonl": handoff._sha(data)
            for split, data in self.inputs.items() for part in ("source", handoff.MODEL)}
        for split, data in self.inputs.items():
            for part in ("source", handoff.MODEL):
                self.write(f"tmp/external_nonquantized/{self.run_id}/{part}/{split}.jsonl", data)
        self.protocol["fingerprint"] = handoff._fingerprint(self.protocol)
        fingerprint = self.protocol["fingerprint"]
        self.write_json(self.result_prefix + "protocol.json", self.protocol)
        self.write_json(self.result_prefix + "preparation.json", {
            "status": "prepared", "same_original_records": True, "protocol_fingerprint": fingerprint,
            "parent_protocol_fingerprint": self.parent["fingerprint"], "tokenization_matches_reference": True,
            "prepared_files_sha256": self.protocol["prepared_files_sha256"]})
        for name in ("preflight.json", "trainer_preflight.json"):
            self.write_json(self.result_prefix + name, {"protocol_fingerprint": fingerprint, "models": {
                handoff.MODEL: {"status": "passed", "quantized_linear_layers": 0}}})
        self.write_json(self.result_prefix + "memory_check.json", {
            "protocol_fingerprint": fingerprint, "fits_dedicated_budget": True})
        for lr, nll in ((1e-4, 2.5), (2e-4, 2.0)):
            key = f"{handoff.MODEL}_lr{lr:.0e}"
            adapter = f"tmp/external_nonquantized/{self.run_id}/runs/{key}/best_adapter"
            self.write_json(self.result_prefix + key + ".json", {
                "status": "completed", "model": handoff.MODEL, "protocol_fingerprint": fingerprint,
                "learning_rate": lr, "adapter": adapter, "training_metrics": {"epoch": 3.0},
                "selected_development": {"eval_loss": nll}})
            self.write_json(adapter + "/adapter_config.json", {**self.parent["lora"], "peft_type": "LORA",
                            "base_model_name_or_path": self.model["id"], "revision": self.model["revision"]})
            self.write(adapter + "/adapter_model.safetensors", self.adapter_bytes())
            # These must never enter the portable package.
            self.write(adapter + "/tokenizer_config.json", b'{"token": "private"}')
            self.write(adapter + "/optimizer.pt", b"optimizer")
        self.write_json(self.result_prefix + f"{handoff.MODEL}_selected.json", {
            "status": "selected", "protocol_fingerprint": fingerprint, "learning_rate": 2e-4,
            "adapter": f"tmp/external_nonquantized/{self.run_id}/runs/{handoff.MODEL}_lr2e-04/best_adapter",
            "development_nll": 2.0})

    @staticmethod
    def adapter_bytes(name="base_model.model.layers.0.q_proj.lora_A.weight"):
        header = json.dumps({name: {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]}}).encode()
        return struct.pack("<Q", len(header)) + header + b"\0" * 4

    def export(self):
        return handoff.write_result_bundle(self.root, self.run_id, self.directory / "results.zip")

    def destination(self):
        target = self.directory / "receiver"
        for relative in ("results/dolly_sft_v1/protocol.json", "results/dolly_sft_additional_v1/registry.json",
                         f"results/dolly_sft_additional_v1/{handoff.MODEL}/preparation.json",
                         "scripts/dolly_external.py", "scripts/dolly_sft.py", "scripts/dolly_study.py",
                         "scripts/dolly_handoff.py", "scripts/preflight_external_gemma.py"):
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.root / relative).read_bytes())
        return target

    def repack(self, source, transform):
        with zipfile.ZipFile(source) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        transform(files)
        target = self.directory / "tampered.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return target

    def test_input_roundtrip_keeps_exact_original_bytes_only(self):
        self.write("tmp/dolly_sft_v1/gpt2_base/train.jsonl", b"encoded tokens")
        bundle = handoff.pack_inputs(self.root, self.directory / "inputs.zip")
        self.assertEqual(handoff.read_inputs(bundle, self.parent), self.inputs)
        with zipfile.ZipFile(bundle) as archive:
            self.assertEqual(len(archive.namelist()), 5)

    def test_input_hash_count_and_id_checks(self):
        self.write("tmp/dolly_sft_v1/train.jsonl", b"wrong\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            handoff.pack_inputs(self.root, self.directory / "inputs.zip")
        self.write("tmp/dolly_sft_v1/train.jsonl", self.inputs["train"])
        self.parent["counts"]["train"] = 2
        self.parent["fingerprint"] = handoff._fingerprint(self.parent, parent=True)
        self.write_json("results/dolly_sft_v1/protocol.json", self.parent)
        with self.assertRaisesRegex(ValueError, "count"):
            handoff.pack_inputs(self.root, self.directory / "inputs.zip")
        self.parent["counts"]["train"] = 1
        self.parent["fingerprint"] = handoff._fingerprint(self.parent, parent=True)
        self.write_json("results/dolly_sft_v1/protocol.json", self.parent)
        self.write("tmp/dolly_sft_v1/development.jsonl", self.inputs["train"])
        self.parent["prepared_files_sha256"]["development.jsonl"] = handoff._sha(self.inputs["train"])
        self.write_json("results/dolly_sft_v1/protocol.json", self.parent)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            handoff.pack_inputs(self.root, self.directory / "inputs.zip")

    def test_input_receiver_requires_same_canonical_protocol(self):
        bundle = handoff.pack_inputs(self.root, self.directory / "inputs.zip")
        different = copy.deepcopy(self.parent)
        different["training_seed"] = 42
        different["fingerprint"] = handoff._fingerprint(different, parent=True)
        with self.assertRaisesRegex(ValueError, "does not match"):
            handoff.read_inputs(bundle, different)

    def test_corrupted_payload_rejected_before_parsing(self):
        bundle = handoff.pack_inputs(self.root, self.directory / "inputs.zip")
        tampered = self.repack(bundle, lambda files: files.update({"inputs/train.jsonl": b"tampered"}))
        with self.assertRaisesRegex(ValueError, "checksum"):
            handoff.read_inputs(tampered, self.parent)

    def test_unsafe_archive_names_never_extracted(self):
        bundle = self.export()
        for name in ("../escape", "C:/escape", "x\\escape", "x/NUL", "x/a:secret", "/escape"):
            with self.subTest(name=name):
                tampered = self.repack(bundle, lambda files: files.update({name: b"unsafe"}))
                with self.assertRaises(ValueError):
                    handoff.import_result_bundle(self.destination(), tampered)
                self.assertFalse((self.directory / "escape").exists())

    def test_case_colliding_archive_paths_rejected(self):
        bundle = self.export()
        tampered = self.repack(bundle, lambda files: files.update({"MANIFEST.JSON": files["manifest.json"]}))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            handoff.import_result_bundle(self.destination(), tampered)

    def test_result_roundtrip_is_idempotent_and_leaves_canonical_results_unchanged(self):
        bundle = self.export()
        target = self.destination()
        parent_bytes = (target / "results/dolly_sft_v1/protocol.json").read_bytes()
        imported = handoff.import_result_bundle(target, bundle)
        self.assertEqual(imported, target / "results/external_nonquantized" / self.run_id)
        self.assertEqual(handoff.import_result_bundle(target, bundle), imported)
        self.assertEqual(parent_bytes, (target / "results/dolly_sft_v1/protocol.json").read_bytes())
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read("manifest.json"))
            self.assertIs(manifest["evaluated"], False)
            self.assertFalse(any("tokenizer" in name or "optimizer" in name for name in names))
            self.assertEqual(len(names), 1 + len(handoff.SUMMARY_NAMES) + 4)

    def test_conflict_aborts_before_creating_other_run_files(self):
        bundle = self.export()
        target = self.destination()
        conflict = target / (self.result_prefix + "memory_check.json")
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            handoff.import_result_bundle(target, bundle)
        self.assertEqual(conflict.read_bytes(), b"existing")
        self.assertEqual(list(conflict.parent.iterdir()), [conflict])

    def test_incomplete_candidate_and_partial_epochs_rejected(self):
        relative = self.result_prefix + f"{handoff.MODEL}_lr1e-04.json"
        record = self.read_json(relative)
        for changes, reason in (({"status": "training"}, "not completed"),
                                ({"training_metrics": {"epoch": 2.99}}, "all epochs"),
                                ({"protocol_fingerprint": "stale"}, "identity mismatch"),
                                ({"adapter": "C:/private/best_adapter"}, "Nonportable")):
            with self.subTest(changes=changes):
                self.write_json(relative, {**record, **changes})
                with self.assertRaisesRegex(ValueError, reason):
                    self.export()

    def test_selected_lr_adapter_and_nll_must_match_minimum(self):
        relative = self.result_prefix + f"{handoff.MODEL}_selected.json"
        original = self.read_json(relative)
        for changes in ({"learning_rate": 1e-4}, {"adapter": "wrong"}, {"development_nll": 0.0}):
            with self.subTest(changes=changes):
                self.write_json(relative, {**original, **changes})
                with self.assertRaisesRegex(ValueError, "winning candidate"):
                    self.export()

    def test_nonfinite_development_loss_rejected(self):
        relative = self.result_prefix + f"{handoff.MODEL}_lr1e-04.json"
        record = self.read_json(relative)
        record["selected_development"]["eval_loss"] = float("nan")
        self.write(relative, json.dumps(record).encode())
        with self.assertRaisesRegex(ValueError, "Invalid development loss"):
            self.export()

    def test_adapter_config_and_base_weights_rejected(self):
        prefix = f"tmp/external_nonquantized/{self.run_id}/runs/{handoff.MODEL}_lr1e-04/best_adapter/"
        config = self.read_json(prefix + "adapter_config.json")
        self.write_json(prefix + "adapter_config.json", {**config, "r": 8})
        with self.assertRaisesRegex(ValueError, "LoRA setting mismatch"):
            self.export()
        self.write_json(prefix + "adapter_config.json", config)
        self.write(prefix + "adapter_model.safetensors", self.adapter_bytes("model.layers.0.q_proj.weight"))
        with self.assertRaisesRegex(ValueError, "other than the trained LoRA"):
            self.export()

    def test_changed_source_and_altered_experiment_conditions_rejected(self):
        self.write("scripts/dolly_external.py", b"changed source")
        with self.assertRaisesRegex(ValueError, "Source fingerprint mismatch"):
            self.export()
        self.write("scripts/dolly_external.py", b"# pinned external source\n")
        self.protocol["epochs"] = 1
        self.protocol["fingerprint"] = handoff._fingerprint(self.protocol)
        self.write_json(self.result_prefix + "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "condition differs.*epochs"):
            self.export()

    def test_refuses_overwrite_of_existing_bundle(self):
        self.export()
        with self.assertRaisesRegex(ValueError, "overwrite bundle"):
            self.export()

    def test_invalid_run_ids_rejected(self):
        for run_id in ("../bad", "C:\\bad", "", "a" * 65):
            with self.subTest(run_id=run_id), self.assertRaisesRegex(ValueError, "run ID"):
                handoff.write_result_bundle(self.root, run_id, self.directory / "bad.zip")


if __name__ == "__main__":
    unittest.main()
