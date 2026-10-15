"""Exercise external-run isolation and failure/resume behavior without loading ML."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dolly_external as external


class ExternalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_id = "gemma_4090_test"
        patcher = patch.object(external, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.local, self.out = external.paths(self.run_id)
        self.local.mkdir(parents=True)
        self.info = {"gpu": "RTX 4090", "total_memory_bytes": 24 * 2**30, "free_memory_bytes": 23 * 2**30,
                     "cuda": "12.8", "python": "3.12", "platform": "test", "versions": {"torch": "test"}}
        self.spec = {"fingerprint": "test", "models": {external.MODEL: {"id": "google/gemma-2-9b-it", "revision": "pinned"}}}

    def write(self, path, record):
        external.write_json(path, record)

    def fake_probe(self, command, **kwargs):
        kind = command[-1]
        filename = "preflight.json" if kind == "manual" else "trainer_preflight.json"
        self.write(self.out / filename, {"protocol_fingerprint": "test", "models": {external.MODEL: {
            "status": "passed", "peak_allocated_mib": 20000, "available_before_probe_bytes": 22 * 2**30}}})
        return SimpleNamespace(returncode=0)

    def fake_train(self, names):
        self.assertEqual(names, [external.MODEL])
        for lr in (1e-4, 2e-4):
            key = f"{external.MODEL}_lr{lr:.0e}"
            adapter = self.local / "runs" / key / "best_adapter"
            self.write(adapter / "adapter_config.json", {"base_model_name_or_path": "C:/old/cache"})
            self.write(self.out / f"{key}.json", {"best_checkpoint": str(adapter.parent / "checkpoint-2085"),
                       "adapter": adapter.relative_to(self.root).as_posix()})

    def engine(self):
        return SimpleNamespace(LR_CANDIDATES=(1e-4, 2e-4), train=self.fake_train)

    def run_training(self, engine=None, probe=None):
        with patch.object(external, "validate_run", return_value=self.spec), \
                patch.object(external, "load_engine", return_value=engine or self.engine()), \
                patch.object(external, "hardware", return_value=self.info), \
                patch.object(external.subprocess, "run", side_effect=probe or self.fake_probe) as child:
            external.train(self.run_id)
        return child

    def test_only_gemma_sft_runs_and_adapter_metadata_is_portable(self):
        child = self.run_training()
        self.assertEqual(child.call_count, 2)
        for call in child.call_args_list:
            command = call.args[0]
            self.assertIn("_probe", command)
            self.assertNotIn("evaluate_dolly_sft.py", " ".join(command))
        record = external.read(self.out / f"{external.MODEL}_lr1e-04.json")
        self.assertFalse(Path(record["best_checkpoint"]).is_absolute())
        config = external.read(self.root / record["adapter"] / "adapter_config.json")
        self.assertEqual(config["base_model_name_or_path"], "google/gemma-2-9b-it")
        self.assertEqual(config["revision"], "pinned")
        status = external.read(self.out / "worker_status.json")
        self.assertEqual(status["status"], "completed")
        self.assertFalse(status["evaluated"])
        self.assertFalse((self.root / "results/dolly_sft_v1").exists())

    def test_resume_reuses_probes_only_on_the_same_device(self):
        self.run_training()
        same_device = self.run_training()
        self.assertEqual(same_device.call_count, 0)
        self.info["gpu"] = "A different BF16 GPU"
        other_device = self.run_training()
        self.assertEqual(other_device.call_count, 2)

    def test_failed_probe_never_starts_training_or_falls_back(self):
        calls = []
        engine = self.engine()
        engine.train = lambda names: calls.append(names)
        with self.assertRaisesRegex(RuntimeError, "No quantization fallback"):
            self.run_training(engine=engine, probe=lambda *a, **kw: SimpleNamespace(returncode=20))
        self.assertEqual(calls, [])
        self.assertEqual(external.read(self.out / "worker_status.json")["status"], "failed")
        self.assertFalse((self.out / "memory_check.json").exists())

    def test_cached_probes_must_also_fit_current_free_vram(self):
        self.run_training()
        self.info["free_memory_bytes"] = 12 * 2**30
        with self.assertRaisesRegex(RuntimeError, "dedicated VRAM budget"):
            self.run_training()
        self.assertFalse(external.read(self.out / "memory_check.json")["fits_dedicated_budget"])

    def test_dedicated_budget_uses_child_probe_free_memory(self):
        def over_budget(command, **kwargs):
            result = self.fake_probe(command, **kwargs)
            name = "preflight.json" if command[-1] == "manual" else "trainer_preflight.json"
            report = external.read(self.out / name)
            report["models"][external.MODEL]["available_before_probe_bytes"] = 19 * 2**30
            self.write(self.out / name, report)
            return result
        with self.assertRaisesRegex(RuntimeError, "dedicated VRAM budget"):
            self.run_training(probe=over_budget)
        self.assertFalse(external.read(self.out / "memory_check.json")["fits_dedicated_budget"])

    def test_run_id_cannot_escape_external_namespace(self):
        for value in ("../other", "a/b", "a\\b", "", "a.b", "a" * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                external.paths(value)

    def test_tokenized_bytes_preserve_the_reference_on_either_platform(self):
        records = [{"id": "raw:1", "labels": [-100, 107]}]
        lf = (json.dumps(records[0]) + "\n").encode()
        crlf = lf.replace(b"\n", b"\r\n")
        self.assertEqual(external.encoded_bytes(records, external.digest(lf)), lf)
        self.assertEqual(external.encoded_bytes(records, external.digest(crlf)), crlf)
        with self.assertRaisesRegex(RuntimeError, "tokenization differs"):
            external.encoded_bytes(records, "wrong")

    def test_source_hash_normalizes_windows_checkout_newlines(self):
        path = self.root / "source.py"
        path.write_bytes(b"a=1\r\nb=2\r\n")
        self.assertEqual(external.source_sha(path), external.digest(b"a=1\nb=2\n"))

    def test_help_does_not_import_training_libraries(self):
        script = Path(external.__file__)
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=True)
        self.assertIn("pack-inputs", result.stdout)
        self.assertIn("train", result.stdout)

    def test_prepare_fetches_only_pinned_gemma_and_preserves_shared_rows(self):
        self.root.joinpath("scripts").mkdir()
        for relative in external.SOURCE_FILES.values():
            (self.root / relative).write_text("# test source\n", encoding="utf-8")
        parent = {"fingerprint": "parent", "source_script_sha256": external.source_sha(self.root / "scripts/dolly_sft.py"),
                  "counts": {split: 1 for split in external.SPLITS}}
        self.write(self.root / "results/dolly_sft_v1/protocol.json", parent)
        entry = {"id": "google/gemma-2-9b-it", "revision": "pinned", "weight_files": ["model-1.safetensors"]}
        self.write(self.root / "results/dolly_sft_additional_v1/registry.json", {external.MODEL: entry})
        shared, encoded_hashes = {}, {}
        for index, split in enumerate(external.SPLITS):
            row = {"id": f"raw:{index}", "instruction": "full prompt", "input": "context", "output": "full response"}
            shared[split] = (json.dumps(row) + "\n").encode()
            encoded = {"id": row["id"], "input_ids": [1, 2, 107], "labels": [-100, 2, 107],
                       "attention_mask": [1, 1, 1], "prompt_tokens": 1}
            encoded_hashes[f"tmp/dolly_sft_additional_v1/{external.MODEL}/{external.MODEL}/{split}.jsonl"] = external.digest(
                (json.dumps(encoded) + "\n").encode())
        reference = {"revision": "pinned", "parent_protocol_fingerprint": "parent", "prepared_files_sha256": encoded_hashes}
        self.write(self.root / f"results/dolly_sft_additional_v1/{external.MODEL}/preparation.json", reference)
        snapshot = self.root / "snapshot"
        fetched = []
        hub = SimpleNamespace(get_token=lambda: "local-auth", snapshot_download=lambda model, **kw: fetched.append((model, kw)) or str(snapshot))
        tokenizer = SimpleNamespace(model_max_length=8192)
        engine = SimpleNamespace(tokenizer_for=lambda name: tokenizer,
            encode=lambda name, tok, row: {"input_ids": [1, 2, 107], "labels": [-100, 2, 107],
                                          "attention_mask": [1, 1, 1], "prompt_tokens": 1})
        with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                patch("dolly_handoff.read_inputs", return_value=shared), \
                patch.object(external, "load_engine", return_value=engine), \
                patch.object(external, "versions", return_value={"torch": "test"}):
            external.prepare(self.run_id, self.root / "inputs.zip")
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0][0], entry["id"])
        self.assertEqual(fetched[0][1]["revision"], "pinned")
        for split in external.SPLITS:
            self.assertEqual((self.local / "source" / f"{split}.jsonl").read_bytes(), shared[split])
        spec = external.read(self.out / "protocol.json")
        self.assertEqual(spec["precision"][external.MODEL]["fallback"], "forbidden")
        self.assertEqual(spec["fingerprint"], external.fingerprint(spec))
        self.assertNotIn("local-auth", json.dumps(spec))
        self.assertEqual(external.read(self.root / "results/dolly_sft_v1/protocol.json"), parent)

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "filelock is required for lock integration")
    def test_shared_worker_lock_blocks_external_gpu_work(self):
        from filelock import FileLock, Timeout
        shared = self.root / "tmp/dolly_sft_v1/worker.lock"
        shared.parent.mkdir(parents=True)
        with FileLock(str(shared)), self.assertRaises(Timeout):
            with external.run_lock(self.run_id, gpu=True):
                self.fail("Concurrent GPU worker acquired the shared lock")


if __name__ == "__main__":
    unittest.main()
