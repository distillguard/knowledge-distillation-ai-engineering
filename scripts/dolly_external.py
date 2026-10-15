"""Run the pinned Gemma-2-9B-IT Dolly LoRA SFT on a separate BF16 GPU."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemma2_9b_it"
SPLITS = ("train", "development", "selection_evaluation")
SOURCE_FILES = {
    "source_script_sha256": "scripts/dolly_external.py",
    "engine_script_sha256": "scripts/dolly_sft.py",
    "study_script_sha256": "scripts/dolly_study.py",
    "handoff_script_sha256": "scripts/dolly_handoff.py",
    "external_probe_script_sha256": "scripts/preflight_external_gemma.py",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_sha(path):
    return digest(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def fingerprint(spec):
    return digest(json.dumps({k: v for k, v in spec.items() if k != "fingerprint"}, sort_keys=True).encode())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def paths(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must be 1-64 letters, digits, underscores or hyphens, starting with a letter/digit")
    return ROOT / "tmp/external_nonquantized" / run_id, ROOT / "results/external_nonquantized" / run_id


def parent_protocol():
    parent = read(ROOT / "results/dolly_sft_v1/protocol.json")
    if source_sha(ROOT / "scripts/dolly_sft.py") != parent["source_script_sha256"]:
        raise RuntimeError("The training engine differs from the pinned parent experiment")
    return parent


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "sentencepiece", "protobuf")}


def validate_run(run_id, check_versions=True):
    local, out = paths(run_id)
    spec = read(out / "protocol.json")
    if spec.get("fingerprint") != fingerprint(spec):
        raise RuntimeError("External protocol fingerprint mismatch")
    if spec["run_id"] != run_id or spec["parent_protocol_fingerprint"] != parent_protocol()["fingerprint"]:
        raise RuntimeError("Run identity or parent experiment mismatch")
    entry = read(ROOT / "results/dolly_sft_additional_v1/registry.json")[MODEL]
    if spec["models"] != {MODEL: {key: entry[key] for key in ("id", "revision")}}:
        raise RuntimeError("Pinned Gemma model/revision mismatch")
    if spec["precision"][MODEL]["base_weights"] != "BF16, no weight quantization":
        raise RuntimeError("This runner only accepts nonquantized BF16 weights")
    for key, relative in SOURCE_FILES.items():
        if spec[key] != source_sha(ROOT / relative):
            raise RuntimeError(f"Source changed since preparation: {relative}; use a new run-id")
    for relative, expected in spec["prepared_files_sha256"].items():
        candidate = (ROOT / relative).resolve()
        if not candidate.is_relative_to(local.resolve()) or digest(candidate.read_bytes()) != expected:
            raise RuntimeError(f"Prepared data changed: {relative}")
    if check_versions and spec["versions"] != versions():
        raise RuntimeError("Library versions changed since preparation; resume in the original environment")
    return spec


@contextmanager
def run_lock(run_id, gpu=False):
    from filelock import FileLock
    local, _ = paths(run_id)
    local.mkdir(parents=True, exist_ok=True)
    with FileLock(str(local / "run.lock"), timeout=0):
        if gpu:
            shared = ROOT / "tmp/dolly_sft_v1/worker.lock"
            shared.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(shared), timeout=0):
                yield
        else:
            yield


def load_engine(run_id):
    """Reuse the unchanged training loop with one model and isolated paths."""
    module_spec = importlib.util.spec_from_file_location("_external_gemma_engine", ROOT / "scripts/dolly_sft.py")
    engine = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = engine
    module_spec.loader.exec_module(engine)
    local, out = paths(run_id)
    entry = read(ROOT / "results/dolly_sft_additional_v1/registry.json")[MODEL]
    snapshot_path = Path(read(local / "machine.json")["snapshot"])
    engine.MODELS = {MODEL: (entry["id"], entry["revision"])}
    engine.LOCAL, engine.OUT = local, out
    engine.protocol = lambda: validate_run(run_id)
    engine.snapshot = lambda name: snapshot_path

    def tokenizer_for(name):
        tokenizer = engine.AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True, trust_remote_code=False)
        tokenizer.model_max_length = read(snapshot_path / "config.json")["max_position_embeddings"]
        if "<end_of_turn>" not in tokenizer.get_vocab():
            raise RuntimeError("Gemma end_of_turn token is missing")
        tokenizer.eos_token = "<end_of_turn>"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def render_prompt(name, tokenizer, instruction, context=""):
        body = f"### Instruction:\n{instruction}" + (f"\n\n### Input:\n{context}" if context else "")
        return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)

    def load_model(name, adapter=None):
        engine.set_seed(0)
        model, loading = engine.AutoModelForCausalLM.from_pretrained(
            snapshot_path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            weights_only=True, dtype=engine.torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Incomplete model load: {loading}")
        if (getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
                or any(isinstance(module, (engine.bnb.nn.Linear4bit, engine.bnb.nn.Linear8bitLt)) for module in model.modules())):
            raise RuntimeError("Quantized layers are forbidden in this BF16 run")
        model.requires_grad_(False)
        model.config.use_cache = False
        for module in model.modules():
            if "norm" in type(module).__name__.lower():
                module.to(engine.torch.float32)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = (engine.PeftModel.from_pretrained(model, adapter, is_trainable=False) if adapter else
                 engine.get_peft_model(model, engine.LoraConfig(**engine.LORA)))
        if not adapter:
            trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
            if not trainable or any("lora_" not in name or parameter.dtype != engine.torch.float32 for name, parameter in trainable):
                raise RuntimeError("Only FP32 LoRA parameters may be trainable")
        return model

    engine.tokenizer_for, engine.render_prompt, engine.load_model = tokenizer_for, render_prompt, load_model
    parent = parent_protocol()
    if (engine.LORA != parent["lora"] or list(engine.LR_CANDIDATES) != parent["learning_rate_candidates"]
            or engine.EPOCHS != parent["epochs"] or engine.ACCUMULATION != parent["effective_batch_examples"]):
        raise RuntimeError("The engine no longer matches the common SFT settings")
    return engine


def encoded_bytes(records, expected):
    """Match the original manifest, including either platform's JSONL newlines."""
    data = "".join(json.dumps(record) + "\n" for record in records).encode()
    for candidate in (data, data.replace(b"\n", b"\r\n")):
        if digest(candidate) == expected:
            return candidate
    raise RuntimeError("Gemma tokenization differs from the original experiment; do not change/drop/truncate records")


def prepare(run_id, inputs, cache_dir=None):
    from dolly_handoff import read_inputs
    parent = parent_protocol()
    shared = read_inputs(inputs, parent)
    local, out = paths(run_id)
    if (out / "protocol.json").exists():
        validate_run(run_id)
        print(f"Already prepared: {run_id}", flush=True)
        return
    if any((local / "runs").glob("*/checkpoint-*")):
        raise RuntimeError("Training checkpoints exist without a valid protocol; inspect this run before proceeding")
    from huggingface_hub import get_token, snapshot_download
    token = get_token()
    fallback_token = Path.home() / ".cache/huggingface/token"
    if token is None and fallback_token.is_file():
        token = fallback_token.read_text(encoding="utf-8").strip()
    entry = read(ROOT / "results/dolly_sft_additional_v1/registry.json")[MODEL]
    cache = Path(cache_dir).resolve() if cache_dir else ROOT / "tmp/hf_home/hub"
    snapshot = snapshot_download(entry["id"], revision=entry["revision"], cache_dir=str(cache), token=token,
                                 allow_patterns=[*entry["weight_files"], "model.safetensors.index.json", "config.json",
                                                 "generation_config.json", "tokenizer*", "special_tokens_map.json"], max_workers=2)
    local.mkdir(parents=True, exist_ok=True)
    write_json(local / "machine.json", {"snapshot": str(Path(snapshot).resolve()), "cache_dir": str(cache)})
    engine = load_engine(run_id)
    tokenizer = engine.tokenizer_for(MODEL)
    reference = read(ROOT / f"results/dolly_sft_additional_v1/{MODEL}/preparation.json")
    if reference["revision"] != entry["revision"] or reference["parent_protocol_fingerprint"] != parent["fingerprint"]:
        raise RuntimeError("Gemma preparation manifest belongs to another experiment")
    prepared, lengths = {}, {}
    for split in SPLITS:
        records = [json.loads(line) for line in shared[split].decode("utf-8").splitlines()]
        encoded = [{"id": row["id"], **engine.encode(MODEL, tokenizer, row)} for row in records]
        limit = min(1024, tokenizer.model_max_length)
        if split != "selection_evaluation" and any(len(row["input_ids"]) > limit for row in encoded):
            raise RuntimeError("A complete shared training/development example exceeds the accepted context")
        expected = reference["prepared_files_sha256"][f"tmp/dolly_sft_additional_v1/{MODEL}/{MODEL}/{split}.jsonl"]
        outputs = {local / "source" / f"{split}.jsonl": shared[split],
                   local / MODEL / f"{split}.jsonl": encoded_bytes(encoded, expected)}
        for path, data in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            prepared[path.relative_to(ROOT).as_posix()] = digest(data)
        lengths[split] = {"n": len(encoded), "max_total_tokens": max(len(row["input_ids"]) for row in encoded),
                         "max_prompt_tokens": max(row["prompt_tokens"] for row in encoded)}
    actual_versions = versions()
    spec = {key: value for key, value in parent.items() if key not in ("fingerprint", "prepared_files_sha256")}
    spec.update(run_id=run_id, format_version=1, parent_protocol_fingerprint=parent["fingerprint"],
                models={MODEL: {key: entry[key] for key in ("id", "revision")}},
                **{key: source_sha(ROOT / relative) for key, relative in SOURCE_FILES.items()},
                source_hash_policy="UTF-8 source bytes with CRLF normalized to LF", versions=actual_versions,
                prepared_files_sha256=prepared, observed_lengths=lengths, max_total_training_tokens=1024,
                max_prompt_tokens=None, response_truncation=False,
                length_basis="Same original records and complete responses; Gemma manifest tokenization parity; train/development <=1024",
                precision={MODEL: {"base_weights": "BF16, no weight quantization", "compute": "BF16 autocast",
                                   "normalization": "FP32", "trainable_lora": "FP32", "fallback": "forbidden"}},
                prompt_policy="Same instruction/context; Gemma native chat template and end_of_turn target; complete responses",
                scope="Independent LoRA SFT and development-NLL selection only; no downstream evaluation")
    spec["fingerprint"] = fingerprint(spec)
    write_json(out / "preparation.json", {"status": "prepared", "model": MODEL, "revision": entry["revision"],
               "parent_protocol_fingerprint": parent["fingerprint"], "protocol_fingerprint": spec["fingerprint"],
               "same_original_records": True, "tokenization_matches_reference": True, "lengths": lengths,
               "prepared_files_sha256": prepared})
    write_json(out / "protocol.json", spec)
    print(f"Prepared {run_id}: BF16 only, {lengths}", flush=True)


def hardware(engine):
    torch = engine.torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This profile requires a single GPU process")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    free, total = torch.cuda.mem_get_info()
    return {"gpu": torch.cuda.get_device_name(0), "total_memory_bytes": total, "free_memory_bytes": free,
            "cuda": torch.version.cuda, "python": platform.python_version(), "platform": platform.platform(),
            "versions": versions(), "recorded_at_utc": datetime.now(timezone.utc).isoformat()}


def probe(run_id, kind):
    from preflight_external_gemma import run_manual, run_trainer
    engine = load_engine(run_id)
    hardware(engine)
    (run_manual if kind == "manual" else run_trainer)(engine)


def train(run_id):
    spec = validate_run(run_id)
    engine = load_engine(run_id)
    local, out = paths(run_id)
    info = hardware(engine)
    hardware_fingerprint = digest(json.dumps({key: info[key] for key in
        ("gpu", "total_memory_bytes", "cuda", "python", "platform", "versions")}, sort_keys=True).encode())
    write_json(out / "runtime.json", {**info, "protocol_fingerprint": spec["fingerprint"]})
    status = {"run_id": run_id, "model": MODEL, "protocol_fingerprint": spec["fingerprint"], "status": "running"}
    try:
        for kind, filename in (("manual", "preflight.json"), ("trainer", "trainer_preflight.json")):
            path = out / filename
            previous = read(path) if path.is_file() else {}
            if (previous.get("hardware_fingerprint") == hardware_fingerprint
                    and previous.get("protocol_fingerprint") == spec["fingerprint"]
                    and previous.get("models", {}).get(MODEL, {}).get("status") == "passed"):
                continue
            write_json(out / "worker_status.json", {**status, "stage": f"{kind}_preflight"})
            with (local / f"{kind}_probe.log").open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-u", "-B", "-X", "utf8", __file__, "_probe",
                                         "--run-id", run_id, "--kind", kind], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"BF16 {kind} preflight failed. No quantization fallback. Inspect {local / (kind + '_probe.log')}")
            report = read(path)
            report["hardware_fingerprint"] = hardware_fingerprint
            write_json(path, report)
        reports = [read(out / name)["models"][MODEL] for name in ("preflight.json", "trainer_preflight.json")]
        peak = max(report["peak_allocated_mib"] for report in reports) * 2**20
        fits = peak <= info["free_memory_bytes"] and all(
            report["peak_allocated_mib"] * 2**20 <= report["available_before_probe_bytes"] for report in reports)
        write_json(out / "memory_check.json", {"protocol_fingerprint": spec["fingerprint"],
                   "available_before_probe_bytes": info["free_memory_bytes"], "peak_allocated_bytes": peak,
                   "probe_budgets": [{"available_bytes": report["available_before_probe_bytes"],
                                       "peak_allocated_bytes": report["peak_allocated_mib"] * 2**20} for report in reports],
                   "fits_dedicated_budget": fits, "quantization_fallback": False})
        if not fits:
            raise RuntimeError("BF16 preflight exceeded the measured dedicated VRAM budget; experiment settings were not reduced")
        write_json(out / "worker_status.json", {**status, "stage": "training"})
        engine.train([MODEL])
        # Keep provenance portable; optimizer checkpoints remain local for epoch resume.
        for lr in engine.LR_CANDIDATES:
            path = out / f"{MODEL}_lr{lr:.0e}.json"
            record = read(path)
            checkpoint = Path(record["best_checkpoint"])
            if checkpoint.is_absolute():
                record["best_checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
                write_json(path, record)
            adapter_config = ROOT / record["adapter"] / "adapter_config.json"
            config = read(adapter_config)
            config["base_model_name_or_path"] = spec["models"][MODEL]["id"]
            config["revision"] = spec["models"][MODEL]["revision"]
            write_json(adapter_config, config)
        write_json(out / "worker_status.json", {**status, "status": "completed", "stage": "sft_selected", "evaluated": False})
    except Exception as error:
        write_json(out / "worker_status.json", {**status, "status": "failed", "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="{pack-inputs,prepare,train,export,import}")
    pack = commands.add_parser("pack-inputs", help="Package the existing common split on the original PC")
    pack.add_argument("--output", type=Path, required=True)
    prep = commands.add_parser("prepare", help="Import shared examples and download only pinned Gemma-9B")
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--inputs", type=Path, required=True)
    prep.add_argument("--cache-dir", type=Path)
    training = commands.add_parser("train", help="BF16 preflight, two LR candidates and selection; no evaluation")
    training.add_argument("--run-id", required=True)
    export = commands.add_parser("export", help="Package completed summaries and both best adapters")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    incoming = commands.add_parser("import", help="Validate and import into isolated external result paths")
    incoming.add_argument("--bundle", type=Path, required=True)
    internal = commands.add_parser("_probe")
    internal.add_argument("--run-id", required=True)
    internal.add_argument("--kind", choices=("manual", "trainer"), required=True)
    args = parser.parse_args(argv)
    from dolly_handoff import import_result_bundle, pack_inputs, write_result_bundle
    if args.command == "pack-inputs":
        print(pack_inputs(ROOT, args.output))
    elif args.command == "import":
        print(import_result_bundle(ROOT, args.bundle))
    elif args.command == "_probe":
        probe(args.run_id, args.kind)
    else:
        with run_lock(args.run_id, gpu=args.command == "train"):
            if args.command == "prepare":
                prepare(args.run_id, args.inputs, args.cache_dir)
            elif args.command == "train":
                train(args.run_id)
            else:
                validate_run(args.run_id, check_versions=False)
                print(write_result_bundle(ROOT, args.run_id, args.output))


if __name__ == "__main__":
    main()
