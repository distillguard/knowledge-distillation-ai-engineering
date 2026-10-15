"""Append paper model pairs without changing the fingerprinted four-model engine."""

import argparse
import gc
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "tmp/dolly_sft_additional_v1/deps"))
import dolly_sft as base

ROOT, OUT, LOCAL, HUB = base.ROOT, base.OUT, base.LOCAL, base.HUB
ADDED_OUT, ADDED_LOCAL = ROOT / "results/dolly_sft_additional_v1", ROOT / "tmp/dolly_sft_additional_v1"
REGISTRY = ADDED_OUT / "registry.json"
LABELS = {"gpt2_base": "GPT-2 base", "gpt2_xl": "GPT-2 XL", "qwen_08b": "Qwen3.5-0.8B", "qwen_9b": "Qwen3.5-9B",
          "opt_125m": "OPT-125M", "opt_27b": "OPT-2.7B", "openllama_3b": "OpenLLaMA-3B", "openllama_7b": "OpenLLaMA-7B",
          "gemma2_2b_it": "Gemma-2-2B-IT", "gemma2_9b_it": "Gemma-2-9B-IT"}
MODELS = tuple(LABELS)
ADDED = tuple(name for name in MODELS if name not in base.MODELS)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def result_dir(name):
    return OUT if name in base.MODELS else ADDED_OUT / name


def selected_record(name):
    return read(result_dir(name) / f"{name}_selected.json")


def snapshot(name):
    if name in base.MODELS:
        return base.snapshot(name)
    entry = read(REGISTRY)[name]
    return HUB / ("models--" + entry["id"].replace("/", "--")) / "snapshots" / entry["revision"]


def tokenizer_for(name):
    if name in base.MODELS:
        return base.tokenizer_for(name)
    tokenizer = base.AutoTokenizer.from_pretrained(snapshot(name), local_files_only=True,
                                                  trust_remote_code=False, use_fast=not name.startswith("openllama"))
    tokenizer.model_max_length = read(snapshot(name) / "config.json")["max_position_embeddings"]
    tokenizer.pad_token = tokenizer.eos_token
    if name.startswith("gemma"):
        assert "<end_of_turn>" in tokenizer.get_vocab()
        tokenizer.eos_token = "<end_of_turn>"
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def render_prompt(name, tokenizer, instruction, context=""):
    if name in base.MODELS:
        return base.render_prompt(name, tokenizer, instruction, context)
    if name.startswith("gemma"):
        body = f"### Instruction:\n{instruction}" + (f"\n\n### Input:\n{context}" if context else "")
        return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    prompt = base.render_prompt("gpt2_base", tokenizer, instruction, context)
    # OpenLLaMA's official card requires its slow tokenizer and recommends a BOS token.
    return tokenizer.bos_token + prompt if name.startswith("openllama") else prompt


def generation_eos_ids(name, tokenizer):
    path = snapshot(name) / "generation_config.json"
    native = read(path).get("eos_token_id", []) if path.exists() else []
    native = native if isinstance(native, list) else [native]
    return list(dict.fromkeys([tokenizer.eos_token_id] + [value for value in native if value is not None]))


def model_protocol(name):
    if name in base.MODELS:
        return base.protocol()
    spec = read(result_dir(name) / "protocol.json")
    assert spec["study_script_sha256"] == base.sha(__file__), "Additional-model source changed after preparation"
    assert spec["engine_script_sha256"] == base.sha(base.__file__)
    assert spec["parent_protocol_fingerprint"] == base.protocol()["fingerprint"]
    for relative, expected in spec["prepared_files_sha256"].items():
        assert base.sha(ROOT / relative) == expected, relative
    return spec


def load_model(name, adapter=None):
    if name in base.MODELS:
        return base.load_model(name, adapter)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    import torch
    import bitsandbytes as bnb
    spec = model_protocol(name)
    nf4 = spec["precision"][name]["base_weights"] == "NF4 with double quantization"
    quant = base.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16) if nf4 else None
    base.set_seed(0)
    model, loading = base.AutoModelForCausalLM.from_pretrained(snapshot(name), local_files_only=True,
        trust_remote_code=False, use_safetensors=any(snapshot(name).glob("*.safetensors")), weights_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", quantization_config=quant, output_loading_info=True)
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), loading
    modules = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert bool(modules) == nf4 and all(m.weight.quant_state.quant_type == "nf4" and m.weight.quant_state.nested for m in modules)
    model.config.use_cache = False
    if nf4:
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        model.get_input_embeddings().to(torch.bfloat16); model.get_output_embeddings().to(torch.bfloat16)
    else:
        model.requires_grad_(False)
        for module in model.modules():
            if "norm" in type(module).__name__.lower():
                module.to(torch.float32)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False) if adapter else get_peft_model(model, LoraConfig(**base.LORA))
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    return model


def engine_for(name):
    # Configure a separate module instance so the original source/checkpoint contract remains intact.
    module_spec = importlib.util.spec_from_file_location("_dolly_engine_" + name, base.__file__)
    engine = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = engine
    module_spec.loader.exec_module(engine)
    entry = read(REGISTRY)[name]
    engine.MODELS = {name: (entry["id"], entry["revision"])}
    engine.LOCAL, engine.OUT = ADDED_LOCAL / name, result_dir(name)
    engine.snapshot, engine.tokenizer_for, engine.render_prompt = snapshot, tokenizer_for, render_prompt
    engine.load_model = load_model
    engine.protocol = lambda: model_protocol(name)
    return engine


def fetch(names):
    from huggingface_hub import get_token, snapshot_download
    token = get_token()
    standard_token = Path.home() / ".cache/huggingface/token"
    if token is None and standard_token.is_file():
        token = standard_token.read_text(encoding="utf-8").strip()
    registry = read(REGISTRY)
    for name in names:
        entry = registry[name]
        base.write_json(result_dir(name) / "preparation.json", {"status": "downloading", "model": name})
        patterns = entry["weight_files"] + ["config.json", "generation_config.json", "tokenizer*", "vocab.json",
                                               "merges.txt", "special_tokens_map.json", "pytorch_model.bin.index.json"]
        print("FETCH", name, entry["id"], entry["revision"], flush=True)
        snapshot_download(entry["id"], revision=entry["revision"], cache_dir=str(HUB), allow_patterns=patterns,
                          token=token, max_workers=2)
        engine, tokenizer = engine_for(name), tokenizer_for(name)
        lengths, prepared = {}, {}
        for split in ("train", "development", "selection_evaluation"):
            source = LOCAL / f"{split}.jsonl"
            records = base.rows(source)
            encoded = [{"id": row["id"], **engine.encode(name, tokenizer, row)} for row in records]
            limit = min(1024, tokenizer.model_max_length)
            checked = [r for r in encoded if split != "selection_evaluation"]
            assert all(len(r["input_ids"]) <= limit for r in checked), "Full shared examples exceed common native context; do not silently truncate or drop rows"
            path = engine.LOCAL / name / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in encoded), encoding="utf-8")
            prepared[path.relative_to(ROOT).as_posix()] = base.sha(path)
            prepared[source.relative_to(ROOT).as_posix()] = base.sha(source)
            lengths[split] = {"n": len(encoded), "max_total_tokens": max(len(r["input_ids"]) for r in encoded),
                               "max_prompt_tokens": max(r["prompt_tokens"] for r in encoded)}
        base.write_json(result_dir(name) / "preparation.json", {"status": "prepared", "model": name,
            "revision": entry["revision"], "parent_protocol_fingerprint": base.protocol()["fingerprint"],
            "same_original_records": True, "lengths": lengths, "prepared_files_sha256": prepared})
        print("PREPARED", name, json.dumps(lengths), flush=True)
    for first, second in (("opt_125m", "opt_27b"), ("openllama_3b", "openllama_7b"), ("gemma2_2b_it", "gemma2_9b_it")):
        if (snapshot(first) / "tokenizer_config.json").exists() and (snapshot(second) / "tokenizer_config.json").exists():
            assert tokenizer_for(first).get_vocab() == tokenizer_for(second).get_vocab(), (first, second)


def prepare_protocol(name, precision, reason):
    import hashlib
    prepared = read(result_dir(name) / "preparation.json")
    assert prepared["status"] == "prepared"
    spec = base.protocol().copy()
    spec["versions"] = {**spec["versions"], **{package: version(package) for package in ("sentencepiece", "protobuf")}}
    parent = spec.pop("fingerprint")
    entry = read(REGISTRY)[name]
    spec.update(models={name: {"id": entry["id"], "revision": entry["revision"]}}, parent_protocol_fingerprint=parent,
                study_script_sha256=base.sha(__file__), engine_script_sha256=base.sha(base.__file__),
                source_script_sha256=base.sha(__file__), prepared_files_sha256=prepared["prepared_files_sha256"],
                max_total_training_tokens=1024, max_prompt_tokens=None,
                length_basis="Preserve all original four-model study record IDs and complete responses; 1024 is the original GPT-2 native context capacity. Added tokenizers may use more than 512 tokens for the same text.",
                observed_lengths=prepared["lengths"], precision={name: {"base_weights": "NF4 with double quantization" if precision == "nf4" else "BF16, no weight quantization",
                    "compute": "BF16 autocast", "normalization": "FP32", "trainable_lora": "FP32", "reason": reason}},
                prompt_policy="Same instruction/context; OPT MiniLLM wrapper, OpenLLaMA slow tokenizer plus BOS, Gemma native chat with end_of_turn target; complete responses")
    spec["fingerprint"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    destination = result_dir(name) / "protocol.json"
    if destination.exists() and read(destination)["fingerprint"] != spec["fingerprint"] and any((ADDED_LOCAL / name / "runs").glob("*/checkpoint-*")):
        raise RuntimeError("Cannot change an additional-model protocol after formal training checkpoints exist")
    base.write_json(destination, spec)


def probe(name):
    import torch
    try:
        available, _ = torch.cuda.mem_get_info()
        engine = engine_for(name)
        engine.preflight([name])
        sys.modules["dolly_sft"] = engine
        probe_spec = importlib.util.spec_from_file_location("_trainer_probe", ROOT / "scripts/preflight_dolly_trainer.py")
        module = importlib.util.module_from_spec(probe_spec); probe_spec.loader.exec_module(module)
        module.main()
        peak = max(read(engine.OUT / filename)["models"][name]["peak_allocated_mib"] for filename in ("preflight.json", "trainer_preflight.json")) * 2**20
        base.write_json(engine.OUT / "memory_check.json", {"protocol_fingerprint": model_protocol(name)["fingerprint"],
                                                           "available_before_probe_bytes": available, "peak_allocated_bytes": peak,
                                                           "fits_dedicated_budget": peak <= available})
        if peak > available:
            print("DEDICATED_VRAM_BUDGET_EXCEEDED", name, flush=True)
            sys.exit(20)
    except torch.cuda.OutOfMemoryError:
        print("CUDA_OUT_OF_MEMORY", name, flush=True)
        sys.exit(20)


def train(name):
    import torch
    if (result_dir(name) / f"{name}_selected.json").exists():
        assert selected_record(name)["protocol_fingerprint"] == model_protocol(name)["fingerprint"]
        return
    destination = result_dir(name) / "protocol.json"
    has_checkpoints = any((ADDED_LOCAL / name / "runs").glob("*/checkpoint-*"))
    if not has_checkpoints:
        physical = torch.cuda.get_device_properties(0).total_memory
        weight_bytes = read(REGISTRY)[name]["base_weight_bytes"]
        choices = ("nf4",) if weight_bytes >= physical else ("bf16", "nf4")
        for precision in choices:
            reason = "BF16 first; full load/update/checkpoint probe on otherwise idle GPU" if precision == "bf16" else "BF16 base weights exceed dedicated VRAM or BF16 load/update probe does not fit the measured dedicated-memory budget"
            prepare_protocol(name, precision, reason)
            checks = [result_dir(name) / filename for filename in ("preflight.json", "trainer_preflight.json", "memory_check.json")]
            fingerprint = model_protocol(name)["fingerprint"]
            if all(path.exists() and read(path).get("protocol_fingerprint") == fingerprint for path in checks):
                manual, trainer, memory = map(read, checks)
                if (manual["models"][name]["status"] == trainer["models"][name]["status"] == "passed"
                        and memory["fits_dedicated_budget"]):
                    break
            log_path = ADDED_LOCAL / f"{name}_{precision}_probe.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-u", "-B", "-X", "utf8", __file__, "probe", "--model", name],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            if result.returncode != 20:
                raise RuntimeError(f"Probe failed, not a CUDA OOM: {log_path}")
        else:
            raise RuntimeError(f"Both precisions failed: {name}")
    assert destination.exists()
    engine_for(name).train([name])
    gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "probe", "train"))
    parser.add_argument("--model", choices=(*ADDED, "all"), default="all")
    args = parser.parse_args()
    names = list(ADDED) if args.model == "all" else [args.model]
    if args.stage == "fetch":
        fetch(names)
    else:
        for name in names:
            (probe if args.stage == "probe" else train)(name)


if __name__ == "__main__":
    main()
