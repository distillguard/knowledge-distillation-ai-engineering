"""Download the pinned base pair and verify concurrent CUDA inference, without training."""

import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "tmp" / "hf_home")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = (
    ("teacher", "openai-community/gpt2-xl", "15ea56dee5df4983c59b2538573817e1667135e2"),
    ("student", "openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
)
FILES = (
    "config.json", "model.safetensors", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
)
PROMPT = "Knowledge distillation transfers"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this preflight; CPU fallback would not verify GPU fit.")

    snapshots = {}
    for role, model_id, revision in MODELS:
        print(f"Preparing {role}: {model_id}@{revision}", flush=True)
        snapshots[role] = Path(snapshot_download(
            repo_id=model_id, revision=revision, allow_patterns=list(FILES),
            cache_dir=ROOT / "tmp" / "hf_home" / "hub", token=False,
        ))
        if not (snapshots[role] / "model.safetensors").is_file():
            raise RuntimeError(f"Missing safe weights for {model_id}")
        print(f"Assets ready: {role}", flush=True)

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    tokenizers, models, records = {}, {}, {}
    load_started = perf_counter()
    for role, model_id, revision in MODELS:
        print(f"Loading {role} on CUDA, float16, eager attention", flush=True)
        tokenizers[role] = AutoTokenizer.from_pretrained(
            snapshots[role], local_files_only=True, use_fast=True, trust_remote_code=False,
        )
        models[role] = AutoModelForCausalLM.from_pretrained(
            snapshots[role], local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map={"": "cuda:0"}, attn_implementation="eager", trust_remote_code=False,
        ).eval().requires_grad_(False)
        model = models[role]
        assert all(p.device.type == "cuda" for p in model.parameters())
        assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
        records[role] = {
            "model_id": model_id, "revision": revision, "checkpoint_stage": "base_pretrained",
            "unique_parameters": sum(p.numel() for p in model.parameters()),
            "layers": model.config.n_layer, "hidden_size": model.config.n_embd,
            "attention_heads": model.config.n_head, "vocab_size": model.config.vocab_size,
            "context_capacity": model.config.n_positions, "dtype": "float16", "device": "cuda:0",
        }

    teacher_vocab = tokenizers["teacher"].get_vocab()
    assert teacher_vocab == tokenizers["student"].get_vocab(), "Vocabulary IDs differ"
    tokenizer_data = [json.loads(tokenizers[role].backend_tokenizer.to_str()) for role in ("teacher", "student")]
    assert tokenizer_data[0] == tokenizer_data[1], "Tokenizer behavior differs"
    vocab_hash = hashlib.sha256(json.dumps(teacher_vocab, sort_keys=True).encode()).hexdigest()
    encoded = {role: tokenizer(PROMPT, return_tensors="pt") for role, tokenizer in tokenizers.items()}
    assert torch.equal(encoded["teacher"]["input_ids"], encoded["student"]["input_ids"])

    logits = {}
    with torch.inference_mode():
        for role, model in models.items():
            output = model(**{name: value.to("cuda") for name, value in encoded[role].items()}, use_cache=False)
            last_logits = output.logits[0, -1].float()
            assert last_logits.shape == (len(teacher_vocab),) and torch.isfinite(last_logits).all()
            probabilities = last_logits.softmax(-1)
            assert torch.isclose(probabilities.sum(), torch.tensor(1.0, device="cuda"), atol=1e-6)
            logits[role] = last_logits.cpu().contiguous()
            print(f"Forward passed: {role}; parameters={records[role]['unique_parameters']:,}", flush=True)
            del output, last_logits, probabilities

    torch.cuda.synchronize()
    result = {
        "status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "two base models, same-input inference preflight only; not a KD experiment",
        "training_performed": False, "sft_performed": False, "mc_dropout_tested": False,
        "models_concurrently_resident": True, "models": records,
        "tokenizer_backend_equal": True, "vocabulary_sha256": vocab_hash,
        "probe": {"prompt": PROMPT, "input_ids": encoded["teacher"]["input_ids"][0].tolist(),
                  "sampling": False, "temperature_transform": None, "logits_finite": True,
                  "logits_file": "tmp/two_model_probe.safetensors"},
        "gpu": torch.cuda.get_device_name(0), "attention_implementation": "eager",
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
        "load_and_forward_seconds": round(perf_counter() - load_started, 2),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "huggingface_hub", "safetensors", "accelerate")},
        "python": platform.python_version(), "torch_cuda_build": torch.version.cuda,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["No Dolly preparation, training, gate, Revival or long-input test",
                        "Single-prompt inference memory does not establish training memory requirements",
                        "Models are released when this command exits; no persistent server is started"],
    }
    save_file(logits, ROOT / "tmp" / "two_model_probe.safetensors")
    (ROOT / "results").mkdir(exist_ok=True)
    output_path = ROOT / "results" / "model_preflight.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
