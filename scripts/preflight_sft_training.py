"""One disposable SFT update per resource probe, never a trained baseline."""

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp/training_deps"))
import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from audit_dataset_candidates import prompt
from load_two_models import MODELS


def main():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    hub = ROOT / "tmp/hf_home/hub"
    tokenizer = AutoTokenizer.from_pretrained(hub / "models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e", local_files_only=True)
    tokenizer.model_max_length = sys.maxsize
    data_path = hub / "datasets--databricks--databricks-dolly-15k/snapshots/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl"
    examples = []
    for index, line in enumerate(data_path.open(encoding="utf-8")):
        row = json.loads(line)
        text = prompt(row["instruction"], row["context"])
        prefix = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.encode(text + row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
        examples.append((index, prefix, tokens))
    result = {"purpose": "disposable one-update resource calibration; no SFT scores or saved model",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "imported_local_source_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                                                for name in ("audit_dataset_candidates.py", "load_two_models.py")},
              "gpu": torch.cuda.get_device_name(0), "microbatch": 1,
              "sequence_length_basis": {"512": "MiniLLM published total context for GPT-2 evaluation/training configuration",
                                        "1024": "GPT-2 native context capacity"},
              "optimizer": {"name": "bitsandbytes.AdamW8bit", "lr": 1e-5, "betas": [0.9, 0.999],
                            "eps": 1e-8, "weight_decay": 0.01,
                            "lr_basis": "member of MiniLLM >1.3B SFT LR candidates; calibration only, not selected optimum",
                            "other_values": "library defaults, calibration only; not a final training protocol"},
              "gradient_checkpointing": True, "attention": "sdpa", "autocast": "bfloat16, cache disabled",
              "probe_order": "For each model/dtype, restart from pretrained weights, update once at 512, then once at 1024",
              "timing_limit": "Includes finite-value checks and optimizer initialization; not steady-state training throughput",
              "windows_memory_limit": "PyTorch allocator values do not separate WDDM dedicated and shared GPU memory",
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes")},
              "standard_fp32_adamw_minimum_bytes": {role: count * 16 for role, count in (("teacher", 1557611200), ("student", 124439808))},
              "probes": []}
    for role, model_id, revision in MODELS:
        # FP32 weights preserve master precision; BF16-only weights are a separate feasibility probe.
        for dtype in (torch.float32, torch.bfloat16):
            model = optimizer = output = loss = ids = labels = mask = None
            torch.cuda.empty_cache()
            path = hub / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
            failed = False
            for capacity in (512, 1024):
                probe = {"role": role, "model_id": model_id, "revision": revision,
                         "parameter_dtype": str(dtype), "sequence_capacity": capacity,
                         "result_is_trained_baseline": False}
                if failed:
                    probe["status"] = "not_run_after_shorter_probe_oom"
                    result["probes"].append(probe)
                    continue
                try:
                    if model is None:
                        torch.manual_seed(0)  # Repeatable engineering probe, not a chosen training seed.
                        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                                                   torch_dtype=dtype, low_cpu_mem_usage=True,
                                                                   attn_implementation="sdpa").to("cuda").train()
                        model.config.use_cache = False
                        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
                    index, prefix, tokens = max((e for e in examples if len(e[2]) <= capacity and len(e[2]) > len(e[1])), key=lambda e: len(e[2]))
                    ids = torch.tensor([tokens + [tokenizer.eos_token_id] * (capacity - len(tokens))], device="cuda")
                    mask = torch.arange(capacity, device="cuda")[None, :] < len(tokens)
                    labels = ids.clone()
                    labels[:, :len(prefix)] = -100
                    labels[~mask] = -100
                    assert (labels[:, 1:] != -100).any()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
                        loss = output.loss
                    assert torch.isfinite(loss), "Nonfinite loss"
                    loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    before = model.transformer.h[-1].mlp.c_proj.weight.detach().clone()
                    optimizer.step()
                    torch.cuda.synchronize()
                    changed = not torch.equal(before, model.transformer.h[-1].mlp.c_proj.weight)
                    assert changed and all(torch.isfinite(p).all() for p in model.parameters())
                    probe.update(status="passed", loss=float(loss.detach()), parameter_updated=changed,
                                 seconds=round(perf_counter() - started, 3), source_row_zero_based=index,
                                 actual_tokens=len(tokens), supervised_tokens=int((labels[:, 1:] != -100).sum()),
                                 peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2),
                                 peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 2))
                    optimizer.zero_grad(set_to_none=True)
                    del before
                except torch.cuda.OutOfMemoryError:
                    probe.update(status="cuda_out_of_memory", peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2),
                                 peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 2))
                    failed = True
                result["probes"].append(probe)
                print(json.dumps(probe), flush=True)
                output = loss = ids = labels = mask = None
            model = optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
    destination = ROOT / "results/dataset_candidates/training_preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
