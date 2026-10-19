"""Disposable joint KD engineering probe for the two shortlisted BF16 pairs."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
import math
from time import perf_counter

from filelock import FileLock
import dolly_study as study
from dolly_sft import ROOT, LOCAL, rows, sha, write_json
import torch

PAIRS = {"gpt2": ("gpt2_xl", "gpt2_base"), "opt": ("opt_27b", "opt_125m")}
OUT = ROOT / "results/kd_feasibility_v1"


def soft_loss(teacher_logits, student_logits, labels):
    mask = labels[:, 1:] != -100
    assert mask.any() and teacher_logits.shape == student_logits.shape
    lp = teacher_logits[:, :-1].detach().float().log_softmax(-1)
    lq = student_logits[:, :-1].float().log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    forward = (p * (lp - lq)).sum(-1)
    reverse = (q * (lq - lp)).sum(-1)
    cp = 1 + (p * lp).sum(-1) / math.log(lp.shape[-1])
    cq = 1 + (q * lq).sum(-1) / math.log(lq.shape[-1])
    gate = (cp - cq).sigmoid()
    loss = (gate * forward + (1 - gate) * reverse)[mask].mean()
    assert torch.isfinite(loss)
    return loss, {"forward_kl": float(forward[mask].detach().mean()),
                  "reverse_kl": float(reverse[mask].detach().mean()),
                  "mean_soft_gate": float(gate[mask].detach().mean()), "response_tokens": int(mask.sum())}


@contextmanager
def mc_dropout(model):
    assert model.config.model_type in ("gpt2", "opt") and not getattr(model.config, "layerdrop", 0)
    flags = [(module, module.training) for module in model.modules()]
    changes = []
    try:
        model.train()
        for module in model.modules():
            if type(module).__name__ == "GPT2Attention":
                changes.append((module.attn_dropout, "p", module.attn_dropout.p))
                module.attn_dropout.p = .1
            elif type(module).__name__ == "OPTAttention":
                changes.append((module, "dropout", module.dropout))
                module.dropout = .1
        assert len(changes) == model.config.num_hidden_layers, "Attention dropout coverage differs"
        yield len(changes)
    finally:
        for module, attribute, value in changes:
            setattr(module, attribute, value)
        for module, training in flags:
            module.training = training


def bald(model, inputs, labels):
    probabilities_sum = entropy_sum = None
    first = None
    changed_entries = 0
    mask = labels[:, 1:] != -100
    assert mask.any()
    maxima = []
    with torch.no_grad():
        for _ in range(3):
            lp = model(**inputs, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            probabilities = lp.exp()
            if first is None:
                first = probabilities
            else:
                changed_entries += int((probabilities != first)[mask].sum())
            entropy = -(probabilities * lp).sum(-1)
            probabilities_sum = probabilities.clone() if probabilities_sum is None else probabilities_sum + probabilities
            entropy_sum = entropy if entropy_sum is None else entropy_sum + entropy
            maxima.append(probabilities.max(-1).values)
        information = torch.special.entr(probabilities_sum / 3).sum(-1) - entropy_sum / 3
        assert torch.isfinite(information).all()
        top1 = torch.stack(maxima)
        return {"mean_bald": float(information[mask].mean()), "minimum_token_bald": float(information[mask].min()),
                "mean_top1_probability_range": float((top1.max(0).values - top1.min(0).values)[mask].mean()),
                "response_tokens": int(mask.sum()), "passes": 3,
                "probability_entries_differing_from_first_pass": changed_entries}


def train_rows(name):
    engine = study if name.startswith("gpt2_") else study.engine_for(name)
    path = engine.LOCAL / name / "train.jsonl"
    return path, rows(path)


def update(teacher, student, optimizer, row, clip):
    student.train()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    values = {key: torch.tensor([row[key]], device="cuda") for key in ("input_ids", "attention_mask", "labels")}
    inputs = {key: values[key] for key in ("input_ids", "attention_mask")}
    parameter = next(p for name, p in student.named_parameters() if "lora_B" in name and p.requires_grad)
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    started = perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher_logits = teacher(**inputs, use_cache=False).logits
    with torch.autocast("cuda", dtype=torch.bfloat16):
        student_logits = student(**inputs, use_cache=False).logits
    assert torch.isfinite(teacher_logits).all() and torch.isfinite(student_logits).all()
    loss, result = soft_loss(teacher_logits, student_logits, values["labels"])
    loss.backward()
    gradients = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(p.grad is None for p in teacher.parameters())
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], clip)
    optimizer.step()
    assert not torch.equal(before, parameter), "Disposable student LoRA did not update"
    torch.cuda.synchronize()
    result.update(loss=float(loss.detach()), seconds=perf_counter() - started, student_updated=True,
                  input_tokens=len(row["input_ids"]), source_id=row["id"])
    optimizer.zero_grad(set_to_none=True)
    return result


def rollout(student, tokenizer, name, row):
    student.eval()
    student.gradient_checkpointing_disable()
    prompt = row["input_ids"][:row["prompt_tokens"]]
    inputs = torch.tensor([prompt], device="cuda")
    torch.manual_seed(10); torch.cuda.manual_seed_all(10)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = student.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=True,
                                  max_new_tokens=256, min_new_tokens=256, do_sample=True, temperature=1,
                                  top_k=0, top_p=1, no_repeat_ngram_size=6,
                                  eos_token_id=study.generation_eos_ids(name, tokenizer), pad_token_id=tokenizer.eos_token_id)
    assert output.shape[1] == len(prompt) + 256
    sequence = output[0].cpu().tolist()
    return {"id": row["id"], "input_ids": sequence, "attention_mask": [1] * len(sequence),
            "labels": [-100] * len(prompt) + sequence[len(prompt):], "prompt_tokens": len(prompt)}


def run(pair):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    teacher_name, student_name = PAIRS[pair]
    shortlist = study.read(ROOT / "results/model_selection_v1/shortlist.json")["shortlisted_pairs"]
    assert any((r["teacher"], r["student"]) == PAIRS[pair] for r in shortlist)
    protocols = {name: study.model_protocol(name) for name in PAIRS[pair]}
    selected = {name: study.selected_record(name) for name in PAIRS[pair]}
    assert all(value["status"] == "selected" and value["protocol_fingerprint"] == protocols[name]["fingerprint"] for name, value in selected.items())
    versions = {key: version(key) for p in protocols.values() for key in p["versions"]}
    assert all(all(versions[key] == value for key, value in p["versions"].items()) for p in protocols.values())
    adapters = {name: ROOT / value["adapter"] / "adapter_model.safetensors" for name, value in selected.items()}
    hashes = {name: sha(path) for name, path in adapters.items()}
    assert all(p["precision"][name]["base_weights"] == "BF16, no weight quantization" for name, p in protocols.items())
    prepared = {name: train_rows(name) for name in PAIRS[pair]}
    teacher_rows, student_rows = (prepared[name][1] for name in PAIRS[pair])
    assert len(teacher_rows) == len(student_rows) == 11117
    assert all(all(a[key] == b[key] for key in ("id", "input_ids", "labels", "prompt_tokens")) for a, b in zip(teacher_rows, student_rows))
    longest = max(student_rows, key=lambda r: len(r["input_ids"]))
    prompt_longest = max(student_rows, key=lambda r: r["prompt_tokens"])
    tokenizers = {name: study.tokenizer_for(name) for name in PAIRS[pair]}
    assert tokenizers[teacher_name].get_vocab() == tokenizers[student_name].get_vocab()
    assert all(getattr(tokenizers[teacher_name], key) == getattr(tokenizers[student_name], key) for key in ("bos_token_id", "eos_token_id", "pad_token_id"))
    assert prompt_longest["prompt_tokens"] + 256 <= min(t.model_max_length for t in tokenizers.values())
    device = torch.cuda.get_device_properties(0)
    spec = {"pair": pair, "source_script_sha256": sha(__file__), "models": selected, "adapter_sha256": hashes, "versions": versions,
            "training_protocol_fingerprints": {name: p["fingerprint"] for name, p in protocols.items()},
            "prepared_training_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path, _ in prepared.values()},
            "gpu": {"name": device.name, "total_memory_bytes": device.total_memory},
            "cases": {"reference": {"id": longest["id"], "tokens": len(longest["input_ids"])},
                      "rollout": {"id": prompt_longest["id"], "prompt_tokens": prompt_longest["prompt_tokens"]}},
            "loss": {"type": "differentiable tokenwise full-vocabulary soft-gated FKL/RKL", "temperature": 1, "margin": 0,
                     "basis": "untransformed distributions and zero gate offset for an engineering probe; not main-study optimum selection"},
            "generation": {**study.read(ROOT / "results/generation_benchmarks_v1" / student_name / "protocol.json")["generation"],
                           "min_new_tokens": 256, "use_cache": True, "eos_token_ids": study.generation_eos_ids(student_name, tokenizers[student_name]),
                           "pad_token_id": tokenizers[student_name].eos_token_id,
                           "basis": "existing sampling settings; force the existing maximum budget for cache stress, not a benchmark response"},
            "mc": {"passes": 3, "attention_dropout": .1, "other_dropout_rates_unchanged": True,
                   "distribution_temperature": 1, "temperature_basis": "identity transform for an engineering probe; main-study value unselected",
                   "pooling": "response positions; streaming full-vocabulary sums; engineering scope",
                   "source": "https://arxiv.org/html/2609.36734v1#S8.SS2"},
            "microbatch": 1, "compute": "BF16 autocast; FP32 KL and trainable adapters", "tf32": False,
            "memory_budget_basis": "peak CUDA reserved bytes must fit CUDA free bytes before loading; no CPU offload; two engineering cases only",
            "optimizer": protocols[student_name]["optimizer"], "learning_rate": selected[student_name]["learning_rate"],
            "warmup_steps": 0, "optimizer_updates": 2, "formal_kd_experiment": False, "checkpoint_saved": False,
            "limitations": "two engineering cases only; no full-data nondegeneracy claim, rejection quantiles, Revival filtering or long-run convergence test"}
    spec["fingerprint"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    folder = OUT / pair
    if (folder / "protocol.json").exists():
        assert study.read(folder / "protocol.json") == spec, "Preserve the existing probe identity"
    if (folder / "result.json").exists():
        cached = study.read(folder / "result.json")
        assert cached["fingerprint"] == spec["fingerprint"]
        if cached["status"] == "passed":
            print(f"Verified existing engineering probe: {pair}; not a fresh hardware measurement", flush=True)
            return
    write_json(folder / "protocol.json", spec)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    available, _ = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    teacher = student = optimizer = model = parameter = inputs = labels = None
    result = {"fingerprint": spec["fingerprint"], "formal_kd_experiment": False, "checkpoint_saved": False,
              "available_before_bytes": available, "steps": {}}
    stage = "load_teacher"
    started = perf_counter()
    try:
        teacher = study.load_model(teacher_name, adapters[teacher_name].parent).eval().requires_grad_(False)
        teacher.gradient_checkpointing_disable()
        stage = "load_student"
        student = study.load_model(student_name, adapters[student_name].parent)
        for name, parameter in student.named_parameters():
            parameter.requires_grad_("lora_" in name)
        assert all(p.dtype == torch.float32 for p in student.parameters() if p.requires_grad)
        assert not any(getattr(m, "is_loaded_in_4bit", False) for m in (teacher, student))
        assert teacher.config.vocab_size == student.config.vocab_size
        assert max(tokenizers[student_name].get_vocab().values()) < student.config.vocab_size
        assert all(p.device.type == "cuda" for model in (teacher, student) for p in model.parameters())
        cfg = protocols[student_name]["optimizer"]
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                     lr=selected[student_name]["learning_rate"], betas=tuple(cfg["betas"]), eps=cfg["eps"],
                                     weight_decay=cfg["weight_decay"], foreach=False)
        stage = "reference_update"
        result["steps"][stage] = update(teacher, student, optimizer, longest, cfg["max_grad_norm"])
        stage = "student_rollout"
        generated = rollout(student, tokenizers[student_name], student_name, prompt_longest)
        result["steps"][stage] = {"source_id": generated["id"], "prompt_tokens": generated["prompt_tokens"], "generated_tokens": 256, "forced_budget": True}
        stage = "rollout_update"
        result["steps"][stage] = update(teacher, student, optimizer, generated, cfg["max_grad_norm"])
        inputs = {key: torch.tensor([generated[key]], device="cuda") for key in ("input_ids", "attention_mask")}
        labels = torch.tensor([generated["labels"]], device="cuda")
        for role, model in (("teacher", teacher), ("student", student)):
            stage = role + "_mc"
            model.gradient_checkpointing_disable()
            with mc_dropout(model) as layers, torch.autocast("cuda", dtype=torch.bfloat16):
                result["steps"][stage] = {**bald(model, inputs, labels), "attention_layers_forced": layers,
                                        "source_id": generated["id"], "prefix": "forced_student_rollout"}
            assert result["steps"][stage]["mean_bald"] > 0, "MC signal is degenerate on this probe"
            assert result["steps"][stage]["probability_entries_differing_from_first_pass"] > 0, "Identical MC probabilities"
        torch.cuda.synchronize()
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed_cuda_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
                      failed_stage=stage, error_type=type(error).__name__)
        raise
    finally:
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      seconds=perf_counter() - started, fits_measured_dedicated_budget=torch.cuda.max_memory_reserved() <= available)
        result["source_adapters_unchanged"] = all(sha(path) == hashes[name] for name, path in adapters.items())
        if not result["source_adapters_unchanged"]:
            result["status"] = "failed_source_adapter_changed"
        if result.get("status") == "passed" and not result["fits_measured_dedicated_budget"]:
            result["status"] = "failed_dedicated_budget"
        write_json(folder / "result.json", result)
        teacher = student = optimizer = model = parameter = inputs = labels = None
        gc.collect(); torch.cuda.empty_cache()
    assert result["status"] == "passed", result["status"]
    print(json.dumps(result), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=PAIRS, required=True)
    args = parser.parse_args(argv)
    with FileLock(str(LOCAL / "worker.lock"), timeout=0):
        run(args.pair)


if __name__ == "__main__":
    main()
