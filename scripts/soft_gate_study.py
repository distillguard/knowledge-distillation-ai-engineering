"""Prepare the isolated GPT-2 gate study and evaluate matched original baselines."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch
from filelock import FileLock
from transformers import AutoModelForCausalLM, set_seed

import dolly_sft as base

ROOT = base.ROOT
OUT = ROOT / "results/soft_gate_gradient_v1"
LOCAL = ROOT / "tmp/soft_gate_gradient_v1"
NAMES = ("gpt2_base", "gpt2_xl")
SEEDS = (10, 20, 30, 40, 50)
COMMIT = "6f53ef41c951de0962bbf27fe3f5819eabc8ce0e"


def encode(tokenizer, row):
    prompt = base.render_prompt("gpt2_base", tokenizer, row["instruction"], row["input"])
    source = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + row["output"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(source) > 256:
        return None
    retained = full[:512]
    labels = [-100] * len(source) + retained[len(source):]
    assert len(labels) == len(retained) and any(t != -100 for t in labels[1:])
    return {"input_ids": retained, "labels": labels, "prompt_ids": source,
            "full_tokens": len(full), "truncated": len(full) > 512,
            "prompt_boundary_changed": full[:len(source)] != source}


def soft_loss(teacher, student, labels, detach_gate=False):
    mask = labels[:, 1:] != -100
    assert mask.any() and teacher.shape == student.shape
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    lp = teacher[:, :-1].detach().to(dtype).log_softmax(-1)
    lq = student[:, :-1].to(dtype).log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    forward = (p * (lp - lq)).sum(-1)
    reverse = (q * (lq - lp)).sum(-1)
    confidence_gap = ((p * lp).sum(-1) - (q * lq).sum(-1)) / math.log(lp.shape[-1])
    gate = (confidence_gap - 0.05).sigmoid()
    weight = gate.detach() if detach_gate else gate
    loss = (weight * forward + (1 - weight) * reverse)[mask].mean()
    selection = ((forward - reverse).detach() * gate)[mask].mean()
    return loss, selection


def gradient_audit():
    generator = torch.Generator().manual_seed(10)
    teacher = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    student = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 2, 3, -100, -100]])
    allow, selection = soft_loss(teacher, student, labels)
    detach, _ = soft_loss(teacher, student, labels, True)
    whole = torch.autograd.grad(allow, student, retain_graph=True)[0]
    learning = torch.autograd.grad(detach, student, retain_graph=True)[0]
    additional = torch.autograd.grad(selection, student)[0]
    assert torch.equal(allow.detach(), detach.detach())
    assert torch.allclose(whole - learning, additional, atol=1e-12, rtol=1e-10)
    assert additional.norm() > 0 and teacher.grad is None
    return {"status": "passed", "scope": "synthetic CPU logits; not actual-model training",
            "same_forward_loss": True, "max_decomposition_absolute_error": float((whole-learning-additional).abs().max()),
            "learning_gradient_norm": float(learning.norm()), "selection_gradient_norm": float(additional.norm()),
            "teacher_gradient": None, "margin": 0.05}


def prepare():
    sources = json.loads((OUT / "official_sources.json").read_text(encoding="utf-8"))
    assert sources["commit"] == COMMIT
    source_root = ROOT / ".sources" / ("minillm-" + COMMIT) / "minillm"
    for name, item in sources["files"].items():
        assert base.sha(source_root / name) == item["sha256"], name
    tokenizers = {name: base.tokenizer_for(name) for name in NAMES}
    assert tokenizers[NAMES[0]].get_vocab() == tokenizers[NAMES[1]].get_vocab()
    source = base.HUB / "datasets--MiniLLM--dolly/snapshots" / base.DATA_REVISION
    raw, test = base.rows(source / "raw.jsonl"), base.rows(source / "valid.jsonl")
    assert len(raw) == 12435 and len(test) == 500
    prepared, audit = {}, {}
    for split, records in (("validation", list(enumerate(raw[:1000]))),
                           ("train", list(enumerate(raw[1000:], 1000))),
                           ("evaluation", list(enumerate(test)))):
        kept, rejected = [], []
        for index, row in records:
            assert all(isinstance(row[key], str) for key in ("instruction", "input", "output"))
            values = [encode(tokenizers[name], row) for name in NAMES]
            assert values[0] == values[1]
            if values[0] is None:
                rejected.append(index)
            else:
                kept.append({"id": f"{'valid' if split == 'evaluation' else 'raw'}:{index}",
                             "instruction": row["instruction"], "input": row["input"],
                             "references": [row["output"]], **values[0]})
        assert kept
        path = LOCAL / "prepared" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
        if path.exists():
            assert path.read_text(encoding="utf-8") == text, "Prepared data changed"
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
        lengths = sorted(len(row["input_ids"]) for row in kept)
        audit[split] = {"input_rows": len(records), "retained_rows": len(kept), "prompt_over_256_source_rows": rejected,
                        "response_truncated_at_total_512": sum(row["truncated"] for row in kept),
                        "prompt_boundary_changed": sum(row["prompt_boundary_changed"] for row in kept),
                        "token_length_min_median_max": [min(lengths), statistics.median(lengths), max(lengths)],
                        "response_prediction_tokens": sum(sum(t != -100 for t in row["labels"][1:]) for row in kept),
                        "prepared_file": path.relative_to(ROOT).as_posix(), "sha256": base.sha(path)}
        prepared[split] = kept
    group = lambda row: " ".join(row["instruction"].lower().split())
    groups = {key: {group(row) for row in records} for key, records in prepared.items()}
    overlaps = {f"{a}_vs_{b}": len(groups[a] & groups[b])
                for a, b in (("train", "validation"), ("train", "evaluation"), ("validation", "evaluation"))}
    spec = {"status": "prepared", "scope": "data and original-baseline evaluation; SFT/KD execution not frozen",
            "official_sources_manifest_sha256": base.sha(OUT / "official_sources.json"), "official_code_commit": COMMIT,
            "producer_sha256": base.sha(__file__), "models": {n: dict(zip(("id", "revision"), base.MODELS[n])) for n in NAMES},
            "dataset": {"id": "MiniLLM/dolly", "revision": base.DATA_REVISION,
                        "raw_sha256": base.sha(source / "raw.jsonl"), "evaluation_sha256": base.sha(source / "valid.jsonl")},
            "preprocessing": "official concat tokenization, prompt<=256, total training sequence truncated to 512; stable raw ID order",
            "ordering_deviation": "official multiprocessing imap_unordered does not guarantee the original processed row order",
            "instruction_group_overlap_counts": overlaps, "overlap_policy": "reported, not silently removed from the reproduction split",
            "data": audit, "generation": {"seeds": list(SEEDS), "left_padded_prompt_length": 256, "max_length": 512,
            "temperature": 1.0, "top_k": 0, "top_p": 1.0, "no_repeat_ngram_size": 6, "batch_size": 8,
            "batch_basis": "official run_eval.sh passes eval-batch-size 8", "eos": 50256},
            "inference": {"dtype": "float16", "attention": "eager", "quantization": None},
            "nll": "response-token-weighted mean; include retained EOS only; no prompt or padding targets",
            "versions": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "tokenizers")}}
    import hashlib
    spec["fingerprint"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = OUT / "preparation.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == spec
    else:
        base.write_json(path, spec)
    base.write_json(OUT / "gradient_preflight.json", gradient_audit())
    print(json.dumps({"prepared": {s: a["retained_rows"] for s, a in audit.items()},
                      "overlaps": overlaps, "fingerprint": spec["fingerprint"]}), flush=True)


def status(stage, **fields):
    base.write_json(OUT / "worker_status.json", {"pid": os.getpid(), "stage": stage,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields})


def nll(model, records):
    total, count = 0., 0
    with torch.inference_mode():
        for row in records:
            ids = torch.tensor([row["input_ids"]], device="cuda")
            labels = torch.tensor([row["labels"][1:]], device="cuda")
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, :-1].float()
            total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"))
            count += int((labels != -100).sum())
    return {"response_token_nll": total / count, "response_token_ppl": math.exp(total/count), "target_tokens": count}


def generate(model, tokenizer, records, folder, spec, seed):
    path = folder / f"seed{seed}_responses.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"fingerprint": spec["fingerprint"], "seed": seed, "responses": []}
    assert result["fingerprint"] == spec["fingerprint"] and result["seed"] == seed
    outputs = result["responses"]
    assert [r["id"] for r in outputs] == [r["id"] for r in records[:len(outputs)]]
    set_seed(seed)
    if outputs:
        torch.set_rng_state(torch.tensor(list(bytes.fromhex(result["cpu_rng"])), dtype=torch.uint8))
        torch.cuda.set_rng_state(torch.tensor(list(bytes.fromhex(result["cuda_rng"])), dtype=torch.uint8))
    for start in range(len(outputs), len(records), spec["generation"]["batch_size"]):
        batch = records[start:start + spec["generation"]["batch_size"]]
        values = torch.full((len(batch), 256), tokenizer.eos_token_id, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(values)
        for i, row in enumerate(batch):
            prompt = row["prompt_ids"]
            values[i, -len(prompt):] = torch.tensor(prompt, device="cuda")
            mask[i, -len(prompt):] = 1
        with torch.inference_mode():
            response_ids = model.generate(input_ids=values, attention_mask=mask, max_length=512, do_sample=True,
                temperature=1.0, top_k=0, top_p=1.0, no_repeat_ngram_size=6, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)[:, 256:]
        for row, sequence in zip(batch, response_ids.tolist()):
            ended = tokenizer.eos_token_id in sequence
            tokens = sequence[:sequence.index(tokenizer.eos_token_id) + 1] if ended else sequence
            outputs.append({"id": row["id"], "response": tokenizer.decode(tokens, skip_special_tokens=True),
                            "generated_tokens": len(tokens), "length_capped": not ended and len(tokens) == 256})
        result.update(cpu_rng=bytes(torch.get_rng_state().tolist()).hex(), cuda_rng=bytes(torch.cuda.get_rng_state().tolist()).hex())
        base.write_json(path, result)
        status("original_generation", model=folder.name, seed=seed, completed_items=len(outputs), total_items=len(records))
    return result


def baseline(wait):
    spec = json.loads((OUT / "preparation.json").read_text(encoding="utf-8"))
    assert spec["producer_sha256"] == base.sha(__file__)
    for item in spec["data"].values():
        assert base.sha(ROOT / item["prepared_file"]) == item["sha256"]
    LOCAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCAL / "baseline.lock"), timeout=0):
        status("waiting_for_shared_gpu_lock")
        with FileLock(str(base.LOCAL / "worker.lock"), timeout=-1 if wait else 0):
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            records = base.rows(LOCAL / "prepared/evaluation.jsonl")
            validation = base.rows(LOCAL / "prepared/validation.jsonl")
            for name in NAMES:
                destination = OUT / "original" / name
                if (destination / "completion.json").exists():
                    done = json.loads((destination / "completion.json").read_text(encoding="utf-8"))
                    assert done["fingerprint"] == spec["fingerprint"]
                    continue
                status("original_loading", model=name)
                tokenizer = base.tokenizer_for(name)
                model = AutoModelForCausalLM.from_pretrained(base.snapshot(name), local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, dtype=torch.float16,
                    attn_implementation="eager", device_map={"": 0})
                model.eval().requires_grad_(False)
                folder = LOCAL / "original" / name
                try:
                    scores_path = destination / "nll.json"
                    if not scores_path.exists():
                        status("original_nll", model=name)
                        base.write_json(scores_path, {"fingerprint": spec["fingerprint"],
                            "validation": nll(model, validation), "evaluation": nll(model, records)})
                    for seed in SEEDS:
                        generate(model, tokenizer, records, folder, spec, seed)
                        print("COMPLETED_ORIGINAL_DOLLY_SEED", name, seed, flush=True)
                    base.write_json(destination / "completion.json", {"status": "generation_completed_pending_scoring",
                        "fingerprint": spec["fingerprint"], "model": name, "seeds": list(SEEDS),
                        "raw_responses": [{"path": (folder / f"seed{s}_responses.json").relative_to(ROOT).as_posix(),
                                           "sha256": base.sha(folder / f"seed{s}_responses.json")} for s in SEEDS]})
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            status("original_dolly_generation_completed_pending_scoring", training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "baseline"))
    parser.add_argument("--wait-for-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        prepare()
    else:
        try:
            baseline(args.wait_for_gpu)
        except Exception as error:
            status("failed", error_type=type(error).__name__, error=str(error), training_started=False)
            raise


if __name__ == "__main__":
    main()
