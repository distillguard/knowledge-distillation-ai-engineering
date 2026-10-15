"""Prepare, verify and train four Dolly LoRA candidates; quantize only Qwen-9B."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "tmp/dolly_sft_v1"
OUT = ROOT / "results/dolly_sft_v1"
HUB = ROOT / "tmp/hf_home/hub"
os.environ["HF_HOME"] = str(HUB.parent)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Qwen3_5ForCausalLM, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

MODELS = {
    "gpt2_base": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "gpt2_xl": ("openai-community/gpt2-xl", "15ea56dee5df4983c59b2538573817e1667135e2"),
    "qwen_08b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "qwen_9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
}
LR_CANDIDATES = (1e-4, 2e-4)
MAX_LENGTH, MAX_PROMPT, EPOCHS, ACCUMULATION = 512, 256, 3, 16
LORA = {"r": 64, "lora_alpha": 16, "lora_dropout": 0.05,
        "target_modules": "all-linear", "bias": "none", "task_type": "CAUSAL_LM"}
DATA_REVISION = "d131636138be0ff3fd0901c12c1739844d337279"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def snapshot(name):
    repo, revision = MODELS[name]
    cache = Path("C:/lightjudge/.cache/huggingface/hub") if name == "qwen_9b" else HUB
    path = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Pinned snapshot unavailable: {repo}@{revision}")
    return path


def tokenizer_for(name):
    tokenizer = AutoTokenizer.from_pretrained(snapshot(name), local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 1024 if name.startswith("gpt2") else 262144
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def render_prompt(name, tokenizer, instruction, context=""):
    body = f"### Instruction:\n{instruction}"
    if context:
        body += f"\n\n### Input:\n{context}"
    if name.startswith("qwen"):
        return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    introduction = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                    if context else "Below is an instruction that describes a task. ")
    return introduction + "Write a response that appropriately completes the request.\n\n" + body + "\n\n### Response:\n"


def encode(name, tokenizer, row):
    prompt = render_prompt(name, tokenizer, row["instruction"], row["input"])
    source = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(row["output"], add_special_tokens=False) + [tokenizer.eos_token_id]
    # Separate source/target tokenization follows QLoRA's causal-LM collator and prevents prompt-label leakage.
    return {"input_ids": source + target, "attention_mask": [1] * (len(source) + len(target)),
            "labels": [-100] * len(source) + target, "prompt_tokens": len(source)}


def prepare():
    source = HUB / "datasets--MiniLLM--dolly/snapshots" / DATA_REVISION
    raw, evaluation = rows(source / "raw.jsonl"), rows(source / "valid.jsonl")
    assert len(raw) == 12435 and len(evaluation) == 500
    tokenizers = {name: tokenizer_for(name) for name in MODELS}
    for first, second in (("gpt2_base", "gpt2_xl"), ("qwen_08b", "qwen_9b")):
        assert tokenizers[first].get_vocab() == tokenizers[second].get_vocab(), (first, second)
    group = lambda row: " ".join(row["instruction"].casefold().split())
    heldout_groups = {group(row) for row in evaluation}
    development_groups = {group(row) for row in raw[:1000]}
    selected = {"train": [], "development": [], "selection_evaluation": []}
    excluded = Counter()
    encoded = {name: {split: [] for split in selected} for name in MODELS}
    seen = set()
    for split, candidates in (("development", list(enumerate(raw[:1000]))),
                              ("train", list(enumerate(raw[1000:], 1000))),
                              ("selection_evaluation", list(enumerate(evaluation)))):
        for index, row in candidates:
            identifier = f"{'valid' if split == 'selection_evaluation' else 'raw'}:{index}"
            assert all(isinstance(row[field], str) for field in ("instruction", "input", "output"))
            reason = None
            if split != "selection_evaluation":
                if group(row) in heldout_groups:
                    reason = "instruction_overlaps_published_evaluation"
                elif split == "train" and group(row) in development_groups:
                    reason = "instruction_overlaps_development"
                elif not row["instruction"].strip() or not row["output"].strip():
                    reason = "empty_instruction_or_response"
                exact = (row["instruction"], row["input"], row["output"])
                if reason is None and exact in seen:
                    reason = "exact_duplicate"
                seen.add(exact)
            tokens = {name: encode(name, tok, row) for name, tok in tokenizers.items()}
            if reason is None and any(t["prompt_tokens"] > MAX_PROMPT for t in tokens.values()):
                reason = "common_prompt_over_256"
            if reason is None and split != "selection_evaluation" and any(len(t["input_ids"]) > MAX_LENGTH for t in tokens.values()):
                reason = "common_complete_example_over_512"
            if reason:
                excluded[f"{split}:{reason}"] += 1
                continue
            selected[split].append({"id": identifier, "instruction_group_sha256": hashlib.sha256(group(row).encode()).hexdigest(),
                                    "topic": row.get("topic"), "instruction": row["instruction"],
                                    "input": row["input"], "output": row["output"]})
            for name, value in tokens.items():
                assert len(value["labels"]) == len(value["input_ids"]) and any(x != -100 for x in value["labels"][1:])
                encoded[name][split].append({"id": identifier, **value})
    groups = {s: {r["instruction_group_sha256"] for r in records} for s, records in selected.items()}
    assert not groups["train"] & groups["development"] and not groups["train"] & groups["selection_evaluation"]
    assert not groups["development"] & groups["selection_evaluation"]
    protocol = {"source_script_sha256": sha(__file__), "models": {n: {"id": s[0], "revision": s[1]} for n, s in MODELS.items()},
                "dataset": {"id": "MiniLLM/dolly", "revision": DATA_REVISION,
                            "file_sha256": {f: sha(source / f) for f in ("raw.jsonl", "valid.jsonl")}},
                "split_basis": "MiniLLM script dev_num=1000, first raw records; valid.jsonl is selection evaluation, already observed in prior public-SFT work",
                "group_policy": "casefold and whitespace-normalized instruction across all contexts; conservative exact grouping, no complete semantic contamination audit",
                "counts": {s: len(v) for s, v in selected.items()}, "excluded": dict(excluded),
                "precision": {name: {"base_weights": "NF4 with double quantization" if name == "qwen_9b" else "BF16, no weight quantization",
                                     "compute": "BF16 autocast", "normalization": "FP32", "trainable_lora": "FP32"}
                              for name in MODELS},
                "prompt_policy": "same instruction/context; GPT-2 MiniLLM wrapper, Qwen native chat wrapper with thinking disabled",
                "max_total_training_tokens": MAX_LENGTH, "max_prompt_tokens": MAX_PROMPT,
                "response_truncation": False, "lora": LORA, "learning_rate_candidates": LR_CANDIDATES,
                "epochs": EPOCHS, "epoch_basis": "MiniLLM preparation phase uses three SFT epochs; bounded initial model-selection experiment, not original full SFT-baseline reproduction",
                "learning_rate_and_lora_basis": "QLoRA official 7B/13B and 33B scripts at 7f4e95a68dc076bea9b3a413d2b512eca6d004e5; published values are candidates, not asserted optima",
                "effective_batch_examples": ACCUMULATION, "training_seed": 0,
                "optimizer": {"name": "torch.optim.AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                              "weight_decay": 0, "max_grad_norm": 0.3},
                "scheduler": {"name": "constant_with_warmup", "warmup_ratio": 0.03,
                              "basis": "QLoRA warmup ratio with explicit warmup implementation; differs from its constant scheduler CLI"},
                "checkpoint_and_validation_cadence": "end of every complete epoch; latest checkpoints support resume",
                "selection": "lowest mean per-example response-token development NLL checkpoint, then lowest across the two LR candidates, independently per model; not a cross-tokenizer ranking metric",
                "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}}
    protocol["fingerprint"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    existing = OUT / "protocol.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8"))["fingerprint"] != protocol["fingerprint"] and any((LOCAL / "runs").glob("*/checkpoint-*")):
        raise RuntimeError("Refusing to change protocol after training checkpoints exist")
    for split, records in selected.items():
        path = LOCAL / f"{split}.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        for name in MODELS:
            path = LOCAL / name / f"{split}.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in encoded[name][split]), encoding="utf-8")
    protocol["prepared_files_sha256"] = {p.relative_to(LOCAL).as_posix(): sha(p) for p in LOCAL.glob("**/*.jsonl")}
    write_json(existing, protocol)
    print(json.dumps({"counts": protocol["counts"], "excluded": protocol["excluded"], "fingerprint": protocol["fingerprint"]}), flush=True)


def protocol():
    result = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert result["source_script_sha256"] == sha(__file__), "Source changed; prepare and reverify before training"
    for relative, expected in result["prepared_files_sha256"].items():
        assert sha(LOCAL / relative) == expected, relative
    return result


def load_model(name, adapter=None):
    set_seed(0)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16) if name == "qwen_9b" else None
    cls = Qwen3_5ForCausalLM if name.startswith("qwen") else AutoModelForCausalLM
    model, loading = cls.from_pretrained(snapshot(name), local_files_only=True, trust_remote_code=False,
                                         use_safetensors=True, dtype=torch.bfloat16, device_map={"": 0},
                                         attn_implementation="sdpa", quantization_config=quant, output_loading_info=True)
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), loading
    assert bool(getattr(model, "is_loaded_in_4bit", False)) == (name == "qwen_9b")
    assert not hasattr(model.model if name.startswith("qwen") else model, "visual")
    quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert bool(quantized) == (name == "qwen_9b")
    assert all(m.weight.quant_state.quant_type == "nf4" and m.weight.quant_state.nested for m in quantized)
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        # Restore BF16 vocabulary matrices after k-bit preparation, as in the QLoRA implementation.
        model.get_input_embeddings().to(torch.bfloat16)
        model.get_output_embeddings().to(torch.bfloat16)
    else:
        model.requires_grad_(False)
        for module in model.modules():
            if "norm" in type(module).__name__.lower():
                module.to(torch.float32)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False) if adapter else get_peft_model(model, LoraConfig(**LORA))
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    return model


def features(name, split):
    return [{k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in rows(LOCAL / name / f"{split}.jsonl")]


class Progress(TrainerCallback):
    def __init__(self, path, identity):
        self.path, self.identity = path, identity

    def on_log(self, args, state, control, logs=None, **kwargs):
        write_json(self.path, {**self.identity, "status": "training", "step": state.global_step,
                               "max_steps": state.max_steps, "epoch": state.epoch, "latest_log": logs,
                               "updated_at_utc": datetime.now(timezone.utc).isoformat()})


def preflight(names):
    spec = protocol()
    destination = OUT / "preflight.json"
    report = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {}
    if report.get("protocol_fingerprint") != spec["fingerprint"]:
        report = {"protocol_fingerprint": spec["fingerprint"], "models": {}}
    for name in names:
        started = perf_counter(); torch.cuda.reset_peak_memory_stats()
        print(f"PREFLIGHT {name}", flush=True)
        model = load_model(name)
        tokenizer = tokenizer_for(name)
        data = features(name, "train")
        example = max(data, key=lambda r: len(r["input_ids"]))
        shortest = min(data, key=lambda r: len(r["input_ids"]))
        padded = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)([shortest, example])
        for i, original in enumerate((shortest, example)):
            end = len(original["input_ids"]) - 1
            assert padded["labels"][i, end] == tokenizer.eos_token_id
            assert (padded["labels"][i, end + 1:] == -100).all()
            assert (padded["attention_mask"][i, end + 1:] == 0).all()
        batch = {k: torch.tensor([v], device="cuda") for k, v in example.items()}
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        assert trainable and all(p.dtype == torch.float32 for _, p in trainable)
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=2e-4, weight_decay=0, foreach=False)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
        before = {n: p.detach().clone() for n, p in trainable if "lora_B" in n}
        optimizer.step()
        assert any(not torch.equal(before[n], p) for n, p in trainable if n in before)
        assert all(torch.isfinite(p).all() for _, p in trainable)
        model.eval(); model.gradient_checkpointing_disable()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=1).logits[:, -1].float()
        folder = LOCAL / "preflight" / name; model.save_pretrained(folder)
        model.load_adapter(folder, adapter_name="reload_check", is_trainable=False); model.set_adapter("reload_check"); model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=1).logits[:, -1].float()
        assert torch.equal(actual, expected), "Reloaded adapter logits differ"
        prompt_length = next(i for i, value in enumerate(shortest["labels"]) if value != -100)
        probe_ids = torch.tensor([shortest["input_ids"][:prompt_length]], device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids),
                                       max_new_tokens=8, do_sample=False, use_cache=True,
                                       eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        assert generated.shape[1] > probe_ids.shape[1]
        record = {"status": "passed", "training_performed": "one disposable update, not a completed SFT baseline",
                  "trainable_parameters": sum(p.numel() for _, p in trainable), "sequence_tokens": len(example["input_ids"]),
                  "loss": float(loss.detach()), "reload_logits_equal": True,
                  "quantized_linear_layers": sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()),
                  "eos_preserved_and_padding_masked": True,
                  "cached_generation_probe_tokens": generated.shape[1] - probe_ids.shape[1],
                  "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                  "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
                  "seconds_including_load_and_checks": round(perf_counter() - started, 3)}
        report["models"][name] = record; write_json(destination, report)
        print(name, json.dumps(record), flush=True)
        del model, optimizer, batch, trainable, before, expected, actual, loss, padded, data, generated, probe_ids
        gc.collect(); torch.cuda.empty_cache()


def train(names):
    spec = protocol()
    checks = json.loads((OUT / "preflight.json").read_text(encoding="utf-8"))
    assert checks["protocol_fingerprint"] == spec["fingerprint"]
    assert all(checks["models"].get(n, {}).get("status") == "passed" for n in MODELS)
    trainer_checks = json.loads((OUT / "trainer_preflight.json").read_text(encoding="utf-8"))
    assert trainer_checks["protocol_fingerprint"] == spec["fingerprint"]
    assert all(trainer_checks["models"].get(n, {}).get("status") == "passed" for n in MODELS)
    for name in names:
        records = []
        for lr in LR_CANDIDATES:
            key = f"{name}_lr{lr:.0e}"
            destination = OUT / f"{key}.json"
            if destination.exists():
                previous = json.loads(destination.read_text(encoding="utf-8"))
                assert previous["protocol_fingerprint"] == spec["fingerprint"]
                if previous["status"] == "completed":
                    records.append(previous); continue
            run = LOCAL / "runs" / key; run.mkdir(parents=True, exist_ok=True)
            identity = {"model": name, "learning_rate": lr, "protocol_fingerprint": spec["fingerprint"]}
            write_json(destination, {**identity, "status": "training"})
            model, tokenizer = load_model(name), tokenizer_for(name)
            torch.cuda.reset_peak_memory_stats(); started = perf_counter()
            args = TrainingArguments(output_dir=str(run), num_train_epochs=EPOCHS,
                                     per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                     gradient_accumulation_steps=ACCUMULATION, learning_rate=lr,
                                     optim="adamw_torch", weight_decay=0, adam_beta1=0.9, adam_beta2=0.999,
                                     adam_epsilon=1e-8, lr_scheduler_type="constant_with_warmup", warmup_steps=0.03,
                                     max_grad_norm=0.3, bf16=True, tf32=False,
                                     gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                                     eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                                     load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                                     prediction_loss_only=True, seed=0, data_seed=0,
                                     dataloader_num_workers=0, report_to=[], disable_tqdm=True,
                                     logging_steps=10, logging_first_step=True, label_names=["labels"])
            trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                              train_dataset=features(name, "train"), eval_dataset=features(name, "development"),
                              data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
                              callbacks=[Progress(OUT / f"{key}_progress.json", identity)])
            baseline = trainer.evaluate(metric_key_prefix="before_sft")
            result = trainer.train(resume_from_checkpoint=get_last_checkpoint(str(run)))
            metrics = trainer.evaluate()
            assert math.isfinite(metrics["eval_loss"])
            best = run / "best_adapter"; trainer.save_model(best)
            record = {**identity, "status": "completed", "training_metrics": result.metrics,
                      "before_sft_development": baseline, "selected_development": metrics,
                      "best_checkpoint": trainer.state.best_model_checkpoint,
                      "adapter": best.relative_to(ROOT).as_posix(), "trainer_log_history": trainer.state.log_history,
                      "seconds": round(perf_counter() - started, 3),
                      "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                      "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2)}
            write_json(destination, record); records.append(record)
            del trainer, model; gc.collect(); torch.cuda.empty_cache()
        chosen = min(records, key=lambda r: r["selected_development"]["eval_loss"])
        write_json(OUT / f"{name}_selected.json", {"status": "selected", "protocol_fingerprint": spec["fingerprint"],
                                                "learning_rate": chosen["learning_rate"], "adapter": chosen["adapter"],
                                                "development_nll": chosen["selected_development"]["eval_loss"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "train"))
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    args = parser.parse_args()
    names = list(MODELS) if args.model == "all" else [args.model]
    if args.stage == "prepare":
        prepare()
    elif args.stage == "preflight":
        preflight(names)
    else:
        train(names)


if __name__ == "__main__":
    main()
