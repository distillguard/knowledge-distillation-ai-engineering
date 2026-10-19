"""Full-parameter GPT-2 base SFT search and fresh three-epoch KD initialization."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from filelock import FileLock
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import soft_gate_study as study

ROOT, LOCAL = study.ROOT, study.LOCAL / "student"
OUT = study.OUT / "student"
TEACHER_ID = "MiniLLM/teacher-gpt2-1.5B"
TEACHER_REVISION = "4a8d07d52ad67b7b87fec9aba25b1521173eb658"
LRS, BATCHES = (5e-4, 1e-4, 5e-5), (32, 64)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare():
    parent = read(study.OUT / "preparation.json")
    teacher = study.base.HUB / "models--MiniLLM--teacher-gpt2-1.5B/snapshots" / TEACHER_REVISION
    index = read(teacher / "pytorch_model.bin.index.json")
    config = read(teacher / "config.json")
    assert (config["n_layer"], config["n_embd"], config["vocab_size"]) == (48, 1600, 50257)
    tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True, trust_remote_code=False)
    assert tokenizer.get_vocab() == study.base.tokenizer_for("gpt2_base").get_vocab()
    audited = {}
    for split in ("train", "validation", "evaluation"):
        records = study.base.rows(study.LOCAL / "prepared" / f"{split}.jsonl")
        for row in records:
            encoded = study.encode(tokenizer, {**row, "output": row["references"][0]})
            assert all(encoded[key] == row[key] for key in ("input_ids", "labels", "prompt_ids"))
        audited[split] = len(records)
    files = sorted(set(index["weight_map"].values()) | {"config.json", "pytorch_model.bin.index.json",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"})
    teacher_spec = {"id": TEACHER_ID, "revision": TEACHER_REVISION,
        "snapshot": teacher.relative_to(ROOT).as_posix(), "training_performed_here": False,
        "source": f"https://huggingface.co/{TEACHER_ID}/tree/{TEACHER_REVISION}",
        "provenance": "author-released Dolly-SFT teacher linked to MiniLLM; model card identifies GPT-2 XL and Dolly",
        "upstream_hyperparameter_trace": "release does not independently establish every training hyperparameter or checkpoint-selection record",
        "teacher_tokenizer_matches_prepared_records": audited,
        "files": {name: {"sha256": study.base.sha(teacher / name), "bytes": (teacher / name).stat().st_size} for name in files}}
    teacher_spec["fingerprint"] = fingerprint(teacher_spec)
    destination = study.OUT / "teacher.json"
    if destination.exists():
        assert read(destination) == teacher_spec
    else:
        study.base.write_json(destination, teacher_spec)
    spec = {"parent_preparation_fingerprint": parent["fingerprint"], "teacher_fingerprint": teacher_spec["fingerprint"],
        "producer_sha256": study.base.sha(__file__), "model": parent["models"]["gpt2_base"],
        "method": "full-parameter SFT, no LoRA and no weight quantization",
        "training_seed": 10, "epoch_sampler_seed": "epoch number, as PyTorch DistributedSampler default seed=0 + epoch",
        "sft_search": {"learning_rates": list(LRS), "effective_batches": list(BATCHES), "epochs": 20,
                       "selection": "highest mean validation ROUGE-L F1; ties use lower batch-mean validation NLL, then fixed candidate order"},
        "initialization": {"epochs": 3, "restart_from_original": True,
                           "learning_rate_and_batch": "selected SFT search condition", "selection": "lowest batch-mean validation response NLL"},
        "optimizer": {"name": "torch.optim.AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
                      "no_decay_name_fragments": ["bias", "ln_f.weight", "ln_1.weight", "ln_2.weight", "ln_cross_attn"]},
        "scheduler": {"name": "CosineAnnealingLR", "eta_min": 1e-7, "warmup": 0,
                      "horizon": "ceil(train_rows / effective_batch) * epochs; stepped on applied updates"},
        "gradient_clip": 1.0, "microbatch": "effective_batch / 16, preserving official per-device batch scale",
        "loss_reduction": "sum response-token CE over the effective batch divided by all its unmasked targets",
        "precision": {"parameters_and_adam_state": "float32", "autocast_compute": "float16",
                      "gradient_scaler": "PyTorch GradScaler init_scale=2048, growth_interval=2000", "tf32": False},
        "validation": {"batch": 8, "generation_seed": 10, "every": "complete epoch", "preserve_training_rng": True,
                       "generation": parent["generation"], "response_nll": "both global-token mean and mean of batch response-token means"},
        "porting_differences": ["single GPU and stable raw order replace original 16-rank unordered processed-data execution",
            "all records retained per epoch, including final partial effective batch; no distributed shard drop/floor-step early stop",
            "global-token loss accumulation replaces the original average of rank-local token means",
            "FP32 master parameters with FP16 autocast and PyTorch loss scaling replace DeepSpeed FP16/ZeRO-1; loss-scale hysteresis differs",
            "validation uses fixed seed 10 and restores training RNG; original sampling may consume the shared RNG stream",
            "current installed library versions are the parent protocol versions; not the original patched Transformers stack"],
        "checkpoints": "separate best and epoch-boundary resume checkpoint for each condition; interruption replays unfinished epoch"}
    spec["fingerprint"] = fingerprint(spec)
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == spec, "Preserve the existing SFT protocol"
    else:
        study.base.write_json(path, spec)
    print("STUDENT_PROTOCOL_READY", spec["fingerprint"], flush=True)


def collate(records, device):
    ids = torch.full((len(records), 512), 50256, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    positions = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(records):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], device=device)
        mask[i, :n] = 1
        positions[i, :n] = torch.arange(n, device=device)
        labels[i, :n] = torch.tensor(row["labels"], device=device)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}, labels


def response_loss(model, records, device):
    inputs, labels = collate(records, device)
    logits = model(**inputs, use_cache=False).logits[:, :-1].float()
    shifted = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1), reduction="sum")
    count = int((shifted != -100).sum())
    assert count > 0
    return loss, count


def parameter_groups(model):
    no_decay = ("bias", "ln_f.weight", "ln_1.weight", "ln_2.weight", "ln_cross_attn")
    return [{"params": [p for n, p in model.named_parameters() if not any(s in n for s in no_decay)]},
            {"params": [p for n, p in model.named_parameters() if any(s in n for s in no_decay)], "weight_decay": 0.0}]


def load_model():
    set_seed(10)
    model = AutoModelForCausalLM.from_pretrained(study.base.snapshot("gpt2_base"), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.float32, device_map={"": 0}, attn_implementation="eager")
    assert sum(p.numel() for p in model.parameters()) == 124439808
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def validate(model, tokenizer, records):
    import sys
    sys.path.append(str(ROOT / "tmp/eval_deps"))
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rng, cuda_rng, training = torch.get_rng_state(), torch.cuda.get_rng_state(), model.training
    model.eval()
    model.gradient_checkpointing_disable()
    set_seed(10)
    sums, counts, batch_nlls, scores, outputs = [], [], [], [], []
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(records), 8):
                batch = records[start:start + 8]
                loss, count = response_loss(model, batch, "cuda")
                sums.append(float(loss)); counts.append(count); batch_nlls.append(float(loss)/count)
                ids = torch.full((len(batch), 256), 50256, device="cuda", dtype=torch.long)
                mask = torch.zeros_like(ids)
                for i, row in enumerate(batch):
                    ids[i, -len(row["prompt_ids"]):] = torch.tensor(row["prompt_ids"], device="cuda")
                    mask[i, -len(row["prompt_ids"]):] = 1
                tokens = model.generate(input_ids=ids, attention_mask=mask, max_length=512, do_sample=True,
                    temperature=1.0, top_k=0, top_p=1.0, no_repeat_ngram_size=6, eos_token_id=50256,
                    pad_token_id=50256, use_cache=True)[:, 256:]
                for row, sequence in zip(batch, tokens.tolist()):
                    response = tokenizer.decode(sequence, skip_special_tokens=True)
                    scores.append(scorer.score(row["references"][0], response)["rougeL"])
                    outputs.append({"id": row["id"], "response": response})
        result = {"validation_token_nll": math.fsum(sums)/sum(counts),
                  "validation_batch_mean_nll": statistics.mean(batch_nlls),
                  "validation_rougeL_precision": 100*statistics.mean(s.precision for s in scores),
                  "validation_rougeL_recall": 100*statistics.mean(s.recall for s in scores),
                  "validation_rougeL_f1": 100*statistics.mean(s.fmeasure for s in scores)}
        assert all(math.isfinite(v) for v in result.values())
        return result, outputs
    finally:
        torch.set_rng_state(rng); torch.cuda.set_rng_state(cuda_rng)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train(training)


def progress(stage, **fields):
    study.base.write_json(OUT / "worker_status.json", {"pid": os.getpid(), "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields})
    lines = ["# GPT-2 학생 SFT 진행 기록", "", f"현재 단계: **{stage}**", "",
        "교사는 MiniLLM 공식 Dolly-SFT XL을 재사용하며, 학생은 원본 GPT-2 base에서 전체 파라미터 SFT한다.", "",
        "| 조건 | 완료 epoch | 최적 검증 ROUGE-L F1 | 상태 |", "|---|---:|---:|---|"]
    for path in sorted(OUT.glob("sft_lr*_b*.json")):
        item = read(path)
        lines.append(f"| {path.stem} | {len(item['history'])}/20 | {item['best']['validation_rougeL_f1']:.4f} | {item['status']} |")
    if (OUT / "selected.json").exists():
        item = read(OUT / "selected.json")
        lines += ["", f"선택 조건: LR `{item['learning_rate']:.0e}`, effective batch `{item['effective_batch']}`, 검증 ROUGE-L F1 `{item['best']['validation_rougeL_f1']:.4f}`."]
    if (OUT / "initialization.json").exists():
        item = read(OUT / "initialization.json")
        lines += ["", f"별도 초기화: {len(item['history'])}/3 epoch, 상태 `{item['status']}`."]
    lines += ["", "| SFT 후 평가 대상 | 생성 채점 | 분류 평가 |", "|---|---|---|"]
    for name in ("teacher_sft", "student_sft", "student_init"):
        destination = study.OUT / "sft" / name
        lines.append(f"| {name} | {'완료' if (destination / 'summary.json').exists() else '대기'} | {'완료' if (destination / 'classification.json').exists() else '대기'} |")
    lines += ["", "[연구계획서](RESEARCH_PLAN_SOFT_GATE_GRADIENT.md) · [원본 기준선](SOFT_GATE_PROGRESS.md)", "",
        "기존 LoRA 예비 결과와 별도 실행이다. 생성 5 seed는 학습 반복이 아니다. 미완료 조건은 최종 성능으로 해석하지 않는다.", ""]
    (ROOT / "docs/SOFT_GATE_STUDENT_PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")


def update(model, optimizer, scaler, records, microbatch):
    total_targets = sum(sum(t != -100 for t in r["labels"][1:]) for r in records)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for start in range(0, len(records), microbatch):
        with torch.autocast("cuda", dtype=torch.float16):
            loss, count = response_loss(model, records[start:start + microbatch], "cuda")
        assert torch.isfinite(loss), "Nonfinite training loss"
        loss_sum += float(loss.detach())
        scaler.scale(loss / total_targets).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    return {"loss_sum": loss_sum, "targets": total_targets, "update_applied": scaler.get_scale() >= scale,
            "gradient_norm": float(norm) if torch.isfinite(norm) else None}


def probe(spec, records):
    folder = LOCAL / "probe"
    path = OUT / "preflight.json"
    if path.exists():
        assert read(path)["protocol_fingerprint"] == spec["fingerprint"]
        return
    progress("student_full_parameter_probe")
    model = load_model()
    model.train()
    optimizer = torch.optim.AdamW(parameter_groups(model), lr=LRS[0], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=2048, growth_interval=2000)
    parameter = model.transformer.h[0].mlp.c_fc.weight
    before = parameter.detach().clone()
    worst = max(records, key=lambda row: len(row["input_ids"]))
    torch.cuda.reset_peak_memory_stats()
    result = update(model, optimizer, scaler, [worst]*64, 4)
    assert result["update_applied"] and not torch.equal(before, parameter)
    model.save_pretrained(folder, safe_serialization=True)
    restored = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, dtype=torch.float32)
    assert torch.equal(parameter.detach().cpu(), restored.transformer.h[0].mlp.c_fc.weight)
    study.base.write_json(path, {"status": "passed", "protocol_fingerprint": spec["fingerprint"],
        "source_id": worst["id"], "sequence_tokens": len(worst["input_ids"]), "effective_batch": 64,
        "microbatch": 4, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "load_update_save_reload": True})
    del restored, model, optimizer, scaler, before, parameter
    gc.collect(); torch.cuda.empty_cache()


def train_one(spec, records, validation, lr, batch, epochs, role):
    key = f"sft_lr{lr:.0e}_b{batch}" if role == "sft" else "initialization"
    result_path, run = OUT / f"{key}.json", LOCAL / key
    if result_path.exists():
        result = read(result_path)
        assert result["protocol_fingerprint"] == spec["fingerprint"]
        if result["status"] == "completed":
            assert study.base.sha(ROOT / result["checkpoint"] / "model.safetensors") == result["checkpoint_sha256"]
            return result
    run.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(), study.base.tokenizer_for("gpt2_base")
    optimizer = torch.optim.AdamW(parameter_groups(model), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=math.ceil(len(records)/batch)*epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", init_scale=2048, growth_interval=2000)
    state_path = run / "resume.pt"
    result = {"status": "training", "protocol_fingerprint": spec["fingerprint"], "role": role,
              "learning_rate": lr, "effective_batch": batch, "history": [], "best": None,
              "checkpoint": (run / "best").relative_to(ROOT).as_posix()}
    try:
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            assert state["result"]["protocol_fingerprint"] == spec["fingerprint"]
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
            torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
            result = state["result"]
            del state
        for epoch in range(len(result["history"]), epochs):
            progress("student_sft" if role == "sft" else "student_initialization", condition=key, epoch=epoch, epochs=epochs)
            model.train()
            order = torch.randperm(len(records), generator=torch.Generator().manual_seed(epoch)).tolist()
            total_loss, total_targets, applied = 0., 0, 0
            started = time.monotonic()
            for start in range(0, len(order), batch):
                selected = [records[i] for i in order[start:start + batch]]
                values = update(model, optimizer, scaler, selected, batch//16)
                total_loss += values["loss_sum"]; total_targets += values["targets"]
                if values["update_applied"]:
                    scheduler.step(); applied += 1
            metrics, responses = validate(model, tokenizer, validation)
            record = {"epoch": epoch + 1, "train_response_token_nll": total_loss/total_targets,
                      "optimizer_updates_applied": applied, "seconds": time.monotonic()-started, **metrics}
            best = result["best"]
            improved = (best is None or ((metrics["validation_rougeL_f1"], -metrics["validation_batch_mean_nll"]) >
                        (best["validation_rougeL_f1"], -best["validation_batch_mean_nll"]) if role == "sft"
                        else metrics["validation_batch_mean_nll"] < best["validation_batch_mean_nll"]))
            if improved:
                model.save_pretrained(run / "best", safe_serialization=True)
                tokenizer.save_pretrained(run / "best")
                study.base.write_json(run / "best_validation_responses.json", responses)
                result["best"] = record
            result["history"].append(record)
            temporary = state_path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "result": result, "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state()}, temporary)
            temporary.replace(state_path)
            study.base.write_json(result_path, result)
            progress("student_sft" if role == "sft" else "student_initialization", condition=key, epoch=epoch+1, epochs=epochs)
            print("COMPLETED_STUDENT_EPOCH", key, epoch+1, json.dumps(metrics), flush=True)
        result.update(status="completed", checkpoint_sha256=study.base.sha(run / "best/model.safetensors"))
        study.base.write_json(result_path, result)
        return result
    finally:
        del model, optimizer, scheduler, scaler
        gc.collect(); torch.cuda.empty_cache()


def evaluate(spec, parent, validation):
    from run_soft_gate_stage1 import classify, classification

    records = study.base.rows(study.LOCAL / "prepared/evaluation.jsonl")
    tokenizer = study.base.tokenizer_for("gpt2_base")
    data, manifest = classification.classification_data({"gpt2_base": tokenizer})
    teacher = read(study.OUT / "teacher.json")
    for name, info in teacher["files"].items():
        assert study.base.sha(ROOT / teacher["snapshot"] / name) == info["sha256"]
    roles = {"teacher_sft": (teacher["snapshot"], teacher["files"], False)}
    for name, filename in (("student_sft", "selected.json"), ("student_init", "initialization.json")):
        item = read(OUT / filename)
        checkpoint = ROOT / item["checkpoint"]
        assert study.base.sha(checkpoint / "model.safetensors") == item["checkpoint_sha256"]
        roles[name] = (item["checkpoint"], {p.name: {"sha256": study.base.sha(p)} for p in sorted(checkpoint.iterdir()) if p.is_file()}, True)
    for name, (snapshot, files, safetensors) in roles.items():
        destination, folder = study.OUT / "sft" / name, study.LOCAL / "sft" / name
        identity = {"role": name, "parent_preparation_fingerprint": parent["fingerprint"],
            "student_protocol_fingerprint": spec["fingerprint"], "checkpoint_files": files,
            "generation": parent["generation"], "inference": parent["inference"], "classification_data": manifest,
            "classification_helper_sha256": study.base.sha(ROOT / "scripts/run_soft_gate_stage1.py"),
            "producer_sha256": study.base.sha(__file__)}
        identity["fingerprint"] = fingerprint(identity)
        protocol = destination / "evaluation_protocol.json"
        if protocol.exists():
            assert read(protocol) == identity
        else:
            study.base.write_json(protocol, identity)
        if (destination / "completion.json").exists():
            assert read(destination / "completion.json")["fingerprint"] == identity["fingerprint"]
            continue
        progress("sft_evaluation_loading", model=name)
        model = AutoModelForCausalLM.from_pretrained(ROOT / snapshot, local_files_only=True,
            trust_remote_code=False, use_safetensors=safetensors, dtype=torch.float16,
            attn_implementation="eager", device_map={"": 0}).eval().requires_grad_(False)
        try:
            nll_path = destination / "nll.json"
            if nll_path.exists():
                assert read(nll_path)["fingerprint"] == identity["fingerprint"]
            else:
                progress("sft_response_nll", model=name)
                study.base.write_json(nll_path, {"fingerprint": identity["fingerprint"],
                    "validation": study.nll(model, validation), "evaluation": study.nll(model, records)})
            # Reuse the tested baseline batch/RNG resume path, directing its status to this report.
            original_status = study.status
            try:
                study.status = lambda _stage, **fields: progress("sft_generation", **fields)
                for seed in parent["generation"]["seeds"]:
                    study.generate(model, tokenizer, records, folder, identity, seed)
                    print("COMPLETED_SFT_DOLLY_SEED", name, seed, flush=True)
            finally:
                study.status = original_status
            classify(name, model, data, identity["fingerprint"], phase="sft", tokenizer_name="gpt2_base")
            study.base.write_json(destination / "completion.json", {"status": "generation_completed_pending_scoring",
                "fingerprint": identity["fingerprint"], "model": name, "seeds": parent["generation"]["seeds"],
                "raw_responses": [{"path": (folder / f"seed{s}_responses.json").relative_to(ROOT).as_posix(),
                                   "sha256": study.base.sha(folder / f"seed{s}_responses.json")} for s in parent["generation"]["seeds"]]})
        finally:
            del model
            gc.collect(); torch.cuda.empty_cache()


def worker():
    spec = read(OUT / "protocol.json")
    assert spec["producer_sha256"] == study.base.sha(__file__)
    for name in study.NAMES:
        assert read(study.OUT / "original" / name / "summary.json")["status"] == "completed", "Original baseline first"
    parent = read(study.OUT / "preparation.json")
    assert parent["fingerprint"] == spec["parent_preparation_fingerprint"]
    for item in parent["data"].values():
        assert study.base.sha(ROOT / item["prepared_file"]) == item["sha256"]
    with FileLock(str(study.base.LOCAL / "worker.lock"), timeout=-1):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        records = study.base.rows(study.LOCAL / "prepared/train.jsonl")
        validation = study.base.rows(study.LOCAL / "prepared/validation.jsonl")
        probe(spec, records)
        results = [train_one(spec, records, validation, lr, batch, 20, "sft") for lr in LRS for batch in BATCHES]
        chosen = max(results, key=lambda r: (r["best"]["validation_rougeL_f1"], -r["best"]["validation_batch_mean_nll"]))
        study.base.write_json(OUT / "selected.json", {"status": "selected", "protocol_fingerprint": spec["fingerprint"],
            "learning_rate": chosen["learning_rate"], "effective_batch": chosen["effective_batch"],
            "checkpoint": chosen["checkpoint"], "checkpoint_sha256": chosen["checkpoint_sha256"], "best": chosen["best"]})
        progress("student_sft_selected")
        train_one(spec, records, validation, chosen["learning_rate"], chosen["effective_batch"], 3, "initialization")
        progress("student_sft_and_initialization_completed_pending_final_evaluation")
        evaluate(spec, parent, validation)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT / "tmp/supplementary_metrics_v1/deps"),
                       USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
    subprocess.run([sys.executable, "-u", "-B", "-X", "utf8", str(ROOT / "scripts/score_soft_gate_baselines.py"),
                    "--phase", "sft"], cwd=ROOT, env=environment, check=True)
    progress("completed_sft_preparation_and_evaluation_kd_not_started")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "worker"))
    args = parser.parse_args(argv)
    LOCAL.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare()
        progress("prepared_waiting_for_original_baselines")
    else:
        try:
            with FileLock(str(LOCAL / "student.lock"), timeout=0):
                worker()
        except Exception as error:
            progress("failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
