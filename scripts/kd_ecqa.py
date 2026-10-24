"""Guarded, nonquantized ECQA full-vocabulary FKL/RKL pathways and feasibility probe."""
import argparse
from datetime import datetime, timezone
import gc
import ast
import importlib.metadata
import importlib.util
import json
import math
import os
import time

import ecqa_study as base
import gpu_safety
import sft_ecqa as sft

LOCAL = base.ROOT / "local/ecqa_kd_v1"
OUT = base.OUT / "kd"
LR = 5e-6
TOKEN_CHUNK = 16


def status(stage, **fields):
    base.write(LOCAL / "worker_status.json", {"stage": stage, "pid": os.getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(), **fields})


def token_kl(student_logits, teacher_logits, direction, chunk_size=TOKEN_CHUNK):
    import torch
    from torch.utils.checkpoint import checkpoint
    assert direction in ("fkl", "rkl") and chunk_size > 0
    assert student_logits.ndim == 2 and student_logits.shape == teacher_logits.shape
    assert student_logits.shape[0] > 0
    teacher_logits = teacher_logits.detach()

    def reduction(student, teacher):
        logq = student.float().log_softmax(-1)
        logp = teacher.float().log_softmax(-1)
        if direction == "fkl":
            return (logp.exp() * (logp-logq)).sum()
        return (logq.exp() * (logq-logp)).sum()

    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, student_logits.shape[0], chunk_size):
        student = student_logits[start:start+chunk_size]
        teacher = teacher_logits[start:start+chunk_size]
        # Recompute temporary FP32 probabilities; retain the exact full vocabulary and gradient.
        value = checkpoint(reduction, student, teacher, use_reentrant=False) if torch.is_grad_enabled() and student.requires_grad else reduction(student, teacher)
        total = total + value
    return total / student_logits.shape[0]


def response_kl(student, teacher, inputs, direction, guard=None):
    import torch
    labels = inputs["labels"]
    assert labels.shape[0] == 1
    n = int((labels[0] != -100).sum())
    assert n > 0 and bool((labels[0, -n:] != -100).all()) and bool((labels[0, :-n] == -100).all())
    kwargs = {k: inputs[k] for k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        teacher_output = teacher(**kwargs, use_cache=False, logits_to_keep=n+1)
        target = teacher_output.logits[:, :-1].detach()
    if guard:
        guard.check()
    output = student(**kwargs, use_cache=False, logits_to_keep=n+1)
    loss = token_kl(output.logits[:, :-1].reshape(n, -1), target.reshape(n, -1), direction)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite response KL")
    if guard:
        guard.check()
    return loss, output


def training_response_kl(student, teacher, inputs, direction, accelerator, guard=None):
    assert accelerator.num_processes == 1 and accelerator.mixed_precision == "bf16"
    assert callable(student._original_forward)

    def forward(**kwargs):
        # Keep the same BF16 autocast, without Accelerate's redundant full FP32 logits copy.
        with accelerator.autocast():
            return student._original_forward(**kwargs)

    return response_kl(forward, teacher, inputs, direction, guard)


def selected(name):
    record = base.read(base.OUT / "sft" / f"{name}_selected.json")
    choice = record["selected"]
    adapter = base.ROOT / choice["adapter_path"]
    assert base.digest(adapter / "adapter_model.safetensors") == choice["adapter_sha256"]
    return choice, adapter


def load_selected(name, trainable, adapter_override=None):
    import torch
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM
    choice, adapter = selected(name)
    if adapter_override is not None:
        adapter = adapter_override
    model, loading = Qwen3_5ForCausalLM.from_pretrained(base.snapshots()[name], local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", output_loading_info=True)
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), loading
    assert not hasattr(model.model, "visual") and not getattr(model, "is_loaded_in_4bit", False)
    assert all(p.device.type == "cuda" for p in model.parameters())
    model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    model.config.use_cache = False
    if trainable:
        assert any(p.requires_grad for p in model.parameters())
        assert all("lora_" in n and p.dtype == torch.float32 for n, p in model.named_parameters() if p.requires_grad)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval().requires_grad_(False)
    return model


def memory_snapshot(stage):
    import psutil
    import torch
    free, total = torch.cuda.mem_get_info()
    return {"stage": stage, "free_vram_bytes": free, "total_vram_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "available_ram_bytes": psutil.virtual_memory().available}


def probe():
    import torch
    from filelock import FileLock
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, set_seed
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (LOCAL / "RUN_DISABLED.json").exists():
        raise RuntimeError("Inspect the KD stop marker before retrying")
    record = {"status": "running", "teacher": "qwen35_4b", "student": "qwen35_2b",
        "teacher_frozen": True, "quantization": None, "cpu_offload": False,
        "script_sha256": base.digest(__file__), "sft_helper_sha256": base.digest(sft.__file__),
        "evaluator_sha256": base.digest(base.__file__), "safety_sha256": base.digest(gpu_safety.__file__),
        "safety": gpu_safety.LIMITS, "loss": "full vocabulary, fixed gold prefixes, response and EOS KL",
        "temperature": 1., "microbatch": 1, "gradient_accumulation": 16, "learning_rate": LR,
        "token_chunk": TOKEN_CHUNK, "probe_updates_used_for_training": False,
        "memory": [], "updates": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    with FileLock(str(LOCAL / "worker.lock"), timeout=0), FileLock(str(base.ROOT / "local/gpu.lock"), timeout=0):
        teacher = student = optimizer = None
        try:
            rows, data_spec = base.prepare()
            record["dataset_fingerprint"] = data_spec["fingerprint"]
            path = base.snapshots()["qwen35_4b"]
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            assert all(base.digest(p / "tokenizer.json") == base.digest(path / "tokenizer.json") for p in base.snapshots().values())
            features = [(r["id"], sft.encode(r, tokenizer, 256)) for r in rows["train"]]
            longest = max(features, key=lambda r: len(r[1]["input_ids"]))
            response = max(features, key=lambda r: sum(x != -100 for x in r[1]["labels"]))
            record["probe_rows"] = [{"id": idx, "sequence_tokens": len(f["input_ids"]),
                "response_tokens": sum(x != -100 for x in f["labels"])} for idx, f in [longest, response]]
            for name in ["qwen35_4b", "qwen35_2b"]:
                choice, _ = selected(name)
                record[name+"_initialization"] = {k: choice[k] for k in ("epoch", "adapter_path", "adapter_sha256")}
            set_seed(10)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # Configure once for the whole process, rather than shrinking the cap at each model load.
            record["allocator"] = gpu_safety.configure()
            torch.cuda.reset_peak_memory_stats()
            status("loading_teacher")
            teacher = load_selected("qwen35_4b", False)
            sft.memory_check()
            record["memory"].append(memory_snapshot("teacher_loaded"))
            base.write(OUT / "pair_probe.json", record)
            status("loading_student")
            student = load_selected("qwen35_2b", True)
            sft.memory_check()
            record["memory"].append(memory_snapshot("pair_loaded"))
            base.write(OUT / "pair_probe.json", record)
            before = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(student).items()}
            teacher_versions = {n: p._version for n, p in teacher.named_parameters()}
            parameters = [p for p in student.parameters() if p.requires_grad]
            for direction in ("fkl", "rkl"):
                optimizer = torch.optim.AdamW(parameters, lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
                student.zero_grad(set_to_none=True)
                started = time.monotonic()
                losses, minimum_free = [], float("inf")
                for micro in range(16):
                    idx, feature = (longest, response)[micro % 2]
                    status("probing_update", direction=direction, micro_step=micro+1, total_micro_steps=16)
                    guard = gpu_safety.stopping_guard()
                    inputs = {k: torch.tensor([v], device="cuda") for k, v in feature.items()}
                    loss, outputs = response_kl(student, teacher, inputs, direction, guard)
                    losses.append(float(loss.detach()))
                    (loss/16).backward()
                    guard.check()
                    minimum_free = min(minimum_free, guard.minimum_free_vram)
                    assert all(p.grad is None for p in teacher.parameters())
                    del inputs, loss, outputs
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                assert float(norm) > 0
                optimizer.step()
                sft.memory_check()
                after = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(student).items()}
                assert any(not torch.equal(before[k], after[k]) for k in before)
                folder = LOCAL / "probe_adapter" / direction
                student.save_pretrained(folder, safe_serialization=True)
                disk = load_file(str(folder / "adapter_model.safetensors"))
                assert set(disk) == set(after) and all(torch.equal(disk[k], after[k]) for k in after)
                set_peft_model_state_dict(student, disk)
                assert all(torch.equal(v.detach().cpu(), disk[k]) for k, v in get_peft_model_state_dict(student).items())
                set_peft_model_state_dict(student, before)
                assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in get_peft_model_state_dict(student).items())
                assert teacher_versions == {n: p._version for n, p in teacher.named_parameters()}
                student.zero_grad(set_to_none=True)
                record["updates"].append({"direction": direction, "effective_batch": 16,
                    "loss": math.fsum(losses)/16, "gradient_norm": float(norm),
                    "seconds": time.monotonic()-started, "minimum_free_vram_bytes": minimum_free,
                    "load_update_save_restore": "passed", "teacher_unchanged": True})
                record["memory"].append(memory_snapshot(direction+"_updated"))
                base.write(OUT / "pair_probe.json", record)
                del optimizer, after, disk
                optimizer = None
                gc.collect()
                torch.cuda.empty_cache()
            record["status"] = "passed"
            record["completed_utc"] = datetime.now(timezone.utc).isoformat()
            base.write(OUT / "pair_probe.json", record)
            status("probe_passed", teacher="qwen35_4b", student="qwen35_2b")
        except Exception as error:
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            record["reason"] = str(error)
            record["memory"].append(memory_snapshot("failure"))
            base.write(OUT / "pair_probe.json", record)
            base.write(LOCAL / "RUN_DISABLED.json", {"reason": str(error), "error_type": type(error).__name__,
                "created_utc": datetime.now(timezone.utc).isoformat(), "automatic_restart": False})
            status("failed", reason=str(error), error_type=type(error).__name__)
            raise
        finally:
            del teacher, student, optimizer
            gc.collect()
            torch.cuda.empty_cache()


STAGES = {
    "ta_fkl": ("qwen35_4b", "qwen35_2b", "fkl", None),
    "ta_rkl": ("qwen35_4b", "qwen35_2b", "rkl", None),
    "direct_fkl": ("qwen35_4b", "qwen35_08b", "fkl", None),
    "direct_rkl": ("qwen35_4b", "qwen35_08b", "rkl", None),
    "via_fkl": ("qwen35_2b", "qwen35_08b", "fkl", "ta_fkl"),
    "via_rkl": ("qwen35_2b", "qwen35_08b", "rkl", "ta_rkl"),
}


def isolated_helper(module, label):
    # Separate module globals keep the frozen SFT response/result paths untouched.
    spec = importlib.util.spec_from_file_location(label, module.__file__)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def choose(candidates):
    assert len(candidates) == 3 and {r["epoch"] for r in candidates} == {1, 2, 3}
    assert all(r["learning_rate"] == LR for r in candidates)
    return max(candidates, key=lambda r: (r["dev"]["accuracy"], r["dev"]["macro_f1"],
        -r["dev"]["choice_nll"], -r["epoch"]))


def verify_probe():
    record = base.read(OUT / "pair_probe.json")
    assert record["status"] == "passed" and {r["direction"] for r in record["updates"]} == {"fkl", "rkl"}
    assert all(r["load_update_save_restore"] == "passed" and r["teacher_unchanged"] for r in record["updates"])
    assert record["quantization"] is None and not record["cpu_offload"] and record["temperature"] == 1.
    assert all(r["effective_batch"] == 16 and r["minimum_free_vram_bytes"] >= gpu_safety.LIMITS["minimum_free_vram_bytes"] for r in record["updates"])
    for name in ("qwen35_4b", "qwen35_2b"):
        assert record[name+"_initialization"]["adapter_sha256"] == selected(name)[0]["adapter_sha256"]
    source = LOCAL / "source_snapshot/kd_ecqa.py"
    assert base.digest(source) == record["script_sha256"]
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name in ("token_kl", "response_kl")}
    assert functions(source) == functions(base.Path(__file__))
    for key, module in (("sft_helper_sha256", sft), ("evaluator_sha256", base), ("safety_sha256", gpu_safety)):
        assert record[key] == base.digest(module.__file__)
    return record


def stage_probe(student, teacher, feature, direction, folder):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    before = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(student).items()}
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR,
        betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
    inputs = {k: torch.tensor([v], device="cuda") for k, v in feature.items()}
    guard = gpu_safety.stopping_guard()
    student.zero_grad(set_to_none=True)
    loss, outputs = response_kl(student, teacher, inputs, direction, guard)
    loss.backward()
    guard.check()
    norm = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    assert float(norm) > 0 and all(p.grad is None for p in teacher.parameters())
    optimizer.step()
    sft.memory_check()
    after = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(student).items()}
    assert any(not torch.equal(before[k], after[k]) for k in before)
    student.save_pretrained(folder, safe_serialization=True)
    disk = load_file(str(folder / "adapter_model.safetensors"))
    assert set(disk) == set(after) and all(torch.equal(disk[k], after[k]) for k in after)
    set_peft_model_state_dict(student, disk)
    assert all(torch.equal(v.detach().cpu(), disk[k]) for k, v in get_peft_model_state_dict(student).items())
    set_peft_model_state_dict(student, before)
    assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in get_peft_model_state_dict(student).items())
    student.zero_grad(set_to_none=True)
    result = {"status": "passed", "load_update_save_restore": "passed", "probe_update_used_for_training": False,
        "loss": float(loss.detach()), "gradient_norm": float(norm), "memory": memory_snapshot("stage_probe"),
        "minimum_free_vram_bytes": guard.minimum_free_vram}
    del optimizer, inputs, loss, outputs, before, after, disk
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_stage(stage, tokenizer, rows, features, spec, teacher_adapter):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    teacher_name, student_name, direction, _ = STAGES[stage]
    folder, public = LOCAL / stage, OUT / stage
    summary = public / "training.json"
    if summary.exists():
        saved = base.read(summary)
        assert saved["status"] == "completed" and saved["fingerprint"] == spec["fingerprint"]
        assert saved["global_step"] == 1419 and saved["epochs"] == 3 and saved["teacher_unchanged"]
        for epoch in (1, 2, 3):
            item = base.read(public / f"epoch{epoch}.json")
            assert item["fingerprint"] == spec["fingerprint"] and item["global_step"] == 473*epoch
            assert base.digest(base.ROOT / item["adapter_path"] / "adapter_model.safetensors") == item["adapter_sha256"]
        return
    set_seed(10)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_safety.configure()
    torch.cuda.reset_peak_memory_stats()
    teacher = student = trainer = None
    try:
        status("loading_teacher", experiment=stage)
        teacher = load_selected(teacher_name, False, teacher_adapter)
        status("loading_student", experiment=stage)
        student = load_selected(student_name, True)
        sft.memory_check()
        teacher_versions = {n: p._version for n, p in teacher.named_parameters()}
        status("resource_probe", experiment=stage)
        extremes = [max(features, key=lambda r: len(r["input_ids"])),
                    max(features, key=lambda r: sum(x != -100 for x in r["labels"]))]
        probes = [stage_probe(student, teacher, feature, direction, folder / "probe_adapter") for feature in extremes]
        base.write(public / "probe.json", {"fingerprint": spec["fingerprint"], "probes": probes})
        dev_helper = isolated_helper(sft, "kd_dev_helper")
        dev_helper.status = lambda phase, **fields: status(phase, experiment=stage, **fields)

        class KlTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                self.item_guard = gpu_safety.stopping_guard()
                loss, outputs = training_response_kl(model, teacher, inputs, direction, self.accelerator, self.item_guard)
                return (loss, outputs) if return_outputs else loss

            def training_step(self, *args, **kwargs):
                result = super().training_step(*args, **kwargs)
                self.item_guard.check()
                return result

        class EpochCallback(TrainerCallback):
            def on_log(self, args, state, control, **kwargs):
                base.write(public / "history.json", {"fingerprint": spec["fingerprint"], "log_history": state.log_history})

            def on_step_end(self, args, state, control, **kwargs):
                sft.memory_check()
                assert all(p.grad is None for p in teacher.parameters())
                status("training", experiment=stage, epoch=state.epoch, global_step=state.global_step, total_steps=state.max_steps)

            def on_epoch_end(self, args, state, control, **kwargs):
                epoch = round(state.epoch)
                assert epoch in (1, 2, 3) and math.isclose(state.epoch, epoch)
                scores, predictions = dev_helper.dev_choices(student, tokenizer, rows["dev"], 256, student_name, stage)
                adapter = folder / f"epoch{epoch}_adapter"
                student.save_pretrained(adapter, safe_serialization=True)
                item = {"fingerprint": spec["fingerprint"], "epoch": epoch, "global_step": state.global_step,
                    "learning_rate": LR, "dev": scores, "adapter_path": adapter.relative_to(base.ROOT).as_posix(),
                    "adapter_sha256": base.digest(adapter / "adapter_model.safetensors")}
                base.fixed_write(public / f"epoch{epoch}.json", item)
                base.write(folder / f"epoch{epoch}_dev.json", {"fingerprint": spec["fingerprint"], "rows": predictions})
                report()
                print(json.dumps({"epoch_completed": stage, "epoch": epoch, "dev": scores}), flush=True)

        args = TrainingArguments(output_dir=str(folder / "checkpoints"), num_train_epochs=3,
            per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=LR,
            lr_scheduler_type="linear", warmup_steps=71, optim="adamw_torch", adam_beta1=.9, adam_beta2=.999,
            adam_epsilon=1e-8, weight_decay=0., max_grad_norm=1., bf16=True, tf32=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="no", save_strategy="steps", save_steps=50, save_total_limit=2, logging_steps=25,
            report_to=[], seed=10, data_seed=10, dataloader_num_workers=0, remove_unused_columns=False)
        trainer = KlTrainer(model=student, args=args, train_dataset=features, callbacks=[EpochCallback()])
        trainer.model_accepts_loss_kwargs = False
        checkpoint = get_last_checkpoint(str(folder / "checkpoints")) if (folder / "checkpoints").exists() else None
        status("starting_training", experiment=stage, resume_checkpoint=checkpoint)
        started = time.monotonic()
        trainer.train(resume_from_checkpoint=checkpoint)
        assert math.isclose(trainer.state.epoch, 3) and trainer.state.global_step == 1419
        assert teacher_versions == {n: p._version for n, p in teacher.named_parameters()}
        base.write(summary, {"status": "completed", "fingerprint": spec["fingerprint"], "epochs": 3,
            "global_step": 1419, "teacher_unchanged": True, "seconds_this_session": time.monotonic()-started,
            "resumed_from_checkpoint": checkpoint, "log_history": trainer.state.log_history})
    finally:
        del trainer, teacher, student
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_stage(stage, rows, data_spec, selection):
    evaluator = isolated_helper(base, "kd_test_helper")
    evaluator.LOCAL = LOCAL / stage
    evaluator.OUT = OUT / stage
    student_name = STAGES[stage][1]
    adapter = base.ROOT / selection["selected"]["adapter_path"]
    assert base.digest(adapter / "adapter_model.safetensors") == selection["selected"]["adapter_sha256"]
    result = evaluator.evaluate(student_name, base.snapshots()[student_name], rows["test"], "test", 256,
        data_spec, adapter=adapter, training=selection,
        report_status=lambda phase, **fields: status(phase, experiment=stage, **fields))
    protocol = base.read(evaluator.OUT / "protocols" / f"{student_name}_sft_test_256.json")
    records = [base.read(evaluator.LOCAL / "responses" / (student_name+"_sft") / "test/256" / (r["id"]+".json"))
               for r in rows["test"]]
    assert result["status"] == "completed" and result["overall"]["n"] == 2194
    assert protocol["fingerprint"] == result["fingerprint"] and protocol["max_new_tokens"] == 256
    for item, row in zip(records, rows["test"]):
        assert item["id"] == row["id"] and item["gold"] == row["gold"] and item["fingerprint"] == result["fingerprint"]
        assert 0 < item["output_tokens"] <= 256
        assert base.inspect_response(item["response"]) == {k: item[k] for k in ("predicted", "explanation_present", "format_valid")}
    assert base.metrics(records) == result["overall"]
    base.write(evaluator.OUT / "completion_verification.json", {"status": "passed", "fingerprint": result["fingerprint"],
        "n": 2194, "raw_answers_reparsed": True, "metrics_recomputed": True, "max_new_tokens": 256})
    return result


def paired_counts(first, second):
    assert len(first) == len(second) and len(first) > 0
    counts = {"both_correct": 0, "first_only_correct": 0, "second_only_correct": 0,
        "both_wrong": 0, "same_wrong_answer": 0}
    for a, b in zip(first, second):
        assert a["id"] == b["id"] and a["gold"] == b["gold"]
        ac, bc = a["predicted"] == a["gold"], b["predicted"] == b["gold"]
        key = "both_correct" if ac and bc else ("first_only_correct" if ac else ("second_only_correct" if bc else "both_wrong"))
        counts[key] += 1
        counts["same_wrong_answer"] += int(not ac and not bc and a["predicted"] == b["predicted"])
    return {"n": len(first), **counts,
        "second_minus_first_accuracy_pp": 100*(counts["second_only_correct"]-counts["first_only_correct"])/len(first)}


def completed_figures(results):
    if not results:
        return ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    labels = list(results)
    for ax, metric, ylabel in zip(axes, ("accuracy", "macro_f1"), ("Test ACC (%)", "Test Macro-F1")):
        scale = 100 if metric == "accuracy" else 1
        values = [scale*results[k]["overall"][metric] for k in labels]
        bars = ax.bar(labels, values, color=["#3267a8" if "fkl" in k else "#d77328" for k in labels])
        ax.bar_label(bars, fmt="%.2f" if scale == 100 else "%.4f", padding=3, fontsize=8)
        ax.set(ylabel=ylabel, ylim=(0, 100 if scale == 100 else 1))
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=.15)
    for suffix in ("png", "svg"):
        fig.savefig(folder / ("kd_comparison."+suffix), dpi=180)
    plt.close(fig)
    return "\n\n![완료된 KD 모델 비교](artifacts/ecqa/kd/figures/kd_comparison.png)\n\n완료·원문 검증된 전체 test 결과만 표시했다.\n"


def report():
    import render_results
    render_results.render()
    lines = ["| 단계 | 학생 | KL | 상태 | 선정 epoch | Dev ACC (%) | Test ACC (%) | Macro-F1 | 설명 상한/2,194 |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    results, raw, costs = {}, {}, {}
    for stage, (_, student, direction, _) in STAGES.items():
        public = OUT / stage
        selection = base.read(public / "selected.json") if (public / "selected.json").exists() else None
        result_path = public / "sft" / f"{student}.json"
        verified = (public / "completion_verification.json").exists()
        result = base.read(result_path) if result_path.exists() and verified else None
        if result:
            assert result["status"] == "completed" and result["overall"]["n"] == 2194
            assert base.read(public / "completion_verification.json")["fingerprint"] == result["fingerprint"]
            results[stage] = result
            raw[stage] = sorted([base.read(p) for p in (LOCAL / stage / "responses" / (student+"_sft") / "test/256").glob("*.json")], key=lambda r: r["id"])
            training = base.read(public / "training.json")
            costs[stage] = {"training_dev_seconds_this_session": training["seconds_this_session"],
                "training_was_resumed": bool(training["resumed_from_checkpoint"]),
                "test_generation_seconds": result["overall"]["total_generation_seconds"]}
        epochs = sum((public / f"epoch{ep}.json").exists() for ep in (1, 2, 3))
        state = "평가 완료" if result else ("test 대기·진행" if selection else (f"학습·검증 {epochs}/3 epoch" if (public / "training_protocol.json").exists() else "미실행"))
        choice = selection["selected"] if selection else None
        lines.append(f"| {stage} | {student} | {direction.upper()} | {state} | {choice['epoch'] if choice else '—'} | "
            + (f"{100*choice['dev']['accuracy']:.2f}" if choice else "—") + " | "
            + (f"{100*result['overall']['accuracy']:.2f} | {result['overall']['macro_f1']:.4f} | {result['overall']['length_cap_count']}" if result else "— | — | —") + " |")
    comparisons = {}
    for direction in ("fkl", "rkl"):
        direct, via = "direct_"+direction, "via_"+direction
        if direct in raw and via in raw:
            comparisons[direction] = paired_counts(raw[direct], raw[via])
    if results:
        _, data_spec = base.prepare()
        base.write(OUT / "comparisons.json", {"dataset_fingerprint": data_spec["fingerprint"],
            "paired_direct_vs_intermediate": comparisons, "costs": costs,
            "cost_scope": "Training+dev current session and saved test generation; excludes model load, probes and prior interrupted sessions",
            "single_seed": 10, "explanation_factuality_reviewed": False})
    content = "\n".join(lines) + "\n\nT=1·순수 KL·동일 정답/사람 설명 prefix·전체 어휘. 단일 seed이며 설명 사실성은 미검토다.\n"
    content += completed_figures(results)
    if comparisons:
        content += "\n| 경로 비교 | 중간 − 직접 ACC (%p) | 직접만 정답 | 중간만 정답 |\n|---|---:|---:|---:|\n"
        for direction, item in comparisons.items():
            content += f"| {direction.upper()} | {item['second_minus_first_accuracy_pp']:+.3f} | {item['first_only_correct']} | {item['second_only_correct']} |\n"
    for name in ("RESEARCH_PLAN.md", "RESEARCH_REPORT.md", "RESULTS_VISUALIZATION.md"):
        path = base.ROOT / name
        text = path.read_text(encoding="utf-8")
        assert text.count("<!-- KD_STATUS_START -->") == 1
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(render_results.update_section(text, "KD_STATUS", content), encoding="utf-8")
        temporary.replace(path)


def run():
    import torch
    from filelock import FileLock
    from transformers import AutoTokenizer
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (LOCAL / "RUN_DISABLED.json").exists():
        raise RuntimeError("Inspect the KD stop marker before restarting")
    with FileLock(str(LOCAL / "worker.lock"), timeout=0), FileLock(str(base.ROOT / "local/gpu.lock"), timeout=0):
        try:
            pair_probe = verify_probe()
            rows, data_spec = base.prepare()
            assert data_spec["fingerprint"] == pair_probe["dataset_fingerprint"]
            assert [len(rows[k]) for k in ("train", "dev", "test")] == [7566, 1079, 2194]
            assert base.read(base.OUT / "generation_limit.json")["selected_max_new_tokens"] == 256
            tokenizer = AutoTokenizer.from_pretrained(base.snapshots()["qwen35_4b"], local_files_only=True, trust_remote_code=False)
            features = [sft.encode(r, tokenizer, 256) for r in rows["train"]]
            assert math.ceil(len(features)/16)*3 == 1419
            for stage, (teacher_name, student_name, direction, teacher_stage) in STAGES.items():
                teacher_choice = base.read(OUT / teacher_stage / "selected.json")["selected"] if teacher_stage else selected(teacher_name)[0]
                student_choice, _ = selected(student_name)
                teacher_adapter = base.ROOT / teacher_choice["adapter_path"]
                assert base.digest(teacher_adapter / "adapter_model.safetensors") == teacher_choice["adapter_sha256"]
                spec = {"stage": stage, "teacher": teacher_name, "student": student_name, "direction": direction,
                    "teacher_initialization": teacher_choice, "student_initialization": student_choice,
                    "base_revisions": {k: v[1] for k, v in base.MODELS.items()},
                    "dataset_fingerprint": data_spec["fingerprint"], "test_ordered_ids_sha256": base.fingerprint([r["id"] for r in rows["test"]]),
                    "tokenizer_sha256": base.digest(base.snapshots()[teacher_name] / "tokenizer.json"),
                    "loss": "pure full-vocabulary response/explanation/EOS KL; prompt masked; fixed original gold prefixes",
                    "temperature": 1., "cross_entropy_mixture": False, "student_rollout": False,
                    "learning_rate": LR, "epochs": 3, "updates_per_epoch": 473, "maximum_updates": 1419,
                    "warmup_updates": 71, "recipe": {**sft.RECIPE, "learning_rates_by_model": None, "learning_rate": LR,
                        "basis": "User-fixed KD reference LR; not an ECQA optimum or full MiniLLM reproduction"},
                    "fresh_optimizer_scheduler_per_stage": True, "continue_selected_sft_lora": True,
                    "test_used_for_selection": False, "selection": "dev ACC, Macro-F1, lower A-E conditional NLL, earlier epoch",
                    "evaluation_max_new_tokens": 256, "truncation": False, "cpu_offload": False,
                    "kl_math_dtype": "float32", "token_chunk": TOKEN_CHUNK,
                    "student_logits_handling": "Same BF16 Accelerate autocast; omit redundant full FP32 output copy, retain FP32 KL within token chunks",
                    "runtime_environment": {k: os.environ.get(k) for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM")},
                    "pair_probe_sha256": base.digest(OUT / "pair_probe.json"), "pair_probe_source_sha256": pair_probe["script_sha256"],
                    "probe_loss_ast_matches": True, "script_sha256": base.digest(__file__),
                    "sft_helper_sha256": base.digest(sft.__file__), "evaluator_sha256": base.digest(base.__file__),
                    "safety_sha256": base.digest(gpu_safety.__file__), "safety": gpu_safety.LIMITS,
                    "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")}}
                spec["fingerprint"] = base.fingerprint(spec)
                base.fixed_write(OUT / stage / "training_protocol.json", spec)
                report()
                train_stage(stage, tokenizer, rows, features, spec, teacher_adapter)
                gc.collect()
                torch.cuda.empty_cache()
                candidates = [base.read(OUT / stage / f"epoch{epoch}.json") for epoch in (1, 2, 3)]
                selection = {"status": "selected", "fingerprint": spec["fingerprint"], "selected": choose(candidates), "test_used_for_selection": False}
                base.fixed_write(OUT / stage / "selected.json", selection)
                result = evaluate_stage(stage, rows, data_spec, selection)
                report()
                pending_path = LOCAL / "completion_reports.json"
                pending = base.read(pending_path) if pending_path.exists() else {}
                pending.setdefault(stage+":"+result["fingerprint"], {"stage": stage, "fingerprint": result["fingerprint"],
                    "reported": False, "selected_epoch": selection["selected"]["epoch"], "overall": result["overall"]})
                base.write(pending_path, pending)
                print(json.dumps({"test_completed": stage, "fingerprint": result["fingerprint"], "overall": result["overall"]}), flush=True)
            status("completed", training_stages=6, final_students=4)
            report()
        except Exception as error:
            base.write(LOCAL / "RUN_DISABLED.json", {"reason": str(error), "error_type": type(error).__name__,
                "created_utc": datetime.now(timezone.utc).isoformat(), "automatic_restart": False})
            status("failed", reason=str(error), error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "run", "report"))
    action = parser.parse_args().action
    {"probe": probe, "run": run, "report": report}[action]()
