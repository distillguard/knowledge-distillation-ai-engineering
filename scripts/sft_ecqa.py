"""Run original benchmarks, then fixed-LR ECQA SFT epoch selection and matched test evaluation."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import math
from pathlib import Path
import time

import ecqa_study as base
import gpu_safety
import render_results

LOCAL = base.ROOT / "local/ecqa_sft_v1"
OUT = base.OUT / "sft"
LEARNING_RATES = {"qwen35_4b": 5e-5, "qwen35_2b": 5e-5, "qwen35_08b": 1e-4}
EPOCHS = 3
RECIPE = {"lora_rank": 8, "lora_alpha": 32, "lora_dropout": 0., "target_modules": "all-linear",
    "microbatch": 1, "gradient_accumulation": 16, "learning_rates_by_model": LEARNING_RATES,
    "epochs": EPOCHS, "early_stopping": False,
    "optimizer": "AdamW", "betas": [.9, .999], "epsilon": 1e-8, "weight_decay": 0.,
    "gradient_clip": 1., "schedule": "linear", "warmup_ratio": .05, "seed": 10,
    "dtype": "BF16 base; FP32 trainable LoRA", "quantization": None,
    "basis": "User-fixed size-specific SFT LR inspired by MiniLLM baseline search ranges; retained LoRA recipe, not an ECQA optimum or full MiniLLM reproduction"}


def status(stage, **fields):
    base.write(LOCAL / "worker_status.json", {"stage": stage, "pid": __import__("os").getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(), **fields})


def encode(row, tokenizer, cap):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": base.prompt(row, cap)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    target = "ANSWER: " + row["gold"] + "\nEXPLANATION: " + row["explanation"]
    response = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    assert response and all(x >= 0 for x in response)
    return {"input_ids": prefix+response, "attention_mask": [1]*(len(prefix)+len(response)),
            "labels": [-100]*len(prefix)+response}


def response_loss(model, inputs):
    import torch
    labels = inputs["labels"]
    assert labels.shape[0] == 1
    n = int((labels[0] != -100).sum())
    assert n > 0 and bool((labels[0, -n:] != -100).all())
    # Same response CE as full logits, without allocating logits for masked prompt positions.
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    use_cache=False, logits_to_keep=n+1)
    loss = torch.nn.functional.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
                                            labels[:, -n:].reshape(-1))
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite response CE")
    return loss, outputs


def memory_check():
    import psutil
    import torch
    free, _ = torch.cuda.mem_get_info()
    reason = gpu_safety.violation(free, psutil.virtual_memory().available, 0.)
    if reason:
        raise gpu_safety.ResourceLimitError(reason)


def update_probe(model, feature, folder, lr):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    before = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0., foreach=False)
    inputs = {k: torch.tensor([v], device="cuda") for k, v in feature.items()}
    started = time.monotonic()
    model.train()
    memory_check()
    loss, _ = response_loss(model, inputs)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    assert float(norm) > 0
    optimizer.step()
    memory_check()
    after = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    assert any(not torch.equal(before[k], after[k]) for k in before)
    model.save_pretrained(folder, safe_serialization=True)
    disk = load_file(str(folder / "adapter_model.safetensors"))
    assert set(disk) == set(after) and all(torch.equal(disk[k], after[k]) for k in after)
    set_peft_model_state_dict(model, before)
    assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in get_peft_model_state_dict(model).items())
    model.zero_grad(set_to_none=True)
    result = {"load_update_save_restore": "passed", "sequence_tokens": len(feature["input_ids"]),
        "response_tokens": sum(x != -100 for x in feature["labels"]), "loss": float(loss.detach()),
        "gradient_norm": float(norm), "seconds": time.monotonic()-started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(), "probe_update_used_for_training": False}
    del optimizer, inputs, loss, params, before, after, disk
    gc.collect()
    torch.cuda.empty_cache()
    return result


def dev_choices(model, tokenizer, rows, cap, name, trial):
    import torch
    from transformers import GenerationConfig
    header, labels, suffix = base.token_layout(tokenizer)
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=len(header)+1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
    records, was_training = [], model.training
    model.eval()
    try:
        with torch.inference_mode():
            for i, row in enumerate(rows):
                memory_check()
                if i % 25 == 0:
                    status("validating", model=name, trial=trial, completed=i, total=len(rows))
                text = tokenizer.apply_chat_template([{"role": "user", "content": base.prompt(row, cap)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
                formatter = base.AnswerFirst(inputs.input_ids.shape[-1], header, labels, suffix)
                output = model.generate(**inputs, generation_config=config, logits_processor=[formatter],
                    stopping_criteria=[gpu_safety.stopping_guard()])
                tokens = output[0, inputs.input_ids.shape[-1]:].tolist()
                assert tokens[:-1] == header and tokens[-1] in labels
                predicted = base.LABELS[labels.index(tokens[-1])]
                assert formatter.selected_label == predicted
                records.append({"id": row["id"], "gold": row["gold"], "predicted": predicted,
                    "choice_probabilities": formatter.probabilities, "choice_log_probabilities": formatter.log_probabilities,
                    "hit_length_cap": False, "explanation_present": False, "format_valid": True,
                    "output_tokens": len(tokens), "seconds": 0.})
        scores = base.metrics(records)
        for key in ("empty_explanation_count", "format_error_count", "mean_output_tokens", "total_generation_seconds", "length_cap_count"):
            scores.pop(key)
        return scores, records
    finally:
        model.train(was_training)


def trial_key(lr):
    return "lr" + format(lr, ".0e").replace("-0", "-")


def select_candidate(name, candidates):
    assert len(candidates) == EPOCHS
    assert {(r["learning_rate"], r["epoch"]) for r in candidates} == {
        (LEARNING_RATES[name], ep) for ep in range(1, EPOCHS+1)}
    return max(candidates, key=lambda r: (r["dev"]["accuracy"], r["dev"]["macro_f1"],
               -r["dev"]["choice_nll"], -r["epoch"]))


def train_trial(name, lr, tokenizer, path, rows, features, spec, cap):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    key = trial_key(lr)
    folder = LOCAL / name / key
    summary = OUT / "trials" / f"{name}_{key}_training.json"
    if summary.exists():
        completed = base.read(summary)
        assert completed["fingerprint"] == spec["fingerprint"] and completed["global_step"] == spec["maximum_updates"]
        for epoch in range(1, EPOCHS+1):
            candidate = base.read(OUT / "trials" / f"{name}_{key}_epoch{epoch}.json")
            assert base.digest(base.ROOT / candidate["adapter_path"] / "adapter_model.safetensors") == candidate["adapter_sha256"]
        return
    set_seed(RECIPE["seed"])
    model = get_peft_model(base.load_model(path), LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32,
                          lora_dropout=0., target_modules="all-linear", bias="none"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    try:
        status("resource_probe", model=name, trial=key)
        probe_features = [max(features, key=lambda r: len(r["input_ids"])),
                          max(features, key=lambda r: sum(x != -100 for x in r["labels"]))]
        probes = [update_probe(model, feature, folder / "probe_adapter", lr) for feature in probe_features]
        base.write(OUT / "trials" / f"{name}_{key}_probe.json", {"fingerprint": spec["fingerprint"], "probes": probes})

        class ResponseTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                loss, outputs = response_loss(model, inputs)
                return (loss, outputs) if return_outputs else loss

        class EpochCallback(TrainerCallback):
            def on_log(self, args, state, control, **kwargs):
                base.write(OUT / "trials" / f"{name}_{key}_history.json", {
                    "status": "running", "fingerprint": spec["fingerprint"], "log_history": state.log_history})

            def on_step_end(self, args, state, control, **kwargs):
                memory_check()
                status("training", model=name, trial=key, epoch=state.epoch,
                       global_step=state.global_step, total_steps=state.max_steps)

            def on_epoch_end(self, args, state, control, **kwargs):
                epoch = round(state.epoch)
                assert epoch in range(1, EPOCHS+1) and math.isclose(state.epoch, epoch)
                scores, predictions = dev_choices(model, tokenizer, rows["dev"], cap, name, key)
                adapter = folder / f"epoch{epoch}_adapter"
                model.save_pretrained(adapter, safe_serialization=True)
                record = {"fingerprint": spec["fingerprint"], "model": name, "learning_rate": lr,
                    "epoch": epoch, "global_step": state.global_step, "dev": scores,
                    "adapter_path": adapter.relative_to(base.ROOT).as_posix(),
                    "adapter_sha256": base.digest(adapter / "adapter_model.safetensors")}
                base.fixed_write(OUT / "trials" / f"{name}_{key}_epoch{epoch}.json", record)
                base.write(folder / f"epoch{epoch}_dev.json", {"fingerprint": spec["fingerprint"], "rows": predictions})
                print("EPOCH_DEV_COMPLETED", name, key, epoch, scores["accuracy"], scores["macro_f1"], flush=True)
                render_results.render()

        args = TrainingArguments(output_dir=str(folder / "checkpoints"), num_train_epochs=EPOCHS,
            per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=lr,
            lr_scheduler_type="linear", warmup_steps=spec["warmup_updates"], optim="adamw_torch",
            adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8, weight_decay=0., max_grad_norm=1.,
            bf16=True, tf32=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="no", save_strategy="steps", save_steps=50, save_total_limit=2,
            logging_steps=25, report_to=[], seed=10, data_seed=10, dataloader_num_workers=0,
            remove_unused_columns=False)
        trainer = ResponseTrainer(model=model, args=args, train_dataset=features, callbacks=[EpochCallback()])
        trainer.model_accepts_loss_kwargs = False
        checkpoint = get_last_checkpoint(str(folder / "checkpoints")) if (folder / "checkpoints").exists() else None
        started = time.monotonic()
        trainer.train(resume_from_checkpoint=checkpoint)
        assert math.isclose(trainer.state.epoch, EPOCHS) and trainer.state.global_step == spec["maximum_updates"]
        base.write(summary, {"status": "completed", "fingerprint": spec["fingerprint"], "learning_rate": lr,
            "epochs": trainer.state.epoch, "global_step": trainer.state.global_step,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "seconds_this_session": time.monotonic()-started, "log_history": trainer.state.log_history})
        del trainer
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run():
    from filelock import FileLock
    from transformers import AutoTokenizer
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (LOCAL / "RUN_DISABLED.json").exists():
        raise RuntimeError("SFT safety stop marker exists; inspect before restarting")
    with FileLock(str(LOCAL / "worker.lock"), timeout=0):
        try:
            if not all((base.OUT / "baselines" / f"{name}.json").exists() for name in base.MODELS):
                status("running_original_benchmarks")
                base.run()
            rows, data_spec = base.prepare()
            paths = base.snapshots()
            cap = base.read(base.OUT / "generation_limit.json")["selected_max_new_tokens"]
            with FileLock(str(base.ROOT / "local/gpu.lock"), timeout=0):
                for name, path in paths.items():
                    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
                    features = [encode(row, tokenizer, cap) for row in rows["train"]]
                    lr = LEARNING_RATES[name]
                    updates = math.ceil(len(features)/16)*EPOCHS
                    spec = {"model": base.MODELS[name][0], "revision": base.MODELS[name][1],
                        "dataset_fingerprint": data_spec["fingerprint"], "recipe": RECIPE,
                        "learning_rate": lr,
                        "updates_per_epoch": math.ceil(len(features)/16), "maximum_updates": updates,
                        "warmup_updates": math.ceil(.05*updates), "response_limit_during_evaluation": cap,
                        "maximum_training_sequence_tokens": max(len(r["input_ids"]) for r in features),
                        "target": "Original ECQA gold ANSWER, original human explanation, EOS; response-only CE",
                        "selection": "dev ACC, macro-F1, lower answer-choice NLL, earlier epoch; LR fixed before SFT",
                        "fresh_lora_per_model": True, "test_used_for_selection": False, "truncation": False,
                        "script_sha256": base.digest(__file__), "evaluator_sha256": base.digest(base.__file__),
                        "gpu_safety_sha256": base.digest(gpu_safety.__file__), "resource_safety": gpu_safety.LIMITS,
                        "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")}}
                    spec["fingerprint"] = base.fingerprint(spec)
                    base.fixed_write(OUT / f"{name}_training_protocol.json", spec)
                    train_trial(name, lr, tokenizer, path, rows, features, spec, cap)
                    candidates = [base.read(OUT / "trials" / f"{name}_{trial_key(lr)}_epoch{ep}.json")
                                  for ep in range(1, EPOCHS+1)]
                    selected = select_candidate(name, candidates)
                    selection = {"status": "selected", "fingerprint": spec["fingerprint"],
                                 "selected": selected, "test_used_for_selection": False}
                    base.fixed_write(OUT / f"{name}_selected.json", selection)
                    adapter = base.ROOT / selected["adapter_path"]
                    assert base.digest(adapter / "adapter_model.safetensors") == selected["adapter_sha256"]
                    result = base.evaluate(name, path, rows["test"], "test", cap, data_spec,
                        adapter=adapter, training=selection, report_status=status)
                    assert result["overall"]["n"] == len(rows["test"])
                    render_results.render()
                    print("SFT_TEST_COMPLETED", name, result["overall"]["accuracy"], result["overall"]["macro_f1"], flush=True)
            status("completed", models=list(base.MODELS), baseline_states=6)
            render_results.render()
        except Exception as error:
            base.write(LOCAL / "RUN_DISABLED.json", {"reason": str(error), "error_type": type(error).__name__,
                "automatic_restart": False, "created_utc": datetime.now(timezone.utc).isoformat()})
            status("failed", error_type=type(error).__name__, reason=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "report"))
    if parser.parse_args().action == "run":
        run()
    else:
        render_results.render()
