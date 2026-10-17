"""Disposable BF16 Gemma checks that never keep two models/adapters on the GPU.

The fingerprinted study engine is supplied by dolly_external. These checks use
its existing features/loader without changing either formal training script.
"""

import gc
import hashlib
import math
from pathlib import Path
import tempfile
from time import perf_counter


def _name(engine):
    names = list(engine.MODELS)
    if names != ["gemma2_9b_it"]:
        raise ValueError("External preflight requires only gemma2_9b_it")
    return names[0]


def _memory(torch):
    return {
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
    }


def _source_hash(spec):
    actual = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert actual == spec["external_probe_script_sha256"], "External probe source changed after preparation"
    return actual


def run_manual(engine):
    """Check a real update, masking and independent save/reload with bounded VRAM."""
    import torch

    name, spec = _name(engine), engine.protocol()
    source_hash = _source_hash(spec)
    engine.LOCAL.mkdir(parents=True, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    available, _ = torch.cuda.mem_get_info()
    started = perf_counter()
    model, tokenizer = engine.load_model(name), engine.tokenizer_for(name)
    data = engine.features(name, "train")
    longest = max(data, key=lambda row: len(row["input_ids"]))
    shortest = min(data, key=lambda row: len(row["input_ids"]))
    padded = engine.DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)([shortest, longest])
    for index, row in enumerate((shortest, longest)):
        final = len(row["input_ids"]) - 1
        assert padded["labels"][index, final] == tokenizer.eos_token_id
        assert (padded["labels"][index, final + 1:] == -100).all()
        assert (padded["attention_mask"][index, final + 1:] == 0).all()
        assert row["labels"][-1] == tokenizer.eos_token_id
        prompt_end = next(i for i, token in enumerate(row["labels"]) if token != -100)
        assert row["labels"][:prompt_end] == [-100] * prompt_end
        assert row["labels"][prompt_end:] == row["input_ids"][prompt_end:]
    del padded
    batch = {key: torch.tensor([value], device="cuda") for key, value in longest.items()}
    trainable = [(key, value) for key, value in model.named_parameters() if value.requires_grad]
    assert trainable and all("lora_" in key and value.dtype == torch.float32 for key, value in trainable)
    assert not getattr(model, "is_loaded_in_4bit", False)
    assert not getattr(model, "is_loaded_in_8bit", False)
    assert not any("4bit" in type(module).__name__.lower() or "8bit" in type(module).__name__.lower()
                   for module in model.modules())
    trainable_count = sum(value.numel() for _, value in trainable)
    optimizer = torch.optim.AdamW([value for _, value in trainable], lr=2e-4, weight_decay=0, foreach=False)
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch, use_cache=False).loss
    assert torch.isfinite(loss)
    loss_value = float(loss.detach())
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for _, value in trainable)
    # One nonzero-gradient tensor is enough to verify an update. Copying every
    # LoRA B tensor would inflate the engineering probe's peak beyond training.
    parameter = next(value for key, value in trainable
                     if "lora_B" in key and torch.count_nonzero(value.grad).item())
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameter)
    assert all(torch.isfinite(value).all() for _, value in trainable)
    model.zero_grad(set_to_none=True)
    del optimizer, trainable, parameter, before, loss
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    model.gradient_checkpointing_disable()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         logits_to_keep=1).logits[:, -1].float().cpu()
    with tempfile.TemporaryDirectory(prefix="manual_preflight_", dir=engine.LOCAL) as temporary:
        folder = Path(temporary)
        model.save_pretrained(folder)
        assert (folder / "adapter_config.json").is_file()
        assert (folder / "adapter_model.safetensors").is_file()
        # A complete reload exercises the transferable adapter without carrying
        # the old frozen base, optimizer or a second adapter into the VRAM peak.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = engine.load_model(name, folder)
        model.eval()
        model.gradient_checkpointing_disable()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           logits_to_keep=1).logits[:, -1].float().cpu()
        assert torch.equal(actual, expected), "Reloaded adapter logits differ"
        prompt_length = next(i for i, token in enumerate(shortest["labels"]) if token != -100)
        prompt = torch.tensor([shortest["input_ids"][:prompt_length]], device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                       max_new_tokens=8, do_sample=False, use_cache=True,
                                       eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        generated_count = generated.shape[1] - prompt.shape[1]
        assert generated_count > 0
        del model, batch, prompt, generated, expected, actual
        gc.collect()
        torch.cuda.empty_cache()
    record = {
        "status": "passed", "training_performed": "one disposable update, not a completed SFT baseline",
        "trainable_parameters": trainable_count, "sequence_tokens": len(longest["input_ids"]),
        "loss": loss_value, "reload_logits_equal": True, "independent_base_and_adapter_reload": True,
        "quantized_linear_layers": 0, "eos_preserved_and_padding_masked": True,
        "cached_generation_probe_tokens": generated_count,
        "available_before_probe_bytes": available,
        **_memory(torch), "seconds_including_load_and_checks": round(perf_counter() - started, 3),
    }
    report = {"protocol_fingerprint": spec["fingerprint"],
              "probe_script_sha256": source_hash, "models": {name: record}}
    engine.write_json(engine.OUT / "preflight.json", report)
    print(name, record, flush=True)
    return report


def run_trainer(engine):
    """Check a second longest-sequence update with resident AdamW states."""
    import torch
    from transformers.trainer_utils import get_last_checkpoint

    name, spec = _name(engine), engine.protocol()
    source_hash = _source_hash(spec)
    engine.LOCAL.mkdir(parents=True, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    available, _ = torch.cuda.mem_get_info()
    started = perf_counter()
    model, tokenizer = engine.load_model(name), engine.tokenizer_for(name)
    data = engine.features(name, "train")
    longest = max(data, key=lambda row: len(row["input_ids"]))
    # AdamW materializes its moment tensors on the first update. A second
    # effective batch is required to test activations while those states live.
    # Repeating the longest example makes this independent of Trainer shuffle.
    batch = [longest] * (2 * engine.ACCUMULATION)
    with tempfile.TemporaryDirectory(prefix="trainer_preflight_", dir=engine.LOCAL) as temporary:
        args = engine.TrainingArguments(
            output_dir=temporary, max_steps=2, learning_rate=2e-4,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=engine.ACCUMULATION,
            optim="adamw_torch", weight_decay=0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
            # The engineering update needs nonzero LR; formal training retains
            # the original engine's three percent warmup and three epochs.
            lr_scheduler_type="constant_with_warmup", warmup_steps=0, max_grad_norm=0.3,
            bf16=True, tf32=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            prediction_loss_only=True, seed=0, data_seed=0, dataloader_num_workers=0,
            report_to=[], disable_tqdm=True, logging_steps=1, label_names=["labels"],
        )
        trainer = engine.Trainer(
            model=model, args=args, processing_class=tokenizer, train_dataset=batch,
            eval_dataset=[max(engine.features(name, "development"), key=lambda row: len(row["input_ids"]))],
            data_collator=engine.DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        )
        parameter = next(value for key, value in model.named_parameters()
                         if value.requires_grad and "lora_B" in key)
        before = parameter.detach().clone()
        result = trainer.train()
        assert trainer.state.global_step == 2 and math.isfinite(result.metrics["train_loss"])
        assert not torch.equal(before, parameter), "Trainer did not change LoRA weights"
        checkpoint = Path(get_last_checkpoint(temporary))
        assert (checkpoint / "optimizer.pt").is_file()
        assert (checkpoint / "adapter_model.safetensors").is_file()
        assert math.isfinite(trainer.state.best_metric)
        record = {
            "status": "passed", "optimizer_updates": 2, "real_training_baseline": False,
            "accumulated_examples": engine.ACCUMULATION, "probe_examples": len(batch),
            "resident_optimizer_states_exercised": True, "longest_sequence_in_every_microbatch": True,
            "sequence_tokens": len(longest["input_ids"]), "checkpoint_step": trainer.state.global_step,
            "checkpoint_has_optimizer_and_adapter": True, "lora_weights_changed": True,
            "probe_warmup_steps": 0, "training_metrics": result.metrics,
            "probe_validation_loss": trainer.state.best_metric,
            "available_before_probe_bytes": available,
            **_memory(torch), "seconds": round(perf_counter() - started, 3),
        }
        del trainer, model, parameter, before
        gc.collect()
        torch.cuda.empty_cache()
    report = {"protocol_fingerprint": spec["fingerprint"],
              "probe_script_sha256": source_hash, "models": {name: record}}
    engine.write_json(engine.OUT / "trainer_preflight.json", report)
    print(name, record, flush=True)
    return report
