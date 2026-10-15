"""Verify one actual accumulated Trainer update, validation and checkpoint per model."""

import gc
import math
from pathlib import Path
from time import perf_counter
import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from dolly_sft import ACCUMULATION, LOCAL, MODELS, OUT, features, load_model, protocol, sha, tokenizer_for, write_json


def main():
    spec = protocol()
    report = {"protocol_fingerprint": spec["fingerprint"], "probe_script_sha256": sha(__file__), "models": {}}
    for name in MODELS:
        print(f"TRAINER PREFLIGHT {name}", flush=True)
        model, tokenizer = load_model(name), tokenizer_for(name)
        data = features(name, "train")
        longest = max(data, key=lambda x: len(x["input_ids"]))
        # One effective batch exercises accumulation; include the longest accepted sequence.
        batch = data[:ACCUMULATION - 1] + [longest]
        run = LOCAL / "trainer_preflight" / spec["fingerprint"][:12] / name
        if get_last_checkpoint(str(run)) if run.exists() else False:
            raise RuntimeError(f"Existing probe checkpoint; inspect it before overwriting: {run}")
        args = TrainingArguments(output_dir=str(run), max_steps=1, learning_rate=2e-4,
                                 per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                 gradient_accumulation_steps=ACCUMULATION, optim="adamw_torch", weight_decay=0,
                                 # A one-step engineering probe needs a nonzero LR; formal training retains 3% warmup.
                                 lr_scheduler_type="constant_with_warmup", warmup_steps=0, max_grad_norm=0.3,
                                 bf16=True, tf32=False, gradient_checkpointing=True,
                                 gradient_checkpointing_kwargs={"use_reentrant": False},
                                 eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                                 load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                                 prediction_loss_only=True, seed=0, data_seed=0, dataloader_num_workers=0,
                                 report_to=[], disable_tqdm=True, logging_steps=1, label_names=["labels"])
        trainer = Trainer(model=model, args=args, processing_class=tokenizer, train_dataset=batch,
                          eval_dataset=[max(features(name, "development"), key=lambda x: len(x["input_ids"]))],
                          data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = perf_counter()
        parameter = next(p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)
        before = parameter.detach().clone()
        result = trainer.train()
        assert trainer.state.global_step == 1 and math.isfinite(result.metrics["train_loss"])
        assert not torch.equal(before, parameter), "Trainer did not change LoRA weights"
        checkpoint = Path(get_last_checkpoint(str(run)))
        assert (checkpoint / "optimizer.pt").is_file() and (checkpoint / "adapter_model.safetensors").is_file()
        assert math.isfinite(trainer.state.best_metric)
        report["models"][name] = {"status": "passed", "optimizer_updates": 1,
                                  "real_training_baseline": False, "accumulated_examples": len(batch),
                                  "checkpoint_has_optimizer_and_adapter": True,
                                  "lora_weights_changed": True, "probe_warmup_steps": 0,
                                  "training_metrics": result.metrics, "probe_validation_loss": trainer.state.best_metric,
                                  "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                                  "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
                                  "seconds": round(perf_counter() - started, 3)}
        write_json(OUT / "trainer_preflight.json", report)
        print(name, report["models"][name], flush=True)
        del trainer, model, parameter, before; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
