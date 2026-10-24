"""Re-evaluate frozen 4B SFT epochs on full train/dev with identical response CE."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import math
import os
import time

import ecqa_study as base
import gpu_safety
import render_results
import sft_ecqa as sft

LOCAL = base.ROOT / "local/ecqa_ce_v1"
OUT = base.OUT / "ce_audit"
NAME = "qwen35_4b"


def status(stage, **fields):
    base.write(LOCAL / "worker_status.json", {"stage": stage, "pid": os.getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(), **fields})


def summarize(records):
    assert records and len({r["id"] for r in records}) == len(records)
    assert all(math.isfinite(r["ce"]) and r["ce"] >= 0 and r["response_tokens"] > 0 for r in records)
    return {"n": len(records), "response_ce": math.fsum(r["ce"] for r in records)/len(records),
            "supervised_tokens": sum(r["response_tokens"] for r in records),
            "seconds": math.fsum(r["seconds"] for r in records)}


def prepare():
    rows, data = base.prepare()
    training = base.read(sft.OUT / f"{NAME}_training_protocol.json")
    completed = base.read(sft.OUT / "trials" / f"{NAME}_lr5e-5_training.json")
    assert completed["status"] == "completed" and completed["epochs"] == 3 and completed["global_step"] == 1419
    assert completed["fingerprint"] == training["fingerprint"]
    for key, path in (("script_sha256", sft.__file__), ("evaluator_sha256", base.__file__),
                      ("gpu_safety_sha256", gpu_safety.__file__)):
        assert training[key] == base.digest(path)
    epochs = [base.read(sft.OUT / "trials" / f"{NAME}_lr5e-5_epoch{ep}.json") for ep in (1, 2, 3)]
    for ep, item in enumerate(epochs, 1):
        assert item["epoch"] == ep and item["fingerprint"] == training["fingerprint"]
        assert base.digest(base.ROOT / item["adapter_path"] / "adapter_model.safetensors") == item["adapter_sha256"]
    cap = base.read(base.OUT / "generation_limit.json")["selected_max_new_tokens"]
    assert cap == training["response_limit_during_evaluation"] == 256
    spec = {"model": training["model"], "revision": training["revision"],
        "training_fingerprint": training["fingerprint"], "dataset_fingerprint": data["fingerprint"],
        "epochs": [{k: row[k] for k in ("epoch", "adapter_path", "adapter_sha256")} for row in epochs],
        "counts": {split: len(rows[split]) for split in ("train", "dev")},
        "ordered_ids_sha256": {split: base.fingerprint([r["id"] for r in rows[split]]) for split in ("train", "dev")},
        "loss": "Same sft_ecqa.response_loss: full-vocabulary teacher-forced response CE, including ANSWER, explanation and EOS; prompt masked",
        "aggregation": "Arithmetic mean of per-example mean response-token CE, identical weighting for train/dev",
        "model_mode": "eval; inference_mode; no gradient or optimizer; batch size 1",
        "dtype": "BF16 base with original FP32 LoRA", "truncation": False,
        "prompt_output_limit": cap, "generation": False, "test_used": False, "selection_influence": False,
        "resource_safety": gpu_safety.LIMITS,
        "script_sha256": base.digest(__file__), "training_script_sha256": base.digest(sft.__file__),
        "evaluator_sha256": base.digest(base.__file__), "gpu_safety_sha256": base.digest(gpu_safety.__file__),
        "packages": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "peft")}}
    spec["fingerprint"] = base.fingerprint(spec)
    base.fixed_write(OUT / "protocol.json", spec)
    return rows, spec


def completed_baselines():
    count = 0
    for directory in ("baselines", "sft"):
        for name in base.MODELS:
            path = base.OUT / directory / f"{name}.json"
            if path.exists():
                result = base.read(path)
                assert result["status"] == "completed" and result["overall"]["n"] == 2194
                count += 1
    return count


def evaluate_split(model, features, rows, epoch, split, spec):
    import torch
    path = LOCAL / f"epoch{epoch}_{split}.json"
    saved = base.read(path) if path.exists() else {"fingerprint": spec["fingerprint"], "records": []}
    assert saved["fingerprint"] == spec["fingerprint"]
    records = saved["records"]
    assert len(records) <= len(rows)
    for row, feature, record in zip(rows, features, records):
        assert record["id"] == row["id"]
        assert record["response_tokens"] == sum(x != -100 for x in feature["labels"])
        assert math.isfinite(record["ce"]) and record["ce"] >= 0
    model.eval().requires_grad_(False)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    with torch.inference_mode():
        for i in range(len(records), len(rows)):
            guard = gpu_safety.stopping_guard()
            started = time.monotonic()
            inputs = {k: torch.tensor([v], device="cuda") for k, v in features[i].items()}
            loss, outputs = sft.response_loss(model, inputs)
            value = float(loss)
            guard.check()
            records.append({"id": rows[i]["id"], "ce": value,
                "response_tokens": sum(x != -100 for x in features[i]["labels"]),
                "seconds": time.monotonic()-started})
            del inputs, loss, outputs
            if len(records) % 50 == 0:
                base.write(path, saved)
                status("evaluating_ce", epoch=epoch, split=split, completed=len(records), total=len(rows))
    base.write(path, saved)
    assert len(records) == len(rows) and not any(p.grad is not None for p in model.parameters())
    result = {"status": "completed", "fingerprint": spec["fingerprint"], "epoch": epoch,
              "split": split, **summarize(records)}
    base.fixed_write(OUT / f"epoch{epoch}_{split}.json", result)
    return result


def run():
    from filelock import FileLock
    from peft import PeftModel
    import torch
    from transformers import AutoTokenizer
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (LOCAL / "RUN_DISABLED.json").exists():
        raise RuntimeError("CE audit safety stop marker exists; inspect before restarting")
    with FileLock(str(LOCAL / "worker.lock"), timeout=0):
        try:
            rows, spec = prepare()
            while completed_baselines() < 6:
                status("waiting_primary_baselines", completed_baselines=completed_baselines(), total_baselines=6)
                time.sleep(30)
            status("waiting_gpu")
            lock = FileLock(str(base.ROOT / "local/gpu.lock"))
            with lock.acquire(timeout=-1, poll_interval=10):
                path = base.snapshots()[NAME]
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
                features = {split: [sft.encode(row, tokenizer, spec["prompt_output_limit"]) for row in rows[split]]
                            for split in ("train", "dev")}
                results = []
                for item in spec["epochs"]:
                    ep = item["epoch"]
                    adapter = base.ROOT / item["adapter_path"]
                    assert base.digest(adapter / "adapter_model.safetensors") == item["adapter_sha256"]
                    status("loading_model", epoch=ep)
                    model = PeftModel.from_pretrained(base.load_model(path), adapter, is_trainable=False)
                    try:
                        pair = {split: evaluate_split(model, features[split], rows[split], ep, split, spec)
                                for split in ("train", "dev")}
                        assert base.digest(adapter / "adapter_model.safetensors") == item["adapter_sha256"]
                        results.append({"epoch": ep, **pair})
                    finally:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                base.write(OUT / "summary.json", {"status": "completed", "fingerprint": spec["fingerprint"],
                    "model": spec["model"], "aggregation": spec["aggregation"], "epochs": results,
                    "selection_influence": False, "completed_utc": datetime.now(timezone.utc).isoformat()})
            status("completed", epochs=3, evaluated_examples=sum(len(rows[s]) for s in ("train", "dev"))*3)
            render_results.render()
        except Exception as error:
            base.write(LOCAL / "RUN_DISABLED.json", {"reason": str(error), "error_type": type(error).__name__,
                "automatic_restart": False, "created_utc": datetime.now(timezone.utc).isoformat()})
            status("failed", reason=str(error), error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    action = parser.parse_args().action
    if action == "run":
        run()
    elif action == "prepare":
        _, protocol = prepare()
        print(protocol["fingerprint"])
    else:
        render_results.render()
