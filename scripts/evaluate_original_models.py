"""Evaluate pinned original weights under the matched SFT inference conditions."""
import argparse
import csv
import gc
import hashlib
import json
import math
import statistics

import dolly_study as study
import evaluate_generation_benchmarks as evaluation
from dolly_scope import MODELS
from dolly_sft import LOCAL as SFT_LOCAL, ROOT, rows, sha, write_json
from filelock import FileLock
import torch

OUT = ROOT / "results/original_model_evaluation_v1"
LOCAL = ROOT / "tmp/original_model_evaluation_v1"


def identity(name, data):
    training = study.model_protocol(name)
    spec = {"phase": "before_our_dolly_sft", "model": training["models"][name],
            "precision": training["precision"][name], "adapters_disabled": True,
            "matched_training_protocol_fingerprint": training["fingerprint"],
            "data_protocol_fingerprint": data["fingerprint"],
            "source_script_sha256": sha(__file__),
            "generation_script_sha256": sha(evaluation.__file__),
            "dolly_generation_script_sha256": sha(evaluation.legacy.__file__),
            "versions": training["versions"],
            "generation": {"seed": 10, "max_new_tokens": 256, "temperature": 1,
                           "top_k": 0, "top_p": 1, "no_repeat_ngram_size": 6, "microbatch": 1,
                           "sample_sd_across_generation_seeds": None}}
    spec["fingerprint"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


def load_completed(name, key, spec, records):
    path = OUT / name / f"{key}.json"
    if not path.exists():
        return None
    record = study.read(path)
    assert record["status"] == "completed" and record["fingerprint"] == spec["fingerprint"]
    assert record["phase"] == "before_our_dolly_sft" and record["n"] == len(records)
    assert record["seed"] == 10 and record["sample_sd_across_generation_seeds"] is None
    csv_path = ROOT / record["per_item_csv"]
    assert sha(csv_path) == record["per_item_csv_sha256"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        items = list(csv.DictReader(handle))
    assert [r["id"] for r in items] == [r["id"] for r in records]
    for metric in ("rouge1", "rouge2", "rougeL", "exact_match", "generated_tokens", "empty"):
        assert math.isclose(statistics.mean(float(r[metric]) for r in items),
                            record["metrics"][metric], rel_tol=0, abs_tol=1e-12)
    raw = ROOT / record["raw_responses"]
    assert raw.exists() and sha(raw) == record["raw_responses_sha256"], "Original response cache missing or changed"
    return record


def generate(name, key, model, tokenizer, records, spec):
    if key == "dolly":
        legacy = evaluation.legacy
        previous = legacy.OUT, legacy.LOCAL
        try:
            legacy.OUT, legacy.LOCAL = LOCAL / name / "legacy", LOCAL / name
            original = rows(SFT_LOCAL / "selection_evaluation.jsonl")
            assert [r["id"] for r in original] == [r["id"] for r in records]
            legacy.generate(name, model, tokenizer, original, spec["fingerprint"])
            raw = legacy.LOCAL / "evaluation" / f"{name}_seed10_responses.json"
            generated = study.read(raw)
            for response in generated:
                response["length_capped"] = None
            cap_rate = study.read(legacy.OUT / f"{name}_generation_seed10.json")["length_cap_rate"]
        finally:
            legacy.OUT, legacy.LOCAL = previous
    else:
        previous = evaluation.LOCAL
        try:
            evaluation.LOCAL = LOCAL
            generated = evaluation.generate(name, key, model, tokenizer, records, spec["fingerprint"] + ":" + key)
            raw = LOCAL / "evaluation" / name / f"{key}_seed10.json"
            cap_rate = None
        finally:
            evaluation.LOCAL = previous
    previous = evaluation.OUT
    try:
        evaluation.OUT = LOCAL / "published"
        evaluation.publish(name, key, records, generated, spec, cap_rate)
        published = study.read(evaluation.OUT / name / f"{key}.json")
    finally:
        evaluation.OUT = previous
    path = OUT / name / f"{key}.json"
    csv_path = OUT / name / f"{key}_per_item.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / published["per_item_csv"]).replace(csv_path)
    write_json(path, {**published, "phase": spec["phase"],
                      "per_item_csv": csv_path.relative_to(ROOT).as_posix(),
                      "raw_responses": raw.relative_to(ROOT).as_posix(), "raw_responses_sha256": sha(raw)})
    load_completed(name, key, spec, records)


def classify(name, model, tokenizer, spec):
    legacy = evaluation.legacy
    data, manifest = legacy.classification_data({name: tokenizer})
    details = {"original_model_fingerprint": spec["fingerprint"], "classification": manifest}
    fingerprint = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    path = OUT / name / "classification.json"
    if path.exists():
        result = study.read(path)
        assert result["status"] == "completed" and result["fingerprint"] == fingerprint
    else:
        write_json(OUT / name / "classification_protocol.json", {**details, "fingerprint": fingerprint})
        previous = legacy.LOCAL
        try:
            legacy.LOCAL = LOCAL / name
            scores = legacy.classify(name, model, data)
        finally:
            legacy.LOCAL = previous
        files = {key: LOCAL / name / "evaluation" / f"{name}_{key}_predictions.json" for key in data}
        result = {"status": "completed", "phase": spec["phase"], "fingerprint": fingerprint,
                  "classification": scores,
                  "predictions": {key: {"path": value.relative_to(ROOT).as_posix(), "sha256": sha(value)}
                                  for key, value in files.items()}}
    assert result["phase"] == spec["phase"] and set(result["classification"]) == set(data)
    for key, records in data.items():
        saved = result["predictions"][key]
        assert sha(ROOT / saved["path"]) == saved["sha256"]
        predictions = study.read(ROOT / saved["path"])
        assert predictions["source_rows"] == [r["source_row"] for r in records]
        assert predictions["labels"] == [r["label"] for r in records]
        probabilities = predictions["positive_conditional_probability"]
        assert len(probabilities) == len(predictions["predictions"]) == len(records)
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        assert all(p in (0, 1) for p in predictions["predictions"])
        assert legacy.classification_metrics(predictions["labels"], predictions["predictions"], probabilities) == result["classification"][key]
    if not path.exists():
        write_json(path, result)


def evaluate(name, keys, classification=False):
    data = evaluation.prepared_protocol()
    spec = identity(name, data)
    protocol_path = OUT / name / "protocol.json"
    if protocol_path.exists():
        assert study.read(protocol_path) == spec, "Original evaluation identity changed"
    records = {key: rows(evaluation.LOCAL / "prepared" / f"{key}.jsonl") for key in keys}
    pending = [key for key in keys if load_completed(name, key, spec, records[key]) is None]
    if not pending and not classification:
        print(f"Verified complete original evaluation: {name}", flush=True)
        return
    write_json(protocol_path, spec)
    # The loader reads only pinned base weights; disabling the disposable adapter excludes SFT updates.
    model = study.load_model(name)
    tokenizer = study.tokenizer_for(name)
    model.eval().requires_grad_(False)
    model.gradient_checkpointing_disable()
    layers = []
    try:
        from peft.tuners.lora import LoraLayer
        layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
        assert layers and not any(parameter.requires_grad for parameter in model.parameters())
        with model.disable_adapter():
            assert all(layer.disable_adapters for layer in layers)
            for key in pending:
                generate(name, key, model, tokenizer, records[key], spec)
                print(f"Completed original evaluation: {name} {key}", flush=True)
            if classification:
                classify(name, model, tokenizer, spec)
    finally:
        del model, layers
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), required=True)
    parser.add_argument("--benchmark", nargs="+", choices=evaluation.BENCHMARKS, default=["dolly"])
    parser.add_argument("--classification", action="store_true")
    args = parser.parse_args(argv)
    names = MODELS if args.model == "all" else (args.model,)
    with FileLock(str(SFT_LOCAL / "worker.lock"), timeout=0):
        for name in names:
            evaluate(name, args.benchmark, args.classification)


if __name__ == "__main__":
    main()
