"""Score saved original/SFT responses with the pinned, isolated metric stack."""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import statistics
import string
import time

from filelock import FileLock
from prepare_supplementary_metrics import ROOT, LOCAL, OUT, read, sha, write_json
import supplementary_metrics as metrics

BATCH_SIZE = 64  # Official BERTScorer default; also bound the per-call CPU embedding cache.
BENCHMARKS = ("dolly", "self_inst", "sni_11", "vicuna")
MEANS = ("meteor", "bertscore_precision", "bertscore_recall", "bertscore_f1")


def fingerprint(value, excluded=("fingerprint",)):
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k not in excluded}, sort_keys=True).encode()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepared_data():
    data = read(ROOT / "results/generation_benchmarks_v1/protocol.json")
    assert data["fingerprint"] == fingerprint(data) and data["status"] == "passed"
    assert data["preparation_script_sha256"] == sha(ROOT / "scripts/prepare_generation_benchmarks.py")
    assert data["tokenizer_study_script_sha256"] == sha(ROOT / "scripts/dolly_study.py")
    for path, expected in data["files_sha256"].items():
        assert sha(ROOT / path) == expected
    return data


def source(name, phase, key, data, rows):
    from rouge_score import rouge_scorer

    training_dir = ROOT / ("results/dolly_sft_v1" if name.startswith("gpt2_") else f"results/dolly_sft_additional_v1/{name}")
    training = read(training_dir / "protocol.json")
    excluded = ("fingerprint", "prepared_files_sha256") if name.startswith("gpt2_") else ("fingerprint",)
    assert training["fingerprint"] == fingerprint(training, excluded)
    source_script = "dolly_sft.py" if name.startswith("gpt2_") else "dolly_study.py"
    assert training["source_script_sha256"] == sha(ROOT / "scripts" / source_script)
    parent = ROOT / "results" / ("original_model_evaluation_v1" if phase == "original" else "generation_benchmarks_v1") / name
    protocol, result = read(parent / "protocol.json"), read(parent / f"{key}.json")
    assert protocol["fingerprint"] == fingerprint(protocol) == result["fingerprint"]
    assert protocol["data_protocol_fingerprint"] == data["fingerprint"]
    assert result["status"] == "completed" and result["n"] == len(rows) == data["counts"][key]["included"]
    assert result["seed"] == 10 and result["sample_sd_across_generation_seeds"] is None
    if phase == "original":
        assert protocol["phase"] == result["phase"] == "before_our_dolly_sft" and protocol["adapters_disabled"] is True
        assert protocol["model"] == training["models"][name] and protocol["precision"] == training["precision"][name]
        assert protocol["matched_training_protocol_fingerprint"] == training["fingerprint"]
        assert protocol["source_script_sha256"] == sha(ROOT / "scripts/evaluate_original_models.py")
        assert protocol["generation_script_sha256"] == sha(ROOT / "scripts/evaluate_generation_benchmarks.py")
        assert protocol["dolly_generation_script_sha256"] == sha(ROOT / "scripts/evaluate_dolly_sft.py")
        raw_path = ROOT / result["raw_responses"]
        assert raw_path.resolve().is_relative_to(ROOT / "tmp/original_model_evaluation_v1")
        expected_raw = (ROOT / "tmp/original_model_evaluation_v1" / name / "evaluation" / f"{name}_seed10_responses.json" if key == "dolly"
                        else ROOT / "tmp/original_model_evaluation_v1/evaluation" / name / f"{key}_seed10.json")
        assert raw_path.resolve() == expected_raw.resolve()
        assert sha(raw_path) == result["raw_responses_sha256"]
        if key == "dolly":
            legacy = read(ROOT / "tmp/original_model_evaluation_v1" / name / "legacy" / f"{name}_generation_seed10.json")
            assert legacy["fingerprint"] == protocol["fingerprint"] and legacy["seed"] == 10 and legacy["n"] == len(rows)
            assert 0 <= legacy["length_cap_rate"] <= 1 and legacy["length_cap_rate"] == result["metrics"]["length_capped"]
    else:
        assert phase == "sft" and protocol["training_protocol_fingerprint"] == training["fingerprint"]
        chosen = read(training_dir / f"{name}_selected.json")
        assert protocol["selected_adapter"] == chosen and chosen["protocol_fingerprint"] == training["fingerprint"]
        assert sha(ROOT / chosen["adapter"] / "adapter_model.safetensors") == protocol["adapter_sha256"]
        assert protocol["evaluation_script_sha256"] == sha(ROOT / "scripts/evaluate_generation_benchmarks.py")
        assert protocol["dolly_evaluation_script_sha256"] == sha(ROOT / "scripts/evaluate_dolly_sft.py")
        raw_path = (ROOT / "tmp/dolly_sft_v1/evaluation" / f"{name}_seed10_responses.json" if key == "dolly"
                    else ROOT / "tmp/generation_benchmarks_v1/evaluation" / name / f"{key}_seed10.json")
        if key == "dolly":
            legacy = read(ROOT / "results/dolly_sft_v1" / f"{name}_generation_seed10.json")
            assert legacy["fingerprint"] == protocol["dolly_evaluation_script_sha256"] + training["fingerprint"] + protocol["adapter_sha256"]
            assert legacy["length_cap_rate"] == result["metrics"]["length_capped"]
    assert protocol["generation"]["seed"] == 10 and protocol["generation"]["sample_sd_across_generation_seeds"] is None
    assert {k: protocol["generation"][k] for k in ("max_new_tokens", "temperature", "top_k", "top_p", "no_repeat_ngram_size", "microbatch")} == dict(max_new_tokens=256, temperature=1, top_k=0, top_p=1, no_repeat_ngram_size=6, microbatch=1)
    csv_path = ROOT / result["per_item_csv"]
    assert csv_path.resolve().is_relative_to(parent) and sha(csv_path) == result["per_item_csv_sha256"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        items = list(csv.DictReader(handle))
    generated = read(raw_path)
    ids = [row["id"] for row in rows]
    assert ids == [row["id"] for row in generated] == [row["id"] for row in items]
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == data["counts"][key]["included_ids_sha256"]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    normalize = lambda text: " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())
    for row, output, item in zip(rows, generated, items):
        assert row["references"] and all(isinstance(r, str) and r.strip() for r in row["references"])
        assert isinstance(output["response"], str) and item["category"] == row["category"]
        assert isinstance(output["generated_tokens"], int) and 0 <= output["generated_tokens"] <= 256
        scores = [scorer.score(reference, output["response"]) for reference in row["references"]]
        expected = {m: 100 * max(score[m].fmeasure for score in scores) for m in ("rouge1", "rouge2", "rougeL")}
        expected.update(exact_match=100 * int(any(normalize(output["response"]) == normalize(r) for r in row["references"])),
                        generated_tokens=output["generated_tokens"], empty=int(not output["response"].strip()))
        for metric, value in expected.items():
            assert math.isclose(float(item[metric]), value, rel_tol=0, abs_tol=1e-12), (row["id"], metric)
        if key == "dolly":
            assert item["length_capped"] == ""
        else:
            assert int(item["length_capped"]) == int(output["length_capped"])
    for metric in ("rouge1", "rouge2", "rougeL", "exact_match", "generated_tokens", "empty"):
        assert math.isclose(statistics.mean(float(row[metric]) for row in items), result["metrics"][metric], rel_tol=0, abs_tol=1e-12)
    identity = {"generation_fingerprint": protocol["fingerprint"], "raw_responses_sha256": sha(raw_path),
                "primary_csv_sha256": result["per_item_csv_sha256"], "training_protocol_fingerprint": training["fingerprint"]}
    return identity, result, items, generated


def encoded_lengths(tokenizer, text):
    text = text.strip()
    ids = (tokenizer.encode(text, add_special_tokens=True, add_prefix_space=True, truncation=False)
           if text else tokenizer.build_inputs_with_special_tokens([]))
    return len(ids), min(len(ids), tokenizer.model_max_length)


def score_outputs(scorer, rows, generated):
    assert rows and [r["id"] for r in rows] == [r["id"] for r in generated]
    items = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows, batch_generated = rows[start:start + BATCH_SIZE], generated[start:start + BATCH_SIZE]
        p, r, f = scorer.score([r["response"] for r in batch_generated], [r["references"] for r in batch_rows], batch_size=BATCH_SIZE)
        values = [tensor.tolist() for tensor in (p, r, f)]
        assert all(len(v) == len(batch_rows) for v in values)
        for index, (row, output) in enumerate(zip(batch_rows, batch_generated)):
            reference_lengths = [encoded_lengths(scorer._tokenizer, ref) for ref in row["references"]]
            full, retained = encoded_lengths(scorer._tokenizer, output["response"])
            item = {"id": row["id"], "meteor": 100 * metrics.meteor(output["response"], row["references"]),
                    **{metric: 100 * values[column][index] for column, metric in enumerate(MEANS[1:])},
                    "bert_reference_full_tokens": json.dumps([n for n, _ in reference_lengths]),
                    "bert_reference_retained_tokens": json.dumps([n for _, n in reference_lengths]),
                    "bert_reference_truncated_count": sum(a > b for a, b in reference_lengths),
                    "bert_response_full_tokens": full, "bert_response_retained_tokens": retained,
                    "bert_response_truncated": int(full > retained)}
            assert all(math.isfinite(item[metric]) for metric in MEANS)
            items.append(item)
        print(f"SCORE_METRICS {len(items)}/{len(rows)}", flush=True)
    return items


def length_audit(items):
    return {"native_max_tokens_including_special_tokens": 512,
             "reference_count": sum(len(json.loads(row["bert_reference_full_tokens"])) for row in items),
             "truncated_reference_count": sum(int(row["bert_reference_truncated_count"]) for row in items),
             "items_with_truncated_reference": sum(int(row["bert_reference_truncated_count"]) > 0 for row in items),
             "items_with_truncated_response": sum(int(row["bert_response_truncated"]) for row in items),
             "max_reference_full_tokens": max(n for row in items for n in json.loads(row["bert_reference_full_tokens"])),
             "max_response_full_tokens": max(int(row["bert_response_full_tokens"]) for row in items)}


def publish(folder, spec, primary, primary_items, scored, generated):
    assert [r["id"] for r in primary_items] == [r["id"] for r in scored]
    combined = [{**left, **right} for left, right in zip(primary_items, scored)]
    path = folder / "per_item.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0])); writer.writeheader(); writer.writerows(combined)
    path.with_suffix(".tmp").replace(path)
    result = {"status": "completed", "fingerprint": spec["fingerprint"], "model": spec["model"], "phase": spec["phase"],
              "benchmark": spec["benchmark"], "n": len(combined), "seed": 10, "sample_sd_across_generation_seeds": None,
              "metrics": {metric: statistics.mean(row[metric] for row in scored) for metric in MEANS},
              "diversity": metrics.distinct4([row["response"] for row in generated]), "bert_length_audit": length_audit(scored),
              "primary_metrics": primary["metrics"], "per_item_csv": path.relative_to(ROOT).as_posix(), "per_item_csv_sha256": sha(path)}
    write_json(folder / "result.json", result)
    return result


def load_completed(folder, spec, rows, generated):
    path = folder / "result.json"
    if not path.exists():
        return None
    result = read(path)
    assert result["status"] == "completed" and result["fingerprint"] == spec["fingerprint"]
    assert result["model"] == spec["model"] and result["phase"] == spec["phase"] and result["benchmark"] == spec["benchmark"]
    assert result["n"] == len(rows) and result["seed"] == 10 and result["sample_sd_across_generation_seeds"] is None
    csv_path = ROOT / result["per_item_csv"]
    assert sha(csv_path) == result["per_item_csv_sha256"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        items = list(csv.DictReader(handle))
    assert [row["id"] for row in items] == [row["id"] for row in rows]
    for metric in MEANS:
        assert math.isclose(statistics.mean(float(row[metric]) for row in items), result["metrics"][metric], rel_tol=0, abs_tol=1e-12)
    assert result["diversity"] == metrics.distinct4([row["response"] for row in generated])
    assert result["bert_length_audit"] == length_audit(items)
    return result


def main(argv=None):
    names = tuple(read(ROOT / "results/generation_benchmarks_v1/completion.json")["study_models"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*names, "all"), default="all")
    parser.add_argument("--phase", choices=("original", "sft", "all"), default="all")
    parser.add_argument("--benchmark", nargs="+", choices=BENCHMARKS, default=["dolly"])
    args = parser.parse_args(argv)
    names = names if args.model == "all" else (args.model,)
    phases = ("original", "sft") if args.phase == "all" else (args.phase,)
    with FileLock(str(ROOT / "tmp/dolly_sft_v1/worker.lock"), timeout=0):
        assets, data = read(OUT / "assets.json"), prepared_data()
        assert assets["fingerprint"] == fingerprint(assets) and assets["status"] == "prepared"
        for relative, item in assets["files"].items():
            assert sha(ROOT / relative) == item["sha256"] and (ROOT / relative).stat().st_size == item["bytes"]
        assert assets["metric_source_sha256"]["helpers"] == sha(metrics.__file__)
        assert assets["metric_source_sha256"]["prepare"] == sha(ROOT / "scripts/prepare_supplementary_metrics.py")
        assert importlib.metadata.version("rouge-score") == "0.1.2"
        pending = []
        for phase in phases:
            for name in names:
                for key in args.benchmark:
                    rows = records(ROOT / "tmp/generation_benchmarks_v1/prepared" / f"{key}.jsonl")
                    input_identity, primary, primary_items, generated = source(name, phase, key, data, rows)
                    spec = {"model": name, "phase": phase, "benchmark": key, "source": input_identity,
                            "assets_fingerprint": assets["fingerprint"], "data_protocol_fingerprint": data["fingerprint"],
                            "scoring_script_sha256": sha(__file__), "batch_size": BATCH_SIZE, "item_chunk_size": BATCH_SIZE,
                            "encoder_dtype": "float32", "tf32": False,
                            "rouge_score_version": "0.1.2", "per_item_metric_scale": "METEOR and raw BERTScore P/R/F1 multiplied by 100; corpus Distinct-4 is a ratio"}
                    spec["fingerprint"] = fingerprint(spec)
                    folder = OUT / phase / name / key
                    if (folder / "protocol.json").exists():
                        assert read(folder / "protocol.json") == spec, "Scoring identity changed; preserve existing results"
                    if load_completed(folder, spec, rows, generated) is None:
                        pending.append((folder, spec, rows, generated, primary, primary_items))
        if not pending:
            print("Verified complete supplementary scoring; no metric model loaded", flush=True)
            return
        import nltk
        import torch
        from bert_score import BERTScorer
        import bert_score.scorer as scorer_module
        import bert_score.utils as scorer_utils

        assert all(importlib.metadata.version(name) == version for name, version in assets["versions"].items())
        assert sha(scorer_module.__file__) == assets["metric_source_sha256"]["bert_score_scorer"]
        assert sha(scorer_utils.__file__) == assets["metric_source_sha256"]["bert_score_utils"]
        nltk.data.path.insert(0, str(LOCAL / "nltk_data"))
        assert metrics.wordnet.get_version() == assets["wordnet"]["version"]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        scorer = BERTScorer(model_type=str(ROOT / assets["model"]["snapshot"]), num_layers=17, lang="en", idf=False,
                            rescale_with_baseline=False, use_fast_tokenizer=False, device="cuda", batch_size=BATCH_SIZE)
        assert not scorer._tokenizer.is_fast and scorer._tokenizer.model_max_length == 512
        assert len(scorer._model.encoder.layer) == 17 and next(scorer._model.parameters()).dtype == torch.float32
        probe = scorer.score(["The car is red.", ""], [["The car is red."], ["The car is red."]], batch_size=BATCH_SIZE)[2]
        assert math.isclose(float(probe[0]), 1, abs_tol=1e-5) and float(probe[1]) == 0
        for folder, spec, rows, generated, primary, primary_items in pending:
            print(f"BEGIN_METRICS {spec['phase']} {spec['model']} {spec['benchmark']}", flush=True)
            write_json(folder / "protocol.json", spec)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            scored = score_outputs(scorer, rows, generated)
            publish(folder, spec, primary, primary_items, scored, generated)
            load_completed(folder, spec, rows, generated)
            write_json(LOCAL / "runtime" / spec["phase"] / spec["model"] / f"{spec['benchmark']}.json",
                       {"fingerprint": spec["fingerprint"], "seconds": time.monotonic() - started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "bert_score_library_hash": scorer.hash})
            print(f"COMPLETED_METRICS {spec['phase']} {spec['model']} {spec['benchmark']}", flush=True)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
