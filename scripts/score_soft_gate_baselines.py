"""Score original and SFT baselines in the existing isolated metric environment."""
import argparse
import csv
import importlib.metadata
import json
import statistics

from filelock import FileLock
from prepare_supplementary_metrics import ROOT, LOCAL as ASSET_LOCAL, read, sha, write_json
import evaluate_supplementary_metrics as existing
import supplementary_metrics as metrics

OUT = ROOT / "results/soft_gate_gradient_v1"
LOCAL = ROOT / "tmp/soft_gate_gradient_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("original", "sft"), default="original")
    phase = parser.parse_args().phase
    import nltk
    import torch
    from bert_score import BERTScorer
    from rouge_score import rouge_scorer

    with FileLock(str(ROOT / "tmp/dolly_sft_v1/worker.lock"), timeout=-1):
        assets = read(ROOT / "results/supplementary_metrics_v1/assets.json")
        assert assets["fingerprint"] == existing.fingerprint(assets)
        assert all(importlib.metadata.version(k) == v for k, v in assets["versions"].items())
        for relative, item in assets["files"].items():
            assert sha(ROOT / relative) == item["sha256"]
        assert assets["metric_source_sha256"]["helpers"] == sha(metrics.__file__)
        nltk.data.path.insert(0, str(ASSET_LOCAL / "nltk_data"))
        spec = read(OUT / "preparation.json")
        rows = existing.records(LOCAL / "prepared/evaluation.jsonl")
        excluded = set(read(OUT / "data_quality.json")["excluded_evaluation_ids"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        scorer = BERTScorer(model_type=str(ROOT / assets["model"]["snapshot"]), num_layers=17,
            lang="en", idf=False, rescale_with_baseline=False, use_fast_tokenizer=False,
            device="cuda", batch_size=existing.BATCH_SIZE)
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        names = ("gpt2_base", "gpt2_xl") if phase == "original" else ("teacher_sft", "student_sft", "student_init")
        for name in names:
            folder = OUT / phase / name
            identity = spec if phase == "original" else read(folder / "evaluation_protocol.json")
            if phase == "sft":
                assert identity["parent_preparation_fingerprint"] == spec["fingerprint"]
            completion = read(folder / "completion.json")
            assert completion["fingerprint"] == identity["fingerprint"]
            assert completion["seeds"] == spec["generation"]["seeds"]
            assert len(completion["raw_responses"]) == len(spec["generation"]["seeds"])
            results = []
            for seed, item in zip(spec["generation"]["seeds"], completion["raw_responses"]):
                assert sha(ROOT / item["path"]) == item["sha256"]
                destination = folder / f"seed{seed}_metrics.json"
                if destination.exists():
                    saved = read(destination)
                    assert saved["fingerprint"] == identity["fingerprint"] and saved["assets_fingerprint"] == assets["fingerprint"]
                    assert saved["producer_sha256"] == sha(__file__)
                    assert sha(ROOT / saved["per_item_csv"]) == saved["per_item_csv_sha256"]
                    results.append(saved)
                    continue
                generated = read(ROOT / item["path"])["responses"]
                scored = existing.score_outputs(scorer, rows, generated)
                for row, output, values in zip(rows, generated, scored):
                    score = rouge.score(row["references"][0], output["response"])["rougeL"]
                    values.update(rougeL_precision=100*score.precision, rougeL_recall=100*score.recall,
                                  rougeL_f1=100*score.fmeasure, generated_tokens=output["generated_tokens"],
                                  empty=int(not output["response"].strip()), length_capped=int(output["length_capped"]))
                columns = ["rougeL_precision", "rougeL_recall", "rougeL_f1", *existing.MEANS,
                           "generated_tokens", "empty", "length_capped"]
                subsets = {"full": scored, "instruction_disjoint": [r for r in scored if r["id"] not in excluded]}
                summary = {key: {"n": len(items), **{c: statistics.mean(r[c] for r in items) for c in columns}}
                           for key, items in subsets.items()}
                csv_path = destination.with_suffix(".csv")
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
                    writer.writeheader()
                    writer.writerows(scored)
                saved = {"status": "completed", "fingerprint": identity["fingerprint"], "assets_fingerprint": assets["fingerprint"],
                    "seed": seed, "metrics": summary, "distinct4": metrics.distinct4([r["response"] for r in generated]),
                    "bert_length_audit": existing.length_audit(scored), "per_item_csv": csv_path.relative_to(ROOT).as_posix(),
                    "per_item_csv_sha256": sha(csv_path), "producer_sha256": sha(__file__)}
                write_json(destination, saved)
                results.append(saved)
                print("COMPLETED_NEW_BASELINE_METRICS", name, seed, flush=True)
            summary = {key: {metric: {"mean": statistics.mean(r["metrics"][key][metric] for r in results),
                "sample_sd_across_generation_seeds": statistics.stdev(r["metrics"][key][metric] for r in results)}
                for metric in results[0]["metrics"][key] if metric != "n"} for key in ("full", "instruction_disjoint")}
            write_json(folder / "summary.json", {"status": "completed", "model": name, "phase": phase,
                "fingerprint": identity["fingerprint"], "generation_seeds": spec["generation"]["seeds"],
                "metrics": summary,
                "distinct4_mean": statistics.mean(r["distinct4"]["distinct4"] for r in results)
                    if all(r["distinct4"]["distinct4"] is not None for r in results) else None,
                "uncertainty_scope": "generation sampling only; no independent training repeats"})


if __name__ == "__main__":
    main()
