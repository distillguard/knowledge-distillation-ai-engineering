"""Evaluate selected SFT adapters on four generation benchmarks before classification."""
import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import string

import torch
import dolly_study  # Make the existing local SentencePiece dependencies visible before Transformers imports.
import evaluate_dolly_sft as legacy
from dolly_sft import ROOT, rows, sha, write_json
from dolly_study import MODELS, generation_eos_ids, load_model, model_protocol, read, render_prompt, selected_record, tokenizer_for
from prepare_generation_benchmarks import BENCHMARKS, LOCAL, NEW_TOKENS, OUT, prepared_protocol


def identity(name, data):
    chosen, training = selected_record(name), model_protocol(name)
    assert chosen["protocol_fingerprint"] == training["fingerprint"] and legacy.SEEDS == (10,)
    spec = {"data_protocol_fingerprint": data["fingerprint"], "training_protocol_fingerprint": training["fingerprint"],
            "selected_adapter": chosen, "adapter_sha256": sha(ROOT / chosen["adapter"] / "adapter_model.safetensors"),
            "evaluation_script_sha256": sha(__file__), "dolly_evaluation_script_sha256": sha(legacy.__file__),
            "generation": {"seed": 10, "max_new_tokens": NEW_TOKENS, "temperature": 1, "top_k": 0, "top_p": 1,
                           "no_repeat_ngram_size": 6, "microbatch": 1, "sample_sd_across_generation_seeds": None}}
    spec["fingerprint"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


def load_completed(name, data):
    spec, completed = identity(name, data), {}
    for key in BENCHMARKS:
        path = OUT / name / f"{key}.json"
        if not path.exists():
            continue
        record = read(path)
        assert record["status"] == "completed" and record["fingerprint"] == spec["fingerprint"]
        assert record["n"] == data["counts"][key]["included"]
        assert record["sample_sd_across_generation_seeds"] is None
        assert sha(ROOT / record["per_item_csv"]) == record["per_item_csv_sha256"]
        with (ROOT / record["per_item_csv"]).open(encoding="utf-8", newline="") as handle:
            per_item = list(csv.DictReader(handle))
        assert [r["id"] for r in per_item] == [r["id"] for r in rows(LOCAL / "prepared" / f"{key}.jsonl")]
        assert math.isclose(statistics.mean(float(r["rougeL"]) for r in per_item), record["metrics"]["rougeL"], abs_tol=1e-12)
        completed[key] = record
    return completed


def score_references(scorer, text, references):
    assert references
    scores = [scorer.score(reference, text) for reference in references]
    normalize = lambda value: " ".join(value.lower().translate(str.maketrans("", "", string.punctuation)).split())
    return {**{metric: 100 * max(score[metric].fmeasure for score in scores) for metric in ("rouge1", "rouge2", "rougeL")},
            "exact_match": 100 * int(any(normalize(text) == normalize(reference) for reference in references))}


def restore_progress(path, fingerprint, records):
    if not path.exists():
        return []
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["fingerprint"] == fingerprint, "Generation resume identity mismatch"
    completed = state["records"]
    assert [r["id"] for r in completed] == [r["id"] for r in records[:len(completed)]]
    assert len(completed) <= len(records)
    torch.set_rng_state(state["rng_cpu"])
    torch.cuda.set_rng_state_all(state["rng_cuda"])
    return completed


def save_progress(path, fingerprint, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"fingerprint": fingerprint, "records": records, "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all()}, temporary)
    temporary.replace(path)


def generate(name, key, model, tokenizer, records, fingerprint):
    path = LOCAL / "evaluation" / name / f"{key}_seed10.pt"
    torch.manual_seed(10); torch.cuda.manual_seed_all(10)
    completed = restore_progress(path, fingerprint, records)
    eos = generation_eos_ids(name, tokenizer)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(len(completed), len(records)):
            row = records[index]
            ids = tokenizer(render_prompt(name, tokenizer, row["instruction"], row["input"]),
                            return_tensors="pt", add_special_tokens=False).to("cuda")
            assert ids.input_ids.shape[1] == row["prompt_tokens"][name]
            output = model.generate(**ids, max_new_tokens=NEW_TOKENS, do_sample=True, temperature=1, top_k=0,
                                    top_p=1, no_repeat_ngram_size=6, use_cache=True,
                                    eos_token_id=eos, pad_token_id=tokenizer.eos_token_id)
            continuation = output[0, ids.input_ids.shape[1]:]
            completed.append({"id": row["id"], "response": tokenizer.decode(continuation, skip_special_tokens=True),
                              "generated_tokens": len(continuation),
                              "length_capped": len(continuation) == NEW_TOKENS and int(continuation[-1]) not in eos})
            if (index + 1) % 50 == 0:
                save_progress(path, fingerprint, completed)
                print(f"GENERATE_PRIMARY {name} {key}: {index + 1}/{len(records)}", flush=True)
    save_progress(path, fingerprint, completed)
    write_json(path.with_suffix(".json"), completed)
    return completed


def publish(name, key, records, generated, spec, legacy_cap_rate=None):
    assert [r["id"] for r in records] == [r["id"] for r in generated] and records
    scorer = legacy.rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_item = [{"id": row["id"], "category": row["category"], **score_references(scorer, output["response"], row["references"]),
                 "generated_tokens": output["generated_tokens"], "empty": int(not output["response"].strip()),
                 "length_capped": int(output["length_capped"]) if output["length_capped"] is not None else None}
                for row, output in zip(records, generated)]
    path = OUT / name / f"{key}_per_item.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_item[0])); writer.writeheader(); writer.writerows(per_item)
    path.with_suffix(".tmp").replace(path)
    means = {metric: statistics.mean(row[metric] for row in per_item) for metric in
             ("rouge1", "rouge2", "rougeL", "exact_match", "generated_tokens", "empty")}
    means["length_capped"] = legacy_cap_rate if legacy_cap_rate is not None else statistics.mean(row["length_capped"] for row in per_item)
    assert all(math.isfinite(value) for value in means.values())
    write_json(OUT / name / f"{key}.json", {"status": "completed", "fingerprint": spec["fingerprint"], "n": len(records),
               "seed": 10, "metrics": means, "sample_sd_across_generation_seeds": None,
               "per_item_csv": path.relative_to(ROOT).as_posix(), "per_item_csv_sha256": sha(path)})


def evaluate(name):
    data = prepared_protocol()
    spec = identity(name, data)
    done = load_completed(name, data)
    if len(done) == len(BENCHMARKS):
        print(f"Verified complete generation evaluation: {name}", flush=True)
        return
    write_json(OUT / name / "protocol.json", spec)
    tokenizer = tokenizer_for(name)
    model = load_model(name, ROOT / spec["selected_adapter"]["adapter"])
    model.eval().requires_grad_(False); model.gradient_checkpointing_disable()
    for key in BENCHMARKS:
        if key in done:
            continue
        records = rows(LOCAL / "prepared" / f"{key}.jsonl")
        cap_rate = None
        if key == "dolly":
            original = rows(legacy.LOCAL / "selection_evaluation.jsonl")
            legacy_fp = sha(legacy.__file__) + spec["training_protocol_fingerprint"] + spec["adapter_sha256"]
            legacy.generate(name, model, tokenizer, original, legacy_fp)
            generated = read(legacy.LOCAL / "evaluation" / f"{name}_seed10_responses.json")
            # Legacy raw responses lack per-item EOS flags; retain its measured aggregate cap rate.
            for record in generated:
                record["length_capped"] = None
            cap_rate = read(legacy.OUT / f"{name}_generation_seed10.json")["length_cap_rate"]
        else:
            generated = generate(name, key, model, tokenizer, records, spec["fingerprint"] + ":" + key)
        publish(name, key, records, generated, spec, cap_rate)
        print(f"Completed primary generation benchmark: {name} {key}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    evaluate(parser.parse_args().model)
