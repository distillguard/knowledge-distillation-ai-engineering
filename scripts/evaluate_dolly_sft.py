"""Evaluate each newly trained adapter on the fixed model-selection records."""

import argparse
from collections import Counter
import gc
import inspect
import json
import math
from pathlib import Path
import statistics
import sys

from dolly_sft import HUB, LOCAL, OUT, ROOT, rows, sha, write_json
from dolly_study import MODELS, generation_eos_ids, load_model, model_protocol, render_prompt, selected_record, tokenizer_for
import torch
import pyarrow.parquet as pq
# These packages are pure Python; keep the current environment's compiled dependencies first.
sys.path.append(str(ROOT / "tmp/eval_deps"))
from rouge_score import rouge_scorer

SEEDS = (10,)
DATA = {"sst2": ("stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb", "data/validation-00000-of-00001.parquet"),
        "boolq": ("super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", "boolq/validation-00000-of-00001.parquet")}


def classification_metrics(labels, predictions, probabilities):
    matrix = [[sum(y == a and p == b for y, p in zip(labels, predictions)) for b in (0, 1)] for a in (0, 1)]
    tn, fp = matrix[0]; fn, tp = matrix[1]
    recalls = [tn / (tn + fp), tp / (tp + fn)]
    f1s = [2 * tn / (2 * tn + fn + fp) if tn else 0, 2 * tp / (2 * tp + fp + fn) if tp else 0]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"n": len(labels), "accuracy": (tn + tp) / len(labels), "macro_f1": sum(f1s) / 2,
            "balanced_accuracy": sum(recalls) / 2, "class_recalls": recalls,
            "weighted_f1": (f1s[0] * (tn + fp) + f1s[1] * (tp + fn)) / len(labels),
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0,
            "positive_class_conditional_label_brier": statistics.mean((p - y) ** 2 for p, y in zip(probabilities, labels)),
            "majority_class_accuracy": max(Counter(labels).values()) / len(labels),
            "confusion_matrix_true_rows_predicted_columns": matrix}


def classification_data(tokenizers):
    output, manifest = {}, {}
    frozen_rows = json.loads((OUT / "evaluation_preflight.json").read_text(encoding="utf-8"))["classification"]
    for dataset, (repo, revision, filename) in DATA.items():
        path = HUB / ("datasets--" + repo.replace("/", "--")) / "snapshots" / revision / filename
        source = pq.read_table(path).to_pylist()
        kept, excluded = [], []
        words = ("negative", "positive") if dataset == "sst2" else ("No", "Yes")
        for index, row in enumerate(source):
            if index in frozen_rows[dataset]["excluded_source_rows"]:
                excluded.append(index)
                continue
            instruction, context = (("Determine the sentiment of the input sentence. Please respond as positive or negative.", row["sentence"])
                                    if dataset == "sst2" else (f'Read the input passage and answer the question: {row["question"].rstrip(" ?")}? Your answer should be “Yes” or “No”.', row["passage"]))
            item = {"source_row": index, "label": row["label"], "models": {}}
            for name, tokenizer in tokenizers.items():
                prompt = render_prompt(name, tokenizer, instruction, context)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                targets = [tokenizer.encode(w, add_special_tokens=False) for w in words]
                assert all(targets)
                assert len(ids) + max(map(len, targets)) - 1 <= tokenizer.model_max_length, (dataset, index, name, "full shared input exceeds native context")
                item["models"][name] = {"prompt": ids, "targets": targets}
            kept.append(item)
        assert len(kept) + len(excluded) == len(source)
        assert len(kept) == frozen_rows[dataset]["included_rows"]
        output[dataset] = kept
        manifest[dataset] = {"repo": repo, "revision": revision, "sha256": sha(path),
                             "included_rows": len(kept), "excluded_source_rows": excluded,
                             "context_limit_basis": "fixed original four-tokenizer audit: SST-2 872 and BoolQ 3269; added models keep the same rows and full input within their native context",
                             "decision": "conditional label-sequence log probability; not generation answer parsing"}
    return output, manifest


def last_logits(model, input_ids, count=1):
    kwargs = {"logits_to_keep": count} if "logits_to_keep" in inspect.signature(model.get_base_model().forward).parameters else {}
    values = torch.tensor([input_ids], device="cuda")
    return model(input_ids=values, attention_mask=torch.ones_like(values), use_cache=False, **kwargs).logits[0, -count:].float()


def classify(name, model, data):
    result = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for dataset, records in data.items():
            labels, predictions, probabilities = [], [], []
            for index, row in enumerate(records):
                value = row["models"][name]; targets = value["targets"]
                if all(len(t) == 1 for t in targets):
                    logp = last_logits(model, value["prompt"]).log_softmax(-1)[0]
                    scores = torch.stack([logp[t[0]] for t in targets])
                else:
                    scores = torch.stack([last_logits(model, value["prompt"] + target[:-1], len(target)).log_softmax(-1)
                                          .gather(1, torch.tensor(target, device="cuda")[:, None]).sum() for target in targets])
                conditional = scores.softmax(-1)
                labels.append(row["label"]); predictions.append(int(conditional.argmax())); probabilities.append(float(conditional[1]))
                if (index + 1) % 250 == 0:
                    print(f"EVAL {name} {dataset}: {index + 1}/{len(records)}", flush=True)
            result[dataset] = classification_metrics(labels, predictions, probabilities)
            write_json(LOCAL / "evaluation" / f"{name}_{dataset}_predictions.json",
                       {"source_rows": [r["source_row"] for r in records], "labels": labels,
                        "predictions": predictions, "positive_conditional_probability": probabilities})
    return result


def summarize_generation(summaries):
    assert summaries
    return {metric: {"mean": statistics.mean(r[metric] for r in summaries),
                     "sample_sd_across_generation_seeds": statistics.stdev(r[metric] for r in summaries) if len(summaries) > 1 else None}
            for metric in ("rouge1", "rouge2", "rougeL", "length_cap_rate", "empty_rate", "mean_generated_tokens")}


def generate(name, model, tokenizer, selected, fingerprint):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    eos_ids = generation_eos_ids(name, tokenizer)
    summaries = []
    for seed in SEEDS:
        destination = OUT / f"{name}_generation_seed{seed}.json"
        response_path = LOCAL / "evaluation" / f"{name}_seed{seed}_responses.json"
        if destination.exists():
            previous = json.loads(destination.read_text(encoding="utf-8"))
            assert previous["fingerprint"] == fingerprint
            summaries.append(previous); continue
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        totals, generated, cap, empty = Counter(), [], 0, 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, row in enumerate(selected):
                prompt = render_prompt(name, tokenizer, row["instruction"], row["input"])
                ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                output = model.generate(**ids, max_new_tokens=256, do_sample=True, temperature=1, top_k=0, top_p=1,
                                        no_repeat_ngram_size=6, use_cache=True, eos_token_id=eos_ids,
                                        pad_token_id=tokenizer.eos_token_id)
                continuation = output[0, ids.input_ids.shape[1]:]
                text = tokenizer.decode(continuation, skip_special_tokens=True)
                generated.append({"id": row["id"], "response": text, "generated_tokens": len(continuation)})
                for metric, score in scorer.score(row["output"], text).items():
                    totals[metric] += 100 * score.fmeasure
                cap += int(len(continuation) == 256 and int(continuation[-1]) not in eos_ids)
                empty += int(not text.strip())
                if (index + 1) % 50 == 0:
                    print(f"GENERATE {name} seed={seed}: {index + 1}/{len(selected)}", flush=True)
                    write_json(response_path.with_name(response_path.stem + "_progress.json"), generated)
        write_json(response_path, generated)
        summary = {"fingerprint": fingerprint, "seed": seed, "n": len(selected),
                   **{metric: total / len(selected) for metric, total in totals.items()},
                   "length_cap_rate": cap / len(selected), "empty_rate": empty / len(selected),
                   "mean_generated_tokens": statistics.mean(x["generated_tokens"] for x in generated)}
        write_json(destination, summary); summaries.append(summary)
    return summarize_generation(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    args = parser.parse_args()
    test = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], [.1] * 4)
    assert test["accuracy"] == .5 and math.isclose(test["macro_f1"], 1 / 3) and test["balanced_accuracy"] == .5
    assert test["mcc"] == 0 and test["confusion_matrix_true_rows_predicted_columns"] == [[2, 0], [2, 0]]
    one_seed = summarize_generation([dict.fromkeys(("rouge1", "rouge2", "rougeL", "length_cap_rate", "empty_rate", "mean_generated_tokens"), 1.0)])
    assert all(score["mean"] == 1.0 and score["sample_sd_across_generation_seeds"] is None for score in one_seed.values())
    selected = rows(LOCAL / "selection_evaluation.jsonl")
    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        training = model_protocol(name)
        tokenizer = tokenizer_for(name)
        data, manifest = classification_data({name: tokenizer})
        specification = {"training_protocol_fingerprint": training["fingerprint"], "evaluation_script_sha256": sha(__file__),
                     "classification": manifest, "generation": {"seeds": SEEDS, "max_new_tokens": 256,
                     "phase": "model_selection", "seed_basis": "User decision: one common seed, 10 (first MiniLLM seed); five seeds reserved for the selected-model main experiment",
                     "eos_token_ids": generation_eos_ids(name, tokenizer),
                     "temperature": 1, "top_k": 0, "top_p": 1, "no_repeat_ngram_size": 6,
                     "microbatch": 1, "basis": "MiniLLM sampling settings and seeds; local changes: microbatch 1 and fixed new-token cap 256, originally 512 minus the first four tokenizers' common prompt cap 256. Added tokenizers keep complete identical text, so their prompts may exceed 256. This is not the original variable 512-minus-padded-prompt response budget",
                     "evaluation_is_independent_final_test": False}}
        chosen = selected_record(name)
        assert chosen["protocol_fingerprint"] == training["fingerprint"]
        fingerprint = sha(__file__) + training["fingerprint"] + sha(ROOT / chosen["adapter"] / "adapter_model.safetensors")
        destination = OUT / f"{name}_evaluation.json"
        if destination.exists():
            assert json.loads(destination.read_text(encoding="utf-8"))["fingerprint"] == fingerprint
            continue
        write_json(OUT / f"{name}_evaluation_protocol.json", {**specification, "selected_adapter": chosen, "fingerprint": fingerprint})
        model = load_model(name, ROOT / chosen["adapter"])
        model.eval().requires_grad_(False); model.gradient_checkpointing_disable()
        result = {"status": "completed", "fingerprint": fingerprint,
                  "classification": classify(name, model, data),
                  "dolly_generation": generate(name, model, tokenizer, selected, fingerprint)}
        write_json(destination, result)
        del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
