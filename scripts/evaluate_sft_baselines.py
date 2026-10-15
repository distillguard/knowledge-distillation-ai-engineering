"""Evaluate the authors' two SFT checkpoints; this script does not train models."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import string
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp" / "eval_deps"))
os.environ["HF_HOME"] = str(ROOT / "tmp" / "hf_home")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from rouge_score import rouge_scorer
from safetensors.torch import load_file, load_model
from sklearn import metrics
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

MODELS = {
    "teacher": ("MiniLLM/teacher-gpt2-1.5B", "4a8d07d52ad67b7b87fec9aba25b1521173eb658"),
    "student": ("MiniLLM/SFT-gpt2-120M", "b9d32f7ca7e06832c7fe9e8211b53c732fbae5b4"),
}
DATA = {
    "dolly_raw": ("MiniLLM/dolly", "d131636138be0ff3fd0901c12c1739844d337279", "raw.jsonl"),
    "dolly": ("MiniLLM/dolly", "d131636138be0ff3fd0901c12c1739844d337279", "valid.jsonl"),
    "sst2": ("stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb", "data/validation-00000-of-00001.parquet"),
    "boolq": ("super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", "boolq/validation-00000-of-00001.parquet"),
}
# MiniLLM Appendix B.2 and scripts/gpt2/eval/run_eval.sh at the cited source commit.
SEEDS, BATCH = (10, 20, 30, 40, 50), 8
HEADER = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
OUT = ROOT / "results" / "sft_baselines"
RAW = ROOT / "tmp" / "sft_eval"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize(text):
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def text_f1(prediction, reference):
    p, r = normalize(prediction).split(), normalize(reference).split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    return 2 * common / (len(p) + len(r))


def classification_metrics(labels, predictions, probabilities):
    assert len(labels) == len(predictions) == len(probabilities) and len(labels) > 0
    return {
        "n": len(labels), "accuracy": metrics.accuracy_score(labels, predictions),
        "balanced_accuracy": metrics.balanced_accuracy_score(labels, predictions),
        "macro_f1": metrics.f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0),
        "weighted_f1": metrics.f1_score(labels, predictions, labels=[0, 1], average="weighted", zero_division=0),
        "mcc": metrics.matthews_corrcoef(labels, predictions),
        "positive_class_brier": metrics.brier_score_loss(labels, probabilities),
        "confusion_matrix_true_rows_predicted_columns": metrics.confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "per_class": metrics.classification_report(labels, predictions, labels=[0, 1], output_dict=True, zero_division=0),
        "majority_class_accuracy": max(Counter(labels).values()) / len(labels),
    }


def self_check():
    assert normalize("The CAT,  sat!") == "the cat sat"
    assert text_f1("red red blue", "red blue blue") == 2 / 3
    result = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], [.1] * 4)
    assert result["accuracy"] == .5 and math.isclose(result["macro_f1"], 1 / 3)
    assert result["confusion_matrix_true_rows_predicted_columns"] == [[2, 0], [2, 0]]
    assert SCORER.score("a good answer", "a good answer")["rougeL"].fmeasure == 1


def prepare_data():
    data, manifest = {}, {}
    for name, (repo, revision, filename) in DATA.items():
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                   cache_dir=ROOT / "tmp/hf_home/hub", token=False))
        rows = pq.read_table(path).to_pylist() if filename.endswith(".parquet") else [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        data[name] = rows
        manifest[name] = {"repo": repo, "revision": revision, "file": filename,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
    assert len(data["dolly"]) == 500 and len(data["sst2"]) == 872 and len(data["boolq"]) == 3270
    key = lambda row: " ".join((row["instruction"] + "\n" + row["input"]).lower().split())
    raw_keys = set(map(key, data["dolly_raw"]))
    overlaps = [i for i, row in enumerate(data["dolly"]) if key(row) in raw_keys]
    manifest["dolly_overlap"] = {"definition": "lowercase/whitespace-normalized instruction+input matches published raw.jsonl",
                                  "evaluation_row_indices": overlaps, "n": len(overlaps),
                                  "caveat": "Published raw file is not a complete checkpoint training-membership audit; semantic duplicates not checked"}
    return data, manifest, set(overlaps)


def classify(model, tokenizer, rows, name):
    words = ["negative", "positive"] if name == "sst2" else ["No", "Yes"]
    label_ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]
    assert all(len(ids) == 1 for ids in label_ids), "Need sequence label scoring for multi-token labels"
    candidates, excluded = [], []
    for row in rows:
        if name == "sst2":
            instruction = "Determine the sentiment of the input sentence. Please respond as positive or negative."
            content = row["sentence"]
        else:
            question = row["question"].rstrip(" ?")
            instruction = f'Read the input passage and answer the question: {question}? Your answer should be “Yes” or “No”.'
            content = row["passage"]
        prompt = HEADER + f"### Instruction:\n{instruction}\n\n### Input:\n{content}\n\n### Response:\n"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert all(tokenizer.encode(prompt + word, add_special_tokens=False) == ids + label for word, label in zip(words, label_ids))
        if len(ids) > model.config.n_positions:
            excluded.append(row["idx"])
        else:
            candidates.append((row, ids))
    records = []
    with torch.inference_mode():
        for start in range(0, len(candidates), BATCH):
            part = candidates[start:start + BATCH]
            batch = tokenizer.pad({"input_ids": [ids for _, ids in part]}, padding=True, return_tensors="pt").to("cuda")
            positions = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
            output = model(**batch, position_ids=positions, use_cache=False)
            logits = output.logits[:, -1].float()
            assert torch.isfinite(logits).all()
            selected = logits[:, [ids[0] for ids in label_ids]]
            probabilities = selected.softmax(-1)
            masses = (selected.logsumexp(-1) - logits.logsumexp(-1)).exp()
            for (row, _), probability, mass in zip(part, probabilities.cpu().tolist(), masses.cpu().tolist()):
                records.append({"id": row["idx"], "label": row["label"], "predicted": int(probability[1] > probability[0]),
                                "p_positive_given_labels": probability[1], "raw_label_probability_mass": mass})
            del output, logits
            if start % (BATCH * 50) == 0:
                print(f"{name}: {len(records)}/{len(candidates)}", flush=True)
    result = classification_metrics([r["label"] for r in records], [r["predicted"] for r in records],
                                    [r["p_positive_given_labels"] for r in records])
    result.update({"label_words_0_1": words, "excluded_over_native_context_ids": excluded,
                   "mean_raw_label_probability_mass": float(np.mean([r["raw_label_probability_mass"] for r in records])),
                   "decision": "argmax of two label logits; ties select class 0; Brier uses probabilities conditional on those labels"})
    return result, records


def reference_diagnostics(model, tokenizer, rows):
    sums, counts, correct, means, clipped = 0., 0, 0, [], 0
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            examples = []
            for row in rows[start:start + BATCH]:
                prompt = tokenizer.encode(row["prompt"].replace("<n>", "\n"), add_special_tokens=False)
                response = tokenizer.encode(row["output"], add_special_tokens=False)
                assert 0 < len(prompt) <= 256 and response
                clipped += int(len(response) > 256)
                response = response[:256]
                examples.append((prompt + response, [-100] * len(prompt) + response))
            width = max(len(ids) for ids, _ in examples)
            ids = torch.full((len(examples), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
            labels = torch.full_like(ids, -100)
            mask = torch.zeros_like(ids)
            for index, (values, target) in enumerate(examples):
                ids[index, :len(values)] = torch.tensor(values, device="cuda")
                labels[index, :len(values)] = torch.tensor(target, device="cuda")
                mask[index, :len(values)] = 1
            output = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits, targets = output.logits[:, :-1].float(), labels[:, 1:]
            losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none", ignore_index=-100)
            valid = targets != -100
            sums += losses.sum().item()
            counts += valid.sum().item()
            correct += ((logits.argmax(-1) == targets) & valid).sum().item()
            means.extend((losses.sum(-1) / valid.sum(-1)).cpu().tolist())
            del output, logits, losses
    return {"response_token_nll": sums / counts, "response_token_perplexity": math.exp(sums / counts),
            "teacher_forced_response_token_accuracy": correct / counts, "scored_reference_tokens": counts,
            "mean_example_response_nll": float(np.mean(means)), "references_clipped_at_256_tokens": clipped,
            "eos_scored": False, "note": "Reference prefix only, at most 256 response tokens; not free-generation correctness"}


def generation_metrics(records):
    assert records
    result = {name: float(np.mean([r[name] for r in records])) for name in
              ["rouge1", "rouge2", "rougeL", "normalized_exact_match", "word_overlap_f1", "generated_tokens", "hit_length_cap", "empty"]}
    ngrams = Counter()
    for record in records:
        words = record["text"].split()
        ngrams.update(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    total = sum(ngrams.values())
    result["distinct_4_whitespace"] = len(ngrams) / total if total else None
    result["corpus_4gram_count"] = total
    result["n"] = len(records)
    return result


def generate_seed(model, tokenizer, rows, overlap, seed, role, fingerprint):
    summary_path = OUT / f"{role}_dolly_seed{seed}.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        assert saved["fingerprint"] == fingerprint, "Refusing to reuse results from a different evaluation"
        return saved
    set_seed(seed)
    config = GenerationConfig(do_sample=True, temperature=1., top_k=0, top_p=1., num_beams=1,
                              no_repeat_ngram_size=6, repetition_penalty=1., max_new_tokens=256,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            part = rows[start:start + BATCH]
            tokens = [tokenizer.encode(r["prompt"].replace("<n>", "\n"), add_special_tokens=False) for r in part]
            assert all(0 < len(ids) <= 256 for ids in tokens)
            inputs = tokenizer.pad({"input_ids": tokens}, padding="max_length", max_length=256, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, generation_config=config)[:, 256:].cpu().tolist()
            for index, (row, ids) in enumerate(zip(part, output), start=start):
                stopped = tokenizer.eos_token_id in ids
                if stopped:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(ids, skip_special_tokens=True).replace("<n>", "\n").strip()
                score = SCORER.score(row["output"], text)
                records.append({"row_index": index, "topic": row["topic"], "text": text,
                                **{key: value.fmeasure for key, value in score.items()},
                                "normalized_exact_match": float(normalize(text) == normalize(row["output"])),
                                "word_overlap_f1": text_f1(text, row["output"]), "generated_tokens": len(ids),
                                "hit_length_cap": float(not stopped and len(ids) == 256), "empty": float(not text)})
            if start % (BATCH * 10) == 0:
                print(f"{role} Dolly seed {seed}: {len(records)}/{len(rows)}", flush=True)
    assert len(records) == len(rows)
    write_json(RAW / f"{role}_dolly_seed{seed}_responses.json", records)
    groups = defaultdict(list)
    for record in records:
        groups[record["topic"]].append(record)
    result = {"seed": seed, "fingerprint": fingerprint, "all_500": generation_metrics(records),
              "excluding_flagged_input_overlap": generation_metrics([r for r in records if r["row_index"] not in overlap]),
              "by_topic": {key: generation_metrics(value) for key, value in groups.items()}}
    write_json(summary_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), nargs="+", default=list(MODELS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_test:
        print("PASS: imbalance-aware classification, text normalization/F1, ROUGE self-checks")
        return
    assert torch.cuda.is_available()
    data, manifest, overlap = prepare_data()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {"checkpoint_source": "author-released SFT; no local training", "training_performed": False,
                "generation": {"seeds": SEEDS, "batch": BATCH, "padded_prompt_length": 256, "max_new_tokens": 256,
                               "temperature": 1., "top_k": 0, "top_p": 1., "no_repeat_ngram_size": 6},
                "precision": "float16", "attention": "eager", "script_sha256": script_hash,
                "sources": {"paper": "https://arxiv.org/html/2306.08543v4",
                            "code_commit": "48fca1acc61ffa20ebb7c31c74a1d8e33499b476"}, "data": manifest}
    write_json(OUT / "protocol.json", protocol)
    RAW.mkdir(parents=True, exist_ok=True)
    for role in args.model:
        started = perf_counter()
        write_json(OUT / f"{role}_protocol.json", protocol)
        model_id, revision = MODELS[role]
        print(f"Loading {role}: {model_id}@{revision}", flush=True)
        patterns = ["*.json", "merges.txt", "vocab.json"] + (["pytorch_model*.bin"] if role == "teacher" else ["model.safetensors"])
        snapshot = snapshot_download(model_id, revision=revision, allow_patterns=patterns,
                                     cache_dir=ROOT / "tmp/hf_home/hub", token=False)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        torch.cuda.reset_peak_memory_stats()
        if role == "student":
            # The published safetensors file stores the shared embedding as lm_head.weight.
            config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation="eager")
            weights = Path(snapshot) / "model.safetensors"
            load_model(model, weights, strict=True)
            saved = load_file(weights)
            assert all(torch.equal(model.state_dict()[key], value.half()) for key, value in saved.items())
            assert torch.equal(model.get_input_embeddings().weight, saved["lm_head.weight"].half())
            assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
            del saved
            model = model.to("cuda:0")
        else:
            model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                       use_safetensors=False, weights_only=True,
                                                       torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                       device_map={"": "cuda:0"}, attn_implementation="eager")
        model = model.eval().requires_grad_(False)
        fingerprint = hashlib.sha256(json.dumps([protocol, model_id, revision], sort_keys=True).encode()).hexdigest()
        result = {"model_id": model_id, "revision": revision, "fingerprint": fingerprint,
                  "protocol_file": f"{role}_protocol.json",
                  "parameters": sum(p.numel() for p in model.parameters()), "classification": {}}
        for name in ["sst2", "boolq"]:
            score, records = classify(model, tokenizer, data[name], name)
            write_json(RAW / f"{role}_{name}_predictions.json", records)
            result["classification"][name] = score
            write_json(OUT / f"{role}_progress.json", result)
            print(role, name, json.dumps(score), flush=True)
        result["reference_diagnostics"] = reference_diagnostics(model, tokenizer, data["dolly"])
        write_json(OUT / f"{role}_progress.json", result)
        result["dolly_seeds"] = [generate_seed(model, tokenizer, data["dolly"], overlap, seed, role, fingerprint) for seed in SEEDS]
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["seconds"] = round(perf_counter() - started, 2)
        result["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        result["versions"] = {name: importlib.metadata.version(name) for name in
                              ["torch", "transformers", "rouge-score", "scikit-learn", "pyarrow", "nltk"]}
        write_json(OUT / f"{role}.json", result)
        print("COMPLETE", role, flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
