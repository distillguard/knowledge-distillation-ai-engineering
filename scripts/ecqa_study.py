"""Prepare ECQA and benchmark original Qwen models; never train or call an API."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata

import gpu_safety

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local/ecqa_baseline_v1"
DATA = ROOT / "local/ecqa_data_v1"
SOURCE = ROOT / "local/ecqa_token_audit_v1/source"
OUT = ROOT / "artifacts/ecqa"
MODELS = {
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen35_2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "qwen35_08b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
}
LABELS = "ABCDE"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def fixed_write(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError(f"Existing frozen artifact differs: {path}")
    write(path, value)


def status(stage, **kwargs):
    write(LOCAL / "worker_status.json", {"stage": stage, "pid": os.getpid(),
          "updated_utc": datetime.now(timezone.utc).isoformat(), **kwargs})


def normalize_question(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def clean_split(rows, splits):
    """Preserve test, then dev; assign repeated questions only to their highest-priority split."""
    owners, result, excluded = {}, {}, {}
    for split in ("test", "dev", "train"):
        result[split], excluded[split] = [], []
        for idx in splits[split]:
            key = normalize_question(rows[idx]["question"])
            if key in owners and owners[key] != split:
                excluded[split].append(idx)
            else:
                owners[key] = split
                result[split].append(rows[idx])
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not ({r["id"] for r in result[first]} & {r["id"] for r in result[second]})
        assert not ({normalize_question(r["question"]) for r in result[first]} &
                    {normalize_question(r["question"]) for r in result[second]})
    return result, excluded


def prepare():
    audit = read(ROOT / "local/ecqa_token_audit_v1/token_length_audit.json")
    for name, expected in audit["sha256"].items():
        if name.startswith("qwen35_2b/"):
            continue
        assert digest(SOURCE / name) == expected, name
    questions = {}
    for name in ("train_rand_split.jsonl", "dev_rand_split.jsonl"):
        for line in (SOURCE / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["id"] not in questions
            questions[row["id"]] = row
    explanations, variants = {}, {}
    for line in (SOURCE / "ecqa.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert isinstance(row["explanation"], str) and row["explanation"].strip()
        explanations.setdefault(row["id"], row)
        variants.setdefault(row["id"], [])
        if row["explanation"] not in variants[row["id"]]:
            variants[row["id"]].append(row["explanation"])
    rows = {}
    for idx, annotation in explanations.items():
        original = questions[idx]
        options = {c["label"]: c["text"] for c in original["question"]["choices"]}
        assert sorted(options) == list(LABELS) and original["answerKey"] in options
        rows[idx] = {"id": idx, "question": original["question"]["stem"], "options": options,
                     "gold": original["answerKey"], "explanation": annotation["explanation"],
                     "reference_explanations": variants[idx], "positives": annotation["positives"],
                     "negatives": annotation["negatives"],
                     "question_concept": original["question"]["question_concept"]}
    splits = {key: (SOURCE / "author_split" / name).read_text(encoding="utf-8").splitlines()
              for key, name in (("train", "train_ids.txt"), ("dev", "val_ids.txt"), ("test", "test_ids.txt"))}
    prepared, excluded = clean_split(rows, splits)
    for split, items in prepared.items():
        fixed_write(DATA / f"{split}.json", items)
    spec = {"status": "completed", "source_revision": audit["revisions"]["ecqa_commit"],
            "source_sha256": {k: v for k, v in audit["sha256"].items() if not k.startswith("qwen35_2b/")},
            "original_author_counts": {k: len(v) for k, v in splits.items()},
            "clean_counts": {k: len(v) for k, v in prepared.items()},
            "priority": ["test", "dev", "train"],
            "group_rule": "NFKC, casefold, whitespace-normalized question; higher-priority split owns the question",
            "excluded_lower_priority_ids": excluded, "test_ids_unchanged": [r["id"] for r in prepared["test"]] == splits["test"],
            "annotation_variants": {k: len(v) for k, v in variants.items() if len(v) > 1},
            "annotation_policy": "Keep every explanation variant; first annotation is the canonical text, matching author join code. No explanation enters benchmark prompts.",
            "unassigned_annotation_ids": sorted(set(rows) - set().union(*map(set, splits.values()))),
            "data_sha256": {s: digest(DATA / f"{s}.json") for s in prepared},
            "disjoint_ids": True, "disjoint_normalized_questions": True,
            "training_started": False, "api_calls": 0,
            "dataset_license": "https://github.com/dair-iitd/ECQA-Dataset/blob/main/LICENSE.md"}
    spec["fingerprint"] = fingerprint(spec)
    fixed_write(OUT / "data_manifest.json", spec)
    return prepared, spec


def snapshots():
    archive = ROOT / "local/archive/legacy_before_ecqa_20261021/tmp/hf_home/hub"
    paths = {}
    for name, (model_id, revision) in MODELS.items():
        cache = ROOT / "local/model_cache/hub" if name == "qwen35_2b" else archive
        path = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
        assert path.is_dir() and (path / "tokenizer.json").is_file(), path
        index = path / "model.safetensors.index.json"
        weights = sorted(set(read(index)["weight_map"].values())) if index.exists() else ["model.safetensors"]
        assert all((path / w).is_file() and (path / w).stat().st_size > 0 for w in weights), path
        paths[name] = path
    return paths


def architecture(paths):
    result = {}
    for name, path in paths.items():
        config = read(path / "config.json")
        text = config["text_config"]
        result[name] = {"model": MODELS[name][0], "revision": MODELS[name][1],
                       "config_sha256": digest(path / "config.json"), "tokenizer_sha256": digest(path / "tokenizer.json"),
                       "text_config": {k: text[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
                           "num_attention_heads", "num_key_value_heads", "head_dim", "linear_num_key_heads",
                           "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "vocab_size",
                           "max_position_embeddings", "hidden_act", "rms_norm_eps", "tie_word_embeddings")},
                       "layer_counts": dict(Counter(text["layer_types"]))}
    assert len({x["tokenizer_sha256"] for x in result.values()}) == 1
    fixed_write(OUT / "model_architecture.json", result)


def prompt(row, cap):
    instruction = ("Answer the commonsense multiple-choice question. Give your selected answer FIRST, "
        "then a concise English explanation. Use exactly this format without markdown or JSON:\n"
        "ANSWER: X\nEXPLANATION: Your explanation.\nReplace X with one of A, B, C, D, E. "
        f"Do not repeat the question or options. Keep the entire response within {cap} output tokens.")
    return instruction + "\n\n" + row["question"] + "\n\n" + "\n".join(c + ") " + row["options"][c] for c in LABELS)


def inspect_response(text):
    lines = text.splitlines()
    match = re.fullmatch(r"ANSWER: ([A-E])", lines[0]) if lines else None
    suffix = "\n".join(lines[1:])
    explanation = suffix[len("EXPLANATION:"):].strip() if suffix.startswith("EXPLANATION:") else ""
    return {"predicted": match.group(1) if match else None, "explanation_present": bool(explanation),
            "format_valid": match is not None and bool(explanation)}


def token_layout(tokenizer):
    header = tokenizer.encode("ANSWER:", add_special_tokens=False)
    labels = [tokenizer.encode(" " + c, add_special_tokens=False) for c in LABELS]
    suffix = tokenizer.encode("\nEXPLANATION:", add_special_tokens=False)
    assert all(len(ids) == 1 for ids in labels) and len({ids[0] for ids in labels}) == 5
    for c, ids in zip(LABELS, labels):
        assert tokenizer.encode(f"ANSWER: {c}\nEXPLANATION:", add_special_tokens=False) == header + ids + suffix
    return header, [ids[0] for ids in labels], suffix


class AnswerFirst:
    def __init__(self, prompt_length, header, labels, suffix):
        self.prompt_length, self.header, self.labels, self.suffix = prompt_length, header, labels, suffix
        self.probabilities, self.log_probabilities, self.selected_label = None, None, None

    def __call__(self, input_ids, scores):
        import torch
        offset = input_ids.shape[1] - self.prompt_length
        if offset < len(self.header):
            allowed = [self.header[offset]]
        elif offset == len(self.header):
            allowed = self.labels
            values = scores[0, allowed].detach().float()
            assert torch.isfinite(values).all()
            logs = values.log_softmax(-1)
            self.log_probabilities, self.probabilities = logs.cpu().tolist(), logs.exp().cpu().tolist()
        elif offset < len(self.header) + 1 + len(self.suffix):
            allowed = [self.suffix[offset-len(self.header)-1]]
        else:
            return scores
        constrained = torch.full_like(scores, -float("inf"))
        constrained[:, allowed] = scores[:, allowed]
        if offset == len(self.header):
            self.selected_label = LABELS[self.labels.index(int(constrained[0].argmax()))]
        return constrained


def metrics(records):
    assert records
    matrix = [[0]*6 for _ in LABELS]
    for row in records:
        assert row["gold"] in LABELS and row["predicted"] in [*LABELS, None]
        matrix[LABELS.index(row["gold"])][LABELS.index(row["predicted"]) if row["predicted"] else 5] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp, support, predicted = matrix[i][i], sum(matrix[i]), sum(r[i] for r in matrix)
        precision, recall = tp/predicted if predicted else 0., tp/support if support else 0.
        per_class[label] = {"support": support, "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.}
    n = len(records)
    assert all(len(r["choice_probabilities"]) == 5 and abs(sum(r["choice_probabilities"])-1) < 1e-5 for r in records)
    nll = -sum(r["choice_log_probabilities"][LABELS.index(r["gold"])] for r in records)/n
    brier = sum(sum((p-float(i == LABELS.index(r["gold"])))**2 for i, p in enumerate(r["choice_probabilities"])) for r in records)/n
    return {"n": n, "correct": sum(matrix[i][i] for i in range(5)),
            "accuracy": sum(matrix[i][i] for i in range(5))/n,
            **{"macro_"+key: sum(v[key] for v in per_class.values())/5 for key in ("precision", "recall", "f1")},
            "choice_nll": nll, "brier_score": brier, "per_class": per_class,
            "confusion_matrix": matrix, "invalid_answer_count": sum(row[5] for row in matrix),
            "length_cap_count": sum(r["hit_length_cap"] for r in records),
            "empty_explanation_count": sum(not r["explanation_present"] for r in records),
            "format_error_count": sum(not r["format_valid"] for r in records),
            "mean_output_tokens": sum(r["output_tokens"] for r in records)/n,
            "total_generation_seconds": sum(r["seconds"] for r in records)}


def probe_rows(dev, tokenizer):
    def length(row):
        return len(tokenizer.encode(prompt(row, 256), add_special_tokens=False))
    def response_length(row):
        return max(len(tokenizer.encode("ANSWER: " + row["gold"] + "\nEXPLANATION: " + text,
                   add_special_tokens=False)) + 1 for text in row["reference_explanations"])
    selected = [min(dev, key=lambda r: (length(r), r["id"])),
                max(dev, key=lambda r: (length(r), r["id"])),
                max(dev, key=lambda r: (response_length(r), r["id"])),
                max(dev, key=lambda r: (length(r)+response_length(r), r["id"]))]
    return list({r["id"]: r for r in selected}.values())


def load_model(path):
    import torch
    from transformers import Qwen3_5ForCausalLM
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_safety.configure()
    torch.cuda.reset_peak_memory_stats()
    model, loading = Qwen3_5ForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Weight loading mismatch: " + str(loading))
    if hasattr(model.model, "visual") or getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Expected original nonquantized language-only model")
    return model.eval().requires_grad_(False)


def evaluate(name, path, rows, split, cap, data_spec, adapter=None, training=None, report_status=status):
    import torch
    from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    run_key = name + ("_sft" if adapter is not None else "")
    generation_path = path / "generation_config.json"
    source_eos = (read(generation_path) if generation_path.exists() else read(path / "config.json")["text_config"]).get("eos_token_id")
    eos_ids = source_eos if isinstance(source_eos, list) else [source_eos]
    eos_ids = sorted({x for x in [*eos_ids, tokenizer.eos_token_id] if x is not None})
    assert eos_ids
    spec = {"model": MODELS[name][0], "revision": MODELS[name][1], "training": training or "none; original post-trained model",
            "split": split, "n": len(rows), "ordered_ids_sha256": fingerprint([r["id"] for r in rows]),
            "dataset_fingerprint": data_spec["fingerprint"], "test_used_for_selection": False,
            "max_new_tokens": cap, "instruction": prompt({"question": "", "options": {c: "" for c in LABELS}}, cap),
            "dtype": "bfloat16", "quantization": None, "batch_size": 1, "thinking": False,
            "do_sample": False, "num_beams": 1, "truncation": False, "temperature": None,
            "eos_token_ids": eos_ids,
            "format": "Force ANSWER header and EXPLANATION separator; model selects A-E, explanation is free generation",
            "choice_distribution": "A-E conditional softmax at the ANSWER position, temperature 1",
            "tie_rule": "greedy torch.argmax, lowest vocabulary token ID",
            "tokenizer_sha256": digest(path / "tokenizer.json"), "script_sha256": digest(__file__),
            "safety_sha256": digest(Path(gpu_safety.__file__)), "safety": gpu_safety.LIMITS,
            "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "safetensors")}}
    if adapter is not None:
        spec["adapter_sha256"] = digest(Path(adapter) / "adapter_model.safetensors")
    spec["fingerprint"] = fingerprint(spec)
    folder = LOCAL / "responses" / run_key / split / str(cap)
    protocol = OUT / "protocols" / f"{run_key}_{split}_{cap}.json"
    fixed_write(protocol, spec)
    records, pending = [], []
    for row in rows:
        saved = folder / (row["id"] + ".json")
        if saved.exists():
            item = read(saved)
            assert item["fingerprint"] == spec["fingerprint"] and item["gold"] == row["gold"] and item["id"] == row["id"]
            assert inspect_response(item["response"]) == {k: item[k] for k in ("predicted", "explanation_present", "format_valid")}
            records.append(item)
        else:
            pending.append(row)
    if pending:
        report_status("loading_model", model=name, split=split, completed=len(records), total=len(rows), cap=cap)
        model = load_model(path)
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval().requires_grad_(False)
        eos = eos_ids
        assert all(isinstance(x, int) for x in eos)
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=cap,
            eos_token_id=eos, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
        layout = token_layout(tokenizer)
        try:
            with torch.inference_mode():
                for row in pending:
                    report_status("generating", model=name, split=split, completed=len(records), total=len(rows), current_id=row["id"], cap=cap)
                    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt(row, cap)}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
                    formatter = AnswerFirst(inputs.input_ids.shape[-1], *layout)
                    guard = gpu_safety.stopping_guard()
                    started = time.monotonic()
                    output = model.generate(**inputs, generation_config=config,
                        logits_processor=LogitsProcessorList([formatter]), stopping_criteria=StoppingCriteriaList([guard]))
                    tokens = output[0, inputs.input_ids.shape[-1]:].tolist()
                    response = tokenizer.decode(tokens, skip_special_tokens=True)
                    info = inspect_response(response)
                    assert formatter.selected_label == info["predicted"] and formatter.probabilities is not None
                    assert 0 < len(tokens) <= cap
                    item = {"id": row["id"], "gold": row["gold"], "fingerprint": spec["fingerprint"],
                            "response": response, **info, "choice_probabilities": formatter.probabilities,
                            "choice_log_probabilities": formatter.log_probabilities,
                            "hit_length_cap": len(tokens) == cap and tokens[-1] not in eos,
                            "ended_with_eos": tokens[-1] in eos, "output_tokens": len(tokens),
                            "input_tokens": inputs.input_ids.shape[-1], "seconds": time.monotonic()-started,
                            "minimum_free_vram_bytes": guard.minimum_free_vram}
                    write(folder / (row["id"] + ".json"), item)
                    records.append(item)
                    del inputs, output
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    by_id = {r["id"]: r for r in records}
    records = [by_id[r["id"]] for r in rows]
    assert len(records) == len(rows) and len(by_id) == len(rows)
    result = {"status": "completed", "model": MODELS[name][0], "revision": MODELS[name][1],
              "fingerprint": spec["fingerprint"], "dataset_fingerprint": data_spec["fingerprint"],
              "max_new_tokens": cap, "split": split, "overall": metrics(records),
              "completed_utc": datetime.now(timezone.utc).isoformat(), "explanation_factuality_reviewed": False}
    if split == "test":
        write(OUT / ("sft" if adapter is not None else "baselines") / f"{name}.json", result)
    else:
        write(OUT / "probes" / f"{name}_{cap}.json", result)
    return result


def run():
    from filelock import FileLock
    from transformers import AutoTokenizer
    import render_results
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (LOCAL / "RUN_DISABLED.json").exists():
        raise RuntimeError("Safety stop marker exists; inspect its cause before restarting")
    with FileLock(str(LOCAL / "worker.lock"), timeout=0), FileLock(str(ROOT / "local/gpu.lock"), timeout=0):
        try:
            prepared, data_spec = prepare()
            paths = snapshots()
            architecture(paths)
            tokenizer = AutoTokenizer.from_pretrained(paths["qwen35_4b"], local_files_only=True, trust_remote_code=False)
            selected = probe_rows(prepared["dev"], tokenizer)
            cap_path = OUT / "generation_limit.json"
            if cap_path.exists():
                cap = read(cap_path)["selected_max_new_tokens"]
            else:
                cap = 256
                probes = [evaluate(name, paths[name], selected, "dev_probe", cap, data_spec) for name in MODELS]
                if any(r["overall"]["length_cap_count"] or r["overall"]["format_error_count"] for r in probes):
                    cap = 512
                    probes = [evaluate(name, paths[name], selected, "dev_probe", cap, data_spec) for name in MODELS]
                write(cap_path, {"status": "completed", "selected_max_new_tokens": cap,
                    "rule": "Start at 256 from reference maximum 190; if any model hits the cap or lacks valid explanation on deterministic dev extremes, use common 512. No expansion beyond 512.",
                    "dev_probe_ids": [r["id"] for r in selected], "probe_selection": "shortest input, longest input, longest reference output, longest combined sample; unique IDs",
                    "probe_scope": "dev extremes only; not evidence that every future generation terminates below cap",
                    "remaining_probe_caps": {r["model"]: r["overall"]["length_cap_count"] for r in probes},
                    "test_used_for_selection": False, "dataset_fingerprint": data_spec["fingerprint"]})
            render_results.render()
            for name in MODELS:
                evaluate(name, paths[name], prepared["test"], "test", cap, data_spec)
                render_results.render()
                print(json.dumps({"model_completed": name, "max_new_tokens": cap}, ensure_ascii=True), flush=True)
            status("completed", models=list(MODELS), test_n=len(prepared["test"]), cap=cap)
        except Exception as error:
            write(LOCAL / "RUN_DISABLED.json", {"reason": str(error), "error_type": type(error).__name__,
                  "automatic_restart": False, "created_utc": datetime.now(timezone.utc).isoformat()})
            status("failed", error_type=type(error).__name__, reason=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    action = parser.parse_args().action
    if action == "prepare":
        prepared, spec = prepare()
        architecture(snapshots())
        print(json.dumps({"counts": spec["clean_counts"], "fingerprint": spec["fingerprint"]}))
    elif action == "run":
        run()
    else:
        import render_results
        render_results.render()
