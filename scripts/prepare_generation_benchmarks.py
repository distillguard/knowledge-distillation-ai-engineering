"""Freeze the paper's held-out generation data without changing SFT records."""
import argparse
import hashlib
import json

import dolly_study
from dolly_sft import LOCAL as SFT_LOCAL, ROOT, protocol as sft_protocol, rows, sha, write_json
from dolly_study import MODELS, REGISTRY, read, render_prompt, tokenizer_for

OUT = ROOT / "results/generation_benchmarks_v1"
LOCAL = ROOT / "tmp/generation_benchmarks_v1"
SOURCES = {
    "self_inst": ("MiniLLM/self-inst", "ad9edcc55ec5623f6d9d2472963bda3701681af8", "valid.jsonl"),
    "sni_11": ("MiniLLM/sinst", "84b8356a5b631002f92ad73adebd5ac9db5612cd", "11_/valid.jsonl"),
    "vicuna": ("MiniLLM/Vicuna", "05a18702266a6c51547034268d6d58bce214bd77", "valid.jsonl"),
}
BENCHMARKS = {"dolly": "Dolly", "self_inst": "Self-Instruct", "sni_11": "S-NI (11_)", "vicuna": "Vicuna"}
PAIRS = (("gpt2_xl", "gpt2_base"), ("qwen_9b", "qwen_08b"), ("opt_27b", "opt_125m"),
         ("openllama_7b", "openllama_3b"), ("gemma2_9b_it", "gemma2_2b_it"))
NEW_TOKENS = 256


def group(instruction):
    return " ".join(instruction.casefold().split())


def normalize(row, identifier):
    assert isinstance(row["instruction"], str) and row["instruction"].strip(), identifier
    context = row.get("input", "")
    references = row["output"] if isinstance(row["output"], list) else [row["output"]]
    assert isinstance(context, str) and references and all(isinstance(x, str) and x.strip() for x in references), identifier
    return {"id": identifier, "instruction": row["instruction"], "input": context,
            "references": references, "category": str(row.get("category", row.get("topic", "unspecified"))),
            "prompt_tokens": {}}


def fetch():
    from huggingface_hub import hf_hub_download
    for key, (repo, revision, filename) in SOURCES.items():
        hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename=filename,
                        local_dir=LOCAL / "sources" / key)
        print(f"Fetched pinned {key}", flush=True)


def pinned_models(training):
    return {**training["models"], **{n: {k: r[k] for k in ("id", "revision")} for n, r in read(REGISTRY).items()}}


def prepared_protocol():
    spec = read(OUT / "protocol.json")
    payload = {k: v for k, v in spec.items() if k != "fingerprint"}
    assert spec["fingerprint"] == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert spec["preparation_script_sha256"] == sha(__file__), "Generation data preparation source changed"
    assert spec["tokenizer_study_script_sha256"] == sha(dolly_study.__file__)
    training = sft_protocol()
    assert spec["training_protocol_fingerprint"] == training["fingerprint"]
    assert spec["tokenizer_models"] == pinned_models(training)
    for path, expected in spec["files_sha256"].items():
        assert sha(ROOT / path) == expected, path
    return spec


def prepare():
    frozen = read(OUT / "protocol.json") if (OUT / "protocol.json").exists() else None
    if frozen:
        assert frozen["preparation_script_sha256"] == sha(__file__), "Generation preparation source changed"
        if all((ROOT / path).is_file() for path in frozen["files_sha256"]):
            prepared_protocol()
            print("Existing generation data contract verified", flush=True)
            return
    training = sft_protocol()
    protected = {group(r["instruction"]) for split in ("train", "development") for r in rows(SFT_LOCAL / f"{split}.jsonl")}
    source_rows = {"dolly": rows(SFT_LOCAL / "selection_evaluation.jsonl")}
    source_info = {"dolly": {"source": "unchanged SFT selection_evaluation.jsonl", "n": len(source_rows["dolly"])}}
    files = {str((SFT_LOCAL / f"{s}.jsonl").relative_to(ROOT).as_posix()): sha(SFT_LOCAL / f"{s}.jsonl")
             for s in ("train", "development", "selection_evaluation")}
    tokenizer_models = pinned_models(training)
    for key, (repo, revision, filename) in SOURCES.items():
        path = LOCAL / "sources" / key / filename
        source_rows[key] = rows(path)
        files[path.relative_to(ROOT).as_posix()] = sha(path)
        source_info[key] = {"id": repo, "revision": revision, "file": filename, "sha256": sha(path), "n": len(source_rows[key])}
    records, exclusions, duplicates, reformatted = {}, {}, {}, {}
    for key, values in source_rows.items():
        records[key], exclusions[key] = [], []
        for index, value in enumerate(values):
            identifier = value["id"] if key == "dolly" else f"{key}:{index}"
            item = normalize(value, identifier)
            if group(item["instruction"]) in protected:
                exclusions[key].append({"id": identifier, "reason": "exact_instruction_overlap_with_train_or_development"})
            else:
                records[key].append(item)
        prompts = [group(r["instruction"] + "\n" + r["input"]) for r in records[key]]
        duplicates[key] = len(prompts) - len(set(prompts))
        reformatted[key] = sum(render_prompt("gpt2_base", None, r["instruction"], r.get("input", "")) !=
                               r.get("prompt", "").replace("<n>", "\n") for r in values) if key != "dolly" else 0
    audit = {}
    for name in MODELS:
        tokenizer = tokenizer_for(name)
        audit[name] = {"native_context": tokenizer.model_max_length, "benchmarks": {}}
        for key, values in records.items():
            lengths = []
            for row in values:
                count = len(tokenizer.encode(render_prompt(name, tokenizer, row["instruction"], row["input"]), add_special_tokens=False))
                row["prompt_tokens"][name] = count
                lengths.append(count)
                if count + NEW_TOKENS > tokenizer.model_max_length:
                    exclusions[key].append({"id": row["id"], "reason": "prompt_plus_generation_exceeds_native_context", "model": name})
            audit[name]["benchmarks"][key] = {"checked": len(lengths), "max_prompt_tokens": max(lengths, default=0)}
        print(f"CPU tokenizer audit passed: {name}", flush=True)
    counts = {}
    for key, values in records.items():
        excluded = {x["id"] for x in exclusions[key]}
        kept = [r for r in values if r["id"] not in excluded]
        assert kept and len({r["id"] for r in kept}) == len(kept)
        if key == "dolly":
            assert not excluded and [r["id"] for r in kept] == [r["id"] for r in source_rows[key]]
        path = LOCAL / "prepared" / f"{key}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8", newline="\n")
        files[path.relative_to(ROOT).as_posix()] = sha(path)
        counts[key] = {"source": len(source_rows[key]), "included": len(kept), "excluded": len(source_rows[key]) - len(kept),
                       "included_ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in kept]).encode()).hexdigest()}
    spec = {"status": "passed", "preparation_script_sha256": sha(__file__), "tokenizer_study_script_sha256": sha(dolly_study.__file__),
            "tokenizer_models": tokenizer_models, "training_protocol_fingerprint": training["fingerprint"],
            "sources": source_info, "counts": counts, "files_sha256": files, "tokenizer_audit": audit,
            "exclusions": exclusions, "duplicate_normalized_prompts_retained": duplicates, "gpt2_prompt_reformatted_count": reformatted,
            "policy": {"sampling": "all distributed rows of Self-Instruct, S-NI 11_, and Vicuna; unchanged Dolly 500",
                       "overlap": "exclude casefold/whitespace-normalized instruction matches to SFT train/development; semantic contamination not audited",
                       "length": "same full text and shared IDs for all ten tokenizers; exclude any row whose full prompt plus 256 new tokens exceeds a native context; no text truncation",
                       "prompt": "same model-native wrapper as Dolly SFT, preserving instruction/input fields",
                       "sni": "publisher's 11_ directory, not a new reference-length filter",
                       "generation": "same seed 10 and sampling as existing Dolly selection; not an exact paper reproduction"}}
    spec["fingerprint"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if frozen:
        assert spec == frozen, "Reconstructed generation data differ from the frozen contract; do not replace completed results"
    else:
        write_json(OUT / "protocol.json", spec)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "prepare"))
    args = parser.parse_args()
    fetch() if args.stage == "fetch" else prepare()
