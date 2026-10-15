"""Audit every downloaded candidate record; do not split, truncate or train."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp/eval_deps"))
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

HUB = ROOT / "tmp/hf_home/hub"
OUT = ROOT / "results/dataset_candidates"
SOURCES = {
    "dolly": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a"),
    "metamath": ("meta-math/MetaMathQA", "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"),
}
INTRO = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
CONTEXT_INTRO = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"


def normalize(value):
    return " ".join(value.casefold().split())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prompt(instruction, context=""):
    return ((CONTEXT_INTRO if context else INTRO) + f"### Instruction:\n{instruction}"
            + (f"\n\n### Input:\n{context}" if context else "") + "\n\n### Response:\n")


def length_summary(lengths):
    values = np.asarray(lengths, dtype=np.int64)
    return {"records": len(values), "total_tokens": int(values.sum()),
            "min": int(values.min()), "max": int(values.max()),
            "percentiles": dict(zip(("p25", "p50", "p75", "p90", "p95", "p99"),
                                    np.percentile(values, [25, 50, 75, 90, 95, 99]).tolist())),
            "complete_records_at_most_native_1024": int((values <= 1024).sum()),
            "complete_records_over_native_1024": int((values > 1024).sum())}


def main():
    started = perf_counter()
    tokenizer_path = HUB / "models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.model_max_length = sys.maxsize  # Measure full lengths; never truncate in this audit.
    assert len(tokenizer) == 50257 and tokenizer.eos_token_id == 50256
    result = {"script_sha256": file_hash(Path(__file__)), "tokenizer_revision": tokenizer_path.name,
              "audit_only": True, "training_performed": False,
              "template": {"single_turn": "MiniLLM pinned process_data_dolly.py instruction/input template + response + EOS",
                           "multi_turn": "Proposed audit format: shared introduction, repeated Instruction/Response headers, one final EOS",
                           "multi_turn_is_original_paper_template": False,
                           "special_tokens_added": False, "truncation": False}, "datasets": {}}
    # Self-check serialization and the native capacity boundary, independently of corpus contents.
    assert prompt("Q").endswith("### Response:\n") and "### Input:\nx" in prompt("Q", "x")
    assert length_summary([1024, 1025])["complete_records_over_native_1024"] == 1

    for name, (repo, revision) in SOURCES.items():
        directory = HUB / ("datasets--" + repo.replace("/", "--")) / "snapshots" / revision
        files = sorted(directory.rglob("*.parquet")) if name == "ultrachat" else [directory / ("databricks-dolly-15k.jsonl" if name == "dolly" else "MetaMathQA-395K.json")]
        record = {"repo": repo, "revision": revision,
                  "files": [{"name": p.relative_to(directory).as_posix(), "bytes": p.stat().st_size,
                             "sha256": file_hash(p)} for p in files], "splits": {}}
        split_keys = {}
        splits = ("train_sft", "test_sft") if name == "ultrachat" else ("train",)
        for split in splits:
            if name == "ultrachat":
                rows = (row for p in files if p.name.startswith(split) for batch in pq.ParquetFile(p).iter_batches(batch_size=256) for row in batch.to_pylist())
            elif name == "dolly":
                rows = (json.loads(line) for line in files[0].open(encoding="utf-8"))
            else:
                rows = iter(json.loads(files[0].read_text(encoding="utf-8")))
            counts, groups, queries, ids, lengths = Counter(), Counter(), Counter(), set(), []
            full_records, first_users = set(), set()
            pending = []
            for index, row in enumerate(rows):
                if name == "ultrachat":
                    messages = row["messages"]
                    assert isinstance(row["prompt_id"], str) and messages
                    roles = [m["role"] for m in messages]
                    assert all(isinstance(m["content"], str) for m in messages)
                    counts["invalid_role_alternation"] += int(roles != ["user", "assistant"] * (len(roles) // 2))
                    counts["assistant_turns"] += roles.count("assistant")
                    counts["empty_message_content"] += sum(not m["content"].strip() for m in messages)
                    counts["prompt_not_first_user_message"] += int(row["prompt"] != messages[0]["content"])
                    counts["normalized_prompt_not_first_user_message"] += int(normalize(row["prompt"]) != normalize(messages[0]["content"]))
                    first_users.add(digest(normalize(messages[0]["content"])))
                    text = INTRO + "\n\n".join(("### Instruction:\n" if m["role"] == "user" else "### Response:\n") + m["content"] for m in messages)
                    group = normalize(row["prompt"])
                    ids.add(row["prompt_id"])
                    query = group
                else:
                    instruction = row["instruction"] if name == "dolly" else row["query"]
                    response = row["response"]
                    context = row["context"] if name == "dolly" else ""
                    assert all(isinstance(v, str) for v in (instruction, response, context))
                    counts["empty_instruction"] += int(not instruction.strip())
                    counts["empty_response"] += int(not response.strip())
                    counts["with_context"] += int(bool(context.strip()))
                    counts[row["category"] if name == "dolly" else row["type"]] += 1
                    if name == "metamath":
                        counts["answer_marker_present"] += int("The answer is:" in response)
                    text = prompt(instruction, context) + response
                    query = normalize(instruction) + "\0" + normalize(context)
                    # Preserve mathematical variable case when grouping original questions.
                    group = " ".join(row["original_question"].split()) if name == "metamath" else query
                groups[digest(group)] += 1
                queries[digest(query)] += 1
                full_records.add(digest(text))
                pending.append(text)
                if len(pending) == 256:
                    lengths.extend(len(tokens) + 1 for tokens in tokenizer(pending, add_special_tokens=False)["input_ids"])
                    pending.clear()
                if (index + 1) % 50000 == 0:
                    print(f"{name}/{split}: {index + 1:,} records", flush=True)
            if pending:
                lengths.extend(len(tokens) + 1 for tokens in tokenizer(pending, add_special_tokens=False)["input_ids"])
            assert sum(groups.values()) == len(lengths) == sum(queries.values())
            record["splits"][split] = {"counts": dict(counts), "normalized_problem_groups": len(groups),
                                        "largest_group_rows": max(groups.values()),
                                        "duplicate_query_rows_beyond_first": len(lengths) - len(queries),
                                        "exact_serialized_duplicate_rows_beyond_first": len(lengths) - len(full_records),
                                        "unique_prompt_ids": len(ids) if ids else None,
                                        "lengths_prompt_and_response_with_eos": length_summary(lengths)}
            split_keys[split] = (set(groups), ids, first_users)
        if name == "ultrachat":
            record["published_sft_train_test_overlap"] = {
                "normalized_first_prompts": len(split_keys["train_sft"][0] & split_keys["test_sft"][0]),
                "prompt_ids": len(split_keys["train_sft"][1] & split_keys["test_sft"][1]),
                "normalized_first_user_messages": len(split_keys["train_sft"][2] & split_keys["test_sft"][2])}
        result["datasets"][name] = record
        OUT.mkdir(parents=True, exist_ok=True)
        destination = OUT / "audit.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        print(f"Finished {name}: {perf_counter() - started:.1f}s", flush=True)
    print("Complete audit saved to results/dataset_candidates/audit.json", flush=True)


if __name__ == "__main__":
    main()
