"""Portable, checksummed input and SFT-only result bundles (standard library only).

Bundles deliberately contain no credentials, base-model weights, optimizer state,
or evaluation results. Importing an SFT result preserves its external run namespace;
it does not promote the result into the canonical experiment or mark it evaluated.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemma2_9b_it"
SPLITS = ("train", "development", "selection_evaluation")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
FORMAT = "dolly-external-handoff-v1"
SUMMARY_NAMES = ("protocol.json", "preparation.json", "preflight.json",
                 "trainer_preflight.json", "memory_check.json",
                 f"{MODEL}_lr1e-04.json", f"{MODEL}_lr2e-04.json", f"{MODEL}_selected.json")
ADAPTER_NAMES = ("adapter_config.json", "adapter_model.safetensors")
SHARED_CONDITIONS = ("dataset", "counts", "lora", "learning_rate_candidates", "epochs",
                     "effective_batch_examples", "training_seed", "optimizer", "scheduler",
                     "response_truncation", "selection", "checkpoint_and_validation_cadence")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    value = json.loads(data)
    _require(isinstance(value, dict), "Expected a JSON object")
    return value


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _fingerprint(protocol, parent=False):
    excluded = {"fingerprint", "prepared_files_sha256"} if parent else {"fingerprint"}
    value = {key: val for key, val in protocol.items() if key not in excluded}
    return _sha(json.dumps(value, sort_keys=True).encode())


def _parent(root):
    protocol = _json(_read_file(Path(root) / "results/dolly_sft_v1/protocol.json"))
    _require(protocol.get("fingerprint") == _fingerprint(protocol, parent=True),
             "Canonical parent protocol fingerprint is invalid")
    return protocol


def _run_id(value):
    _require(isinstance(value, str) and RUN_ID.fullmatch(value) is not None, "Invalid external run ID")
    return value


def _relative(value):
    _require(isinstance(value, str) and value and "\\" not in value and ":" not in value,
             "Bundle paths must be portable relative POSIX paths")
    _require(not any(ord(char) < 32 or char in '<>"|?*' for char in value), "Unsafe bundle path characters")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/")),
             f"Unsafe bundle path: {value}")
    _require(not any(part.endswith((".", " ")) for part in path.parts), f"Unsafe bundle path: {value}")
    # Names resolving to Windows devices are unsafe even on another operating system.
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    _require(not any(part.split(".")[0].upper() in reserved for part in path.parts),
             f"Unsafe bundle path: {value}")
    return path


def _safe_target(root, relative):
    path = _relative(relative)
    root = Path(root).resolve()
    target = root.joinpath(*path.parts)
    # Refuse any existing symlink, including a parent, instead of following it.
    current = root
    for part in path.parts:
        current = current / part
        _require(not current.is_symlink(), f"Symlinks are not valid handoff targets: {relative}")
        if current != target and current.exists():
            _require(current.is_dir(), f"A parent of the handoff target is not a directory: {relative}")
    _require(target.resolve().is_relative_to(root), f"Target escapes repository: {relative}")
    return target


def _read_file(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"Missing regular file: {path}")
    return path.read_bytes()


def _validate_splits(files, protocol):
    seen = set()
    for split in SPLITS:
        data = files[split]
        expected = protocol["prepared_files_sha256"][f"{split}.jsonl"]
        _require(_sha(data) == expected, f"Original {split} SHA256 differs from parent protocol")
        records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        _require(len(records) == protocol["counts"][split], f"Original {split} count differs from parent protocol")
        ids = [row.get("id") for row in records]
        _require(all(isinstance(value, str) and value for value in ids), f"Invalid {split} record IDs")
        _require(len(set(ids)) == len(ids) and not seen.intersection(ids), f"Duplicate/overlapping {split} IDs")
        seen.update(ids)


def _write_bundle(destination, kind, metadata, files):
    destination = Path(destination)
    _require(not destination.exists(), f"Refusing to overwrite bundle: {destination}")
    manifest = {"format": FORMAT, "kind": kind, **metadata,
                "files": {name: {"sha256": _sha(data), "bytes": len(data)}
                          for name, data in sorted(files.items())}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Build completely before publishing, so interrupted export leaves no final bundle.
    with tempfile.TemporaryDirectory(prefix="dolly-bundle-", dir=destination.parent) as scratch:
        temporary = Path(scratch) / "bundle.zip"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.writestr("manifest.json", _json_bytes(manifest))
            for name, data in sorted(files.items()):
                _relative(name)
                archive.writestr(name, data)
        # A same-volume hard link publishes the complete zip atomically and fails
        # if a destination appeared concurrently. No partial final zip is exposed.
        os.link(temporary, destination)
    return destination


def _read_bundle(bundle, kind):
    with zipfile.ZipFile(bundle) as archive:
        entries = archive.infolist()
        _require(len(entries) <= 32, "Too many entries for a Dolly handoff bundle")
        names = [entry.filename for entry in entries]
        _require(len(names) == len(set(name.casefold() for name in names)), "Duplicate bundle paths")
        for entry in entries:
            _relative(entry.filename)
            _require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16),
                     "Directories/symlinks are not valid bundle entries")
            _require(not entry.flag_bits & 1, "Encrypted bundle entries are unsupported")
            _require(entry.file_size <= 2 * 1024**3, "Bundle entry is too large for a LoRA handoff")
        _require(sum(entry.file_size for entry in entries) <= 4 * 1024**3, "Bundle exceeds handoff size limit")
        _require("manifest.json" in names, "Missing bundle manifest")
        _require(archive.getinfo("manifest.json").file_size <= 1024**2, "Bundle manifest is too large")
        manifest = _json(archive.read("manifest.json"))
        _require(manifest.get("format") == FORMAT and manifest.get("kind") == kind,
                 f"Expected a {kind} bundle")
        specifications = manifest.get("files")
        _require(isinstance(specifications, dict), "Invalid bundle file manifest")
        _require(set(names) == {"manifest.json", *specifications}, "Unexpected or missing bundle files")
        files = {}
        for name, spec in specifications.items():
            _relative(name)
            _require(isinstance(spec, dict), "Invalid bundle file specification")
            data = archive.read(name)
            _require(len(data) == spec.get("bytes") and _sha(data) == spec.get("sha256"),
                     f"Bundle checksum mismatch: {name}")
            files[name] = data
    return manifest, files


def pack_inputs(root, destination):
    """Pack the original common JSONL records after verifying the canonical protocol."""
    root = Path(root)
    parent = _parent(root)
    files = {split: _read_file(_safe_target(root, f"tmp/dolly_sft_v1/{split}.jsonl")) for split in SPLITS}
    _validate_splits(files, parent)
    payload = {f"inputs/{split}.jsonl": data for split, data in files.items()}
    payload["parent_protocol.json"] = _read_file(root / "results/dolly_sft_v1/protocol.json")
    return _write_bundle(destination, "inputs", {"parent_protocol_fingerprint": parent["fingerprint"]}, payload)


def read_inputs(bundle, expected_protocol):
    """Return {split: original_jsonl_bytes}; never extract or modify local files."""
    manifest, payload = _read_bundle(bundle, "inputs")
    expected_names = {"parent_protocol.json", *(f"inputs/{split}.jsonl" for split in SPLITS)}
    _require(set(payload) == expected_names, "Input bundle must contain only parent protocol and three original splits")
    archived = _json(payload["parent_protocol.json"])
    _require(archived.get("fingerprint") == _fingerprint(archived, parent=True), "Archived parent fingerprint is invalid")
    _require(expected_protocol.get("fingerprint") == _fingerprint(expected_protocol, parent=True),
             "Expected parent fingerprint is invalid")
    _require(archived == expected_protocol, "Input bundle parent protocol does not match the local canonical protocol")
    _require(manifest.get("parent_protocol_fingerprint") == archived["fingerprint"], "Input manifest parent mismatch")
    files = {split: payload[f"inputs/{split}.jsonl"] for split in SPLITS}
    _validate_splits(files, expected_protocol)
    return files


def _validate_protocol(root, protocol, run_id):
    parent = _parent(root)
    _require(protocol.get("fingerprint") == _fingerprint(protocol), "External protocol fingerprint is invalid")
    _require(protocol.get("run_id") == run_id, "External protocol run ID mismatch")
    _require(protocol.get("parent_protocol_fingerprint") == parent["fingerprint"], "External parent fingerprint mismatch")
    registry = _json(_read_file(Path(root) / "results/dolly_sft_additional_v1/registry.json"))
    expected_model = {key: registry[MODEL][key] for key in ("id", "revision")}
    _require(protocol.get("models") == {MODEL: expected_model}, "External base model/revision differs from registry")
    for key in SHARED_CONDITIONS:
        _require(protocol.get(key) == parent.get(key), f"External experiment condition differs from parent: {key}")
    _require(protocol.get("max_total_training_tokens") == 1024 and protocol.get("max_prompt_tokens") is None,
             "Gemma must preserve the additional-model complete-response length policy")
    precision = protocol.get("precision", {}).get(MODEL, {})
    _require(precision.get("base_weights") == "BF16, no weight quantization", "External run is not nonquantized BF16")
    _require(precision.get("normalization") == "FP32" and precision.get("trainable_lora") == "FP32",
             "External FP32 normalization/LoRA condition mismatch")
    allowed_losses = (
        {"implementation": "transformers", "chunk_tokens": None,
         "reference": "transformers.loss.loss_utils.ForCausalLMLoss"},
        {"implementation": "checkpointed-ce", "chunk_tokens": 64,
         "reference": "transformers.loss.loss_utils.ForCausalLMLoss"},
    )
    _require(protocol.get("causal_lm_loss") in allowed_losses,
             "Unsupported external causal LM loss implementation, chunk size, or reference")
    sources = {"source_script_sha256": "scripts/dolly_external.py",
               "engine_script_sha256": "scripts/dolly_sft.py", "study_script_sha256": "scripts/dolly_study.py",
               "handoff_script_sha256": "scripts/dolly_handoff.py",
               "external_probe_script_sha256": "scripts/preflight_external_gemma.py",
               "chunked_loss_script_sha256": "scripts/dolly_chunked_loss.py"}
    _require(protocol.get("source_hash_policy") == "UTF-8 source bytes with CRLF normalized to LF",
             "External source hash policy mismatch")
    for key, relative in sources.items():
        source = _read_file(_safe_target(root, relative)).replace(b"\r\n", b"\n")
        _require(protocol.get(key) == _sha(source),
                 f"Source fingerprint mismatch: {relative}; use the checkout that produced this run")
    reference = _json(_read_file(Path(root) / f"results/dolly_sft_additional_v1/{MODEL}/preparation.json"))
    _require(reference.get("revision") == expected_model["revision"] and
             reference.get("parent_protocol_fingerprint") == parent["fingerprint"],
             "Reference Gemma tokenization belongs to another experiment")
    expected_prepared = {}
    for split in SPLITS:
        source = f"tmp/external_nonquantized/{run_id}/source/{split}.jsonl"
        encoded = f"tmp/external_nonquantized/{run_id}/{MODEL}/{split}.jsonl"
        original_encoded = f"tmp/dolly_sft_additional_v1/{MODEL}/{MODEL}/{split}.jsonl"
        expected_prepared[source] = parent["prepared_files_sha256"][f"{split}.jsonl"]
        expected_prepared[encoded] = reference["prepared_files_sha256"][original_encoded]
    _require(protocol.get("prepared_files_sha256") == expected_prepared,
             "External original records or Gemma tokenization fingerprints differ from the reference")
    return parent


def _validate_adapter_weights(data):
    """Read only the safetensors header; reject base weights mislabeled as an adapter."""
    _require(len(data) >= 10, "Invalid adapter safetensors file")
    header_bytes = struct.unpack("<Q", data[:8])[0]
    _require(0 < header_bytes <= min(16 * 1024**2, len(data) - 8), "Invalid safetensors header size")
    header = _json(data[8:8 + header_bytes])
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    _require(tensors, "Adapter safetensors file contains no tensors")
    ranges = []
    for name, value in tensors.items():
        _require(name.endswith((".lora_A.weight", ".lora_B.weight")),
                 "Result file contains tensors other than the trained LoRA adapter")
        _require(isinstance(value, dict) and value.get("dtype") == "F32", "LoRA tensor is not FP32")
        shape, offsets = value.get("shape"), value.get("data_offsets")
        _require(isinstance(shape, list) and len(shape) == 2 and
                 all(isinstance(dim, int) and dim > 0 for dim in shape), "Invalid LoRA tensor shape")
        _require(isinstance(offsets, list) and len(offsets) == 2 and
                 all(isinstance(offset, int) for offset in offsets), "Invalid LoRA tensor offsets")
        start, end = offsets
        _require(0 <= start < end <= len(data) - 8 - header_bytes and
                 end - start == math.prod(shape) * 4, "Invalid LoRA tensor data length")
        ranges.append((start, end))
    position = 0
    for start, end in sorted(ranges):
        _require(start == position, "Overlapping or missing LoRA tensor bytes")
        position = end
    _require(position == len(data) - 8 - header_bytes, "Unexpected non-adapter bytes in safetensors file")


def _validate_result(root, run_id, files):
    run_id = _run_id(run_id)
    result_prefix = f"results/external_nonquantized/{run_id}/"
    for name in SUMMARY_NAMES:
        _require(result_prefix + name in files, f"Missing completed-run summary: {name}")
    records = {name: _json(files[result_prefix + name]) for name in SUMMARY_NAMES}
    protocol = records["protocol.json"]
    _validate_protocol(root, protocol, run_id)
    fingerprint = protocol["fingerprint"]
    prepared = records["preparation.json"]
    _require(prepared.get("status") == "prepared" and prepared.get("same_original_records") is True,
             "External run does not confirm the original common records")
    _require(prepared.get("protocol_fingerprint") == fingerprint and
             prepared.get("parent_protocol_fingerprint") == protocol["parent_protocol_fingerprint"] and
             prepared.get("prepared_files_sha256") == protocol["prepared_files_sha256"] and
             prepared.get("tokenization_matches_reference") is True,
             "Preparation report does not match the external protocol and reference tokenization")
    for name in ("preflight.json", "trainer_preflight.json", "memory_check.json"):
        record = records[name]
        _require(record.get("protocol_fingerprint") == fingerprint, f"Stale {name}")
        if name != "memory_check.json":
            _require(record.get("models", {}).get(MODEL, {}).get("status") == "passed", f"Unpassed {name}")
    _require(records["preflight.json"]["models"][MODEL].get("quantized_linear_layers") == 0,
             "Nonquantized preflight must confirm zero quantized layers")
    _require(records["memory_check.json"].get("fits_dedicated_budget") is True, "Memory preflight has not passed")
    allowed = {result_prefix + name for name in SUMMARY_NAMES}
    if result_prefix + "runtime.json" in files:
        runtime = _json(files[result_prefix + "runtime.json"])
        _require(runtime.get("protocol_fingerprint") == fingerprint, "Stale runtime report")
        allowed.add(result_prefix + "runtime.json")
    candidates = []
    for lr in protocol["learning_rate_candidates"]:
        key = f"{MODEL}_lr{lr:.0e}"
        record = records[f"{key}.json"]
        _require(record.get("status") == "completed" and record.get("model") == MODEL,
                 f"Candidate is not completed: {key}")
        _require(record.get("protocol_fingerprint") == fingerprint and record.get("learning_rate") == lr,
                 f"Candidate identity mismatch: {key}")
        epoch = record.get("training_metrics", {}).get("epoch")
        _require(isinstance(epoch, (int, float)) and math.isfinite(epoch) and epoch >= protocol["epochs"],
                 f"Candidate has not completed all epochs: {key}")
        loss = record.get("selected_development", {}).get("eval_loss")
        _require(type(loss) in (int, float) and math.isfinite(loss) and loss >= 0,
                 f"Invalid development loss: {key}")
        adapter = f"tmp/external_nonquantized/{run_id}/runs/{key}/best_adapter"
        _require(record.get("adapter") == adapter, f"Nonportable or mismatched adapter path: {key}")
        for name in ADAPTER_NAMES:
            path = f"{adapter}/{name}"
            _require(path in files and files[path], f"Missing adapter file: {path}")
            allowed.add(path)
        config = _json(files[f"{adapter}/adapter_config.json"])
        _validate_adapter_weights(files[f"{adapter}/adapter_model.safetensors"])
        for setting in ("r", "lora_alpha", "lora_dropout", "bias", "task_type"):
            _require(config.get(setting) == protocol["lora"][setting], f"Adapter LoRA setting mismatch: {key}/{setting}")
        _require(config.get("peft_type") == "LORA", f"Unexpected adapter type: {key}")
        _require(config.get("base_model_name_or_path") == protocol["models"][MODEL]["id"] and
                 config.get("revision") == protocol["models"][MODEL]["revision"],
                 f"Adapter must identify the portable pinned base model/revision: {key}")
        # PEFT expands all-linear into a module list; target_modules cannot be compared literally.
        candidates.append(record)
    chosen = min(candidates, key=lambda record: record["selected_development"]["eval_loss"])
    selected = records[f"{MODEL}_selected.json"]
    _require(selected.get("status") == "selected" and selected.get("protocol_fingerprint") == fingerprint,
             "Missing or stale LR selection")
    _require(selected.get("learning_rate") == chosen["learning_rate"] and selected.get("adapter") == chosen["adapter"]
             and selected.get("development_nll") == chosen["selected_development"]["eval_loss"],
             "Selected LR, adapter, and development NLL do not match the winning candidate")
    _require(set(files) == allowed, "Result bundle contains unexpected files (only SFT summaries and adapters are allowed)")
    return protocol


def write_result_bundle(root, run_id, destination):
    """Export both completed LR adapters and their unchanged SFT-only summaries."""
    root = Path(root)
    run_id = _run_id(run_id)
    prefix = f"results/external_nonquantized/{run_id}/"
    paths = [prefix + name for name in SUMMARY_NAMES]
    if _safe_target(root, prefix + "runtime.json").exists():
        paths.append(prefix + "runtime.json")
    for lr in (1e-4, 2e-4):
        adapter = f"tmp/external_nonquantized/{run_id}/runs/{MODEL}_lr{lr:.0e}/best_adapter"
        paths.extend(f"{adapter}/{name}" for name in ADAPTER_NAMES)
    files = {path: _read_file(_safe_target(root, path)) for path in paths}
    protocol = _validate_result(root, run_id, files)
    for relative, expected in protocol["prepared_files_sha256"].items():
        _require(_sha(_read_file(_safe_target(root, relative))) == expected,
                 f"Prepared data changed before export: {relative}")
    return _write_bundle(destination, "sft-results", {"run_id": run_id,
                         "protocol_fingerprint": protocol["fingerprint"],
                         "parent_protocol_fingerprint": protocol["parent_protocol_fingerprint"],
                         "evaluated": False}, files)


def import_result_bundle(root, bundle):
    """Validate first, then import into the same external namespace without conflicts.

    Existing identical files are reusable. Any differing file aborts before writes.
    Canonical experiment results and checkpoints are never destinations.
    """
    root = Path(root)
    manifest, files = _read_bundle(bundle, "sft-results")
    run_id = _run_id(manifest.get("run_id"))
    protocol = _validate_result(root, run_id, files)
    _require(manifest.get("protocol_fingerprint") == protocol["fingerprint"] and
             manifest.get("parent_protocol_fingerprint") == protocol["parent_protocol_fingerprint"] and
             manifest.get("evaluated") is False, "Result manifest identity/state mismatch")
    targets = {path: _safe_target(root, path) for path in files}
    for relative, target in targets.items():
        if target.exists():
            _require(target.is_file() and target.read_bytes() == files[relative],
                     f"Refusing to overwrite conflicting external result: {relative}")
    for relative, target in targets.items():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects concurrent runs as well as accidental replacement.
        with target.open("xb") as output:
            output.write(files[relative])
    return root / "results/external_nonquantized" / run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pack-inputs", "export", "import"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.stage == "pack-inputs":
        result = pack_inputs(args.root, args.bundle)
    elif args.stage == "export":
        if args.run_id is None:
            parser.error("export requires --run-id")
        result = write_result_bundle(args.root, args.run_id, args.bundle)
    else:
        result = import_result_bundle(args.root, args.bundle)
    print(result)


if __name__ == "__main__":
    main()
