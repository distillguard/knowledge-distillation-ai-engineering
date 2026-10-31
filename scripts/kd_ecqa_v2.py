"""Fixed epoch-3 ECQA KD, seeds 10/20/30; immutable v1 inputs, isolated v2 outputs."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import kd_ecqa as kd

base = kd.base
LOCAL = base.ROOT / "local/ecqa_kd_v2"
OUT = base.OUT / "kd_v2"
INVENTORY = LOCAL / "planning/20261027/reuse_inventory.json"
SEEDS = (10, 20, 30)
PINNED = {
    "kd_ecqa.py": "cbf1b65f43168e46ebe585726e797a25f28137b318732dd54f8cb1370481b895",
    "sft_ecqa.py": "064c605d30ce93a0e39c680cb9e320d1f247926e92be759338d82abbe251f103",
    "ecqa_study.py": "369c09865d8032c3b5ca7cc125d2f6cfe033e760b2c3e1f48ede1f2f16719648",
    "gpu_safety.py": "67eb21c36e8b11c00cbac8ac29503589d70fcb6b00b0103ffdbcb40b1ec9810c",
}
ENVIRONMENT = {"PYTORCH_ALLOC_CONF": "backend:native,garbage_collection_threshold:0.8",
    "PYTORCH_CUDA_ALLOC_CONF": "backend:native,garbage_collection_threshold:0.8",
    "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
BEGIN, END = "<!-- ECQA_KD_V2_START -->", "<!-- ECQA_KD_V2_END -->"
MDS = ("RESEARCH_PLAN.md", "RESEARCH_REPORT.md", "RESULTS_VISUALIZATION.md")


def now():
    return datetime.now(timezone.utc).isoformat()


def status(phase, **fields):
    base.write(LOCAL / "worker_status.json", {"stage": phase, "pid": os.getpid(), "updated_utc": now(), **fields})


def trainer_source():
    """Change only the three seed constants in the pinned, already-tested training function."""
    source = inspect.getsource(kd.train_stage)
    replacements = {"set_seed(10)": "set_seed(EXPERIMENT_SEED)",
                    "seed=10, data_seed=10": "seed=EXPERIMENT_SEED, data_seed=EXPERIMENT_SEED"}
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    original, changed = ast.parse(inspect.getsource(kd.train_stage)), ast.parse(source)
    changed_seeds = 0
    for node in ast.walk(changed):
        if isinstance(node, ast.Name) and node.id == "EXPERIMENT_SEED":
            node.__class__ = ast.Constant
            node.__dict__ = {"value": 10}
            changed_seeds += 1
    assert changed_seeds == 3 and ast.dump(original, include_attributes=False) == ast.dump(changed, include_attributes=False)
    return source


def engine_for(seed):
    assert seed in SEEDS
    engine = kd.isolated_helper(kd, f"ecqa_kd_v2_seed{seed}")
    engine.LOCAL, engine.OUT = LOCAL / f"seed{seed}", OUT / f"seed{seed}"
    engine.EXPERIMENT_SEED = seed
    engine.status = lambda phase, **fields: status(phase, seed=seed, **fields)
    engine.report = report
    exec(compile(trainer_source(), str(Path(__file__).resolve()), "exec"), engine.__dict__)
    return engine


def verify_preservation():
    inventory = base.read(INVENTORY)
    for filename, sha in PINNED.items():
        assert base.digest(base.ROOT / "scripts" / filename) == sha, filename
    for name, sha in inventory["preserved_sha256"].items():
        path = base.ROOT / name
        if name in MDS and (LOCAL / "md_before" / name).exists():
            assert path.read_bytes().split(BEGIN.encode())[0].rstrip() == (LOCAL / "md_before" / name).read_bytes().rstrip(), name
        else:
            assert base.digest(path) == sha, name
    assert inventory["seeds"] == list(SEEDS) and inventory["new_training_stages"] == 13
    assert inventory["new_full_tests"] == 15 and inventory["fixed_updates"] == 1419
    return inventory


def initialize():
    inventory = verify_preservation()
    assert all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "Allocator/offline environment differs"
    LOCAL.mkdir(parents=True, exist_ok=True)
    sources = {name: base.digest(base.ROOT / "scripts" / name) for name in (*PINNED, "kd_ecqa_v2.py")}
    manifest = {"version": 2, "seeds": list(SEEDS), "fixed_epoch": 3, "updates": 1419,
        "selection": "fixed epoch 3, independent of dev/test", "sources": sources,
        "inventory_sha256": base.digest(INVENTORY), "environment": ENVIRONMENT,
        "safety": kd.gpu_safety.LIMITS, "training_implementation": "pinned v1 function; only 3 seed expressions parameterized",
        "training_ast_sha256": base.fingerprint(ast.dump(ast.parse(trainer_source()), include_attributes=False))}
    manifest["fingerprint"] = base.fingerprint(manifest)
    base.fixed_write(LOCAL / "run_manifest.json", manifest)
    for name in sources:
        dest = LOCAL / "run_source" / name
        if dest.exists():
            assert base.digest(dest) == sources[name]
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base.ROOT / "scripts" / name, dest)
    for name in MDS:
        dest = LOCAL / "md_before" / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base.ROOT / name, dest)
    return inventory, manifest


def protocol(seed, stage, teacher, manifest):
    # Preserve all original learning and numerical conditions; record v2 differences explicitly.
    spec = deepcopy(base.read(kd.OUT / stage / "training_protocol.json"))
    spec.pop("fingerprint")
    spec.update(version=2, seed=seed, data_seed=seed, selection="fixed epoch 3, independent of dev/test",
        teacher_initialization=teacher, student_initialization=kd.selected(kd.STAGES[stage][1])[0],
        runtime_environment=ENVIRONMENT, script_sha256=base.digest(__file__),
        run_manifest_fingerprint=manifest["fingerprint"], pinned_v1_script_sha256=PINNED["kd_ecqa.py"])
    spec["recipe"]["seed"] = seed
    spec["fingerprint"] = base.fingerprint(spec)
    return spec


def fixed_selection(epochs, fingerprint):
    assert len(epochs) == 3 and {r["epoch"] for r in epochs} == {1, 2, 3}
    assert all(r["global_step"] == 473*r["epoch"] and r["dev"]["n"] == 1079 for r in epochs)
    chosen = next(r for r in epochs if r["epoch"] == 3)
    return {"status": "selected", "fingerprint": fingerprint, "selected": chosen,
        "selection": "fixed epoch 3, independent of dev/test", "test_used_for_selection": False}


def reuse_training(engine, row, spec):
    stage = row["stage"]
    assert row["seed"] == 10 and row["reuse_training"] and stage != "via_fkl"
    source_protocol = base.read(kd.OUT / stage / "training_protocol.json")
    assert source_protocol["fingerprint"] == row["source_training_protocol_fingerprint"]
    assert source_protocol["teacher_initialization"]["adapter_sha256"] == spec["teacher_initialization"]["adapter_sha256"]
    assert source_protocol["student_initialization"]["adapter_sha256"] == spec["student_initialization"]["adapter_sha256"]
    for epoch in (1, 2, 3):
        src = kd.OUT / stage / f"epoch{epoch}.json"
        original = base.read(src)
        assert base.digest(base.ROOT / original["adapter_path"] / "adapter_model.safetensors") == original["adapter_sha256"]
        item = {**original, "fingerprint": spec["fingerprint"], "reused_from": str(src.relative_to(base.ROOT)),
            "source_fingerprint": original["fingerprint"],
            "dev_records_path": str((kd.LOCAL / stage / f"epoch{epoch}_dev.json").relative_to(base.ROOT))}
        base.fixed_write(engine.OUT / stage / f"epoch{epoch}.json", item)
    summary = base.read(kd.OUT / stage / "training.json")
    assert summary["status"] == "completed" and summary["global_step"] == 1419 and summary["teacher_unchanged"]
    assert original["adapter_sha256"] == row["reuse_adapter_sha256"]
    base.fixed_write(engine.OUT / stage / "training.json", {**summary, "fingerprint": spec["fingerprint"],
        "reused_from": str((kd.OUT / stage / "training.json").relative_to(base.ROOT)),
        "new_training_seconds": 0, "source_fingerprint": summary["fingerprint"]})
    base.fixed_write(engine.OUT / stage / "probe.json", {"reused_from": str((kd.OUT / stage / "probe.json").relative_to(base.ROOT)),
        "source_sha256": base.digest(kd.OUT / stage / "probe.json")})


def validate_checkpoint(engine, stage, spec):
    """No GPU allocation; reject incomplete or foreign checkpoints before Trainer can resume."""
    import torch
    from safetensors.torch import load_file
    folder = engine.LOCAL / stage
    binding = folder / "checkpoint_protocol.json"
    base.fixed_write(binding, {"fingerprint": spec["fingerprint"], "seed": engine.EXPERIMENT_SEED,
        "teacher_sha256": spec["teacher_initialization"]["adapter_sha256"]})
    checkpoints = sorted((folder / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        return None
    checkpoint = checkpoints[-1]
    step = int(checkpoint.name.split("-")[-1])
    assert 0 < step <= 1419 and base.read(checkpoint / "trainer_state.json")["global_step"] == step
    adapter = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
    assert adapter and all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in adapter.values())
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    assert optimizer["state"] and all(int(v["step"]) == step and torch.isfinite(v["exp_avg"]).all()
        and torch.isfinite(v["exp_avg_sq"]).all() for v in optimizer["state"].values())
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    assert scheduler["last_epoch"] == step
    rng = torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=False)
    assert {"python", "numpy", "cpu", "cuda"} <= rng.keys()
    files = {p.name: base.digest(p) for p in checkpoint.iterdir() if p.is_file()}
    record = {"status": "passed", "checkpoint": str(checkpoint.relative_to(base.ROOT)), "step": step,
        "fingerprint": spec["fingerprint"], "files_sha256": files, "verified_utc": now()}
    base.write(folder / "latest_checkpoint_verification.json", record)
    return record


def raw_records(folder, rows, fingerprint):
    records = [base.read(folder / (r["id"]+".json")) for r in rows]
    for item, row in zip(records, rows):
        assert (item["id"], item["gold"], item["fingerprint"]) == (row["id"], row["gold"], fingerprint)
        assert 0 < item["output_tokens"] <= 256
        assert base.inspect_response(item["response"]) == {k: item[k] for k in ("predicted", "explanation_present", "format_valid")}
    return records


def test_stage(engine, row, data, data_spec, selection):
    stage, name = row["stage"], row["student"]
    public = engine.OUT / stage
    result_path = public / "sft" / f"{name}.json"
    if result_path.exists():
        result = base.read(result_path)
    elif row["reuse_test"]:
        assert row["seed"] == 10 and row["reuse_training"]
        src = kd.OUT / stage / "sft" / f"{name}.json"
        old_protocol = base.read(kd.OUT / stage / "protocols" / f"{name}_sft_test_256.json")
        assert old_protocol["adapter_sha256"] == selection["selected"]["adapter_sha256"]
        old_result = base.read(src)
        assert old_protocol["fingerprint"] == old_result["fingerprint"]
        base.fixed_write(public / "protocols" / f"{name}_sft_test_256.json", old_protocol)
        result = {**old_result, "reused_from": str(src.relative_to(base.ROOT)),
            "v2_training_fingerprint": selection["fingerprint"], "new_generation_seconds": 0}
        base.fixed_write(result_path, result)
    else:
        status("starting_test", seed=row["seed"], experiment=stage, endpoint_epoch=3)
        result = engine.evaluate_stage(stage, data, data_spec, selection)
    protocol_path = public / "protocols" / f"{name}_sft_test_256.json"
    saved_protocol = base.read(protocol_path)
    assert saved_protocol["adapter_sha256"] == selection["selected"]["adapter_sha256"]
    assert result["status"] == "completed" and result["dataset_fingerprint"] == data_spec["fingerprint"]
    assert saved_protocol["fingerprint"] == result["fingerprint"] and result["overall"]["n"] == 2194
    assert saved_protocol["max_new_tokens"] == 256 and saved_protocol["batch_size"] == 1
    folder = (kd.LOCAL if row["reuse_test"] else engine.LOCAL) / stage / "responses" / (name+"_sft") / "test/256"
    records = raw_records(folder, data["test"], result["fingerprint"])
    assert base.metrics(records) == result["overall"]
    base.write(public / "completion_verification.json", {"status": "passed", "fingerprint": result["fingerprint"],
        "adapter_sha256": selection["selected"]["adapter_sha256"], "n": 2194,
        "raw_answers_reparsed": True, "metrics_recomputed": True,
        "raw_directory": str(folder.relative_to(base.ROOT)),
        "raw_manifest_sha256": base.fingerprint({r["id"]: base.digest(folder / (r["id"]+".json")) for r in data["test"]})})
    queue_path = LOCAL / "completion_reports.json"
    queue = base.read(queue_path) if queue_path.exists() else {}
    queue.setdefault(f'{row["seed"]}:{stage}:{result["fingerprint"]}', {"seed": row["seed"], "stage": stage,
        "fingerprint": result["fingerprint"], "reported": False, "overall": result["overall"], "reused": row["reuse_test"]})
    base.write(queue_path, queue)
    report()
    print(json.dumps({"test_completed": stage, "seed": row["seed"], "overall": result["overall"]}), flush=True)


def report():
    """CPU-only append/replace of v2 sections; the completed v1 text remains byte-for-byte intact."""
    from filelock import FileLock
    LOCAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCAL / "report.lock"), timeout=60):
        lines = ["## 강화 통제 실험 v2 진행 기록", "", "고정 SFT 초기화, KD 3에포크·1,419 updates, 시드 10·20·30. v1의 검증 선정 결과와 별도 비교한다.",
            "완료된 dev 에포크만 아래에 기록한다. 주 시험 모델은 dev 점수와 관계없이 항상 3에포크다.", "",
            "| 시드 | 단계 | 에포크 | dev ACC(%) | Macro-F1 | NLL | Brier | 학습 처리 |", "|---|---|---:|---:|---:|---:|---:|---|"]
        tests = ["", "| 시드 | 단계 | 시험 상태 | ACC(%) | Macro-F1 | NLL | Brier | 설명 상한/2,194 |", "|---|---|---|---:|---:|---:|---:|---:|"]
        epochs_done = tests_done = 0
        for seed in SEEDS:
            for stage, (_, name, _, _) in kd.STAGES.items():
                folder = OUT / f"seed{seed}" / stage
                for ep in (1, 2, 3):
                    path = folder / f"epoch{ep}.json"
                    if not path.exists():
                        continue
                    item = base.read(path)
                    d = item["dev"]
                    assert d["n"] == 1079
                    lines.append(f'| {seed} | {stage} | {ep} | {100*d["accuracy"]:.4f} | {d["macro_f1"]:.6f} | {d["choice_nll"]:.6f} | {d["brier_score"]:.6f} | '+("v1 재사용" if "reused_from" in item else "v2 학습")+" |")
                    epochs_done += 1
                result_path = folder / "sft" / f"{name}.json"
                verification = folder / "completion_verification.json"
                if result_path.exists() and verification.exists():
                    result = base.read(result_path)
                    assert result["fingerprint"] == base.read(verification)["fingerprint"]
                    d = result["overall"]
                    tests.append(f'| {seed} | {stage} | 전체 검증 완료'+(" (재사용)" if "reused_from" in result else "")+f' | {100*d["accuracy"]:.4f} | {d["macro_f1"]:.6f} | {d["choice_nll"]:.6f} | {d["brier_score"]:.6f} | {d["length_cap_count"]} |')
                    tests_done += 1
                else:
                    tests.append(f"| {seed} | {stage} | 미완료 | — | — | — | — | — |")
        lines += tests + ["", f"완료 기록: dev {epochs_done}/54개 에포크, 전체 시험 {tests_done}/18개 단계 (재사용 포함).",
            "NLL·Brier가 주 지표이고 ACC·Macro-F1은 보조 지표다. 미완료 시험은 최종 성능으로 해석하지 않는다.",
            "3시드 평균·표본 표준편차·경로 차이·상호작용·대응 검정은 전체 시험 완료 후 계산한다.",
            "총 계산량은 중간 교사 학습을 포함하므로 직접 경로와 같지 않다. 고정 SFT 초기화에서의 제한된 시드 반복이며 설명 사실성은 평가하지 않는다.", ""]
        content = "\n".join(lines)
        for name in MDS:
            path = base.ROOT / name
            text = path.read_bytes()
            if BEGIN.encode() in text:
                assert text.count(BEGIN.encode()) == text.count(END.encode()) == 1
                head, rest = text.split(BEGIN.encode())
                _, tail = rest.split(END.encode())
            else:
                head, tail = text + b"\n\n", b"\n"
            updated = head + (BEGIN+"\n"+content+END).encode("utf-8") + tail
            temporary = path.with_suffix(".md.tmp")
            temporary.write_bytes(updated)
            temporary.replace(path)
        base.write(LOCAL / "progress.json", {"dev_epochs_completed": epochs_done, "full_tests_completed": tests_done, "updated_utc": now()})


def run():
    from filelock import FileLock
    import torch
    from transformers import AutoTokenizer
    LOCAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCAL / "worker.lock"), timeout=0):
        if (LOCAL / "RUN_DISABLED.json").exists():
            raise RuntimeError("Inspect and preserve the v2 stop marker before recovery")
        status("waiting_gpu_lock")
        with FileLock(str(base.ROOT / "local/gpu.lock"), timeout=-1):
            try:
                inventory, manifest = initialize()
                kd.verify_probe()
                data, data_spec = base.prepare()
                assert data_spec["fingerprint"] == inventory["dataset_fingerprint"]
                assert [len(data[k]) for k in ("train", "dev", "test")] == [7566, 1079, 2194]
                tokenizer = AutoTokenizer.from_pretrained(base.snapshots()["qwen35_4b"], local_files_only=True, trust_remote_code=False)
                features = None
                for seed in SEEDS:
                    engine = engine_for(seed)
                    for stage, (teacher_name, _, _, parent) in kd.STAGES.items():
                        row = next(r for r in inventory["rows"] if r["seed"] == seed and r["stage"] == stage)
                        teacher = base.read(engine.OUT / parent / "selected.json")["selected"] if parent else kd.selected(teacher_name)[0]
                        if parent:
                            assert teacher["epoch"] == 3 and teacher["global_step"] == 1419
                        assert base.digest(base.ROOT / teacher["adapter_path"] / "adapter_model.safetensors") == teacher["adapter_sha256"]
                        spec = protocol(seed, stage, teacher, manifest)
                        base.fixed_write(engine.OUT / stage / "training_protocol.json", spec)
                        status("preparing_stage", seed=seed, experiment=stage)
                        if row["reuse_training"]:
                            reuse_training(engine, row, spec)
                        else:
                            validate_checkpoint(engine, stage, spec)
                            if features is None:
                                features = [kd.sft.encode(r, tokenizer, 256) for r in data["train"]]
                            engine.train_stage(stage, tokenizer, data, features, spec, base.ROOT / teacher["adapter_path"])
                        epochs = [base.read(engine.OUT / stage / f"epoch{ep}.json") for ep in (1, 2, 3)]
                        selection = fixed_selection(epochs, spec["fingerprint"])
                        base.fixed_write(engine.OUT / stage / "selected.json", selection)
                        report()
                        test_stage(engine, row, data, data_spec, selection)
                status("all_tests_completed_analysis_pending", training_stages=18, final_students=12)
                report()
            except Exception as error:
                base.write(LOCAL / "RUN_DISABLED.json", {"reason": str(error), "error_type": type(error).__name__,
                    "created_utc": now(), "automatic_restart": False})
                status("failed", reason=str(error), error_type=type(error).__name__)
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "report", "check"))
    action = parser.parse_args().action
    if action == "check":
        inventory = verify_preservation()
        trainer_source()
        print(json.dumps({"status": "passed", "preserved_files": len(inventory["preserved_sha256"]), "cuda_used": False}))
    else:
        {"run": run, "report": report}[action]()
