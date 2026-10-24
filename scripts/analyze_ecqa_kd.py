"""Verify the completed ECQA study and render its CPU-only final analysis."""
from datetime import datetime, timezone
import hashlib
import json
import math
import re

import ecqa_study as base
import kd_ecqa as kd
import render_results

LOCAL = base.ROOT / "local/ecqa_kd_v1"
OUT = base.OUT / "kd"
SOURCE_SHA = "cbf1b65f43168e46ebe585726e797a25f28137b318732dd54f8cb1370481b895"
TRAIN_FPS = dict(zip(kd.STAGES, [
    "e3a3449c59ee37c871514301da3b27c79af64f3a03423476fff2e8c6b09cafef",
    "32979a9a7377616e48806f59bf1912c376eceac68bd92860fa4815c1cdcd9406",
    "d9caf7a5344fe92863a267d9e0ec84d521658639d63e42a0277e03b202da12ad",
    "85bfb89a537e980f67173e1490425ce23397acf1103ad376a42329b75226cc64",
    "cb6c8bf138c64268f2e0e90e2920476ecfb7c58fcff8a5c6bb3a2019da33ab4b",
    "3a6767cc8d6220a20aea2cd4a06ab87564bdcc5340eb365a22c966b93493ef85",
]))
CONDITIONS = ["sft_08b", "direct_fkl", "direct_rkl", "via_fkl", "via_rkl"]
LABELS = ["0.8B SFT", "직접 FKL", "직접 RKL", "중간 FKL", "중간 RKL"]


def fingerprint_checked(path):
    value = base.read(path)
    assert value["fingerprint"] == base.fingerprint({k: v for k, v in value.items() if k != "fingerprint"}), path
    return value


def teacher_transfer(teacher, student):
    pair = kd.paired_counts(teacher, student)
    correct = pair["both_correct"] + pair["first_only_correct"]
    wrong = pair["both_wrong"] + pair["second_only_correct"]
    assert correct + wrong == pair["n"]
    assert pair["same_wrong_answer"] <= pair["both_wrong"]
    return {**pair, "teacher_correct": correct, "teacher_wrong": wrong,
            "correct_retention_rate": pair["both_correct"] / correct if correct else None,
            "same_error_transfer_rate": pair["same_wrong_answer"] / wrong if wrong else None,
            "teacher_error_corrected_rate": pair["second_only_correct"] / wrong if wrong else None}


def verify_records(name, public, folder, test, dataset_fp):
    result = base.read(public / "sft" / f"{name}.json")
    protocol = fingerprint_checked(public / "protocols" / f"{name}_sft_test_256.json")
    assert result["status"] == "completed" and result["overall"]["n"] == 2194
    assert result["fingerprint"] == protocol["fingerprint"]
    assert result["dataset_fingerprint"] == dataset_fp
    expected = {"n": 2194, "split": "test", "max_new_tokens": 256, "batch_size": 1,
                "thinking": False, "do_sample": False, "num_beams": 1, "truncation": False,
                "quantization": None, "test_used_for_selection": False, "dataset_fingerprint": dataset_fp}
    assert all(protocol[k] == v for k, v in expected.items())
    files = list(folder.glob("*.json"))
    assert {p.stem for p in files} == {r["id"] for r in test} and len(files) == 2194
    records, hashes = [], {}
    for row in test:
        path = folder / (row["id"] + ".json")
        data = path.read_bytes()
        item = json.loads(data)
        hashes[row["id"]] = hashlib.sha256(data).hexdigest()
        assert item["id"] == row["id"] and item["gold"] == row["gold"]
        assert item["fingerprint"] == protocol["fingerprint"] and 0 < item["output_tokens"] <= 256
        assert base.inspect_response(item["response"]) == {k: item[k] for k in ("predicted", "explanation_present", "format_valid")}
        assert item["hit_length_cap"] == (item["output_tokens"] == 256 and not item["ended_with_eos"])
        records.append(item)
    assert base.metrics(records) == result["overall"], name
    return result, protocol, records, base.fingerprint(hashes)


def costs(results, training):
    records = {stage: {"training_dev_current_session_seconds": training[stage]["seconds_this_session"],
                      "training_resumed": bool(training[stage]["resumed_from_checkpoint"]),
                      "prior_aborted_trainer_elapsed_seconds": 0,
                      "test_generation_seconds": results[stage]["overall"]["total_generation_seconds"]}
               for stage in kd.STAGES}
    evidence = []
    for stamp, stage, step in [("20261022T034736Z", "ta_fkl", 113),
                               ("20261022T035015Z", "ta_fkl", 0),
                               ("20261022T132429Z", "ta_rkl", 748)]:
        path = LOCAL / "recovery" / stamp / "training_worker_error.log"
        matches = re.findall(r"(\d+)/1419\s+\[([0-9:]+)[<,]", path.read_text(encoding="utf-8-sig"))
        assert matches and int(matches[-1][0]) == step
        elapsed = sum(int(v) * 60 ** i for i, v in enumerate(reversed(matches[-1][1].split(":"))))
        records[stage]["prior_aborted_trainer_elapsed_seconds"] += elapsed
        evidence.append({"stage": stage, "last_displayed_step": step, "trainer_elapsed_seconds": elapsed,
                         "log": path.relative_to(base.ROOT).as_posix(), "log_sha256": base.digest(path)})
    for value in records.values():
        value["recorded_training_dev_including_aborts_seconds"] = value["training_dev_current_session_seconds"] + value["prior_aborted_trainer_elapsed_seconds"]
        value["recorded_training_dev_plus_generation_seconds"] = value["recorded_training_dev_including_aborts_seconds"] + value["test_generation_seconds"]
    branches = {}
    for stage in CONDITIONS[1:]:
        stages = [stage] if stage.startswith("direct") else ["ta_" + stage.split("_")[1], stage]
        branches[stage] = {"stages": stages, **{k: sum(records[s][k] for s in stages) for k in
                          ["recorded_training_dev_including_aborts_seconds", "test_generation_seconds", "recorded_training_dev_plus_generation_seconds"]}}
    start = base.read(LOCAL / "recovery/20261022T034736Z/launch.json")["started_utc"]
    finish = results["via_rkl"]["completed_utc"]
    wall = (datetime.fromisoformat(finish) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
    measured = sum(x["recorded_training_dev_plus_generation_seconds"] for x in records.values())
    assert wall >= measured
    return {"by_stage": records, "by_branch": branches, "aborted_session_evidence": evidence,
            "recorded_training_dev_plus_generation_seconds": measured,
            "kd_launch_to_last_test_wall_seconds": wall,
            "kd_launch_utc": start, "last_test_completed_utc": finish,
            "scope": "Trainer time includes dev and recorded aborted attempts (integer-second progress logs), including repeated updates. Generation is saved per-question model generation only. Their sum excludes loading, probes, recovery diagnostics and idle/restart gaps. Wall span includes those within the KD launch-to-test interval, excludes earlier pair probes and SFT/baseline preparation, and is not measured GPU compute time."}


def figures(analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [analysis["final_conditions"][s] for s in CONDITIONS]
    names = ["SFT", "Direct FKL", "Direct RKL", "Via FKL", "Via RKL"]
    colors = ["#687787", "#3267a8", "#d77328", "#3267a8", "#d77328"]
    folder = OUT / "figures"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, key, scale, label, limits in zip(axes, ["accuracy", "macro_f1", "length_cap_count"],
            [100, 1, 1], ["Test ACC (%)", "Test Macro-F1", "256-token cap count / 2,194"],
            [(64.8, 65.9), (.648, .658), (-.4, 6.2)]):
        values = [v[key] * scale for v in rows]
        ax.scatter(range(5), values, c=colors, s=65)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.4f}" if key != "length_cap_count" else str(int(v)), (i, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
        ax.set(xticks=range(5), xticklabels=names, ylabel=label, ylim=limits, xlim=(-.6, 4.6))
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("ECQA: 0.8B SFT and four KD conditions; n=2,194; seed=10", fontsize=11)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"kd_final_conditions.{suffix}", dpi=180)
    plt.close(fig)
    effects = analysis["path_effects_pp"]
    fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
    vals = [effects[k] for k in ("delta_f", "delta_r", "delta_r_minus_delta_f")]
    ax.scatter(vals, range(3), c=[colors[1], colors[2], "#687787"], s=75)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.4f} pp", (v, i), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(yticks=range(3), yticklabels=["Via - Direct: FKL", "Via - Direct: RKL", "Interaction: delta R - delta F"],
           xlabel="Test ACC difference (percentage points)", xlim=(-.55, .12), ylim=(-.6, 2.65))
    ax.grid(axis="x", alpha=.2)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"kd_path_effects.{suffix}", dpi=180)
    plt.close(fig)


def markdown(analysis):
    rows = analysis["final_conditions"]
    lines = ["## ECQA 증류 최종 비교 (2026-10-24)", "",
             "여섯 단계 모두 3 epoch·1,419 updates·각 epoch dev 1,079건과 선정 adapter를 검증했고, 각 모델 test 2,194건의 첫 ANSWER를 원문에서 재파싱해 전체 지표를 재계산했다. 학습·평가 조건과 보호 기준은 유지했다.", "",
             "| 0.8B 조건 | 선정 epoch | Dev ACC (%) | Test ACC (%) | Macro-F1 | 설명 상한/2,194 | NLL ↓ | Brier ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stage, label in zip(CONDITIONS, LABELS):
        v = rows[stage]
        lines.append(f"| {label} | {v['selected_epoch']} | {100*v['dev_accuracy']:.4f} | {100*v['accuracy']:.4f} | {v['macro_f1']:.6f} | {v['length_cap_count']} | {v['choice_nll']:.6f} | {v['brier_score']:.6f} |")
    lines += ["", "설명 상한은 출력 256토큰에 도달하고 EOS로 종료되지 않은 건수다. 상한에 도달해도 유효한 첫 답은 유지했으며 재호출·제외·상한 확대는 하지 않았다. NLL/Brier는 ANSWER 위치의 A–E 조건부 분포이며 설명 사실성 점수가 아니다.", "",
              "![최종 다섯 조건](artifacts/ecqa/kd/figures/kd_final_conditions.png)", "",
              "| 경로 비교 | 중간−직접 ACC (%p) | 공통 정답 | 직접만 정답 | 중간만 정답 | 공통 오답 |",
              "|---|---:|---:|---:|---:|---:|"]
    for direction in ("fkl", "rkl"):
        p = analysis["paired_direct_vs_intermediate"][direction]
        lines.append(f"| {direction.upper()} | {p['second_minus_first_accuracy_pp']:+.6f} | {p['both_correct']} | {p['first_only_correct']} | {p['second_only_correct']} | {p['both_wrong']} |")
    e = analysis["path_effects_pp"]
    lines += ["", f"ΔF=ACC(중간 FKL)−ACC(직접 FKL)={e['delta_f']:+.6f}%p, ΔR=ACC(중간 RKL)−ACC(직접 RKL)={e['delta_r']:+.6f}%p, **ΔR−ΔF={e['delta_r_minus_delta_f']:+.6f}%p**다. 이는 같은 test 문항의 서술적 차이이며 학습 seed 간 변동을 검증한 결과는 아니다.", "",
              "이번 고정 조건에서는 직접 RKL의 ACC·Macro-F1이 가장 높았고, 중간 경로는 두 KL 모두 대응 직접 경로보다 ACC가 낮았다. 중간 FKL은 SFT와 정답 수가 같고 중간 RKL은 SFT보다 2건 많았다. FKL은 SFT보다 NLL/Brier가 개선됐지만 RKL은 두 지표가 악화되어 정답률과 확률 품질의 결론을 구분해야 한다.", "",
              "![경로 효과와 상호작용](artifacts/ecqa/kd/figures/kd_path_effects.png)", "",
              "| 교사→학생 단계 | 교사 정답 유지 / 교사 정답 | 같은 오답 전달 / 교사 오답 | 교사 오답 교정 / 교사 오답 |",
              "|---|---:|---:|---:|"]
    for stage, v in analysis["teacher_transfer"].items():
        lines.append(f"| {stage} ({v['teacher']}) | {v['both_correct']}/{v['teacher_correct']} ({100*v['correct_retention_rate']:.2f}%) | {v['same_wrong_answer']}/{v['teacher_wrong']} ({100*v['same_error_transfer_rate']:.2f}%) | {v['second_only_correct']}/{v['teacher_wrong']} ({100*v['teacher_error_corrected_rate']:.2f}%) |")
    lines += ["", "정답 유지는 교사가 맞힌 문항을 학생도 맞힌 비율이고, 같은 오답 전달은 교사가 틀린 문항에서 학생이 교사와 같은 선택지를 낸 비율이다. 학생이 다른 오답을 낸 경우는 같은 오답 전달에 포함하지 않는다. 이 일치는 학습으로 인한 인과적 전달이나 설명의 사실성을 입증하지 않는다.", "",
              "| KD 단계 | 현재 세션 train+dev (h) | 이전 중단 trainer (h) | train+dev 합 (h) | test 생성 (h) |",
              "|---|---:|---:|---:|---:|"]
    for stage, v in analysis["costs"]["by_stage"].items():
        lines.append(f"| {stage} | {v['training_dev_current_session_seconds']/3600:.3f} | {v['prior_aborted_trainer_elapsed_seconds']/3600:.3f} | {v['recorded_training_dev_including_aborts_seconds']/3600:.3f} | {v['test_generation_seconds']/3600:.3f} |")
    lines += ["", "| 최종 경로 | 포함 KD 단계 | train+dev (h) | test 생성 (h) | 관측 합 (h) |",
              "|---|---|---:|---:|---:|"]
    for stage, v in analysis["costs"]["by_branch"].items():
        lines.append(f"| {stage} | {' + '.join(v['stages'])} | {v['recorded_training_dev_including_aborts_seconds']/3600:.3f} | {v['test_generation_seconds']/3600:.3f} | {v['recorded_training_dev_plus_generation_seconds']/3600:.3f} |")
    c = analysis["costs"]
    lines += ["", f"관측 train/dev(중단 전 로그 포함)+test 생성 합은 {c['recorded_training_dev_plus_generation_seconds']/3600:.3f}시간이다. 중단 시도는 ta_fkl 1,228+6초, ta_rkl 9,161초를 보존 로그에서 추가했으며 재계산된 업데이트 비용도 포함한다. 이 합은 로딩·probe·복구 진단·대기 시간을 제외하므로 전체 소요나 GPU 계산시간으로 부르지 않는다. 최초 정규 KD launch부터 마지막 test 완료까지의 경과 시간은 {c['kd_launch_to_last_test_wall_seconds']/3600:.3f}시간으로, 그 구간의 로딩·중단·진단·대기는 포함하지만 앞선 SFT/기준선과 최초 pair probe는 제외한다. 경로별 합에는 중간모델의 전체 test도 포함하며 공통 SFT 초기화 비용은 포함하지 않았다.", "",
              "중간 경로는 최종 학생의 3 epoch 예산은 같아도 2B KD 학습·평가를 추가로 사용한다. 단일 seed·한 모델 계열·한 데이터셋이며, 설명 사실성은 미검토다. 추가 seed나 추가 탐색은 수행하지 않았다.", "",
              "[최종 집계 및 검증](artifacts/ecqa/kd/final_analysis.json)"]
    return "\n".join(lines) + "\n"


def main():
    toy_teacher = [dict(id=str(i), gold=g, predicted=p) for i, (g, p) in enumerate(zip("ABCD", "ABAB"))]
    toy_student = [dict(id=str(i), gold=g, predicted=p) for i, (g, p) in enumerate(zip("ABCD", "ACCB"))]
    toy = teacher_transfer(toy_teacher, toy_student)
    assert toy["teacher_correct"] == toy["teacher_wrong"] == 2
    assert toy["correct_retention_rate"] == toy["same_error_transfer_rate"] == toy["teacher_error_corrected_rate"] == .5
    assert base.read(LOCAL / "worker_status.json")["stage"] == "completed"
    assert not (LOCAL / "RUN_DISABLED.json").exists() and base.digest(base.ROOT / "scripts/kd_ecqa.py") == SOURCE_SHA
    preservation = base.read(LOCAL / "baseline_preservation.json")
    assert all(base.digest(base.ROOT / p) == sha for p, sha in preservation.items())
    test = base.read(base.ROOT / "local/ecqa_data_v1/test.json")
    dev = base.read(base.ROOT / "local/ecqa_data_v1/dev.json")
    dataset_fp = "82bc1dd2fc2c400ca1a4f779cd68e86093fd9f792fdab27c73426bece9015677"
    results, protocols, raw, hashes, training, selections = {}, {}, {}, {}, {}, {}
    for key, name in [("sft_4b", "qwen35_4b"), ("sft_2b", "qwen35_2b"), ("sft_08b", "qwen35_08b")]:
        folder = base.ROOT / "local/ecqa_baseline_v1/responses" / (name + "_sft") / "test/256"
        results[key], protocols[key], raw[key], hashes[key] = verify_records(name, base.OUT, folder, test, dataset_fp)
        selections[key] = kd.selected(name)[0]
        assert protocols[key]["adapter_sha256"] == selections[key]["adapter_sha256"]
        print(json.dumps({"verified_test": key, "n": 2194}), flush=True)
    for stage, (teacher, student, direction, parent) in kd.STAGES.items():
        public, folder = OUT / stage, LOCAL / stage
        spec = fingerprint_checked(public / "training_protocol.json")
        assert spec["fingerprint"] == TRAIN_FPS[stage] and spec["script_sha256"] == SOURCE_SHA
        for k, name in [("sft_helper_sha256", "sft_ecqa.py"), ("evaluator_sha256", "ecqa_study.py"), ("safety_sha256", "gpu_safety.py")]:
            assert spec[k] == base.digest(base.ROOT / "scripts" / name)
        assert spec["teacher_initialization"] == (selections[parent] if parent else kd.selected(teacher)[0])
        assert spec["student_initialization"] == kd.selected(student)[0]
        t = base.read(public / "training.json")
        assert t["status"] == "completed" and t["epochs"] == 3 and t["global_step"] == 1419 and t["teacher_unchanged"]
        assert t["fingerprint"] == spec["fingerprint"]
        candidates = []
        for ep in (1, 2, 3):
            candidate = base.read(public / f"epoch{ep}.json")
            data = base.read(folder / f"epoch{ep}_dev.json")
            assert candidate["fingerprint"] == data["fingerprint"] == spec["fingerprint"]
            assert candidate["global_step"] == ep * 473 and candidate["learning_rate"] == 5e-6
            assert len(data["rows"]) == 1079 and [(x["id"], x["gold"]) for x in data["rows"]] == [(x["id"], x["gold"]) for x in dev]
            scores = base.metrics(data["rows"])
            assert all(scores[k] == v for k, v in candidate["dev"].items())
            assert base.digest(base.ROOT / candidate["adapter_path"] / "adapter_model.safetensors") == candidate["adapter_sha256"]
            candidates.append(candidate)
        sel = base.read(public / "selected.json")
        assert sel["selected"] == kd.choose(candidates) and sel["test_used_for_selection"] is False
        probe = base.read(public / "probe.json")
        assert probe["fingerprint"] == spec["fingerprint"] and all(p["status"] == "passed" and p["load_update_save_restore"] == "passed" and p["probe_update_used_for_training"] is False for p in probe["probes"])
        response_folder = folder / "responses" / (student + "_sft") / "test/256"
        results[stage], protocols[stage], raw[stage], hashes[stage] = verify_records(student, public, response_folder, test, dataset_fp)
        assert protocols[stage]["training"] == sel and protocols[stage]["adapter_sha256"] == sel["selected"]["adapter_sha256"]
        verification = base.read(public / "completion_verification.json")
        assert verification["status"] == "passed" and verification["fingerprint"] == results[stage]["fingerprint"] and verification["n"] == 2194
        training[stage], selections[stage] = t, sel["selected"]
        print(json.dumps({"verified_stage": stage, "dev_epochs": 3, "test_n": 2194}), flush=True)
    common = ["n", "split", "ordered_ids_sha256", "dataset_fingerprint", "max_new_tokens", "instruction", "dtype", "quantization", "batch_size", "thinking", "do_sample", "num_beams", "truncation", "temperature", "eos_token_ids", "format", "choice_distribution", "tie_rule", "tokenizer_sha256", "script_sha256", "safety_sha256", "safety"]
    assert all(p[k] == protocols["sft_08b"][k] for p in protocols.values() for k in common)
    pairs = {d: kd.paired_counts(raw["direct_" + d], raw["via_" + d]) for d in ("fkl", "rkl")}
    effects = {"delta_f": pairs["fkl"]["second_minus_first_accuracy_pp"], "delta_r": pairs["rkl"]["second_minus_first_accuracy_pp"]}
    effects["delta_r_minus_delta_f"] = effects["delta_r"] - effects["delta_f"]
    net = lambda d: results["via_" + d]["overall"]["correct"] - results["direct_" + d]["overall"]["correct"]
    assert math.isclose(effects["delta_r_minus_delta_f"], 100 * (net("rkl") - net("fkl")) / 2194, abs_tol=1e-12)
    transfer = {}
    for stage, (teacher, _, _, parent) in kd.STAGES.items():
        teacher_key = parent or "sft_4b"
        transfer[stage] = {"teacher": teacher_key, **teacher_transfer(raw[teacher_key], raw[stage])}
    conditions = {s: {**results[s]["overall"], "selected_epoch": selections[s]["epoch"], "dev_accuracy": selections[s]["dev"]["accuracy"]} for s in CONDITIONS}
    analysis = {"status": "verified_completed", "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_fingerprint": dataset_fp,
                "single_seed": 10, "explanation_factuality_reviewed": False, "training_protocol_fingerprints": TRAIN_FPS,
                "evaluation_fingerprints": {k: v["fingerprint"] for k, v in results.items()}, "raw_sha256_manifest_fingerprints": hashes,
                "verified_test_rows_per_model": 2194, "verified_test_models": 9, "verified_dev_epochs": 18,
                "final_conditions": conditions, "paired_direct_vs_intermediate": pairs, "path_effects_pp": effects,
                "teacher_transfer": transfer, "costs": costs(results, training),
                "analysis_script_sha256": base.digest(__file__), "training_script_sha256": SOURCE_SHA}
    base.write(OUT / "final_analysis.json", analysis)
    kd.report()
    figures(analysis)
    content = markdown(analysis)
    for name in ("RESEARCH_PLAN.md", "RESEARCH_REPORT.md", "RESULTS_VISUALIZATION.md"):
        path = base.ROOT / name
        text = path.read_text(encoding="utf-8")
        if "<!-- KD_FINAL_START -->" not in text:
            text = text.replace("<!-- KD_STATUS_END -->", "<!-- KD_STATUS_END -->\n\n<!-- KD_FINAL_START -->\n\n<!-- KD_FINAL_END -->")
        text = text.replace("<!-- KD_FINAL_START -->\n<!-- KD_FINAL_END -->", "<!-- KD_FINAL_START -->\n\n<!-- KD_FINAL_END -->")
        text = render_results.update_section(text, "KD_FINAL", content)
        assert text.count(content) == 1, name
        text = text.replace("KD 네 조건의 전체 평가가 나오기 전에는 증류 경로나 KL 방향의 효과를 판단하지 않는다.", "KD 네 조건의 전체 평가·검증을 완료했으며, 최종 비교와 경로 효과는 위 표와 최종 집계에 제시했다.")
        temp = path.with_suffix(".md.tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)
    assert all(base.digest(base.ROOT / p) == sha for p, sha in preservation.items())
    assert base.digest(base.ROOT / "scripts/kd_ecqa.py") == SOURCE_SHA
    import torch
    assert not torch.cuda.is_initialized()
    print(json.dumps({"completed": True, "path_effects_pp": effects, "recorded_cost_hours": analysis["costs"]["recorded_training_dev_plus_generation_seconds"] / 3600,
                      "wall_hours": analysis["costs"]["kd_launch_to_last_test_wall_seconds"] / 3600}), flush=True)


if __name__ == "__main__":
    main()
