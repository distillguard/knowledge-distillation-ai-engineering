"""Report paired teacher/student generation scores without mixing in classification."""
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dolly_sft import ROOT, write_json
from dolly_study import LABELS, result_dir
from dolly_scope import EXCLUDED_MODELS, MODELS, REQUIRED_BENCHMARKS
from evaluate_generation_benchmarks import load_completed
from prepare_generation_benchmarks import BENCHMARKS, OUT, PAIRS, prepared_protocol


def main():
    data = prepared_protocol()
    runs = {name: load_completed(name, data) for name in MODELS
            if (result_dir(name) / f"{name}_selected.json").exists()}
    complete = [name for name, run in runs.items() if all(key in run for key in REQUIRED_BENCHMARKS)]
    extended_complete = [name for name, run in runs.items() if len(run) == len(BENCHMARKS)]
    records = [{"model": name, "benchmark": key, "n": value["n"], "seed": 10, **value["metrics"]}
               for name, run in runs.items() for key, value in run.items()]
    fields = ["model", "benchmark", "n", "seed", "rouge1", "rouge2", "rougeL", "exact_match", "generated_tokens", "empty", "length_capped"]
    with (OUT / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    lines = ["# Dolly 예비 생성 평가와 기존 외부 평가", "",
             f"예비 주 평가: Dolly 500문항. 완료 {len(complete)}/{len(MODELS)}모델. 외부 3종과 SST-2·BoolQ의 새 평가는 후보 축소 후 진행한다. [기존 보조 분류 결과](../dolly_sft_v1/REPORT.md)는 보존한다.", "",
             "현재는 4계열·8모델의 예비 비교다. Qwen 두 모델은 2026-10-16 범위 축소로 제외했고 기존 SFT·부분 평가 결과는 보존한다. 후보군을 좁힌 뒤 원문 기반 SFT 탐색을 적용하며, 본 KD는 한 조합에서 진행한다. 아래 외부 점수는 이미 완료한 결과이며 미실행 외부 항목은 후속으로 표시한다.", "",
             "모든 모델에 동일 원문 ID·새 토큰 예산 256·생성 seed 10을 적용한다. 원 논문 그대로의 재현은 아니다. 단일 seed이므로 seed 간 SD와 오차막대를 표시하지 않는다. ROUGE는 정답과의 겹침을 측정하며 사실성이나 사람의 선호를 직접 측정하지 않는다.", "",
             "이 보고서는 SFT 후 모델 간의 1차 후보 비교다. NLL은 각 모델의 체크포인트 선택이며 최종 조합의 순위가 아니다. METEOR·BERTScore와 원본 대비 SFT 전후의 생성 평가는 아직 미실행이다. 최종 조합은 보완 평가·오류 분석·후보 KD 실행 점검 후 선정한다. [단계별 완료 기준](../../docs/GENERATION_EVALUATION.md#조합-선정까지의-단계와-완료-기준)을 따른다. ROUGE-1/2와 exact match 등 현재 계산한 전체 지표는 [scores.csv](scores.csv)에 있다.", "",
             "| 모델 | Dolly | Self-Instruct | S-NI (11_) | Vicuna | 네 과제 평균 |", "|---|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        cells = [f'{run[key]["metrics"]["rougeL"]:.2f}' if key in run else ("대기" if key in REQUIRED_BENCHMARKS else "후속") for key in BENCHMARKS]
        average = f'{statistics.mean(v["metrics"]["rougeL"] for v in run.values()):.2f}' if name in extended_complete else "—"
        lines.append(f'| {LABELS[name]} | {" | ".join(cells)} | {average} |')
    lines += ["", "과제 평균은 네 benchmark 평균에 같은 가중치를 준 값이며 전체 문항을 합친 평균과 다르다. 미완료 모델은 평균을 계산하지 않는다. 교사의 자격에 임의의 최소 점수 차이를 두지 않는다.", "",
              "## 동일 문항의 교사−학생 차이", "", "| 조합 | Benchmark | 평균 ROUGE-L 차이 | 교사 우위 / 동률 / 학생 우위 문항 |", "|---|---|---:|---|"]
    pairs = []
    for teacher, student in PAIRS:
        if teacher not in runs or student not in runs:
            continue
        for key in BENCHMARKS:
            if key not in runs[teacher] or key not in runs[student]:
                continue
            values = []
            for name in (teacher, student):
                with (ROOT / runs[name][key]["per_item_csv"]).open(encoding="utf-8", newline="") as handle:
                    values.append(list(csv.DictReader(handle)))
            assert [r["id"] for r in values[0]] == [r["id"] for r in values[1]]
            differences = [float(a["rougeL"]) - float(b["rougeL"]) for a, b in zip(*values)]
            delta = statistics.mean(differences)
            assert abs(delta - (runs[teacher][key]["metrics"]["rougeL"] - runs[student][key]["metrics"]["rougeL"])) < 1e-10
            record = {"teacher": teacher, "student": student, "benchmark": key, "n": len(differences), "mean_rougeL_difference": delta,
                      "teacher_higher": sum(d > 0 for d in differences), "tie": sum(d == 0 for d in differences), "student_higher": sum(d < 0 for d in differences)}
            pairs.append(record)
            lines.append(f'| {LABELS[teacher]} → {LABELS[student]} | {BENCHMARKS[key]} | {delta:+.2f} | {record["teacher_higher"]} / {record["tie"]} / {record["student_higher"]} |')
    lines += ["", "우위 문항 수는 동일 문항의 ROUGE-L 비교이며 사람의 선호 승률이 아니다. 이 결과는 모델 선정 자료이므로 독립적인 최종 test로 취급하지 않는다.", "",
              "## 자료와 전처리", "", "| Benchmark | 공식 배포 행 수 | 공통 평가 행 수 | 제외 행 수 |", "|---|---:|---:|---:|"]
    for key, count in data["counts"].items():
        lines.append(f'| {BENCHMARKS[key]} | {count["source"]} | {count["included"]} | {count["excluded"]} |')
    lines += ["", "S-NI는 MiniLLM의 공식 `11_` 배포 구간을 사용한다. 다중 정답은 지표별 최고 점수를 사용한다. 학습·개발 집합과 정규화 instruction이 같은 문항 및 어느 모델에서든 native context를 넘는 문항은 출력 확인 전에 공통 제외한다. 문항·응답을 자르거나 성능에 따라 재선정하지 않는다. 중복 prompt는 원문대로 보존하고 수를 protocol에 기록한다. 의미상 누수 전체를 감사한 것은 아니다.", ""]
    if records:
        names = [n for n in MODELS if runs.get(n)]
        grid = np.array([[runs[n][key]["metrics"]["rougeL"] if key in runs[n] else np.nan for key in BENCHMARKS] for n in names])
        figure, axis = plt.subplots(figsize=(10, max(3.5, .55 * len(names) + 1.5)), layout="constrained")
        plotted = axis.imshow(np.ma.masked_invalid(grid), vmin=0, vmax=100, cmap="Blues", aspect="auto")
        axis.set(xticks=range(4), xticklabels=list(BENCHMARKS.values()), yticks=range(len(names)), yticklabels=[LABELS[n] for n in names], title="Primary generation evaluation · seed 10")
        for index in np.ndindex(grid.shape):
            axis.text(index[1], index[0], f"{grid[index]:.2f}" if np.isfinite(grid[index]) else ("pending" if list(BENCHMARKS)[index[1]] in REQUIRED_BENCHMARKS else "deferred"), ha="center", va="center", color="white" if grid[index] > 60 else "black")
        figure.colorbar(plotted, ax=axis, label="ROUGE-L F1 × 100")
        figure.savefig(OUT / "generation_scores.png", dpi=180); plt.close(figure)
        lines += ["![생성 ROUGE-L](generation_scores.png)", ""]
    if pairs:
        names = list(dict.fromkeys((r["teacher"], r["student"]) for r in pairs))
        lookup = {(r["teacher"], r["student"], r["benchmark"]): r["mean_rougeL_difference"] for r in pairs}
        grid = np.array([[lookup.get((*pair, key), np.nan) for key in BENCHMARKS] for pair in names])
        bound = float(np.nanmax(np.abs(grid))) or 1
        figure, axis = plt.subplots(figsize=(12, max(3.5, .65 * len(names) + 1.5)), layout="constrained")
        plotted = axis.imshow(np.ma.masked_invalid(grid), vmin=-bound, vmax=bound, cmap="RdBu", aspect="auto")
        axis.set(xticks=range(4), xticklabels=list(BENCHMARKS.values()), yticks=range(len(names)),
                 yticklabels=[LABELS[t] + " → " + LABELS[s] for t, s in names], title="Paired teacher − student gap · seed 10")
        for index in np.ndindex(grid.shape):
            value = grid[index]
            axis.text(index[1], index[0], f"{value:+.2f}" if np.isfinite(value) else ("pending" if list(BENCHMARKS)[index[1]] in REQUIRED_BENCHMARKS else "deferred"), ha="center", va="center",
                      color="white" if np.isfinite(value) and abs(value) > .6 * bound else "black")
        figure.colorbar(plotted, ax=axis, label="Teacher − student ROUGE-L")
        figure.savefig(OUT / "teacher_student_gaps.png", dpi=180); plt.close(figure)
        lines += ["![교사−학생 생성 점수 차이](teacher_student_gaps.png)", ""]
    write_json(OUT / "paired_comparisons.json", {"data_protocol_fingerprint": data["fingerprint"], "comparisons": pairs})
    write_json(OUT / "completion.json", {"data_protocol_fingerprint": data["fingerprint"], "study_models": list(MODELS),
                                         "excluded_models": list(EXCLUDED_MODELS), "required_benchmarks": list(REQUIRED_BENCHMARKS),
                                         "completed_models": complete, "completed_extended_models": extended_complete,
                                         "completed_benchmarks": {name: list(run) for name, run in runs.items()}})
    (OUT / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Verified generation report: {len(complete)}/{len(MODELS)} models complete", flush=True)


if __name__ == "__main__":
    main()
