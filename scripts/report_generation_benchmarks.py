"""Report paired teacher/student generation scores without mixing in classification."""
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dolly_sft import ROOT, write_json
from dolly_study import LABELS, MODELS, result_dir
from evaluate_generation_benchmarks import load_completed
from prepare_generation_benchmarks import BENCHMARKS, OUT, PAIRS, prepared_protocol


def main():
    data = prepared_protocol()
    runs = {name: load_completed(name, data) for name in MODELS
            if (result_dir(name) / f"{name}_selected.json").exists()}
    complete = [name for name, run in runs.items() if len(run) == len(BENCHMARKS)]
    records = [{"model": name, "benchmark": key, "n": value["n"], "seed": 10, **value["metrics"]}
               for name, run in runs.items() for key, value in run.items()]
    fields = ["model", "benchmark", "n", "seed", "rouge1", "rouge2", "rougeL", "exact_match", "generated_tokens", "empty", "length_capped"]
    with (OUT / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    lines = ["# 생성 벤치마크의 교사·학생 비교", "",
             f"주 평가: 네 생성 benchmark. 완료 {len(complete)}/10모델. SST-2·BoolQ는 [보조 분류 진단](../dolly_sft_v1/REPORT.md)으로 분리한다.", "",
             "모든 모델에 동일 원문 ID·새 토큰 예산 256·생성 seed 10을 적용한다. 원 논문 그대로의 재현은 아니다. 단일 seed이므로 seed 간 SD와 오차막대를 표시하지 않는다. ROUGE는 정답과의 겹침을 측정하며 사실성이나 사람의 선호를 직접 측정하지 않는다.", "",
             "| 모델 | Dolly | Self-Instruct | S-NI (11_) | Vicuna | 네 과제 평균 |", "|---|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        cells = [f'{run[key]["metrics"]["rougeL"]:.2f}' if key in run else "대기" for key in BENCHMARKS]
        average = f'{statistics.mean(v["metrics"]["rougeL"] for v in run.values()):.2f}' if name in complete else "—"
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
            axis.text(index[1], index[0], f"{grid[index]:.2f}" if np.isfinite(grid[index]) else "pending", ha="center", va="center", color="white" if grid[index] > 60 else "black")
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
            axis.text(index[1], index[0], f"{value:+.2f}" if np.isfinite(value) else "pending", ha="center", va="center",
                      color="white" if np.isfinite(value) and abs(value) > .6 * bound else "black")
        figure.colorbar(plotted, ax=axis, label="Teacher − student ROUGE-L")
        figure.savefig(OUT / "teacher_student_gaps.png", dpi=180); plt.close(figure)
        lines += ["![교사−학생 생성 점수 차이](teacher_student_gaps.png)", ""]
    write_json(OUT / "paired_comparisons.json", {"data_protocol_fingerprint": data["fingerprint"], "comparisons": pairs})
    write_json(OUT / "completion.json", {"data_protocol_fingerprint": data["fingerprint"], "completed_models": complete,
                                         "completed_benchmarks": {name: list(run) for name, run in runs.items()}})
    (OUT / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Verified generation report: {len(complete)}/10 models complete", flush=True)


if __name__ == "__main__":
    main()
