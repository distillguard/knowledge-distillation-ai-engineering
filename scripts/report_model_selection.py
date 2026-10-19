"""Compare verified original/SFT generation results without inventing missing scores."""
import csv
import importlib.metadata
import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluate_supplementary_metrics as scoring
from dolly_study import LABELS, result_dir
from prepare_generation_benchmarks import PAIRS
from rouge_score import rouge_scorer

ROOT = scoring.ROOT
OUT = ROOT / "results/model_selection_v1"
DISPLAY = {"rougeL": "ROUGE-L", "meteor": "METEOR", "bertscore_f1": "BERTScore F1"}
ROUGE_COMPONENTS = ("rougeL_precision", "rougeL_recall", "rougeL_f1")


def rouge_components(rows, generated, primary_items):
    assert rows and [r["id"] for r in rows] == [r["id"] for r in generated] == [r["id"] for r in primary_items], "ROUGE component IDs differ"
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    items = []
    for row, output, primary in zip(rows, generated, primary_items):
        assert row["references"] and all(isinstance(r, str) and r.strip() for r in row["references"])
        assert isinstance(output["response"], str)
        scores = [scorer.score(ref, output["response"])["rougeL"] for ref in row["references"]]
        index = max(range(len(scores)), key=lambda i: scores[i].fmeasure)
        chosen = scores[index]
        assert math.isclose(100 * chosen.fmeasure, float(primary["rougeL"]), rel_tol=0, abs_tol=1e-12), (row["id"], "ROUGE-L F1 differs")
        items.append({"id": row["id"], "reference_index_0based": index, "reference_count": len(scores),
                      **dict(zip(ROUGE_COMPONENTS, (100 * chosen.precision, 100 * chosen.recall, 100 * chosen.fmeasure)))})
    return items


def component_means(items):
    return {key: statistics.mean(float(row[key]) for row in items) for key in ROUGE_COMPONENTS}


def paired(left, right, metric):
    assert left and [r["id"] for r in left] == [r["id"] for r in right], "Paired IDs differ"
    if not all(metric in row for row in left + right):
        return None
    differences = [float(b[metric]) - float(a[metric]) for a, b in zip(left, right)]
    return {"n": len(differences), "difference": statistics.mean(differences),
            "higher": sum(v > 0 for v in differences), "tie": sum(v == 0 for v in differences),
            "lower": sum(v < 0 for v in differences)}


def verified_sources(names, data):
    runs = {}
    assets = scoring.read(scoring.OUT / "assets.json")
    assert assets["fingerprint"] == scoring.fingerprint(assets)
    for phase in ("original", "sft"):
        for name in names:
            for key in scoring.BENCHMARKS:
                parent = ROOT / "results" / ("original_model_evaluation_v1" if phase == "original" else "generation_benchmarks_v1") / name
                if not (parent / f"{key}.json").exists():
                    continue
                rows = scoring.records(ROOT / "tmp/generation_benchmarks_v1/prepared" / f"{key}.jsonl")
                identity, primary, items, generated = scoring.source(name, phase, key, data, rows)
                precision = scoring.read(result_dir(name) / "protocol.json")["precision"][name]["base_weights"]
                components = rouge_components(rows, generated, items)
                assert math.isclose(component_means(components)["rougeL_f1"], primary["metrics"]["rougeL"], rel_tol=0, abs_tol=1e-12)
                run = {"primary": primary, "items": items, "source": identity, "precision": precision,
                       "rouge_components": components, "supplementary": None}
                folder = scoring.OUT / phase / name / key
                if (folder / "result.json").exists():
                    spec = scoring.read(folder / "protocol.json")
                    assert spec["fingerprint"] == scoring.fingerprint(spec)
                    assert spec["source"] == identity and spec["assets_fingerprint"] == assets["fingerprint"]
                    assert spec["data_protocol_fingerprint"] == data["fingerprint"]
                    assert spec["scoring_script_sha256"] == scoring.sha(scoring.__file__)
                    assert (spec["model"], spec["phase"], spec["benchmark"]) == (name, phase, key)
                    result = scoring.load_completed(folder, spec, rows, generated)
                    assert result["primary_metrics"] == primary["metrics"]
                    run["supplementary"] = result
                    with (ROOT / result["per_item_csv"]).open(encoding="utf-8", newline="") as handle:
                        run["items"] = list(csv.DictReader(handle))
                runs[phase, name, key] = run
    return runs


def values(run):
    result = dict(run["primary"]["metrics"])
    if run.get("rouge_components"):
        result.update(component_means(run["rouge_components"]))
    if run["supplementary"]:
        result.update(run["supplementary"]["metrics"])
        result.update(run["supplementary"]["diversity"])
    return result


def external_grid(runs, ordered, metric):
    return np.array([[values(runs[phase, name, key]).get(metric, np.nan)
                      if (phase, name, key) in runs else np.nan
                      for key in scoring.BENCHMARKS[1:]] for phase, name in ordered], dtype=float)


def main():
    data = scoring.prepared_data()
    names = scoring.read(ROOT / "results/generation_benchmarks_v1/completion.json")["study_models"]
    runs = verified_sources(names, data)
    OUT.mkdir(parents=True, exist_ok=True)
    component_rows = [{"model": name, "phase": phase, "benchmark": key, "seed": 10, **item}
                      for (phase, name, key), run in runs.items() for item in run["rouge_components"]]
    component_path = OUT / "rougeL_components_per_item.csv"
    with component_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader(); writer.writerows(component_rows)
    scoring.write_json(OUT / "rougeL_components.json", {
        "data_protocol_fingerprint": data["fingerprint"], "scale": "all P/R/F1 are multiplied by 100",
        "aggregation": "arithmetic mean of per-item precision, recall and F1 separately; mean F1 is not F1 of mean P/R",
        "reference_selection": "same reference maximizing ROUGE-L F1; first reference on ties; do not maximize P and R independently",
        "tokenizer": "Google ROUGE default tokenizer with use_stemmer=True; same full reference/response as primary scoring",
        "rouge_score_version": importlib.metadata.version("rouge-score"),
        "rouge_scorer_sha256": scoring.sha(rouge_scorer.__file__), "report_script_sha256": scoring.sha(__file__),
        "per_item_csv": str(component_path.relative_to(ROOT)).replace("\\", "/"),
        "per_item_csv_sha256": scoring.sha(component_path), "n_scored_responses": len(component_rows),
        "sample_sd_across_generation_seeds": None,
        "conditions": [{"model": name, "phase": phase, "benchmark": key, "n": len(run["rouge_components"]),
                        "source": run["source"], **component_means(run["rouge_components"])}
                       for (phase, name, key), run in runs.items()]})
    scores = [{"model": name, "phase": phase, "precision": run["precision"], "benchmark": key, "n": run["primary"]["n"],
               "seed": 10, **values(run)} for (phase, name, key), run in runs.items()]
    fields = list(dict.fromkeys(field for row in scores for field in row))
    with (OUT / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(scores)
    changes, gaps = [], []
    for name in names:
        for key in scoring.BENCHMARKS:
            if all((phase, name, key) in runs for phase in ("original", "sft")):
                left, right = (runs[phase, name, key]["items"] for phase in ("original", "sft"))
                for metric in DISPLAY:
                    result = paired(left, right, metric)
                    if result is not None:
                        changes.append({"model": name, "benchmark": key, "metric": metric, **result})
    for teacher, student in PAIRS:
        for phase in ("original", "sft"):
            for key in scoring.BENCHMARKS:
                if all((phase, name, key) in runs for name in (teacher, student)):
                    left, right = (runs[phase, name, key]["items"] for name in (student, teacher))
                    for metric in DISPLAY:
                        result = paired(left, right, metric)
                        if result is not None:
                            gaps.append({"teacher": teacher, "student": student, "phase": phase,
                                         "benchmark": key, "metric": metric, **result})
    scoring.write_json(OUT / "comparisons.json", {"data_protocol_fingerprint": data["fingerprint"],
                       "sources": [{"phase": phase, "model": name, "benchmark": key, **run["source"]}
                                   for (phase, name, key), run in runs.items()],
                       "sft_minus_original": changes, "teacher_minus_student": gaps,
                       "sample_sd_across_generation_seeds": None, "automatic_pair_selection_performed": False})
    coverage = {phase: {"generation": sum((phase, name, "dolly") in runs for name in names),
                        "supplementary": sum(bool(runs.get((phase, name, "dolly"), {}).get("supplementary")) for name in names)}
                for phase in ("original", "sft")}
    assert all(0 <= v["supplementary"] <= v["generation"] <= len(names) for v in coverage.values())
    lines = ["# 원본과 Dolly SFT의 생성 평가 비교", "",
             f'완료: 원본 Dolly {coverage["original"]["generation"]}/{len(names)}, SFT Dolly {coverage["sft"]["generation"]}/{len(names)}, 보완 채점 원본 {coverage["original"]["supplementary"]}/{len(names)}·SFT {coverage["sft"]["supplementary"]}/{len(names)}. 최종 조합은 후보 후속 검증 후 선정한다.', "",
             "동일 원문 ID·prompt·모델별 정밀도·생성 seed 10·새 토큰 상한 256의 결과다. 원본은 우리 Dolly SFT 전 checkpoint이며 Gemma-IT는 이미 instruction tuning을 거친 모델이다. 미완료 점수는 공란으로 두고 완료한 응답·CSV·명세 hash와 지표를 재검산한다.", "",
             "ROUGE-L·METEOR·raw BERTScore F1은 ×100이다. Distinct-4는 corpus 고유/전체 4-gram 비율이며 클수록 사실성이나 성능이 좋다는 뜻이 아니다. 단일 seed이므로 seed 간 SD·오차막대는 없다. 아래 결과만으로 최종 조합을 자동 선정하지 않는다.", "",
             "| 모델 | 단계 | Dolly ROUGE-L | METEOR | BERTScore F1 | Distinct-4 |",
             "|---|---|---:|---:|---:|---:|"]
    for name in names:
        for phase in ("original", "sft"):
            run = runs.get((phase, name, "dolly"))
            scores_for_run = values(run) if run else {}
            cells = [f"{scores_for_run[m]:.2f}" if scores_for_run.get(m) is not None else "—" for m in DISPLAY]
            distinct = scores_for_run.get("distinct4")
            lines.append(f'| {LABELS[name]} | {phase} | {" | ".join(cells)} | {distinct:.4f} |' if distinct is not None
                         else f'| {LABELS[name]} | {phase} | {" | ".join(cells)} | — |')
    lines += ["", "## ROUGE-L Precision·Recall·F1", "",
              "위 ROUGE-L은 문항별 F1 평균 ×100이다. 25.92는 원래 척도에서 0.2592이며 정답률이 아니다. 아래 P·R·F1도 모두 ×100으로 표시한다. 문항마다 P·R·F1을 구한 뒤 각각 평균하며, 평균 P·R의 조화평균을 평균 F1로 대체하지 않는다.", "",
              "기존 scorer와 같은 Google ROUGE tokenizer·Porter stemming을 사용하며 reference와 응답 전체를 유지했다. 다중 reference는 F1을 최대화한 같은 reference의 P/R을 사용한다. 동률이면 첫 reference를 사용한다. 모든 문항의 F1을 기존 결과 CSV와 대조했고 추가 학습·생성은 없다. P/R은 순서가 유지되는 어휘 일치 비율이며 내용의 정오나 사실성을 판정하지 않는다.", "",
              "| 모델 | 단계 | Dolly n | Precision | Recall | F1 |", "|---|---|---:|---:|---:|---:|"]
    for name in names:
        for phase in ("original", "sft"):
            run = runs.get((phase, name, "dolly"))
            if run:
                mean = component_means(run["rouge_components"])
                lines.append(f'| {LABELS[name]} | {phase} | {len(run["rouge_components"])} | {mean["rougeL_precision"]:.2f} | {mean["rougeL_recall"]:.2f} | {mean["rougeL_f1"]:.2f} |')
    lines += ["", "완료된 외부 benchmark의 P/R/F1도 [전체 요약 CSV](scores.csv)와 [구성값·설정 JSON](rougeL_components.json)에 있다. [문항별 P/R/F1 CSV](rougeL_components_per_item.csv)에는 텍스트 원문 대신 ID·선택 reference 위치·구성값을 기록한다.", "",
              "정의와 구현: [Google ROUGE scorer](https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py).", ""]
    external_names = [name for name in names if any((phase, name, key) in runs
                      for phase in ("original", "sft") for key in scoring.BENCHMARKS[1:])]
    if external_names:
        lines += ["", "## 후보의 외부 생성 평가", "",
                  "완료된 조건만 표에 수록한다. 대시와 그림의 pending은 생성 또는 해당 보완 채점이 미완료라는 뜻이다. 전체 reference의 ROUGE·METEOR와 BERTScore의 길이 제한을 구분하며 benchmark 간 단순 평균으로 순위를 만들지 않는다.", "",
                  "| 모델 | 단계 | 자료 | n | ROUGE-L | METEOR | BERTScore F1 | Distinct-4 |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for name in external_names:
            for phase in ("original", "sft"):
                for key in scoring.BENCHMARKS[1:]:
                    run = runs.get((phase, name, key))
                    if run is None:
                        continue
                    observed = values(run)
                    cells = [f"{observed[m]:.2f}" if observed.get(m) is not None else "—" for m in DISPLAY]
                    distinct = observed.get("distinct4")
                    tail = f"{distinct:.4f}" if distinct is not None else "—"
                    lines.append(f'| {LABELS[name]} | {phase} | {key} | {run["primary"]["n"]} | {" | ".join(cells)} | {tail} |')
        ordered = [(phase, name) for name in external_names for phase in ("original", "sft")]
        figure, axes = plt.subplots(1, 3, figsize=(16, max(4, .48 * len(ordered) + 1.4)), layout="constrained")
        for index, (axis, (metric, title)) in enumerate(zip(axes, DISPLAY.items())):
            grid = external_grid(runs, ordered, metric)
            plotted = axis.imshow(np.ma.masked_invalid(grid), vmin=0, vmax=100, cmap="Blues", aspect="auto")
            axis.set(xticks=range(3), xticklabels=scoring.BENCHMARKS[1:], yticks=range(len(ordered)),
                     yticklabels=[LABELS[name] + " / " + phase for phase, name in ordered] if index == 0 else [], title=title)
            for y, x in np.ndindex(grid.shape):
                value = grid[y, x]
                axis.text(x, y, f"{value:.2f}" if np.isfinite(value) else "pending", ha="center", va="center",
                          color="white" if np.isfinite(value) and value > 60 else "black", fontsize=9)
        figure.colorbar(plotted, ax=axes, label="Metric × 100")
        figure.suptitle("Candidate external generation · original / SFT · seed 10 · no cross-benchmark averaging")
        figure.savefig(OUT / "external_comparison.png", dpi=180); plt.close(figure)
        lines += ["", "![후보 외부 생성 평가](external_comparison.png)", ""]
    for title, records in (("SFT − 원본", changes), ("교사 − 학생", gaps)):
        lines += ["", f"## {title}", "", "| 모델·조합 | 단계 | 자료 | 지표 | 평균 차이 | 증가 / 동률 / 감소 문항 |",
                  "|---|---|---|---|---:|---|"]
        for record in records:
            label = LABELS[record["model"]] if "model" in record else LABELS[record["teacher"]] + " → " + LABELS[record["student"]]
            lines.append(f'| {label} | {record.get("phase", "sft−original")} | {record["benchmark"]} | {DISPLAY[record["metric"]]} | {record["difference"]:+.2f} | {record["higher"]} / {record["tie"]} / {record["lower"]} |')
    lines += ["", "문항 수는 해당 자동 지표의 대응 차이이며 사람의 선호 승률이 아니다. 교사−학생 차이와 SFT−원본 차이, 이후 본 KD 학생−동일 SFT 학생의 개선을 구분한다.", "",
              "## BERTScore 길이 감사", "", "| 모델 | 단계 | 자료 | reference 영향 문항 | 응답 영향 문항 | 전체 reference 수 |",
              "|---|---|---|---:|---:|---:|"]
    for (phase, name, key), run in runs.items():
        if run["supplementary"]:
            audit = run["supplementary"]["bert_length_audit"]
            lines.append(f'| {LABELS[name]} | {phase} | {key} | {audit["items_with_truncated_reference"]} | {audit["items_with_truncated_response"]} | {audit["reference_count"]} |')
    lines += ["", "BERTScore encoder의 native 상한은 특수 토큰 포함 512다. METEOR·ROUGE는 전체 reference를 사용한다. 세부 토큰 수·고유/전체 4-gram 수는 각 결과 CSV·JSON에 남긴다.", ""]
    if runs:
        labels = [LABELS[name] + (" [NF4]" if "NF4" in scoring.read(result_dir(name) / "protocol.json")["precision"][name]["base_weights"] else " [BF16]") for name in names]
        figure, axes = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
        for axis, (metric, title) in zip(axes, DISPLAY.items()):
            for phase, offset, color in (("original", -.19, "#8b98a5"), ("sft", .19, "#2878b5")):
                heights = [values(runs[phase, name, "dolly"]).get(metric, np.nan)
                           if (phase, name, "dolly") in runs else np.nan for name in names]
                axis.barh(np.arange(len(names)) + offset, heights, height=.36, label=phase, color=color)
                for y, value in enumerate(heights):
                    if np.isfinite(value):
                        axis.text(value + .4, y + offset, f"{value:.2f}", va="center", fontsize=8)
                    else:
                        axis.text(.8, y + offset, "pending", va="center", fontsize=8, color=color)
            axis.set(yticks=range(len(names)), yticklabels=labels, title=title, xlim=(0, 100))
            axis.set_ylim(len(names) - .5, -.5); axis.grid(axis="x", alpha=.2)
        figure.suptitle("Dolly 500 · original vs SFT · generation seed 10 · same precision within model")
        figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        figure.savefig(OUT / "dolly_before_after.png", dpi=180); plt.close(figure)
        lines += ["![원본과 SFT 생성 평가](dolly_before_after.png)", ""]
    lines += ["전체 지표: [scores.csv](scores.csv), 대응 차이와 입력 근거: [comparisons.json](comparisons.json). 최종 선정에는 후보의 외부 평가·보조 분류/보정 진단·응답 검토·실제 증류 경로 점검이 남아 있다.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Verified original/SFT report: {len(runs)} benchmark results", flush=True)


if __name__ == "__main__":
    main()
