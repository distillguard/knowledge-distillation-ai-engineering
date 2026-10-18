"""Validate cached candidate label probabilities and report classification/ECE diagnostics."""
import csv
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluate_original_models as original
from dolly_study import LABELS, model_protocol, selected_record
from dolly_sft import ROOT, LOCAL, OUT as SFT_OUT, sha, write_json
import supplementary_metrics as metrics

OUT = ROOT / "results/candidate_diagnostics_v1"
legacy = original.evaluation.legacy


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def validate_predictions(prediction, records, expected):
    assert prediction["source_rows"] == [r["source_row"] for r in records]
    assert prediction["labels"] == [r["label"] for r in records]
    probabilities, predicted = prediction["positive_conditional_probability"], prediction["predictions"]
    assert len(probabilities) == len(predicted) == len(records)
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    assert all(p in (0, 1) for p in predicted)
    computed = legacy.classification_metrics(prediction["labels"], predicted, probabilities)
    assert computed == expected, "Cached classification metrics disagree with probabilities"
    return {**computed, "positive_prediction_rate": statistics.mean(predicted),
            "calibration": metrics.calibration(prediction["labels"], predicted, probabilities)}


def load_run(name, phase, data, manifest):
    parent = ROOT / "results/original_model_evaluation_v1" / name
    path = parent / "classification.json" if phase == "original" else SFT_OUT / f"{name}_evaluation.json"
    if not path.exists():
        return None
    result = original.study.read(path)
    assert result["status"] == "completed"
    training = model_protocol(name)
    if phase == "original":
        base = original.identity(name, original.evaluation.prepared_protocol())
        assert original.study.read(parent / "protocol.json") == base
        details = {"original_model_fingerprint": base["fingerprint"], "classification": manifest}
        expected_fp = fingerprint(details)
        assert original.study.read(parent / "classification_protocol.json") == {**details, "fingerprint": expected_fp}
        assert result["phase"] == "before_our_dolly_sft"
        files = {key: ROOT / result["predictions"][key]["path"] for key in data}
        for key, value in files.items():
            assert value.resolve() == (ROOT / "tmp/original_model_evaluation_v1" / name / "evaluation" / f"{name}_{key}_predictions.json").resolve()
            assert sha(value) == result["predictions"][key]["sha256"]
    else:
        assert phase == "sft"
        chosen = selected_record(name)
        assert chosen["protocol_fingerprint"] == training["fingerprint"]
        expected_fp = sha(legacy.__file__) + training["fingerprint"] + sha(ROOT / chosen["adapter"] / "adapter_model.safetensors")
        spec = original.study.read(SFT_OUT / f"{name}_evaluation_protocol.json")
        assert spec["fingerprint"] == expected_fp and spec["classification"] == manifest
        assert spec["training_protocol_fingerprint"] == training["fingerprint"] and spec["selected_adapter"] == chosen
        assert spec["evaluation_script_sha256"] == sha(legacy.__file__)
        files = {key: LOCAL / "evaluation" / f"{name}_{key}_predictions.json" for key in data}
    assert result["fingerprint"] == expected_fp and set(result["classification"]) == set(data)
    computed = {key: validate_predictions(original.study.read(files[key]), records, result["classification"][key]) for key, records in data.items()}
    spec = {"model": name, "phase": phase, "source_evaluation_fingerprint": expected_fp,
            "training_protocol_fingerprint": training["fingerprint"], "classification": manifest,
            "prediction_sha256": {key: sha(value) for key, value in files.items()},
            "classifier_script_sha256": sha(legacy.__file__), "calibration_helper_sha256": sha(metrics.__file__),
            "ece": {"bins": 15, "bounds": "equal-width (lower,upper]", "confidence": "max(p_positive,1-p_positive)",
                    "scope": "two conditional label-sequence probabilities; not full-vocabulary entropy confidence",
                    "temperature_fitting": False, "source": "https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py"}}
    spec["fingerprint"] = fingerprint(spec)
    folder = OUT / phase / name
    if (folder / "protocol.json").exists():
        assert original.study.read(folder / "protocol.json") == spec, "Preserve the existing diagnostic identity"
    if (folder / "result.json").exists():
        previous = original.study.read(folder / "result.json")
        assert previous["fingerprint"] == spec["fingerprint"] and previous["classification"] == computed
    else:
        write_json(folder / "protocol.json", spec)
        write_json(folder / "result.json", {"status": "completed", "fingerprint": spec["fingerprint"], "model": name,
                                          "phase": phase, "classification": computed, "producer_script_sha256": sha(__file__)})
    return {"fingerprint": spec["fingerprint"], "classification": computed}


def main():
    shortlist = original.study.read(ROOT / "results/model_selection_v1/shortlist.json")
    chosen = {value[key] for value in shortlist["shortlisted_pairs"] for key in ("teacher", "student")}
    names = [name for name in original.MODELS if name in chosen]
    # Only source labels/IDs are needed; the tokenizer/context audit was frozen before inference.
    data, manifest = legacy.classification_data({})
    runs = {(phase, name): value for name in names for phase in ("original", "sft")
            if (value := load_run(name, phase, data, manifest)) is not None}
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["model", "phase", "dataset", "n", "accuracy", "macro_f1", "balanced_accuracy", "weighted_f1", "mcc",
              "majority_class_accuracy", "positive_prediction_rate", "class0_recall", "class1_recall", "brier", "ece"]
    table = []
    for (phase, name), run in runs.items():
        for key, value in run["classification"].items():
            row = {"model": name, "phase": phase, "dataset": key, **{field: value[field] for field in fields[3:11]}}
            row.update(positive_prediction_rate=value["positive_prediction_rate"], class0_recall=value["class_recalls"][0],
                       class1_recall=value["class_recalls"][1], brier=value["positive_class_conditional_label_brier"], ece=value["calibration"]["ece"])
            table.append(row)
    with (OUT / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(table)
    lines = ["# 후보의 분류 편향과 calibration 진단", "",
             f"원본 {sum(p == 'original' for p, _ in runs)}/{len(names)}·SFT {sum(p == 'sft' for p, _ in runs)}/{len(names)}모델 완료. SST-2 872·BoolQ 3,269건의 고정 source row·label·파일 hash를 확인하고 기존 확률에서 지표를 재계산했다. 미완료 항목은 채우지 않는다.", "",
             "두 label 후보의 조건부 sequence 확률로 판정한다. 자유 생성의 의미상 정답률이나 전체 어휘의 entropy confidence와 다르다. ECE는 Guo 공식 기본 15개 등폭 (lower,upper] bin이며 temperature를 학습하지 않는다. 본 생성 평가의 보조 진단이며 최종 조합을 accuracy 하나로 고르지 않는다.", "",
             "| 모델 | 단계 | 자료 | Acc % | Macro-F1 % | Balanced Acc % | 다수 클래스 % | 양성 예측 % | ECE % ↓ | Brier ↓ |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        cells = [f'{100*row[key]:.2f}' for key in ("accuracy", "macro_f1", "balanced_accuracy", "majority_class_accuracy", "positive_prediction_rate", "ece")]
        lines.append(f'| {LABELS[row["model"]]} | {row["phase"]} | {row["dataset"]} | {" | ".join(cells)} | {row["brier"]:.4f} |')
    lines += ["", "SST-2의 양성은 positive, BoolQ는 Yes다. weighted-F1·MCC·클래스별 recall은 [scores.csv](scores.csv), 혼동행렬과 15개 bin의 n·평균 confidence·정답률·ECE 기여도는 모델별 result.json에 있다. 빈 bin의 평균은 null이다. 예측 원문·문항별 확률은 Git에서 제외한 tmp에 보관한다.", ""]
    for dataset in data:
        ordered = [(phase, name) for name in names for phase in ("original", "sft")]
        figure, axes = plt.subplots(1, 2, figsize=(12, max(4, .52 * len(ordered) + 1.5)), layout="constrained", gridspec_kw={"width_ratios": [3, 2]})
        for axis, columns, title, cmap in ((axes[0], ("accuracy", "macro_f1", "balanced_accuracy"), "Higher is better", "Blues"),
                                          (axes[1], ("ece", "brier"), "Lower is better", "Oranges")):
            lookup = {(row["phase"], row["model"]): row for row in table if row["dataset"] == dataset}
            grid = np.array([[lookup[pair][column] if pair in lookup else np.nan for column in columns] for pair in ordered])
            plotted = axis.imshow(np.ma.masked_invalid(grid), vmin=0, vmax=1, cmap=cmap, aspect="auto")
            axis.set(xticks=range(len(columns)), xticklabels=columns, yticks=range(len(ordered)),
                     yticklabels=[LABELS[name] + " / " + phase for phase, name in ordered] if axis is axes[0] else [], title=title)
            for y, x in np.ndindex(grid.shape):
                value = grid[y, x]
                text = "pending" if not np.isfinite(value) else (f"{value:.3f}" if columns[x] == "brier" else f"{100*value:.2f}%")
                axis.text(x, y, text, ha="center", va="center", color="white" if np.isfinite(value) and value > .6 else "black", fontsize=9)
            figure.colorbar(plotted, ax=axis, label="Fraction")
        figure.suptitle(f"{dataset} · original / SFT · conditional binary labels · ECE15, no temperature fit")
        figure.savefig(OUT / f"{dataset}.png", dpi=180); plt.close(figure)
        lines += [f'![{dataset} 진단]({dataset}.png)', ""]
    write_json(OUT / "completion.json", {"study_models": names, "completed": [{"model": name, "phase": phase, "fingerprint": run["fingerprint"]} for (phase, name), run in runs.items()],
                                          "classifier_source_manifest": manifest, "report_script_sha256": sha(__file__)})
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Verified candidate diagnostics: {len(runs)}/{2*len(names)} model phases", flush=True)


if __name__ == "__main__":
    main()
