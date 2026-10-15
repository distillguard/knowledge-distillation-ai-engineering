"""Make tables and figures only from completed, matching Dolly SFT evaluations."""

import csv
from datetime import datetime, timezone
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dolly_sft import LR_CANDIDATES, MODELS as INITIAL_MODELS, OUT, ROOT, sha
from dolly_study import LABELS, MODELS, model_protocol, result_dir, selected_record
from evaluate_dolly_sft import SEEDS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replace_progress(text, block):
    start, end = "<!-- SFT_PROGRESS:START -->", "<!-- SFT_PROGRESS:END -->"
    assert text.count(start) == text.count(end) == 1, "Progress markers must occur exactly once"
    first, rest = text.split(start)
    _, last = rest.split(end)
    return first + start + "\n\n" + block + "\n\n" + end + last


def progress():
    lines = ["# Dolly 모델별 SFT 진행", "", "5개 계열·10개 모델. 완료는 두 학습률 후보 학습과 검증 기반 adapter 선택을 모두 마친 상태다. 수치는 직접 실행한 결과만 기록한다.", "",
             "| 모델 | 완료 LR 후보 | 선택 LR | 검증 NLL | 가중치 | 선정 평가 | SFT 완료 확인 시각 (UTC) |", "|---|---|---|---|---|---|---|"]
    selected, evaluated = [], []
    for name, label in LABELS.items():
        folder = result_dir(name)
        completed = []
        for lr in LR_CANDIDATES:
            path = folder / f"{name}_lr{lr:.0e}.json"
            if path.exists() and (run := read(path))["status"] == "completed":
                completed.append(run)
        chosen_path = folder / f"{name}_selected.json"
        spec = model_protocol(name) if name in INITIAL_MODELS or (folder / "protocol.json").exists() else None
        assert all(run["protocol_fingerprint"] == spec["fingerprint"] for run in completed)
        precision = spec["precision"][name]["base_weights"] if spec else "GPU 점검 후 확정"
        precision = "NF4" if "NF4" in precision else ("BF16" if "BF16" in precision else precision)
        lr, loss, finished = "—", "—", "—"
        if chosen_path.exists():
            chosen = selected_record(name)
            assert len(completed) == 2 and chosen["protocol_fingerprint"] == spec["fingerprint"]
            assert math.isfinite(chosen["development_nll"])
            assert math.isclose(chosen["development_nll"], min(r["selected_development"]["eval_loss"] for r in completed))
            selected.append(name)
            lr, loss = f'{chosen["learning_rate"]:.0e}', f'{chosen["development_nll"]:.4f}'
            finished = datetime.fromtimestamp(chosen_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        evaluation_path = OUT / f"{name}_evaluation.json"
        if evaluation_path.exists():
            evaluated.append(name)
        lines.append(f'| {label} | {len(completed)}/2 | {lr} | {loss} | {precision} | {"완료" if evaluation_path.exists() else "대기"} | {finished} |')
    lines += ["", "NLL은 각 모델 내부의 checkpoint·학습률 선택에 사용한다. 다른 tokenizer의 NLL로 성능 순위를 매기지 않는다.", "",
              "각 모델 SFT 후 SST-2 872건·BoolQ 3,269건·Dolly 500문항(생성 seed 10)을 평가하고 다음 모델로 진행한다. 단일 생성 seed의 표준편차는 미추정이다."]
    (ROOT / "docs/SFT_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for filename in ("README.md", "docs/RESEARCH_PLAN.md", "docs/RESEARCH_REPORT.md", "docs/DOLLY_SFT_MODEL_SELECTION.md"):
        path = ROOT / filename
        progress_link = "docs/SFT_PROGRESS.md" if path.parent == ROOT else "SFT_PROGRESS.md"
        report_link = "results/dolly_sft_v1/REPORT.md" if path.parent == ROOT else "../results/dolly_sft_v1/REPORT.md"
        block = (f"현재 비교 범위는 **5개 계열·10개 모델**이며, SFT 선택 완료 **{len(selected)}/10**, 선정 평가 완료 **{len(evaluated)}/10**이다. "
                 f"Dolly 생성 평가는 공통 seed 10 한 개를 사용한다. 모델별 상태·선택 LR·검증 NLL·실제 정밀도는 [SFT 진행 기록]({progress_link}), "
                 f"평가 점수와 그림은 [직접 SFT 평가 결과]({report_link})를 따른다.")
        path.write_text(replace_progress(path.read_text(encoding="utf-8"), block), encoding="utf-8")
    return selected


def main():
    assert SEEDS == (10,), "This report is for single-seed model selection"
    assert replace_progress("a<!-- SFT_PROGRESS:START -->old<!-- SFT_PROGRESS:END -->b", "new") == "a<!-- SFT_PROGRESS:START -->\n\nnew\n\n<!-- SFT_PROGRESS:END -->b"
    runs = {name: read(OUT / f"{name}_evaluation.json") for name in MODELS if (OUT / f"{name}_evaluation.json").exists()}
    if not runs:
        selected = progress()
        (OUT / "REPORT.md").write_text(f"# Dolly 모델 SFT 평가 결과\n\nSFT 선택 완료 {len(selected)}/10. 아직 완료된 선정 평가가 없다. [모델별 학습 상태](../../docs/SFT_PROGRESS.md)를 따른다.\n", encoding="utf-8")
        print("Updated SFT progress; no completed evaluations yet", flush=True)
        return
    labels = []
    colors = plt.get_cmap("tab10")(np.arange(len(runs)))
    records = []
    for name, run in runs.items():
        spec = model_protocol(name)
        evaluation = read(OUT / f"{name}_evaluation_protocol.json")
        chosen = selected_record(name)
        assert evaluation["selected_adapter"] == chosen
        labels.append(LABELS[name] + (" NF4" if "NF4" in spec["precision"][name]["base_weights"] else ""))
        assert evaluation["generation"]["seeds"] == list(SEEDS) and evaluation["generation"]["phase"] == "model_selection"
        assert run["status"] == "completed" and evaluation["training_protocol_fingerprint"] == spec["fingerprint"]
        assert run["fingerprint"] == evaluation["fingerprint"] == (sha(ROOT / "scripts/evaluate_dolly_sft.py") +
                spec["fingerprint"] + sha(ROOT / chosen["adapter"] / "adapter_model.safetensors"))
        assert set(run["classification"]) == {"sst2", "boolq"}
        for dataset, scores in run["classification"].items():
            assert scores["n"] == (872 if dataset == "sst2" else 3269)
            records.append({"model": name, "dataset": dataset, "n": scores["n"], **{key: scores[key] for key in
                           ("accuracy", "macro_f1", "balanced_accuracy", "weighted_f1", "mcc",
                            "positive_class_conditional_label_brier", "majority_class_accuracy")}})
        for seed in SEEDS:
            generation = read(OUT / f"{name}_generation_seed{seed}.json")
            assert generation["fingerprint"] == run["fingerprint"] and generation["n"] == 500
        assert all(scores["sample_sd_across_generation_seeds"] is None for scores in run["dolly_generation"].values())
        records.append({"model": name, "dataset": "dolly_generation", "n": 500,
                        **{key + "_" + stat: value for key, scores in run["dolly_generation"].items()
                           for stat, value in scores.items() if value is not None}})
    selected = progress()
    fields = list(dict.fromkeys(key for record in records for key in record))
    assert all(math.isfinite(v) for row in records for v in row.values() if isinstance(v, (int, float)))
    with (OUT / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, layout="constrained")
    x, width = np.arange(2), .8 / len(runs)
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        for index, (name, run) in enumerate(runs.items()):
            bars = axis.bar(x + (index - (len(runs) - 1) / 2) * width, [100 * run["classification"][d][metric] for d in ("sst2", "boolq")],
                            width, label=labels[index], color=colors[index])
            axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=3, rotation=90 if len(runs) > 5 else 0)
        axis.set(title=title, ylim=(0, 110), xticks=x, xticklabels=["SST-2 (n=872)", "BoolQ (n=3,269)"])
        axis.grid(axis="y", alpha=.18); axis.set_axisbelow(True)
    axes[0].set_ylabel("Score (%)"); axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    figure.suptitle(f"Own Dolly LoRA SFT · {len(runs)}/10 evaluated candidates")
    figure.savefig(OUT / "classification.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 5.5), layout="constrained")
    values = [run["dolly_generation"]["rougeL"] for run in runs.values()]
    bars = axis.bar(labels, [v["mean"] for v in values], color=colors)
    axis.bar_label(bars, fmt="%.2f", padding=6)
    axis.tick_params(axis="x", labelrotation=30)
    axis.set(title="Dolly model selection · 500 prompts · generation seed 10", ylabel="ROUGE-L F1 × 100", ylim=(0, 100))
    axis.grid(axis="y", alpha=.18); axis.set_axisbelow(True)
    figure.savefig(OUT / "dolly_rouge.png", dpi=180); plt.close(figure)
    figure, axes = plt.subplots(math.ceil(len(runs) / 2), 2, figsize=(11, 3.4 * math.ceil(len(runs) / 2)), squeeze=False, layout="constrained")
    for axis, (name, label) in zip(axes.flat, zip(runs, labels)):
        for lr in LR_CANDIDATES:
            run = read(result_dir(name) / f"{name}_lr{lr:.0e}.json")
            assert run["status"] == "completed" and run["protocol_fingerprint"] == model_protocol(name)["fingerprint"]
            logs = [r for r in run["trainer_log_history"] if "eval_loss" in r]
            axis.plot([0] + [r["epoch"] for r in logs], [run["before_sft_development"]["before_sft_loss"]] +
                      [r["eval_loss"] for r in logs], marker="o", label=f"LR={lr:.0e}")
        axis.set(title=label, xlabel="Epoch", ylabel="Mean example response NLL"); axis.legend(frameon=False)
    for axis in axes.flat[len(runs):]:
        axis.set_visible(False)
    figure.suptitle("Development loss · compare checkpoints within each model")
    figure.savefig(OUT / "development_nll.png", dpi=180); plt.close(figure)
    lines = ["# Dolly 모델 SFT 평가 결과", "", f"5개 계열·10개 모델 중 SFT 선택 완료 {len(selected)}/10, 선정 평가 완료 {len(runs)}/10. 이번 직접 LoRA SFT 결과이며 이전 공개 SFT 점수와 구분한다.", "",
             "| 모델 | SST-2 Acc / macro-F1 (%) | BoolQ Acc / macro-F1 (%) | Dolly ROUGE-L (생성 seed 10) |",
             "|---|---|---|---|"]
    for label, run in zip(labels, runs.values()):
        scores = run["classification"]; rouge = run["dolly_generation"]["rougeL"]
        cells = [f'{100*scores[d]["accuracy"]:.2f} / {100*scores[d]["macro_f1"]:.2f}' for d in ("sst2", "boolq")]
        lines.append(f'| {label} | {cells[0]} | {cells[1]} | {rouge["mean"]:.2f} |')
    lines += ["", "모델 선정 단계는 생성 seed 10 한 개를 사용해 seed 간 변동을 추정하지 않는다. 최종 선정 모델의 본 실험에서는 생성 seed 10/20/30/40/50을 사용한다. 학습 seed 반복과 생성 seed 반복은 별개다. 모델 선정 자료이므로 독립적인 최종 test가 아니다.", "",
              "![분류 점수](classification.png)", "", "![Dolly ROUGE-L](dolly_rouge.png)", "", "![검증 손실](development_nll.png)"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote scores.csv, REPORT.md and three figures from matching completed evaluations", flush=True)


if __name__ == "__main__":
    main()
