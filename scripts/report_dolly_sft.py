"""Make tables and figures only from completed, matching Dolly SFT evaluations."""

import csv
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dolly_sft import LR_CANDIDATES, MODELS, OUT, ROOT, protocol, sha
from evaluate_dolly_sft import SEEDS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    spec = protocol()
    assert SEEDS == (10,), "This report is for single-seed model selection"
    runs = {name: read(OUT / f"{name}_evaluation.json") for name in MODELS}
    labels = ["GPT-2 base", "GPT-2 XL", "Qwen3.5-0.8B", "Qwen3.5-9B NF4"]
    colors = ["#93C5FD", "#2563EB", "#FDBA74", "#EA580C"]
    records = []
    for name, run in runs.items():
        evaluation = read(OUT / f"{name}_evaluation_protocol.json")
        chosen = evaluation["selected_adapter"]
        assert evaluation["generation"]["seeds"] == list(SEEDS) and evaluation["generation"]["phase"] == "model_selection"
        assert run["status"] == "completed" and evaluation["training_protocol_fingerprint"] == spec["fingerprint"]
        assert run["fingerprint"] == evaluation["fingerprint"] == (sha(ROOT / "scripts/evaluate_dolly_sft.py") +
                spec["fingerprint"] + sha(ROOT / chosen["adapter"] / "adapter_model.safetensors"))
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
    fields = list(dict.fromkeys(key for record in records for key in record))
    assert all(math.isfinite(v) for row in records for v in row.values() if isinstance(v, (int, float)))
    with (OUT / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    x, width = np.arange(2), .19
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        for index, (name, run) in enumerate(runs.items()):
            bars = axis.bar(x + (index - 1.5) * width, [100 * run["classification"][d][metric] for d in ("sst2", "boolq")],
                            width, label=labels[index], color=colors[index])
            axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        axis.set(title=title, ylim=(0, 100), xticks=x, xticklabels=["SST-2 (n=872)", "BoolQ (n=3,269)"])
        axis.grid(axis="y", alpha=.18); axis.set_axisbelow(True)
    axes[0].set_ylabel("Score (%)"); axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Own Dolly LoRA SFT · Qwen-9B only: NF4 · candidate comparison")
    figure.savefig(OUT / "classification.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    values = [run["dolly_generation"]["rougeL"] for run in runs.values()]
    bars = axis.bar(labels, [v["mean"] for v in values], color=colors)
    axis.bar_label(bars, fmt="%.2f", padding=6)
    axis.set(title="Dolly model selection · 500 prompts · generation seed 10", ylabel="ROUGE-L F1 × 100", ylim=(0, 100))
    axis.grid(axis="y", alpha=.18); axis.set_axisbelow(True)
    figure.savefig(OUT / "dolly_rouge.png", dpi=180); plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for axis, (name, label) in zip(axes.flat, zip(MODELS, labels)):
        for lr in LR_CANDIDATES:
            run = read(OUT / f"{name}_lr{lr:.0e}.json")
            assert run["status"] == "completed" and run["protocol_fingerprint"] == spec["fingerprint"]
            logs = [r for r in run["trainer_log_history"] if "eval_loss" in r]
            axis.plot([0] + [r["epoch"] for r in logs], [run["before_sft_development"]["before_sft_loss"]] +
                      [r["eval_loss"] for r in logs], marker="o", label=f"LR={lr:.0e}")
        axis.set(title=label, xlabel="Epoch", ylabel="Mean example response NLL"); axis.legend(frameon=False)
    figure.suptitle("Development loss · compare checkpoints within each model")
    figure.savefig(OUT / "development_nll.png", dpi=180); plt.close(figure)
    lines = ["# Dolly 네 모델 SFT 평가 결과", "", "이번 직접 LoRA SFT의 선정 평가 결과다. 9B만 NF4이며 이전 공개 SFT 점수와 구분한다.", "",
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
