"""Make comparison tables and figures from completed SFT evaluations."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sft_baselines"


def main():
    runs = {role: json.loads((OUT / f"{role}.json").read_text(encoding="utf-8")) for role in ("teacher", "student")}
    protocols = {role: json.loads((OUT / f"{role}_protocol.json").read_text(encoding="utf-8")) for role in runs}
    assert {k: v for k, v in protocols["teacher"].items() if k != "script_sha256"} == {k: v for k, v in protocols["student"].items() if k != "script_sha256"}
    for role, run in runs.items():
        fingerprint = hashlib.sha256(json.dumps([protocols[role], run["model_id"], run["revision"]], sort_keys=True).encode()).hexdigest()
        assert run["fingerprint"] == fingerprint
        assert all(seed["fingerprint"] == fingerprint for seed in run["dolly_seeds"])
    for run in runs.values():
        assert [item["seed"] for item in run["dolly_seeds"]] == [10, 20, 30, 40, 50]
        assert all(item["all_500"]["n"] == 500 for item in run["dolly_seeds"])
    for dataset in ("sst2", "boolq"):
        assert runs["teacher"]["classification"][dataset]["n"] == runs["student"]["classification"][dataset]["n"]
        assert runs["teacher"]["classification"][dataset]["excluded_over_native_context_ids"] == runs["student"]["classification"][dataset]["excluded_over_native_context_ids"]

    rows = []
    for role, run in runs.items():
        for dataset, scores in run["classification"].items():
            rows.append({"role": role, "dataset": dataset, **{key: scores[key] for key in
                         ("n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "positive_class_brier", "mean_raw_label_probability_mass", "majority_class_accuracy")}})
    with (OUT / "classification.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dolly = []
    for role, run in runs.items():
        for subset in ("all_500", "excluding_flagged_input_overlap"):
            record = {"role": role, "subset": subset, "n": run["dolly_seeds"][0][subset]["n"]}
            for key in ("rouge1", "rouge2", "rougeL", "normalized_exact_match", "word_overlap_f1", "generated_tokens", "hit_length_cap", "empty", "distinct_4_whitespace"):
                values = [seed[subset][key] for seed in run["dolly_seeds"]]
                record[key + "_mean"] = float(np.mean(values))
                record[key + "_sample_std"] = float(np.std(values, ddof=1))
            record.update({"reference_all500_" + key: value for key, value in run["reference_diagnostics"].items()})
            dolly.append(record)
    with (OUT / "dolly.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(dolly[0]))
        writer.writeheader()
        writer.writerows(dolly)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"teacher": "#2563EB", "student": "#EA580C"}
    labels = {"teacher": "GPT-2 XL SFT", "student": "GPT-2 base SFT"}
    x, width = np.arange(2), .32
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout="constrained")
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        for offset, (role, run) in zip((-.5, .5), runs.items()):
            values = [100 * run["classification"][name][metric] for name in ("sst2", "boolq")]
            bars = axis.bar(x + offset * width, values, width, color=colors[role], label=labels[role])
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
        axis.set(title=title, ylim=(0, 100), xticks=x, xticklabels=[f"SST-2 (n={runs['teacher']['classification']['sst2']['n']:,})", f"BoolQ (n={runs['teacher']['classification']['boolq']['n']:,})"])
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Score (%)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    figure.suptitle("Author-released Dolly SFT checkpoints · zero-shot label scoring", fontsize=12)
    figure.savefig(OUT / "classification.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = np.arange(3)
    for offset, role in zip((-.5, .5), runs):
        record = next(row for row in dolly if row["role"] == role and row["subset"] == "all_500")
        keys = ("rouge1", "rouge2", "rougeL")
        values = [100 * record[key + "_mean"] for key in keys]
        bars = axis.bar(x + offset * width, values, width, yerr=[100 * record[key + "_sample_std"] for key in keys], capsize=4, color=colors[role], label=labels[role])
        axis.bar_label(bars, fmt="%.2f", padding=5, fontsize=10)
    axis.set(title="Dolly evaluation · 500 prompts\nMean ± sample SD over 5 generation seeds", ylabel="ROUGE F1 × 100", ylim=(0, 100), xticks=x, xticklabels=["ROUGE-1", "ROUGE-2", "ROUGE-L"])
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.savefig(OUT / "dolly_rouge.png", dpi=180)
    plt.close(figure)
    print("Wrote classification.csv, dolly.csv and two figures from complete evaluations")


if __name__ == "__main__":
    main()
