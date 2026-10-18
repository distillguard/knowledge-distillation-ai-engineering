"""Make tables and figures only from completed, matching Dolly SFT evaluations."""

import csv
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dolly_sft import LR_CANDIDATES, MODELS as INITIAL_MODELS, OUT, ROOT, sha
from dolly_study import LABELS, model_protocol, result_dir, selected_record
from dolly_scope import MODELS, REQUIRED_BENCHMARKS
from evaluate_dolly_sft import SEEDS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def epoch_losses(history):
    losses = {}
    for entry in history:
        if "eval_loss" in entry:
            # Trainer re-evaluates the restored best checkpoint at the final epoch.
            losses.setdefault(entry["epoch"], entry["eval_loss"])
    return losses


def replace_progress(text, block):
    start, end = "<!-- SFT_PROGRESS:START -->", "<!-- SFT_PROGRESS:END -->"
    assert text.count(start) == text.count(end) == 1, "Progress markers must occur exactly once"
    first, rest = text.split(start)
    _, last = rest.split(end)
    return first + start + "\n\n" + block + "\n\n" + end + last


def progress():
    generation_path = ROOT / "results/generation_benchmarks_v1/completion.json"
    generation = read(generation_path) if generation_path.exists() else {"completed_models": [], "completed_benchmarks": {}}
    if generation_path.exists():
        assert generation["data_protocol_fingerprint"] == read(generation_path.parent / "protocol.json")["fingerprint"]
        assert generation["study_models"] == list(MODELS) and generation["required_benchmarks"] == list(REQUIRED_BENCHMARKS)
    lines = ["# Dolly 모델별 SFT 진행", "", f"4개 계열·{len(MODELS)}개 모델의 예비 비교. 완료는 두 학습률 후보 학습과 검증 기반 adapter 선택을 모두 마친 상태다. 수치는 직접 실행한 결과만 기록한다.", "",
             "Qwen 두 모델은 2026-10-16 범위 축소로 제외했다. 기존 결과·checkpoint는 보존하며 아래 완료 수와 현재 순위에는 포함하지 않는다. 원문 기반 SFT 탐색·학생 초기화·본 KD는 최종 한 조합 선정 이후의 별도 설계다.", "",
             "| 모델 | 완료 LR 후보 | 선택 LR | 검증 NLL | 가중치 | 예비 Dolly 생성 | 기존 Dolly·분류 보조 평가 |", "|---|---|---|---|---|---|---|"]
    selected, evaluated = [], []
    for name in MODELS:
        label = LABELS[name]
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
        lr, loss = "—", "—"
        if chosen_path.exists():
            chosen = selected_record(name)
            assert len(completed) == 2 and chosen["protocol_fingerprint"] == spec["fingerprint"]
            assert math.isfinite(chosen["development_nll"])
            assert math.isclose(chosen["development_nll"], min(r["selected_development"]["eval_loss"] for r in completed))
            selected.append(name)
            lr, loss = f'{chosen["learning_rate"]:.0e}', f'{chosen["development_nll"]:.4f}'
        evaluation_path = OUT / f"{name}_evaluation.json"
        if evaluation_path.exists():
            evaluated.append(name)
        primary = "dolly" in generation["completed_benchmarks"].get(name, [])
        lines.append(f'| {label} | {len(completed)}/2 | {lr} | {loss} | {precision} | {"완료" if primary else "대기"} | {"완료" if evaluation_path.exists() else "후속"} |')
    lines += ["", "NLL은 각 모델 내부의 checkpoint·학습률 선택에 사용한다. 다른 tokenizer의 NLL로 성능 순위를 매기지 않는다.", "",
              "각 모델 SFT 후 예비 Dolly 500문항을 먼저 평가한다. 외부 세 benchmark와 SST-2·BoolQ는 후보 축소 후 진행하며 이미 완료한 결과는 보존한다. Dolly 생성은 기존 응답을 검증해 재사용하며 중복 생성하지 않는다. 생성 seed는 10 한 개이고 seed 간 표준편차는 미추정이다.", "",
              "[생성 벤치마크의 교사·학생 비교](../results/generation_benchmarks_v1/REPORT.md)", "",
              "동일 Dolly의 원본/SFT 비교·보완 채점 후 유망 조합을 추리고 외부 생성 평가·보조 진단·오류 분석·KD 실행 점검으로 최종 조합을 선정한다. 선정 전 추가 SFT는 시작하지 않는다. METEOR·BERTScore·Distinct-4와 후보 분류의 ECE는 추가 실행 승인 상태이며 아직 미구현·미채점이다. 단계별 완료 기준은 [생성 평가 계획](GENERATION_EVALUATION.md#조합-선정까지의-단계와-완료-기준)을 따른다."]
    (ROOT / "docs/SFT_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for filename in ("README.md", "docs/RESEARCH_PLAN.md", "docs/RESEARCH_REPORT.md", "docs/DOLLY_SFT_MODEL_SELECTION.md"):
        path = ROOT / filename
        progress_link = "docs/SFT_PROGRESS.md" if path.parent == ROOT else "SFT_PROGRESS.md"
        report_link = "results/dolly_sft_v1/REPORT.md" if path.parent == ROOT else "../results/dolly_sft_v1/REPORT.md"
        generation_link = "results/generation_benchmarks_v1/REPORT.md" if path.parent == ROOT else "../results/generation_benchmarks_v1/REPORT.md"
        block = (f"현재 예비 비교 범위는 **4개 계열·{len(MODELS)}개 모델**이며, SFT 선택 완료 **{len(selected)}/{len(MODELS)}**, Dolly 예비 생성 완료 **{len(generation['completed_models'])}/{len(MODELS)}**, 기존 Dolly·분류 보조 평가 완료 **{len(evaluated)}/{len(MODELS)}**이다. "
                 "외부 3종·새 보조 분류 평가는 후보 축소 후 수행한다. "
                 f"생성 평가는 공통 seed 10 한 개를 사용한다. 모델별 상태·선택 LR·검증 NLL·실제 정밀도는 [SFT 진행 기록]({progress_link}), "
                 f"교사·학생의 생성 성능 차이는 [생성 주 평가]({generation_link}), 분류 편향 등은 [기존 Dolly·분류 보조 평가]({report_link})를 따른다. 최종 조합은 아직 확정하지 않았다.")
        path.write_text(replace_progress(path.read_text(encoding="utf-8"), block), encoding="utf-8")
    return selected


def main():
    assert SEEDS == (10,), "This report is for single-seed model selection"
    assert replace_progress("a<!-- SFT_PROGRESS:START -->old<!-- SFT_PROGRESS:END -->b", "new") == "a<!-- SFT_PROGRESS:START -->\n\nnew\n\n<!-- SFT_PROGRESS:END -->b"
    runs = {name: read(OUT / f"{name}_evaluation.json") for name in MODELS if (OUT / f"{name}_evaluation.json").exists()}
    if not runs:
        selected = progress()
        (OUT / "REPORT.md").write_text(f"# Dolly 모델 SFT 평가 결과\n\nSFT 선택 완료 {len(selected)}/{len(MODELS)}. 아직 완료된 선정 평가가 없다. [모델별 학습 상태](../../docs/SFT_PROGRESS.md)를 따른다.\n", encoding="utf-8")
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
    figure.suptitle(f"Auxiliary classification after Dolly LoRA SFT · {len(runs)}/{len(MODELS)} candidates")
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
            losses = epoch_losses(run["trainer_log_history"])
            axis.plot([0, *losses], [run["before_sft_development"]["before_sft_loss"], *losses.values()],
                      marker="o", label=f"LR={lr:.0e}")
        axis.set(title=label, xlabel="Epoch", ylabel="Mean example response NLL"); axis.legend(frameon=False)
    for axis in axes.flat[len(runs):]:
        axis.set_visible(False)
    figure.suptitle("Development loss · compare checkpoints within each model")
    figure.savefig(OUT / "development_nll.png", dpi=180); plt.close(figure)
    lines = ["# 기존 Dolly·분류 보조 평가", "", "교사·학생 선정의 주 자료는 [네 생성 벤치마크 비교](../generation_benchmarks_v1/REPORT.md)다. 아래 SST-2·BoolQ는 편향·확률 품질의 보조 진단이며, 현재 Brier score는 MiniLLM의 ECE 재현과 구분한다.", "",
             f"4개 계열·{len(MODELS)}개 모델의 예비 비교 중 SFT 선택 완료 {len(selected)}/{len(MODELS)}, 기존 평가 완료 {len(runs)}/{len(MODELS)}. Qwen 두 모델의 기존 결과는 범위 밖으로 보존한다. 이번 직접 LoRA SFT 결과이며 이전 공개 SFT 점수와 구분한다.", "",
             "| 모델 | SST-2 Acc / macro-F1 (%) | BoolQ Acc / macro-F1 (%) | Dolly ROUGE-L (생성 seed 10) |",
             "|---|---|---|---|"]
    for label, run in zip(labels, runs.values()):
        scores = run["classification"]; rouge = run["dolly_generation"]["rougeL"]
        cells = [f'{100*scores[d]["accuracy"]:.2f} / {100*scores[d]["macro_f1"]:.2f}' for d in ("sst2", "boolq")]
        lines.append(f'| {label} | {cells[0]} | {cells[1]} | {rouge["mean"]:.2f} |')
    lines += ["", "모델 선정 단계는 생성 seed 10 한 개를 사용해 seed 간 변동을 추정하지 않는다. 최종 선정 모델의 본 실험에서는 생성 seed 10/20/30/40/50을 사용한다. 학습 seed 반복과 생성 seed 반복은 별개다. 모델 선정 자료이므로 독립적인 최종 test가 아니다.", "",
              "검증 손실 곡선은 각 에포크 종료 시 측정값을 표시한다. 학습 종료 후 최적 checkpoint를 복원해 재평가한 값은 해당 에포크의 추가 측정으로 표시하지 않는다.", "",
              "![분류 점수](classification.png)", "", "![Dolly ROUGE-L](dolly_rouge.png)", "", "![검증 손실](development_nll.png)"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote scores.csv, REPORT.md and three figures from matching completed evaluations", flush=True)


if __name__ == "__main__":
    main()
