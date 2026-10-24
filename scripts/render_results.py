"""Render the three study documents and figures from completed aggregate artifacts."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/ecqa"
DOCS = ("RESEARCH_PLAN.md", "RESEARCH_REPORT.md", "RESULTS_VISUALIZATION.md")
NAMES = ("qwen35_4b", "qwen35_2b", "qwen35_08b")
DISPLAY = {"qwen35_4b": "Qwen3.5-4B", "qwen35_2b": "Qwen3.5-2B", "qwen35_08b": "Qwen3.5-0.8B"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_section(text, marker, body):
    pattern = rf"(<!-- {marker}_START -->\n).*?(\n<!-- {marker}_END -->)"
    updated, n = re.subn(pattern, lambda m: m.group(1) + body + m.group(2), text, flags=re.S)
    if n > 1:
        raise ValueError("Duplicate document section: " + marker)
    return updated


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    token_data = read(OUT / "token_lengths.json")["groups"]["all_unique_annotations"]
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    for y, (key, label) in enumerate((("question_options", "Question + choices"),
            ("explanation", "Reference explanation"), ("answer_explanation", "Answer + explanation + EOS"))):
        s = token_data[key]
        ax.plot([s["min"], s["max"]], [y, y], color="#3267a8", linewidth=3)
        ax.scatter(s["mean"], y, color="#112d4e", label="Mean" if y == 0 else None, zorder=3)
        ax.scatter(s["p99"], y, marker="s", color="#d77328", label="P99" if y == 0 else None, zorder=3)
        ax.text(s["max"]+4, y, str(s["max"]), va="center", fontsize=9)
    ax.set_yticks(range(3), ["Question + choices", "Reference explanation", "Answer + explanation + EOS"])
    ax.axvline(256, color="#666666", linestyle="--", linewidth=1, label="Candidate output cap 256")
    ax.set(xlabel="Qwen3.5 tokens (n = 10,962 unique questions)", xlim=(0, 280), ylim=(-.6, 2.6))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.15)
    for suffix in ("png", "svg"):
        fig.savefig(figures / ("reference_token_lengths."+suffix), dpi=180)
    plt.close(fig)
    data_path = OUT / "data_manifest.json"
    if data_path.exists():
        data = read(data_path)
        counts = data["clean_counts"]
        data_body = (f"고정 분할: **train {counts['train']:,} / dev {counts['dev']:,} / test {counts['test']:,}건**. "
            "test ID·순서 보존, ID·정규화 질문의 분할 간 교집합 0. "
            "원문 저자 분할에서 dev 11건·train 32건을 낮은 우선순위 분할에서 제외했다. "
            "[전처리 명세](artifacts/ecqa/data_manifest.json).")
    else:
        data_body = "고정 분할 준비 중. 원문과 토큰 길이 집계만 완료했다."
    worker_path = ROOT / "local/ecqa_baseline_v1/worker_status.json"
    worker = read(worker_path) if worker_path.exists() else {}
    sft_worker_path = ROOT / "local/ecqa_sft_v1/worker_status.json"
    sft_worker = read(sft_worker_path) if sft_worker_path.exists() else {}
    table = ["| 모델 | 상태 | 판정 | ACC (%) | Macro-F1 | 상한 도달 | 빈 설명 |", "|---|---|---|---:|---:|---:|---:|"]
    completed = []
    for name in NAMES:
        for stage in ("original", "sft"):
            path = OUT / ("baselines" if stage == "original" else "sft") / (name+".json")
            result = read(path) if path.exists() else None
            if result and result.get("status") == "completed":
                assert result["overall"]["n"] == 2194
                s = result["overall"]
                label = result["model"].split("/")[-1]
                table.append(f"| {label} | {stage} | 완료 | {100*s['accuracy']:.2f} | {s['macro_f1']:.4f} | {s['length_cap_count']} | {s['empty_explanation_count']} |")
                completed.append((label, stage, result))
            else:
                active = worker if stage == "original" else sft_worker
                state = "진행 중" if active.get("model") == name and active.get("stage") not in ("completed", "failed") else "대기"
                if active.get("stage") == "failed":
                    state = "중단 기록 확인"
                table.append(f"| {DISPLAY[name]} | {stage} | {state} | — | — | — | — |")
    limit_path = OUT / "generation_limit.json"
    if limit_path.exists():
        limit = read(limit_path)
        limit_body = f"공통 새 출력 상한: **{limit['selected_max_new_tokens']}토큰**. dev 극값 probe {len(limit['dev_probe_ids'])}문항×세 모델 기준으로 고정했다. test는 선정에 사용하지 않았다. [선정 명세](artifacts/ecqa/generation_limit.json)."
    else:
        limit_body = "공통 새 출력 상한: **dev probe 준비·진행 중**. 참조 출력 최대 190토큰에 따라 256부터 검사한다."
    figure_body = "완료된 모델 점수가 아직 없어 성능 그래프를 만들지 않았다."
    if completed:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        labels = [label+"\n"+stage for label, stage, _ in completed]
        for ax, key, title, factor in zip(axes, ("accuracy", "macro_f1", "choice_nll"),
                ("Accuracy (%)", "Macro-F1", "Choice NLL"), (100, 1, 1)):
            values = [row["overall"][key]*factor for _, _, row in completed]
            bars = ax.bar(range(len(values)), values, color=["#3267a8" if stage == "original" else "#d77328" for _, stage, _ in completed])
            ax.set_xticks(range(len(values)), labels, rotation=35, ha="right", fontsize=8)
            ax.set_title(title)
            if key == "accuracy": ax.set_ylim(0, 100)
            if key == "macro_f1": ax.set_ylim(0, 1)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        for suffix in ("png", "svg"):
            fig.savefig(figures / ("baseline_comparison."+suffix), dpi=180)
        plt.close(fig)
        figure_body = "![완료된 원본·SFT 벤치마크](artifacts/ecqa/figures/baseline_comparison.png)\n\n완료된 전체 test 결과만 표시했다. 미완료 모델의 막대와 seed 오차막대는 없다."
        for label, stage, row in completed:
            fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
            matrix = row["overall"]["confusion_matrix"]
            im = ax.imshow(matrix, cmap="Blues")
            ax.set(xticks=range(6), xticklabels=[*"ABCDE", "invalid"], yticks=range(5), yticklabels=list("ABCDE"),
                   xlabel="Prediction", ylabel="Gold", title=label+" / "+stage)
            fig.colorbar(im, ax=ax)
            for y, values in enumerate(matrix):
                for x, value in enumerate(values):
                    ax.text(x, y, str(value), ha="center", va="center", fontsize=8,
                            color="white" if value > max(map(max, matrix))/2 else "black")
            fig.savefig(figures / (label+"_"+stage+"_confusion.png"), dpi=180)
            plt.close(fig)
    sft_body = "SFT 학습·검증 곡선은 아직 생성되지 않았다."
    candidate_paths = sorted((OUT / "sft/trials").glob("*_epoch*.json"))
    if candidate_paths:
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
        for column, name in enumerate(NAMES):
            key, label, color = ("lr1e-4", "LR 1e-4 (fixed)", "#d77328") if name == "qwen35_08b" else ("lr5e-5", "LR 5e-5 (fixed)", "#3267a8")
            history_path = OUT / "sft/trials" / f"{name}_{key}_history.json"
            if history_path.exists():
                history = [r for r in read(history_path)["log_history"] if "loss" in r]
                if history:
                    axes[0, column].plot([r["step"] for r in history], [r["loss"] for r in history], label=label, color=color)
            candidates = [read(p) for p in candidate_paths if p.name.startswith(name+"_"+key+"_")]
            if candidates:
                axes[1, column].plot([r["epoch"] for r in candidates], [100*r["dev"]["accuracy"] for r in candidates], marker="o", label=label, color=color)
                axes[2, column].plot([r["epoch"] for r in candidates], [r["dev"]["choice_nll"] for r in candidates], marker="o", label=label, color=color)
            axes[0, column].set_title(DISPLAY[name])
            for row, (xlabel, ylabel) in enumerate((("Update", "Response-token train CE"), ("Epoch", "Dev accuracy (%)"), ("Epoch", "Dev answer-choice NLL"))):
                axes[row, column].set(xlabel=xlabel, ylabel=ylabel)
                if axes[row, column].lines:
                    axes[row, column].legend(fontsize=8)
                else:
                    axes[row, column].text(.5, .5, "Not yet available", transform=axes[row, column].transAxes, ha="center")
        for suffix in ("png", "svg"):
            fig.savefig(figures / ("sft_training_curves."+suffix), dpi=180)
        plt.close(fig)
        sft_body = "![SFT 학습·검증 곡선](artifacts/ecqa/figures/sft_training_curves.png)\n\n4B·2B는 LR 5e-5, 0.8B는 1e-4로 고정한 실제 기록만 표시했다. 조기 종료 없이 3 epoch를 학습하며, train CE와 dev 정답 NLL은 서로 다른 지표다."
    ce_body = "4B의 epoch 1·2·3을 고정해 train 7,566건·dev 1,079건 전체를 같은 response CE로 재평가한다. 기존 여섯 기준 상태 작업이 GPU를 반납한 뒤 실행하며, 아직 완료되지 않은 CE 값은 표시하지 않는다."
    ce_path = OUT / "ce_audit/summary.json"
    if ce_path.exists():
        ce = read(ce_path)
        assert ce["status"] == "completed" and [r["epoch"] for r in ce["epochs"]] == [1, 2, 3]
        assert ce["fingerprint"] == read(OUT / "ce_audit/protocol.json")["fingerprint"]
        assert all(r["train"]["n"] == 7566 and r["dev"]["n"] == 1079 for r in ce["epochs"])
        fig, ax = plt.subplots(figsize=(7, 4.4), layout="constrained")
        for split, label, color in (("train", "Train (eval mode; n=7,566)", "#3267a8"),
                                    ("dev", "Dev (eval mode; n=1,079)", "#d77328")):
            values = [row[split]["response_ce"] for row in ce["epochs"]]
            ax.plot([1, 2, 3], values, marker="o", label=label, color=color)
            for ep, value in enumerate(values, 1):
                ax.annotate(f"{value:.4f}", (ep, value), xytext=(0, 8 if split == "dev" else -14),
                            textcoords="offset points", ha="center", color=color, fontsize=9)
        ax.set(xticks=[1, 2, 3], xlabel="Frozen SFT checkpoint epoch",
               ylabel="Mean per-example response CE (nats)", title="Qwen3.5-4B: matched train/dev CE")
        ax.margins(y=.12)
        ax.grid(alpha=.2)
        ax.legend()
        for suffix in ("png", "svg"):
            fig.savefig(figures / ("qwen35_4b_matched_ce."+suffix), dpi=180)
        plt.close(fig)
        ce_table = ["| Epoch | Train eval CE | Dev CE | Dev − Train |", "|---|---:|---:|---:|"]
        for row in ce["epochs"]:
            train, dev = row["train"]["response_ce"], row["dev"]["response_ce"]
            ce_table.append(f"| {row['epoch']} | {train:.6f} | {dev:.6f} | {dev-train:+.6f} |")
        ce_body = ("![4B의 동일 CE 학습·검증 비교](artifacts/ecqa/figures/qwen35_4b_matched_ce.png)\n\n"
            + "\n".join(ce_table) + "\n\n각 epoch 가중치를 고정하고 eval 모드에서 원본 정답·설명·EOS 전체를 teacher forcing으로 평가했다. "
            "prompt를 제외한 전체 어휘 token CE를 문항별 평균한 뒤 문항 간 평균했다. "
            "학습 도중 로그 평균·A~E 조건부 NLL과 구분하며, 이 재평가로 체크포인트를 다시 선정하지 않는다. "
            "[집계 JSON](artifacts/ecqa/ce_audit/summary.json).")
    for name in DOCS:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        for marker, body in (("DATA_STATUS", data_body), ("BASELINE_STATUS", "\n".join(table)),
                             ("GENERATION_STATUS", limit_body), ("BENCHMARK_FIGURES", figure_body),
                             ("SFT_FIGURES", sft_body), ("CE_AUDIT", ce_body)):
            text = update_section(text, marker, body)
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


if __name__ == "__main__":
    render()
