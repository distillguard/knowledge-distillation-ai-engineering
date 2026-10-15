"""Count Dolly's source records and the published 500-record evaluation package."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sft_baselines"
NAMES = {
    "open_qa": "일반 지식 질문 (open_qa)", "general_qa": "자유 질문·지시 (general_qa)",
    "classification": "분류 (classification)", "closed_qa": "본문 기반 질문 (closed_qa)",
    "brainstorming": "아이디어 제안 (brainstorming)", "information_extraction": "정보 추출 (information_extraction)",
    "summarization": "요약 (summarization)", "creative_writing": "창작 (creative_writing)",
}


def main():
    specs = {
        "original": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "databricks-dolly-15k.jsonl"),
        "published_evaluation": ("MiniLLM/dolly", "d131636138be0ff3fd0901c12c1739844d337279", "valid.jsonl"),
    }
    result, data = {}, {}
    for name, (repo, revision, filename) in specs.items():
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                   cache_dir=ROOT / "tmp/hf_home/hub", token=False))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        category, context = ("category", "context") if name == "original" else ("topic", "input")
        assert all(isinstance(row[category], str) and isinstance(row[context], str) for row in rows)
        counts = Counter(row[category] for row in rows)
        assert set(counts) == set(NAMES) and sum(counts.values()) == len(rows)
        result[name] = {"repo": repo, "revision": revision, "file": filename,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
                        "fields": sorted(rows[0]), "categories": dict(counts),
                        "nonempty_context_rows": sum(bool(row[context].strip()) for row in rows)}
        data[name] = rows
    assert len(data["original"]) == 15011 and len(data["published_evaluation"]) == 500
    key = lambda instruction, context: (" ".join(instruction.lower().split()), " ".join(context.lower().split()))
    original_keys = {key(row["instruction"], row["context"]) for row in data["original"]}
    original_instructions = {key(row["instruction"], "")[0] for row in data["original"]}
    result["evaluation_input_matches_original"] = sum(key(row["instruction"], row["input"]) in original_keys for row in data["published_evaluation"])
    result["evaluation_instruction_matches_original"] = sum(key(row["instruction"], "")[0] in original_instructions for row in data["published_evaluation"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dolly_composition.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    font = FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    categories = sorted(NAMES, key=lambda item: result["original"]["categories"][item], reverse=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, layout="constrained")
    for axis, name, title, color in zip(axes, specs, ("Dolly 원본 · 15,011행", "이번 생성 평가 · 500문항"), ("#64748B", "#2563EB")):
        summary = result[name]
        counts = [summary["categories"][category] for category in categories]
        shares = [100 * count / summary["rows"] for count in counts]
        bars = axis.barh(range(len(categories)), shares, color=color, height=.65)
        axis.bar_label(bars, labels=[f"{count:,} ({share:.1f}%)" for count, share in zip(counts, shares)], padding=4, fontsize=10)
        axis.set_yticks(range(len(categories)), [NAMES[category] for category in categories], fontproperties=font)
        axis.set_xlim(0, 40)
        axis.set_xlabel("각 자료 안에서의 비중 (%)", fontproperties=font)
        axis.set_title(title, fontproperties=font, fontsize=13)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    figure.suptitle("원본과 평가용 자료의 과제별 구성", fontproperties=font, fontsize=15)
    figure.savefig(OUT / "dolly_composition.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
