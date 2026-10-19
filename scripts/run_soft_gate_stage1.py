"""Run original-model metrics, then the approved direct student SFT."""
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from filelock import FileLock
import torch
from transformers import AutoModelForCausalLM

import soft_gate_study as study
import evaluate_dolly_sft as classification
import supplementary_metrics as metrics


def quality_audit():
    records = {key: study.base.rows(study.LOCAL / "prepared" / f"{key}.jsonl")
               for key in ("train", "validation", "evaluation")}
    group = lambda row: " ".join(row["instruction"].lower().split())
    seen = {group(row) for key in ("train", "validation") for row in records[key]}
    excluded = [row["id"] for row in records["evaluation"] if group(row) in seen]
    result = {"definition": "lowercase and whitespace-normalized instruction overlap with training or validation",
              "excluded_evaluation_ids": excluded, "full_n": len(records["evaluation"]),
              "instruction_disjoint_n": len(records["evaluation"]) - len(excluded),
              "scope": "exact instruction groups only; not a semantic audit or a previously unseen test set"}
    study.base.write_json(study.OUT / "data_quality.json", result)
    return result


def classify(name, model, data, fingerprint, phase="original", tokenizer_name=None):
    output = {}
    for key, records in data.items():
        path = study.LOCAL / phase / name / f"{key}_predictions.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            assert saved["fingerprint"] == fingerprint
        else:
            saved = {"fingerprint": fingerprint, "source_rows": [], "labels": [],
                     "predictions": [], "positive_conditional_probability": []}
        count = len(saved["labels"])
        assert saved["source_rows"] == [r["source_row"] for r in records[:count]]
        assert saved["labels"] == [r["label"] for r in records[:count]]
        assert len(saved["predictions"]) == len(saved["positive_conditional_probability"]) == count
        with torch.inference_mode():
            for row in records[count:]:
                value = row["models"][tokenizer_name or name]
                scores = []
                if all(len(target) == 1 for target in value["targets"]):
                    ids = torch.tensor([value["prompt"]], device="cuda")
                    logp = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float().log_softmax(-1)
                    scores = [logp[target[0]] for target in value["targets"]]
                else:
                    for target in value["targets"]:
                        ids = torch.tensor([value["prompt"] + target[:-1]], device="cuda")
                        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -len(target):].float()
                        scores.append(logits.log_softmax(-1).gather(1, torch.tensor(target, device="cuda")[:, None]).sum())
                probability = torch.stack(scores).softmax(-1)
                saved["source_rows"].append(row["source_row"])
                saved["labels"].append(row["label"])
                saved["predictions"].append(int(probability.argmax()))
                saved["positive_conditional_probability"].append(float(probability[1]))
                if len(saved["labels"]) % 250 == 0:
                    study.base.write_json(path, saved)
                    study.status(f"{phase}_classification", model=name, dataset=key,
                                 completed_items=len(saved["labels"]), total_items=len(records))
        study.base.write_json(path, saved)
        labels, predicted, probabilities = (saved[k] for k in ("labels", "predictions", "positive_conditional_probability"))
        assert len(labels) == len(records)
        wrong = [max(p, 1-p) for y, pred, p in zip(labels, predicted, probabilities) if y != pred]
        output[key] = {**classification.classification_metrics(labels, predicted, probabilities),
                       "calibration": metrics.calibration(labels, predicted, probabilities),
                       "mean_confidence_on_errors": statistics.mean(wrong) if wrong else None}
    study.base.write_json(study.OUT / phase / name / "classification.json",
                          {"status": "completed", "fingerprint": fingerprint, "metrics": output})


def report(stage):
    spec = json.loads((study.OUT / "preparation.json").read_text(encoding="utf-8"))
    quality = json.loads((study.OUT / "data_quality.json").read_text(encoding="utf-8"))
    lines = ["# Soft Gate 본 연구 진행 기록", "", f"현재 단계: **{stage}**", "",
             "[새 연구계획서](RESEARCH_PLAN_SOFT_GATE_GRADIENT.md) · [실시간 실행 상태](../results/soft_gate_gradient_v1/worker_status.json)", "",
             "## 완료한 준비", "", f"- MiniLLM 공식 소스 commit: `{study.COMMIT}`. 외부 코드는 실행하지 않고 설정을 대조했다.",
             "- CPU logits에서 같은 순전파 값·선택 gradient 분해·teacher detach·마스크를 검증했다. 실제 모델 증류 검증은 아직 아니다.",
             f"- Dolly: 학습 {spec['data']['train']['retained_rows']:,}건, 검증 {spec['data']['validation']['retained_rows']:,}건, 평가 {quality['full_n']}건.",
             f"- 평가의 지시문 중복 제외 부분집합: {quality['instruction_disjoint_n']}건. 전체 점수와 분리해 보고한다.",
             "- 원문 concat tokenization·학습 입력 512토큰 자르기·평가 left padding 256·FP16·eager attention·batch 8·생성 seed 10/20/30/40/50을 별도 명세했다.", "",
             "## 원본 모델 평가", "", "| 모델 | 생성 응답 수 | 분류 평가 | 보완 채점 |", "|---|---:|---|---|"]
    for name in study.NAMES:
        total = sum(len(json.loads(p.read_text(encoding="utf-8"))["responses"])
                    for p in (study.LOCAL / "original" / name).glob("seed*_responses.json"))
        classification_done = (study.OUT / "original" / name / "classification.json").exists()
        scoring_done = (study.OUT / "original" / name / "summary.json").exists()
        lines.append(f"| {name} | {total}/2500 | {'완료' if classification_done else '대기'} | {'완료' if scoring_done else '대기'} |")
    lines += ["", "## 다음 단계", "", "교사는 MiniLLM 공식 Dolly-SFT XL을 재사용한다. 원본 기준 성능 확인 후 학생 base의 전체 파라미터 SFT 탐색(3 LR × 2 batch, 각 20 epoch)과 선택 조건의 별도 3 epoch 초기화를 순서대로 실행한다. [학생 진행 기록](SOFT_GATE_STUDENT_PROGRESS.md)에 실제 완료 epoch와 점수를 남긴다. gate의 prefix/gradient 추정과 학습량은 KD 전에 확정한다. 기존 LoRA 결과를 새 본 학습으로 표시하지 않는다.", "",
              "OPT는 확장 연구로 미뤘고 진행 중이던 평가는 재개 파일을 보존해 중단했다. 이 실행은 예약 자동화가 아니며 기존 작업과 같은 GPU 잠금을 사용한다.", ""]
    (study.ROOT / "docs/SOFT_GATE_PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    study.LOCAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(study.LOCAL / "stage1.lock"), timeout=0):
        try:
            spec = json.loads((study.OUT / "preparation.json").read_text(encoding="utf-8"))
            quality_audit()
            report("기존 GPU 작업 종료 후 원본 기준선 평가")
            study.baseline(wait=True)
            with FileLock(str(study.base.LOCAL / "worker.lock"), timeout=-1):
                data, manifest = classification.classification_data({n: study.base.tokenizer_for(n) for n in study.NAMES})
                study.base.write_json(study.OUT / "classification_protocol.json", {"data": manifest,
                    "classifier_sha256": study.base.sha(__file__), "data_helper_sha256": study.base.sha(classification.__file__),
                    "calibration_helper_sha256": study.base.sha(metrics.__file__), "ece_bins": 15,
                    "scope": "shared 872 SST-2 / 3269 BoolQ native-context rows; conditional label-sequence probabilities"})
                for name in study.NAMES:
                    if (study.OUT / "original" / name / "classification.json").exists():
                        continue
                    study.status("original_classification_loading", model=name)
                    model = AutoModelForCausalLM.from_pretrained(study.base.snapshot(name), local_files_only=True,
                        trust_remote_code=False, use_safetensors=True, dtype=torch.float16,
                        attn_implementation="eager", device_map={"": 0}).eval().requires_grad_(False)
                    try:
                        classify(name, model, data, spec["fingerprint"])
                    finally:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                    report("원본 분류 평가 진행")
            environment = os.environ.copy()
            environment.update(PYTHONPATH=str(study.ROOT / "tmp/supplementary_metrics_v1/deps"),
                               USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
            subprocess.run([sys.executable, "-u", "-B", "-X", "utf8",
                            str(study.ROOT / "scripts/score_soft_gate_baselines.py")],
                           cwd=study.ROOT, env=environment, check=True)
            study.status("stage1_completed", training_started=False)
            report("원본 Dolly 5-seed·분류·보완 채점 완료")
            study.status("student_pipeline_running")
            subprocess.run([sys.executable, "-u", "-B", "-X", "utf8",
                            str(study.ROOT / "scripts/train_soft_gate_student.py"), "worker"],
                           cwd=study.ROOT, check=True)
            study.status("sft_preparation_pipeline_completed", kd_started=False)
            report("원본 기준선·학생 SFT·3 epoch 초기화 및 평가 완료, KD 명세 검토 대기")
        except Exception as error:
            study.status("failed", error_type=type(error).__name__, error=str(error))
            report("실패: 실행 상태의 원인 확인 필요")
            raise


if __name__ == "__main__":
    main()
