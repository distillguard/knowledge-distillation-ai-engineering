# Soft Gate 본 연구 진행 기록

현재 단계: **GPT-2 원본 기준선 평가 진행 · OPT 확장 연구로 보류**

[새 연구계획서](RESEARCH_PLAN_SOFT_GATE_GRADIENT.md) · 실시간 실행 상태는 로컬 `results/soft_gate_gradient_v1/worker_status.json` 참조

## 완료한 준비

- MiniLLM 공식 소스 commit: `6f53ef41c951de0962bbf27fe3f5819eabc8ce0e`. 외부 코드는 실행하지 않고 설정을 대조했다.
- CPU logits에서 같은 순전파 값·선택 gradient 분해·teacher detach·마스크를 검증했다. 실제 모델 증류 검증은 아직 아니다.
- Dolly: 학습 11,435건, 검증 1,000건, 평가 500건.
- 평가의 지시문 중복 제외 부분집합: 487건. 전체 점수와 분리해 보고한다.
- 원문 concat tokenization·학습 입력 512토큰 자르기·평가 left padding 256·FP16·eager attention·batch 8·생성 seed 10/20/30/40/50을 별도 명세했다.

## 원본 모델 평가

| 모델 | 생성 응답 수 | 분류 평가 | 보완 채점 |
|---|---:|---|---|
| gpt2_base | 2440/2500 | 대기 | 대기 |
| gpt2_xl | 0/2500 | 대기 | 대기 |

## 다음 단계

교사는 MiniLLM 공식 Dolly-SFT XL을 재사용한다. 원본 기준 성능 확인 후 학생 base의 전체 파라미터 SFT 탐색(3 LR × 2 batch, 각 20 epoch)과 선택 조건의 별도 3 epoch 초기화를 순서대로 실행한다. [학생 진행 기록](SOFT_GATE_STUDENT_PROGRESS.md)에 실제 완료 epoch와 점수를 남긴다. gate의 prefix/gradient 추정과 학습량은 KD 전에 확정한다. 기존 LoRA 결과를 새 본 학습으로 표시하지 않는다.

OPT는 확장 연구로 미뤘고 진행 중이던 평가는 재개 파일을 보존해 중단했다. 이 실행은 예약 자동화가 아니며 기존 작업과 같은 GPU 잠금을 사용한다.
