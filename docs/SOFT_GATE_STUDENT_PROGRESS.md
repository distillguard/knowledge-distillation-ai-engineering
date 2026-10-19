# GPT-2 학생 SFT 진행 기록

현재 단계: **prepared_waiting_for_original_baselines**

교사는 MiniLLM 공식 Dolly-SFT XL을 재사용하며, 학생은 원본 GPT-2 base에서 전체 파라미터 SFT한다.

| 조건 | 완료 epoch | 최적 검증 ROUGE-L F1 | 상태 |
|---|---:|---:|---|

| SFT 후 평가 대상 | 생성 채점 | 분류 평가 |
|---|---|---|
| teacher_sft | 대기 | 대기 |
| student_sft | 대기 | 대기 |
| student_init | 대기 | 대기 |

[연구계획서](RESEARCH_PLAN_SOFT_GATE_GRADIENT.md) · [원본 기준선](SOFT_GATE_PROGRESS.md)

기존 LoRA 예비 결과와 별도 실행이다. 생성 5 seed는 학습 반복이 아니다. 미완료 조건은 최종 성능으로 해석하지 않는다.
