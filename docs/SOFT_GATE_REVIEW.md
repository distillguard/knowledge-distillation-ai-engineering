# Soft gate 선행연구와 검증 범위

확인일: 2026-10-19, Asia/Seoul. **주제 검토 자료이며 본 실험 명세가 아니다.** 사용자는 기존 모델 평가를 계속하며 연구 주제를 검토하기로 했다. 양자화 영향은 연구 주제 후보에서 제외했다.

## 검토 질문

같은 Soft gate 계산에서 gate 경로의 역전파만 허용하거나 차단하여, KL 혼합 자체와 추가 그라디언트의 효과를 구분할 수 있는가?

## 선행연구 근거

| 연구 | 직접 확인한 내용 | 제안과의 중복 및 차이 |
|---|---|---|
| [CaRE-KD, 2026, §3.1·§9.1](https://arxiv.org/html/2609.36734v1#S3.SS1) | 혼합 손실의 그라디언트를 KL 학습 항과 gate 선택 항으로 분해한다. gate를 detach하면 선택 항이 사라진다고 명시한다. §4.5·Figure 3은 Hard/Soft와 설정 민감도를 비교한다. | 추가 항의 존재와 소멸은 이미 알려진 결과다. 공개 원문에서 동일 Soft gate의 detach 유무만 바꾼 생성 성능 대조는 확인하지 못했다. 공식 supplementary 구현은 확보·검증하지 못했다. |
| [ToDi, EMNLP 2025, 식 (10)·(11)·Appendix C](https://arxiv.org/html/2505.16297v2#A3) | sigmoid 기반 FKL/RKL 혼합 가중치에 stop-gradient를 사용하고, Jeffreys divergence와의 관계 및 미분의 차이를 분석한다. | Soft 혼합 가중치를 detach해 학습 신호를 바꾸는 발상과 분석은 강하게 겹친다. 다만 ToDi는 어휘 항목별 가중치이고 CaRE는 각 예측 위치의 entropy confidence로 정한 스칼라 gate다. 동일한 알고리즘은 아니다. |
| [SDAR, 2026, Appendix A.2, Proposition 5](https://arxiv.org/html/2605.15155v1#A1.SS2) | gate를 detach하지 않을 때 추가되는 자기 참조 그라디언트 항을 유도하고, 이를 차단하는 설계 이유를 설명한다. | gate 역전파의 역할과 안정성을 묻는 관점이 겹친다. agentic RL·학생이 생성한 토큰의 log-probability gap을 이용하므로 CaRE의 전체 어휘 KL 혼합에 대한 효과로 옮겨 해석할 수 없다. |
| [Unified Per-Token Gating, 2026, §5.2·§7](https://arxiv.org/html/2609.11768v1) | 혼합 비율 크기 및 평균 혼합 비율을 맞춘 대조를 수행한다. ToDi의 stop-gradient를 생략한 대리 비교임을 밝히며, 짧은 분류 출력과 단일 조합으로 적용 범위를 제한한다. | gate 구조를 통제해 효과를 분리하는 분석도 선행 사례가 있다. 이 논문의 다중 변경 비교를 detach 단독 효과의 증거로 사용할 수 없다. |

ToDi의 [고정 commit 구현](https://github.com/jungseongryong/ToDi/blob/1d1dd9efbdbeeae281130b184f29fde2aaf79cff/code/criterions/various_divergence.py)에서 `compute_ToDi`의 두 sigmoid 가중치에 `.detach()`가 적용된 것을 확인했다. 해당 외부 코드는 실행하지 않았다.

## 효과에 관한 근거와 한계

- **미분 경로의 차이:** CaRE의 기존 수식에서 직접 따른다. 작은 CPU 수식 예제에서도 같은 입력의 손실값은 같고 그라디언트는 달라지는 것을 확인했다. 실제 교사·학생 모델의 학습이나 성능 평가가 아니다.
- **관련 방법의 생성 점수:** ToDi 원문 Table 3의 GPT-2 1.5B → 120M DollyEval ROUGE-L은 FKL 24.06, RKL 24.22, ToDi 24.81이다. 이는 ToDi 전체 방법의 결과이며 같은 entropy gate에서 detach만 바꾼 효과를 뜻하지 않는다. 우리 LoRA 예비 결과와도 실행 조건이 다르다.
- **우리 조건의 실제 이득:** 동일한 출발점의 Soft detach/비detach 독립 학습·생성 비교는 수행하지 않았다. 기존 SFT 점수, 수식상 추가 항, 다른 연구의 방법 전체 점수를 근거로 개선 폭을 추정하지 않는다.
- **부정 결과의 해석:** 성능 차이를 확인하지 못하더라도 곧바로 동등성이나 효과 부재를 뜻하지 않는다. 학습 반복의 변동과 추정 불확실성을 함께 봐야 한다. 생성 seed 반복은 독립 학습 반복의 대체가 아니다.

## 현재 판단

**단독 신규 방법의 기여로 채택할 근거는 현재 부족하다.** 원문 재현에 붙이는 기전 분석 또는 보조 ablation으로 검토할 수 있으나, 동일 대조를 찾지 못했다는 사실을 최초성의 근거로 삼지 않는다. 검색은 위 원문·공개 구현과 관련 검색 결과를 확인한 범위이며 문헌 전체를 포괄하지 않는다.

새 주제·모델·온도·margin·학습 횟수는 확정하지 않았고 새 KD 실행은 시작하지 않았다. 기존 8모델 예비 SFT 결과와 진행 중인 원본/SFT 후속 평가는 유지한다. 본 KD는 교사 응답 품질과 별도의 실행 명세 검토 이후에 진행한다.
