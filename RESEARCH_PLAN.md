# 연구 계획과 방법

[문서 안내](README.md) · [방법](RESEARCH_PLAN.md) · [결과·해석](RESEARCH_REPORT.md) · [비교 요약](RESULTS_VISUALIZATION.md)

갱신일: 2026-10-24. 원본·SFT 여섯 기준 상태, 4B CE 재평가, 여섯 KD 단계의 학습·검증 자료 선정·전체 시험 평가를 완료했다. 이번 갱신은 기존 결과의 문서 정리이며 추가 실험은 수행하지 않았다.

## 1. 목적과 연구 질문

**ECQA에서 증류 경로와 KL 방향에 따른 정확도 및 확률 점수 비교**

ECQA의 사람 정답·설명으로 Qwen3.5 모델의 기준 성능을 확보하고, 4B→0.8B와 4B→2B→0.8B 경로에서 정방향 KL(FKL)과 역방향 KL(RKL)을 비교했다. [최종 국문 논문](docs/PAPER_KO.md)과 같은 연구 범위이며, 이 문서는 상세 재현 조건을 보충한다.

### 연구 질문

- RQ1. 같은 KL 방향에서 중간 2B 모델을 경유하면 최종 0.8B의 정답률·Macro-F1이 개선되는가?
- RQ2. 중간 경로의 효과는 FKL과 RKL 사이에서 달라지는가?
- RQ3. 정확도와 확률 점수(NLL·Brier)의 순위는 일치하는가? 교사·학생의 문항별 정오 중첩과 출력 상한 도달은 어떤 양상을 보이는가?

새 알고리즘의 개발보다 통제된 비교와 관찰 결과를 기여로 삼는다. 한 모델 계열·한 데이터셋의 결과로 파라미터 격차 완화의 일반 법칙을 입증하지 않는다.

## 2. 비교 설계

| 단계 | 산출물 | 범위 |
|---|---|---|
| 1 | 원본 4B·2B·0.8B 벤치마크 | 추가 학습 없이 같은 ECQA test 평가 |
| 2 | 각 모델의 정답·설명 SFT와 재평가 | 원본별 fresh LoRA, 모델별 LR 고정, 3 epoch 후 dev 선정 |
| 3 | 공통 4B 교사·초기화·KD 명세 고정 | 기준선을 확인한 뒤 네 비교에 공통 적용 |
| 4 | 직접·중간 경로의 FKL·RKL | 아래 네 학생과 두 중간 교사 |
| 5 | 결과 비교와 해석 | 대응 문항 분석, 시각화, 한계 기록 |

원본 3개와 SFT 후 3개는 총 **6개 기준 상태**다. SFT는 4B·2B의 LR을 `5e-5`, 0.8B의 LR을 `1e-4`로 사전에 고정한다. 조기 종료 없이 각 3 epoch를 모두 학습한 뒤, 저장한 세 체크포인트 중 하나를 dev로 선정한다. test로 LR·epoch를 고르지 않는다.

[직접·중간 경로 도식](RESULTS_VISUALIZATION.md#1-증류-경로)을 참고한다.

| 조건 | 경로 | 최종 학생 |
|---|---|---|
| D-FKL | 4B → 0.8B, FKL | 별도 0.8B |
| D-RKL | 4B → 0.8B, RKL | 별도 0.8B |
| M-FKL | 4B → 2B → 0.8B, FKL → FKL | 별도 0.8B |
| M-RKL | 4B → 2B → 0.8B, RKL → RKL | 별도 0.8B |

두 2B 중간 교사는 서로 다른 학습 결과다. KD는 중간 교사 2회와 최종 학생 4회로 구분한다. **공통 교사는 4B SFT 선정 epoch 2**, 두 중간모델은 **같은 2B SFT 선정 epoch 2**, 네 최종 학생은 **같은 0.8B SFT 선정 epoch 3** adapter의 독립 사본에서 시작한다. SFT LoRA를 이어 학습하되 KD 단계마다 optimizer·scheduler·seed를 새로 초기화하며, 분기 간 가중치·optimizer 상태를 공유하지 않는다. 원본/SFT 조합을 추가 요인으로 늘리지 않는다.

## 3. 데이터와 분할

CommonsenseQA의 질문·A~E 선택지·정답과 ECQA의 사람 작성 설명을 문항 ID로 결합한다. [공식 자료와 구조](https://github.com/dair-iitd/ECQA-Dataset), [ECQA 논문](https://aclanthology.org/2021.acl-long.238/).

| 필드 | 내용 | 평가 입력 제공 |
|---|---|---|
| `id` | 원본 문항 ID | 식별에만 사용 |
| `question`, `options` | 질문과 A~E 선택지 | 제공 |
| `gold` | 원본 정답 문자 | 제공하지 않음 |
| `explanation` | 첫 주석의 통합 설명 | 제공하지 않음 |
| `reference_explanations` | 같은 ID에 존재하는 모든 설명 변형 | 제공하지 않음 |
| `positives`, `negatives` | 정답 근거와 오답 반박 | 제공하지 않음 |
| `question_concept` | 질문의 중심 개념 | 제공하지 않음 |

저자 분할은 7,598/1,090/2,194건이지만 서로 같은 ID·질문이 겹친다. **test → dev → train 우선순위**로 같은 질문은 가장 높은 우선순위의 분할에만 둔다. 질문 비교는 Unicode NFKC·대소문자·공백 정규화로 한정하며 임의 유사도 임계값을 쓰지 않는다. 테스트는 원래 2,194개 ID와 순서를 보존한다.

고정 분할: **train 7,566 / dev 1,079 / test 2,194건**. test ID·순서 보존, ID·정규화 질문의 분할 간 교집합 0. 원문 저자 분할에서 dev 11건·train 32건을 낮은 우선순위 분할에서 제외했다. 전처리 명세(로컬: `artifacts/ecqa/data_manifest.json`).

서로 다른 설명을 가진 중복 ID 1개는 모든 설명을 보존한다. 현재 train에는 해당 ID가 없다. 123개 설명 ID는 원래 저자 분할에 포함되지 않아 추가 편입하지 않는다. 원문 자료·응답·가중치는 `local/`에만 보관한다. [데이터 라이선스](https://github.com/dair-iitd/ECQA-Dataset/blob/main/LICENSE.md).

## 4. 모델과 SFT 설정

Qwen3.5-4B·2B·0.8B는 같은 토크나이저를 사용하지만 레이어 수·표현 차원·헤드 구성이 다르다. 구성 실측표는 [연구보고서](RESEARCH_REPORT.md)에 둔다. 약 4B/2B/0.8B는 모델의 표기상 규모이며 실제 학습 파라미터 수로 혼동하지 않는다.

| 항목 | SFT 비교 기준 | 근거와 제한 |
|---|---|---|
| 기준 LR | 4B·2B: `5e-5` = **0.00005**, 0.8B: `1e-4` = **0.0001** | 사전 고정, LR 탐색 없음 |
| 학습량 | 각 모델 **3 epoch 전체 학습**, 매 epoch 저장·dev 평가 | **early stopping 없음**, 완료 후 체크포인트 선정 |
| 초기화 | 원본 모델에서 모델별 fresh LoRA | 각 모델의 원본 가중치에서 독립 초기화 |
| LoRA | r=8, α=32, dropout=0, all-linear | 기존 검증 코드의 기준 설정, ECQA 최적값은 아님 |
| 배치 | microbatch 1 × gradient accumulation 16 | 유효 배치 16, 마지막 불완전 배치 포함 |
| Optimizer | AdamW, β=(0.9,0.999), ε=1e-8 | weight decay=0, gradient clipping=1 |
| Scheduler | linear, 전체 학습 update의 warmup 5% | 최종 분할·3 epoch 기준으로 계산 |
| 정밀도 | BF16 base, FP32 학습 LoRA | 양자화 없이 동일 원칙, 자원 probe 먼저 |
| Seed | 10 | 현재 단일 seed, seed 분산을 추정하지 않음 |
| 손실 | 응답·EOS의 token CE | prompt·padding은 제외, 정답·설명 원문은 자르지 않음 |
| 선정 | dev ACC → Macro-F1 → 낮은 정답 위치 NLL → 이른 epoch | 각 모델의 epoch 1·2·3에 같은 순서 적용, test 미사용 |

`5e-5 = 5×10^-5`, `1e-4 = 1×10^-4`이며 뒤의 값은 앞의 값의 2배다. LR 고정은 모델별 **기준·최대 학습률**을 사전에 정한다는 뜻이다. 실제 step별 LR은 유지한 linear scheduler와 warmup에 따라 변한다.

고정 train 7,566건과 유효 배치 16에서 epoch당 473 updates, 모델당 1,419 updates, warmup 71 updates다. SFT는 총 3회·9 model-epochs·4,257 updates이며 epoch별 dev 평가는 총 9회다. 3 epoch 일정의 epoch 1은 독립적인 1 epoch 학습과 scheduler가 달라 같은 궤적이라고 표현하지 않는다. 같은 epoch라도 모델 구조·토큰 수·계산량은 같지 않다. **위 LR은 SFT용**이며 KD는 아래 별도 명세의 LR·학습량·초기화를 사용했다.

## 5. KD 손실과 학습 조건

같은 응답 위치에서 어휘 v에 대한 교사 확률을 p(v), 학생 확률을 q(v)라 하면 FKL은 `Σ p(v) log[p(v)/q(v)]`, RKL은 `Σ q(v) log[q(v)/p(v)]`다. Σ는 전체 어휘에 대한 합, log는 자연로그다. 교사는 동결하고 학생에만 gradient를 전달한다. T=1은 분포를 변형하지 않는 기준이며 최적 온도가 아니다. CE 혼합이나 온도 탐색은 수행하지 않았다.

응답·설명 구간의 **전체 어휘 분포**를 같은 고정 참조 prefix에서 비교했다. A~E 5개 분포만의 증류와 구분한다. 같은 tokenizer 파일 확인만으로 chat template·EOS·마스크까지 같다고 간주하지 않으며 실제 입력을 대조했다. 전체 logits를 데이터셋 단위로 저장하지 않고, 메모리·수치 안정성과 실제 load/update/save/restore를 사전에 검사했다.

이는 고정 문맥 token-level KL 비교이며, 학생 rollout과 정책경사 최적화를 포함한 MiniLLM의 완전 재현이 아니다. FKL/RKL 두 방향에서 prefix 출처를 바꾸지 않는다.

### KD 고정 명세

| 항목 | 고정 조건 |
|---|---|
| 학습 타깃·prefix | 원본 ECQA ANSWER·사람 설명·EOS의 모든 응답 위치; 질문·선택지와 앞선 정답/설명 token을 같은 teacher-forced prefix로 제공 |
| 손실 | FKL `KL(p교사∥q학생)` 또는 RKL `KL(q학생∥p교사)`; 전체 어휘 248,320개, prompt·padding 제외 |
| 온도·혼합 | T=1, 순수 KL; CE 혼합·온도 탐색 없음. RKL은 q의 가중치와 log q 모두 미분 |
| KD LR | **여섯 학습 단계 모두 `5e-6` = 0.000005**, SFT LR과 구분 |
| 학습량 | 단계당 3 epoch 전체·1,419 updates, early stopping 없음; epoch당 473 updates |
| 선정 | 각 단계 dev 1,079건에서 ACC → Macro-F1 → 낮은 A~E 조건부 정답 NLL → 이른 epoch; test 선정 없음 |
| LoRA | 선정된 SFT adapter의 r=8·α=32·dropout=0·all-linear 유지, 각 실험의 사본만 업데이트 |
| 배치·optimizer | microbatch1×누적16; AdamW β .9/.999·ε1e-8·weight decay0·clip1 |
| Scheduler·seed | 단계별 새 linear·warmup71 updates(전체의 5% 올림), seed10·같은 데이터 순서 |
| 정밀도 | BF16 base·FP32 trainable LoRA·FP32 KL, 양자화·CPU offload 없음; 교사는 eval·완전 동결 |
| 메모리 계산 | 응답 logits만 계산; FP32 KL 임시 확률은 16 response-token씩 재계산. 어휘·응답을 자르지 않으며 dense KL과 값·gradient 일치 검사 통과 |
| 평가 | 선정된 2B 교사 2개·0.8B 학생 4개만 같은 test 2,194건; greedy·non-thinking·batch1·ANSWER→EXPLANATION·새 출력 최대256 |

KD LR은 [MiniLLM의 Phase 2 설정](https://arxiv.org/html/2306.08543v4#A2.SS1)을 출발점으로 사전 고정했다. 해당 논문의 rollout·정책 최적화와 현재 고정 참조 prefix 방식은 다르며, 이 LR의 최적성을 주장하지 않는다. 기존 정답·사람 설명은 SFT와 prefix로 사용하고, KD의 직접 손실은 교사 분포와 학생 분포 사이 KL이다. 평가 입력에는 참조 정답·설명을 제공하지 않는다.

실행 순서는 `ta_fkl → ta_rkl → direct_fkl → direct_rkl → via_fkl → via_rkl`이다. 단계마다 3 epoch·dev 선정·test를 완료한 후 다음 단계로 이어간다. 중간 경로의 두 번째 교사는 해당 방향으로 KD 후 dev 선정한 2B이며, 원래 2B SFT를 그대로 사용하는 것이 아니다. 최종 학생의 추가 학습 예산은 네 조건에서 같고, 중간 경로의 추가 2B 학습·평가 비용은 별도 집계한다. 따라서 경로 효과와 전체 계산량 효과를 완전히 분리했다고 주장하지 않는다.

## 6. 모델 선정과 평가

원본·SFT·KD 모두 질문·선택지만 입력하고, 비사고 모드·탐욕 선택·배치 1로 `ANSWER: X` 다음에 `EXPLANATION: ...`을 생성했다. 형식 구분자를 강제하고 답은 A–E로 제한했다. 자유 생성과 같은 평가로 간주하지 않는다.

dev 극값 4문항×세 모델 검사를 거쳐 공통 생성 상한을 256토큰으로 고정했다. 이 값은 답·설명·형식 표지를 합친 새 출력 길이이며 학습 시 원문을 자르는 한도가 아니다. test 상한 도달 문항도 유효한 첫 답을 유지했고 재호출·제외·한도 확대를 하지 않았다.

| 지표 | 정의와 해석 |
|---|---|
| ACC / Macro-F1 | 첫 답의 정답률(%) / A–E 각 선택지 F1의 동일 가중 평균(0–1) |
| NLL | 첫 ANSWER 위치에서 A–E에 재정규화한 정답 확률의 음의 자연로그를 문항 평균 |
| Brier | 같은 확률 벡터와 정답 원-핫 벡터의 제곱차 합을 문항 평균; 선택지 수로 나누지 않음 |
| 설명 상한 | EOS 없이 256토큰에 도달한 건수; 설명 사실성이나 실제 내용 손실의 직접 측정이 아님 |
| 경로 효과 | ΔF=ACC(중간 FKL)−ACC(직접 FKL), ΔR=ACC(중간 RKL)−ACC(직접 RKL); 단위 %p |
| 차이의 차이 | ΔR−ΔF; 같은 test 문항에서 계산한 서술적 비교이며 유의성 검정이 아님 |

모든 단계는 전체 학습을 마친 뒤 dev ACC→Macro-F1→낮은 정답 NLL→이른 저장 회차 순으로 선정했다. test는 모델·학습률·상한 선정에 사용하지 않았다. 설명의 사실성·충실성은 평가하지 않았다. 교사·학생의 동일 오답은 예측 중첩으로 해석하며 인과적인 전달로 단정하지 않는다.

## 7. 구현 검증과 기록 보존

모델별 chat template 파일은 달랐지만 train 전체 input IDs·마스크·labels와 세 분할의 평가 prompt IDs·종료 토큰이 일치함을 확인했다. 16개 응답 토큰 단위 재계산은 전체 어휘 KL과 loss·gradient 일치 검사를 거쳤으며 어휘·설명 절단이 아니다.

VRAM 여유 보호로 중단된 세션은 가중치·optimizer·scheduler·난수 상태를 보존한 checkpoint에서 복구했다. 학생 logits의 불필요한 FP32 중복 복사를 생략한 변경은 동일 BF16 autocast·FP32 KL 아래 CPU 회귀 검사와 GPU loss·gradient 비교로 확인했다. 진단 업데이트는 본 학습에 사용하지 않았다. 중단·재계산 시간은 결과 문서의 비용에 범위를 밝혀 반영했다.

원본 응답·체크포인트·소스와 프로토콜 해시·복구 로그·HWP·PDF는 로컬에 보존한다. 이번 Git 갱신은 MD만 포함한다. 완료 검증 기록과 원래 실험 결과를 다시 쓰지 않았으며, 임시 학습 점검 예약은 완료 후 삭제했다.

## 8. 선행연구와 주장 범위

- [TAKD](https://arxiv.org/abs/1902.03393): 중간 교사와 모델 용량 격차의 선행 근거. 영상 분류 결과를 ECQA 성능 보장으로 옮기지 않는다.
- [MiniLLM](https://arxiv.org/abs/2306.08543v4): 생성형 LLM에서 RKL을 다루는 선행연구. 본 고정-prefix token loss와 구현 범위가 다르다.
- [GKD](https://arxiv.org/abs/2306.13649): 손실 방향과 학습 prefix 출처를 함께 통제할 근거.
- [Beyond One-Step Distillation, EACL 2026 SRW](https://aclanthology.org/2026.eacl-srw.13/): 중간 모델을 경유하는 다단계 LLM 증류 선행연구. 중간 모델·RKL의 결합 자체를 새 기여라고 주장하지 않는다.

이번 연구의 기여는 ECQA의 같은 데이터·모델 계열·최종 학생 예산에서 경로×KL 방향의 서술적 차이, 정오 중첩, 정답률과 확률 점수의 불일치를 분석한 것이다. 보편적인 성능 향상·인과적 오류 전달·학회 채택을 입증한 연구로 표현하지 않는다.
