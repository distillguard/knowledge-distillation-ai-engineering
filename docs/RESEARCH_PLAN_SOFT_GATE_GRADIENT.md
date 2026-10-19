# GPT-2 지식증류의 Soft Gate 역전파 효과 연구계획서

작성일: 2026-10-19, Asia/Seoul
상태: 데이터·CPU 수식 사전 검증 완료. 원본 모델 기준선 평가부터 실행하며 본 학습 명세는 별도로 동결한다.

이 문서는 사용자가 확정한 **Dolly·GPT-2 XL → GPT-2 base·동일 soft gate의 역전파 허용/차단 비교**를 위한 새 계획서다. 기존 연구계획서·다중 교사 검토·8모델 예비 결과는 당시 기록으로 보존한다. 이 문서 작성으로 기존 실행이나 결과 파일의 설정이 바뀌지는 않는다.

실제 진행은 [새 연구 진행 기록](SOFT_GATE_PROGRESS.md)과 [학생 SFT 진행 기록](SOFT_GATE_STUDENT_PROGRESS.md), 사전 명세는 [데이터·원본 평가 protocol](../results/soft_gate_gradient_v1/preparation.json)과 [학생 학습 protocol](../results/soft_gate_gradient_v1/student/protocol.json)을 따른다. 모델 구조·토크나이저·활성함수·옵티마이저의 원리와 실제 값은 [인공지능공학 구성](GPT2_AI_ENGINEERING.md)에 설명한다. 준비 상태와 실제 학습 완료를 구분한다.

## 1. 연구 목적과 범위

**연구 목적은 confidence 기반 soft gate의 추가 gradient가 생성 능력과 confidence의 적절성에 미치는 영향을 분리하는 것이다.** MiniLLM의 모델·Dolly·SFT 준비·평가 절차를 기준으로 삼고, CaRE의 KL 혼합 gate를 분석 대상으로 사용한다.

| 항목 | 선정 내용 |
|---|---|
| 교사 | GPT-2 XL, 약 1.5B. Dolly SFT 후 동결 |
| 학생 | GPT-2 base, 약 124M. 동일한 SFT 초기화에서 조건별 독립 학습 |
| 학습 자료 | Dolly. 외부 생성 벤치마크와 SST-2·BoolQ는 평가용 |
| 핵심 조작 | 같은 soft gate에서 gate 경로의 gradient만 허용하거나 차단 |
| 주요 결과 | 생성 성능, 오류와 confidence의 관계, 추가 학습 신호의 역할 |
| 실행 제약 | 기존에 확인한 단일 16GB GPU. 실행 시 실제 여유 VRAM과 자원을 다시 확인 |

다중 교사·다른 아키텍처 간 증류·새 트랜스포머·양자화 효과·한국어 확장은 이번 본 질문에 포함하지 않는다. 메모리와 시간은 실행 제약이며 효율성 순위를 연구 목표로 두지 않는다. 두 모델 종류를 사용하되 비교 조건별 학생 학습과 체크포인트는 따로 필요하다.

## 2. 연구 질문과 판단 근거

| 연구 질문 | 확인할 내용 | 결론에 필요한 근거 |
|---|---|---|
| RQ1. 추가 gradient가 언제 생성 성능을 개선하는가? | 교사·학생의 초기 정답성, confidence 차이, 응답 특성에 따른 허용/차단 효과 | 동일 출발점의 생성 점수 차이와 문항별 변화. 조건부 분석 기준은 평가 결과를 보기 전에 정함 |
| RQ2. confidence 상승은 실제 오류 감소와 함께 나타나는가? | 올바른 답에 대한 확신과 오답에 대한 확신을 구분 | 생성 정답성 판정·분류 정답률과 confidence를 함께 관찰. entropy 감소만으로 개선을 판정하지 않음 |
| RQ3. 차단이 유용한 신호까지 제거하는가? | 추가 신호 제거에 따른 생성 성능과 calibration의 상충 | 성능·ECE/Brier·추가 gradient의 크기와 방향을 함께 비교 |

허용이 더 좋다거나 차단이 과신을 줄인다는 결론을 미리 정하지 않는다. 차이가 작으면 측정 불확실성과 학습 변동을 먼저 살핀다. 한 쌍의 결과는 우선 GPT-2와 해당 학습 조건에 한정한다.

### 2.1. 질문별 검증 절차

공통 비교는 같은 학생 초기화에서 시작한 Soft-Allow와 Soft-Detach다. 자료 순서·seed·optimizer·학습량·선택 규칙을 맞추고 gate 경로의 gradient만 조작한다. checkpoint 선택은 validation을 사용하며 평가 500문항으로 조건을 재선택하지 않는다.

| 질문 | 측정·비교 절차 | 지지하거나 반박하는 관측 |
|---|---|---|
| RQ1 | 동일 문항·생성 seed의 Allow−Detach ROUGE-L/METEOR/BERTScore 차이를 구한다. 학습 전의 교사·학생 응답 품질 차이, 공통 prefix의 confidence 차이, 고정된 자료 속성과 연결한다. 전체 500건과 중복 제외 487건을 분리한다. | 평균 개선뿐 아니라 어느 초기 조건에서 개선·악화가 나타나는지 확인. 교사가 정확하고 학생이 어려워하는 조건에서 추가 효과가 있는지는 관측할 가설이며 선결론이 아님 |
| RQ2 | SST-2·BoolQ의 accuracy/macro-F1과 ECE/Brier·오답 확신을 같이 비교한다. Dolly는 entropy 기반 confidence 변화와 익명 응답 오류 검토를 연결한다. | 확신 상승과 오류 감소가 함께 나타나는지, 오답 확신만 증가하는지 구분. entropy와 BERTScore를 문장 정답 확률·사실 정확도로 간주하지 않음 |
| RQ3 | 같은 checkpoint·동일 prefix·dropout RNG에서 실제 모델의 A/B gradient norm과 방향을 측정한다. attention·MLP·공유 embedding/LM head 등 중복 없는 파라미터 그룹으로도 분석한다. 동일 Adam 상태에서 한 update를 가한 검증 loss 진단과 최종 생성·calibration을 연결한다. | 차단이 과신을 완화하면서 생성 성능을 손상하는지, 불필요한 추가 신호를 제거하는지, 차이가 검출되지 않는지 비교. norm·cosine만으로 신호의 유용성을 판정하지 않음 |

여기서 A는 gate를 상수로 둔 두 KL의 혼합 gradient이고 B는 gate가 학생에 의존하면서 생기는 추가 gradient다. Allow는 A+B, Detach는 A다. 순간 gradient 진단의 prefix는 같게 고정하지만 최종 생성 평가는 각 학습된 모델이 자체적으로 생성하게 한다. 한 update 진단은 원본 checkpoint·Adam 상태를 복제해 수행하고 본 학습이나 평가 조건 선택에 섞지 않는다.

문항별 효과와 초기 confidence는 연속값 관계를 먼저 분석한다. 결과를 본 뒤 임의의 ‘높음/낮음’ 임계값을 골라 유리한 구간만 제시하지 않는다. 생성 오류 검토는 모델·조건을 가리고 사실 오류·지시 준수·핵심 누락의 판정 기준을 먼저 고정한다. 자동 유사도 지표와 사람의 오류 판정은 다른 근거로 보고한다.

문항·생성 seed를 대응시킨 비교와 불확실성을 보고하되, 생성 5 seed의 변동은 학습 자체의 반복 변동이 아니다. 독립 학습 반복 수와 통계 절차는 예비 분산·실행 예산을 확인해 본 KD 전에 정한다. 단일 학습 실행이면 결론도 그 실행의 범위로 제한한다. 위 표는 실제 모델 진단의 계획이며 현재 CPU logits 사전 검증의 통과만으로 완료 처리하지 않는다.

## 3. 선행연구와 기여 범위

| 근거 | 이번 연구에서의 역할 | 구분 사항 |
|---|---|---|
| [MiniLLM, §2·§3·부록 B](https://arxiv.org/html/2306.08543v4) | GPT-2 조합, SFT 준비, 생성 평가의 기준 | 전체 방법은 시퀀스 Reverse KL의 policy gradient 최적화이며 soft gate는 없음 |
| [CaRE-KD, §2.1·§3.1·§9.1](https://arxiv.org/html/2609.36734v1) | confidence gate와 학습/선택 gradient 분해의 직접 근거 | 추가 항의 존재와 detach 시 제거는 이미 알려진 내용 |
| [ToDi, 식 10·11·부록 C](https://arxiv.org/html/2505.16297v2) | 혼합 가중치에 stop-gradient를 적용하는 관련 사례 | 어휘 항목별 가중치로, CaRE의 예측 위치별 entropy gate와 다름 |

기여 후보는 **동일한 soft gate의 역전파 경로를 통제하고, 도움이 되는 조건·과신과의 관계·제거의 대가를 함께 설명하는 실증적 근거**다. detach 자체의 최초성이나 새로운 gradient 분해를 주장하지 않는다. 방법 제안으로 확장하려면 관측한 실패 원인과 이를 해결하는 추가 규칙의 필요성·효과가 별도로 입증되어야 한다. 조건부 차단이나 gradient 보정은 현재 필수 실험에 추가하지 않는다.

**2026-10-19 선행연구 재대조:** RQ1의 confidence에 따른 지도 효과·gate 민감도는 CaRE의 Figure 1·§4.5, RQ2의 정확도/calibration 관계는 MiniLLM의 SST-2·BoolQ Table 2와 CaRE §13.1, RQ3의 추가 항과 detach 효과는 CaRE §9.1·ToDi 부록 C에 선행 내용이 있다. 현재 계획은 재현·추가 분석의 출발점이며 독창성이 검증된 제출 주제로 확정하지 않는다. 동일한 연속 gate의 값·margin·prefix를 유지한 detach 단독 생성 대조는 확인한 원문에서 찾지 못했지만, 이것은 전 문헌에서의 부재나 최초성 증명이 아니다. SDAR의 [gate 미차단 추가 항 분석](https://arxiv.org/html/2605.15155v1#A1.SS2) 등 인접 연구도 중복 검토 대상이다. 모델 계열 변경·보조지표 추가만으로 새로운 기여가 성립했다고 주장하지 않는다. 기준선과 학생 준비는 계속하고 KD 전 기여 범위를 다시 판단한다.

## 4. 비교 대상의 수학적 정의

같은 prefix에서 교사 분포를 \(p\), 학생 분포를 \(q_\theta\), 공통 어휘를 \(V\)라 둔다. 교사는 동결하고 학생 분포는 미분 가능하게 유지한다.

\[
c(r)=1-\frac{H(r)}{\log |V|},\qquad
g_\theta=\sigma\bigl(c(p)-c(q_\theta)-m\bigr).
\]

\[
L_F=\mathrm{KL}(p\Vert q_\theta),\qquad
L_R=\mathrm{KL}(q_\theta\Vert p).
\]

| 조건 | 손실 | 차이 |
|---|---|---|
| Soft-Allow | \(g_\theta L_F+(1-g_\theta)L_R\) | gate를 포함한 학생 경로 미분 |
| Soft-Detach | \(\mathrm{sg}(g_\theta)L_F+[1-\mathrm{sg}(g_\theta)]L_R\) | gate 경로만 stop-gradient |

CaRE 식 7에 따라, 고정 prefix에서 허용 조건의 gradient는 다음과 같다.

\[
\nabla_\theta L
=\underbrace{g_\theta\nabla L_F+(1-g_\theta)\nabla L_R}_{A:\;\text{모방 학습 신호}}
+\underbrace{(L_F-L_R)\nabla g_\theta}_{B:\;\text{추가 선택 신호}}.
\]

Detach 조건에는 \(A\)가 남는다. 같은 파라미터·입력·확률적 연산 상태에서 두 손실의 순전파 값은 같고 gradient 차이는 \(B\)다. 독립 학습이 진행되면 모델 상태와 gate 값도 달라질 수 있다. detach는 hard gate 전환이나 학생 전체 학습 차단이 아니다.

margin은 [CaRE §4.5](https://arxiv.org/html/2609.36734v1)의 작은 margin 사례 **m=0.05**를 원문 근거의 공통 시작값으로 채택한다. 우리 조건의 최적값이라는 의미는 아니다. 추가 gate 기울기 계수나 KD 온도 탐색은 도입하지 않고, 두 분포의 기본 softmax를 사용한다. 생성 sampling temperature와 이 분포 정의를 구분한다. loss는 응답 예측 위치에만 적용하며 padding·prompt는 제외한다.

\(c(q)\)는 분포의 집중도이지 문장 정답 확률이 아니다. \(B\)와 \(A\)의 방향 일치도는 기전 진단이며, Adam의 실제 업데이트나 평가 성능 향상을 보장하는 정리가 아니다.

## 5. MiniLLM 기준의 학습 단계

### 5.1. SFT와 초기화

아래 학습량·탐색·선택 절차는 [MiniLLM 부록 B.1](https://arxiv.org/html/2306.08543v4)에 따른다.

| 단계 | 학습량 | 학습률 후보 | 배치 후보 | 선택 |
|---|---|---|---|---|
| 교사 XL SFT | 10 epoch | 5e-5·1e-5·5e-6 | 32·64 | 검증 ROUGE-L |
| 학생 base SFT 기준선 | 20 epoch | 5e-4·1e-4·5e-5 | 32·64 | 검증 ROUGE-L |
| 증류 전 학생 초기화 | 3 epoch | 학생 SFT에서 선택한 학습률 | 선택한 배치 | 검증 loss |

두 gate 조건은 동일한 학생 초기화 체크포인트에서 시작한다. SFT 기준선과 3 epoch 초기화를 혼동하지 않는다. 전체 파라미터 학습을 목표로 하며, 기존 LoRA 결과를 원문 조건의 초기화로 간주하지 않는다. 공식 공개 교사·초기화 가중치를 활용할 경우 revision·출처·데이터·선택 절차를 확인하고 직접 학습 결과와 구분한다.

2026-10-19 사용자가 **공식 교사 재사용·학생 직접 SFT**를 선택했다. 교사는 [MiniLLM/teacher-gpt2-1.5B](https://huggingface.co/MiniLLM/teacher-gpt2-1.5B/tree/4a8d07d52ad67b7b87fec9aba25b1521173eb658)를 동결해 재사용한다. 위 교사 행은 원문 재학습 절차의 참조이며 이 컴퓨터에서 10 epoch를 학습했다는 뜻이 아니다. 공개 모델 카드로 확인되지 않는 원 저자의 개별 탐색·체크포인트 이력은 추정하지 않는다. [교사 provenance](../results/soft_gate_gradient_v1/teacher.json)에 revision·파일 해시·전체 준비 record의 tokenizer 일치를 기록했다.

학생은 LR 3개 × effective batch 2개의 **6조건을 각각 원본 base에서 20 epoch** 학습하고 검증 ROUGE-L로 선택한다. 그 조건으로 원본에서 다시 시작한 3 epoch 학습의 최소 검증 loss checkpoint를 KD 초기화로 사용한다. 20 epoch 실행의 앞 3 epoch를 재사용하면 cosine 스케줄의 길이가 달라지므로 별도 실행한다. 단일 GPU의 FP32 파라미터·Adam 상태, FP16 autocast, gradient accumulation을 사용하며 원문의 DeepSpeed 다중 GPU 실행과 달라지는 loss 평균·데이터 순서·마지막 배치·loss scaling을 protocol에 명시했다. 실제 GPU load/update/save/reload 검증을 통과해야 학생 본 학습을 시작한다.

[공식 base SFT 실행기](https://github.com/microsoft/LMOps/blob/main/minillm/scripts/gpt2/sft/sft_base.sh)와 [XL 실행기](https://github.com/microsoft/LMOps/blob/main/minillm/scripts/gpt2/sft/sft_xlarge.sh)의 공통 기본값은 seed 10, cosine scheduler, warmup 0, weight decay 1e-2, clipping 1.0, 전체 길이 512, prompt 길이 256이다. 제공 스크립트의 단일 학습률은 예시 실행값이므로 원문의 탐색 결과와 동일하다고 단정하지 않는다. 정밀도·optimizer의 실제 값은 고정할 코드와 대조한다.

### 5.2. MiniLLM 증류 기준선과 gate 실험의 구분

| MiniLLM 기준선 항목 | 원문 기준 |
|---|---|
| 학습률·mini-batch | 5e-6·64 |
| 학습량 | 5,000 step |
| 응답 수집·내부 최적화 | 256개 수집·4 inner epoch |
| teacher-mixed sampling | alpha=0.2 |
| PPO clipping | 0.2 |
| 선택 | 검증 ROUGE-L |

위 값은 MiniLLM 재현 근거다. **gate 목적함수에 PPO의 4 inner epoch를 자동 적용하거나, 토큰별 RKL만 계산하고 MiniLLM 전체 재현이라고 표시하지 않는다.**

핵심 대조는 Soft-Allow와 Soft-Detach다. SFT 기준선과 MiniLLM 기준선은 이 대조의 위치를 해석하는 참조다. Hard gate·FKL 단독·ToDi 전체 재현은 최초 필수 범위에 자동 추가하지 않는다.

gate 학습의 prefix 출처와 gradient 추정 방식은 실행 전 확정 항목이다. 정답 prefix의 teacher forcing, 학생 생성 prefix의 토큰 손실, MiniLLM의 시퀀스 policy gradient는 서로 다른 방법이다. 원문 기준선을 먼저 확인한 뒤 두 gate 조건에 하나의 동일한 방식을 적용한다. 생성 경로를 포함한 목적함수라면 §4의 고정-prefix 미분과 별도로 해당 gradient를 확인한다. 이 결정 전에는 gate의 최종 epoch·step·학습률이 원문 그대로 확정되었다고 보고하지 않는다.

### 5.3. 코드와 원문의 차이

[공식 증류 실행기](https://github.com/microsoft/LMOps/blob/main/minillm/scripts/gpt2/minillm/train_base_xl.sh)는 per-device batch 16과 기본 GPU 수 16을 지정하며, 원문 mini-batch 64와 표기상 차이가 있다. 본 계획은 논문 값 64를 재현 기준으로 두고 실제 전역 배치·gradient accumulation·rollout 단위를 확인한다. GPU 수만 1로 바꿔 같은 실험이라고 부르지 않는다.

또한 실행기는 `--total-iters 5000`과 `--epochs 10`을 함께 사용하고 [trainer](https://github.com/microsoft/LMOps/blob/main/minillm/minillm/trainer.py)는 step 및 데이터 epoch 종료 조건을 검사한다. 따라서 5,000 step·데이터 10 epoch·PPO 내부 4 epoch를 구분하고, 실제 종료 이유와 optimizer 업데이트 수를 남긴다. 원문과 코드 중 어느 기준을 적용했는지도 기록한다.

## 6. Seed와 반복의 해석

| 용도 | 기준값 | 근거와 해석 |
|---|---|---|
| SFT 기본 학습 | 10 | 공식 SFT 실행기 |
| MiniLLM 기본 학습 | 10 | 공식 증류 실행기 |
| PPO 처리 | 42 | 공식 `--seed-ppo` |
| 보조 LM 데이터 | 7 | 공식 `--seed-lm`. no_pt에서는 해당 자료를 사용하지 않음 |
| 생성 평가 | 10·20·30·40·50 | [공식 평가 실행기](https://github.com/microsoft/LMOps/blob/main/minillm/scripts/gpt2/eval/run_eval.sh) |

생성 5 seed는 같은 학습 체크포인트에서의 sampling 반복이다. 학습 5회가 아니며 PPO seed 42도 별도의 독립 학습 반복을 뜻하지 않는다. gate 조건끼리는 동일한 학습 seed·프롬프트 순서·난수 설정을 짝지어 적용한다. 학생이 달라지면 같은 난수 설정에서도 생성 문장은 달라질 수 있다.

학습 안정성이나 보편적 우열을 주장하려면 독립 학습 반복이 별도로 필요하다. 추가 학습 seed의 수와 값은 현재 확정하지 않았으며, 실행 시간·예비 변동·주장 범위를 근거로 평가 전에 정한다. 한 번만 학습했다면 그 제한을 명시한다.

## 7. 데이터와 기존 결과의 역할

학습 자료는 Dolly로 유지한다. MiniLLM 본 설정은 보조 OpenWebText 손실을 포함하므로, 이 계획의 참조는 [공식 Dolly-only `no_pt` 변형](https://github.com/microsoft/LMOps/blob/main/minillm/scripts/gpt2/minillm/train_base_xl_no_pt.sh)이다. 학습 seed·epoch를 따르더라도 MiniLLM 본 설정 전체와 동일하다고 표현하지 않는다.

새 본 실험의 record ID·순서·tokenizer·프롬프트·길이 검사·마스크·EOS를 공식 데이터와 대조하고 train/validation/evaluation별 해시와 실제 건수를 기록한다. 현재 예비 분할 11,117/975/500은 추가 중복·길이 필터가 적용된 값이므로 그대로 원문 분할이라고 쓰지 않는다. 원문 재현 분할과 중복 검사에 따른 변경이 필요하면 그 차이를 공개한다. 검증 자료로 학습률·체크포인트를 선택하고 평가 자료로 재선택하지 않는다.

2026-10-19 사전 검증에서 공식 소스 commit `6f53ef41c951de0962bbf27fe3f5819eabc8ce0e`를 고정했다. 공식 concat tokenization과 입력 512토큰 자르기를 적용한 실제 건수는 학습 **11,435**, 검증 **1,000**, 평가 **500**이다. 각각 198·18·13건의 정답 응답이 NLL 계산 입력에서 잘리며, 생성 채점의 reference 원문은 유지한다. 공식 처리기의 비결정적 `imap_unordered` 대신 원본 record ID 순서를 고정한 차이를 명시했다.

평가 500건 중 정규화 지시문이 학습 또는 검증과 겹치는 13건을 확인했다. 원문 전체 점수와 이 13건을 제외한 **487건 부분집합 점수**를 함께 산출한다. 이는 정확한 지시문 중복에 대한 진단이며 의미 중복까지 제거한 새 test라는 뜻은 아니다. [검사 결과](../results/soft_gate_gradient_v1/data_quality.json)를 보존한다.

기존 예비 protocol(`results/dolly_sft_v1/protocol.json`)의 seed 0·LoRA·2개 LR·3 epoch와 원본/SFT 비교(`results/model_selection_v1/REPORT.md`)는 로컬에 보존한다. 이 이전 자료는 현재 원격 저장소의 공개 범위에서 제외한다. 기존 결과를 새 본 실험 결과로 옮겨 적지 않는다. 이미 모델 선택에서 관찰한 Dolly 500문항은 완전히 미관찰한 최종 test로 부르지 않는다. 외부 평가도 사전 관찰 여부를 기록한다.

## 8. 평가와 시각화

### 8.1. 생성 성능

원문 평가 대상은 DollyEval·Self-Instruct·Vicuna·S-NI·UnNI다. 기존 외부 3종 응답·행 수를 공식 버전과 먼저 대조하고, UnNI를 포함한 원문 전체 범위의 확보 여부를 구분한다. 데이터 이름이 같아도 다른 행을 비교하지 않는다.

최종 생성은 seed 10/20/30/40/50, sampling temperature 1과 공식 길이·EOS·디코딩 규칙을 기준으로 한다. 현재 예비 설정의 `max_new_tokens=256`·`no_repeat_ngram_size=6`을 자동 승계하지 않는다. 원문의 max length 512는 생성 토큰 512개 추가와 다르다.

ROUGE-L의 P/R/F1을 보고하고, 같은 저장 응답에 METEOR·BERTScore·Distinct-4를 적용한다. 생성 길이·빈 응답·상한 도달도 확인한다. 0~1과 0~100 표시를 명시한다. 보완 채점 때문에 SFT epoch를 늘리지는 않는다. ROUGE-L을 정확도나 사실 정답률로 표현하지 않는다.

### 8.2. Confidence와 오류

SST-2·BoolQ는 생성 평가를 대체하지 않는 보조 진단이다. label 확률로 accuracy·macro-F1·balanced accuracy·ECE·Brier score를 산출하고 클래스 비율과 다수 클래스 기준선을 함께 확인한다. template·label scoring·평가 행·ECE 구간은 조건 간 동일하게 고정한다. 정규화 entropy confidence를 정답 확률처럼 ECE에 직접 넣지 않는다.

Dolly의 정답성은 reference와 제공 context를 근거로 한 사전 rubric으로 판단한다. 자유 생성의 타당한 다른 답을 오답으로 처리하지 않으며, 조건명을 가린 오류 사례 검토를 포함한다. 검토 표본과 평가자 수는 본 평가 전 확정하고 일부 사례의 판정을 전체 정확도로 확대하지 않는다. 기존 METEOR·BERTScore만으로 사실 오류 감소를 확정하지 않는다.

### 8.3. 기전 진단과 그림

| 자료 | 설명할 질문 |
|---|---|
| 같은 업데이트 시점의 검증 생성 성능 곡선 | 추가 신호가 도움이 되거나 차이를 만드는 시점 |
| confidence–정답성 관계와 reliability diagram | 확신 상승이 오류 감소와 연결되는지 |
| 고정 진단 prefix에서 A/B의 norm·방향 일치도·gate 분포 | 제거한 신호의 크기와 모방 신호와의 관계 |
| 문항별 허용/차단 차이와 익명 응답 사례 | 평균 점수에 가려진 개선·악화와 다양성 변화 |

기전 비교는 같은 파라미터·prefix·dropout 상태에서 수행한다. 업데이트 수가 같은 중간 체크포인트 비교와 각 조건의 검증 최적 체크포인트 비교를 구분한다. 전자는 진행 과정, 후자는 선택 절차를 거친 결과를 설명한다.

생성 seed의 평균·표준편차는 sampling 변동으로 표시한다. 독립 학습의 표준편차와 혼합하지 않는다. 문항별 차이를 짝지어 분석하고, 여러 seed의 같은 문항을 모두 독립 표본으로 취급하지 않는다. 하위 집단 분석·구간·불확실성 산출 방식은 평가 전에 고정하고 사후 탐색은 별도로 표시한다.

## 9. 실행 순서와 필수 산출물

| 단계 | 작업 | 완료 기준 |
|---|---|---|
| 1. 명세 확인 | 공식 코드 commit·가중치 revision·Dolly 분할·정밀도·전역 배치·prefix 방식 대조 | 원문 재현값과 변경값이 구분된 새 protocol |
| 2. SFT 준비 | 교사 SFT·학생 기준선 탐색·3 epoch 초기화 | 각 checkpoint의 출처와 validation 선택 근거 |
| 3. 수식·자원 점검 | 같은 순전파 값, gradient 차이 B, 교사 동결, 마스크, 실제 load/update/save 확인 | 검증 기록과 전용 VRAM·속도 실측. 기존 GPU 작업과 중복 실행 없음 |
| 4. 기준선 확인 | Dolly-only MiniLLM 참조 실행의 종료·평가 절차 확인 | 실제 구현과 원문 차이를 설명할 수 있는 결과 |
| 5. 핵심 대조 | 동일 학생 초기화의 Soft-Allow/Soft-Detach 독립 학습 | 동일 조건·학습량·선택 규칙의 완료 기록 |
| 6. 평가·분석 | 생성 5 seed·보완 채점·calibration·기전 진단 | 점수표·그림·오류 사례·해석의 한계 |
| 7. 보고 | 세 질문별 결과와 주장 정리 | 근거·반례·재현 조건이 연결된 연구보고서 |

새 결과의 예정 경로는 `results/soft_gate_gradient_v1/`, 대형 checkpoint·응답·로그는 Git에서 제외된 별도 로컬 경로다. 실제 생성 전 경로·용량·ignore 규칙을 확인한다. 기존 `results/dolly_sft_v1/`와 checkpoint를 덮어쓰지 않는다. 실행 기록에는 source hash·실제 파라미터·업데이트 수·완료/실패 상태를 포함한다.

현재 남은 KD 동결 항목은 **gate 학습의 prefix/gradient 추정과 학습량, MiniLLM PPO 전역 배치 대응, 독립 학습 반복·정답성 검토·통계 절차**다. 교사 재사용 경로와 학생 SFT의 optimizer·정밀도·배치 대응은 별도 protocol에 고정했다. 확인하지 않은 KD 값을 기존 예비 설정이나 임의의 최적값으로 채우지 않는다.

메타 장치에서 실제 config의 파라미터 수를 확인한 결과 XL은 1,557,611,200개다. FP16 가중치·gradient, FP32 master·Adam의 두 상태를 합한 이론적 크기는 약 **23.21GiB**로 activation을 포함하지 않아도 16GB를 넘는다. 이 때문에 교사는 공식 SFT 가중치를 재사용한다. [메모리 추정](../results/soft_gate_gradient_v1/training_memory_estimate.json)은 학생의 실제 GPU probe와 구분한다.

## 10. 기대 기여와 한계

기대 산출물은 soft gate의 분포 혼합 효과와 gate 경로의 추가 gradient 효과를 구분하는 대조, confidence와 실제 오류의 관계를 보여주는 자료, 허용/차단이 유리한 조건에 대한 설명이다. 개선 폭·우세 조건·학술대회 채택 가능성은 미리 단정하지 않는다.

한 GPT-2 조합, Dolly-only 적응, 기존 평가 자료의 사전 관찰, 독립 학습 반복의 범위를 결론에 포함한다. 효과가 확인되지 않아도 원인과 불확실성을 보고하고, 새로운 제어 규칙의 필요성이 입증되면 별도 후속 방법으로 검토한다. 학습·평가의 완료 여부는 실제 실행 결과로 확인한다.

**OPT는 GPT-2 본 실험 이후의 확장 연구로 둔다.** 먼저 한 쌍에서 gradient 대조와 평가 절차를 검증하고, 효과와 실행 가능성을 확인한 뒤 다른 모델 계열에서의 일반화를 검토한다. 기존 OPT 원본 평가는 중단 시점의 응답·RNG를 로컬에 보존했으며 현재 GPT 실행의 완료 조건에 포함하지 않는다.

## 11. 단계별 지표와 기록 시점

| 단계 | 필수 기록 | 해석·선택 |
|---|---|---|
| 데이터 사전 점검 | 분할별 건수·중복·길이·잘림·마스크·해시 | 성능 지표와 구분 |
| 원본 기준선 | 응답 토큰 평균 NLL/PPL, 아래 생성·분류 지표 | SFT 전후 변화의 출발점 |
| SFT 학습·선택 | train loss·검증 NLL·검증 ROUGE-L·학습/검증 차이 | 교사/학생 SFT는 ROUGE-L, 3 epoch 학생 초기화는 loss 선택 |
| 증류 과정 | FKL·RKL·혼합 loss, gate 분포, 교사/학생 confidence 차이, A/B norm·방향 일치도, 검증 성능 | 동일 checkpoint·prefix의 미분 진단과 독립 학습 곡선을 구분 |
| 생성 최종 평가 | ROUGE-L P/R/F1, METEOR, BERTScore P/R/F1, Distinct-4, 길이·빈 응답·상한 도달률 | 저장 응답 재채점. 보완 지표 때문에 SFT epoch를 늘리지 않음 |
| Confidence·오류 | SST-2/BoolQ accuracy·macro-F1·balanced accuracy·ECE·Brier, 오답 평균 확신, 정답→오답/오답→정답 전환 | Dolly 생성의 정답성은 익명 오류 검토로 별도 확인 |
| 최종 비교 | 학생 초기화 대비 변화량·허용−차단 차이·문항별 변화·생성 seed 평균/SD | sampling 변동을 독립 학습 변동과 구분 |

실행기는 `scripts/run_soft_gate_stage1.py`다. 공유 GPU 잠금 아래 원본 GPT-2 두 모델의 Dolly 5-seed 생성·NLL·분류·보완 채점을 실행한 뒤 학생 SFT 탐색과 별도 3 epoch 초기화를 순서대로 연결한다. 이후 공식 교사·선택된 학생 SFT·학생 초기화를 동일한 Dolly 생성·NLL·분류·보완 지표로 평가한다. KD 학습은 미확정 명세를 고정한 다음 진행한다. 수식·mask·RNG 재개·공유 잠금·토큰 단위 gradient accumulation·optimizer 그룹의 CPU 테스트 8개를 통과했다. 실제 완료 단계는 진행 기록과 결과 파일을 따른다.
