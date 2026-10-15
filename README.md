# 지식증류 · AI Engineering

**큰 교사 모델의 예측을 작은 학생 모델이 배우는 지식증류에서, 학습 방식에 따라 학생의 응답과 성능이 어떻게 달라지는지 탐구한다.**

현재는 **Dolly를 공통 데이터로 고정하고, GPT-2 XL/base와 Qwen3.5-9B/0.8B를 각각 SFT해 비교하는 단계**다. 모델별 기준선을 확보한 뒤 KL 방향·confidence gate·Revival에 따른 변화를 분석한다. 주요 산출물은 확률분포·학습 신호·응답 변화를 설명하는 시각화와 평가 결과다.

## 실험 설계

1. **네 모델의 독립적인 SFT:** 각 모델의 원본 공개 checkpoint에서 시작해 같은 Dolly 학습 자료의 기준 응답으로 LoRA 추가 학습한다. Qwen 원본은 이미 post-training을 거친 모델이다. 이 단계에는 지식증류를 적용하지 않는다.
2. **SFT 기준선 비교:** 같은 평가 문항에서 생성 품질·과제 수행·오류 유형을 확인한다. 각 계열에서 교사가 학생보다 어떤 문제를 잘 풀고, 어떤 문제에서 실패하는지 함께 살펴본다.
3. **지식증류 비교:** 선정한 계열의 교사–학생 쌍에서 고정 Forward/Reverse KL, Hard/Soft gate, 같은 손실의 Revival 유무와 결합을 비교한다. 학생의 SFT 기준선 대비 분포·학습 신호·갱신·응답 변화를 시각화한다.

모델 크기만으로 교사의 지도가 유익하다고 가정하지 않는다. 공통 분할·평가 절차를 유지하고 학습률·checkpoint는 검증 자료로 선택한다. 원문 사용값과 현재 조건에서 검증해 선택한 값을 구분하며, 근거 없는 설정을 기본값으로 확정하지 않는다.

## 모델·데이터 구성

| 계열 | 교사 후보 | 학생 후보 | 현재 확인 범위 |
|---|---|---|---|
| **GPT-2** | [GPT-2 XL](https://huggingface.co/openai-community/gpt2-xl), 약 1.56B | [GPT-2 base](https://huggingface.co/openai-community/gpt2), 약 124M | 둘 다 BF16 가중치, 양자화 없음, LoRA 학습 |
| **Qwen3.5** | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 9B만 NF4·double quantization과 LoRA, 0.8B는 BF16 가중치와 LoRA |

양자화는 16GB GPU에서 9B를 다루기 위한 제약이다. LoRA는 학습할 가중치의 범위를 정하는 별도 선택이며 네 모델에 공통 적용한다. GPT-2는 pretrained 모델, Qwen은 post-trained 모델이므로 시작점·크기·tokenizer·정밀도 차이를 포함한 후보 조합 비교로 해석한다. OPT와 더 큰 GPT-2 모델은 이번 실행에 포함하지 않는다.

공통 데이터는 [Databricks Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)다. 사람이 작성한 영어 지시·응답 **15,011건**으로, 질의응답·분류·정보 추출·요약·창작 등 8개 과제를 포함한다.

- `instruction`: 수행할 지시나 질문
- `context`: 응답에 필요한 참고 본문. 없는 예제도 있음
- `response`: SFT에서 학습할 기준 응답
- `category`: 과제 유형

Dolly를 고정해 모델·증류 방식의 차이에 집중한다. 실제 입력은 MiniLLM이 공개한 Dolly 분할이다. 지시문 중복과 네 tokenizer의 길이를 함께 확인한 공통 분할은 **학습 11,117건·검증 975건·선정 평가 500건**이다. 응답을 잘라 맞추지 않고 길이 초과 예제를 공통 제외한다. UltraChat·MetaMathQA는 확장 후보로 보관한다. 상세 근거·제외 수·설정 선택은 [현재 SFT 비교 계획](DOLLY_SFT_MODEL_SELECTION.md)에 기록한다.

## 평가 기준

| 평가 대상 | 확인할 지표와 자료 |
|---|---|
| **Dolly의 생성 응답** | ROUGE-L·reference NLL, 과제별 응답 품질·지시 준수·실패 사례 |
| **정답 label이 있는 공통 과제** | SST-2·BoolQ 등의 accuracy·macro-F1·클래스별 오류. 공통 평가 문항과 판정 규칙 사용 |
| **증류 전후의 학생 변화** | 학생의 SFT 기준선 대비 수행 변화, 교사·학생의 확률분포, gate·불확실성·갱신 수용/거절 |

자유 생성의 ROUGE를 문항 정답률로 해석하지 않는다. Revival의 지도 유익성을 분석할 때는 신뢰할 수 있는 정오 판정 기준을 별도로 마련한다. 학습·설정 선택용 검증·최종 평가의 역할을 구분하고, 이미 모델 선정에 사용한 점수를 독립적인 최종 증거로 부르지 않는다.

## 핵심 개념

| 개념 | 역할 |
|---|---|
| **KL** | 교사와 학생의 확률분포 차이를 측정하는 학습 기준. Forward KL은 교사가 지지하는 후보를 학생이 놓치는 데, Reverse KL은 학생이 고르는 후보를 교사가 지지하지 않는 데 민감하다. |
| **Confidence gate** | 다음 토큰 확률이 얼마나 집중되는지를 나타내는 confidence를 비교해 KL 방식을 조절한다. Hard gate는 하나를 선택하고 Soft gate는 두 KL을 연속적인 비중으로 섞는다. |
| **Revival** | 모델 일부를 무작위로 끄며 여러 번 예측해 불확실성을 추정하고, 각 모델의 과거 기준과 비교한다. 교사가 상대적으로 불확실하고 학생이 상대적으로 안정적인 조건에서 해당 증류 갱신을 억제한다. |

높은 confidence가 정답을 보장하지는 않는다. 이번 연구에서는 각 장치의 동작과 실제 응답·성능 변화를 함께 확인한다. Hinton은 지식증류의 기초, MiniLLM은 KL 방향, CaRE-KD는 gate와 Revival의 근거로 사용한다.

## 진행 현황

2026-10-15 기준, **네 모델의 사전 점검을 마치고 본 SFT를 시작했다.** 실행은 GPT-2 base → XL → Qwen3.5-0.8B → 9B 순서이며 모델별 두 학습률 후보를 비교한다.

- [x] 참고논문 해설과 실험 비교 방향 정리.
- [x] GPT-2 XL/base의 GPU 적재·추론 및 일회성 학습 갱신 점검.
- [x] Dolly 원자료 확보·버전 고정·구성 및 길이 집계. UltraChat·MetaMathQA의 예비 조사 기록도 확보.
- [x] 저자가 공개한 Dolly-SFT XL/base 모델의 기준선 평가와 결과 시각화.
- [x] 네 공개 모델의 revision 고정과 계열 내 어휘 ID 일치 확인.
- [x] 공통 분할·중복 제외·길이 제한·두 학습률 후보의 사전 명세.
- [x] 수정된 정밀도에서 네 모델의 실제 LoRA 갱신·재적재와 16건 누적 Trainer checkpoint 저장 확인.
- [ ] Dolly에서 네 모델의 본 SFT와 공통 평가 수행.
- [ ] 지식증류 비교 실험과 응답 변화 분석·시각화.

현재 완료된 성능 점수는 저자가 공개한 GPT-2 SFT 모델의 평가 결과다. 이번 직접 학습 결과는 후보별 완료 후 별도 기록한다. 본 SFT가 진행 중이며 지식증류 결과는 아직 없다. 일회성 갱신 통과는 장기 학습 안정성을 보장하지 않는다.

## 관련 문서

- [연구계획](RESEARCH_PLAN.md) · [연구보고서 초안](RESEARCH_REPORT.md)
- [현재 Dolly 네 모델 SFT 비교 계획](DOLLY_SFT_MODEL_SELECTION.md) · [실행 명세와 점검 결과](results/dolly_sft_v1/)
- [데이터 선정 조사](DATA_SELECTION.md) · [이전 데이터 후보별 SFT 비교 설계](DATASET_SFT_COMPARISON.md)
- [공개 SFT 모델 평가 결과](SFT_BASELINE_EVALUATION.md) · [Dolly 구성](DOLLY_DATA_OVERVIEW.md)
- 논문 해설: [Hinton](papers/01_hinton_2015_ko.md) · [MiniLLM](papers/02_minillm_2024_ko.md) · [CaRE-KD](papers/03_care_kd_2026_ko.md)

연결된 계획·보고서의 과거 검토에는 이전의 데이터 3종 × 모델 2종 설계가 남아 있다. 현재 실행 범위는 **Dolly 고정·GPT-2와 Qwen 네 모델의 LoRA SFT 비교·9B만 NF4**다. 진행 상태는 결과 폴더의 실행 기록을 따른다.
