# Knowledge Distillation · AI Engineering

지식증류 인공지능공학: 생성형 언어모델의 지식증류 원리와 학생 모델의 응답 변화를 탐구하는 학습 저장소.

연구 방향은 **참고문헌의 인사이트를 수식·그래프·표·사례로 인용하고, 그 적용 조건을 설명·검증하는 것**이다. 학습 시간과 메모리는 이 컴퓨터에서 연구를 수행할 수 있는지 판단하는 제약으로 기록한다.

**현재 목표는 ‘생성형 언어모델의 지식증류 원리와 학생 모델의 응답 변화 탐구’다.** 2026-10-15에 확정했다. Hinton을 기초로, MiniLLM을 KL 방향의 근거로, CaRE의 gate와 Revival을 손실 적응·학습 갱신 거절 비교의 근거로 사용한다. **Dolly-15K를 공통 instruction 데이터로 우선 선정하고, Forward/Reverse KL과 Hard/Soft gate에 따른 손실 변화, Revival에 따른 학습 갱신 수용·거절, 두 장치의 결합을 비교·시각화한다.** 모델·데이터 revision/분할·실행 프로토콜·설정·예산은 아직 미정이다. 이전 LightJudge 실험은 필요할 때 대조할 기존 근거이며, 이 저장소의 출발 가설이나 실행 계획으로 자동 승계하지 않는다.

**기초 이해 → MiniLLM의 KL·CaRE의 gate/Revival 분석 → 같은 모델·데이터에서 손실 적응과 갱신 거절 비교·시각화** 순서로 진행한다. MiniLLM의 원문 구성을 먼저 검토하되 전체 재현을 필수 목표로 삼지 않는다. 구현할 방법과 변경 사항을 밝히고, 일부 요소만 적용한 실험을 MiniLLM 전체 재현으로 표시하지 않는다. **CaRE의 gate와 Revival은 현재 범위에 포함하고 한국어 적용은 후속 탐구로 둔다.** 전체 MiniLLM·CaRE-KD 재현을 필수 목표로 삼지 않는다. [계획서 §5](RESEARCH_PLAN.md)에 목표와 범위를 정리했다.

[연구보고서 초안](RESEARCH_REPORT.md)에 연구 질문·이론·비교 방법·모델 추천·시각화·결과 작성 틀을 정리했다. 실험 결과는 아직 없으며 1차 모델 추천은 GPT-2 XL 교사 → GPT-2 base 학생이다.

[논문 기반 연구계획서](RESEARCH_PLAN.md)에 논문별 가정·근거, 연구 질문 선정 기준, 모델·데이터 선정 절차와 T를 비롯한 설정의 검증·선택 방법을 정리했다. 원문 사용값과 현재 조건의 최적값을 구분하며, 미검증 수치는 미정으로 남긴다.

[모델 선정용 벤치마크 검토](MODEL_BENCHMARK_REVIEW.md)에 능력별 평가 후보, 공식 보고서의 소형 모델 비교 사례와 교사·학생을 같은 문제에서 확인하는 절차를 정리했다. 공개 점수는 선정 근거로 기록하고, 로컬 실측과 구분한다.

[데이터 선정 근거와 후보](DATA_SELECTION.md)에서 기준 데이터 Dolly-15K와 MiniLLM의 원문 분할·전처리를 대조한다. 공통 분할·실제 사용 규모는 미정이다. 기존 UltraChat·MetaMathQA·WizardCoder와 한국어 자료 조사는 후속 탐구 자료로 보관한다.

## 읽기 순서

| 순서 | 논문별 한국어 문서 | 붙잡을 질문 | 문서 상태 |
|---|---|---|---|
| 1 | [Hinton 외: 신경망의 지식 증류](papers/01_hinton_2015_ko.md) | 확률분포에서 무엇을 전달하며, T가 학습 신호를 어떻게 바꾸는가? | 한국어 해설·독립적인 수식 풀이. 전문 번역은 미작성 |
| 2 | [MiniLLM: 대규모 언어모델의 지식 증류](papers/02_minillm_2024_ko.md) | 생성에서는 왜 KL 방향과 응답을 얻는 방식이 함께 중요한가? | 한국어 해설·독립적인 수식 풀이. 전문 번역은 미작성 |
| 3 | [CaRE-KD: 중요한 것의 증류](papers/03_care_kd_2026_ko.md) | 손실을 바꾸는 것과 학습 갱신을 거르는 것은 어떻게 다른가? | gate·Revival의 개별 역할과 결합이 현재 분석 대상. 전체 benchmark 재현 범위는 미정 |

문서의 `[원문 요지]`와 `[해설]`을 구분한다. 논문의 주장은 저자의 주장으로 읽고, 가정·실험 근거·해석 범위를 함께 확인한다. 수치 예시는 `[학습용 예시]`, 연구 후보는 `[열린 질문]`으로 표시한다. 예시 수치를 논문의 실측 결과로 사용하지 않는다.

## 진행 흐름

1. **기초 이해:** 다음 토큰 예측과 문장 생성 → 출력분포 → 교사·학생 → soft target·온도·CE/KL.
2. **논문 분석:** Hinton의 기초, MiniLLM의 KL 방향, CaRE의 gate·Revival 정의와 근거를 수식·그림·표에 연결한다.
3. **제한된 실험과 해석:** 같은 교사·학생·Dolly 분할·학습 context에서 KL·gate를 비교하고 같은 손실의 Revival 유무를 대조한다. 초기화·불확실성 추정·실제 갱신·공통 평가·로컬 실행 가능성을 확인한 뒤 손실·gate·수용/거절·응답 변화를 시각화한다.

일차 산출물은 **원리 설명, 논문 근거 정리, 증류 방식별 시각화와 응답·평가 결과의 해석**이다. [계획서 §5.2.1](RESEARCH_PLAN.md)에 비교 방법과 시각화 설계를 정리했다. 추가 기전 분석이나 새 방법 제안은 관측 결과를 바탕으로 필요할 때 후속 탐구로 검토한다.

읽기는 한 개념씩 진행한다. **문장 → 모르는 개념 → 숫자 예시 → 수식 → 논문의 가정으로 돌아가기**를 반복한다. 읽기 단계에서 새 모델이나 데이터가 해결책이라고 먼저 확정하지 않는다.

세 문헌의 분석 문서는 작성했으며, 공식 구현·실행 설정의 확인은 일부 남아 있다. [계획서](RESEARCH_PLAN.md)의 §4.4–4.5에 인용할 자료와 분석 상태를, §6에 향후 선정 기준을 정리했다. 2026-10-14에 확인한 장비는 **RTX 5060 Ti 16GB급, 시스템 RAM 31.62 GiB, Ryzen 5 7500F**다. 장비 확인만으로 특정 모델의 학습 가능성을 확정하지 않는다.

## 세 논문을 연결하는 기준

| 관점 | Hinton | MiniLLM | CaRE-KD |
|---|---|---|---|
| 출발 문제 | 배포하기 어려운 모델의 예측 정보를 전달 | 용량이 제한된 학생이 복잡한 생성분포를 학습 | 교사의 지도 신호가 항상 믿을 만하지 않음 |
| 먼저 이해할 대상 | 입력에 조건부인 클래스 분포 | 입력에 조건부인 응답 시퀀스 분포 | 같은 생성 context에서의 토큰 분포와 불확실성 |
| 핵심 이론 | softmax 온도와 soft CE의 미분 | KL 방향과 표본분포를 함께 미분 | 가중 손실의 곱미분과 선택적 갱신 |
| 연결할 질문 | 어떤 확률 차이가 유용한 지식인가? | 어디에서 생성한 응답으로 학습하는가? | confidence가 유익한 지도 신호를 구분하는가? |

이 표는 문헌을 연결하는 해설이다. 현재는 Hinton·MiniLLM과 CaRE의 gate·Revival을 연결하며 각 구성요소와 결합의 역할을 구분한다. 상세 근거는 각 논문 문서의 출처를 따른다.

## 원문과 버전

기준일: 2026-10-14, Asia/Seoul.

| 논문 | 고정한 버전 | 공식 원문 |
|---|---|---|
| Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network* | arXiv:1503.02531v1, 2015-03-09 | [원문](https://arxiv.org/abs/1503.02531v1) · [PDF](https://arxiv.org/pdf/1503.02531v1) |
| Gu, Dong, Wei, Huang, *MiniLLM: Knowledge Distillation of Large Language Models* | arXiv:2306.08543v4, 2024-04-10; ICLR 2024 | [원문](https://arxiv.org/abs/2306.08543v4) · [HTML](https://arxiv.org/html/2306.08543v4) · [공식 코드](https://github.com/microsoft/LMOps/tree/main/minillm) |
| Sengupta, Seth, Chakraborty, *Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models* | arXiv:2609.36734v1, 2026-09-29 | [원문](https://arxiv.org/abs/2609.36734v1) · [HTML](https://arxiv.org/html/2609.36734v1) · [PDF](https://arxiv.org/pdf/2609.36734v1) |

MiniLLM에는 더 최근 버전이 있지만 처음 제공받은 학습 맥락과 맞추어 v4를 고정했다. 버전을 바꾸면 변경 내용을 먼저 대조하고 문서에 명시한다. 논문에 적힌 설정과 공식 실행 코드에서 확인한 설정은 따로 기록한다.

## 막히는 개념에 필요한 보조 논문

논문 수를 늘리는 것이 목적이 아니다. 아래 문헌은 해당 개념에서 막힐 때 필요한 절만 읽는다.

| 읽을 때 | 공식 문헌 | 확인할 개념 |
|---|---|---|
| MiniLLM의 학습 prefix가 이해되지 않을 때 | [Agarwal 외, On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) | 학생이 생성한 시퀀스로 학습하는 이유와 분포 불일치 |
| confidence와 정답률을 혼동할 때 | [Guo 외, On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | calibration의 정의; 증류 온도와 사후 temperature scaling의 구분 |
| CaRE의 MC dropout·BALD에서 막힐 때 | [Gal·Ghahramani, Dropout as a Bayesian Approximation, ICML 2016](https://proceedings.mlr.press/v48/gal16.html) | 확률적 forward 간 불일치와 불확실성 추정의 전제 |

현재는 읽기 후보와 연결 이유를 확인한 상태다. 이 보조 논문들의 전문 분석을 완료했다고 표시하지 않는다.

## 번역·출처 표시

- CaRE-KD 원문은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)이다. 한국어 번역과 재구성 사실, 원저자와 원문 링크를 해당 문서에 표시한다. 번역과 해설은 원저자의 공식 번역이나 보증이 아니다.
- 확인한 Hinton·MiniLLM arXiv판에는 [arXiv 배포 라이선스](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)가 붙어 있다. 이를 일반적인 번역·재배포 허가로 취급하지 않는다. 현재 공개 문서는 짧은 원문 요지, 수학적 사실, 직접 작성한 풀이와 읽기 질문으로 구성한다. 원문 전체 번역을 했다고 표시하지 않는다.
- 원문 PDF·HTML과 추출 텍스트는 로컬 `.sources/`에만 보관하며 Git에서 제외한다. 공개 저장소에는 출처 링크와 작성한 학습 문서만 올린다.
- 모든 원문의 라이선스와 작성한 해설의 권리를 한꺼번에 덮는 저장소 전체 라이선스는 지정하지 않았다.

## 다음 읽기

[첫 문서](papers/01_hinton_2015_ko.md)의 **“한 문제에서 학생은 어떤 신호를 받는가?”**부터 시작한다. 이어 [MiniLLM 분석](papers/02_minillm_2024_ko.md)으로 생성 LLM의 증류를 이해한다. 다음 결정은 응답에서 관찰할 변화와 실행할 실험 범위이며, 이를 정한 뒤 모델·데이터를 선정한다.
