# 생성형 언어모델의 지식증류에서 손실 적응과 선택적 학습 갱신의 비교 및 시각화

작성일: 2026-10-15, Asia/Seoul

<!-- SFT_PROGRESS:START -->

현재 비교 범위는 **5개 계열·10개 모델**이며, SFT 선택 완료 **2/10**, 생성 주 평가 4종 완료 **1/10**, 기존 Dolly·분류 보조 평가 완료 **2/10**이다. 생성 평가는 공통 seed 10 한 개를 사용한다. 모델별 상태·선택 LR·검증 NLL·실제 정밀도는 [SFT 진행 기록](SFT_PROGRESS.md), 교사·학생의 생성 성능 차이는 [생성 주 평가](../results/generation_benchmarks_v1/REPORT.md), 분류 편향 등은 [기존 Dolly·분류 보조 평가](../results/dolly_sft_v1/REPORT.md)를 따른다. 최종 조합은 아직 확정하지 않았다.

<!-- SFT_PROGRESS:END -->

문서 상태: **연구보고서 초안. 현재 실행은 Dolly 고정·GPT-2·Qwen3.5·OPT·OpenLLaMA·Gemma-2의 총 10개 모델 LoRA SFT 비교다. 기존 GPT-2 두 모델·Qwen0.8B는 BF16, Qwen9B는 NF4이며, 추가 모델은 가중치 크기와 실제 전용 VRAM 점검에 따라 정밀도를 정한다.** 공통 분할·설정 근거·실행 상태는 [현재 비교 계획](DOLLY_SFT_MODEL_SELECTION.md)과 [실행 결과](../results/dolly_sft_v1)를 따른다. 아래의 데이터 3종 비교와 두 모델 관련 내용은 이전 검토·관측 기록이며, 공개 SFT 모델 평가와 이번 직접 학습 결과를 합치지 않는다. KD 방법 비교 결과는 아직 없다.

이 문서는 연구의 문제·이론·방법·모델 추천과 보고할 자료를 연결한다. 상세 실행·설정 선택 절차는 [연구계획서](RESEARCH_PLAN.md), 기존 분석은 [Hinton](../papers/01_hinton_2015_ko.md)·[MiniLLM](../papers/02_minillm_2024_ko.md)·[CaRE-KD](../papers/03_care_kd_2026_ko.md)를 따른다. 원문 주장, 우리 설계, 실제 관측을 구분하며 결과가 없는 곳에 수치를 채우지 않는다.

| 항목 | 현재 상태 |
|---|---|
| 연구 중심 | KL 방향·Hard/Soft gate의 손실 적응, Revival의 갱신 거절, 두 장치 결합의 역할 비교 |
| 현재 데이터 | Dolly 고정. 공통 학습 11,117건·검증 975건·선정 평가 500건. UltraChat·MetaMathQA는 후속 후보 |
| 비교 모델 | GPT-2 XL/base, Qwen3.5-9B/0.8B, OPT-2.7B/125M, OpenLLaMA-7B/3B, Gemma-2-9B-IT/2B-IT. LoRA 공통 적용, 모델별 원본 revision·어휘 대응·정밀도와 근거 기록 |
| 양자화 조건 분담 | 팀원은 비양자화 SFT, 현재 컴퓨터는 기존 BF16/NF4 정책 유지. 동일 모델·자료·LoRA·학습·평가 조건을 대조하고 외부 결과는 분리 보관. 비교 결과 미확보 |
| 추가 비교 후보 | SeqKD와 교사 응답 증강. 실제 실험 포함 여부는 미정 |
| 실험·수치·그래프 | 기반 모델 점검과 공개 SFT 기준선 평가를 별도 기록. KD 학습·방법 비교는 미실행. §9.3은 KD 예정 자료 |
| 생성 평가 반복 | 모델 선정은 공통 seed 10 한 개, 최종 선정 모델의 본 실험은 10/20/30/40/50의 5개 생성 seed. 현재 단일 seed에서 표준편차 미추정 |
| 후속 범위 | 한국어 적용, 다른 모델·과제로의 확장 |

## 1. 초록

본 연구는 생성형 언어모델의 지식증류에서 **어떤 손실로 배울지 결정하는 것**과 **학습 갱신을 받아들일지 결정하는 것**의 역할을 비교·시각화하는 것을 목적으로 한다. Hinton의 soft target·온도 개념, MiniLLM의 KL 방향과 생성 학습 관점, CaRE-KD의 confidence gate와 Revival을 이론적 근거로 삼는다. 같은 교사·학생과 최종 선정 데이터의 공통 조건에서 고정 FKL/RKL, Hard/Soft gate, 같은 손실의 Revival 유무와 결합을 대조하도록 설계한다. 확률분포·gate·불확실성·갱신 수용/거절·생성 응답을 연결해 해석한다. **이 초안에는 KD 방법 비교나 개선 효과에 대한 결론이 없다.**

## 2. 연구 배경과 목적

생성형 LLM의 학생은 교사가 가진 복잡한 응답 분포를 제한된 용량으로 학습한다. 어떤 확률 정보를 전달하고 어떤 생성 context에서 학습하느냐에 따라 학습 신호가 달라질 수 있다. 또한 교사의 confidence와 불확실성은 실제 정답 여부와 같은 개념이 아니므로, 지도 신호를 따르는 방식을 분석할 필요가 있다. [Hinton §2](https://arxiv.org/pdf/1503.02531v1), [MiniLLM §1–2](https://arxiv.org/html/2306.08543v4), [CaRE §2](https://arxiv.org/html/2609.36734v1).

본 연구는 문헌에서 제시한 개입을 분리해 읽고, 실제 모델에서 어떤 차이가 관측되는지 설명하는 비교·분석 연구다. 주요 산출물은 원리 설명, 출처가 있는 수식·자료, 방법별 시각화, 응답·평가 결과와 한계다. 계산 시간과 메모리는 현재 컴퓨터에서 연구를 끝낼 수 있는지 판단하는 제약으로 기록한다.

## 3. 이론적 배경

### 3.1. 생성형 LLM과 지식증류의 위치

생성형 LLM은 입력과 앞서 생성한 토큰에 조건부로 다음 토큰의 확률을 계산한다. 교사 분포 P와 학생 분포 Q를 비교할 때는 같은 입력·prefix·어휘 좌표인지 확인해야 한다.

지식 유형의 일반적인 분류는 응답/출력 기반·특징 기반·관계 기반이다. 확률분포를 이용한 이번 비교는 응답/출력 기반에 속한다. 세 유형 전체를 실험 대상으로 삼는 것은 아니다. [Gou 외, KD Survey §2](https://arxiv.org/pdf/2006.05525v7).

### 3.2. Soft target·온도·CE/KL

실제 정답 토큰을 지도하는 것과 교사의 확률분포를 전달하는 것은 다른 학습 신호다. 온도 T는 logits를 나누어 분포를 바꾸며 그라디언트 크기에도 영향을 준다. Hinton의 온도·보정 설명은 적용 가정과 함께 읽는다. 원문이 특정 값을 사용했다는 사실을 현재 조건의 최적성으로 해석하지 않는다. [Hinton §2–2.1](https://arxiv.org/pdf/1503.02531v1).

같은 context에서 다음과 같이 정의한다.

$$
D_F=KL(P\|Q),\qquad D_R=KL(Q\|P).
$$

FKL/RKL의 차이는 손실 방향의 차이다. 생성 학습에서는 학습 응답과 prefix의 출처도 확인해야 한다. 단순한 토큰 RKL 구현과 policy gradient·sampling·안정화 등을 포함하는 MiniLLM 전체 알고리즘을 구분한다. [MiniLLM §2](https://arxiv.org/html/2306.08543v4).

## 4. 관련 연구 분석

| 문헌 | 분석할 인사이트 | 연결할 근거 | 이번 보고서에서의 역할 |
|---|---|---|---|
| Hinton | soft target·온도와 전달되는 확률 정보 | §2–2.1의 수식, 과제별 온도 실험 조건 | KD의 기초와 설정값 해석 |
| MiniLLM | 생성분포의 KL 방향, 학습 응답·prefix의 출처 | Figure 2–3, §2–3의 방법·평가 | 고정 FKL/RKL을 읽는 배경 |
| CaRE-KD | 손실 적응과 갱신 거절의 분리·결합 | §2–3, Figure 2–3, Table 2 | Hard/Soft gate·Revival 비교의 직접 근거 |

원문 자료는 저자의 주장·보고값으로 인용하고 우리 결과와 분리한다. 원문 그림을 설명하거나 수치로 재작성할 때 버전·조건·축·오차 정의와 변경 여부를 적는다. 수식 기반 예시는 학습용임을 표시한다. 기존 한국어 분석 문서가 있다는 이유만으로 모든 주장을 독립 검증했다고 표시하지 않는다.

## 5. 연구 질문과 비교 방법

### 5.1. 연구 질문

- **RQ1:** 같은 조건에서 FKL과 RKL은 학습 신호·분포·응답을 어떻게 다르게 변화시키는가?
- **RQ2:** Hard/Soft gate의 선택·혼합과 미분 차이는 어떤 동작·결과로 나타나는가?
- **RQ3:** 같은 손실에서 Revival을 적용하면 어떤 갱신이 수용·거절되고 결과는 어떻게 달라지는가?
- **RQ4:** 손실 적응과 갱신 거절을 결합하면 각 장치만 사용한 경우와 무엇이 달라지는가?

### 5.2. 손실 적응: FKL/RKL과 Hard/Soft gate

공통 표현을 다음과 같이 둔다.

$$
L_g=gD_F+(1-g)D_R.
$$

| 조건 | 정의·역할 | 비교할 내용 |
|---|---|---|
| FKL | g=1로 고정 | 해당 context에서 교사 분포를 맞추는 신호 |
| RKL | g=0으로 고정 | 방향 변경에 따른 신호·분포 변화 |
| Hard gate | confidence 차이로 두 분기 중 하나를 선택 | 토큰별 선택과 경계 부근의 동작 |
| Soft gate | confidence 차이로 혼합 비중을 연속적으로 결정 | 혼합 비중과 gate를 통한 미분의 영향 |

CaRE의 기준 정의는 c(P)=1−H(P)/log|V|, Δc=c(P)−c(Q), g_hard=1[Δc≥0], g_soft=σ(Δc−m)이다. Hard/Soft는 **gate**의 구분이며 hard label/soft target과 다르다. 원문에서는 margin·연속 혼합·gate의 미분 여부가 함께 달라지므로 결과를 연속성 하나의 효과라고 단정하지 않는다. [CaRE §2.1·3.1](https://arxiv.org/html/2609.36734v1).

m·추가 scale·온도는 현재 실험 값으로 확정하지 않았다. 기존 분석에서 확인한 식 (3)과 Figure 2의 scale 차이는 공식 실행 config와 대조할 항목이다.

### 5.3. 선택적 갱신: Revival

Revival은 MC dropout으로 추정한 교사·학생 BALD와 각각의 과거 분포 quantile을 이용한다. 단일 분포의 entropy confidence를 그대로 같은 추정량으로 사용하지 않는다. **손실을 선택하는 gate와 갱신을 수용·거절하는 Revival은 신호와 개입 위치가 다르다.** [CaRE §2.2](https://arxiv.org/html/2609.36734v1).

이번 비교에서는 원문의 추정·mask 집계 단위와 실제 optimizer 동작을 확인한다. loss·gradient를 0으로 만드는 것과 optimizer step 생략은 구분한다. MC dropout이 비퇴화 신호를 만드는지와 그 신호의 유익성도 별개로 읽는다. 거절 여부 자체를 실제 오답의 판정으로 사용하지 않는다.

### 5.4. 교사 응답 학습·증강 — 추가 후보

SeqKD는 교사가 생성한 문장을 학습 대상으로 사용하는 방식이다. 같은 선정 데이터의 학습 입력에 교사 응답을 생성하거나 여러 응답을 선별해 추가하는 구성을 관련 방법으로 설명한다. 실제 실험 포함 여부는 아직 미정이다. [SeqKD 원문](https://aclanthology.org/D16-1139/), [MiniLLM baseline](https://arxiv.org/html/2306.08543v4).

추가한다면 교사 응답·원본 응답의 역할, 생성·선별 기준, 같은 원본에서 파생한 group과 학습량을 기록한다. 응답 증강 효과와 단순 학습량 증가의 효과를 구분한다.

## 6. 데이터와 모델 추천

### 6.1. 데이터

현재 학습 후보는 Dolly·UltraChat·MetaMathQA다. 같은 원본 XL/base에서 각각 SFT한 뒤 공통 평가와 과제별 평가로 선정한다. [비교 프로토콜](DATASET_SFT_COMPARISON.md)에 실제 전체 구조·길이·중복 집계, group 분할·assistant mask·채점과 설정 선택 절차를 정리했다. 원자료 revision은 고정했으나 최종 학습 분할·규모·설정은 검증 전이다. 사용자는 별도 시간 제한을 두지 않았다.

train·development·최종 평가의 역할을 분리하며 증강 자료는 원본 문제 group을 따라 관리한다. 원문의 분할 설명과 공개 전처리 코드의 동작 차이는 [데이터 선정 문서](DATA_SELECTION.md)의 확인 사항을 따른다. MiniLLM 원문의 D_PT를 채택할지는 별도 결정이며 일부 요소의 채택을 전체 재현으로 표시하지 않는다.

### 6.2. 1차 추천: GPT-2 XL → GPT-2 base

**현재 공통 비교 모델은 GPT-2 XL과 GPT-2 base다. 데이터별 SFT 수행을 확인한 후 최종 교사·학생을 정한다.** 이 조합은 MiniLLM과 CaRE-KD의 instruction-following 실험 모두에 연결된다. [MiniLLM §3.1](https://arxiv.org/html/2306.08543v4), [CaRE §4](https://arxiv.org/html/2609.36734v1).

| 우선순위 | 교사 기반 모델 ID | 학생 기반 모델 ID | 추천 이유·제약 |
|---|---|---|---|
| 초기 사용 | [openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl) | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | 두 문헌과 공통으로 연결. 기반 모델의 GPU 동시 적재·짧은 입력 추론 확인. 데이터별 본 SFT·장기 학습은 미검증 |
| 대안 | [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b) | [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) | CaRE의 원문 조합. 더 큰 교사의 적재·MC forward 부담을 확인해야 하며 다른 계열을 바로 추가할 필요는 없음 |
| 이후 확장 | 같은 GPT-2 XL 교사 | [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) | 초기 결과 이후 학생 크기에 따른 차이를 추가 관찰할 후보. 초기 필수 실험은 아님 |

논문은 GPT-2 학생을 120M급으로 표시하지만 위 base 모델 카드는 124M으로 설명한다. 실제 선정은 모델 ID·revision으로 고정하고 표기 차이를 기록한다. 숫자가 비슷한 다른 checkpoint로 조용히 교체하지 않는다. [GPT-2 모델 카드](https://huggingface.co/openai-community/gpt2).

추천 근거는 원문과의 연결, logits 계측과 공통 어휘 확인 가능성, dropout 경로, 여러 조건을 반복할 수 있는 실행 범위다. 이후 §7.3의 기반 모델 동시 적재·짧은 추론을 확인했다. 학습·MC 경로의 실행 가능성은 여전히 미검증이며 최신 채팅 모델의 종합 성능 순위로 제시하지 않는다. 초기에는 이 두 모델만 사용하고 OPT·medium은 추가하지 않는다.

### 6.3. 확인한 config와 남은 준비

2026-10-15에 공개 config를 조회했다. GPT-2 XL과 base에는 vocab_size=50,257 및 attention·embedding·residual dropout 설정이 있고, OPT-2.7B/125M에는 vocab_size=50,272와 dropout 설정이 있다. 이는 **배포된 config의 내용**이며 우리 실험의 MC dropout 최적 설정을 뜻하지 않는다. [GPT-2 XL config](https://huggingface.co/openai-community/gpt2-xl/raw/main/config.json), [GPT-2 base config](https://huggingface.co/openai-community/gpt2/resolve/main/config.json), [OPT-2.7B config](https://huggingface.co/facebook/opt-2.7b/raw/main/config.json), [OPT-125M config](https://huggingface.co/facebook/opt-125m/resolve/main/config.json).

같은 vocab_size만으로 어휘 ID·tokenizer의 일치를 확정하지 않는다. 이번에 고정한 기반 revision에서는 전체 어휘 ID와 tokenizer backend, 점검 입력의 ID 일치를 실제 확인했다. 배치 학습용 특수 토큰·padding·prompt·mask는 별도 대조한다. config의 dropout 존재만으로 MC-BALD가 유효하다고 판단하지 않으며 실행 시 활성 경로를 확인한다.

위 ID는 **기반 모델**이다. 원문과 연결하려면 선정 데이터로 준비한 교사·학생의 SFT 초기 상태를 확보·기록해야 한다. 준비된 checkpoint의 데이터 분할·학습 이력을 확인하거나 공통 train 자료로 준비한다. 공개 SFT baseline을 MiniLLM의 초기 checkpoint와 같다고 가정하지 않는다. 실제 사용할 준비된 checkpoint·revision은 미정이다.

## 7. 실험 설계와 실행 환경

### 7.1. 구성요소 대조

| 대조 구조 | 손실 | Revival | 확인할 역할 |
|---|---|---|---|
| 고정 기준선 | FKL 또는 RKL | 없음 | KL 방향의 차이 |
| 손실 적응만 | Hard 또는 Soft | 없음 | 손실 선택·혼합의 역할 |
| 갱신 거절만 | 대응 기준선과 같은 손실 | 있음 | 같은 손실에서 갱신 수용·거절의 역할 |
| 결합 | 대응 조건과 같은 gate | 있음 | 같은 gate에 Revival을 추가한 변화 |

모든 조합의 실행을 확정한 것은 아니다. 원문 설정·구현·자원 확인 후 실제 목록을 동결한다. [CaRE Table 2](https://arxiv.org/html/2609.36734v1#S4.T2)는 같은 손실의 Revival 유무를 대조하는 근거이며 위 표 전체가 원문과 동일한 실험이라고 표시하지 않는다.

### 7.2. 공통 조건과 구현 확인

- 교사·학생 초기 상태, 데이터·분할, tokenizer·prompt·mask, 평가와 학습량 정의를 기록한다.
- reference/teacher/student 중 학습 prefix를 어디에서 얻는지 정하고 조건별로 같은 규칙을 사용한다. SGO와 teacher forcing은 아직 채택 전이다.
- 같은 입력의 고정 진단과 실제 자유 생성·학생 rollout의 변화를 구분한다.
- 같은 손실에서 Revival 유무를 비교하고 처리 입력 수와 실제 갱신 수를 함께 남긴다.
- 서로 다른 목적함수의 training loss 숫자만으로 방법의 성능을 순위화하지 않는다.

### 7.3. 로컬 자원과 실행 통과 기준

기존 장치 조회에서 RTX 5060 Ti 16GB급과 시스템 RAM 31.62 GiB를 확인했다. 2026-10-15 14:27(Asia/Seoul)에 GPT-2 XL/base를 동시에 GPU에 적재하고 같은 짧은 입력의 forward를 통과했다. 기반 모델의 실제 고유 파라미터 수는 각각 **1,557,611,200 / 124,439,808**이었다. revision·입력·검사 항목은 [계획서 §6.4](RESEARCH_PLAN.md), 원자료는 [적재·추론 실행 기록](../results/model_preflight.json)을 따른다.

점검은 float16·eager attention·eval·gradient 비활성 상태에서 `Knowledge distillation transfers`의 마지막 위치를 읽었다. 별도 온도 변환·sampling은 없었다. 전체 50,257개 어휘에서 유한 logits와 정규화된 확률을 확인했다. PyTorch allocator의 peak allocated는 **3,350.71 MiB**, peak reserved는 **3,368.00 MiB**, 다운로드를 제외한 적재·forward 시간은 **8.24초**였다. 전체 장치 메모리나 학습 요구량으로 읽지 않는다.

이 입력·정밀도는 적재와 계측 경로를 확인하기 위한 선택이며 최적 학습 설정이나 대표 평가 표본이 아니다. 가중치·공통 입력 logits는 Git에서 제외한 로컬 `tmp/`에 보관했다. 확인 명령 종료 후 GPU 모델은 해제되며 상시 서버는 시작하지 않았다.

후보 모델의 공통 입력과 계획상 긴 입력에서 손실·gate·MC 추정·실제 갱신·생성·기록을 시험한다. peak memory·시간·저장량과 반복 가능 범위를 확인한 뒤 batch·길이·정밀도·예산을 정한다. 양자화·LoRA·길이 변경을 채택하면 분포·학습 조건에 미치는 차이를 기록한다. GPU 용량만으로 특정 조합의 학습 성공을 보장하지 않는다.

### 7.4. KD 전의 SFT 기준선 평가

2026-10-15에 두 모델의 SFT 후 수행을 확인하는 별도 평가를 추가했다. MiniLLM 저자가 Dolly로 SFT해 공개한 `teacher-gpt2-1.5B`와 `SFT-gpt2-120M`의 revision을 고정해 현재 장치에서 평가한다. 이번 작업에서 두 모델을 직접 재학습한 것은 아니며, SFT 전후 개선 효과를 추정하는 실험도 아니다.

[SFT 기준선 평가 문서](SFT_BASELINE_EVALUATION.md)에 모델·데이터·prompt·집계·중복 점검·실측을 기록한다. Dolly 생성 평가와 SST-2·BoolQ의 accuracy·macro-F1 등 분류 평가를 구분한다. 이 공개 SFT baseline을 MiniLLM의 초기 `init` checkpoint와 같다고 가정하지 않는다. 최종 KD 공통 초기 상태와 최종 평가 집합의 선정은 별도 결정이다.

두 모델의 평가를 완료했다. Dolly ROUGE-L(0–100)은 XL **27.06 ± 0.48**, base **23.00 ± 0.41**이며 ±는 5개 생성 seed의 표본 표준편차다. SST-2 accuracy/macro-F1은 XL 76.49/76.45%, base 62.16/59.53%였다. BoolQ는 XL 62.53/42.98%, base 50.93/50.20%로 지표의 순위가 달랐다. 큰 모델·높은 accuracy 하나로 모든 과제의 지도 유익성을 확정하지 않는다. 전체 표·그림·클래스별 오류·중복 표시 제외 결과는 위 별도 문서와 실행 기록을 따른다.

[Dolly 데이터 구성 문서](DOLLY_DATA_OVERVIEW.md)에 원본 15,011행의 필드·과제 비중과 평가용 500행의 차이를 기록했다. 현재의 두 공개 패키지를 원본의 완전히 같은 행으로 구성한 최종 연구 분할로 취급하지 않는다.

### 7.5. Dolly 직접 SFT의 초기 결과 — 2026-10-16

GPT-2 base와 XL은 공통 Dolly 분할에서 두 LR 후보를 각각 3에포크 LoRA SFT했다. 두 모델 모두 BF16과 LR `2e-4`가 선택됐으며, 최적 checkpoint는 base 3에포크·XL 2에포크다. 이 절의 수치는 §7.4의 공개 SFT checkpoint 평가와 별개의 직접 학습 결과다. 전체 점수·조건·그림은 [직접 SFT 평가 결과](../results/dolly_sft_v1/REPORT.md)를 따른다.

Dolly 500문항·생성 seed 10에서 ROUGE-L은 base 14.68, XL 20.13이다. SST-2 accuracy/macro-F1은 base 65.94/63.95%, XL 62.27/56.27%이며, BoolQ는 base 50.35/49.40%, XL 62.43/42.49%다. XL의 BoolQ accuracy는 다수 클래스만 예측하는 기준선 62.16%와 가깝고, balanced accuracy는 51.14%, 음성 클래스 recall은 4.69%다. 이 prompt의 분류에서는 한쪽 답에 치우친 예측이 관측됐으므로 accuracy만으로 교사의 지도 유익성을 판단하지 않는다.

현재 두 모델의 생성 관측은 seed 하나의 결과이며 seed 간 변동과 통계적 유의성을 추정하지 않는다. 나머지 후보 평가와 실제 증류 전후의 대응 비교가 남아 있다. 학습 곡선은 에포크 종료 시 NLL을 표시하고, 최적 checkpoint 복원 후의 재평가를 마지막 에포크의 추가 관측으로 그리지 않는다.

**평가 역할의 확인:** [MiniLLM §3.1·3.3](https://arxiv.org/html/2306.08543v4)은 생성 benchmark를 주 평가로 삼고, SST-2·BoolQ는 LLaMA-7B 기반 calibration 분석에 사용한다. [CaRE-KD §4](https://arxiv.org/html/2609.36734v1)도 Dolly·Self-Instruct·Super-Natural Instructions·Vicuna의 생성 결과를 중심으로 보고한다. 우리 분류 평가 두 개는 보조 진단이며, Dolly에서 관측한 XL의 +5.45 ROUGE-L과 분류 과제별 결과를 분리해서 해석한다. 현재의 공통 3에포크 LoRA 실행은 MiniLLM의 10/20에포크 SFT 기준선 재현이 아니다. 다만 XL은 두 LR 모두 2에포크에서 검증 NLL이 가장 낮고 3에포크에 상승했으므로, 학습을 늘리면 생성·분류 성능 차이가 커진다고 단정할 근거도 없다. 2026-10-16 사용자 결정으로 [네 생성 benchmark 평가](GENERATION_EVALUATION.md)를 주 평가로 확장했다. 같은 10개 tokenizer에서 2,516문항을 그대로 사용하는 감사를 통과했으며 실제 완료 범위와 교사−학생 차이는 [주 평가 보고서](../results/generation_benchmarks_v1/REPORT.md)를 따른다.

### 7.6. 데이터 비교 준비의 실측

세 원자료의 전체 구조·길이를 확인했다. 검토용 포맷의 완전한 입력+응답이 GPT-2 native 1,024토큰을 넘은 비율은 Dolly 1.38%, UltraChat train_sft 59.63%, MetaMathQA 0.23%다. UltraChat은 대화 전체를 센 값이며 최종 제외율이 아니다. MetaMathQA는 원문 13,929종에서 나온 증강이므로 원문 group 분할이 필요하다. 세 데이터의 직접 SFT accuracy는 아직 측정하지 않았다. 전체 조건·hash·한계는 [비교 프로토콜](DATASET_SFT_COMPARISON.md)과 연결한 JSON을 따른다.

## 8. 설정값의 근거와 재현 기록

| 설정 | 필요한 근거 | 현재 상태 |
|---|---|---|
| 실제 teacher/student checkpoint·revision | 원문 학습 이력·공식 배포·데이터 분할·공통 초기화 | 기반 두 모델 revision 고정. 선정 데이터의 SFT·증류 초기 상태는 미정 |
| 데이터 revision·분할·사용 규모 | 원문·공식 자료·전처리와 중복 검사 | 미정 |
| KD·rollout·uncertainty·평가 온도 | 역할별 원문·코드 경로와 필요한 validation 선택 | 미정 |
| gate margin·scale·미분 | 원문 식·그림·공식 구현 대조 | 미정 |
| MC 수·dropout | 원문 역할·모델별 추정 신호·실제 자원 | 미정 |
| quantile·schedule·mask 집계 | 원문 정의·과거 통계·실제 갱신 경로 | 미정 |
| 학습률·batch·길이·학습량·정밀도 | 원문 조건·실행 시험·사전 선택 절차 | 미정 |
| checkpoint·주 평가 지표·반복 | 원문·과제 목적·변동과 가능한 자원 | 미정 |
| 무작위성 식별자 | 재현 가능한 기록. 성능을 보고 좋은 seed를 선택하지 않음 | 미정 |

원문 고정값과 새 조건의 최적값을 구분한다. 추가 설정 선택이 필요하면 기준·영역·예산·validation 절차를 먼저 정한다. 유한 후보에서 선택한 값을 전역 최적값으로 표현하지 않으며 test를 보고 설정을 다시 고르지 않는다. 온도 탐색 자체를 별도의 주 연구로 확장하지 않는다. 상세 원칙은 [계획서 §7–10](RESEARCH_PLAN.md)을 따른다.

## 9. 평가와 시각화 설계

### 9.1. 평가

생성 과제의 reference overlap을 보는 ROUGE-L을 원문 평가와 연결되는 후보로 검토한다. 사실성·정답성은 별도 채점 근거가 필요하며 창작·요약·QA를 하나의 단순 정확도 사건으로 합치지 않는다. 실제 응답 사례와 공통 정량 지표를 함께 읽는다. [MiniLLM §3](https://arxiv.org/html/2306.08543v4), [CaRE §4.2](https://arxiv.org/html/2609.36734v1).

entropy·KL·gate·gradient·BALD·수용률은 동작을 읽는 진단이다. 낮은 값·높은 수용률을 좋은 생성 품질이나 실제 정답의 대체 지표로 사용하지 않는다. 주 평가 지표·reference·채점 규칙·표본 수·반복 정의는 아직 미정이다. 외부 judge를 쓴다면 rubric·모델 버전·오류·비용을 먼저 확인한다.

### 9.2. 두 모델부터 시작하는 표시 원칙

초기에는 **GPT-2 XL과 GPT-2 base 두 모델만** 사용한다. 공통 입력의 원시 다음 토큰 분포부터 읽고, 선정 데이터의 SFT 후의 분포·학습 신호, gate, Revival을 순서대로 연결한다. 초기 기반 모델의 출력은 추론 파이프라인 점검 자료이며 증류 효과의 증거가 아니다. 전체 진행 순서는 [계획서 §5.2.3](RESEARCH_PLAN.md)을 따른다.

교사·학생 색상, 대응 패널의 축과 토큰 순서를 고정한다. 주요 토큰의 합집합을 표시하고 빠진 확률은 `기타 어휘`로 남기며, KL·entropy는 전체 어휘에서 계산한다. 표시할 토큰 수는 가독성을 위한 설정으로 기록하고 분포를 다시 정규화하지 않는다. 확률 차이는 0을 중심으로, heatmap은 공통 범례·범위·mask로 읽는다.

분포 비교·gate 곡선·Revival 흐름은 각각 한 질문을 설명하는 그림으로 나눈다. 모든 그림에 실제 입력·prefix 위치·checkpoint 단계·revision·정밀도·온도 변환·계산 범위를 남긴다. 원문 인용·수식 예시·기반 모델 추론 점검·선정 데이터의 SFT 후 실측을 구분하며 아직 실행하지 않은 단계의 값을 채우지 않는다.

### 9.3. 예정 자료

| 자료 ID | 그림·표 | 기록할 내용·해석 조건 |
|---|---|---|
| V1 | 전체 학습 흐름도 | 입력·분포·손실·gate·Revival·실제 갱신의 연결 |
| V2 | 같은 context의 FKL/RKL 분포 비교 | 토큰 확률·차이·학습 신호. 표시 토큰 외의 확률질량 보존 |
| V3 | Hard/Soft gate 곡선 | confidence 차이·실제 margin/scale·미분 조건 |
| V4 | 토큰별 heatmap | confidence·gate·KL·학습 신호와 유효 mask |
| V5 | 불확실성과 수용/거절 흐름 | teacher/student BALD·quantile·mask·실제 갱신 |
| V6 | 수용/거절된 자료의 분포 | category·길이·불확실성, 전체 처리 자료와의 비교 |
| V7 | 공통 validation·최종 평가 | 학습량·표본 수·변동과 처리 입력/실제 갱신 수 |
| V8 | 같은 질문의 생성 응답 표 | reference·교사·방식별 학생 응답, 길이·반복·과제 수행 |

이는 예정 자료 목록이며 실제 그림은 아직 생성하지 않았다. 좋은 예제만 선택하지 않고 선정 이유·실패·무변화를 함께 남긴다. 문헌 그림·수식 예시·실측을 구분하고 축·조건·반복·오차 정의를 표시한다.

## 10. 실험 결과 — 실행 후 작성

**현재 KD 학습·평가는 미실행이다. Dolly 직접 SFT의 초기 결과는 §7.5에 기록하며, 아래 KD 방법 비교의 개선·악화·동등성에 관한 결론은 없다.**

실행 후에는 실제 수행한 조건과 누락·실패 이유를 적고, FKL/RKL, Hard/Soft, 같은 손실의 Revival 유무, 결합 결과를 대응 표·그림으로 작성한다. 예상 방향에 맞지 않는 결과와 무변화도 기록한다. SeqKD·응답 증강은 실제 수행한 경우에만 추가 결과로 보고한다. 임의 예시 수치를 측정값처럼 채우지 않는다.

## 11. 논의와 해석의 한계

- confidence·BALD와 실제 정오·지도의 유익성을 구분한다. Revival의 거절 자체가 오류 판정은 아니다.
- Hard/Soft의 margin·혼합·미분 차이를 함께 읽고 연속성 하나의 원인 효과라고 단정하지 않는다.
- 갱신 거절에서 달라지는 입력 노출·실제 갱신 수와 optimizer 동작을 설명한다.
- 학습 loss 감소와 과제 수행 개선을 구분한다. 차이가 검출되지 않았다고 동등성을 선언하지 않는다.
- 실제 모델·분할·길이·평가·원문과의 변경 범위 안에서 결과를 해석한다.
- 작은 GPT-2 조합의 관측을 최신 LLM 전체의 일반적 효과로 확대하지 않는다. 기반 모델과 Dolly 준비 checkpoint를 구분한다.
- 원문의 조건부 이론·보고값과 우리 관측을 구분한다. 개별 구성요소와 전체 CaRE-KD의 결과를 섞지 않는다.

## 12. 결론과 후속 연구 — 결과에 따라 작성

현재 연구 질문·비교 구조·기준 데이터의 방향을 정했다. 모델 확보·동시 적재·짧은 입력 추론을 확인하고 시각화를 분포 → 손실·gate → Revival 순서로 정리했다. 세 후보의 전체 구조·길이 감사와 일회성 갱신을 완료했으며, 현재는 Dolly 분할과 검증 선택 규칙을 고정해 10개 모델의 직접 SFT·평가를 진행한다. 최신 완료 범위는 상단 진행 기록을 따른다. 실험 후 KL 방향·손실 적응·갱신 거절·결합에서 얻은 시사점과 남은 질문을 정리한다. 아직 특정 KD 방법의 우수성이나 새로운 효과를 결론내리지 않는다.

한국어 적용, 다른 모델·과제, 교사 응답 증강 등은 관측 결과와 자원에 근거해 후속 후보로 검토한다. 결론을 먼저 정해 실험 결과를 맞추지 않는다.

## 13. 참고문헌과 부록 기록

- Hinton, Vinyals, Dean (2015), [Distilling the Knowledge in a Neural Network, v1](https://arxiv.org/pdf/1503.02531v1).
- Gu, Dong, Wei, Huang (2024), [MiniLLM, v4](https://arxiv.org/html/2306.08543v4).
- Sengupta, Seth, Chakraborty (2026), [CaRE-KD, v1](https://arxiv.org/html/2609.36734v1).
- Gou 외 (2021), [Knowledge Distillation: A Survey, v7](https://arxiv.org/pdf/2006.05525v7).
- Kim, Rush (2016), [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/).
- 데이터·기반 모델·config의 출처는 §6의 직접 링크를 따른다. 원문 버전과 적재 점검용 두 기반 모델 revision은 고정했다. 세 후보 원자료 revision은 고정했고 데이터별 SFT checkpoint·분할·본 실험 코드의 동결은 남아 있다.

부록에는 실제 모델·데이터 revision, 분할·전처리, 값별 근거, 코드·환경 버전, 실행 로그, 추가 그래프·응답 사례를 남긴다. 현재 [확인 코드](../scripts/load_two_models.py)와 [적재·추론 실행 기록](../results/model_preflight.json)이 있다. KD 학습 로그·방법 비교 결과는 아직 없다. 공개 SFT 기준선 평가와 그림은 §7.4의 별도 기록을 따른다.
