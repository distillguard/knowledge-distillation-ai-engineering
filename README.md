# 지식증류 · AI Engineering

<!-- SFT_PROGRESS:START -->

현재 예비 비교 범위는 **4개 계열·8개 모델**이며, SFT 선택 완료 **6/8**, Dolly 예비 생성 완료 **5/8**, 기존 Dolly·분류 보조 평가 완료 **2/8**이다. 외부 3종·새 보조 분류 평가는 후보 축소 후 수행한다. 생성 평가는 공통 seed 10 한 개를 사용한다. 모델별 상태·선택 LR·검증 NLL·실제 정밀도는 [SFT 진행 기록](docs/SFT_PROGRESS.md), 교사·학생의 생성 성능 차이는 [생성 주 평가](results/generation_benchmarks_v1/REPORT.md), 분류 편향 등은 [기존 Dolly·분류 보조 평가](results/dolly_sft_v1/REPORT.md)를 따른다. 최종 조합은 아직 확정하지 않았다.

<!-- SFT_PROGRESS:END -->

**큰 교사 모델의 예측을 작은 학생 모델이 배우는 지식증류에서, 학습 방식에 따라 학생의 응답과 성능이 어떻게 달라지는지 탐구한다.**

현재는 **Dolly를 공통 데이터로 고정하고, GPT-2·OPT·OpenLLaMA·Gemma-2의 교사·학생 후보 총 8개 모델을 각각 SFT해 비교하는 단계**다. 모델별 기준선을 확보한 뒤 KL 방향·confidence gate·Revival에 따른 변화를 분석한다. 주요 산출물은 확률분포·학습 신호·응답 변화를 설명하는 시각화와 평가 결과다.

## 실험 흐름

**범위 축소 (2026-10-16):** Qwen 두 모델을 제외하고 4계열·8모델의 예비 비교를 진행한다. 예비 생성 평가는 Dolly 500문항부터 수행하며 외부 3종·새 보조 분류 평가는 후보 축소 후 진행한다. 기존 결과는 보존한다. **예비 비교 → 유망 후보의 원문 기반 SFT 탐색·외부 평가 → 최종 한 조합 → 학생 초기화·본 증류 비교**로 진행한다. [원문 설정과 적용 단계](docs/DOLLY_SFT_MODEL_SELECTION.md#minillm-원문-절차와-적용-범위-검토)를 따른다.

**Dolly SFT → 모델별 체크포인트 선택 → Dolly 예비 생성 평가 → 유망 후보의 설정 탐색·외부 평가 → 교사·학생 조합 선정 → 증류 방식 비교**

1. **독립적인 SFT:** 4계열·8개 모델을 같은 Dolly 기준 응답으로 LoRA 학습한다. 학습률 후보마다 같은 공개 revision에서 시작하며 이 단계에는 KD를 적용하지 않는다.
2. **체크포인트 선택:** 각 모델 내부에서 두 LR·epoch별 개발 NLL로 adapter를 선택한다. 다른 tokenizer의 NLL로 모델 순위를 매기지 않는다.
3. **선정 평가:** 우선 Dolly 500문항을 비교하고 외부 3종·새 보조 진단은 유망 후보에 수행한다. 보완 지표·응답 검토·증류 실행 점검 후 한 조합과 선정 근거를 확정한다.
4. **본 증류 비교:** 선정 조합에서 FKL/RKL·Hard/Soft gate·Revival 유무와 결합을 대조한다. 같은 학생의 SFT 기준선 대비 분포·학습 신호·응답 변화를 시각화한다.

현재는 SFT와 선정 평가를 진행 중이다. 원문 사용값과 로컬 검증 선택값을 구분하고, 효율성 순위나 근거 없는 최적 설정을 연구 결론으로 삼지 않는다.

## 모델·데이터 구성

팀원은 비양자화 SFT를 별도로 진행한다. 현재 컴퓨터는 아래 BF16/NF4 정책으로 계속하며, 동일 모델의 양자화·비양자화 결과를 확보하면 [공통 대조 조건](docs/EXPERIMENT_GUIDE.md#양자화-조건의-역할-분담)에 따라 비교한다.

| 계열 | 교사 후보 | 학생 후보 | 현재 확인 범위 |
|---|---|---|---|
| **GPT-2** | [GPT-2 XL](https://huggingface.co/openai-community/gpt2-xl), 약 1.56B | [GPT-2 base](https://huggingface.co/openai-community/gpt2), 약 124M | 둘 다 BF16 가중치, 양자화 없음, LoRA 학습 |
| **OPT** | [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b) | [OPT-125M](https://huggingface.co/facebook/opt-125m) | BF16 먼저 점검, 전용 VRAM에 맞지 않으면 NF4 |
| **OpenLLaMA** | [OpenLLaMA-7B](https://huggingface.co/openlm-research/open_llama_7b) | [OpenLLaMA-3B](https://huggingface.co/openlm-research/open_llama_3b) | BF16 먼저 점검, 전용 VRAM에 맞지 않으면 NF4 |
| **Gemma-2** | [Gemma-2-9B-IT](https://huggingface.co/google/gemma-2-9b-it) | [Gemma-2-2B-IT](https://huggingface.co/google/gemma-2-2b-it) | 9B는 BF16 가중치만으로 VRAM 초과해 NF4, 2B는 실제 점검으로 확정 |

LoRA는 8개 모델에 공통 적용한다. 추가한 OPT·OpenLLaMA·Gemma 조합은 [CaRE의 Dolly 실험](https://arxiv.org/html/2609.36734v1)에서 가져왔다. 추가 모델은 BF16 가중치 크기와 실제 적재·갱신·저장 점검의 전용 VRAM 사용량을 근거로 양자화 여부를 결정한다. GPT-2·OPT·OpenLLaMA는 pretrained, Gemma-IT는 post-trained 모델이므로 시작점·크기·tokenizer·정밀도 차이를 포함한 후보 조합 비교로 해석한다.

공통 데이터는 [Databricks Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)다. 사람이 작성한 영어 지시·응답 **15,011건**으로, 질의응답·분류·정보 추출·요약·창작 등 8개 과제를 포함한다.

- `instruction`: 수행할 지시나 질문
- `context`: 응답에 필요한 참고 본문. 없는 예제도 있음
- `response`: SFT에서 학습할 기준 응답
- `category`: 과제 유형

Dolly를 고정해 모델·증류 방식의 차이에 집중한다. 실제 입력은 MiniLLM이 공개한 Dolly 분할이다. 지시문 중복과 네 tokenizer의 길이를 함께 확인한 공통 분할은 **학습 11,117건·검증 975건·선정 평가 500건**이다. 응답을 잘라 맞추지 않고 길이 초과 예제를 공통 제외한다. UltraChat·MetaMathQA는 확장 후보로 보관한다. 상세 근거·제외 수·설정 선택은 [현재 SFT 비교 계획](docs/DOLLY_SFT_MODEL_SELECTION.md)에 기록한다.

## 평가 기준

| 평가 대상 | 확인할 지표와 자료 |
|---|---|
| **생성 벤치마크의 교사·학생 차이 — 주 평가** | 예비 Dolly 500문항. 후보 후속은 Self-Instruct 242·S-NI (11_) 1,694·Vicuna 80문항. ROUGE-1/2/L·동일 문항의 차이와 오류. 선정 단계는 공통 생성 seed 10 한 개 |
| **분류 편향·확률 품질 — 보조 진단** | SST-2·BoolQ의 accuracy·macro-F1·Brier·클래스별 오류. 현재 Brier와 논문의 ECE를 구분 |
| **증류 전후의 학생 변화** | 학생의 SFT 기준선 대비 수행 변화, 교사·학생의 확률분포, gate·불확실성·갱신 수용/거절 |

자유 생성의 ROUGE를 문항 정답률로 해석하지 않는다. Revival의 지도 유익성을 분석할 때는 신뢰할 수 있는 정오 판정 기준을 별도로 마련한다. 학습·설정 선택용 검증·최종 평가의 역할을 구분하고, 이미 모델 선정에 사용한 점수를 독립적인 최종 증거로 부르지 않는다.

예비 생성 평가는 모델당 Dolly 500문항이며 외부 3종까지 적용한 후보는 총 2,516문항이다. 기존 Dolly 응답은 검증해 재사용한다. 자료 revision·다중 정답·전처리·실행·완료 기준은 [생성 평가 계획](docs/GENERATION_EVALUATION.md)을 따른다. 단일 생성 seed에서는 seed 간 표준편차와 오차막대를 제시하지 않는다. 최종 선정 모델 본 실험의 생성 seed는 10/20/30/40/50이며 학습 seed 반복과 구분한다.

검증 NLL은 각 모델의 SFT 체크포인트를 고르는 기준이며 다른 계열의 순위표가 아니다. 현재 ROUGE-L은 1차 후보 비교이고 최종 조합은 미정이다. METEOR·BERTScore는 보완 지표로 검토했으나 아직 채점하지 않았다. 원본 대비 SFT 전후의 생성 성능 향상도 아직 측정하지 않았다.

## 핵심 개념

| 개념 | 역할 |
|---|---|
| **KL** | 교사와 학생의 확률분포 차이를 측정하는 학습 기준. Forward KL은 교사가 지지하는 후보를 학생이 놓치는 데, Reverse KL은 학생이 고르는 후보를 교사가 지지하지 않는 데 민감하다. |
| **Confidence gate** | 다음 토큰 확률이 얼마나 집중되는지를 나타내는 confidence를 비교해 KL 방식을 조절한다. Hard gate는 하나를 선택하고 Soft gate는 두 KL을 연속적인 비중으로 섞는다. |
| **Revival** | 모델 일부를 무작위로 끄며 여러 번 예측해 불확실성을 추정하고, 각 모델의 과거 기준과 비교한다. 교사가 상대적으로 불확실하고 학생이 상대적으로 안정적인 조건에서 해당 증류 갱신을 억제한다. |

높은 confidence가 정답을 보장하지는 않는다. 이번 연구에서는 각 장치의 동작과 실제 응답·성능 변화를 함께 확인한다. Hinton은 지식증류의 기초, MiniLLM은 KL 방향, CaRE-KD는 gate와 Revival의 근거로 사용한다.

## 진행 현황과 남은 단계

자동 실행은 모델마다 **예비 SFT → 문서 → Dolly 500 생성 → 생성 보고서·문서 → 다음 모델** 순서다. 최신 완료 수는 문서 상단과 [SFT 진행 기록](docs/SFT_PROGRESS.md)을 따른다. 일부 LR·일부 benchmark 완료를 모델 전체 완료로 표시하지 않는다.

SFT 이후 조합 선정까지 남은 작업은 Dolly 예비 비교와 후보군 축소, 해당 후보의 원문 기반 SFT 탐색·외부 생성 평가·보조 진단, 보완 지표의 명세·채점, 모델×benchmark×지표 비교와 오류 검토, 후보 조합의 실제 증류 연산·메모리 점검, 선정 이유와 한계의 기록이다. 세부 완료 기준은 [생성 평가 계획](docs/GENERATION_EVALUATION.md#조합-선정까지의-단계와-완료-기준)에 있다.

직접 SFT 결과와 저자가 공개한 SFT checkpoint의 과거 평가를 분리한다. 원본 대비 SFT 생성 향상 폭·METEOR·BERTScore·본 KD 결과는 아직 없다. 과거 데이터 후보 조사와 한국어 적용은 후속 자료로 보관한다.

## 저장소 구성

```text
README.md                 연구 개요와 문서 진입점
docs/                     공통 실행 지침·연구계획·보고서·데이터·진행 기록
papers/                   참고논문별 한국어 분석
scripts/                  준비·점검·학습·평가·문서 갱신
tests/                    실행 순서와 중복 실행 차단 검사
results/                  검증된 protocol·요약 JSON·CSV·그림
.vscode/tasks.json        모델 하나 또는 전체의 공통 실행 작업
requirements-sft.txt      현재 SFT·평가 환경의 직접 의존성 버전
tmp/                      로컬 모델·자료·checkpoint·응답·로그, Git 제외
.sources/                 내려받은 원문 자료, Git 제외
```

모델별 명령·필수 결과물·완료 판정은 [공통 실행 지침](docs/EXPERIMENT_GUIDE.md#모델별-필수-결과물과-완료-판정)을 따른다. 현재 학습의 checkpoint와 protocol 경로는 유지한다.

## 관련 문서

- [공통 파라미터·장비별 실행 지침](docs/EXPERIMENT_GUIDE.md) · [VSCode 실행 작업](.vscode/tasks.json)
- [연구계획](docs/RESEARCH_PLAN.md) · [연구보고서 초안](docs/RESEARCH_REPORT.md)
- [현재 Dolly 모델 SFT 비교 계획](docs/DOLLY_SFT_MODEL_SELECTION.md) · [모델별 진행](docs/SFT_PROGRESS.md) · [실행 명세와 점검 결과](results/dolly_sft_v1)
- [데이터 선정 조사](docs/DATA_SELECTION.md) · [이전 데이터 후보별 SFT 비교 설계](docs/DATASET_SFT_COMPARISON.md)
- [공개 SFT 모델 평가 결과](docs/SFT_BASELINE_EVALUATION.md) · [Dolly 구성](docs/DOLLY_DATA_OVERVIEW.md)
- 논문 해설: [Hinton](papers/01_hinton_2015_ko.md) · [MiniLLM](papers/02_minillm_2024_ko.md) · [CaRE-KD](papers/03_care_kd_2026_ko.md)

연결된 계획·보고서의 과거 검토에는 이전의 데이터 3종 × 모델 2종 설계가 남아 있다. 현재 실행 범위는 **Dolly 고정·4개 계열의 8개 모델 LoRA SFT 비교**다. 진행 상태는 결과 폴더와 모델별 진행 기록을 따른다.
