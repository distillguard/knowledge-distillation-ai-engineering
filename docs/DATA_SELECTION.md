# 연구 데이터 선정 근거와 후보

갱신일: 2026-10-16, Asia/Seoul.

현재 결정은 **학습 Dolly 고정 → 같은 자료로 5계열·10모델 SFT → 생성·보조 평가 → 조합 선정 → KD 비교**다. 공통 학습 11,117건·개발 검증 975건·Dolly 선정 평가 500건을 사용한다. 외부 Self-Instruct·S-NI·Vicuna는 생성 평가용이며 학습에 포함하지 않는다. 상세 명세는 [현재 SFT 계획](DOLLY_SFT_MODEL_SELECTION.md)과 [생성 평가 계획](GENERATION_EVALUATION.md)을 따른다.

아래 후보 표와 원자료 조사는 데이터의 역할·제작 과정·품질·누수를 검토한 근거로 보관한다. UltraChat·MetaMathQA·한국어·WizardCoder는 현재 SFT의 추가 학습 자료가 아니다. 데이터별 SFT를 진행하던 과거 설계와 예비 갱신 기록은 [후속 비교 문서](DATASET_SFT_COMPARISON.md)에 분리했다.

## 1. 데이터의 역할부터 정한다

| 역할 | 사용하는 정보 | 선정·평가에서 구분할 것 |
|---|---|---|
| 교사/학생 SFT | 입력과 reference 응답 | 원문 초기화 단계인지, 추가 학습인지; reference의 품질 |
| 증류용 입력 | prompt와 필요한 context, 해당 학습 경로의 교사 분포·학생 rollout | reference 응답을 따라 배우는 것과 학생이 생성한 prefix에서 배우는 것 |
| development | 후보의 수행·오류·confidence, 설정과 checkpoint 선택 | 원인 진단·설정 선택에 사용한 자료임을 기록 |
| 최종 평가 | 선택을 동결한 후의 정답·reference·채점 규칙 | 선택용 자료와 분리하고 모델 간 같은 과제 조건 유지 |
| 능력 보존용 자료 | 해당 방법이 요구하는 별도 언어모델링 corpus | 원문 방법의 구성요소인지 확인; 제거·변경하면 그 차이를 기록 |

[설계] 한 데이터셋을 여러 역할에 사용할 수 있지만 record·원문 문제·대화 단위의 분할과 사용 시점을 명시한다. 최종 평가용으로 남긴 benchmark 문항·정답·파생 자료는 학습과 설정 선택에서 제외한다. 데이터에 `test`나 `validation`이라는 이름이 있다는 사실만으로 연구에서의 역할이 확정되지는 않는다.

## 2. 참고문헌과 연결되는 학습 후보

Dolly는 MiniLLM과 연결되는 기준 후보이고 UltraChat·MetaMathQA도 현재 선정 대상이다. 원문과의 연결, 정오 판정 가능성, 모델 문맥 길이와 로컬 실행 범위를 함께 확인한다. WizardCoder와 한국어는 이번 SFT 비교에 추가하지 않는다.

| 후보 | 확인한 자료·구조 | 문헌과 연결되는 연구 | 채택 전에 확인할 것 |
|---|---|---|---|
| [Databricks Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 영어, 직원 작성 instruction·context·response·category; 공개본 `train` 15,011행 | MiniLLM·CaRE의 일반 지시/생성, KL·prefix·confidence 분석 | 원문 분할·prompt·길이 필터, 과제별 reference의 평가 가능성. 사람 작성 응답도 오류·여러 타당한 답이 있을 수 있음 |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 합성 다중 턴 대화; prompt·prompt_id·messages, sft/gen 분할 | CaRE의 chat alignment 조건 | 어느 split·턴·역할을 학습할지, 대화 길이·응답 품질·같은 대화의 분할 간 중복 |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | `train` 395,000행; type·query·original_question·response | CaRE의 수학 추론; 정오가 정의되는 지도 효과 분석 | 원문 문제별 변형 그룹, 답 추출·풀이 품질, 평가 자료와의 중복·길이 |
| WizardCoder 학습 자료 | [CaRE §4·부록 D](https://arxiv.org/html/2609.36734v1)와 [WizardCoder 공식 프로젝트](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder)에서 연결 확인 | CaRE의 코드 생성 조건 | 해당 실험의 정확한 데이터 ID·revision·원자료·라이선스·split 미확인. 이름이 비슷한 공개 mirror로 대체하지 않음 |

MiniLLM v4는 Dolly를 사용하고, CaRE v1은 Dolly·UltraChat·WizardCoder·MetaMathQA를 과제별로 사용한다. 같은 데이터명만으로 전처리와 학습 초기 상태가 같다고 판단하지 않는다. [MiniLLM §3.1](https://arxiv.org/html/2306.08543v4), [CaRE §4·부록 D](https://arxiv.org/html/2609.36734v1).

## 3. 확인한 규모·분할·버전

아래 버전은 공식 카드/API와 대조하고 내려받은 원자료에 고정했다. **원문 실행 당시의 revision이나 새 연구의 채택 규모를 뜻하지 않는다.** 2026-10-15에 Dolly 전체, UltraChat SFT 두 split 전체, MetaMathQA 전체의 구조·길이를 직접 집계했다. 파일 hash와 집계는 [audit.json](../results/dataset_candidates/audit.json), 렌더링 조건과 해석은 [비교 프로토콜](DATASET_SFT_COMPARISON.md)을 따른다. 정답 품질·의미상 중복 감사까지 끝난 것은 아니다.

| 데이터 | 확인한 공개 split과 행 수 | 카드의 라이선스 표시 | 확인한 repository revision |
|---|---|---|---|
| Databricks Dolly-15K | train: 15,011 | CC BY-SA 3.0 | [bdd27f4…](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a) |
| UltraChat 200k | train_sft: 207,865; test_sft: 23,110; train_gen: 256,032; test_gen: 28,304 | MIT | [8049631…](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a) |
| MetaMathQA | train: 395,000 | MIT | [aa4f34d…](https://huggingface.co/datasets/meta-math/MetaMathQA/tree/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18) |

UltraChat의 전체 행 수 515,311을 SFT 학습량으로 적지 않는다. 데이터 카드는 `sft`를 supervised fine-tuning, `gen`을 generation ranking 용도로 구분한다. MetaMathQA 카드는 GSM8K·MATH **train**에서 증강했으며 test에서 증강하지 않았다고 밝힌다. 이는 제공자의 설명이고, 새 연구의 누수 검사가 끝났다는 뜻은 아니다. 원본과 파생 자료의 출처·이용 조건을 유지하며 카드의 표시만으로 모든 자료의 권리를 단일 라이선스로 덮지 않는다.

## 4. 기준 후보 Dolly-15K의 사용 조건 확인

[설계] MiniLLM의 생성 과제를 참고해 Dolly의 원문 분할·전처리를 검토한다. 원문 실험과 연결되고 category와 context를 보존해 응답을 해석할 수 있기 때문이다. 15,011행이라는 규모만으로 학습 가능성·충분한 통계 정밀도를 보장하지 않는다. Dolly도 다른 두 후보와 비교한 뒤 채택 여부를 정한다. 분할·길이 필터·규모는 원문 조건과 실행 시험에 근거해 확정한다.

[원문] MiniLLM §3.1은 context 길이를 초과하는 자료를 필터링한 뒤 validation 1K, test 0.5K를 무작위 분리하고 약 12.5K로 학습했다고 기술한다. 이 숫자는 원문 protocol이며 우리의 표본 수·분할 비율로 자동 채택하지 않는다. 능력 보존용 D_PT에는 GPT-2 계열의 OpenWebText와 다른 계열의 RoBERTa corpus를 사용한다. [고정한 원문](https://arxiv.org/html/2306.08543v4).

[공식 구현] 확인한 LMOps commit은 `48fca1acc61ffa20ebb7c31c74a1d8e33499b476`이다.

- [README §2, line 43–67](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/README.md#L43)은 MiniLLM의 원자료·processed data와 모델 계열별 전처리 경로를 연결한다.
- [전처리 코드 line 71–78](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/tools/process_data_dolly.py#L71)은 `raw.jsonl` 앞 `dev_num`행을 valid로, 나머지를 train으로 나눈다. 이 함수에서 무작위로 섞는 단계는 확인되지 않았다. 이전 단계의 원자료 순서 생성은 추가 확인한다.
- [코드 line 52–58](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/tools/process_data_dolly.py#L52)의 encoder는 tokenizer와 prompt template를 사용하고 prompt 길이를 검사한다. 위 분할 뒤 각 split에 encoder를 적용한다. 원문의 전체 context 필터·분할 설명과 동일한 실행인지 추가 대조한다.
- [GPT-2 전처리 script](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/scripts/gpt2/tools/process_data_dolly.sh)는 prompt-only 경로와 prompt+response 경로를 별도로 만든다. 둘을 같은 reference 지도 방식으로 취급하지 않는다.

[관측] [MiniLLM/dolly](https://huggingface.co/datasets/MiniLLM/dolly/tree/d131636138be0ff3fd0901c12c1739844d337279)에는 `raw.jsonl`과 `valid.jsonl`이 있다. 최초 메타데이터 조회 당시 기본 Viewer는 `text` 컬럼의 `validation` 500행만 표시했다. 이후 2026-10-15에 파일을 직접 읽어 raw 12,435행·valid 500행을 확인했고, [공식 평가 코드](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/evaluate_main.py)가 valid를 test 역할로 읽는 것을 대조했다. 이를 공개 SFT 기준선 평가에 사용하며 논문의 validation 1K나 새 학습 자료로 해석하지 않는다. 해당 패키지의 카드 표시는 Apache 2.0이며 원본 Dolly의 CC BY-SA 3.0 표시와 다르므로, 이를 원본 이용 조건의 변경 근거로 취급하지 않는다.

[관측: SFT 기준선 평가용 점검] 공개 500문항의 prompt는 두 SFT 모델의 공통 tokenizer에서 모두 256토큰 이하였다. 공백·대소문자를 정규화한 instruction+input을 raw와 대조했을 때 평가 10행이 9종의 입력과 일치했다. [SFT 평가 문서](SFT_BASELINE_EVALUATION.md)에 전체 500행과 이 10행을 제외한 490행의 평가를 함께 기록한다. 파일 hash·일치 행 번호는 [protocol](../results/sft_baselines/protocol.json)에 고정한다. 완전한 체크포인트 학습 이력·의미상 중복 감사를 마쳤다고 표시하지 않는다. 이미 기준선 확인에 사용한 평가 집합을 앞으로의 독립 최종 test로 자동 취급하지 않는다.

일반 QA·요약·창작을 같은 정확도 사건으로 합치지 않는다. context를 주는 과제에서 context를 제거하면 문제 자체가 달라진다. reference와 표현이 다르다는 이유만으로 생성 답을 오답으로 분류하지 않는다. [Dolly 카드의 과제·품질 설명](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

### 4.1. 손실 적응·학습 갱신 거절 시각화를 위한 공통 자료

[설계] 최종 선정 데이터의 record ID·입력·원문 속성과 train/development/test 분할을 네 비교 조건에서 공유한다. 학습 응답·prefix의 출처와 reference를 쓰는 방식을 공통으로 정하고, 같은 context에서 FKL/RKL과 gate만 비교할 수 있도록 기록한다. SFT·SeqKD·MiniLLM 전체 학습 파이프라인의 차이를 동시에 바꾸는 비교로 확대하지 않는다.

시각화용 development 예제는 category·길이·context 유무와 선정 이유를 기록한다. 잘 나온 응답만 고르지 않으며 최종 평가 문항은 설정 선택에서 제외한다. 확률·confidence·gate·학습 신호는 같은 prompt·prefix·어휘 좌표에서 비교하고 자유 생성 응답은 별도로 관찰한다. 실제 student rollout은 조건에 따라 달라질 수 있으므로 입력이 동일한 고정 진단과 구분한다.

Revival 비교에는 같은 입력의 teacher/student BALD, 과거 quantile·mask, 수용/거절된 record·집계 단위와 실제 갱신 여부를 연결한다. 수용된 자료만의 결과로 전체 성능을 보고하지 않고 처리한 자료·거절한 자료의 분포와 전체 평가를 함께 남긴다. 입력 노출 수와 실제 갱신 수를 구분한다. 거절을 정답 오류의 자동 판정으로 해석하지 않는다.

Dolly-15K는 일반 instruction 자료의 기준 후보다. MiniLLM 원문의 D_PT·teacher-mixed sampling은 KL 배경을 이해하는 참고이며 이번 비교에 자동 승계하지 않는다. 학습 경로에 별도 손실·자료를 채택하면 이유와 조건별 적용을 기록한다. 원문 전체 재현과 목적함수 비교를 구분한다.

### 4.2. 원본 파일의 구성 확인

2026-10-15에 Databricks 원본 15,011행 파일도 내려받아 네 필드·8개 과제·context 유무를 직접 집계했다. [Dolly 데이터 구성 문서](DOLLY_DATA_OVERVIEW.md)와 [파일 hash·구성 집계](../results/sft_baselines/dolly_composition.json)를 따른다. 원본의 context 보유 비중은 4,467/15,011(29.76%)이며 평가용 500행에서는 91/500(18.2%)다.

공백·대소문자 정규화 후 평가 500개 instruction은 원본에서 모두 찾았으나 instruction+context가 같은 것은 448개였고 52개는 context가 달랐다. 변환·버전 경로의 완전한 동일성은 아직 미확인이다. 두 모델은 같은 공개 prompt로 평가했다. 단순한 행 수 차이로 저자가 실제 사용한 train/development/test 크기나 필터 제외 수를 역산하지 않는다. 원본 전체를 읽어 구성 수를 확인한 것과 전처리·의미상 중복·정답 품질 감사를 마친 것은 다르다.

## 5. 질문에 따라 학습·평가 조합을 좁힌다

| 적용하려는 인사이트 | 검토할 학습 입력 | 평가·진단 자료 후보 | 채택 조건 |
|---|---|---|---|
| 생성에서 KL·prefix·길이가 미치는 영향 | Dolly의 instruction과 필요한 context | 원문의 held-out DollyEval·SelfInst·S-NI 등 생성 평가; 길이별 결과·reference overlap과 별도 품질 진단 | 원문 분할·prompt·표본 생성 경로와 길이별 분석이 확인됨 |
| 교사의 지도가 학생의 정답을 손상시키는 조건 | Dolly 중 정오를 신뢰성 있게 판정할 수 있는 과제, 또는 수학 과제로 정의 변경 | 같은 development 문제에서 교사/학생 정오와 confidence, 학습 후 학생 변화 | 채점 사건이 정의되고 실제 오류 조합이 관측됨. 단순 문자열 일치·ROUGE로 환각 판정하지 않음 |
| 수학에서의 지도 유익성 | MetaMathQA; 원문과 연결할 모델·prompt 확인 | GSM8K·CollegeMath의 잠근 평가 자료와 채점 규칙; 원문 문제별 development | 같은 원문에서 파생한 증강 질문을 group으로 묶고 평가 중복을 검사함 |
| 다중 턴 대화에서의 지도 | UltraChat의 적합한 split·대화 턴 | 원문 chat 평가의 조건 또는 별도 채점 근거 | 대화 context·assistant 학습 mask·reference/judge·긴 입력 예산이 준비됨 |
| 코드에서의 지도 | 정확한 WizardCoder 학습 자료 확인 후 검토 | HumanEval·MBPP의 해당 평가 split·테스트 버전 | 원자료와 전처리·평가 문항 중복·실행 채점 환경이 확인됨 |

아래 일반 생성·수학·대화·코드의 조합은 역할별 후보 조사다. 현재 직접 학습은 Dolly만 사용한다. 오류 보호를 분석하려면 신뢰할 수 있는 정오 판정과 오류 조합을 확인해야 한다. 코드 과제는 후속 조사다. 평가 집합의 출처·버전·원자료는 [벤치마크 검토](MODEL_BENCHMARK_REVIEW.md)와 MiniLLM 논문/공식 코드에서 확인한다. 학습 후보 이름만으로 최종 평가 집합을 자동 선택하지 않는다.

**후속 연구 메모:** 한국어 적용을 시작하면 [계획서 §5.3](RESEARCH_PLAN.md)의 한국어 원생 과제·번역/합성 자료와 채점 근거를 비교한다. [KIT-19](https://arxiv.org/abs/2403.16444v1)나 [KoAlpaca](https://github.com/beomi/KoAlpaca)는 후속 조사 자료이며 현재 학습 후보로 채택하지 않았다. [KMMLU-Redux/Pro](https://arxiv.org/abs/2507.08924v2)는 전문 지식 평가, [KoSimpleQA](https://arxiv.org/abs/2510.18368v1)는 짧은 사실 질문 평가로 역할이 다르다. 최종 평가용 문항을 학습 입력으로 사용하지 않는다. 번역한 Dolly·MetaMathQA 등을 쓸 경우 원문과의 변경, 번역 품질·문제 의미·동일 원문 파생 group·이용 조건을 확인한다.

## 6. 분할·품질·누수를 확인하는 순서

[설계] 전처리 전에 원자료 revision·파일 hash와 record 식별자를 남긴다. 다음 확인 후 변경 이유·제외 수·자료 분포를 기록한다.

1. **같은 문제/대화의 그룹:** MetaMathQA는 `original_question`과 원자료 대응, UltraChat은 대화·prompt_id, Dolly는 같은/유사 instruction·context를 확인한다. 새 분할에서는 관련 변형이 train과 development/test에 나뉘지 않도록 group을 보존한다.
2. **평가 자료와의 겹침:** 정확한 중복뿐 아니라 재표현·답 누출·같은 원문 문제의 변형을 검토한다. 새 학습 자료의 누수 검사와 이미 학습된 모델의 사전학습 오염 여부는 별도 상태다.
3. **reference의 신뢰성:** 인간 작성·합성·수학 풀이·코드의 출처를 구분하고 채점 가능한 정답과 응답 품질을 확인한다. 합성 응답이나 사람이 쓴 문장을 무검증 정답으로 취급하지 않는다.
4. **선정 모델의 tokenizer·template로 길이 확인:** prompt·reference·전체 context·생성 예산의 토큰 분포와 긴 입력을 확인한다. 문자 수를 토큰 수로 대신하지 않는다.
5. **비교 집합 유지:** 모델별 tokenizer·길이 필터로 남는 자료가 다르면 공통 비교 집합 또는 다른 조건임을 명시한다. 원문 재현과 공통 집합으로 변경한 비교를 구분한다.
6. **분할 동결:** 필터·group·split·역할과 근거를 고정하고 선택에 사용한 문항을 기록한다. test 결과를 보고 데이터·길이·T를 다시 정한 뒤 같은 test를 독립 최종 근거로 사용하지 않는다.

## 7. 로컬 실행 범위에서 데이터 양을 정한다

RTX 5060 Ti 16GB급·RAM 약 32GB의 장비와 실행 환경은 [공통 실행 지침](EXPERIMENT_GUIDE.md)에 기록했다. **행 수·압축 파일 크기만으로 GPU 학습 메모리를 판정하지 않는다.** tokenizer·template·batch·학생 학습 경로·교사 forward·rollout/MC를 포함한 예비 실행에서 메모리와 시간을 측정한다.

전체 어휘의 teacher 분포를 저장하는 방법이면 저장 토큰 수×어휘 크기×원소당 bytes의 원시 배열 용량과 부가 파일을 확인한다. 고정 teacher 응답 cache로 학생 생성 prefix가 필요한 방법을 대체하면 원문의 학습 조건이 바뀐다. 능력 보존 corpus를 생략하면 그 영향과 방법 변경도 기록한다.

subset이 필요하면 연구에서 관측해야 하는 과제·난도·길이·오류 조합을 보존하는 표집 근거와 예비 변동을 기록한다. 성능이 잘 나온 문항만 남기지 않는다. 최종 수량·split 비율·최대 길이는 필요한 관측 정밀도와 실제 실행 예산에 근거해 정하며, 임의의 “먼저 N개”·고정 비율·토큰 상한을 새 기본값으로 넣지 않는다.

## 8. 현재 결정과 남은 확인

Dolly의 현재 공통 분할·지시문 중복 제외·전체 응답 보존·mask·최초 네 tokenizer 길이 필터는 동결했다. 추가 모델도 같은 원본 ID를 사용한다. 생성 평가 네 자료는 같은 문항·전체 reference를 보존하고 tokenizer별 native context를 감사했다. 이 검사가 의미상 중복과 모든 사전학습 오염까지 검증한 것은 아니다.

현재 남은 일은 모델별 SFT와 생성·보조 평가, 보완 지표·오류 검토 기준의 확정과 채점, 후보 조합의 실제 KD 경로 확인과 선정 근거다. 현재 학습 데이터를 세 후보 중에서 다시 고르는 절차는 수행하지 않는다. 본 KD의 학습 prefix·보존 corpus·최종 평가 역할은 별도 미정 사항이다.
