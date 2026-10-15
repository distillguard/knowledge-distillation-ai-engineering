# 모델 선정을 위한 벤치마크 검토

공개 벤치마크 조사 기준일: 2026-10-14, Asia/Seoul. 모델 추천·config 확인 갱신일: 2026-10-15.

상태: **두 기반 모델의 적재·추론을 확인하고, 공개 SFT 완료 XL/base의 로컬 benchmark 평가를 추가했다. 직접 학습·KD 비교는 미실행.** 실측과 지표별 해석은 [SFT 기준선 평가](SFT_BASELINE_EVALUATION.md), 선정 원칙은 [연구계획서](RESEARCH_PLAN.md) §6을 따른다. 연구의 중심은 문헌 인사이트의 적용·검증이며, 메모리·시간은 실행 제약이다.

2026-10-15에 목표를 **생성형 언어모델의 지식증류 원리와 학생 모델의 응답 변화 탐구**로 정했다. Hinton·MiniLLM·CaRE를 근거로 KL·Hard/Soft의 손실 적응과 Revival의 갱신 거절을 비교하며, 모델 선정에서는 MiniLLM의 원래 모델·초기 학습·과제·언어·평가 조건을 우선 검토한다. 전체 재현은 필수 목표가 아니며 실제 실험 범위와 로컬 실행 가능성을 확인해 선정한다. 현재는 Dolly·UltraChat·MetaMathQA를 같은 기반 모델에서 각각 SFT해 비교한다. 선정한 데이터의 공통 조건에서 증류 방식별 학습 신호·분포·생성 응답을 시각화한다. 아래 공개 점수는 참고 근거다. CaRE의 gate·Revival과 결합은 현재 비교에 포함하고 한국어 적용은 후속 탐구 자료로 둔다.

## 1. 벤치마크를 선정에 사용하는 순서

[설계] 여러 벤치마크에서 모델의 능력 분포를 확인하고, 선택한 연구 질문에 필요한 관측을 제공하는지 판단한다. 공개 종합 순위만으로 교사·학생을 정하지 않는다.

| 역할 | 확인할 것 | 선정에 사용하는 범위 |
|---|---|---|
| 공개 성능 조사 | 공식 model card·기술보고서·평가 원자료의 과제별 결과 | 후보를 좁히는 사전 근거. 조건이 다른 숫자를 하나의 순위로 합치지 않음 |
| 우리 과제의 기준선 확인 | 같은 development 자료·평가 조건에서 교사와 학생의 수행·오류 | 지도의 유익성과 연구 현상이 실제로 관측되는지 판단 |
| 기전 분석 | confidence·정오·학습 전후 변화, prefix·길이의 영향 | 문헌 인사이트를 설명하는 자료. benchmark 총점으로 대체하지 않음 |
| 최종 평가 | 설정·모델 선택 후 잠근 평가 자료에서 얻은 결과 | 선택에 사용한 자료와 역할을 구분해 주장 범위를 보고 |

공개 test 점수를 보고 모델을 고르면 그 점수는 이미 선정 정보다. 같은 benchmark의 후속 결과를 모델 선정과 완전히 독립인 증거로 부르지 않는다. 새 연구의 학습·설정 선택·최종 평가 역할을 기록하고, 선택에 쓰지 않은 평가 자료나 외부 과제의 필요성을 검토한다. [선택 편향의 근거: Cawley·Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html).

## 2. 먼저 참고문헌의 평가와 연결한다

| 참고문헌 | 원문에서 확인한 평가 | 현재 모델 선정에 주는 기준 |
|---|---|---|
| [Hinton v1, §3·4·6](https://arxiv.org/pdf/1503.02531v1) | MNIST 오류 수, 음성 frame accuracy 등 과제별 평가 | soft target·온도의 신호를 이해하는 근거. 이 점수로 현재 생성 LLM을 순위화하지 않음 |
| [MiniLLM v4, §3.1·3.3·Appendix B](https://arxiv.org/html/2306.08543v4) | DollyEval, SelfInst, VicunaEval, S-NI, U-NI의 ROUGE-L; 앞 세 집합의 GPT-4 feedback; 별도 human evaluation·exposure bias·calibration 분석 | 생성과 prefix의 영향이 질문이면 실제 응답 생성과 길이·동작 분석이 필요. 객관식 지식 점수만으로 대체하지 않음 |
| [CaRE v1, §2·3.1·4.3](https://arxiv.org/html/2609.36734v1) | gate·Revival, Hard/Soft ablation과 같은 손실의 Revival 유무 | 두 장치가 현재 근거. 전체 benchmark·모든 calibration 분석의 재현 범위는 별도로 정함 |

학습 데이터와 benchmark 이름이 겹치면 원문의 held-out split·중복 제거·평가 prompt를 확인한다. Dolly 학습과 DollyEval 평가는 같은 이름만으로 독립성이 확보되지 않는다. 원문 재현을 선택한다면 원문 설정을 따르고, benchmark나 지표를 추가·변경하면 그 목적과 차이를 기록한다.

[데이터 선정 문서](DATA_SELECTION.md)에서 학습 후보·분할·전처리와 평가 자료의 연결을 함께 검토한다. 모델의 공개 benchmark 점수와 새 학습 데이터의 적합성은 별도 근거로 남긴다.

### 2.1. 현재 연구 목적의 모델 추천

**초기 공통 비교는 GPT-2 XL → GPT-2 base로 수행한다. 세 데이터의 SFT 수행·오류 조합을 확인해 최종 선정한다.** MiniLLM §3.1·Table 1과 CaRE §4에 공통으로 연결되는 조합이며, 같은 모델에서 KL·gate·Revival을 여러 조건으로 비교·계측할 출발점으로 추천한다. [MiniLLM 원문](https://arxiv.org/html/2306.08543v4), [CaRE 원문](https://arxiv.org/html/2609.36734v1).

| 우선순위 | 기반 모델 조합 | 근거·현재 상태 |
|---|---|---|
| 초기 사용 | [openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl) → [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | 두 문헌과 공통으로 연결. 기반 revision 고정·동시 적재·짧은 입력 추론 확인. 새 데이터별 SFT checkpoint·장기 학습은 미검증 |
| 대안 | [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b) → [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) | CaRE의 원문 조합. 더 큰 교사의 메모리·MC forward 부담 확인 필요 |
| 이후 확장 | 같은 GPT-2 XL → [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) | 초기 결과 이후 학생 크기 차이를 볼 후보. 초기 필수 실험은 아님 |

논문은 base 학생을 120M급으로 표시하며 위 GPT-2 모델 카드는 124M으로 안내한다. 실제 선정은 모델 ID·revision을 기준으로 기록한다. 상세 추천과 해석 범위는 [연구보고서 §6](RESEARCH_REPORT.md)을 따른다.

2026-10-15에 확인한 GPT-2 XL/base config에는 vocab_size=50,257과 attention·embedding·residual dropout 설정이 있다. 이후 고정한 기반 revision의 tokenizer backend·어휘 ID 전체 일치와 공통 입력 forward를 확인했다. MC-BALD의 유효성이나 최적 dropout을 검증한 결과는 아니다. [XL config](https://huggingface.co/openai-community/gpt2-xl/raw/main/config.json), [base config](https://huggingface.co/openai-community/gpt2/resolve/main/config.json), [로컬 적재·추론 기록](results/model_preflight.json).

[설계] 종합 benchmark 최고점보다 같은 교사·학생에서 손실·gate·불확실성·수용/거절·생성을 반복해 계측할 수 있는지 우선한다. 직접 토큰 KL 비교를 위해 tokenizer·어휘 ID·특수 토큰과 같은 prefix를 확인한다. MC 추정이 0으로 퇴화하면 dropout 경로·원문 설정을 대조한다. GPU 16GB급이라는 정보만으로 실제 학습 가능성을 확정하지 않는다.

위 ID는 기반 모델이다. 선정 데이터로 준비한 교사·학생의 SFT 초기 상태를 확보·기록해야 하며 공개 최종 SFT baseline과 MiniLLM 초기 checkpoint를 같다고 가정하지 않는다. 평가용 공개 SFT checkpoint·revision과 결과는 SFT 기준선 문서에 고정했다. 향후 KD의 공통 초기 상태·prompt·분할 이력 선정은 별도 확인 사항이다. 기반 모델의 짧은 추론과 공개 SFT 모델의 평가를 완료했다. 일회성 optimizer 갱신은 통과했지만 데이터별 본 SFT·MC 추정은 미실행이다. [데이터별 비교 프로토콜](DATASET_SFT_COMPARISON.md)을 따른다. 초기에는 두 모델만 사용하고 대안·확장 모델은 적재하지 않는다.

## 3. 능력별로 검토할 벤치마크

아래는 선택 가능한 평가 축의 조사 목록이다. 현재 목표에 맞는 생성 응답 평가를 우선하고 과제와 로컬 예산을 정한 뒤 필요한 항목만 채택한다. confidence·gate·MC 불확실성·Revival의 갱신 수용/거절은 현재 관찰 대상이고 한국어 평가는 후속 참고다. 아래 benchmark 전부를 실행하기로 확정한 목록은 아니다.

| 평가 축 | 확인한 공식 자료·벤치마크 | 무엇을 관측하는가 | 해석·선택 때 확인할 것 |
|---|---|---|---|
| 지시 따르기 | [IFEval 논문](https://arxiv.org/abs/2311.07911v1) | 검증 가능한 출력 제약의 준수 | prompt/개별 instruction, strict/loose 지표를 구분. 지시 준수는 내용의 사실성과 별도 |
| 지식·추론 | [MMLU-Pro 공식 저장소](https://github.com/TIGER-AI-Lab/MMLU-Pro) | 여러 분야의 지식·추론 문항 수행 | MMLU·MMLU-Pro·Redux의 이름과 변형을 구분. few-shot·CoT·답 추출·과목별 결과 확인 |
| 상식·문맥 이해 | [HellaSwag 논문](https://arxiv.org/abs/1905.07830v1) | 문맥에 적합한 후속 내용을 고르는 능력 | 선택지 likelihood 평가와 자유 생성 평가를 구분. 정확도·길이 정규화 지표의 정의 확인 |
| 수학 | [GSM8K 공식 자료](https://github.com/openai/grade-school-math) | 다단계 산술 문제의 최종 답 | CoT·도구 사용·답 추출·생성 길이·반복 sampling 조건 확인. 최종 답 정답률만으로 풀이 전체의 타당성을 보장하지 않음 |
| 사실성 | [TruthfulQA 공식 저장소](https://github.com/sylinrl/TruthfulQA) | 잘못된 믿음을 유도하는 질문에서 참인 답을 식별하거나 생성 | generation·MC1·MC2·2025년 새 객관식 버전을 구분. 객관식 점수를 자유 생성 환각률로 바꾸어 해석하지 않음 |
| 코드 | [EvalPlus 공식 저장소](https://github.com/evalplus/evalplus): HumanEval/MBPP, HumanEval+/MBPP+ | 생성한 코드의 테스트 통과 | 원본과 추가 테스트 버전, pass@1·생성 수·실행 환경을 구분. 코드를 연구 과제로 정할 때 채택 |
| 한국어 지식 | [KMMLU](https://arxiv.org/abs/2402.11548), [KMMLU-Redux/Pro v2](https://arxiv.org/abs/2507.08924v2) | 한국어 전문 지식 문항 수행 | 한국어가 연구 대상일 때 검토. 한국어 객관식 지식 점수로 한국어 생성·지시 따르기를 모두 평가했다고 주장하지 않음 |
| 한국어 사실 질문 | [KoSimpleQA v1](https://arxiv.org/abs/2510.18368v1) | 한국 문화 지식에 관한 짧은 사실 질문의 응답 | 정답·무응답·오답의 판정 규칙·채점 구현·난도·오염을 확인. 전체 자유 생성 품질의 대체 지표로 사용하지 않음 |
| 분포·불확실성의 동작 | [MiniLLM §3.3](https://arxiv.org/html/2306.08543v4), [CaRE 부록 G](https://arxiv.org/html/2609.36734v1#S13) | calibration, confidence와 정오·지도 효과의 연결 | ECE 등의 정답 사건·확률 정의·binning을 기록. 서로 다른 어휘의 token entropy나 sequence probability를 직접 비교하지 않음 |

학습 전 학생이 거의 풀지 못하는 과제와 이미 대부분 푸는 과제는 작은 변화가 관측되기 어려울 수 있다. development 자료에서 실제 오류와 변동을 확인한다. “MMLU 몇 점 이상” 같은 임의 통과값, 고정 benchmark 개수, 근거 없는 종합 가중치를 현재 기준으로 넣지 않는다.

## 4. 실제 공개 점수의 비교 사례

아래는 **Qwen2.5 기술보고서 v2, §5.2.1, Table 8–10에 저자들이 보고한 instruction-tuned 모델의 값**이다. 원문 표의 일부 열을 재작성했다. Qwen2.5 계열과 다른 소형 모델에서 능력별 차이를 읽는 사례이며, 최종 모델 선정은 별도다. 출처: [고정한 보고서](https://arxiv.org/html/2412.15115v2).

| 모델 | MMLU-Pro | IFEval | GSM8K | MATH | HumanEval | 원문 표 |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B-Instruct | 15.0 | 27.9 | 49.6 | 34.4 | 35.4 | Table 10 |
| Qwen2.5-1.5B-Instruct | 32.4 | 42.5 | 73.2 | 55.2 | 61.6 | Table 10 |
| Qwen2.5-3B-Instruct | 43.7 | 58.2 | 86.7 | 65.9 | 74.4 | Table 9 |
| Gemma-2-2B-IT | 26.7 | 51.0 | 63.2 | 26.6 | 68.9 | Table 9 |
| Phi-3.5-Mini-Instruct | 47.5 | 52.1 | 86.2 | 48.5 | 72.6 | Table 9 |
| Qwen2.5-7B-Instruct | 56.3 | 71.2 | 91.6 | 75.5 | 84.8 | Table 8 |

수치는 원문의 0–100 척도로 표시했다. IFEval은 원문이 밝힌 strict-prompt 결과다. MATH를 MATH-500이나 다른 subset 이름으로 바꾸지 않는다. Base 모델 표의 숫자를 Instruct 모델 결과로 옮기지 않는다. 이 표는 공개 보고값이며 우리 컴퓨터에서 측정한 값이 아니다. 완전한 prompt·few-shot·CoT·decoding·실행 코드와의 일치는 후보 선정 전에 확인해야 한다.

[해석] 원문 보고값에서 Phi-3.5-Mini는 Qwen2.5-3B보다 MMLU-Pro가 높지만 IFEval·MATH는 낮다. 하나의 “전체 우수 모델”로 요약하면 과제별 차이가 사라진다. 오차 정보 없이 작은 차이를 통계적으로 확정하지 않는다. 공개 점수만으로 같은 tokenizer나 교사의 입력별 유익성을 확인할 수 없으므로, 모델·tokenizer revision과 실제 기준선도 필요하다.

## 5. 교사와 학생은 같은 문제에서 함께 비교한다

[설계] 공개 benchmark로 후보를 좁힌 후, 선택한 과제의 development 자료에서 동일한 문제·정오 기준으로 두 모델을 평가한다. 프롬프트는 각 모델이 요구하는 template를 따르되 실질적인 문제 정보와 허용된 도구·생성 조건을 맞추고 차이를 기록한다. 원문 초기화·SFT 상태를 선택했다면 그 상태로 비교한다.

| 교사 / 학생 | 사례에서 확인할 것 | 해석 범위 |
|---|---|---|
| 정답 / 오답 | 학생에게 없는 정보와 교사 지도의 잠재적 유익성 | 실제 학습 개선은 후속 비교가 필요 |
| 오답 / 정답 | 교사를 따라 학생의 기존 정답을 손상시킬 위험 | confidence가 이 경우를 구분하는지 확인 |
| 정답 / 정답 | 답이 같은데 분포·설명·안정성에 차이가 있는지 | 추가 이득이 자동 보장되지 않음 |
| 오답 / 오답 | 같은 오류인지 다른 오류인지, 평가 기준의 문제인지 | 교사의 신호를 전부 무익하다고 단정하지 않음 |

정오가 정의되는 과제에서는 네 경우의 개수·분포, 각 영역의 confidence, 학습 후 학생의 변화와 불확실성을 기록한다. 전체 교사 정확도가 높아도 학생만 맞는 사례가 있을 수 있다. 문헌의 오류 보호 인사이트를 적용하려면 그 영역을 실제로 관측해야 한다. 자유 생성에서는 검증 가능한 채점 규칙을 먼저 정하며, 임의의 문자열 일치를 사실성 판정으로 사용하지 않는다.

문헌과 연결할 자료 후보는 **모델×능력의 점수 표/그림**, **교사·학생 정오 조합 표**, **confidence와 지도 효과의 관계**, **길이에 따른 생성 결과**다. 오차·표본 수·평가 조건을 붙이며, 공개 결과와 우리 실측을 구분한다. 서로 다른 benchmark 점수를 평균하거나 임의로 정규화한 그래프를 새 선정 기준으로 만들지 않는다.

## 6. 선정 전에 고정·확인할 평가 조건

[설계] 후보 비교 기록에 아래 내용을 남긴다. 공개 자료에 없는 항목은 미확인으로 표시하고 다른 출처의 기본값으로 채우지 않는다.

- 모델 ID·revision, Base/Instruct·SFT·adapter 상태, tokenizer·chat template.
- benchmark ID·revision·split·subset·언어, 학습 자료와의 중복·오염 가능성.
- 평가 코드의 버전/commit과 task 설정, few-shot 예시·순서, system prompt.
- likelihood 평가인지 generation 평가인지, CoT/thinking·도구 사용 여부.
- temperature·top-p·생성 길이·stop·답 추출, 반복 sampling의 의미.
- dtype·quantization·backend와 변경 사항, 평가 표본 수·개별 출력·정오·오차 표시.
- judge를 쓰면 모델 버전·rubric·reference·순서 편향과 실제 정오에 대한 점검. 외부 API 없이 가능한 로컬 평가도 검토하며, 대체 채점은 원문과 다른 조건으로 표시.
- 모델을 좁히는 기준과 그때 사용한 자료, 최종 평가에 남겨 둘 자료.

[공식 구현 근거] [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)는 여러 과제의 prompt·지표를 관리하는 실행 후보다. 코드 과제를 선택하면 [EvalPlus](https://github.com/evalplus/evalplus)의 테스트 버전도 대조한다. 도구 설치와 Windows/GPU 호환성 확인, 실제 실행은 아직 하지 않았다. 공개 순위 인터페이스는 모델 찾기에 활용하되 해당 점수의 원자료와 평가 조건까지 확인한다.

RTX 5060 Ti 16GB급에서 **모델별 평가를 차례로 실행할 수 있는지**와 **선택한 교사–학생 학습 경로를 실행할 수 있는지**는 별도로 확인한다. 평가가 가능하다고 교사·학생·optimizer·rollout을 포함한 학습도 가능하다고 판단하지 않는다. 필요한 정밀도·표본 범위·길이를 보존한 실행 시험 후 예산을 정한다.

## 7. 현재 결론과 다음 산출물

지금은 평가 축과 공개 비교 사례를 확보했다. 다음은 MiniLLM의 원문 구성·평가 조건과 이번에 관찰할 응답 변화의 범위를 확인하는 것이다. **문헌에서 설명할 원리 → 원문 모델·평가 근거 → 같은 development 조건의 초기 학생·증류 후 학생 비교 설계 → 로컬 실행 가능성**을 연결해 선정한다. 교사 응답은 해석의 참고로 사용한다. 핵심 비교는 FKL/RKL/Hard/Soft와 같은 손실의 Revival 유무·결합이다. SFT 학생은 초기 상태의 참고 기준선이다. 같은 초기화에서 손실·불확실성·수용/거절·실제 갱신을 계측할 수 있는지 확인한다. 실행 목록은 자원과 설정 근거를 확인한 뒤 확정하며 한국어 평가는 후속 참고로 둔다.

초기 기반 모델은 GPT-2 XL/base로 정했고 revision·어휘·동시 GPU 적재·짧은 입력 추론을 확인했다. 이어 저자의 Dolly SFT 완료 checkpoint를 확보해 Dolly·SST-2·BoolQ 기준선 평가를 추가했다. 이 평가용 revision·protocol과 향후 KD의 공통 초기 상태·최종 평가 선정은 구분한다. KD 통과 기준·설정 탐색·최종 평가 규모는 미정이다. 공개 점수의 완전한 실행 재현, 전체 학습 이력·오염 감사와 데이터별 본 SFT는 수행하지 않았다.
