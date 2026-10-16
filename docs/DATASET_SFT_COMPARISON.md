# 후속 데이터 후보의 비교 설계와 감사 기록

상태 정리: 2026-10-16, Asia/Seoul. **이 문서는 Dolly·UltraChat·MetaMathQA를 각각 SFT하려던 과거 설계와 예비 관측을 보관한다. 현재 실행 지침이 아니다.** 현재는 학습을 Dolly로 고정하고 5계열·10모델의 SFT·선정 평가를 진행한다. [현재 SFT 명세](DOLLY_SFT_MODEL_SELECTION.md)와 [생성 평가 계획](GENERATION_EVALUATION.md)을 따른다.

당시 원자료 다운로드·파일 hash·전체 구조·길이 감사와 두 기반 모델의 일회성 학습 갱신을 확인했다. 세 데이터의 직접 SFT 성능 비교는 수행하지 않았다. 아래 §1–7의 데이터별 학습·평가 규칙은 후속 연구로 채택할 경우 다시 동결할 보관 설계다. 예비 길이·갱신 수치를 성능 점수로 해석하지 않는다.

## 1. 모델과 데이터는 어떤 순서로 정하는가

연구 질문·평가 사건 → 데이터 후보·전처리 → 공통 모델의 SFT 비교 → 모델·데이터 최종 선정 → KD 비교 순서다. 현재 XL/base는 비교의 통제 조건이다. 문맥 길이·실제 수행·교사/학생 오류 조합이 연구 질문에 맞는지 확인한 후 확정한다.

SFT 비교에서 데이터와 모델 초기 상태를 동시에 바꾸지 않는다. 각 데이터의 XL은 같은 XL 기반 revision에서, base는 같은 base 기반 revision에서 출발한다. 데이터 간 순차 추가 학습이나 이미 Dolly로 학습한 모델의 재사용은 이번 비교가 아니다. 두 revision은 [모델 점검 기록](../results/model_preflight.json)에 고정했다.

최고 accuracy 하나로 데이터를 고르지 않는다. KL 방향과 gate의 차이를 관찰할 수 있는 분포·응답, Revival의 지도 유익성을 판정할 정답, 교사·학생의 다양한 오류 조합, 누수·품질·문맥 길이·로컬 반복 가능성을 함께 본다. 두 모델이 모두 거의 풀지 못하거나 오류 조합이 한쪽으로만 몰리는지도 확인한다. 수치 임계값은 임의로 만들지 않는다.

## 2. 원본 전체 집계

[재실행 코드](../scripts/audit_dataset_candidates.py)와 [전체 JSON](../results/dataset_candidates/audit.json). 표본 추정이 아닌 명시한 파일 전체 집계다. UltraChat의 `gen` split은 SFT용 자료가 아니므로 내려받거나 학습량에 합치지 않았다. 원본 행 수는 최종 학습량이 아니다.

| 후보·공개 split | 읽은 행/대화 | 직렬화 길이 중앙값 | 1,024토큰을 초과한 완전한 행/대화 |
|---|---:|---:|---:|
| Dolly 원본 train | 15,011 | 134 | 207 (1.38%) |
| UltraChat train_sft | 207,865 | 1,159 | 123,947 (59.63%) |
| UltraChat test_sft | 23,110 | 1,151 | 13,640 (59.02%) |
| MetaMathQA train | 395,000 | 215 | 912 (0.23%) |

길이는 공통 GPT-2 tokenizer로 prompt+응답+EOS를 센 값이며 잘라내지 않았다. Dolly·MetaMath는 MiniLLM의 instruction/input wrapper를 사용했다. UltraChat은 같은 도입문에 Instruction/Response header를 반복한 **검토용 포맷**으로 대화 전체를 셌다. 원문 UltraChat/Zephyr 템플릿 재현이 아니며, 이 수치는 최종 전처리의 제외율이나 첫 턴 사용 가능 비율이 아니다. `1,024`는 GPT-2의 native context 용량이다. 모든 대화를 자르겠다는 선택값이 아니다.

- **Dolly:** 빈 instruction/response는 0개, 정규화 instruction+context는 14,808종이다. 같은 입력의 반복 203행과 완전히 같은 직렬화 15행은 다른 통계다. context는 4,467행에 있다.
- **UltraChat:** train에 assistant 657,794턴, test에 73,152턴이 있다. 대화 수를 학습 target 턴 수로 표시하지 않는다. 역할 순서 위반은 없었고 빈 message는 train 31개·test 3개였다. `prompt`와 첫 user message의 원문 문자열은 train 12,091개에서 달랐지만 대소문자·공백 정규화 후에는 모두 일치했다. 학습 내용은 `messages`에서 읽는다. 공개 SFT train/test 사이 prompt_id·정규화 첫 질문의 정확한 겹침은 0종이었다. 의미상 중복 검증 완료를 뜻하지 않는다.
- **MetaMathQA:** 원문 문자열 그대로는 original_question 13,929종, 공백만 정리하면 13,928그룹이다. 수학 변수의 대소문자는 유지했다. 증강 395,000행을 395,000개의 독립 문제로 보지 않는다. 빈 query 4행과 완전히 같은 직렬화 8,957행이 있었다. 모든 response에서 `The answer is:` 표지를 찾았지만 답의 정확성을 검증한 것은 아니다. 같은 query의 여러 풀이·답 증강을 모두 불량 중복으로 삭제하지 않는다.

고정 revision·파일별 SHA-256·길이 분포와 MetaMath 8개 type의 수는 JSON을 따른다. 대소문자·공백 정규화 입력 일치는 의미상 동치나 완전한 오염 검사의 대체물이 아니다.

## 3. 전처리와 분할

| 단계 | Dolly | UltraChat | MetaMathQA |
|---|---|---|---|
| 유지할 정보 | instruction/context/response/category | messages의 역할·순서·내용, prompt_id | query/response/type/original_question |
| group 단위 | 같은/유사 instruction+context | prompt_id와 같은 첫 질문의 대화 | original_question와 GSM8K/MATH 원문 대응 |
| 검증 | 빈 필드·원문 context·중복 | 빈 message·역할 순서·대화 중복 | 빈 query·원문 그룹·최종 답·증강 유형 |
| 학습 target | response와 EOS | assistant 응답과 정한 종료 토큰 | 풀이 전체와 최종 답·EOS |
| loss에서 제외 | instruction/context/header/padding | user/header/padding, 정한 정책에 따른 대화 경계 | query/header/padding |
| 길이 문제 | 입력·정답을 보존하는 완전한 예제 확인 | 모든 이전 맥락을 유지하는 턴 단위 또는 완전 대화 비교; 긴 맥락의 임의 삭제 금지 | 최종 답을 잘라낸 풀이를 정상 정답으로 학습하지 않음 |

원자료는 수정하지 않는다. 파생 자료에는 원본 revision·파일·행/대화 ID, group ID, split 역할, 실제 토큰 수·supervised mask, 제외 사유를 남긴다. 수학의 표기·숫자·변수 대소문자와 대화의 역할을 임의로 정규화하지 않는다. 새 special token을 추가해 교사/학생 어휘 좌표를 바꾸지 않는다.

분할은 길이 필터 전에 원문 group에서 정해 파생 턴·증강이 다른 split으로 새지 않게 한다. 공개 test를 train이나 학습률 선택에 쓰지 않는다. UltraChat의 공개 test_sft는 잠근 최종 평가 후보이고 development는 train_sft의 group에서 확보한다. MetaMath는 자체 test가 없으므로 원문 문제의 group을 분리하고 GSM8K/MATH 원본 정답과 연결해 채점한다. 공식 GSM8K/MATH test를 데이터 선택에 먼저 쓰면 독립 최종 test로 다시 부르지 않는다.

이미 확인한 Dolly 500문항·SST-2·BoolQ 점수는 선정 정보다. 이를 완전히 보지 않은 최종 증거로 취급하지 않는다. Dolly의 공개 raw/valid 사이 입력 겹침과 원본 context 차이는 [기존 점검](DATA_SELECTION.md#4-기준-후보-dolly-15k의-사용-조건-확인)을 따른다. 신규 분할 크기·정답 감사·표집/반복 정밀도는 본학습 전 확정하고 숫자의 근거를 기록한다.

## 4. 성능을 어떻게 비교하는가

학습 데이터 3종 × 모델 2종을 **같은 평가 열**에 놓는 표와 각 데이터의 고유 수행을 보는 표를 따로 만든다. 공통 평가에는 같은 문항·prompt·생성/label 판정 규칙을 사용한다. `Dolly-SFT → 공통 평가`, `UltraChat-SFT → 공통 평가`, `MetaMath-SFT → 공통 평가`의 비교다. 데이터별 난도가 다른 정답률을 직접 순위화하지 않는다. SFT 전 기반 모델의 동일 평가도 기록해 변화량을 확인한다.

| 평가 축 | 지표·채점 근거 | 해석 범위 |
|---|---|---|
| 정답 label이 있는 공통 과제 | SST-2/BoolQ 같은 고정 문항의 accuracy·macro-F1·클래스별 오류 | 언어 분류/읽기 수행. 대화·수학 전체 능력 점수가 아님 |
| 수학의 최종 답 | 원문 정답에 대한 answer accuracy, 답 추출 실패율·길이 제한 도달률 | 숫자·분수·부호·단위·기호 동치와 파서 오류를 먼저 검사. 실패는 분모에서 빼지 않음 |
| Dolly 일반 생성 | ROUGE-L·reference NLL·응답 길이·과제별 사례, 별도 품질 검토 | ROUGE나 다음 토큰 accuracy를 문항 정답률로 이름 바꾸지 않음 |
| UltraChat 대화 | held-out assistant NLL, 고정 rubric의 사람/judge 평가·실패 사례 | 합성 reference와 문자열이 다르다는 이유만으로 오답 처리하지 않음. 판정자·rubric 미선정 |
| KD 관측 가능성 | 같은 문제에서 교사/학생 정오 조합·confidence·불확실성, 분포·길이 | NLL·entropy가 낮다는 이유로 지도가 유익하거나 정답이라고 단정하지 않음 |

MetaMath의 [고정한 공식 GSM8K evaluator](https://github.com/meta-math/MetaMath/blob/fe667b14f9a4c51bde9809adafb82b6b5d255800/eval_gsm8k.py)는 답 표지 뒤 숫자를 읽고 반올림한다. 이를 무검증 복사해 근접한 오답을 정답으로 만들지 않는다. 공식 채점과 수치 동치 판정의 차이, 정수·소수·분수·comma·parse failure에 대한 작은 self-check를 먼저 준비한다. 수학의 기호 동치를 단순 문자열 일치로 일반화하지 않는다.

**현재 새 점수는 전부 미측정이다.** [기존 SFT 점수](SFT_BASELINE_EVALUATION.md)는 저자가 공개한 Dolly-SFT checkpoint의 참고 결과이며 아래 세 조건의 직접 학습 비교표를 채우는 값으로 재사용하지 않는다.

| 새 SFT 학습 데이터 | XL의 공통/과제별 평가 | base의 공통/과제별 평가 |
|---|---|---|
| Dolly | 미실행 | 미실행 |
| UltraChat | 미실행 | 미실행 |
| MetaMathQA | 미실행 | 미실행 |

## 5. 학습 설정의 근거와 선택

SFT 단계에는 KD 온도 T나 confidence gate를 넣지 않는다. KD 설정 탐색과 데이터 준비용 SFT를 혼합하지 않는다.

[MiniLLM B.1](https://arxiv.org/html/2306.08543v4)은 SFT baseline에 대해 <1.3B의 LR 후보 `[5e-4, 1e-4, 5e-5]`·20 epochs, 큰 모델의 `[5e-5, 1e-5, 5e-6]`·10 epochs, batch 후보 `[32,64]`와 validation ROUGE-L 선택을 보고한다. **원문의 탐색 근거이지 세 데이터에서 최적이라는 보장이 아니다.** MiniLLM 초기화의 3 epochs/validation loss 선택은 별도 단계다.

[MetaMath 공식 설정](https://github.com/meta-math/MetaMath/tree/fe667b14f9a4c51bde9809adafb82b6b5d255800#supervised-fine-tuning)은 LLaMA-2 7B에 LR `2e-5`, batch `128`, 3 epochs, 최대 길이 `512`, cosine scheduler를 보고한다. 이를 GPT-2에 그대로 적용한 뒤 최적 설정이라고 표시하지 않는다. MetaMath 원문 조건과 공통 모델·토큰 예산을 통제하는 비교는 다른 실험이다.

선택 순서는 전처리·development·채점 동결 → 출처 있는 후보와 탐색 규칙 등록 → 학습·validation 기록 → checkpoint/설정 선택 → 잠근 평가다. 같은 모델 내 데이터 비교의 주 조건은 같은 supervised-token 노출 예산과 같은 탐색 기회로 두는 방향으로 설계한다. 같은 행 수나 같은 epoch 수는 실제 학습량을 같게 만들지 않는다. task별 최대 수행을 찾는 실험을 별도로 하면 탐색량과 다른 학습량을 명시한다. NLL은 같은 자료에서 checkpoint를 고르는 데 쓸 수 있지만 서로 다른 데이터의 NLL 숫자만으로 우수 데이터를 고르지 않는다.

별도 시간 제한이 없어도 총 학습량·종료 규칙을 무한대로 두지 않는다. development 변화·정밀도·원문 조건을 근거로 범위와 종료 규칙을 사전에 기록한다. LR·epoch·batch·분할 비율을 편의상 하나로 확정하거나, test를 본 후 유리한 설정만 남기지 않는다. 현재 본학습의 값은 검증 선택 전이다.

## 6. 이 컴퓨터에서 실행한 학습 점검

[코드](../scripts/preflight_sft_training.py), [실측 JSON](../results/dataset_candidates/training_preflight.json). 기반 XL/base를 각각 다시 불러와 microbatch 1, 길이 512와 1,024에서 response-only loss·backward·optimizer.step을 실행했다. 모든 parameter의 유한 gradient와 유한 갱신 후 가중치, 실제 한 가중치의 변화를 확인했다. 각 dtype의 512 점검 뒤 1,024 점검은 한 번 갱신된 상태에서 수행했다. checkpoint를 저장하지 않았다.

공통 실행 경로는 SDPA·gradient checkpointing·BF16 autocast(cache off)·**bitsandbytes 0.47.0 AdamW8bit**다. 일반 FP32 AdamW의 가중치+gradient+두 moment만으로 XL은 약 23.21GiB여서 16GB GPU의 범위를 넘는다. 8비트 optimizer는 메모리를 줄이기 위한 원문과의 변경이다. 모델 가중치의 8비트 양자화나 LoRA를 적용한 것은 아니다. [Windows/CUDA 12.8·sm120 지원 근거](https://huggingface.co/docs/bitsandbytes/v0.47.0/installation), [8비트 optimizer 설명](https://huggingface.co/docs/bitsandbytes/v0.47.0/optimizers).

| 경로 | XL 최대 PyTorch allocated | base 최대 PyTorch allocated | 일회성 갱신 |
|---|---:|---:|---|
| FP32 parameter + BF16 autocast + 8bit optimizer | 15,954MiB | 1,600MiB | 두 길이 통과 |
| BF16 parameter + BF16 autocast + 8bit optimizer | 9,702MiB | 1,132MiB | 두 길이 통과 |

BF16 parameter 경로는 FP32 master weight를 유지한 학습과 수치 조건이 다르며 최종 기본값으로 채택하지 않았다. 두 경로의 속도를 모델 성능·최적성으로 해석하지 않는다. FP32 XL의 reserved peak는 16,404MiB로 물리 VRAM 16,311MiB보다 컸다. Windows WDDM의 전용/공유 메모리를 분리 측정하지 않았으므로 전용 VRAM 안에서 안정적으로 돌아간다고 결론내리지 않는다. 장기 실행·gradient accumulation·checkpoint 재개·KD 동시 학습은 미검증이다. 측정 시간은 초기화·유한값 검사도 포함해 본학습 완료 시간을 예측하는 처리량이 아니다.

라이브러리는 로컬 `tmp/training_deps`에 설치했다. 원본 데이터·가중치와 함께 Git에서 제외한다. 재실행에는 기존 평가 환경과 `bitsandbytes==0.47.0`, 로컬 감사용 `pyarrow==26.0.0`이 필요하다. 표에 보인 LR·seed 등 실행 점검 값은 JSON에 근거와 역할을 기록했으며 본학습의 검증 선택값이 아니다.

## 7. 다음 실행에서 고정할 것

1. 원문 group·평가 중복·불량 필드 처리와 UltraChat 대화 길이 정책을 확정하고 제외 원인을 집계한다.
2. 최종 개발/평가 manifest, assistant mask·EOS·padding·긴 정답 보존을 검사한다.
3. 안정적인 precision/optimizer·누적 갱신·저장/재개 경로를 확인하고 LR·학습량 선택 규칙을 등록한다.
4. 같은 기반 모델에서 데이터별 SFT·동일 평가를 수행하고, 성능·정오 조합·분포 시각화의 근거로 모델과 데이터를 최종 선정한다.

현재 단계에서 특정 데이터의 우수성이나 accuracy를 추정해 채우지 않는다.
