# Dolly 네 모델 SFT 비교 계획

갱신일: 2026-10-15. 목적은 이후 지식증류 비교에 사용할 교사·학생 조합의 선정이다. 네 모델을 Dolly 기준 응답으로 각각 추가 학습하고 공통 문항에서 평가한다. KD 손실·gate·Revival은 이 단계에 적용하지 않는다.

## 모델과 정밀도

| 역할 후보 | 원본 공개 모델 | 기반 가중치 | 학습 범위 |
|---|---|---|---|
| GPT-2 교사 | openai-community/gpt2-xl | BF16, 양자화 없음 | LoRA |
| GPT-2 학생 | openai-community/gpt2 | BF16, 양자화 없음 | LoRA |
| Qwen 교사 | Qwen/Qwen3.5-9B | NF4·double quantization, BF16 계산 | LoRA, 즉 QLoRA |
| Qwen 학생 | Qwen/Qwen3.5-0.8B | BF16, 양자화 없음 | LoRA |

9B만 16GB GPU의 적재 제약 때문에 양자화한다. GPT-2까지 낮은 비트로 통일하지 않는다. 정규화 모듈과 학습할 LoRA 가중치는 FP32를 사용한다. 동결한 embedding·출력 행렬은 BF16이다. 양자화 여부와 전체 미세조정/LoRA 여부는 다른 결정이며, 이번 네 모델은 모두 LoRA 실험이다.

원본 model ID·revision·소스 해시·라이브러리 버전은 [protocol.json](results/dolly_sft_v1/protocol.json)에 고정한다. 과거 연구의 학습된 adapter는 초기값으로 사용하지 않는다. Qwen은 공개 checkpoint에서 텍스트 언어모델 부분만 불러오고 vision encoder는 사용하지 않는다.

GPT-2는 pretrained 모델, Qwen은 이미 post-training을 거친 모델이다. [Qwen 공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-0.8B#model-overview)에서 학습 단계를 확인할 수 있다. 모델 크기·학습 이력·어휘·prompt 형식·9B 정밀도가 함께 다르므로 순수한 아키텍처 효과나 양자화 효과로 해석하지 않는다. 두 계열 각각의 교사–학생 격차와 오류 유형을 확인한다.

## 데이터와 전처리

실제 입력은 [MiniLLM/dolly](https://huggingface.co/datasets/MiniLLM/dolly)의 고정 revision이다. 원래 Dolly-15K 15,011건과 MiniLLM 배포 파일의 행 수를 동일시하지 않는다. 배포본 `raw.jsonl`은 12,435건, `valid.jsonl`은 500건이다.

1. [MiniLLM 공식 전처리](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/scripts/tools/process_data_dolly.sh)의 `dev_num=1000`에 따라 raw의 첫 1,000건을 개발 검증 후보로 두고 나머지를 학습 후보로 둔다.
2. 공개 valid 500건은 모델 선정용 생성 평가로 둔다. 지시문을 대소문자·공백 정규화해 묶고, 다른 context가 있어도 같은 지시문이면 분할 사이 중복을 제거한다. 공개 평가와의 중복 제외를 먼저 적용하고, 개발 검증과 겹친 학습 문항을 제외한다. 학습·검증의 완전 중복도 제거한다.
3. 네 tokenizer 모두에서 prompt 256토큰 이하, 학습·검증의 전체 prompt+응답+EOS 512토큰 이하인 같은 원본 record만 남긴다. 긴 응답을 자르지 않는다. 256은 위 전처리 코드, 512는 MiniLLM의 모델 길이와 로컬 실행 범위를 근거로 한 제한이다. 보수적인 공통 필터는 긴 응답의 비율을 줄이는 선택 편향을 갖는다.
4. 최종 공통 분할은 **학습 11,117건·검증 975건·선정 평가 500건**이다. 제외 사유별 수와 준비 파일 SHA-256은 protocol에 기록한다. 지시문 기준 분할 간 중복이 없음을 assert한다. 의미가 비슷한 모든 문항의 오염까지 검증한 것은 아니다.
5. GPT-2는 MiniLLM wrapper, Qwen은 native chat template에서 thinking을 끈 prompt를 사용한다. 지시·context·기준 응답의 내용은 같다. 계열 내 어휘 ID 대응은 일치함을 확인했다. 모델별 native EOS를 사용하며 어휘 일치가 모든 특수 토큰 설정의 일치를 뜻하지는 않는다.
6. Prompt와 응답을 따로 tokenize해 연결하고 prompt label을 `-100`으로 마스킹한다. 응답과 마지막 EOS만 학습하며 padding은 손실에서 제외한다. EOS를 padding 토큰으로 사용해도 실제 EOS label이 지워지지 않는지 검사한다.

공개 평가 자료는 이전 공개-SFT 평가에서 이미 관찰한 자료다. 이후 KD 결론의 독립적인 최종 test로 부르지 않는다.

## 학습 설정의 근거와 선택

| 설정 | 사전 명세 | 근거와 한계 |
|---|---|---|
| LoRA | r=64, alpha=16, dropout=0.05, all-linear | QLoRA 공식 13B/33B 스크립트의 초기 후보. 모델별 최적 rank라는 주장은 하지 않음 |
| 학습률 후보 | 0.0001, 0.0002 | 아래 공식 33B·13B 스크립트의 값. 모델마다 검증 손실로 둘 중 선택하며 전체 탐색 공간의 최적값을 보장하지 않음 |
| Epoch 상한 | 후보별 3회 전체 학습 | MiniLLM 부록 B.1의 초기화 단계 근거. 실제 선택은 각 epoch 검증 손실로 수행 |
| Microbatch·누적 | 1건·16회, 유효 batch 16건 | QLoRA 13B 스크립트의 값과 실제 GPU 점검. 마지막 불완전 batch는 그대로 처리 |
| Optimizer | AdamW, betas=(0.9,0.999), eps=1e-8, weight decay=0 | QLoRA 설정·기본 optimizer 값. 최적성 주장은 하지 않음 |
| Gradient clipping | 0.3 | QLoRA 공식 실행 설정 |
| Scheduler | constant_with_warmup, 전체 step의 3% warmup | QLoRA의 warmup 비율을 명시적으로 구현. 원본 CLI의 constant scheduler와 다른 점을 기록 |
| 학습 seed | 0 | QLoRA 실행 설정. 단일 학습 seed 결과로 학습의 통계적 우월성을 주장하지 않음 |
| 검증·저장 | 매 epoch 말 | 전체 학습 pass를 기준으로 checkpoint 선택. 학습 loss 최솟값으로 고르지 않음 |

공식 근거: [QLoRA 13B 스크립트](https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/scripts/finetune_guanaco_13b.sh), [33B 스크립트](https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/scripts/finetune_guanaco_33b.sh), [MiniLLM 부록 B.1](https://arxiv.org/html/2306.08543v4#A2.SS1).

두 학습률은 각각 같은 원본에서 새 LoRA를 초기화해 학습한다. 모델마다 3개 epoch checkpoint 중 검증 손실이 가장 낮은 것을 고르고, 두 학습률의 선택 checkpoint를 같은 검증 기준으로 비교한다. 총 4모델 × 2학습률의 실행이며 완료되지 않은 실행을 결과로 간주하지 않는다.

검증 손실은 microbatch 1에서 응답 토큰 NLL을 문항별 평균한 뒤 전체 문항에 평균한 값이다. 같은 모델의 checkpoint·학습률 선택에 사용한다. 서로 다른 tokenizer의 NLL을 직접 성능 순위로 비교하지 않는다.

3 epoch는 초기화 단계에 근거한 제한이다. 원문의 SFT baseline은 크기에 따라 10/20 epoch와 별도 학습률·batch 탐색 및 ROUGE-L checkpoint 선택을 사용한다. 따라서 이번 결과를 MiniLLM SFT baseline의 완전 재현이나 각 모델의 최상 성능으로 표시하지 않는다. 양자화·LoRA·공통 필터·prompt·scheduler 변경도 재현 차이로 남긴다.

## 공통 평가와 시각화

| 평가 | 공통 자료와 지표 |
|---|---|
| SST-2 | validation 872건, accuracy·macro-F1·balanced accuracy·weighted-F1·MCC·클래스별 recall·혼동행렬 |
| BoolQ | validation 중 공통 길이 기준을 통과한 3,269건, 같은 분류 지표. 원본 row 3154 한 건은 GPT-2의 1,024토큰 용량을 넘으므로 전 모델 공통 제외 |
| Dolly 생성 | 모델 선정 단계: 500문항 × 공통 생성 seed 10 한 개. ROUGE-1/2/L의 문항 평균, 생성 길이·상한 도달률·빈 응답률 |

**반복 수 결정:** 모델 선정 단계에서는 네 모델 모두 같은 seed 10으로 생성 평가한다. 전체 생성 수는 500 × 1 × 4 = 2,000개다. 10은 MiniLLM의 seed 목록 중 첫 값을 공통 조건으로 고정한 것이며 최적 seed라는 의미는 아니다. 최종 선정 모델의 본 실험에서는 MiniLLM을 따라 seed 10/20/30/40/50의 5개 생성 반복을 사용한다. 두 단계의 반복 수는 학습 반복 수와 별개다. 현재 실행은 모델 선정까지만 수행하며 본 실험의 추가 학습·방법 비교는 이후 설계에 따른다.

분류는 같은 지시 내용과 label 후보의 조건부 sequence log probability로 판정한다. 자유 응답을 임의 파싱한 정답률과 구분한다. Brier score는 두 label 후보 사이에서 정규화한 확률임을 명시한다. 자료 revision·제외 행·파일 해시도 평가 명세에 남긴다.

생성은 temperature=1, top-k=0, top-p=1, no-repeat-ngram=6과 생성 seed를 [MiniLLM 평가 코드](https://github.com/microsoft/LMOps/tree/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm)·부록 B.2에서 가져온다. **최대 새 토큰 256과 microbatch 1은 로컬 변경값**이다. 256은 전체 길이 상한 512에서 공통 최대 prompt 256을 뺀 보수적 응답 예산이다. 원본은 512에서 실제 padded prompt 길이를 빼므로 짧은 prompt에 더 긴 응답을 허용한다. 이번 고정 상한은 긴 응답의 점수를 낮출 수 있어 상한 도달률도 함께 보고한다. Batch 변경 때문에 seed가 같아도 원문과 생성 문장의 완전 일치를 요구하지 않는다.

그림에는 네 모델의 SST-2·BoolQ accuracy/macro-F1, 단일 생성 seed의 Dolly ROUGE-L, 학습·검증 손실 곡선을 별도로 제시한다. 현재 단일 seed에서는 생성 seed 간 표준편차를 추정할 수 없으므로 JSON에 null로 남기고 표·그래프에 오차막대나 ±0을 붙이지 않는다. 본 실험의 5개 생성 반복에서 평균·표본 표준편차를 보고한다. 점수가 없는 모델은 공란으로 두며 이전 공개 SFT 모델 점수로 채우지 않는다. Dolly ROUGE를 accuracy로 해석하지 않는다.

## 실행과 상태

현재 환경에서 다음 순서로 실행한다. Python은 Qwen3.5를 지원하는 transformers가 설치된 환경이어야 한다.

```powershell
python -B scripts/dolly_sft.py prepare
python -B scripts/dolly_sft.py preflight
python -B scripts/preflight_dolly_trainer.py
python -B scripts/dolly_sft.py train
python -B scripts/evaluate_dolly_sft.py
python -B scripts/report_dolly_sft.py
```

점검 이후 `python -B scripts/run_dolly_sft.py`로 본 학습 → 평가 → 표·그림 작성을 순서대로 실행할 수 있다. 실행 상태는 `results/dolly_sft_v1/worker_status.json`, 상세 로그는 `tmp/dolly_sft_v1/worker.log`에 남긴다. 실패 시 다음 단계로 진행하지 않고 실패 상태를 저장한다. 같은 worker의 중복 실행은 파일 잠금으로 막는다.

사전 점검은 실제 가중치 갱신·유한 gradient·EOS/padding 마스크·adapter 저장/재적재 logits 일치·cache 생성·16건 누적 Trainer 갱신·검증·optimizer checkpoint 저장을 확인한다. 일회성 점검 점수를 본 SFT 성능으로 보고하지 않는다. Trainer 일회성 점검에만 warmup=0을 사용해 실제 한 번의 가중치 변화를 확인하며, 본 학습은 위 3% warmup을 유지한다.

공개 요약은 `results/dolly_sft_v1/`, 대형 checkpoint·예측·생성 응답과 로그는 Git에서 제외한 `tmp/dolly_sft_v1/`에 저장한다. 학습은 epoch checkpoint에서 재개하며 완료한 후보는 재학습하지 않는다. 생성은 완료 seed 단위로 재개한다. CUDA의 Qwen 고속 커널이 없는 현재 Windows 환경에서는 정확한 PyTorch 대체 구현이 사용되므로 속도는 실측 기록을 따른다. 짧은 점검 통과가 장시간 안정성을 보장하지는 않는다.
