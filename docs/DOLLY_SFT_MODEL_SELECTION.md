# Dolly 모델 SFT 비교 계획

<!-- SFT_PROGRESS:START -->

현재 비교 범위는 **5개 계열·10개 모델**이며, SFT 선택 완료 **2/10**, 생성 주 평가 4종 완료 **2/10**, 기존 Dolly·분류 보조 평가 완료 **2/10**이다. 생성 평가는 공통 seed 10 한 개를 사용한다. 모델별 상태·선택 LR·검증 NLL·실제 정밀도는 [SFT 진행 기록](SFT_PROGRESS.md), 교사·학생의 생성 성능 차이는 [생성 주 평가](../results/generation_benchmarks_v1/REPORT.md), 분류 편향 등은 [기존 Dolly·분류 보조 평가](../results/dolly_sft_v1/REPORT.md)를 따른다. 최종 조합은 아직 확정하지 않았다.

<!-- SFT_PROGRESS:END -->

갱신일: 2026-10-16. 목적은 이후 지식증류 비교에 사용할 교사·학생 조합의 선정이다. 5개 계열의 10개 모델을 Dolly 기준 응답으로 각각 추가 학습하고 공통 문항에서 평가한다. KD 손실·gate·Revival은 이 단계에 적용하지 않는다.

## 진행 흐름과 선정 범위

**Dolly SFT → 모델별 최저 개발 NLL 체크포인트 → 네 생성 평가·보조 진단 → 보완 평가·오류 분석 → 증류 실행 점검 → 한 교사·학생 조합 선정**이다. 자동 worker의 평가 완료와 최종 조합 선정은 별개다. 현재 KD 학습은 실행하지 않았다.

학습은 Dolly만 사용한다. Self-Instruct·S-NI·Vicuna는 생성 평가에만 사용한다. NLL은 각 모델 내부의 LR·checkpoint 선택이고, 최종 조합을 다른 tokenizer의 NLL이나 ROUGE-L 하나로 확정하지 않는다. 보완 지표의 미결정 사항과 단계별 완료 근거는 [생성 평가 계획](GENERATION_EVALUATION.md#조합-선정까지의-단계와-완료-기준)을 따른다.

## 모델과 정밀도

2026-10-15 역할 분담: 팀원은 비양자화 조건을 별도로 진행하고, 이 컴퓨터는 아래 기존 BF16/NF4 정책으로 계속한다. 비교 조건과 외부 결과의 분리 보관 기준은 [공통 지침의 역할 분담](EXPERIMENT_GUIDE.md#양자화-조건의-역할-분담)을 따른다. 팀원 측 결과는 아직 수령하지 않았다.

| 역할 후보 | 원본 공개 모델 | 기반 가중치 | 학습 범위 |
|---|---|---|---|
| GPT-2 교사 | openai-community/gpt2-xl | BF16, 양자화 없음 | LoRA |
| GPT-2 학생 | openai-community/gpt2 | BF16, 양자화 없음 | LoRA |
| Qwen 교사 | Qwen/Qwen3.5-9B | NF4·double quantization, BF16 계산 | LoRA, 즉 QLoRA |
| Qwen 학생 | Qwen/Qwen3.5-0.8B | BF16, 양자화 없음 | LoRA |
| OPT 교사 / 학생 | facebook/opt-2.7b / opt-125m | BF16 먼저 실제 점검, 전용 VRAM에 맞지 않으면 NF4 | LoRA |
| OpenLLaMA 교사 / 학생 | openlm-research/open_llama_7b / open_llama_3b | BF16 먼저 실제 점검, 전용 VRAM에 맞지 않으면 NF4 | LoRA |
| Gemma 교사 / 학생 | google/gemma-2-9b-it / gemma-2-2b-it | 9B는 BF16 가중치만으로 VRAM 초과해 NF4, 2B는 실제 점검 | LoRA |

기존 GPT-2 두 모델과 Qwen0.8B는 BF16, Qwen9B는 NF4를 유지한다. 추가 6개 모델은 BF16 가중치 크기가 물리 VRAM을 넘으면 NF4로 시작하고, 그렇지 않으면 BF16의 실제 적재·갱신·저장 점검부터 한다. CUDA OOM 또는 측정한 전용 VRAM 여유보다 높은 peak allocated가 확인될 때 NF4로 전환한다. 그 밖의 오류를 메모리 부족으로 간주하지 않는다. 실제 정밀도·전환 근거·메모리 수치를 모델별 protocol과 점검 JSON에 남긴다. 정규화와 LoRA는 FP32, 동결 embedding·출력 행렬은 BF16이다. 10개 모델 모두 LoRA 실험이다.

추가 조합은 [CaRE §4의 Dolly 실험](https://arxiv.org/html/2609.36734v1)과 맞춘다. Gemma는 사용자 조건 동의·로컬 인증 후 다시 포함했다. 공식 가중치만 사용하고 원본 revision·파일 형식은 [추가 모델 registry](../results/dolly_sft_additional_v1/registry.json)에 고정한다. 기존 4개 모델의 학습 checkpoint와 protocol은 유지하며 추가 모델은 부모 protocol·동일 원본 자료·실행 소스 해시를 가진 별도 protocol을 만든다.

원본 model ID·revision·소스 해시·라이브러리 버전은 [protocol.json](../results/dolly_sft_v1/protocol.json)에 고정한다. 과거 연구의 학습된 adapter는 초기값으로 사용하지 않는다. Qwen은 공개 checkpoint에서 텍스트 언어모델 부분만 불러오고 vision encoder는 사용하지 않는다.

GPT-2·OPT·OpenLLaMA는 pretrained 모델, Qwen·Gemma-IT는 post-training을 거친 모델이다. [Qwen 공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-0.8B#model-overview)와 [Gemma 공식 카드](https://huggingface.co/google/gemma-2-2b-it)에서 학습 단계를 확인한다. 크기·학습 이력·어휘·prompt 형식·정밀도가 함께 다르므로 순수한 아키텍처 효과나 양자화 효과로 해석하지 않는다. 계열 각각의 교사–학생 격차와 오류 유형을 확인한다.

## 데이터와 전처리

실제 입력은 [MiniLLM/dolly](https://huggingface.co/datasets/MiniLLM/dolly)의 고정 revision이다. 원래 Dolly-15K 15,011건과 MiniLLM 배포 파일의 행 수를 동일시하지 않는다. 배포본 `raw.jsonl`은 12,435건, `valid.jsonl`은 500건이다.

1. [MiniLLM 공식 전처리](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/scripts/tools/process_data_dolly.sh)의 `dev_num=1000`에 따라 raw의 첫 1,000건을 개발 검증 후보로 두고 나머지를 학습 후보로 둔다.
2. 공개 valid 500건은 모델 선정용 생성 평가로 둔다. 지시문을 대소문자·공백 정규화해 묶고, 다른 context가 있어도 같은 지시문이면 분할 사이 중복을 제거한다. 공개 평가와의 중복 제외를 먼저 적용하고, 개발 검증과 겹친 학습 문항을 제외한다. 학습·검증의 완전 중복도 제거한다.
3. 최초 GPT-2·Qwen의 네 tokenizer 모두에서 prompt 256토큰 이하, 학습·검증의 전체 prompt+응답+EOS 512토큰 이하인 같은 원본 record만 남긴다. 긴 응답을 자르지 않는다. 256은 위 전처리 코드, 512는 MiniLLM의 모델 길이와 로컬 실행 범위를 근거로 한 제한이다. 보수적인 공통 필터는 긴 응답의 비율을 줄이는 선택 편향을 갖는다.
4. 최종 공통 분할은 **학습 11,117건·검증 975건·선정 평가 500건**이다. 제외 사유별 수와 준비 파일 SHA-256은 protocol에 기록한다. 지시문 기준 분할 간 중복이 없음을 assert한다. 의미가 비슷한 모든 문항의 오염까지 검증한 것은 아니다.
5. GPT-2는 MiniLLM wrapper, Qwen은 native chat template에서 thinking을 끈 prompt를 사용한다. 지시·context·기준 응답의 내용은 같다. 계열 내 어휘 ID 대응은 일치함을 확인했다. 모델별 native EOS를 사용하며 어휘 일치가 모든 특수 토큰 설정의 일치를 뜻하지는 않는다.
6. Prompt와 응답을 따로 tokenize해 연결하고 prompt label을 `-100`으로 마스킹한다. 응답과 마지막 EOS만 학습하며 padding은 손실에서 제외한다. EOS를 padding 토큰으로 사용해도 실제 EOS label이 지워지지 않는지 검사한다.

공개 평가 자료는 이전 공개-SFT 평가에서 이미 관찰한 자료다. 이후 KD 결론의 독립적인 최종 test로 부르지 않는다.

추가 모델도 위에서 확정한 **동일 원본 record ID와 전체 응답**을 사용한다. tokenizer가 달라졌다는 이유로 행을 추가 제외하거나 응답을 자르지 않는다. 기존 4개 모델은 512토큰 이하이며, 추가 모델은 원문 전체의 실제 길이를 기록하고 원래 GPT-2의 native 용량인 1,024토큰 이내인지 검사한다. 따라서 모든 tokenizer가 512토큰 이하라는 조건으로 확대 해석하지 않는다. OPT는 MiniLLM wrapper, OpenLLaMA는 [공식 권고](https://huggingface.co/openlm-research/open_llama_7b)에 따라 slow tokenizer와 BOS, Gemma는 native chat template과 end_of_turn 응답 종료를 사용한다. 생성은 각 모델의 공식 EOS도 종료 조건에 포함한다.

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

두 학습률은 각각 같은 원본에서 새 LoRA를 초기화해 학습한다. 모델마다 3개 epoch checkpoint 중 검증 손실이 가장 낮은 것을 고르고, 두 학습률의 선택 checkpoint를 같은 검증 기준으로 비교한다. 총 10모델 × 2학습률 = 20개 실행이며 완료되지 않은 실행을 결과로 간주하지 않는다.

검증 손실은 microbatch 1에서 응답 토큰 NLL을 문항별 평균한 뒤 전체 문항에 평균한 값이다. 같은 모델의 checkpoint·학습률 선택에 사용한다. 서로 다른 tokenizer의 NLL을 직접 성능 순위로 비교하지 않는다.

3 epoch는 초기화 단계에 근거한 제한이다. 원문의 SFT baseline은 크기에 따라 10/20 epoch와 별도 학습률·batch 탐색 및 ROUGE-L checkpoint 선택을 사용한다. 따라서 이번 결과를 MiniLLM SFT baseline의 완전 재현이나 각 모델의 최상 성능으로 표시하지 않는다. 양자화·LoRA·공통 필터·prompt·scheduler 변경도 재현 차이로 남긴다.

## 공통 평가와 시각화

**평가 역할:** 2026-10-16 사용자 결정에 따라 Dolly·Self-Instruct·S-NI (11_)·Vicuna의 생성 성능을 주 평가로 삼는다. SST-2·BoolQ는 label 편향과 확률 품질을 살피는 보조 진단이다. [MiniLLM §3.1·3.3](https://arxiv.org/html/2306.08543v4)은 다섯 instruction benchmark를 주 평가로 쓰고 SST-2·BoolQ는 LLaMA-7B 기반 calibration 분석에 사용한다. [CaRE-KD §4](https://arxiv.org/html/2609.36734v1)은 네 공통 생성 benchmark에서 평가한다. 두 분류 점수만으로 교사·학생을 선정하지 않으며, 우리 Brier score를 논문의 ECE 재현으로 표시하지 않는다.

교사가 학생보다 반드시 몇 점 높아야 한다는 고정 임계값은 두지 않는다. 목표 생성 과제의 동일 문항에서 관측한 차이·오류 유형·변동을 근거로 지도 유익성을 판단한다. 현재 한 개 생성 seed로 seed 간 변동이나 우위의 통계적 확실성을 결론내리지 않는다. [생성 평가 계획](GENERATION_EVALUATION.md)의 세 추가 자료를 공식 배포 revision으로 고정했고 모든 tokenizer의 같은 원문·길이·학습/개발 instruction 중복 감사를 통과했다. 학습 파라미터와 기존 평가 이력은 유지하며, 모델 선정 완료에는 네 생성 benchmark의 완료도 요구한다. 실제 완료 범위는 [생성 주 평가](../results/generation_benchmarks_v1/REPORT.md)를 따른다.

| 평가 | 공통 자료와 지표 |
|---|---|
| 생성 주 평가 4종 | Dolly 500·Self-Instruct 242·S-NI (11_) 1,694·Vicuna 80문항, 공통 seed 10. ROUGE-L·문항별 교사−학생 차이. 다중 정답의 최고 점수 사용 |
| SST-2 | validation 872건, accuracy·macro-F1·balanced accuracy·weighted-F1·MCC·클래스별 recall·혼동행렬 |
| BoolQ | validation 중 공통 길이 기준을 통과한 3,269건, 같은 분류 지표. 원본 row 3154 한 건은 GPT-2의 1,024토큰 용량을 넘으므로 전 모델 공통 제외 |
| Dolly 생성 | 모델 선정 단계: 500문항 × 공통 생성 seed 10 한 개. ROUGE-1/2/L의 문항 평균, 생성 길이·상한 도달률·빈 응답률 |

**반복 수 결정:** 모델 선정 단계에서는 10개 모델 모두 같은 seed 10으로 생성 평가한다. Dolly는 500 × 1 × 10 = 5,000개이고 세 추가 자료를 포함한 총 평가 응답은 2,516 × 1 × 10 = 25,160개다. 기존 Dolly 응답은 재사용한다. 10은 MiniLLM의 seed 목록 중 첫 값을 공통 조건으로 고정한 것이며 최적 seed라는 의미는 아니다. 최종 선정 모델의 본 실험에서는 MiniLLM을 따라 seed 10/20/30/40/50의 5개 생성 반복을 사용한다. 두 단계의 반복 수는 학습 반복 수와 별개다. 현재 실행은 모델 선정까지만 수행하며 본 실험의 추가 학습·방법 비교는 이후 설계에 따른다.

분류는 같은 지시 내용과 label 후보의 조건부 sequence log probability로 판정한다. 자유 응답을 임의 파싱한 정답률과 구분한다. Brier score는 두 label 후보 사이에서 정규화한 확률임을 명시한다. 자료 revision·제외 행·파일 해시도 평가 명세에 남긴다.

생성은 temperature=1, top-k=0, top-p=1, no-repeat-ngram=6과 생성 seed를 [MiniLLM 평가 코드](https://github.com/microsoft/LMOps/tree/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm)·부록 B.2에서 가져온다. **최대 새 토큰 256과 microbatch 1은 로컬 변경값**이다. 256은 전체 길이 상한 512에서 공통 최대 prompt 256을 뺀 보수적 응답 예산이다. 원본은 512에서 실제 padded prompt 길이를 빼므로 짧은 prompt에 더 긴 응답을 허용한다. 이번 고정 상한은 긴 응답의 점수를 낮출 수 있어 상한 도달률도 함께 보고한다. Batch 변경 때문에 seed가 같아도 원문과 생성 문장의 완전 일치를 요구하지 않는다.

위 256토큰 생성 예산은 최초 네 tokenizer를 기준으로 정한 값으로 10개 모델에 동일하게 유지한다. 추가 tokenizer의 전체 원문 prompt는 256토큰을 넘을 수 있으므로 총 길이가 모두 512 이하라는 뜻은 아니다. 각 모델의 native context 내에서 입력 원문과 동일한 새 토큰 예산을 유지한다.

그림에는 평가가 완료된 모델의 SST-2·BoolQ accuracy/macro-F1, 단일 생성 seed의 Dolly ROUGE-L, 학습·검증 손실 곡선을 별도로 제시한다. 현재 단일 seed에서는 생성 seed 간 표준편차를 추정할 수 없으므로 JSON에 null로 남기고 표·그래프에 오차막대나 ±0을 붙이지 않는다. 본 실험의 5개 생성 반복에서 평균·표본 표준편차를 보고한다. 점수가 없는 모델은 공란으로 두며 이전 공개 SFT 모델 점수로 채우지 않는다. Dolly ROUGE를 accuracy로 해석하지 않는다.

## 실행과 상태

공통 파라미터 전체 표·현재 장비 사양·장비별 변경 범위·모델 하나씩 실행하는 방법은 [공통 실험 실행 지침](EXPERIMENT_GUIDE.md)을 따른다. 현재 microbatch·누적 횟수는 고정이며, 문서의 장비별 최적화 절차가 자동 적용된 상태는 아니다.

현재 환경에서 다음 순서로 실행한다. Python은 Qwen3.5를 지원하는 transformers가 설치된 환경이어야 한다.

```powershell
python -m pip install --target tmp/dolly_sft_additional_v1/deps sentencepiece==0.2.2 protobuf==7.36.2
python -B scripts/dolly_study.py fetch
python -B scripts/run_dolly_sft.py
```

위 두 tokenizer 의존성은 현재 프로젝트의 `tmp` 안에 설치되어 있다. 추가 모델 protocol에는 해당 버전도 기록한다. 기존 학습 환경 전체를 바꾸지 않고, 새 환경에서 실행할 때만 이 설치 단계를 수행한다.

기존 데이터·점검 파일은 준비되어 있으므로 학습 중 `dolly_sft.py prepare`를 다시 실행하지 않는다. 추가 모델 파일 준비 후 worker는 **모델별 사전 점검 → 두 학습률 SFT → MD 갱신 → 공통 평가 → MD·그림 갱신 → 다음 모델**을 진행한다. 사용자 요청으로 예약된 후속 확인은 삭제했으며 활성 대화에서 epoch 완료 단위로 진행률을 보고한다. step 단위 중간 보고는 생략한다. 모델별 SFT·평가 완료와 실행 실패는 별도로 알린다. 별도 Python worker의 학습·평가·MD 갱신 순서는 유지한다. 실행 상태는 `results/dolly_sft_v1/worker_status.json`, 상세 로그는 `tmp/dolly_sft_v1/worker.log`에 남긴다. 실패 시 다음 단계로 진행하지 않고 실패 상태를 저장한다. 같은 worker의 중복 실행은 파일 잠금으로 막는다.

사전 점검은 실제 가중치 갱신·유한 gradient·EOS/padding 마스크·adapter 저장/재적재 logits 일치·cache 생성·16건 누적 Trainer 갱신·검증·optimizer checkpoint 저장을 확인한다. 일회성 점검 점수를 본 SFT 성능으로 보고하지 않는다. Trainer 일회성 점검에만 warmup=0을 사용해 실제 한 번의 가중치 변화를 확인하며, 본 학습은 위 3% warmup을 유지한다.

공개 요약은 `results/dolly_sft_v1/`, 대형 checkpoint·예측·생성 응답과 로그는 Git에서 제외한 `tmp/dolly_sft_v1/`에 저장한다. 학습은 epoch checkpoint에서 재개하며 완료한 후보는 재학습하지 않는다. 생성은 완료 seed 단위로 재개한다. CUDA의 Qwen 고속 커널이 없는 현재 Windows 환경에서는 정확한 PyTorch 대체 구현이 사용되므로 속도는 실측 기록을 따른다. 짧은 점검 통과가 장시간 안정성을 보장하지는 않는다.
