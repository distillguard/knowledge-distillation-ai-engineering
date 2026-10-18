# 공통 실험 조건과 장비별 실행 지침

갱신일: 2026-10-18. 적용 범위: Dolly 기반 교사·학생 후보의 독립적인 LoRA SFT와 모델 선정 평가.

목적은 컴퓨터나 실행 담당자가 달라도 같은 자료·학습량·선택 기준을 사용하고, 장비에 맞춘 실행 차이는 별도로 기록하는 것이다. 이후 KD 방법 비교는 선정한 교사·학생 조합에서 수행한다. 이 문서는 현재 코드의 조건과 향후 장비 최적화 절차를 구분한다.

## 명세와 작업 범위

| 기준 자료 | 역할 |
|---|---|
| [기존 학습 protocol](../results/dolly_sft_v1/protocol.json) | 공통 데이터 분할·LoRA·학습률·optimizer·학습량·해시의 기준 |
| [추가 모델 registry](../results/dolly_sft_additional_v1/registry.json) | OPT·OpenLLaMA·Gemma의 공식 모델 ID·고정 revision |
| `results/dolly_sft_additional_v1/<model>/protocol.json` | 추가 모델의 실제 정밀도·같은 원본 분할·부모 명세·실행 소스 해시. GPU 점검 시 생성 |
| [상세 선정 계획](DOLLY_SFT_MODEL_SELECTION.md) | 논문·공식 구현의 설정 근거와 원문 재현 차이 |
| [진행 기록](SFT_PROGRESS.md) | 완료한 학습률 후보·선택 결과·평가 상태 |

위 JSON은 실행 기록이다. JSON 숫자만 고쳐도 코드에 적용되는 설정 파일은 아니다. 실행 조건을 바꿀 때는 코드·새 protocol·사전 점검·결과 경로를 함께 맞추며, 기존 checkpoint가 있는 명세를 덮어쓰지 않는다.

| 계열 | 교사 후보 | 학생 후보 | 실행 ID |
|---|---|---|---|
| GPT-2 | XL | base | `gpt2_xl`, `gpt2_base` |
| OPT | 2.7B | 125M | `opt_27b`, `opt_125m` |
| OpenLLaMA | 7B | 3B | `openllama_7b`, `openllama_3b` |
| Gemma-2 | 9B-IT | 2B-IT | `gemma2_9b_it`, `gemma2_2b_it` |

4개 계열은 교사·학생을 합쳐 **8개 모델**이다. 같은 seed에서 학습률 후보 두 개를 독립적으로 학습하므로 총 16개 학습 실행이다. 모델 크기만으로 교사의 우월성을 가정하지 않는다. Gemma-IT의 기존 post-training과 양자화 차이도 해석에 포함한다.

## 양자화 조건의 역할 분담

2026-10-16 범위 축소: Qwen을 제외한 8개 모델의 예비 SFT와 Dolly 500문항 생성 평가를 진행한다. 외부 3종·새 보조 분류 평가는 후보 축소 후 수행한다. 팀원의 비양자화 조건과 비교할 때도 같은 활성 모델·자료·학습 조건을 맞춘다. 모델별 학습량·학습률·seed와 기존 protocol은 변경하지 않는다.

| 구분 | 조건과 상태 |
|---|---|
| 현재 컴퓨터 | 기존 정밀도 정책 유지. GPT-2 두 모델은 BF16. 추가 모델은 기존 전용 VRAM 점검에 따라 BF16/NF4 선택 |
| 팀원 측 | 기반 가중치를 양자화하지 않는 실행. 정확한 BF16/FP16/FP32 형식·장비·라이브러리·실행 결과는 아직 확인하지 않음 |
| 비교 대상 | 같은 모델에서 실제 NF4 학습 결과와 비양자화 학습 결과가 모두 확보된 경우. 현재 실행 전체를 일괄 4비트 실험으로 부르지 않음 |
| 결과 보관 | 외부 요약을 받으면 `results/external_nonquantized/<run_id>/`처럼 별도 경로에 보관. 현재 selected·평가 JSON을 덮어쓰지 않음. 외부 가중치·응답 원문은 `tmp/`에 보관 |

팀원 측도 공식 모델 ID·revision, 같은 원본 record ID·분할·전체 응답, tokenizer/template/EOS, LoRA 구성, optimizer·scheduler·유효 batch 16, 학습률 후보·학습량·학습 seed, 평가 문항·생성 seed를 기록한다. 비양자화라는 이유로 전체 가중치 미세조정을 사용하면 현재 LoRA 조건과 함께 달라진다.

각 조건에서 검증으로 선택한 최종 결과의 비교와, 동일 LR·학습량의 결과를 맞춘 양자화 비교를 구분한다. 선택된 LR·epoch가 다르거나 장비·연산 정밀도가 다르면 그 차이를 함께 보고하며, 관측한 차이를 양자화 하나의 효과로 단정하지 않는다. 다른 GPU의 경과 시간을 양자화만의 속도 효과로 해석하지 않는다. 현재 NF4는 학습 중 기반 가중치에 적용하므로, 비양자화 SFT 완료 후 추론 시에만 양자화하는 실험과 구분한다.

## 공통 학습 파라미터

| 항목 | 현재 공통 조건 |
|---|---|
| 데이터 | `MiniLLM/dolly`, revision `d131636138be0ff3fd0901c12c1739844d337279` |
| 원본 분할 | 학습 11,117건·개발 검증 975건·선정 평가 500건. 모든 모델에서 같은 순서의 원본 record ID |
| 전처리 | 지시문 기준 분할 중복 제거, 전체 응답 보존, prompt label·padding은 `-100`, 응답과 마지막 EOS만 손실 대상 |
| 입력 형식 | 같은 instruction/context. 모델별 검증된 native template·BOS·EOS 적용 |
| 길이 | 최초 GPT-2·Qwen은 prompt ≤256·학습 전체 ≤512. 추가 모델은 같은 원문을 재토큰화하고 전체 ≤1,024 및 native 용량 확인. 응답 자르기·추가 행 제외 없음 |
| 초기 가중치 | 고정한 공식 공개 revision에서 매 학습률 후보를 새로 초기화. 기존 연구 adapter 재사용 없음 |
| 학습 방식 | 동결한 기반 모델 + LoRA. 전체 가중치 미세조정 아님 |
| LoRA | `r=64`, `lora_alpha=16`, `lora_dropout=0.05`, `target_modules=all-linear`, `bias=none`, `task_type=CAUSAL_LM` |
| 학습률 후보 | `1e-4`, `2e-4`. 같은 모델 안에서 검증 NLL로 선택 |
| 학습량 | 학습률 후보마다 3epoch 상한. 매 epoch 검증·저장 |
| 유효 batch | 단일 GPU 기준 16문항, 마지막 불완전 batch 유지 |
| 학습 seed | `seed=0`, `data_seed=0` |
| Optimizer | `adamw_torch`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0` |
| Gradient clipping | `max_grad_norm=0.3` |
| Scheduler | `constant_with_warmup`, 전체 optimizer step의 3% warmup |
| 선택 기준 | 모델·학습률별 최저 개발 검증 NLL checkpoint → 두 학습률 후보 중 최저 NLL adapter |
| NLL 해석 | 현재 microbatch 1 평가의 문항별 응답 토큰 평균. tokenizer가 다른 모델 사이의 성능 순위로 사용하지 않음 |

LoRA와 학습률 후보는 [QLoRA 공식 13B 스크립트](https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/scripts/finetune_guanaco_13b.sh)·[33B 스크립트](https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/scripts/finetune_guanaco_33b.sh), 3epoch는 [MiniLLM 부록 B.1의 초기화 단계](https://arxiv.org/html/2306.08543v4#A2.SS1)에 근거한다. 이 조합이 모든 후보의 최적 설정이라는 의미는 아니다. 원문 SFT baseline의 완전 재현과 구분한다.

## 공통 평가 조건

| 항목 | 현재 조건 |
|---|---|
| SST-2 | 같은 validation 872건, label sequence의 조건부 log probability로 판정 |
| BoolQ | 같은 validation 3,269건. 원본 row 3154는 기존 공통 길이 검사에서 제외 |
| 분류 지표 | accuracy, macro-F1, balanced accuracy, weighted-F1, MCC, 클래스별 recall, 혼동행렬, 두 label 사이에서 정규화한 Brier score |
| 생성 주 평가 | Dolly 500·Self-Instruct 242·S-NI 1,694·Vicuna 80, 예비는 Dolly 500문항 × 8모델 = 4,000응답. 외부 3종은 후보 축소 후 적용하며 전체 적용 후보는 모델당 2,516응답. 공통 생성 seed `10` 한 개 |
| 생성 파라미터 | `do_sample=True`, `temperature=1`, `top_k=0`, `top_p=1`, `no_repeat_ngram_size=6`, `max_new_tokens=256`, `use_cache=True` |
| 생성 종료 | 모델별 tokenizer EOS와 공식 generation config의 EOS. 실제 목록 기록 |
| 현재 생성 지표 | ROUGE-1/2/L F1 ×100, 정규화 exact match, 생성 길이, 상한 도달률, 빈 응답률 |
| 보완 지표 | Dolly 원본/SFT 각각 8모델의 METEOR·BERTScore·Distinct-4 채점 완료. 저장 응답 재사용, 별도 채점 명세·결과. 후보 분류에는 ECE 추가 |
| 반복 해석 | 생성 seed 간 표준편차는 `null`. 오차막대·±0 없음 |
| 최종 선정 후 본 실험 | 생성 seed `10/20/30/40/50` 예정. 학습 seed 5회 반복을 뜻하지 않음 |

생성의 temperature는 sampling 조건이며 KD 분포의 온도 T와 다르다. KD의 T·gate·MC dropout 횟수·Revival 기준은 이 SFT 단계의 설정값이 아니다. 이후 문헌 근거와 검증 절차로 별도 명세한다. 현재 관찰한 선정 평가 자료를 독립적인 최종 test로 부르지 않는다.

검증 NLL은 각 모델의 체크포인트 선택이고 동일 Dolly의 원본/SFT 비교·보완 채점 후 유망 후보를 추린다. 해당 후보의 외부 평가·보조 진단·오류 분석·KD 실행 점검으로 최종 조합을 선정한다. 선정 전 추가 SFT 탐색은 시작하지 않는다. MiniLLM의 전체 SFT 탐색·학생 초기화는 선정한 한 조합의 본 실험 설계로 남긴다. [선정 단계별 완료 기준과 원본 평가 명령](GENERATION_EVALUATION.md#조합-선정까지의-단계와-완료-기준)을 따른다. 원본 8모델의 Dolly 기준선을 완료했으며 SFT 전후 변화는 [비교표](../results/model_selection_v1/REPORT.md)에 반영한다. 공개 SFT checkpoint로 원본을 대체하지 않는다.

보완 채점은 `run_supplementary_metrics.py prepare`로 자산을 준비하고, 원본/SFT 응답을 모두 확보한 뒤 `run_supplementary_metrics.py score`로 실행한다. [전용 환경과 산출물](GENERATION_EVALUATION.md#보완-채점의-전용-환경과-산출물)의 설치·모델별 명령을 따른다. 별도 자식 프로세스의 Transformers 4.46.3·BERTScore slow tokenizer를 사용하며 기존 SFT 환경은 유지한다. 보완 채점도 공통 GPU 잠금을 사용한다.

## 현재 장비와 실행 환경

2026-10-15에 확인한 현재 실행 장비다. 다른 컴퓨터에 이 사양을 적용했다고 기록하지 않는다.

| 항목 | 확인값 |
|---|---|
| CPU | AMD Ryzen 5 7500F, 6 cores |
| RAM | 31.62 GiB |
| GPU | NVIDIA GeForce RTX 5060 Ti, 전용 VRAM 16,311 MiB |
| OS / GPU driver | Windows / NVIDIA 610.88 |
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 |
| Transformers / PEFT | 5.19.0 / 0.21.2 |
| bitsandbytes / Accelerate | 0.50.2 / 1.15.0 |
| datasets | 5.1.0 |
| 추가 tokenizer 의존성 | sentencepiece 0.2.2, protobuf 7.36.2. 프로젝트 `tmp/dolly_sft_additional_v1/deps` 안에 설치 |

현재 직접 의존성 버전은 [requirements-sft.txt](../requirements-sft.txt)에 정리했다. PyTorch의 CUDA wheel은 실제 GPU·driver와 맞춰 설치하고 현재 검증된 wheel은 `2.11.0+cu128`이다. 기존 `requirements-eval.txt`는 과거 공개 SFT 평가용으로 일부 버전이 다르며, 현재 8개 모델 학습 환경과 혼용하지 않는다. 현재 학습 중인 환경은 재설치하지 않았다.

현재 실행은 BF16 계산·FP32 LoRA와 정규화, microbatch 1·gradient accumulation 16, gradient checkpointing 켜기, TF32 끄기, dataloader workers 0을 사용한다. Qwen은 실행 대상에서 제외했으며 모델별 장시간 속도는 실제 로그로 확인한다.

## 장비에 따른 조정 범위

| 항목 | 조정 원칙 | 현재 구현 상태 |
|---|---|---|
| 모델 적재 정밀도 | 기존 GPT-2 두 모델은 BF16. 추가 6개는 BF16 가중치가 물리 VRAM을 넘거나 실제 갱신·저장 점검이 전용 VRAM에 맞지 않으면 NF4 | 추가 모델에서 자동 판단·기록. Gemma9B는 BF16 가중치만으로 현재 VRAM 초과 |
| Microbatch·누적 횟수 | 단일 GPU에서 두 값의 곱 16 유지. 후보 `1×16`, `2×8`, `4×4`, `8×2`, `16×1` 중 실제 처리량·메모리로 선택 | 현재 `1×16` 고정. 다른 조합의 자동 탐색은 미구현 |
| Gradient checkpointing | VRAM에 여유가 있으면 해제 상태도 실측. 학습량·손실 의미 유지 확인 | 현재 모든 모델에서 켜짐. 자동 해제 미구현 |
| 평가 batch | 공통 문항·전체 입력·동일 지표 유지. 문항별 NLL 평균 방식과 생성 난수 처리 변화 확인 | 현재 분류·검증·생성 모두 microbatch 1 |
| Data loader·kernel | CPU·RAM·GPU 환경에 맞춰 처리량을 측정하고 버전·변경값 기록 | 현재 worker 0·검증된 기본 backend. 자동 탐색 미구현 |

Batch 크기와 accumulation의 관계는 [Transformers 문서](https://huggingface.co/docs/transformers/v5.12.0/grad_accumulation), activation 재계산의 메모리·속도 교환은 [gradient checkpointing 문서](https://huggingface.co/docs/transformers/v5.19.0/grad_checkpointing)를 따른다. 문서의 일반 설명을 현재 GPU의 실측 향상률로 쓰지 않는다.

양자화·연산 정밀도는 결과에도 영향을 주는 조건이다. 장비 때문에 바꿨더라도 실제 형식과 이유를 함께 기록하며 BF16 결과와 무조건 동등하다고 간주하지 않는다. BF16을 지원하지 않는 GPU에서 FP16으로 자동 대체하지 않고 별도 실행 조건으로 명세한다.

새 장비 또는 새 실행 profile의 사전 점검은 다음 순서다.

1. GPU·전용 여유 VRAM·RAM·라이브러리 버전을 확인한다. 같은 GPU에서 다른 학습을 겹쳐 실행하지 않는다.
2. 동일 원본 ID·파일 해시·template·EOS·마스킹·실제 최장 입력을 확인한다. 기존 장비의 통과 JSON을 새 장비의 점검 결과로 복사하지 않는다.
3. 폐기 가능한 adapter로 실제 forward/backward·optimizer update·검증·저장/재적재·생성을 검사한다. 배치를 늘릴 때는 최장 입력들이 한 batch에 들어오는 경우도 점검한다.
4. 유효 batch 16과 손실 정규화·문항 가중치가 유지되는지 확인한다. 같은 유효 batch라는 이유만으로 gradient의 완전 일치를 가정하지 않는다.
5. 적재 시간을 제외한 안정 구간의 처리량과 peak VRAM을 측정한다. OOM·전용 메모리 초과를 확인한 경우에만 메모리 대응을 적용하며 다른 오류는 별도로 해결한다.
6. 선택값·실측값·정밀도·소스 해시를 새 profile/protocol에 남기고 본 학습에 적용한다. 현재 저장된 학습 중간에 batch·누적·scheduler를 바꿔 이어 붙이지 않는다.

데이터 수·epoch·학습률 후보·LoRA rank·생성 길이·평가 문항을 줄이는 것은 별도의 연구 설계 변경이다. 장비 조정으로 숨기지 않는다. 예비 Dolly 평가 후 후보를 좁히는 단계 구분을 적용했으며 원문의 전체 SFT 탐색은 아직 실행하지 않았다.

## VSCode와 모델별 실행

저장소 `knowledge-distillation-ai-engineering` 폴더 자체를 VSCode로 열고 `Python: Select Interpreter`에서 위 패키지가 설치된 환경을 선택한다. `.vscode/tasks.json`의 **Dolly: 모델별 학습·평가** 작업에서 실행 ID 하나 또는 `all`을 고른다. 작업은 선택한 Python interpreter를 사용한다. [VSCode 공식 Tasks 문서](https://code.visualstudio.com/docs/debugtest/tasks).

모델별 공통 순서는 **사전 점검 → 두 LR의 3 epoch 예비 SFT → 검증 NLL로 adapter 선택 → 문서 → Dolly 500 생성 → 점수·그림·문서 → 다음 모델**이다. 외부 3종과 새 SST-2·BoolQ는 후보 축소 후 진행한다. 추가 6개 모델의 GPU 점검·정밀도 선택은 기존 실행기가 수행하며 GPT-2 두 모델은 기존 점검·완료 결과를 재사용한다.

아래 인자를 `python -u -B -X utf8 scripts/run_dolly_sft.py` 뒤에 붙인다. 모든 모델에 같은 학습·평가 계약을 적용한다.

| 모델·역할 | 실행 인자 | 기반 가중치 조건 |
|---|---|---|
| GPT-2 base·학생 | `--model gpt2_base` | BF16 |
| GPT-2 XL·교사 | `--model gpt2_xl` | BF16 |
| OPT-125M·학생 | `--model opt_125m` | BF16부터 실제 점검, 맞지 않으면 NF4 |
| OPT-2.7B·교사 | `--model opt_27b` | BF16부터 실제 점검, 맞지 않으면 NF4 |
| OpenLLaMA-3B·학생 | `--model openllama_3b` | BF16부터 실제 점검, 맞지 않으면 NF4 |
| OpenLLaMA-7B·교사 | `--model openllama_7b` | BF16부터 실제 점검, 맞지 않으면 NF4 |
| Gemma-2-2B-IT·학생 | `--model gemma2_2b_it` | BF16부터 실제 점검, 맞지 않으면 NF4 |
| Gemma-2-9B-IT·교사 | `--model gemma2_9b_it` | 현재 장비는 BF16 가중치만으로 VRAM 초과하므로 NF4 |

현재 장비에서 같은 작업을 PowerShell로 실행하는 예시는 다음과 같다. 다른 장비에서는 Python 경로를 해당 환경에 맞춘다. 모델·데이터 준비와 해당 장비의 사전 점검이 완료된 작업 폴더를 전제로 한다. Gemma의 최초 학습 전에는 아래 분할 가중치 인덱스 보완도 수행한다.

```powershell
$StudyPython = 'C:\lightjudge\.venv\Scripts\python.exe'

# 공통 환경 확인
& $StudyPython -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

# 생성 평가 자료 준비: 첫 실행 전, 고정 revision 다운로드와 CPU 감사
& $StudyPython -B -X utf8 scripts/prepare_generation_benchmarks.py fetch
& $StudyPython -B -X utf8 scripts/prepare_generation_benchmarks.py prepare

# GPT-2 base 하나: 두 LR 예비 학습 → MD → Dolly 500 생성 → MD·그림
& $StudyPython -u -B -X utf8 scripts/run_dolly_sft.py --model gpt2_base

# OPT-2.7B 하나: 추가 모델 GPU 점검부터 같은 순서
& $StudyPython -u -B -X utf8 scripts/run_dolly_sft.py --model opt_27b

# 전체 8개 모델 순차 실행·재개. 위 개별 명령과 함께 실행하지 않음
& $StudyPython -u -B -X utf8 scripts/run_dolly_sft.py --model all
```

각 명령은 독립적인 선택지다. 같은 worker 잠금을 사용하므로 기존 전체 실행 중 모델별 실행을 추가하면 학습을 시작하기 전에 종료된다. 저수준 `dolly_sft.py train`을 직접 실행하면 이 worker 잠금을 우회하므로 운영에는 위 실행기를 사용한다. 2026-10-19에 활성 8모델의 예비 SFT·Dolly 생성 worker는 정상 종료했다. 원본/SFT 비교 이후의 후속 평가는 같은 잠금으로 순차 실행하며, 이미 완료한 SFT를 재시작하지 않는다.

단일 모델 실행의 worker 상태가 `completed`라는 것은 `requested_models`에 기록된 요청만 끝났다는 뜻이다. 예비 완료는 활성 8개 모델의 selected와 Dolly 500문항의 검증된 생성 결과로 판단한다. 외부 3종·보조 분류의 전체 완료는 예비 완료 요건이 아니다. worker는 로컬 SentencePiece 의존 경로를 자식 프로세스의 `PYTHONPATH`에 먼저 넣어 Transformers가 tokenizer 의존성을 올바르게 감지하게 한다.

### Gemma 분할 가중치 인덱스 보완

고정된 준비 코드의 다운로드 목록에는 `model.safetensors.index.json`이 빠져 있다. Gemma의 가중치 shard가 모두 있어도 이 인덱스가 없으면 모델 적재가 실패한다. 기존 실행의 소스 지문을 보존하면서 아래 명령으로 동일 revision의 메타데이터를 보완한다. 모델 사용 조건 동의와 로컬 Hugging Face 인증이 필요하다. 토큰은 출력하거나 저장소에 기록하지 않는다.

```powershell
@'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import dolly_study as study
from huggingface_hub import hf_hub_download, get_token
token = get_token()
standard = Path.home() / '.cache/huggingface/token'
if token is None and standard.is_file():
    token = standard.read_text(encoding='utf-8').strip()
registry = study.read(study.REGISTRY)
for name in ('gemma2_2b_it', 'gemma2_9b_it'):
    entry = registry[name]
    path = Path(hf_hub_download(entry['id'], 'model.safetensors.index.json',
                revision=entry['revision'], cache_dir=str(study.HUB), token=token))
    shards = set(json.loads(path.read_text(encoding='utf-8'))['weight_map'].values())
    assert shards == set(entry['weight_files'])
    assert all((path.parent / shard).is_file() for shard in shards)
    print(name, 'index and shards verified')
'@ | & $StudyPython -B -X utf8 -
```

2026-10-17에 이 절차로 2B의 2개 shard·288개 tensor, 9B의 4개 shard·464개 tensor 연결을 확인하고 worker를 재개했다. 기존 가중치와 학습·평가 명세는 그대로 사용한다.

## 모델별 필수 결과물과 완료 판정

아래 `<model>`에는 표의 실행 ID를 넣는다. GPT-2와 제외한 Qwen의 보존 학습 요약 디렉터리는 `results/dolly_sft_v1/`, 추가 6개 모델은 `results/dolly_sft_additional_v1/<model>/`이다. 8개 모델의 평가 요약은 모두 `results/dolly_sft_v1/`에 모이며 파일명에 모델 ID가 들어간다. 기존 checkpoint·파일 해시를 유지하기 위해 실행 중인 결과 디렉터리는 이동하지 않는다.

| 단계 | 필수 파일 | 완료 조건 |
|---|---|---|
| 준비 | 공통 `protocol.json`, 추가 모델별 `preparation.json`·`protocol.json` | 고정 revision·동일 원본 ID·분할 수·자료 해시 일치. 추가 모델 `preparation.status=prepared` |
| GPU 사전 점검 | `preflight.json`, `trainer_preflight.json`; 추가 모델 `memory_check.json` | 해당 protocol fingerprint에서 갱신·유한 gradient·마스킹·재적재·checkpoint 저장 통과. 추가 모델 전용 VRAM 적합성 통과 |
| LR 후보 1 | 학습 요약 디렉터리의 `<model>_lr1e-04.json` | `status=completed`, 유한 검증 NLL, best checkpoint·adapter 경로 존재 |
| LR 후보 2 | 같은 위치의 `<model>_lr2e-04.json` | 위와 동일. LR 1 후보의 adapter에서 이어 학습하지 않음 |
| SFT 선택 | 같은 위치의 `<model>_selected.json` | 두 후보 모두 완료. 선택 LR·자기 모델의 검증 NLL·adapter·protocol fingerprint 기록 |
| 실제 adapter | selected JSON의 `adapter` 경로 안 `adapter_config.json`, `adapter_model.safetensors` | 파일 존재·정상 재적재. Git에 가중치를 올리지 않음 |
| 예비 평가 조건 | `results/generation_benchmarks_v1/<model>/protocol.json` | 선택 adapter·학습 명세·평가 코드 fingerprint 일치, 같은 데이터 revision·ID·생성 조건 |
| Dolly 생성 | 같은 위치의 `<model>_generation_seed10.json` | `seed=10`, `n=500`, ROUGE·생성 길이·상한·빈 응답 지표. 실제 응답 500개는 `tmp/dolly_sft_v1/evaluation/<model>_seed10_responses.json` |
| 예비 생성 주 평가 | `results/generation_benchmarks_v1/<model>/dolly.json`과 문항별 점수 CSV | Dolly 500문항 완료, protocol·adapter fingerprint·CSV hash·동일 ID 일치. SD `null` |
| 후속 외부 평가 | 같은 디렉터리의 `{self_inst,sni_11,vicuna}.json`과 문항별 점수 CSV | 후보 축소 후 수행. 이미 완료한 결과 보존, 예비 전체 완료 요건에는 미포함 |
| 기존 Dolly·분류 보조 평가 | `results/dolly_sft_v1/<model>_evaluation.json` | 기존 완료 결과 보존. 새 분류는 후보 축소 후 수행하며 예비 완료 요건에는 미포함 |
| 표·그림·문서 | `scores.csv`, `classification.png`, `dolly_rouge.png`, `development_nll.png`, `REPORT.md`, `docs/SFT_PROGRESS.md` | 완료한 모델의 실제 결과만 포함. 미완료 모델을 0점으로 채우지 않음 |

단일 LR 후보의 완료, 진행 JSON의 `step=max_steps`, checkpoint 파일 생성만으로 모델 SFT 전체 완료를 선언하지 않는다. 두 LR 후보를 검증해 selected JSON을 만든 시점에 **SFT 완료**, 공통 평가와 지표·fingerprint 검증까지 마친 시점에 **평가 완료**를 보고한다.

모델별 SFT 완료 시 선택 LR·자기 모델의 NLL·실제 정밀도를, 예비 생성 완료 시 Dolly 점수를 보고한다. 외부·보조 결과는 기존 완료 자료와 후보 후속 평가로 구분한다. 고정 점수 차이를 교사 자격 기준으로 쓰지 않는다. 자료·전처리·우선순위·생성 재개는 [생성 평가 계획](GENERATION_EVALUATION.md)을 따른다. 과거 공개 SFT 모델의 점수로 이번 직접 학습 결과를 채우지 않는다. 실패한 모델은 원인과 마지막 정상 checkpoint를 남기고 결과 수치를 만들지 않는다.

## 재개와 산출물

실패 시 실제 실행 프로세스·마지막 로그·checkpoint를 먼저 확인한다. 살아 있는 실행을 중복 시작하지 않는다. 같은 조건의 학습은 마지막 epoch checkpoint에서 재개하고, 완료한 학습률 후보는 건너뛴다. 생성은 완료된 seed 단위로 재사용한다. 조건을 변경한 결과는 이전 완료 JSON과 합치지 않는다.

| 위치 | 저장 내용 | Git |
|---|---|---|
| `results/dolly_sft_v1/` | 기존 모델 학습 요약·선택, 8개 모델 평가·표·그림 | 검증한 완료 요약만 반영 |
| `results/dolly_sft_additional_v1/<model>/` | 추가 모델 준비·protocol·점검·학습·선택 요약 | 검증한 요약만 반영 |
| `results/generation_benchmarks_v1/` | 생성 주 평가 protocol·문항별 점수·집계·교사−학생 차이·그림 | 검증한 실제 결과만 반영. 원문·응답 제외 |
| `docs/SFT_PROGRESS.md`와 연구 MD | 모델별 완료 상태와 실제 선택 근거 | 반영 |
| `tmp/` | 원자료·토큰화 자료·모델 cache·checkpoint·생성 원문·로그 | 제외 |
| 토큰·인증 파일 | 로컬 인증 | 제외. 명령 인자나 문서에 토큰 입력 금지 |

모델별 SFT·평가 완료 후 관련 MD와 검증된 결과만 commit/push한다. `git add .` 대신 대상 파일을 지정하며 진행 중 LR JSON·예시 점수·미완료 실행을 완료 성능으로 올리지 않는다.

문항별 CSV의 SHA-256은 줄바꿈을 포함한 실제 파일 바이트를 대상으로 한다. `.gitattributes`는 결과 CSV의 Git 줄바꿈 변환을 끄므로 Windows·Linux에서 같은 byte와 hash를 유지한다. 2026-10-19에 기존 Git blob의 자동 변환을 발견해 평가 당시 CSV 바이트로 바로잡았으며 수치·문항·기록된 SHA는 바꾸지 않았다. `python -B -X utf8 -m unittest discover -s tests -p test_published_artifacts.py`는 공개 결과 JSON의 CSV hash를 Git index와 현재 checkout 모두에서 검증한다.

## 후보 후속 평가의 공통 실행

SFT 후보는 `python -u -B -X utf8 scripts/evaluate_sft_candidates.py --model gpt2_base gpt2_xl opt_125m opt_27b --classification`으로 외부 생성과 보조 분류를 순차 실행한다. 완료한 동일 fingerprint 결과는 재사용한다. `--classification`을 생략하면 생성만 수행한다. 원본은 기존 실행기의 `--model <모델명> --benchmark dolly self_inst sni_11 vicuna --classification`을 후보별로 사용한다. 두 실행기 모두 같은 `worker.lock`을 사용하며 저수준 평가를 동시에 시작하지 않는다. SFT 조건·checkpoint·기존 결과 경로는 바꾸지 않는다.

분류 진단의 `python -B -X utf8 scripts/report_candidate_diagnostics.py`는 고정 source row·label·parquet와 예측 파일 hash를 확인하고 Acc·macro-F1·balanced accuracy·weighted-F1·MCC·클래스별 recall·Brier·ECE15를 재계산한다. 완료 결과만 `results/candidate_diagnostics_v1/`에 반영한다. 두 조건부 label 확률의 ECE와 전체 어휘 entropy confidence를 구분한다. 현재 기존 GPT-2 SFT 예측으로 계산한 진단이 있으며 후보 원본과 OPT의 완료 범위는 해당 보고서를 따른다.

## 증류 실행 점검의 준비와 범위

[점검 코드](../scripts/probe_kd_pair.py)는 CPU 수식·mask·dropout 복구·잠금 검사를 준비한 상태이며 실제 공동 GPU 점검은 아직 실행하지 않았다. 후보 평가의 GPU 잠금이 해제된 뒤 `python -u -B -X utf8 scripts/probe_kd_pair.py --pair gpt2` 또는 `--pair opt`를 사용한다. 같은 SFT adapter·BF16을 동시 적재하고, 동일 11,117개 준비 입력과 어휘 ID·logits 좌표를 확인한다. 최장 학습 reference와 최장 학습 prompt의 학생 생성에서 각각 한 번의 임시 LoRA 갱신을 확인하고 원본 adapter 파일 hash를 보존한다. 학습된 checkpoint는 저장하지 않는다.

생성의 최소·최대 새 토큰을 기존 상한 256으로 맞추는 것은 cache 한계 점검용 강제 조건이다. benchmark 답변이나 본 실험의 decoding 변경으로 취급하지 않는다. T=1·margin=0은 분포를 변환하지 않고 gate에 offset을 주지 않는 정의상 대조값이며 최적값 선정이 아니다. MC는 [CaRE §8.1–8.2](https://arxiv.org/html/2609.36734v1#S8.SS2)의 3회·attention dropout 0.1을 사용하고 다른 기존 dropout 비율은 유지한다. 모든 training flag·attention dropout을 복구하고, 확률의 실제 변화와 BALD를 확인한다. 이 단계의 응답 위치 pooling·단일 생성 prefix를 논문의 500문항 비퇴화 검증으로 표시하지 않는다. 전체 어휘 streaming 집계·실제 학생 backward/optimizer·생성 cache·MC의 최대 메모리와 실패 단계를 기록한다. 본 KD·온도 탐색·Revival의 quantile/거절 규칙·장기 안정성은 이 점검의 완료 범위가 아니다.
