# RTX 4090에서 Gemma-2-9B 비양자화 SFT 분담하기

이 경로는 별도 PC에서 `google/gemma-2-9b-it`의 **BF16 기반 가중치 + LoRA SFT만** 실행하고, 완료한 산출물을 원래 PC로 가져오기 위한 것이다. 기존 `run_dolly_sft.py`의 10개 모델 실행과 기존 결과는 그대로 둔다. SST-2·BoolQ·Dolly 생성 평가는 이 명령의 작업 범위에 포함하지 않는다. 학습 중 개발 검증 NLL은 checkpoint와 학습률을 선택하기 위해 계산한다.

실행기는 NF4/FP16으로 자동 전환하지 않는다. BF16 지원·실제 forward/backward·optimizer 갱신·검증·adapter 저장/재적재 등의 사전 점검에 실패하면 학습을 중단한다. **RTX 4090 24GB에서 아래 checkpointed CE 조건으로 두 LR의 3epoch SFT를 완료했다.** [실측 결과와 어댑터 다운로드](../results/external_nonquantized/gemma2_9b_bf16_4090_20261015_chunkce/README.md)에 상세 기록을 보존했다. 다른 환경에서 메모리가 부족하면 오류와 점검 기록을 확인한다. 이 실행에서 LoRA rank·입력 길이·학습량을 줄여 결과를 기존 조건으로 기록하지 않는다.

## 유지하는 실험 조건

| 항목 | 별도 PC에서의 조건 |
|---|---|
| 모델 | `google/gemma-2-9b-it`, revision `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| 데이터 | 원래 PC가 준비한 동일 원본 ID·순서·전체 응답. 학습 11,117건·개발 검증 975건·선정 평가 500건 |
| 데이터 revision | `MiniLLM/dolly@d131636138be0ff3fd0901c12c1739844d337279` |
| 입력·손실 | Gemma native chat template, 응답과 마지막 `<end_of_turn>`만 손실 대상, prompt/padding label `-100` |
| 길이 | 학습·개발 검증의 전체 입력 상한 1,024. 응답 자르기·추가 행 제외 없음 |
| 가중치 | 동결한 BF16 기반 가중치, FP32 LoRA·정규화, 양자화 없음 |
| LoRA | `r=64`, `lora_alpha=16`, `lora_dropout=0.05`, `all-linear`, `bias=none`, `CAUSAL_LM` |
| 학습률·학습량 | `1e-4`와 `2e-4`를 각각 원본 가중치에서 독립적으로 3epoch 학습 |
| Batch | GPU 1개, microbatch 1 × gradient accumulation 16, 마지막 불완전 batch 유지 |
| Seed | `seed=0`, `data_seed=0` |
| Optimizer | `adamw_torch`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0`, gradient clipping `0.3` |
| Scheduler | `constant_with_warmup`, 전체 optimizer step의 3% warmup |
| 메모리 설정 | Gradient checkpointing 켜기, TF32 끄기, dataloader workers 0 |
| 아래 4090 예시의 손실 구현 | `checkpointed-ce`: 64토큰 단위로 cross-entropy 중간 계산을 재계산. protocol에 별도 기록 |
| 선택 | 각 LR의 최저 개발 검증 NLL checkpoint → 두 LR 중 최저 NLL adapter |

정밀도와 GPU 환경은 기존 NF4 실행과 다른 조건으로 기록한다. 나중에 결과를 한 저장소에 모으더라도 NF4와 BF16 결과의 차이를 양자화만의 효과라고 단정하지 않는다. 전체 공통 조건은 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)를 따른다.

## 1. 원래 PC에서 공통 입력 묶음 만들기

두 PC에 이 기능이 포함된 동일한 저장소 코드를 준비한다. 소스 fingerprint는 CRLF를 LF로 정규화하므로 Git의 Windows 줄바꿈 변환은 허용하지만, 코드 내용 변경은 같은 실행의 재개에 허용하지 않는다. 원래 PC에서 공통 데이터 준비가 끝난 저장소 루트를 작업 폴더로 사용한다. `pack-inputs`는 이미 만들어진 원본 분할과 기준 protocol을 묶으므로 별도 PC에서 데이터를 다시 무작위 분할하지 않는다.

```powershell
python -B -X utf8 scripts/dolly_external.py pack-inputs --output tmp/gemma_inputs.zip
```

`tmp/gemma_inputs.zip`을 RTX 4090 PC로 전달한다. 이 파일에는 학습 원문과 원본 record ID가 있으므로 Git에는 올리지 않는다. 기존 checkpoint·cache·원본 모델 가중치는 입력 묶음에 포함하지 않는다.

## 2. RTX 4090 PC 환경 준비

저장소 루트에서 새 환경을 만든다. 기존 학습을 실행 중인 환경을 재설치하지 않는다. 현재 기준 환경은 Python 3.12이며, 직접 의존성 버전은 [requirements-sft.txt](../requirements-sft.txt)에 고정돼 있다.

```powershell
py -3.12 -m venv .venv
$SftPython = (Resolve-Path .venv/Scripts/python.exe).Path
& $SftPython -m pip install --upgrade pip

# GPU·driver와 호환되는 PyTorch CUDA wheel을 먼저 설치한다.
# 기존 PC에서 확인한 wheel은 torch 2.11.0+cu128이다.
# 아래는 해당 CUDA 12.8 wheel을 사용하는 예시다.
& $SftPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# PyTorch 설치 뒤 프로젝트의 고정 의존성을 적용한다.
& $SftPython -m pip install -r requirements-sft.txt
& $SftPython -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
```

마지막 명령에서 CUDA 사용과 BF16 지원이 확인되어야 한다. 해당 wheel을 사용할 수 없는 driver 환경이면 GPU·driver에 맞는 CUDA wheel을 먼저 준비한다. 이 절차는 모델 적재나 실제 학습 메모리 점검을 대신하지 않는다. 다른 GPU 작업을 종료해 전용 VRAM을 확보한다. 원본 모델 cache와 checkpoint용 디스크 공간도 준비한다.

Gemma는 Hugging Face 계정에서 [모델 이용 조건과 접근 권한](https://huggingface.co/google/gemma-2-9b-it)을 먼저 확인해야 한다. 접근 권한이 있는 계정으로 로컬 로그인한다. 토큰은 대화형 로그인에 입력하며 명령 인자·Git 파일·결과 JSON에 넣지 않는다.

```powershell
$env:HF_HOME = Join-Path (Get-Location).Path 'tmp/hf_home'
& .venv/Scripts/hf.exe auth login
```

데이터 묶음은 예를 들어 저장소의 `tmp/gemma_inputs.zip`에 둔다. 명령 도움말·묶음 생성/가져오기는 CUDA 학습 라이브러리를 불러오지 않고 사용할 수 있다.

```powershell
& $SftPython -B -X utf8 scripts/dolly_external.py --help
& $SftPython -B -X utf8 scripts/dolly_external.py prepare --help
```

## 3. 전용 실행 준비와 SFT

`run-id`는 이 실행에 고유하게 정하고 두 PC에서 같은 값을 사용한다. 다음 예시는 `gemma2_9b_bf16_4090`이다. 다른 실험 조건이나 손실 구현으로 시작할 때는 새 ID를 사용한다.

```powershell
& $SftPython -u -B -X utf8 scripts/dolly_external.py prepare --run-id gemma2_9b_bf16_4090 --inputs tmp/gemma_inputs.zip --loss-implementation checkpointed-ce

# 선택 사항: allocator의 메모리 단편화 대응. 실행 환경에 기록된다.
$env:PYTORCH_ALLOC_CONF = 'expandable_segments:True'
& $SftPython -u -B -X utf8 scripts/dolly_external.py train --run-id gemma2_9b_bf16_4090
```

다른 드라이브의 모델 cache를 사용할 경우 준비 명령에 `--cache-dir D:/hf-cache/hub`를 지정할 수 있다. 준비는 고정 revision 다운로드, 입력 묶음·record ID 검증, 동일 template의 재토큰화, 전용 protocol 생성을 수행한다. `train`은 해당 PC의 사전 점검을 통과한 뒤 두 LR을 순서대로 실행한다. 같은 PC에서 이 명령을 중복으로 시작하지 않는다.

4090 24GB에서는 기본 Transformers cross-entropy의 중간 텐서로 메모리 부족이 관측되어 위 예시에 `--loss-implementation checkpointed-ce`를 지정했다. 한 문항의 전체 입력과 label을 유지하면서 손실 계산의 중간 텐서를 64토큰 단위로 나누고 backward 때 재계산한다. 원래 causal label shift, `-100` 마스킹, 손실 합산과 Trainer의 공통 `num_items_in_batch` 분모를 유지한다. 데이터·LoRA rank·batch·누적 횟수·epoch·학습률 후보는 위 표와 같다.

옵션을 생략하면 기본 Transformers 손실 구현을 사용한다. 선택한 구현·청크 크기·기준 구현·소스 해시는 protocol의 `causal_lm_loss`와 관련 source hash에 남아 결과 묶음에도 보존된다. FP32 합산 순서가 달라질 수 있으므로 두 구현의 수학적 손실 목적이 같다는 것과 비트 단위 결과가 같다는 것을 구분한다. 이미 준비한 실행에서 손실 옵션만 바꿔 이어 학습하지 않고 새 `run-id`로 준비한다.

실행 위치는 다음과 같다.

| 위치 | 내용 | Git |
|---|---|---|
| `results/external_nonquantized/gemma2_9b_bf16_4090/` | protocol·준비/점검·실행 환경·LR별 학습 요약·선택 결과 | 완료 여부와 내용을 검증한 요약만 반영 |
| `tmp/external_nonquantized/gemma2_9b_bf16_4090/` | 원문·토큰화 자료·epoch checkpoint·선택 adapter·실행 중간 파일 | 제외 |
| `tmp/gemma_inputs.zip`, `tmp/gemma_sft.zip` | PC 간 입력/결과 전달 묶음 | 제외 |

학습이 중단되면 같은 `train --run-id ...` 명령으로 마지막 정상 epoch checkpoint에서 재개한다. 완료한 LR은 다시 학습하지 않는다. epoch 중간까지의 작업은 마지막 정상 epoch부터 다시 수행할 수 있다. 같은 ID의 checkpoint를 이어갈 때 protocol·입력·소스 조건을 바꾸지 않는다.

두 LR이 모두 완료되고 선택 adapter가 생성되어야 SFT 완료다. 한 LR의 종료나 checkpoint 생성만으로 전체 완료로 판단하지 않는다. 이 경로의 완료는 SST-2·BoolQ·Dolly 생성 평가 완료를 뜻하지 않는다.

## 4. 완료한 산출물 내보내기와 원래 PC로 가져오기

4090 PC에서 다음 명령을 실행한다.

```powershell
& $SftPython -B -X utf8 scripts/dolly_external.py export --run-id gemma2_9b_bf16_4090 --output tmp/gemma_sft.zip
```

결과 묶음에는 **각 LR에서 고른 best adapter 두 개**의 가중치·설정, 두 LR의 학습 요약과 최종 선택 근거, protocol 및 파일 해시 manifest가 담긴다. 같은 LR끼리 비교할 때도 해당 adapter를 사용할 수 있다. 두 LR의 3epoch 완료·유한 검증 NLL·선택 일치·BF16 점검 통과를 검사한 뒤 내보낸다. 기반 모델 전체 가중치와 optimizer checkpoint를 매번 전달하는 방식은 아니다. adapter를 사용하는 PC는 동일한 원본 모델 revision과 이용 권한이 필요하다.

`tmp/gemma_sft.zip`을 원래 PC로 전달하고 저장소 루트에서 가져온다.

```powershell
python -B -X utf8 scripts/dolly_external.py import --bundle tmp/gemma_sft.zip
```

가져오기는 묶음의 무결성을 검사하고 실행 ID별 외부 결과 경로에 저장한다. 이미 있는 파일과 내용이 같으면 재사용하고, 다른 내용이면 덮어쓰지 않고 중단한다. Git에 반영할 작은 JSON과 로컬에 둘 adapter를 구분한다. 기존 `dolly_sft_v1`·`dolly_sft_additional_v1`의 selected·평가 결과에 자동으로 합치거나 기존 학습을 재시작하지 않는다. 이후 평가 또는 NF4 결과와의 비교는 가져온 protocol·모델 revision·adapter 경로를 확인해 별도 단계로 진행한다.

두 adapter는 각각 아래 경로에 복원되며, `results/external_nonquantized/<run_id>/gemma2_9b_it_selected.json`의 `adapter` 값이 최종 선택 경로를 가리킨다.

```text
tmp/external_nonquantized/<run_id>/runs/gemma2_9b_it_lr1e-04/best_adapter/
tmp/external_nonquantized/<run_id>/runs/gemma2_9b_it_lr2e-04/best_adapter/
```

adapter 설정에 원래 PC의 snapshot 경로가 남을 수 있으므로 다른 PC에서는 기반 모델을 **protocol에 기록된 공식 ID·revision으로 명시적으로 적재한 뒤** adapter를 붙인다. 충분한 GPU 메모리를 확보한 환경에서 다음과 같이 재적재할 수 있다. 이 예시는 평가를 실행하지 않으며 정량 평가 명세를 대신하지 않는다.

```python
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

run_id = "gemma2_9b_bf16_4090"
out = Path("results/external_nonquantized") / run_id
protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
selected = json.loads((out / "gemma2_9b_it_selected.json").read_text(encoding="utf-8"))
base_spec = protocol["models"]["gemma2_9b_it"]

base = AutoModelForCausalLM.from_pretrained(
    base_spec["id"], revision=base_spec["revision"], dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation="sdpa",
    trust_remote_code=False, use_safetensors=True, weights_only=True,
)
for module in base.modules():
    if "norm" in type(module).__name__.lower():
        module.to(torch.float32)
model = PeftModel.from_pretrained(base, selected["adapter"], is_trainable=False).eval()
tokenizer = AutoTokenizer.from_pretrained(
    base_spec["id"], revision=base_spec["revision"], trust_remote_code=False,
)
tokenizer.eos_token = "<end_of_turn>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

`export` 묶음은 완료한 결과를 전달하는 용도다. 학습 중인 실행을 다른 PC에서 이어갈 목적이라면 optimizer·scheduler·RNG 상태가 있는 전체 epoch checkpoint와 같은 protocol·실행 환경을 별도로 보존해야 한다. 기본 재개 절차는 원래 학습을 실행한 PC에서 수행한다.

Git에 올릴 때는 외부 실행의 검증한 요약 경로만 명시적으로 선택한다. `.gitignore`는 `tmp/`, adapter 가중치, checkpoint를 제외한다. 입력 zip·결과 zip·원문·HF 인증 정보·모델 파라미터를 Git에 추가하지 않는다.

완료한 두 어댑터를 공개 전달할 때는 Git 이력 대신 GitHub Release의 배포 ZIP을 사용한다. 원본 handoff ZIP을 그대로 담고 Gemma 약관·NOTICE·사용 조건·수정 고지를 동봉한 바깥 ZIP으로 배포한다. [완료 결과](../results/external_nonquantized/gemma2_9b_bf16_4090_20261015_chunkce/README.md)에 다운로드·체크섬·내부 ZIP 가져오기 절차가 있다.

## 검증 범위

2026-10-16에 Linux RTX 4090 24GB에서 전체 테스트 43개가 통과했다. 이 중 CUDA BF16·어휘 256,000의 손실값과 gradient를 기존 Hugging Face 구현과 비교하는 검사를 포함한다. 기본 손실 구현은 메모리 부족으로 중단됐다. `checkpointed-ce`와 `expandable_segments:True`에서는 실제 최장 입력 496토큰으로 optimizer 상태를 유지한 두 번째 누적 갱신·검증·checkpoint 저장 및 adapter 독립 재적재까지 통과했으며, peak allocated는 약 21.60GiB였다.

두 LR의 본 학습이 각각 3epoch·2,085 optimizer step까지 완료됐고, 각 epoch 1의 checkpoint를 개발 NLL로 선택했다. 최종 선택은 LR `1e-4`, 개발 NLL `1.3143589496612549`다. SST-2·BoolQ·Dolly 생성 평가는 수행하지 않았다. 결과 묶음의 파일별 SHA256과 완료·선택·소스 조건을 검증하고 원래 PC로 가져오기를 완료했다.
