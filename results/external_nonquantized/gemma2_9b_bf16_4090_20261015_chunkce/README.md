# Gemma-2-9B-IT 비양자화 LoRA SFT 완료 결과

RTX 4090 24GB 한 장에서 두 학습률 후보의 SFT를 각각 3epoch·2,085 optimizer step까지 완료했다. 개발 검증 NLL로 선택한 최종 산출물은 **LR `1e-4`의 epoch 1 어댑터**다. 이 실행은 SFT와 개발 검증 선택만 포함하며, SST-2·BoolQ·Dolly 생성 평가는 수행하지 않았다.

## 결과

| LR | epoch 1 개발 NLL | epoch 2 | epoch 3 | 선택 checkpoint | 학습 루프 시간 |
|---|---:|---:|---:|---|---:|
| **`1e-4` (최종 선택)** | **1.31435895** | 1.33605289 | 1.52233243 | `checkpoint-695` | 8,939.630초 |
| `2e-4` | 1.31931043 | 1.38185918 | 1.74337435 | `checkpoint-695` | 8,905.514초 |

두 후보 모두 SFT 이전 개발 NLL은 3.01846194다. 각 후보의 최저 개발 NLL checkpoint를 복원한 후 두 후보를 비교했다. JSON의 `selected_development.epoch=3.0`은 3epoch 학습 종료 후 복원한 모델을 다시 검증한 시점이다. 선택한 가중치의 학습 시점은 `best_checkpoint`와 epoch별 로그에 따라 **epoch 1, step 695**다.

두 학습 루프의 합계는 약 4시간 57분이다. 본 학습의 최대 allocated CUDA 메모리는 두 후보 모두 약 21.60GiB였다. 이는 PyTorch가 보고한 allocated 값이며 전체 GPU 점유량과 구분한다. 개발 NLL은 같은 tokenizer·분할에서 checkpoint를 선택한 값으로, 독립적인 최종 평가 점수로 해석하지 않는다.

## 고정 조건과 재현 기록

- 실행 ID: `gemma2_9b_bf16_4090_20261015_chunkce`
- 학습 소스 commit: `163d48c9f9cbb844c596b882f010887266e1f5ee`
- 모델: `google/gemma-2-9b-it@11c9b309abf73637e4b6f9a3fa1e92e615547819`
- 데이터: `MiniLLM/dolly@d131636138be0ff3fd0901c12c1739844d337279`, 공통 학습 11,117건·개발 975건. 원본 record·순서·전체 응답과 Gemma 토큰화 해시 일치 확인.
- 동결한 BF16 기반 모델, FP32 LoRA·정규화, 양자화 없음. LoRA r64/alpha16/dropout0.05/all-linear.
- 후보마다 원본 가중치에서 독립 시작. microbatch 1 × accumulation 16, seed/data seed 0, 기존 optimizer·scheduler·입력 조건 유지.
- `checkpointed-ce`, 청크 64토큰, `expandable_segments:True`. 전체 응답·causal shift·마스킹·공통 분모를 유지하며 CE 중간 텐서를 재계산한다. 부동소수점 합산 순서는 기본 구현과 다를 수 있다.
- Protocol fingerprint: `69d62b288478c268486314d8a36e98bd0a5b1460718e89f2263881508a64e056`
- Parent protocol fingerprint: `3de5dbca6509d761d1996c062e4c3a74cbb5aaa0b3b9ea43d6247921f5e92898`

상세 조건은 [protocol.json](protocol.json), 환경은 [runtime.json](runtime.json), 학습 기록은 [LR 1e-4](gemma2_9b_it_lr1e-04.json)·[LR 2e-4](gemma2_9b_it_lr2e-04.json), 최종 선택은 [selected JSON](gemma2_9b_it_selected.json)을 따른다. 준비·적재·두 optimizer 갱신·검증·저장·독립 재적재 점검도 같은 폴더에 보존했다. 실제 Linux RTX 4090에서 전체 테스트 43개가 통과했으며, CUDA BF16 어휘 256,000의 loss·gradient 대조를 포함한다.

## 어댑터 다운로드와 가져오기

가중치는 Git 이력 대신 [GitHub Release `gemma2-9b-bf16-sft-20261016`](https://github.com/kimhrin/knowledge-distillation-ai-engineering/releases/tag/gemma2-9b-bf16-sft-20261016)에 첨부한다. **`gemma2-9b-bf16-sft-release.zip`**을 받고 같은 Release의 `.sha256`으로 확인한다. 바깥 ZIP에는 아래 파일이 있다.

- `gemma2-9b-bf16-sft-results.zip`: 검증한 원본 handoff 묶음. 각 LR의 best adapter 두 개와 결과 JSON·manifest 포함.
- `legal/`: Gemma 약관 전체·금지 사용 정책·필수 NOTICE·변경 내역·하위 사용자에게 적용되는 사용 조건.
- `README.md`, `SHA256SUMS.txt`: 이 설명과 내부 handoff ZIP의 체크섬.

내부 handoff ZIP은 **1,602,036,051 bytes**, SHA256은 다음과 같다.

```text
4bc8d1f037e2b9da17ef9c0657b8e94fafb71d80eaedd5758a553ba74a3a746c
```

바깥 ZIP을 풀고 저장소 루트에서 **내부** ZIP을 가져온다. 가져오기는 두 어댑터의 해시·설정·완료·선택 결과와 소스/protocol 일치를 확인한다.

```sh
python -B -X utf8 scripts/dolly_external.py import --bundle /path/to/gemma2-9b-bf16-sft-results.zip
```

Windows에서 긴 경로 오류가 생기면 PowerShell에서 다음과 같이 실행한다.

```powershell
$SftScript = '\\?\' + (Resolve-Path scripts/dolly_external.py).Path
$SftBundle = (Resolve-Path C:/path/to/gemma2-9b-bf16-sft-results.zip).Path
python -B -X utf8 $SftScript import --bundle $SftBundle
```

어댑터 사용에는 같은 기반 모델 revision과 Hugging Face 접근 권한이 필요하다. 기반 모델 전체 가중치·학습 원문·optimizer checkpoint는 배포 묶음에 포함하지 않는다. 재적재 예시와 세부 절차는 [실행 안내](../../../docs/EXTERNAL_GEMMA_SFT.md)를 따른다.

## 배포 조건

이 두 어댑터는 Dolly SFT로 학습한 Gemma 모델 파생물이며 [Gemma Terms of Use](https://ai.google.dev/gemma/terms)와 [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)를 따른다. 배포에 적용되는 조건은 동봉한 [USE_TERMS.txt](legal/USE_TERMS.txt), 전체 약관과 정책은 같은 `legal/` 폴더, 수정 내역은 [MODIFICATIONS.txt](legal/MODIFICATIONS.txt)에 있다. 재배포할 때 이 조건과 고지를 함께 전달한다.
