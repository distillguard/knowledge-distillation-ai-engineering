# 생성 벤치마크의 교사·학생 선정

기준일: 2026-10-16. 생성 성능의 차이를 모델 선정의 주 근거로 삼는다. SST-2·BoolQ의 accuracy·macro-F1·Brier·클래스별 오류는 편향과 확률 품질의 보조 진단이다. 현재 Brier score를 MiniLLM의 ECE 재현으로 표시하지 않는다. 교사의 자격에 임의의 최소 점수 차이를 두지 않고 과제별 차이와 동일 문항의 오류를 확인한다.

## 문헌과 자료

[MiniLLM §3.1·3.3](https://arxiv.org/html/2306.08543v4)은 다섯 생성 benchmark를 사용하고 SST-2·BoolQ는 LLaMA-7B의 calibration 분석에 사용한다. [CaRE-KD §4](https://arxiv.org/html/2609.36734v1)의 Dolly 학습 instruction-following 실험과 겹치는 네 생성 benchmark를 우선 평가한다. Unnatural Instructions, chat·수학·코드 전용 평가는 이번 추가 실행에 포함하지 않는다.

| 평가 자료 | 공식 배포 파일과 고정 revision | 공통 평가 문항 |
|---|---|---:|
| Dolly | 기존 동결 `selection_evaluation.jsonl`의 ID와 원문 | 500 |
| Self-Instruct | [MiniLLM/self-inst](https://huggingface.co/datasets/MiniLLM/self-inst/tree/ad9edcc55ec5623f6d9d2472963bda3701681af8), `valid.jsonl` | 242 |
| S-NI | [MiniLLM/sinst](https://huggingface.co/datasets/MiniLLM/sinst/tree/84b8356a5b631002f92ad73adebd5ac9db5612cd), `11_/valid.jsonl` | 1,694 |
| Vicuna | [MiniLLM/Vicuna](https://huggingface.co/datasets/MiniLLM/Vicuna/tree/05a18702266a6c51547034268d6d58bce214bd77), `valid.jsonl` | 80 |

모델당 총 2,516문항이다. 논문의 설명상 표본 수와 실제 배포 파일의 행 수를 섞지 않는다. Self-Instruct의 배포 파일은 242행이며 논문에 기술한 252문항을 임의로 복원하지 않았다. S-NI는 배포된 `11_` 구간 자체를 사용하며 새로운 길이 임계값을 만들지 않았다. 다중 정답은 [공식 ROUGE 평가 코드](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/rouge_metric.py)처럼 reference별 점수의 최댓값을 사용한다.

## 전처리와 누수 점검

원문 instruction·input·전체 reference를 보존한다. GPT-2 wrapper와 배포 `prompt`의 일치는 세 추가 자료의 모든 행에서 확인했다. Qwen·Gemma의 chat wrapper와 OpenLLaMA의 BOS는 기존 SFT 조건을 따른다.

학습·개발 집합과 casefold·공백 정규화 instruction이 같은 문항을 출력 확인 전에 공통 제외한다. 모든 tokenizer에서 전체 prompt와 새 토큰 예산 256의 합이 native context에 들어가는지 CPU로 확인한다. 어느 모델에서든 넘으면 그 ID를 전 모델에서 공통 제외한다. 이번 검사에서는 두 사유의 제외가 모두 0건이었고 원문을 자르지 않았다. 모델별 최대 prompt는 [자료 protocol](../results/generation_benchmarks_v1/protocol.json)에 있다.

Dolly의 정규화 prompt 중복 1건은 기존 500개 ID를 유지하기 위해 그대로 보존했다. 나머지 세 배포 집합 내부의 같은 기준 중복은 0건이다. 의미상 중복·기반 모델의 사전학습 오염 전체를 감사한 결과는 아니다.

## 생성과 집계

- 모델 선정용 생성 seed는 기존 결정대로 `10` 한 개다. 학습 seed는 기존 `0`이며 서로 다른 반복이다.
- 새 토큰 상한 256, temperature 1, top-k 0, top-p 1, no-repeat-ngram 6, microbatch 1을 기존 Dolly 평가와 맞춘다. 256은 기존 로컬 응답 예산이며 논문의 모든 생성 설정을 그대로 재현한 것은 아니다.
- 주 지표는 ROUGE-L F1 × 100이다. ROUGE-1/2·정규화 exact match·응답 길이·상한 도달률·빈 응답률도 기록한다. 어휘 겹침을 사실성·사람의 선호로 해석하지 않는다.
- 문항 평균을 benchmark별로 보고하고, 네 benchmark가 모두 끝난 모델에만 과제 평균을 계산한다. [CaRE-KD 표 1](https://arxiv.org/html/2609.36734v1)의 과제 평균 형식에 맞춰 네 평균에 같은 가중치를 준다. 전체 문항을 합쳐 S-NI의 큰 표본 수가 평균을 지배하게 하지 않는다.
- 교사−학생의 평균 차이와 동일 ID에서 ROUGE-L이 높은 쪽·동률 문항 수를 함께 보고한다. 이 문항 수는 사람의 선호 승률과 다르다.
- 단일 생성 seed의 seed 간 SD는 `null`이며 오차막대·±0을 붙이지 않는다. 최종 선정 모델의 본 실험에서만 기존 결정의 생성 5seeds를 적용한다. 현재의 차이로 seed 간 안정성이나 통계적 유의성을 확정하지 않는다.

## 실행 순서와 재개

이미 완료된 GPT-2 base·XL의 생성 평가를 먼저 확인한다. 기존 Dolly 응답과 fingerprint가 일치하면 다시 생성하지 않고 문항별 점수를 재계산한다. 미완료 SFT는 저장된 checkpoint와 기존 LR·에포크·정밀도를 보존해 이어간다.

기존 Dolly 응답 파일에는 문항별 마지막 EOS 여부가 없어 점수 CSV의 `length_capped` 열은 결측으로 둔다. Dolly의 상한 도달률 집계는 검증한 기존 생성 요약을 재사용한다. 새로 생성한 세 benchmark는 EOS를 확인한 문항별 상한 도달 여부를 기록한다.

이후 모델마다 **SFT → 문서 → 생성 4종 → 생성 보고서 → SST-2·BoolQ 보조 평가 → 문서 → 다음 모델** 순서다. 모델별 실행과 전체 실행은 같은 worker 잠금을 사용한다. 새 생성의 중간 응답과 CPU·CUDA RNG 상태는 무시된 `tmp/generation_benchmarks_v1/`에 50문항마다 함께 저장한다. 이는 저장 주기이며 학습·샘플링 파라미터가 아니다. 중단 시 저장한 ID prefix와 fingerprint를 확인한 뒤 같은 RNG 상태에서 재개한다.

처음 준비할 때:

```powershell
python -B -X utf8 scripts/prepare_generation_benchmarks.py fetch
python -B -X utf8 scripts/prepare_generation_benchmarks.py prepare
```

공개 protocol이 있고 준비된 원문 파일만 없는 경우에도 같은 revision에서 자료를 복원한다. 재계산한 protocol 전체가 기존 명세와 같아야 하며 완료한 결과의 명세를 자동 덮어쓰지 않는다. fingerprint를 보호하기 위해 Python 소스와 새 준비 JSONL의 줄바꿈은 LF로 고정한다.

모델별 실행은 기존처럼 `python -u -B -X utf8 scripts/run_dolly_sft.py --model gpt2_base`를 사용한다. 전체는 `--model all`이다. 이미 완료된 SFT와 평가를 검증해 재사용한다. 새 평가 코드만 GPU worker와 따로 동시에 실행하지 않는다.

## 산출물과 완료 기준

[생성 주 평가 보고서](../results/generation_benchmarks_v1/REPORT.md)에 표·그림과 동일 문항 차이를 모은다. `protocol.json`은 자료 revision·원문/준비 파일 hash·모델별 길이 감사·제외 ID를 보존한다. 모델별 `<benchmark>.json`의 완료 상태·평가 fingerprint·문항 수와 `<benchmark>_per_item.csv`의 hash·ID를 검증한다. 네 benchmark가 모두 완료된 모델만 생성 주 평가 완료로 센다.

Git에는 검증한 점수·ID·category·집계·그림과 문서만 올린다. 원문·응답·RNG checkpoint·가중치·캐시·로그는 `tmp/`에 둔다. 기존 `results/dolly_sft_v1/`는 원래 Dolly·분류 평가의 이력을 유지한다. 추가 생성 주 평가가 끝나기 전에 그 이력의 완료를 전체 모델 선정 완료로 해석하지 않는다.
