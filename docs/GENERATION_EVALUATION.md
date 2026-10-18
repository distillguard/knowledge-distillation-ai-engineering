# 생성 벤치마크의 교사·학생 선정

기준일: 2026-10-18. 생성 성능의 차이를 모델 선정의 주 근거로 삼는다. SST-2·BoolQ의 accuracy·macro-F1·Brier·클래스별 오류는 편향과 확률 품질의 보조 진단이다. ECE도 후보의 후속 평가에 추가하되, 기존 Brier를 ECE로 바꾸어 표시하지 않는다. 교사의 자격에 임의의 최소 점수 차이를 두지 않고 과제별 차이와 동일 문항의 오류를 확인한다.

2026-10-16 최신 범위 결정: **Qwen 제외 8모델의 예비 평가는 Dolly 500문항부터 수행하고 외부 세 benchmark는 후보 축소 후 적용**한다. 이미 완료한 외부 평가 결과는 보존한다. 학습은 Dolly만 유지한다. 외부까지 적용한 모델당 2,516개는 학습 표본 수가 아니라 네 자료의 평가 문항 합계다.

## 체크포인트 선택과 조합 선정

현재 검증 NLL은 **각 모델 내부에서** 두 학습률 후보와 에포크별 SFT 체크포인트를 선택하는 기준이다. 다른 tokenizer를 사용하는 모델의 토큰당 NLL로 계열의 순위를 정하지 않는다. Teacher forcing의 NLL 감소만으로 실제 생성 품질의 향상을 확정하지 않는다.

네 생성 benchmark의 ROUGE-L은 논문과 비교하는 1차 후보 평가다. 이 점수 하나로 최종 교사·학생 조합을 확정하지 않는다. 과제별 우위의 일관성, 같은 문항의 응답 오류, 의미 유사성의 보완 평가와 기존 분류 진단을 함께 검토한다. 지표를 임의 가중합하거나 최소 격차를 새로 정하지 않는다. 증류 전 단일 모델의 성능은 이후 증류 효과를 보장하지 않는다.

[MiniLLM §2.3](https://arxiv.org/html/2306.08543v4)은 학생 초기화에 최저 검증 손실을 사용하지만 §3.1의 생성 관련 설정 탐색에는 검증 ROUGE-L을 사용하고, 최종 응답은 ROUGE-L·GPT-4 feedback·사람 평가로 확인한다. 현재의 로컬 NLL 선택을 원문의 전체 생성 평가와 동일하게 표시하지 않는다.

2026-10-18 사용자 결정에 따라 [METEOR](https://www.nltk.org/api/nltk.translate.meteor_score.html)·[BERTScore](https://github.com/Tiiiger/bert_score)·[Distinct-4](https://arxiv.org/html/2306.08543v4#A2.SS4)를 추가하고, 후보의 분류 진단에는 ECE를 포함한다. **현재는 실행 승인 및 설계 단계이며 추가 지표를 아직 구현·채점하지 않았다.** METEOR·BERTScore는 어휘·의미 유사성, Distinct-4는 표현의 다양성을 보완하며 사실성을 직접 판정하지 않는다. 기존 생성 응답과 원본 reference를 재사용하고 추가 SFT나 기존 답변 재생성 없이 별도 채점 명세·결과를 남긴다. 평가 모델·revision·길이 처리·다중 정답·집계를 먼저 고정하며, 실행 중인 생성 명세와 기존 완료 점수는 변경하지 않는다.

**SFT 전후 기준선도 추가 실행한다.** 현재 보유한 SFT 전 값은 개발 NLL이며, 동일 문항의 원본 생성 기준선은 아직 없다. 활성 8개 모델의 원본 checkpoint를 같은 Dolly 500문항·seed 10·prompt·EOS·생성 설정·모델별 정밀도로 평가해 `SFT 전 / SFT 후 / 변화량`을 작성한다. 새로 필요한 것은 원본 모델의 추론이며 SFT를 다시 하지 않는다. Gemma-IT의 원본은 기존 instruction tuning 이후이자 우리 Dolly SFT 이전이다. 저자 공개 SFT checkpoint를 이 원본 기준선으로 대체하지 않는다.

## 조합 선정까지의 단계와 완료 기준

| 단계 | 현재 실행·결정 | 완료 근거 |
|---|---|---|
| SFT 체크포인트 | 각 모델에서 두 LR·epoch별 개발 NLL 선택 | 두 LR 완료·selected adapter hash. 계열 간 NLL 순위와 구분 |
| 예비 생성 평가 | Dolly 500문항, 생성 seed 10 한 개 | Dolly JSON·문항별 CSV·같은 ID·fingerprint 검증 |
| SFT 전후 기준선 | 활성 8개 원본의 동일 Dolly 평가 추가 | 원본과 SFT의 model revision·정밀도·ID·생성 조건 일치, 단계별 별도 fingerprint·응답 경로 |
| 보완 지표 | METEOR·BERTScore·Distinct-4, 미구현·미채점 | 고정 채점 명세 후 원본/SFT 응답 재사용. BERTScore의 평가 모델 길이 제한과 영향 문항 수 공개 |
| 후보 후속 평가 | 공통 Dolly 지표·응답 검토로 후보 축소 후 외부 3종 | 후보 원본/SFT를 동일 조건으로 비교, 검증된 기존 결과 재사용 |
| 보조 진단 | 후보 원본/SFT의 SST-2·BoolQ | accuracy·macro-F1·클래스별 오류·Brier·ECE. 생성 평가의 대체가 아님 |
| 비교·오류 분석 | 모델×benchmark×지표와 계열 내 차이 | 과제별 일관성, 지시 준수·사실 오류·반복·누락, 사례 선정 근거 |
| 증류 실행 점검 | 후보 조합의 logits·어휘·prefix·학생 갱신·메모리 | 실제 KD 연산·필요한 생성·MC 경로의 통과 기록 |
| 최종 조합 | 미정 | model ID·revision·SFT adapter·정밀도·선정 자료·이유와 한계 동결 |

현재 `completion.json`과 자동 보고서의 완료 수는 활성 8모델의 Dolly 예비 평가 완료다. `completed_extended_models`는 네 benchmark를 모두 완료한 보존 결과를 별도 기록한다. 보완 지표·오류 검토·증류 실행 점검·사람의 선정 판단까지 자동으로 완료됐다는 뜻이 아니다. 실제 점수가 있는 지표만 비교표에 채우고 미실행 항목은 미실행으로 표시한다.

보완 지표 채택과 오류 검토 기준은 전체 결과를 보고 유리한 기준으로 바꾸지 않도록 먼저 확정한다. 의미 유사성·reference 겹침·정답률의 역할을 구분하며, 응답 검토의 사실 판정은 별도 근거가 필요하다.

Gemma-2-9B-IT의 현재 두 LR·3epoch SFT와 Dolly 평가를 끝낸 다음, **원본/SFT Dolly 비교 → 보완 채점 → 후보 축소 → 후보의 외부 3종·분류 진단 → 실제 증류 경로 점검 → 추천 조합 보고** 순서로 수행한다. 이번 후속 평가에 새로운 SFT 탐색이나 본 KD 학습은 포함하지 않는다. MiniLLM 원문의 SFT 탐색·학생 초기화 절차는 선정 이후 별도 본 실험 설계로 남긴다. 단순한 SFT worker 종료로 이 후속 작업까지 완료됐다고 보고하지 않는다.

## 보완 지표의 명세 원칙

- METEOR는 NLTK 3.9.2의 공식 기본값(alpha 0.9, beta 3, gamma 0.5)·Porter stemmer·영어 WordNet·최고 reference 점수를 사용한다. 공통 단어 토큰화와 corpus 버전/hash를 기록한다.
- BERTScore는 공식 영어 기본 설정인 RoBERTa-large·17층·IDF 미사용을 기준으로 평가 모델 revision과 tokenizer를 고정한다. 원문 reference는 보존하되 encoder의 native 길이 제한이 적용되는 문항·토큰 수와 처리를 별도로 공개한다. 아직 평가 모델 파일이나 길이 처리 점검을 완료한 상태는 아니다.
- Distinct-4는 응답별 경계를 넘지 않는 4-gram의 전체 고유 개수/전체 개수를 사용한다. 전 모델에 같은 단어 tokenizer를 적용하고 분자·분모도 기록한다. 논문의 5개 생성 seed와 현재 예비 1개 seed를 구분한다.
- ECE는 두 label 후보 사이에서 정규화한 기존 확률로 계산한다. [Guo 외의 공식 구현](https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py)의 15개 등폭 confidence bin을 고정 근거로 삼고 bin별 표본 수·정답률·평균 confidence를 남긴다. 이는 최적 bin 수를 찾았다는 뜻이나 MiniLLM의 정확한 구현 재현이라는 뜻이 아니다. 확률 보정용 temperature를 새로 학습하지 않는다.
- LLM judge·사람의 Win/Tie/Loss·ExAccErr는 문헌의 평가 옵션이며 현재 자동 실행에는 포함하지 않는다. 모델의 자동 채점이나 단순 ROUGE 우위 문항 수를 실제 사람의 선호 평가로 표시하지 않는다.

## 개선량의 비교 기준

`원본 → Dolly SFT → 본 KD`를 구분한다. 현재 선정 단계는 `SFT − 원본`, 이후 본 증류는 `KD 학생 − 동일한 SFT 학생`을 비교한다. 교사−학생 격차는 별도 표로 보고한다. 지표별 증가/감소의 유리한 방향을 명시하고 개선·정체·악화를 모두 남긴다. 본 KD의 공통 초기화·학습 예산·반복 조건은 선정 후 동결하며 현재 예비 결과로 증류 효과를 주장하지 않는다.

[MiniLLM §3.1·부록 B.1](https://arxiv.org/html/2306.08543v4#A2.SS1)은 과제 SFT를 거친 교사와, 사전 탐색한 설정으로 3epoch SFT하고 최저 검증 loss를 선택한 학생에서 증류한다. [DistiLLM 부록 D.2·표 6](https://arxiv.org/html/2402.03898v2#A4.SS2)은 SFT 학생 초기화를 사용하는 주 비교와 SFT 없는 학생의 one-stage 증류를 구분한다. 따라서 원본 학생에서 직접 KD하는 조건을 현재 SFT 출발 조건과 섞지 않는다. CaRE-KD §10.3의 MiniLLM 설정 인용만으로 모든 초기화 세부가 같다고 단정하지 않는다. 원본 성능 측정은 고정 원본 checkpoint를 따로 적재하므로 현재 SFT 종료 이후에도 수행할 수 있다.

## 문헌과 자료

[MiniLLM §3.1·3.3](https://arxiv.org/html/2306.08543v4)은 다섯 생성 benchmark를 사용하고 SST-2·BoolQ는 LLaMA-7B의 calibration 분석에 사용한다. [CaRE-KD §4](https://arxiv.org/html/2609.36734v1)의 Dolly 학습 instruction-following 실험과 겹치는 네 생성 benchmark를 우선 평가한다. Unnatural Instructions, chat·수학·코드 전용 평가는 이번 추가 실행에 포함하지 않는다.

| 평가 자료 | 공식 배포 파일과 고정 revision | 공통 평가 문항 |
|---|---|---:|
| Dolly | 기존 동결 `selection_evaluation.jsonl`의 ID와 원문 | 500 |
| Self-Instruct | [MiniLLM/self-inst](https://huggingface.co/datasets/MiniLLM/self-inst/tree/ad9edcc55ec5623f6d9d2472963bda3701681af8), `valid.jsonl` | 242 |
| S-NI | [MiniLLM/sinst](https://huggingface.co/datasets/MiniLLM/sinst/tree/84b8356a5b631002f92ad73adebd5ac9db5612cd), `11_/valid.jsonl` | 1,694 |
| Vicuna | [MiniLLM/Vicuna](https://huggingface.co/datasets/MiniLLM/Vicuna/tree/05a18702266a6c51547034268d6d58bce214bd77), `valid.jsonl` | 80 |

8모델의 SFT 후 Dolly 예비 평가는 500문항 × 8 = 4,000개 답변이다. 원본 기준선도 동일하게 4,000개를 추가하므로 두 단계가 모두 끝나면 총 8,000개다. 후보 축소 후 외부 3종까지 적용한 모델은 checkpoint 단계마다 총 2,516문항이다. 여러 지표로 같은 답변을 채점해도 새 답변을 생성하지 않는다. 논문의 설명상 표본 수와 실제 배포 파일의 행 수를 섞지 않는다. Self-Instruct의 배포 파일은 242행이며 논문에 기술한 252문항을 임의로 복원하지 않았다. S-NI는 배포된 `11_` 구간 자체를 사용하며 새로운 길이 임계값을 만들지 않았다. 다중 정답은 [공식 ROUGE 평가 코드](https://github.com/microsoft/LMOps/blob/48fca1acc61ffa20ebb7c31c74a1d8e33499b476/minillm/rouge_metric.py)처럼 reference별 점수의 최댓값을 사용한다.

## 전처리와 누수 점검

원문 instruction·input·전체 reference를 보존한다. GPT-2 wrapper와 배포 `prompt`의 일치는 세 추가 자료의 모든 행에서 확인했다. Gemma의 chat wrapper와 OpenLLaMA의 BOS는 기존 SFT 조건을 따른다. 최초 10모델에서 수행한 tokenizer 감사와 공통 ID는 그대로 보존하고 현재 활성 8모델에 적용한다.

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

이후 모델마다 **예비 SFT → 문서 → Dolly 500 생성 → 생성 보고서·문서 → 다음 모델** 순서다. `evaluate_dolly_pilot.py`는 기존 생성 구현의 실행 자료를 Dolly로 제한하며 생성·채점 알고리즘과 기존 결과 fingerprint는 바꾸지 않는다. 외부 3종과 새 SST-2·BoolQ는 현재 worker에서 실행하지 않는다. 모델별 실행과 전체 실행은 같은 worker 잠금을 사용한다.

Dolly는 완료한 seed의 응답·요약을 검증해 재사용하며 미완료 seed를 중단한 경우 같은 seed로 처음부터 재생성한다. 후속 외부 3종의 중간 응답과 CPU·CUDA RNG 상태는 `tmp/generation_benchmarks_v1/`에 50문항마다 함께 저장하며, 해당 경로는 ID prefix·fingerprint·RNG 상태를 검증해 재개한다. 50은 저장 주기이며 학습·샘플링 파라미터가 아니다.

처음 준비할 때:

```powershell
python -B -X utf8 scripts/prepare_generation_benchmarks.py fetch
python -B -X utf8 scripts/prepare_generation_benchmarks.py prepare
```

공개 protocol이 있고 준비된 원문 파일만 없는 경우에도 같은 revision에서 자료를 복원한다. 재계산한 protocol 전체가 기존 명세와 같아야 하며 완료한 결과의 명세를 자동 덮어쓰지 않는다. fingerprint를 보호하기 위해 Python 소스와 새 준비 JSONL의 줄바꿈은 LF로 고정한다.

모델별 실행은 기존처럼 `python -u -B -X utf8 scripts/run_dolly_sft.py --model gpt2_base`를 사용한다. 전체는 `--model all`이다. 이미 완료된 SFT와 평가를 검증해 재사용한다. 새 평가 코드만 GPU worker와 따로 동시에 실행하지 않는다.

## 산출물과 완료 기준

[생성 주 평가 보고서](../results/generation_benchmarks_v1/REPORT.md)에 표·그림과 동일 문항 차이를 모은다. `protocol.json`은 최초 10모델의 자료 revision·원문/준비 파일 hash·길이 감사·제외 ID를 보존한다. 활성 실행 범위는 `dolly_scope.py`와 `completion.json`에 기록한다. 모델별 `<benchmark>.json`의 완료 상태·평가 fingerprint·문항 수와 `<benchmark>_per_item.csv`의 hash·ID를 검증한다. 현재 예비 완료는 Dolly 기준이며 외부까지 네 자료를 마친 결과는 별도 구분한다.

Git에는 검증한 점수·ID·category·집계·그림과 문서만 올린다. 원문·응답·RNG checkpoint·가중치·캐시·로그는 `tmp/`에 둔다. 기존 `results/dolly_sft_v1/`는 원래 Dolly·분류 평가의 이력을 유지한다. 추가 생성 주 평가가 끝나기 전에 그 이력의 완료를 전체 모델 선정 완료로 해석하지 않는다.
