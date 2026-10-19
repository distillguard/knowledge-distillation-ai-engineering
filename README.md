# 지식증류 · 인공지능공학

**GPT-2 XL → GPT-2 base에서 confidence soft gate의 역전파 허용·차단이 생성 성능과 과신에 미치는 영향을 분석한다.** MiniLLM의 모델·Dolly·SFT 준비 절차를 참고하고, CaRE의 KL 혼합 gate를 대조한다. 먼저 GPT-2 한 쌍에서 검증하며 OPT는 이후 확장 연구로 둔다.

현재는 재현·추가 분석의 준비 단계다. RQ1~3의 큰 질문과 detach 자체에는 선행연구가 있으며, 학술대회 제출을 위한 독창성이 확정된 상태는 아니다. 동일 gate의 gradient만 통제한 실증 분석이 어떤 추가 근거를 제공하는지 KD 전에 판단한다.

## 연구 문서

| 문서 | 내용 |
|---|---|
| [연구계획서](docs/RESEARCH_PLAN_SOFT_GATE_GRADIENT.md) | 연구 질문·수식·대조 조건·원문 근거·단계별 지표 |
| [인공지능공학 구성](docs/GPT2_AI_ENGINEERING.md) | Transformer 구조·토크나이저·GELU·Softmax·AdamW·손실·스케줄러·정밀도 |
| [기준선 진행 기록](docs/SOFT_GATE_PROGRESS.md) | 실제 원본 평가의 진행과 완료 여부 |
| [학생 학습 진행 기록](docs/SOFT_GATE_STUDENT_PROGRESS.md) | 조건별 완료 epoch·검증 점수·선택·초기화·SFT 후 평가 |

## 연구 구성

- **교사:** MiniLLM 공식 Dolly-SFT GPT-2 XL, 1,557,611,200 parameters. 공개 가중치를 재사용하고 동결한다. 이 컴퓨터의 직접 교사 SFT 결과로 표현하지 않는다.
- **학생:** 원본 GPT-2 base, 124,439,808 parameters. 전체 파라미터 SFT를 직접 수행한다. 이번 본 준비는 LoRA·가중치 양자화를 사용하지 않는다.
- **자료:** 공식 MiniLLM Dolly 분할과 원문 concat tokenization. 학습 11,435건·검증 1,000건·평가 500건. 잘림과 지시문 중복을 공개하며 평가의 중복 제외 부분집합은 487건이다.
- **핵심 대조:** 동일 초기화·같은 순전파 gate에서 Soft-Allow와 Soft-Detach의 gradient 경로 차이. MiniLLM의 전체 policy-gradient 방법과 토큰 단위 KL 혼합을 구분한다.

## 실행 순서

1. 원본 base·XL의 동일 조건 기준선: 응답 NLL/PPL, Dolly 500 × 생성 5 seed, SST-2·BoolQ 진단, 저장 응답의 보완 채점.
2. 학생 SFT: LR `5e-4 / 1e-4 / 5e-5` × effective batch `32 / 64`, 각 20 epoch. 검증 ROUGE-L로 조건과 checkpoint 선택.
3. 학생 초기화: 선택 LR·batch로 원본에서 새 3 epoch. 최소 검증 loss checkpoint 선택.
4. 공식 교사·선택된 학생 SFT·학생 초기화의 동일 생성·NLL·분류 평가. 보완 지표는 학습 epoch를 추가하지 않고 같은 저장 응답을 채점한다.
5. KD의 prefix·gradient 추정·학습량을 고정한 뒤 gate 대조·분석. 미확정 KD 명세를 임의의 기본값으로 채우지 않는다.

생성 지표는 ROUGE-L P/R/F1, METEOR, BERTScore P/R/F1, Distinct-4와 길이·빈 응답·상한 도달률이다. SST-2·BoolQ의 accuracy·macro-F1·balanced accuracy·ECE·Brier·오답 확신은 보조 진단이다. ROUGE-L은 실제 정답률이 아니며 생성 5 seed의 SD는 독립 학습 반복의 불확실성이 아니다.

## 실행 명세와 코드

[원본·데이터 명세](results/soft_gate_gradient_v1/preparation.json), [학생 명세](results/soft_gate_gradient_v1/student/protocol.json), [교사 출처](results/soft_gate_gradient_v1/teacher.json), [공식 소스 해시](results/soft_gate_gradient_v1/official_sources.json)를 기준으로 실행한다. 고정값은 재현 기준과 검증 선택값을 구분한다. [AdamW 등의 공학적 설명](docs/GPT2_AI_ENGINEERING.md)과 원문의 단일 GPU 이식 차이를 함께 읽는다.

GPU에 맞는 PyTorch 환경을 마련한 뒤 현재 실행 의존성은 `requirements-sft.txt`로 설치한다. BERTScore의 이전 Transformers 스택은 메인 환경에 덮어쓰지 않고 별도 경로에 설치한다.

```powershell
python -m pip install -r requirements-sft.txt
python -m pip install --no-deps --target tmp/supplementary_metrics_v1/deps -r requirements-supplementary-metrics.txt
python scripts/prepare_soft_gate_assets.py
python scripts/soft_gate_study.py prepare
$env:PYTHONPATH = "$PWD/tmp/supplementary_metrics_v1/deps"
python scripts/prepare_supplementary_metrics.py
Remove-Item Env:PYTHONPATH
python scripts/train_soft_gate_student.py prepare
python -m unittest discover -s tests -p 'test_soft_gate*.py' -v
python -u scripts/run_soft_gate_stage1.py
```

기존 결과가 있는 경로에서는 source·자료·환경 fingerprint 불일치를 실패로 처리한다. 현재 실행 기록과 다른 환경에서 재실험할 경우 기존 결과를 덮어쓰지 않고 별도 실행 ID와 차이 명세를 먼저 마련한다. GPU 작업은 공통 잠금으로 직렬화하고 epoch 단위 재개 checkpoint를 유지한다. `dolly_sft.py`·`dolly_study.py`·`evaluate_dolly_sft.py` 등은 재사용 중인 공통 의존성이며 현재 연구의 실행 진입점은 위 두 새 실행기다.

## 공개 범위

원격 저장소에는 현재 계획·설명·필수 코드·테스트·의존성·검증된 요약만 둔다. 현재도 사용하는 분류 행 manifest와 지표 자산 manifest 두 파일은 기존 경로를 유지한다. 이전 논문 분석·8모델 예비 연구·다중교사 설계·OPT 중단 상태는 로컬 기록으로 보존한다. 가중치·checkpoint·원문 응답·로그·캐시·토큰은 공개하지 않는다. 파일별 수치나 완료 표시는 실제 결과 생성 이후에 갱신한다.

## 참고문헌

- [MiniLLM](https://arxiv.org/html/2306.08543v4): 모델·SFT 준비·Dolly-only 참조·생성 평가의 근거.
- [CaRE-KD](https://arxiv.org/html/2609.36734v1): confidence 기반 KL 혼합 gate의 근거.
- [ToDi](https://arxiv.org/html/2505.16297v2): 혼합 가중치의 stop-gradient와 gradient 효과에 관한 관련 연구.
- [MiniLLM 공식 교사](https://huggingface.co/MiniLLM/teacher-gpt2-1.5B/tree/4a8d07d52ad67b7b87fec9aba25b1521173eb658): 재사용하는 교사 가중치.
