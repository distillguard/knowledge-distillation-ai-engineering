# ECQA 지식증류 연구

**Qwen3.5의 직접·중간 증류 경로와 KL 방향을 비교한 연구 문서다.**

원본·SFT 기준선, 두 중간 교사와 네 최종 학생의 전체 학습·평가·검증을 마쳤다. 마지막 문서 정리일은 2026-10-24이며 추가 실험은 수행하지 않았다.

## 읽는 순서

| 순서 | 문서 | 내용 |
|---:|---|---|
| 1 | [연구 계획과 방법](RESEARCH_PLAN.md) | 연구 질문, 데이터, 경로, 초기화, 학습·선정·평가 조건 |
| 2 | [연구 결과와 해석](RESEARCH_REPORT.md) | 기준선, 여섯 단계 완료, 최종 지표, 대응 문항, 비용, 관측·가설·한계 |
| 3 | [실험 흐름과 결과 비교](RESULTS_VISUALIZATION.md) | 경로 도식과 핵심 비교표 |
| 4 | [국문 논문 원고](docs/PAPER_KO.md) | 대학생 논문경진대회 공식 2페이지 양식에 배치한 본문 |
| 참고 | [국문 초록](docs/ABSTRACT_KO.md) | 최종 논문의 결과·주장 범위에 맞춘 별도 요약 |
| 참고 | [기호·약어](docs/NOTATION.md) | 경로 별칭, 수식, 평가 지표, ΔF·ΔR·%p 정의 |

## 결과의 핵심

동일 test 2,194문항의 이번 고정 조건에서는 두 경로의 FKL이 SFT와 대응 RKL보다 낮은 NLL·Brier를 기록해, 첫 답 A-E 조건부 확률 평가에서 우세했다. 정답 수는 SFT 대비 직접 FKL이 1건 많고 중간 FKL은 같았다. 중간 FKL의 NLL 0.867518·Brier 0.459455가 다섯 0.8B 조건 중 가장 낮았고, SFT 대비 감소율은 각각 16.76%·10.62%였다. 이를 핵심 관측으로 정리하고 ACC·혼동 분석은 보조 근거로 제시한다. 기존 dev ACC 우선 모델 선정 기준은 변경하지 않았으며, 이 강조점은 완료 결과의 사후 해석이다. 통계적 유의성·내부 표현·확률 보정의 개선을 입증한 것은 아니다.

ΔF=ACC(중간 FKL)−ACC(직접 FKL), ΔR=ACC(중간 RKL)−ACC(직접 RKL)로 정의하면 각각 −0.045579%p, −0.410210%p이고 ΔR−ΔF는 −0.364631%p다. 단일 시드·모델 계열·데이터셋의 서술적 차이이며 통계적 유의성이나 인과적 원인을 검증한 값은 아니다.

## 공개 범위와 원고 상태

공개 대상은 위 7개 MD, 현재 ECQA 실행·분석 코드와 테스트, 의존성 명세, 집계 JSON과 집계 그림이다. 원본·SFT 기준선과 4B CE audit는 이번 증류 연구의 비교·검증 근거로 포함한다. HWP·PDF·문서 미리보기, 문항 원문·생성 응답, 가중치·체크포인트, 원시 로그, 중복 학습 이력과 운영 기록은 제외하고 로컬에 보존한다. 허용 목록에 없는 파일도 기본적으로 Git에서 제외한다.

공식 양식을 적용한 2페이지 HWP와 PDF는 로컬에 보관했다. 저자명·소속·이메일·지도교수는 입력이 필요하며 학회에 제출하지 않았다. 이전 Dolly·soft-gate 연구 파일 25개는 현재 Git 트리에서 제거하고 로컬 보관본을 유지했다. 과거 커밋 이력은 재작성하지 않았다.

최종 원고는 NLL·Brier 산점도, 중간 FKL−SFT 혼동행렬 차이 그림, 다섯 조건의 종합 성능표를 한 단 너비로 배치했다. 토큰 길이는 최종 10,839문항의 요약 문장으로 줄였다. 두 그림의 수치와 해석 범위는 [비교 요약](RESULTS_VISUALIZATION.md)에 정리했다.

## 코드와 집계 근거

| 경로 | 용도 |
|---|---|
| `scripts/ecqa_study.py`, `scripts/sft_ecqa.py`, `scripts/gpu_safety.py` | 전처리·평가·SFT 및 자원 보호 공통 코드 |
| `scripts/kd_ecqa.py`, `scripts/analyze_ecqa_kd.py` | 여섯 단계 증류와 완료 결과의 CPU 검증·분석 |
| `scripts/audit_ecqa_ce.py`, `scripts/render_results.py` | 기준선 CE 검증과 집계 결과 렌더링 |
| `tests/test_ecqa*.py`, `requirements.txt` | 파서·지표·마스킹·KL 손실·gradient·모델 선정 검사와 실행 의존성 |
| [최종 분석](artifacts/ecqa/kd/final_analysis.json), [완료 검증](artifacts/ecqa/kd/final_verification.json) | 다섯 조건 지표, 경로 효과, 교사·학생 정오 중첩, 범위를 명시한 비용 |
| [전처리 명세](artifacts/ecqa/data_manifest.json), [최종 토큰 길이](artifacts/ecqa/final_data_token_lengths.json) | 자료 fingerprint, 중복 제거 정책, 최종 10,839문항 요약 |
| `artifacts/ecqa/baselines/`, `sft/`, `ce_audit/`, `kd/` | 기준선·단계별 검증/선정·시험 집계와 프로토콜; 개별 응답 원문 제외 |

CPU 단위 검사는 `python -m unittest discover -s tests -p "test_ecqa*.py"`로 실행한다. `run`과 분석 재실행에는 공개하지 않은 로컬 자료·모델 캐시·adapter·응답·복구 기록이 필요하므로, 이 저장소만으로 전체 실행 환경이 갖춰지는 것은 아니다. 완료 결과를 확인하는 데 추가 학습이나 생성은 필요하지 않다. 기록된 SHA-256과 대조할 수 있도록 실험 코드와 집계 JSON의 바이트를 보존했다. `local/`·`output/`·`tmp/` 경로는 로컬 보관 위치이며 공개 다운로드 링크가 아니다.
