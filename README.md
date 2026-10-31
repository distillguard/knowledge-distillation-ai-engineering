# ECQA 지식증류 연구

Qwen3.5에서 직접 증류와 2B 중간 교사 경로를 FKL·RKL로 비교한 연구다. 기존 검증 선정 실험(v1)과 세 시드·고정 E3 실험(v2)의 문서를 분리했다. 최신 v2는 2026-10-31에 전체 18개 시험과 계획한 통계를 검증했다.

## 연구 문서

| 버전 | 연구계획서 | 연구보고서 | 비교 요약 |
|---|---|---|---|
| v2 세 시드·고정 E3 | [계획과 조건](research/v2/RESEARCH_PLAN.md) | [최종 결과와 전체 통계](research/v2/RESEARCH_REPORT.md) | [핵심 비교](research/v2/RESULTS_VISUALIZATION.md) |
| v1 단일 시드·검증 선정 | [계획과 조건](research/v1/RESEARCH_PLAN.md) | [기존 결과와 해석](research/v1/RESEARCH_REPORT.md) | [기존 비교](research/v1/RESULTS_VISUALIZATION.md) |

v1은 2026-10-24에 공개한 연구 내용을 기준으로 분리했다. v2는 시드 10·20·30, 단계별 3에포크·1,419회 갱신, 중간 교사와 학생 모두 E3 평가를 적용했다. 두 버전의 지표와 평가 가중치를 섞지 않는다.

## v2 결과의 핵심

- 중간 FKL의 평균 NLL·Brier는 0.866428·0.459024로 직접 FKL의 0.892186·0.470023보다 낮았다.
- 직접 RKL의 평균 ACC는 65.6639%로 네 조건 중 가장 높았으나 NLL·Brier는 고정 0.8B SFT보다 높았다.
- ACC 경로 차이의 문항 구간은 0을 포함했고 시드별 정확 McNemar 검정의 Holm 보정 p값은 모두 0.05보다 컸다. 이는 정답률 동등성의 입증이 아니다.
- 문항 구간은 고정된 세 KD 학습 결과에 조건부이며 학습 시드 모집단 구간이 아니다. 공통 SFT 초기값 한 개와 중간 경로의 추가 2B 학습을 해석에 포함한다.

## 공개 코드와 집계

| 파일 | 용도 |
|---|---|
| [v2 공개 집계](artifacts/ecqa/kd_v2/publication_summary.json) | 모델별·시드별 지표, 대응 검정, 문항 구간, 검증 에포크, 비용, 검증 SHA |
| [v1 최종 분석](artifacts/ecqa/kd/final_analysis.json) | 기존 선정 모델의 지표·정오 중첩·비용 |
| [v2 실행기](scripts/kd_ecqa_v2.py) | 고정 E3·세 KD 시드의 실행 원본 |
| [통계 계산](scripts/final_statistics_cpu.py) | 같은 문항 ID 부트스트랩·정확 McNemar·시드별 Holm의 계산 원본 |
| [공통 평가](scripts/ecqa_study.py), [SFT](scripts/sft_ecqa.py), [v1 KD](scripts/kd_ecqa.py) | 자료 준비·평가·기존 학습 |
| [GPU 보호](scripts/gpu_safety.py), [관찰 모니터](scripts/monitor_ecqa_gpu.py) | 자원 보호와 별도 CPU 관찰 |
| tests · requirements.txt | CPU 검증과 실행 의존성 |

실험 소스는 완료 당시 바이트를 유지한다. 로컬 자료·모델 캐시·adapter·재사용 명세·통계 벡터·운영 문서가 필요하므로 공개 저장소만으로 전체 학습·검정을 재실행할 수는 없다. 통계 계산은 `python scripts/final_statistics_cpu.py <vectors.json> <audit.json> <out.json>` 형식이며 두 입력은 비공개 로컬 자료다. 원래 실행기의 루트 MD 출력 대상은 로컬 운영 사본이고, 공개 문서는 research/v1·v2의 정리본이다.

CPU 검사는 `python -m unittest discover -s tests -p "test_ecqa*.py"`로 실행한다. 일부 검사는 비공개 집계 입력·운영 wrapper를 필요로 한다. 평가나 학습을 다시 시작하지 않고 공개 보고서와 집계 결과를 읽을 수 있다.

## 공개 범위

연구계획서·연구보고서·비교 요약·코드·테스트·집계 근거만 공개한다. 논문 원고는 Markdown·HWP·HWPX·PDF 모두 공개 목록에서 제외하고 로컬에 보존한다. 문항 원문·생성 응답·가중치·체크포인트·원시 로그·운영 기록·사용자 환경도 제외한다. v2 원본 final_analysis.json은 단계별 학습 로그와 호스트 경로를 포함하므로 로컬에 보존하고, 공개용 publication_summary.json에는 집계와 해시만 담았다.

이번 변경은 현재 파일 목록에서 이전 논문 원고를 제외한다. 기존 커밋 이력은 재작성하지 않는다. 초기화·교사·최종 가중치·생성 시각·기존 결과의 SHA는 보존하며 재사용 결과를 새 학습·새 생성으로 세지 않는다.
