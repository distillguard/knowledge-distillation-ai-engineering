# 원본과 Dolly SFT의 생성 평가 비교

완료: 원본 Dolly 8/8, SFT Dolly 8/8, 보완 채점 원본 8/8·SFT 8/8. 최종 조합은 후보 후속 검증 후 선정한다.

동일 원문 ID·prompt·모델별 정밀도·생성 seed 10·새 토큰 상한 256의 결과다. 원본은 우리 Dolly SFT 전 checkpoint이며 Gemma-IT는 이미 instruction tuning을 거친 모델이다. 미완료 점수는 공란으로 두고 완료한 응답·CSV·명세 hash와 지표를 재검산한다.

ROUGE-L·METEOR·raw BERTScore F1은 ×100이다. Distinct-4는 corpus 고유/전체 4-gram 비율이며 클수록 사실성이나 성능이 좋다는 뜻이 아니다. 단일 seed이므로 seed 간 SD·오차막대는 없다. 아래 결과만으로 최종 조합을 자동 선정하지 않는다.

| 모델 | 단계 | Dolly ROUGE-L | METEOR | BERTScore F1 | Distinct-4 |
|---|---|---:|---:|---:|---:|
| GPT-2 base | original | 6.16 | 10.09 | 78.93 | 0.9818 |
| GPT-2 base | sft | 14.68 | 16.10 | 83.27 | 0.9961 |
| GPT-2 XL | original | 7.50 | 10.83 | 79.58 | 0.9724 |
| GPT-2 XL | sft | 20.13 | 21.18 | 84.73 | 0.9904 |
| OPT-125M | original | 7.40 | 11.48 | 79.60 | 0.9634 |
| OPT-125M | sft | 17.08 | 17.05 | 83.84 | 0.9942 |
| OPT-2.7B | original | 8.66 | 13.94 | 80.63 | 0.9585 |
| OPT-2.7B | sft | 21.52 | 22.41 | 85.28 | 0.9905 |
| OpenLLaMA-3B | original | 9.69 | 12.55 | 79.77 | 0.9620 |
| OpenLLaMA-3B | sft | 21.86 | 21.88 | 85.42 | 0.9946 |
| OpenLLaMA-7B | original | 12.30 | 14.77 | 81.40 | 0.9555 |
| OpenLLaMA-7B | sft | 23.98 | 24.39 | 85.90 | 0.9920 |
| Gemma-2-2B-IT | original | 19.10 | 24.70 | 83.63 | 0.8925 |
| Gemma-2-2B-IT | sft | 24.63 | 23.61 | 86.28 | 0.9935 |
| Gemma-2-9B-IT | original | 21.13 | 26.67 | 84.06 | 0.8781 |
| Gemma-2-9B-IT | sft | 25.92 | 25.62 | 86.54 | 0.9918 |

## SFT − 원본

| 모델·조합 | 단계 | 자료 | 지표 | 평균 차이 | 증가 / 동률 / 감소 문항 |
|---|---|---|---|---:|---|
| GPT-2 base | sft−original | dolly | ROUGE-L | +8.52 | 374 / 31 / 95 |
| GPT-2 base | sft−original | dolly | METEOR | +6.01 | 288 / 20 / 192 |
| GPT-2 base | sft−original | dolly | BERTScore F1 | +4.33 | 474 / 0 / 26 |
| GPT-2 XL | sft−original | dolly | ROUGE-L | +12.63 | 389 / 23 / 88 |
| GPT-2 XL | sft−original | dolly | METEOR | +10.36 | 336 / 19 / 145 |
| GPT-2 XL | sft−original | dolly | BERTScore F1 | +5.15 | 468 / 0 / 32 |
| OPT-125M | sft−original | dolly | ROUGE-L | +9.68 | 326 / 34 / 140 |
| OPT-125M | sft−original | dolly | METEOR | +5.58 | 251 / 22 / 227 |
| OPT-125M | sft−original | dolly | BERTScore F1 | +4.24 | 459 / 0 / 41 |
| OPT-2.7B | sft−original | dolly | ROUGE-L | +12.86 | 365 / 24 / 111 |
| OPT-2.7B | sft−original | dolly | METEOR | +8.47 | 291 / 19 / 190 |
| OPT-2.7B | sft−original | dolly | BERTScore F1 | +4.65 | 455 / 0 / 45 |
| OpenLLaMA-3B | sft−original | dolly | ROUGE-L | +12.17 | 361 / 21 / 118 |
| OpenLLaMA-3B | sft−original | dolly | METEOR | +9.33 | 312 / 19 / 169 |
| OpenLLaMA-3B | sft−original | dolly | BERTScore F1 | +5.66 | 452 / 0 / 48 |
| OpenLLaMA-7B | sft−original | dolly | ROUGE-L | +11.68 | 344 / 17 / 139 |
| OpenLLaMA-7B | sft−original | dolly | METEOR | +9.61 | 312 / 14 / 174 |
| OpenLLaMA-7B | sft−original | dolly | BERTScore F1 | +4.50 | 428 / 0 / 72 |
| Gemma-2-2B-IT | sft−original | dolly | ROUGE-L | +5.52 | 275 / 8 / 217 |
| Gemma-2-2B-IT | sft−original | dolly | METEOR | -1.09 | 189 / 6 / 305 |
| Gemma-2-2B-IT | sft−original | dolly | BERTScore F1 | +2.65 | 383 / 0 / 117 |
| Gemma-2-9B-IT | sft−original | dolly | ROUGE-L | +4.79 | 252 / 4 / 244 |
| Gemma-2-9B-IT | sft−original | dolly | METEOR | -1.06 | 168 / 1 / 331 |
| Gemma-2-9B-IT | sft−original | dolly | BERTScore F1 | +2.48 | 357 / 0 / 143 |

## 교사 − 학생

| 모델·조합 | 단계 | 자료 | 지표 | 평균 차이 | 증가 / 동률 / 감소 문항 |
|---|---|---|---|---:|---|
| GPT-2 XL → GPT-2 base | original | dolly | ROUGE-L | +1.34 | 293 / 38 / 169 |
| GPT-2 XL → GPT-2 base | original | dolly | METEOR | +0.73 | 245 / 28 / 227 |
| GPT-2 XL → GPT-2 base | original | dolly | BERTScore F1 | +0.64 | 304 / 0 / 196 |
| GPT-2 XL → GPT-2 base | sft | dolly | ROUGE-L | +5.45 | 317 / 31 / 152 |
| GPT-2 XL → GPT-2 base | sft | dolly | METEOR | +5.08 | 311 / 24 / 165 |
| GPT-2 XL → GPT-2 base | sft | dolly | BERTScore F1 | +1.47 | 345 / 2 / 153 |
| GPT-2 XL → GPT-2 base | sft | self_inst | ROUGE-L | +5.29 | 137 / 31 / 74 |
| GPT-2 XL → GPT-2 base | sft | sni_11 | ROUGE-L | +8.95 | 1156 / 47 / 491 |
| GPT-2 XL → GPT-2 base | sft | vicuna | ROUGE-L | +2.72 | 54 / 0 / 26 |
| OPT-2.7B → OPT-125M | original | dolly | ROUGE-L | +1.26 | 283 / 34 / 183 |
| OPT-2.7B → OPT-125M | original | dolly | METEOR | +2.47 | 309 / 20 / 171 |
| OPT-2.7B → OPT-125M | original | dolly | BERTScore F1 | +1.03 | 347 / 0 / 153 |
| OPT-2.7B → OPT-125M | sft | dolly | ROUGE-L | +4.44 | 302 / 38 / 160 |
| OPT-2.7B → OPT-125M | sft | dolly | METEOR | +5.36 | 314 / 30 / 156 |
| OPT-2.7B → OPT-125M | sft | dolly | BERTScore F1 | +1.44 | 335 / 6 / 159 |
| OpenLLaMA-7B → OpenLLaMA-3B | original | dolly | ROUGE-L | +2.61 | 292 / 27 / 181 |
| OpenLLaMA-7B → OpenLLaMA-3B | original | dolly | METEOR | +2.22 | 274 / 22 / 204 |
| OpenLLaMA-7B → OpenLLaMA-3B | original | dolly | BERTScore F1 | +1.63 | 338 / 0 / 162 |
| OpenLLaMA-7B → OpenLLaMA-3B | sft | dolly | ROUGE-L | +2.12 | 258 / 32 / 210 |
| OpenLLaMA-7B → OpenLLaMA-3B | sft | dolly | METEOR | +2.51 | 265 / 26 / 209 |
| OpenLLaMA-7B → OpenLLaMA-3B | sft | dolly | BERTScore F1 | +0.48 | 259 / 10 / 231 |
| Gemma-2-9B-IT → Gemma-2-2B-IT | original | dolly | ROUGE-L | +2.03 | 300 / 16 / 184 |
| Gemma-2-9B-IT → Gemma-2-2B-IT | original | dolly | METEOR | +1.98 | 289 / 6 / 205 |
| Gemma-2-9B-IT → Gemma-2-2B-IT | original | dolly | BERTScore F1 | +0.43 | 318 / 2 / 180 |
| Gemma-2-9B-IT → Gemma-2-2B-IT | sft | dolly | ROUGE-L | +1.29 | 242 / 30 / 228 |
| Gemma-2-9B-IT → Gemma-2-2B-IT | sft | dolly | METEOR | +2.00 | 249 / 23 / 228 |
| Gemma-2-9B-IT → Gemma-2-2B-IT | sft | dolly | BERTScore F1 | +0.27 | 245 / 13 / 242 |

문항 수는 해당 자동 지표의 대응 차이이며 사람의 선호 승률이 아니다. 교사−학생 차이와 SFT−원본 차이, 이후 본 KD 학생−동일 SFT 학생의 개선을 구분한다.

## BERTScore 길이 감사

| 모델 | 단계 | 자료 | reference 영향 문항 | 응답 영향 문항 | 전체 reference 수 |
|---|---|---|---:|---:|---:|
| GPT-2 base | original | dolly | 11 | 0 | 500 |
| GPT-2 XL | original | dolly | 11 | 0 | 500 |
| OPT-125M | original | dolly | 11 | 0 | 500 |
| OPT-2.7B | original | dolly | 11 | 0 | 500 |
| OpenLLaMA-3B | original | dolly | 11 | 0 | 500 |
| OpenLLaMA-7B | original | dolly | 11 | 0 | 500 |
| Gemma-2-2B-IT | original | dolly | 11 | 0 | 500 |
| Gemma-2-9B-IT | original | dolly | 11 | 0 | 500 |
| GPT-2 base | sft | dolly | 11 | 0 | 500 |
| GPT-2 XL | sft | dolly | 11 | 0 | 500 |
| OPT-125M | sft | dolly | 11 | 0 | 500 |
| OPT-2.7B | sft | dolly | 11 | 0 | 500 |
| OpenLLaMA-3B | sft | dolly | 11 | 0 | 500 |
| OpenLLaMA-7B | sft | dolly | 11 | 0 | 500 |
| Gemma-2-2B-IT | sft | dolly | 11 | 0 | 500 |
| Gemma-2-9B-IT | sft | dolly | 11 | 0 | 500 |

BERTScore encoder의 native 상한은 특수 토큰 포함 512다. METEOR·ROUGE는 전체 reference를 사용한다. 세부 토큰 수·고유/전체 4-gram 수는 각 결과 CSV·JSON에 남긴다.

![원본과 SFT 생성 평가](dolly_before_after.png)

전체 지표: [scores.csv](scores.csv), 대응 차이와 입력 근거: [comparisons.json](comparisons.json). 최종 선정에는 후보의 외부 평가·보조 분류/보정 진단·응답 검토·실제 증류 경로 점검이 남아 있다.
