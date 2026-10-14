# 중요한 것의 증류: 대규모 언어모델을 위한 신뢰도 기반 선택적 증류 — CaRE-KD

원제: *Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models*

저자: Ayan Sengupta, Vaibhav Seth, Tanmoy Chakraborty

소속: Indian Institute of Technology Delhi

기준: [arXiv:2609.36734v1](https://arxiv.org/abs/2609.36734v1), 2026-09-29

원문: [HTML](https://arxiv.org/html/2609.36734v1) · [PDF](https://arxiv.org/pdf/2609.36734v1)

**번역·이용 고지:** 원문은 [Creative Commons Attribution 4.0 International, CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)로 공개되어 있다. 이 문서는 한국어 번역·재구성과 학습용 해설을 추가한 비공식 자료다. 원저자들이 번역·해설을 작성하거나 보증한 것은 아니다. 원문과 라이선스, 원저자 표시는 재사용할 때도 유지해야 한다.

**범위:** 초록·본문 §1–6은 원문 순서를 따른 한국어 재구성 번역, 부록 A–G는 절별 번역과 증명 풀이, Table 1–10은 수치 전사, Figure 1–13은 한국어 캡션과 원문 링크다. 서지 정보는 영문으로 유지한다. 그림 이미지·축 표기까지 모두 번역한 대조 완역본은 아니다. PDF의 부록 A–G가 HTML에서는 §7–13으로 표시되므로 두 위치를 함께 적는다.

`[번역]`은 저자의 주장이다. `[해체 메모]`는 이 문서에서 추가한 해석·검증 질문이다. 원문에서 발견한 불일치는 번역에서 임의로 고치지 않고 끝부분에 따로 기록한다.

## 읽기 목표

**분포가 더 뾰족한 것, 정답을 더 잘 맞히는 것, 교사의 지도가 학생에게 유익한 것은 어떻게 연결되는가?**

처음에는 §2의 두 구성요소와 §3의 가정만 읽는다. 그다음 실험을 보고, 마지막에 부록의 구현 조건과 해체 메모로 돌아온다.

## 초록

[번역] 지식증류는 작은 용량의 학생 모델이 큰 교사 모델의 출력분포를 맞추도록 학습시킨다. 여기에는 교사가 믿을 만한 판단자라는 암묵적인 가정이 있다. 그러나 대규모 언어모델에서는 교사의 예측이 높은 엔트로피와 환각을 보일 수 있으며, 표준 증류가 잘 보정된 학생의 기존 예측을 손상시킬 수 있다.

저자들은 고정된 목적함수를 불확실성에 적응하는 최적화로 바꾸는 confidence gate 기반 프레임워크 CaRE-KD를 제안한다. 토큰 수준의 CaRE-Divergence는 교사와 학생의 confidence에 따라 Forward·Reverse KL을 전환한다. 배치 수준의 Revival은 교사가 학생보다 더 불확실한 조건에서 갱신을 억제한다. 저자들은 그라디언트 분석을 통해 이 두 수준의 설계가 고정 divergence로 재현하기 어려운 조건부 calibration을 만든다고 설명한다.

여덟 교사–학생 조합과 열한 benchmark에서 instruction following, chat alignment, 코드 생성, 수학 추론을 평가한다. 보고한 개선에는 평균 ROUGE-L 최대 +3.2, MBPP pass@1 +2.1, GSM8k accuracy +1.7, CollegeMath accuracy +1.8과 LLM 평가자의 사실성 점수 개선이 있다. Revival은 특정 손실에만 묶이지 않는 선택적 지도 필터로도 제안된다.

## 1. 서론과 선행 연구

[번역] 지식증류는 큰 교사와 작은 학생의 예측분포를 정렬하는 일반적인 방식이며, 흔히 Forward KL을 사용한다. 이 구성은 교사의 전체 확률질량을 학생이 덮도록 요구하므로 교사가 모든 입력에 일관되게 믿을 만한 지도를 준다는 가정을 포함한다.

교사가 잘 보정되어 있으면 효과적이지만, 높은 엔트로피·내적 불일치·환각이 나타나는 LLM에서는 이 가정이 약해진다. 저자들은 Forward KL이 믿기 어려운 긴 꼬리 예측에도 학생의 확률을 배정하게 하여 확신 있고 올바른 학생의 기존 예측을 손상시킬 수 있다고 설명한다. 원문은 이 실패 양상을 `zero-forcing`이라고 부른다.

Reverse KL이나 Skewed KL, α–β divergence 등은 교사의 꼬리에 대한 민감도를 낮추기 위해 최적화의 성향을 바꾼다. 그러나 고정된 성향은 어떤 토큰에서는 교사가 유익하고 다른 토큰에서는 불확실한 자기회귀 생성의 변화를 반영하기 어렵다.

저자들은 이를 **Fidelity Trap**, 즉 믿기 어려운 지도에도 충실하게 맞추다가 학생의 기존 능력을 손상시키는 문제로 설명한다. Figure 1의 분석에서는 교사가 자신 있을 때의 지도 효과와 불확실할 때의 효과가 다르며, 학생이 원래 더 낮은 불확실성을 갖는 영역에서도 고정 목적함수는 일치를 요구한다고 해석한다.

CaRE-KD의 두 구성요소는 이를 서로 다른 수준에서 다룬다. CaRE-Divergence는 confidence에 따라 토큰별 학습 기하를 바꾸고, Revival은 BALD 기반으로 지도를 받을지 결정한다. 저자들은 이 둘이 정보가 있는 지도는 흡수하고 믿기 어려운 지도는 거르는 데 상보적이라고 주장한다.

제시하는 기여는 다음과 같다.

1. 토큰별 confidence gate로 divergence의 기하를 적응적으로 바꾸는 CaRE-Divergence.
2. 교사의 epistemic uncertainty가 상대적으로 큰 영역의 갱신을 억제하는 Revival.
3. 그라디언트 분석과 entropy confidence가 BALD를 대리할 수 있는 조건.
4. 여러 생성 영역의 평가, 사실성 평가와 재현용 평가 프레임워크.

선행 연구는 logits·시퀀스 증류에서 on-policy 생성과 불완전한 교사 지도 문제로 확장되어 왔다. 저자들은 보조 모델이나 데이터 전체의 사전 제거 없이 학습 중 관측한 신호로 최적화 경로와 지도 수용 여부를 바꾼다는 점을 차별점으로 제시한다. 원문 각 divergence의 구현은 supplementary material에 있다고 명시한다.

### Figure 1 — 문제를 보여주는 관측

[캡션 번역] Dolly-15K에서 7B 교사로부터 Skewed-RKL을 사용해 증류한 OpenLLaMA-3B 학생. (a) 교사가 불확실한 경우 학생의 epistemic uncertainty가 거의 줄지 않으며 KS 거리 변화는 0.4%다. (b) 자신 있는 교사의 지도는 중간값 기준 +4%, 불확실한 교사의 지도는 +0.5%의 효과를 보인다. (c) 고정 목적함수는 믿기 어려운 교사에게도 일치를 강제한다. [원문 그림](https://arxiv.org/html/2609.36734v1#S1.F1)

## 2. 방법

[번역] CaRE-KD는 토큰 수준의 적응과 시퀀스 수준의 거절이라는 두 수준으로 구성된다. 첫 구성요소는 단일 forward의 confidence를 사용하고, 두 번째 구성요소는 epistemic uncertainty를 사용한다.

어휘 집합을 V라 하고, 같은 context에서의 교사와 학생의 다음 토큰 분포를 P와 Q라고 하자. 필요하면 온도를 적용한 분포를 사용한다. 정규화한 엔트로피 confidence는

$$
c(P)=1-\frac{H(P)}{\log|V|}\in[0,1]\tag{1}
$$

이며 H는 Shannon entropy다. c가 클수록 분포가 더 뾰족하다.

### 2.1. 토큰 수준: CaRE-Divergence

[번역] 교사가 제공하는 정보의 상대적 유용성에 따라 적절한 학습 기하가 달라진다는 관점에서, 고정 Forward KL을 다음 혼합으로 바꾼다.

$$
L_{\mathrm{CaRE}}=gD_F+(1-g)D_R,\quad
D_F=KL(P\|Q),\ D_R=KL(Q\|P).\tag{2}
$$

g는 토큰별 스칼라이며 Δc=c(P)−c(Q)에 따라 정한다.

$$
g_{\mathrm{hard}}=\mathbf1[c(P)\ge c(Q)],\qquad
g_{\mathrm{soft}}=\sigma(c(P)-c(Q)-m).\tag{3}
$$

σ는 sigmoid이고 m은 margin이다. 저자들은 g가 1에 가까우면 support를 넓게 덮는 Forward KL, 0에 가까우면 주요 mode에 집중하는 Reverse KL로 해석한다.

### Figure 2 — 두 gate와 학습 기하

[캡션 번역] 교사가 자신 있을 때는 Forward KL의 평균 탐색·recall 성향으로, 불확실할 때는 Reverse KL의 mode 탐색·precision 성향으로 전환하여 꼬리 잡음의 영향을 억제한다. [원문 그림](https://arxiv.org/html/2609.36734v1#S2.F2)

### 2.2. 시퀀스 수준: Revival

[번역] 단일 forward의 엔트로피는 입력 자체의 모호성과 환각에 관련된 epistemic uncertainty를 구분하지 못한다. 그래서 MC dropout으로 BALD를 추정한다.

$$
I_{\mathrm{BALD}}(x)=H[\mathbb E_\omega P(y|x,\omega)]
-\mathbb E_\omega[H(P(y|x,\omega))].\tag{4}
$$

이는 확률적으로 달라진 모델 상태 ω 사이의 예측 불일치를 측정한다. 교사·학생의 시퀀스 BALD를 I_T, I_S라고 하면

$$
M_{\mathrm{Revival}}=
\mathbf1[I_T>Q_\tau(I_T)\ \land\ I_S<Q_\tau(I_S)].\tag{5}
$$

Qτ는 각 모델의 과거 BALD에서 따로 계산한 rolling τ 분위수다. 절대 임계값을 대신하여 상대적인 분위수를 사용하면 학습 중 BALD의 규모가 달라지는 영향을 줄일 수 있다. 교사가 자기 분포에서 상대적으로 불확실하고 학생이 자기 분포에서 상대적으로 자신 있는 경우에 거절을 활성화한다. τ의 시간 변화도 실험한다.

배치 B에 대한 목적함수는

$$
L_{\mathrm{CaRE\text{-}KD}}=(1-M_{\mathrm{Revival}})
\frac1{|B|}\sum_{x\in B}L_{\mathrm{CaRE}}(P(x),Q(x)).\tag{6}
$$

M=1이면 갱신을 억제하고, 아니면 gated loss를 최적화한다.

[해체 메모] 식 (5)는 단순한 `I_T>I_S`가 아니다. 각 모델의 분위수 조건 두 개를 결합한다. **CaRE-Divergence는 어떻게 배울지, Revival은 해당 지도를 받아 갱신할지**를 다룬다. gate만 사용한 실험에서 Revival의 거절 효과까지 구현했다고 말할 수 없다.

## 3. 이론적 결과

[번역] 저자들은 CaRE-Divergence가 Skewed KL과 α–β divergence와 어떻게 다른지 그라디언트 수준에서 분석하고 entropy confidence와 BALD의 연결 조건을 정한다. 이 결과는 실험에서 관측한 동작을 설명하기 위한 것이며, 독립적인 최적화 보장이 아니다. 증명은 PDF 부록 C, HTML §9에 있다.

### 3.1. 명제 3.1: 그라디언트 분해

교사는 θ에 대해 고정되고, 학생 Qθ와 gate g(θ)는 미분 가능하다고 하자.

$$
\nabla_\theta L_{\mathrm{CaRE}}
=\underbrace{g\nabla_\theta D_F+(1-g)\nabla_\theta D_R}_{\text{learning signal}}
+\underbrace{(D_F-D_R)\nabla_\theta g}_{\text{selection signal}}.\tag{7}
$$

soft gate에 대해서는

$$
\nabla_\theta g=-g(1-g)\nabla_\theta c(Q_\theta)
$$

이므로 selection 항은

$$
-g(1-g)(D_F-D_R)\nabla_\theta c(Q_\theta)
$$

다.

[번역] 원문은 D_F>D_R인 경우 이 선택 신호가 학생 confidence를 높이는 방향으로 작용한다고 해석한다. 고정 skew나 고정 β에는 이런 confidence 의존 항이 없다. hard gate는 detach한 step function으로 ∇g=0이므로 토큰별 분기 선택만 남는다. 저자들은 hard의 분기 선택을 peak 성능, soft의 selection 항을 설정 변화에 대한 robustness와 연결한다. Revival이 활성화되어 전체 그라디언트가 0이면 거절한 지도 때문에 학생이 악화되지 않는다는 주장을 덧붙인다.

[해체 메모] 원문 설명의 부호와 “교사가 상대적으로 불확실함”의 연결은 뒤의 해체 메모에서 별도로 확인한다. 이 번역은 그 설명을 수학적으로 추가 증명한 것으로 취급하지 않는다.

### 3.2. 명제 3.2: 고정 SRKL·α–β와의 구분

[번역] 학생 confidence가 더 커서 g=0인 조건에서는 CaRE의 토큰별 그라디언트가 Skewed Reverse KL의 λ→1 극한과 일치한다. 그러나 CaRE는 토큰마다 g가 달라지므로 모든 토큰에 같은 λ 또는 β를 적용하는 전역 설정으로 일반적인 토큰별 동작을 재현할 수 없다고 주장한다.

### 3.3. 가정 3.3과 정리 3.4: entropy confidence와 BALD

[번역] **Sharpness Hypothesis:** cross-entropy로 학습한 LLM은 각 확률적 모델 상태에서 예측이 뾰족하여 H[P(y|x,ω)]≈0이라고 가정한다.

여기에 단일 forward 분포 P̂가 posterior predictive mean EωP(y|x,ω)에 가깝다는 근사를 더하면 c(P̂)를 크게 하는 것이 BALD를 작게 하는 것과 대략 연결된다. 이 조건에서는 단일 forward의 entropy confidence를 계산이 저렴한 대리 신호로 사용할 수 있다.

[해체 메모] “각 pass가 뾰족하다”와 “한 pass가 평균 예측에 가깝다”는 별도 가정이다. 첫 번째 조건만으로 두 번째 조건이 따라오지는 않는다.

### 3.4. 정리 3.5: 조건부 양방향 calibration

[번역] support 조건 아래에서 두 KL 분기의 전역 최소점은 Q=P다. 최적화 동안 분기가 고정되어 있고 Qθ가 P로 수렴한다고 **가정하면** H(Qθ)도 H(P)로 수렴한다. 학생의 초기 엔트로피가 교사보다 낮으면 어느 시점에는 증가해야 하고, 높으면 어느 시점에는 감소해야 한다. 매 단계의 단조 변화는 요구하지 않는다.

[해체 메모] 여기서 calibration은 교사의 엔트로피 수준으로 향하는 성질이다. 실제 정답에 대한 확률 보정이나 ECE 최적성을 자동으로 증명하지 않는다. 제한된 모델이 P를 표현할 수 있는지, 동적인 soft gate에서 같은 조건이 유지되는지는 별도 문제다.

## 4. 실험 구성과 결과

[번역] instruction following·chat alignment, 코드 생성, 수학 추론을 평가한다. 일반 instruction following에는 Dolly-15K를 사용하며 다음 모델 조합을 다룬다.

| 영역 | 교사 → 학생 | 학습 데이터 | 평가 |
|---|---|---|---|
| instruction following | GPT-2 XL → base / large | Dolly-15K | Dolly, Self-Instruct, S-NI, Vicuna |
| instruction following | OPT-2.7B → 125M | Dolly-15K | 위 네 benchmark |
| instruction following | Gemma-2-9B-Instruct → 2B-Instruct | Dolly-15K | 위 네 benchmark |
| instruction following | OpenLLaMA-7B → 3B | Dolly-15K | 위 네 benchmark |
| chat alignment | Qwen2-7B → 1.5B | UltraChat-200K | AlpacaEval, Evol-Instruct, UltraFeedback |
| 코드 | DeepSeek-Coder-6.9B-Instruct → 1.3B | WizardCoder | HumanEval, MBPP |
| 수학 | Qwen2.5-Math-7B-Instruct → 1.5B | MetaMathQA | GSM8k, CollegeMath |

instruction following과 chat에는 ROUGE-L과 GPT-5-Mini의 0–100 사실성 점수를 사용한다. 코드는 실행 기반 pass@1, 수학은 exact-match accuracy로 평가한다. ROUGE-L 개선의 유의성은 모델 조합 사이의 paired t-test로 평가한다. Table 1–10의 원문 수치는 문서 뒤에 모았다.

### 4.1. instruction following 결과

[번역] SGO는 학생 rollout, non-SGO는 teacher forcing 설정이다. Table 1·5의 셀에는 ROUGE-L / 사실성 점수가 적혀 있다.

SGO에서 CaRE-Divergence는 Forward KL 대비 GPT2-base +2.6/+4.3, GPT2-large +2.0/+3.4, OPT-125M +3.2/+3.0을 보고한다. GPT2-base에서는 Skewed-RKL보다 +0.3/+1.5 높다. OpenLLaMA에서는 일부 benchmark가 가장 높지만 평균은 최고 baseline에서 0.2 이내다. 원문은 행별 차상위 baseline에 대한 평균 개선의 paired t-test를 t=3.93, p=0.03으로 제시한다. SFT, KD, SeqKD, ImitKD, MiniLLM, GKD, Distillm, ABKD와의 추가 비교는 Table 6–8에 있다.

저자들은 큰 한 번의 개선보다 여러 조합에서 크게 실패하지 않는 robustness가 장점이라고 해석한다. 본문은 다섯 조합 중 네 조합에서 평균 ROUGE-L이 가장 높고 전체 평균은 20.1 대 Skewed-RKL 19.8이라고 기술한다. 고정 목적함수의 최악 차이는 Forward KL 4.1, Reverse KL 3.0, α–β 1.6, Skewed-RKL 0.6이며 CaRE는 최대 0.2라고 적는다. OpenLLaMA에서 평균 차이 0.18은 보고한 task별 seed SD보다 작아 통계적 동률로 취급한다고 설명한다. Gemma의 Dolly 17.0 대 19.7은 분명한 손실로 인정한다.

적용 범위는 교사 지도의 불완전성과 교사–학생 용량 차이에 달려 있다고 해석한다. 큰 압축비의 GPT2-XL→base, OPT-2.7B→125M에서 더 분명한 개선이 있고, 약 2.3배 차이의 OpenLLaMA에서는 이미 학생이 교사에 가까워 추가 선택의 이득이 작다고 설명한다.

non-SGO에서는 GPT2-base의 19.5/19.7을 제시하며 SRKL 17.7/17.7, FKL 16.1/15.6과 비교한다. 수렴 분석은 Figure 6–7에 있다.

### 4.2. LLM 평가자의 사실성

[번역] ROUGE-L만으로 환각을 판단하기 어렵기 때문에 GPT-5-Mini로 사실성을 추가 평가한다. 대부분의 모델 조합에서 품질 점수와 함께 사실성도 개선되며, GPT2-base에서 SRKL 대비 평균 +1.5, GPT2-large +2.2를 보고한다. Qwen2의 Evol-Instruct 사실성은 53.0으로 Distillm-2의 52.1보다 높다.

Revival이 실제 저사실성 영역을 선택하는지 확인한 OpenLLaMA-Dolly 분석에서는 교사 점수가 활성 영역 25.2, 비활성 영역 28.6이었다. 단측 Wilcoxon p=0.06, 다른 confidence 영역 비교의 Mann–Whitney p=0.26은 유의하지 않다. 저자들은 **Revival이 특정 저사실성 토큰을 선택한다는 기전 수준의 주장은 하지 않고**, 학생의 최종 사실성 점수가 개선되었다는 결과로 범위를 제한한다.

### 4.3. 손실과 독립적인 Revival 효과

[번역] Table 2는 Revival을 붙였을 때와 붙이지 않았을 때의 점수 차이를 제시한다. CaRE-Divergence는 OPT-125M에서 평균 +1.42, S-NI에서 +3.62를 보고한다. 일부 고정 손실도 개선되지만 차이는 변동성이 크며, GPT2-large의 RKL/Self-Instruct는 −1.78이다.

주장은 모든 셀의 개선이 아니라 평균적인 반응에 관한 것이다. CaRE 자체에도 Gemma/Vicuna −0.4, GPT2-large/Dolly −0.04가 있다. task별 delta의 one-sample t-test는 CaRE t=3.98, p<0.001이고, FKL t=−2.55, RKL t=1.24·p=0.25로 보고한다.

### 4.4. 전문 영역

[번역] Table 3에서 chat·코드·수학으로 범위를 확장한다. MBPP는 62.4, GSM8k는 73.9%, CollegeMath는 46.1%다. 비교 baseline 대비 각각 +2.1, +1.7, +1.8을 주요 개선으로 제시한다. 추가 GKD·Distillm 비교에서도 MBPP·GSM8k·CollegeMath가 가장 높고 HumanEval은 동률이라고 기술한다. chat의 두 추가 baseline 셀은 진행 중인 상태여서 “–”로 남겨 두었다.

저자들은 환각 한 단계가 코드·증명·논리의 실패로 이어지는 precision에 민감한 과제에서 선택적 지도의 이점이 크다고 해석한다.

### 4.5. 민감도와 ablation

[번역] GPT·OPT 계열의 설정 변화에서 skip 목표 약 50–70%가 낮은 필터 비율보다 좋고, 시간이 지날수록 선택성이 높아지는 increasing schedule이 좋은 결과를 보인다고 보고한다. MC 표본 수를 늘리면 안정성이 좋아지지만 N=3도 increasing schedule과 함께 충분한 성능을 낸다.

Hard gate는 최고 peak, Soft gate는 설정 변화에 조금 더 높은 robustness를 보인다. 불확실성 추정용 sampling temperature는 중간 값 약 3.0에서 좋은 결과를 보인다고 기술한다.

margin m이 작을 때 좋고, 큰 margin은 대부분의 토큰을 한 기하로 치우치게 한다. m은 sigmoid의 중심을 이동시키며 기울기를 바꾸지 않는다. m=0.05에서도 soft는 미분 가능하고 selection 항을 유지한다. 원문은 작은 margin의 crossover와 hard gate의 전환 위치를 연결하여 해석한다.

### Figure 3 — ablation의 비교 축

[캡션 번역] GPT2-base에서 (a) hard/soft gate, (b) MC dropout 수 N, (c) sampling temperature T, (d) skip 목표 τ, (e) constant/increasing/decreasing schedule, (f) soft margin m을 비교한다. [원문 그림](https://arxiv.org/html/2609.36734v1#S4.F3)

## 5. 논의와 한계

### 5.1. 학생 confidence와 교사 uncertainty의 분리

[번역] 저자들은 표준 증류에서 학생이 교사의 능력뿐 아니라 불확실성까지 따라가는 문제를 강조한다. Figure 4에서 교사의 BALD는 긴 꼬리를 보이지만 Revival을 사용한 학생은 기존 학생의 낮은 BALD에 가까운 집중된 분포를 유지한다고 설명한다. Self-Instruct의 평균 BALD는 교사 0.52, 학생 0.077이며, Vicuna·Dolly의 KS 거리는 0.5를 넘는다.

Skewed-RKL의 KS 변화는 0.840→0.836으로 0.4%, Revival 없는 CaRE-Divergence는 Dolly에서 0.840→0.864로 2.4%, Revival을 추가하면 7%의 변화를 보고한다. 토큰 손실이 주요 동력이고 Revival이 이를 키운다는 해석이다. Revival을 붙인 다른 divergence에서도 관련 패턴을 관측한다. 학습 중 거절 비율이 증가하는 현상을 폭넓은 초기 지도에서 점차 엄격한 필터링으로 이동하는 curriculum으로 해석한다.

### Figure 4 — 불확실성 분포

[캡션 번역] 네 instruction benchmark에서 교사는 높은 uncertainty의 긴 꼬리를, CaRE-KD 학생은 집중된 낮은 BALD를 보인다. Vicuna·Dolly의 KS 거리는 0.5를 넘는다. [원문 그림](https://arxiv.org/html/2609.36734v1#S5.F4)

### 5.2. 계산 비용

[번역] BALD는 배치마다 N번 확률적 forward를 요구하여 N에 거의 선형인 비용을 추가한다. Dolly/OpenLLaMA-3B의 한 epoch는 Revival 없이 5,520초, N=3에서 7,860초로 약 +42%이며 N=7은 약 +98%다. 거절한 배치의 backward를 건너뛰면 일부를 회복하며, skip 목표 70%에서 BALD 추가 비용의 약 12%를 회복한다고 보고한다.

이 약 40% 수치는 instruction following에 한정된다. N=3의 추가 비용은 chat 약 +106%, 코드 +170%, 수학 +148%다. 원문은 이를 더 이상 “modest”하다고 부르지 않겠다고 명시한다. 매우 큰 교사·긴 시퀀스·온라인 갱신에는 Revival이 적합하지 않을 수 있으며, 가정 3.3·정리 3.4가 정당화하는 저렴한 단일-pass entropy 대안을 제시한다. 저자들은 높은 압축비의 오프라인 증류에서는 배포 전에 한 번 발생하는 비용이라는 관점에서 수용 가능하다고 평가한다.

### Figure 5 — 시간 분석

[캡션 번역] Dolly-15K/OpenLLaMA-3B의 한 epoch. (a) MC 수 N에 따라 wall-clock이 증가한다. (b) skip 비율 τ를 늘리면 거절된 배치의 backward를 생략해 추가 비용을 줄인다. [원문 그림](https://arxiv.org/html/2609.36734v1#S5.F5)

### 5.3. 가정의 적용 범위

[번역] 다중-pass 거절의 비용은 계산 예산이 작은 경우에 부담이 된다. entropy와 BALD의 근사 연결은 각 MC 예측이 뾰족한 조건에서 더 타당하다. 원문은 Sharpness Hypothesis를 **선택 가능한 단일-pass 대안의 정당화에만 사용하며, 보고한 결과는 모두 완전한 MC-dropout BALD 추정을 사용한다**고 강조한다.

유효한 여러 continuation이 존재하는 열린 생성에서는 확률적 pass가 정답 후보 자체에 다르게 동의할 수 있어 이 가정이 약해질 수 있다. 높은 aleatoric uncertainty 영역으로 분석을 확장하는 것은 이후 과제로 남긴다.

## 6. 결론

[번역] 저자들은 confidence에 따라 토큰 수준의 최적화 기하를 바꾸고 epistemic rejection으로 지도 수용 여부를 정하는 CaRE-KD를 제안했다. 정보가 있는 지도는 전달하고 불확실한 교사 지도의 영향을 억제하여 기존 학생 예측을 보호한다는 설명이다. instruction following·코드·수학에서 고정 divergence보다 높은 품질과 calibration을 보고하며, 현대 LLM의 증류에서는 교사를 얼마나 모방할지뿐 아니라 언제 믿을지도 조정해야 한다고 결론내린다.

[해체 메모] 결론의 calibration·오류 보호 표현을 읽을 때 §4.2의 유의하지 않은 기전 검사와 부록 G의 ECE 해석을 반드시 함께 읽는다.

## 부록 A / HTML §7. 관련 연구

[번역] LLM 증류는 초기 logit matching에서 자기회귀·생성의 특성을 다루는 방식으로 확장되었다. 시퀀스·white-box 증류, exposure bias, 어휘 불일치, 생성 중 분포 변화가 주요 문제다. GKD는 학생의 생성에 교사 점수를 사용하는 on-policy 증류를, MiniLLM은 Reverse KL의 꼬리 민감도 완화를, 협력적 증류는 학습 진전에 따라 달라지는 교사·학생 역할을 다룬다. 강한 reasoning 모델에서의 증류와 불완전한 교사 지도의 붕괴 가능성도 관련 근거로 제시한다.

목적함수의 기하는 학생의 동작을 바꾼다. Forward KL은 support를 넓게 덮고 Reverse KL은 주요 mode에 집중하지만, 전자는 꼬리 위험, 후자는 다양성 저하라는 절충을 가질 수 있다고 설명한다. Skewed KL·α–β 등 고정 보간과 Pearson correlation·total variation·Rényi·Wasserstein 같은 대안은 표현력이 있어도 전역 설정이라는 한계가 있다고 평가한다.

CaRE-Divergence는 고정 보간 대신 토큰별 switch를 사용하고 Revival은 불확실한 지도를 거절한다. 저자들은 이 결합을 기존 고정 divergence와 uncertainty-aware 방법에 대한 차별점으로 제시한다.

## 부록 B / HTML §8. 알고리즘 상세

### B.1. MC dropout으로 BALD 추정

[번역] S개의 확률적 forward logits는 Z∈R^(S×B×L×V)다. 모델 배포 설정의 dropout에 의존하지 않고, BALD 추정 시 dropout layer를 training mode로 바꾸고 attention dropout을 p=0.1로 설정한다. 추정 후에는 원래 설정을 복원한다.

$$
P_{s,b,t}=softmax(Z_{s,b,t}),\tag{8}
$$

$$
\bar P=\frac1S\sum_sP_s,\tag{9}
$$

$$
H_{total}=H(\bar P),\qquad
H_{aleatoric}=\frac1S\sum_sH(P_s),\tag{10–11}
$$

$$
I_{b,t}=H_{total,b,t}-H_{aleatoric,b,t},\tag{12}
$$

$$
U_{BALD}^{(b)}=\frac1L\sum_tI_{b,t}.\tag{13}
$$

마지막 식은 토큰 평균으로 시퀀스 신호를 만든다.

### B.2. 추정이 상수로 붕괴하지 않는지 확인

[번역] meaningful dropout으로 학습하지 않은 모델의 MC uncertainty가 일정한 값으로 붕괴할 수 있다는 우려를 확인한다. 강제로 p=0.1을 적용한 GPT2-XL·OPT-2.7B에서 Dolly 500시퀀스, N=3을 검사한다. 모든 시퀀스의 BALD가 0이 아니고 top-1 확률도 pass 사이에 변한다. Table 4는 그 수치를 제시한다. 거절은 보정된 절대 BALD 대신 각 모델의 상대 분위수를 사용하므로 거친 추정에도 견딘다는 해석이다.

[해체 메모] 비영 신호가 존재한다는 사실과 그 신호가 실제 오류를 잘 구분한다는 사실은 별도다. 강제 dropout의 이론적 posterior 의미도 적용 모델과 학습 조건을 확인해야 한다.

### B.3. 수치 안정성

[번역] tempered softmax의 P는 수학적으로 full support를 가진다. 실제 구현에서는 log_softmax에서 log-probability를 직접 얻고 유한 floor로 clamp한다. α–β에서는 log 전에 작은 uniform mass도 섞는다. 보고한 실행에는 inf·NaN이 없었다.

## 부록 C / HTML §9. 이론 증명

### C.1. gate의 곱미분

[번역·수식 풀이] g가 미분 가능하고 detach하지 않았다고 하자. P는 고정한다.

$$
\nabla(gD_F+(1-g)D_R)
=\nabla(gD_F)+\nabla((1-g)D_R)\tag{14}
$$

$$
=g\nabla D_F+D_F\nabla g+(1-g)\nabla D_R-D_R\nabla g\tag{15}
$$

$$
=g\nabla D_F+(1-g)\nabla D_R+(D_F-D_R)\nabla g.\tag{16}
$$

Δ=c(P)−c(Qθ)−m이고 P·m이 θ에 대해 상수이므로

$$
\nabla g=\sigma'(\Delta)\nabla\Delta
=-g(1-g)\nabla c(Q_\theta).\tag{17}
$$

따라서 selection 항은

$$
(D_F-D_R)\nabla g=-g(1-g)(D_F-D_R)\nabla c(Q_\theta).\tag{18}
$$

detach한 gate에서는 ∇g=0이어서 이 항이 사라진다.

### C.2. Skewed Reverse KL의 그라디언트

[번역] mixture M=λP+(1−λ)Q, λ∈[0,1], Q=softmax(z)에 대해 KL(Q‖M)을 미분한다. Q 안에 있는 M도 Q에 의존한다.

$$
L=\sum_kQ_k\log Q_k-\sum_kQ_k\log M_k.
$$

첫 항의 미분은 1+log Q_i다. 두 번째 항의 곱미분에서 ∂M_k/∂Q_i=(1−λ)δ_ki를 사용하면

$$
v_i=\frac{\partial L}{\partial Q_i}
=\log\frac{Q_i}{M_i}+1-(1-\lambda)\frac{Q_i}{M_i}.\tag{19}
$$

softmax Jacobian을 곱하면

$$
\frac{\partial L}{\partial z_j}
=Q_j\left(v_j-\sum_kQ_kv_k\right).\tag{20}
$$

중심을 빼는 항에서 상수 1이 사라지므로

$$
\nabla_zL=Q\odot\left[
\log\frac Q M-(1-\lambda)\frac Q M
-\mathbb E_{y\sim Q}\left(\log\frac{Q_y}{M_y}-(1-\lambda)\frac{Q_y}{M_y}\right)\mathbf1
\right].\tag{21}
$$

### C.3. 거절 영역의 worst-case 주장

[번역] M_Revival=1이면 정의에 의해 해당 배치 손실이 0이고 그라디언트도 0이다. 저자들은 학생 파라미터가 그대로이므로 임의의 기준 분포 P*에 대한 KL(P*‖Q_new)=KL(P*‖Q_old)가 성립한다고 설명한다. 즉 거절한 noisy 지도 때문에 악화되지 않는다는 주장이다.

[해체 메모] 실제로 파라미터를 유지하려면 optimizer step도 적절히 건너뛰어야 한다. momentum·weight decay·다른 손실이 있으면 “이 항의 그라디언트가 0”만으로 모든 파라미터 갱신이 0이라는 결론을 내릴 수 없다. 이 문서는 공식 실행 코드의 거절 경로를 확인한 결과가 아니다.

### C.4. SRKL 극한과 동적 분기의 차이

[번역] λ→1이면 M→P, (1−λ)Q/M→0이므로 식 (21)은 Reverse KL의 그라디언트로 향한다. CaRE의 g=0 분기와 토큰별로 일치한다. 그러나 일반적으로 토큰마다 g_t가 다르면 전역 λ를 사용하는 SRKL의 합으로 같은 모든 동작을 만들 수 없다고 주장한다.

[해체 메모] 모든 입력에서 두 규칙이 같은 함수인가라는 질문과 특정 데이터에서 비슷한 결과를 낼 수 있는가는 다른 질문이다. 퇴화한 분포나 gate가 상수인 경우도 구분한다.

### C.5. entropy confidence의 조건부 대리 성질

[번역] BALD 분해는

$$
I_{BALD}(x)=H[\mathbb E_\omega P(y|x,\omega)]
-\mathbb E_\omega H[P(y|x,\omega)].\tag{22}
$$

Sharpness Hypothesis 아래에서 두 번째 항이 약 0이면

$$
I_{BALD}(x)\approx H[P(y|x)],\quad
P(y|x)=\mathbb E_\omega P(y|x,\omega).\tag{23}
$$

|V|가 고정이면 c(P̂)를 크게 하는 것은 H(P̂)를 작게 하는 것과 같다. 단일 pass P̂가 mean predictive distribution에 가깝다는 추가 근사 아래에서 이는 BALD를 작게 하는 것의 대리가 된다. 주장은 두 조건을 전제로 한다.

### C.6–C.7. 과신·저신뢰 학생의 조건부 엔트로피 이동

[번역] 분기가 고정되고 학생이 해당 분기의 전역 최소점에 수렴한다고 하자. 두 KL의 최소점 Q=P와 entropy의 연속성에서 H(Qθ_t)→H(P)가 따른다.

초기 H(Qθ_0)<H(P)이면 극한에 도달하기 위해 어느 순간 entropy가 증가해야 한다. 반대로 H(Qθ_0)>H(P)이면 어느 순간 감소해야 한다. 저자들은 이를 고정 분기 조건의 calibration 또는 sharpening으로 부른다.

[해체 메모] 이 증명은 모델 용량·학습 알고리즘이 수렴을 보장한다고 증명하지 않는다. 수렴을 가정한 후 entropy의 극한을 설명한다. 전 과정에서의 단조성도 결론에 포함되지 않는다.

## 부록 D / HTML §10. 실험 상세

### D.1. 학습 데이터

[번역] Dolly-15K는 약 15K의 single-turn instruction–response로 열린 질의응답·요약·분류·추출·창작을 포함한다. controlled instruction-following 증류의 주 데이터다. UltraChat-200K는 LLM이 생성한 multi-turn 대화로 alignment 학습에 사용한다. WizardCoder는 Evol-Instruct로 만든 프로그래밍 문제와 단계별 풀이로, 추가 filtering·reformatting 없이 코드 모델 학습에 사용한다고 기술한다. MetaMathQA는 산술·대수·여러 단계 추론을 포함하는 수학 문제와 reasoning trace로 math 모델을 학습한다.

### D.2. 평가 benchmark

[번역] Dolly Evaluation은 held-out instruction prompt, Self-Instruct는 다양한 자동 생성 지시, Super-Natural Instructions는 많은 과제 유형, Vicuna는 사용자형 대화 prompt를 포함한다. AlpacaEval은 선호 비교, Evol-Instruct는 복잡한 조합형 지시, UltraFeedback은 선호 기반 피드백을 다룬다. HumanEval·MBPP는 코드를 실행하여 검증한다. GSM8k는 초등 수준 다단계 수학 문제, CollegeMath는 대수·미적분·선형대수·미분방정식 등 대학 교재 문제다.

### D.3. 학습 설정

| 데이터·학생 | 학습률 | batch | epoch | LoRA rank |
|---|---:|---:|---:|---:|
| Dolly, 1B 미만 | 5×10⁻⁵ | 32 | 최대 20 | 16 |
| Dolly, 1B 초과 | 5×10⁻⁵ | 8 | 10 | 16 |
| UltraChat-200K | 5×10⁻⁵ | 원문 이 문단에 미기재 | 3 | 16 |
| WizardCoder | 5×10⁻⁵ | 원문 이 문단에 미기재 | 2 | 16 |
| MetaMathQA | 5×10⁻⁵ | 원문 이 문단에 미기재 | 2 | 16 |

[번역] self-attention과 MLP block의 모든 linear layer에 LoRA를 적용한다. checkpoint는 validation ROUGE-L로 고른다. MC dropout 수는 N∈{3,5,7}이며 기본은 3이다. skip τ는 constant·increasing·decreasing schedule을 평가한다. 불확실성 추정용 sampling temperature 후보는 {0.5,1.0,3.0,5.0}, 기본은 3.0이다.

현재 epoch e, 전체 scheduling horizon T에 대해

$$
s_{inc}(e)=\tau\frac{1-e^{-ke/T}}{1-e^{-k}},\quad k>0,\tag{24}
$$

$$
s_{dec}(e)=\tau\frac{e^{-ae/T}-e^{-a}}{1-e^{-a}},\quad a>0,\tag{25}
$$

$$
\lim_{a\to0}s_{dec}(e)=\tau(1-e/T).\tag{26}
$$

increasing은 초기 폭넓은 지도에서 더 엄격한 거절로, decreasing은 반대 방향으로 이동한다. 이 T는 **시간 horizon**이며 앞의 sampling·증류 온도와 구분한다.

### D.4–D.6. 평가·prompt·하드웨어

[번역] instruction 응답 생성은 temperature 0.8, top-p 0.95, 최대 512토큰이다. AlpacaEval의 reference는 text-davinci-003, Evol-Instruct·UltraFeedback은 gpt-3.5-turbo를 사용한다. 결과는 다섯 random seed의 평균이라고 적는다. 코드·수학은 greedy, 최대 1,024토큰이며 LM-Evaluation-Harness와 EvalPlus를 사용한다.

SGO에서는 학생이 응답을 생성하고 교사는 그 응답을 평가한다. gate와 loss는 같은 on-policy context에서 계산한다. non-SGO에서는 ground-truth 응답을 teacher forcing한 context에서 계산한다. prompt template는 Gu 외의 설정을 따른다. 모든 실험의 하드웨어는 NVIDIA A100 80GB 한 장이며, 평가 생성에는 vLLM을 사용한다.

[해체 메모] “다섯 random seed”의 반복 단위를 공식 실행 명세에서 추가 확인해야 한다. MiniLLM의 다섯 생성 반복 정의를 이 논문에 그대로 대입하지 않는다.

### D.7. 사실성 평가 prompt

[번역] instruction·reference·candidate를 GPT-5-Mini에 단일-turn, stateless로 제공한다. reference와 candidate 순서를 고정하고 temperature 0, 예시당 세 번의 독립 호출을 평균한다. 평가 prompt의 한국어 의미는 다음과 같다.

> 당신은 신뢰할 수 있는 reference와 비교하여 모델 응답의 사실적 정확성을 평가하는 공정한 평가자다. instruction, reference answer, candidate response가 주어진다. 다음 기준으로 candidate에 0–100의 점수를 부여하라.
>
> - 90–100: 모든 사실 주장이 정확하고 reference와 일치한다. 지어낸 entity·수치·날짜·관계가 없다.
> - 70–89: 대부분 정확하며 핵심 의미를 바꾸지 않는 작은 부정확성만 있다.
> - 40–69: 적어도 하나의 중요한 사실 오류 또는 근거 없는 주장이 답을 실질적으로 바꾼다.
> - 10–39: 여러 환각 사실·허구 인용·모순 때문에 대체로 잘못되었다.
> - 0–9: 무관하거나 과제를 거부하거나 전적으로 지어낸 응답이다.
>
> 길이·문체·형식 등 스타일 차이는 무시하고 사실의 차이만 벌점으로 반영하라. 0–100의 정수 하나만 반환하고 다른 설명은 쓰지 마라.

원문은 이 prompt를 모델·데이터·decoding 조건 사이에 고정한다고 명시한다. 위 번역은 독해용이며 재현 때 영어 prompt를 한국어로 바꾸라는 실행 지시가 아니다.

## 부록 E / HTML §11. 추가 결과

[번역] Table 5는 non-SGO, Table 6–8은 여러 baseline에 대한 전체 비교다. ROUGE-L과 LLM 사실성, 괄호 안 seed SD의 의미를 구분한다. 본문은 non-SGO 개선의 paired t-test를 t=12.84, p<0.005로 제시한다. Figure 6–7은 GPT-2와 OPT 학생의 validation CE·exact-match 흐름이며, Figure 8은 OPT-125M의 gate·MC 수·sampling 온도·skip·schedule·margin ablation이다. 수치는 뒤의 표 전사를 따른다.

## 부록 F / HTML §12. 상세 논의

### F.1. 교사의 불확실성과 학생 confidence 분리

[번역] 선택적 거절이 없으면 학생의 증류 후 BALD가 교사의 긴 꼬리를 따라갈 수 있다. Revival을 켜면 학생은 원래의 집중된 낮은 uncertainty에 더 가까운 분포를 유지하고 teacher–student KS 분리가 커진다고 설명한다. 고정 mode-seeking 목적함수도 균일하게 적용하면 믿기 어려운 토큰과의 정렬을 요구한다. CaRE-KD는 divergence 적응과 거절을 결합해 이런 영역의 교사 지도를 아예 무시할 수 있다는 해석이다.

저자들은 학습 기하인 “어떻게 배울지”와 epistemic reliability인 “언제 배울지”를 분리하는 것이 중요하다고 결론낸다.

### F.2. 학습 중 confidence 변화

[번역] M_Revival의 활성 비율을 추적한다. constant·increasing schedule은 초기에는 거절이 드물고 학습 중 점차 늘어 목표 비율 부근에서 안정화된다. increasing은 초기 폭넓은 지도에서 후반 엄격한 거절로 이어져 curriculum과 비슷한 동작을 보인다. decreasing은 후반에 불확실한 지도를 다시 받게 하여 상대 confidence의 성장을 제한한다고 해석한다.

MC 수를 늘리면 궤적이 조금 부드러워져도 schedule의 순서는 유지된다고 보고한다. 저자들은 이를 학생의 상대적 epistemic confidence 증가에 대한 근거로 해석한다.

[해체 메모] 마스크의 활성 빈도는 confidence만 아니라 움직이는 분위수·schedule·두 모델의 분포에도 의존한다. 활성 비율 증가가 능력 증가의 단독 증거인지 별도로 검토해야 한다.

## 부록 G / HTML §13. calibration·선택적 예측·선택 편향

### G.1. ECE를 어떻게 해석하는가?

[번역] Dolly/GPT2 학생의 ECE와 risk–coverage를 보고한다. 정오 기준은 단일 reference의 다음 토큰 exact match다. token accuracy는 약 0.43–0.47로 비슷하며 confidence가 주로 달라진다. Table 9에서 FKL의 평균 confidence/ECE는 0.67/0.22, 기본 CaRE는 0.79/0.34다.

저자들은 이 조건의 ECE를 sharpness에 많이 영향받는 지표로 해석하고, **실제 정답 기반 ECE에서 CaRE가 가장 잘 보정된 방법이라는 주장은 하지 않는다.** 증명한 성질은 학생 entropy가 교사의 entropy로 향하는 조건부 성질이라고 구분한다. uncertainty 분리의 근거로 Dolly의 KS 변화 7% 대 SRKL 0.4%를 제시한다. 모든 목적함수의 risk–coverage curve가 단조적이어서 confidence가 선택적 예측에 사용 가능하다고 설명한다.

[해체 메모] 본문 결론의 “better calibrated”를 실제 정답에 대한 일반적인 calibration 최적성으로 확대하면 이 부록과 충돌한다. 또한 softest CaRE의 ECE 0.28을 FKL 0.22와 “같은 수준”이라고 설명한 문장은 수치 차이를 그대로 남겨 확인해야 한다.

### G.2. 어려운 문제나 다양성을 선택적으로 버리는가?

[번역] OpenLLaMA-Dolly에서 Revival 활성 영역과 반대 영역의 LLM-judged 문제 난도·응답 다양성을 비교한다. 난도는 2.87 대 2.98, Mann–Whitney p=0.37이고, 다양성은 3.96 대 3.99, p=0.63이다. 저자들은 유의한 차이가 없다고 보고하며 어려운 문제 제거 또는 다양성 감소가 관측되지 않았다고 해석한다.

[해체 메모] 유의한 차이를 발견하지 못한 결과를 두 영역의 완전한 동등성 증명으로 바꾸어 적지 않는다.

## Figure 6–13 — 부록 그림 읽기

| 그림 | 한국어 캡션·볼 대상 | 원문 |
|---|---|---|
| 6 | GPT-2 계열의 validation cross-entropy·exact-match 흐름 | [그림](https://arxiv.org/html/2609.36734v1#S11.F6) |
| 7 | OPT 계열의 validation cross-entropy·exact-match 흐름 | [그림](https://arxiv.org/html/2609.36734v1#S11.F7) |
| 8 | OPT-125M의 gate·MC 수·온도·skip·schedule·margin ablation | [그림](https://arxiv.org/html/2609.36734v1#S11.F8) |
| 9 | Skewed-RKL에서 Revival 유무에 따른 OpenLLaMA-3B BALD 분포 | [그림](https://arxiv.org/html/2609.36734v1#S12.F9) |
| 10 | α–β divergence에서 Revival 유무에 따른 같은 학생의 BALD 분포 | [그림](https://arxiv.org/html/2609.36734v1#S12.F10) |
| 11 | skip 목표·schedule에 따른 학습 중 Revival 활성 배치 비율 | [그림](https://arxiv.org/html/2609.36734v1#S12.F11) |
| 12 | MC dropout 표본 수에 따른 Revival 활성 비율 | [그림](https://arxiv.org/html/2609.36734v1#S12.F12) |
| 13 | chat·코드·수학의 BALD 표본 수별 실행 시간 | [그림](https://arxiv.org/html/2609.36734v1#S12.F13) |

## 해체 메모: 원문을 그대로 믿기 전에 확인할 것

### 1. confidence는 무엇을 측정하는가?

c(P)는 분포의 집중도다. 실제 정답과의 일치를 직접 포함하지 않는다. `[학습용 예시]` 네 답 후보에 0.25씩 주고 다른 토큰에 0을 주는 분포를 V=248,320에서 측정하면 c≈0.8884다. 답 후보 사이에서는 전혀 구별하지 못해도 전체 어휘 entropy confidence는 높을 수 있다. 이 예시는 상대 confidence 비교가 반드시 무효라는 증명이 아니라, 측정 공간과 정답 신뢰도를 구분해야 함을 보여준다.

### 2. Soft의 범위와 Figure 2의 scale

m=0.05, Δc∈[−1,1]이면 단위 기울기 sigmoid의 범위는 약 0.2592–0.7211이고, Δc=0이면 0.4875다. Figure 2의 그림처럼 작은 confidence 차이에서 거의 0/1로 전환되는 곡선은 식 (3)의 단위 기울기와 scale이 다르게 보인다. 공식 실행 config에서 추가 scaling이 있는지 확인할 때까지 식·그림·구현을 같은 것으로 처리하지 않는다.

### 3. Soft·Hard는 연속성 하나만 다른가?

hard는 이진 분기와 zero margin, soft는 연속 가중·margin·selection gradient를 함께 포함한다. “부드럽기 때문에 더 좋다”라는 원인 설명에는 이 요소들을 구분하는 근거가 필요하다. g가 0.5 부근이라는 사실만으로 고정 50:50과 같은 그라디언트라고 말할 수도 없다.

### 4. selection gradient의 방향

식 (18)에서 D_F>D_R이면 **그라디언트 벡터**는 −∇c 쪽이다. 일반적인 gradient descent의 **갱신 방향**은 그 반대여서 +∇c 쪽이다. 본문이 “selection signal aligns with +∇c”라고 적은 설명은 두 방향을 구분해 읽어야 한다. 또한 D_F−D_R의 부호와 H(P)−H(Q)의 부호가 일반적으로 같은 의미라고 증명한 것은 아니다.

[학습용 반례, 직접 계산] P=[0.05,0.10,0.85], Q=[0.90,0.05,0.05]이면 H(P)=0.5182>H(Q)=0.3944지만 D_F=2.3330<D_R=2.4250이다. 교사가 상대적으로 더 높은 entropy라는 사실만으로 D_F>D_R을 대체할 수 없다.

### 5. 어느 calibration을 주장하는가?

정리 3.5는 분기 고정·수렴·support 조건 아래에서 교사의 entropy로 향하는 성질이다. 실제 정답에 대한 잘 보정된 확률이나 gate 자체의 최적성을 보장하지 않는다. 부록 G는 실제 ECE 최우수 주장을 명시적으로 제한한다.

### 6. 본문·표·캡션의 불일치

- Table 1 캡션은 최고 평균 ROUGE-L인 조합을 세 개라고 적지만, 본문은 네 개라고 적는다. 전사한 평균 열에서는 네 개로 읽힌다. 이 불일치를 삭제해 맞추지 않는다.
- §4.4에서 MBPP·GSM8k의 비교를 SRKL이라고 부르는 문장과 Table 3의 해당 행 이름 `Distillm2`를 대조해야 한다.
- 본문의 OpenLLaMA와 일부 표의 OpenLLaMA2 표기를 같은 checkpoint로 단정하지 않는다.
- Table 9에서 softest CaRE ECE는 0.28, FKL은 0.22다. “같은 ECE 수준”이라는 본문 표현과 구분한다.
- gate-only CaRE-Divergence와 Revival을 포함한 전체 CaRE-KD의 이름이 결과 설명에서 섞이는 곳이 있다. 공식 실험별 적용 요소를 확인해야 한다.

### 7. 아직 확정하지 않은 연구 질문

1. entropy confidence의 비교는 교사 지도가 유익한 입력을 구분하는가?
2. selection gradient의 추가 효과는 가중치 변화의 효과와 분리되는가?
3. 고정 분기의 이론을 동적인 gate와 제한된 학생 용량에 어떻게 연결할 수 있는가?
4. Revival의 선택 편향과 실제 오류 보호를 어떤 측정으로 확인할 수 있는가?
5. 보고한 robustness는 모델 조합·hyperparameter·학습 seed 중 무엇에 대한 것인가?

이 질문들의 이해와 근거 점검을 끝낸 뒤 연구 주제를 고른다. 이 문서에서 모델·데이터·실험을 확정하거나 실행하지 않았다.

## 주요 용어

| 용어 | 이 문서에서의 의미 |
|---|---|
| confidence | 정규화 entropy로 측정한 분포 집중도 |
| correctness | 외부 정답 기준과 예측이 일치하는지 |
| aleatoric uncertainty | 입력·출력 자체의 모호성과 관련된 불확실성 |
| epistemic uncertainty | 모델 상태의 변화에 따른 예측 불일치와 관련된 불확실성 |
| BALD | 평균 예측 entropy에서 pass별 entropy 평균을 뺀 신호 |
| detach | 해당 경로의 미분을 끊음 |
| SGO | Student-generated outputs; 학생의 생성 context |
| teacher forcing | 주어진 응답의 이전 토큰을 context로 사용 |
| rolling quantile | 과거 관측으로 계속 갱신하는 분위수 |
| calibration | 어떤 기준 분포·정오 정의를 사용하는지 함께 명시해야 하는 확률 보정 성질 |

## 원문 표 1–10과 참고문헌

아래는 CC BY 4.0 원문 표의 숫자·표기와 영문 서지를 전사한다. 표의 merged header는 Markdown 열로 풀고, 모델 이름의 표기를 원문대로 유지한다. 수치나 원문의 불일치를 교정한 데이터가 아니다. 그림의 숫자를 추정해 추가하지 않았다.

<!-- care-tables-and-references -->

### 원문 Table 1 — SGO의 ROUGE-L / LLM 사실성

| 모델 조합 | 손실 | Dolly | Self-Inst | Super-Nat | Vicuna | 평균 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | FKL | 23.4/21.2 | 10.7/15.6 | 19.0/16.4 | 15.0/13.5 | 17.0/16.7 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | RKL | 24.7/20.7 | 11.6/16.4 | 19.2/16.2 | 16.6/14.6 | 18.0/17.0 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | SRKL | 24.9/23.1 | 12.1/17.6 | 23.6 /21.9 | 16.5/15.3 | 19.3/19.5 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | $\alpha$ - $\beta$ Div | 23.9/23.1 | 11.5/16.4 | 21.9/18.3 | 16.1/15.0 | 18.4/18.2 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | CaRE-Div (Ours) | 25.6/24.1 | 12.9/20.1 | 22.8/ 24.4 | 17.3/15.3 | 19.6/21.0 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | FKL | 23.7/21.5 | 10.1/13.6 | 15.7/13.4 | 15.0/14.4 | 16.1/15.7 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | RKL | 19.3/14.7 | 9.9/14.3 | 16.3/13.0 | 15.0/11.3 | 15.1/13.3 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | SRKL | 24.1/20.5 | 11.1/15.0 | 19.9 /17.0 | 16.4 /15.3 | 17.9/16.9 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | $\alpha$ - $\beta$ Div | 23.0/20.1 | 10.7/14.6 | 16.7/15.2 | 15.5/13.3 | 16.5/15.8 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | CaRE-Div (Ours) | 25.4/25.5 | 11.6/16.7 | 19.4/ 18.0 | 16.0/ 16.1 | 18.1/19.1 |
| OPT-2.7B $\rightarrow$ OPT-125M | FKL | 22.4/20.7 | 9.6/14.1 | 17.1/14.5 | 15.5/12.9 | 16.1/15.5 |
| OPT-2.7B $\rightarrow$ OPT-125M | RKL | 24.8/22.7 | 11.2/15.8 | 22.2/18.4 | 16.4/15.9 | 18.6/18.2 |
| OPT-2.7B $\rightarrow$ OPT-125M | SRKL | 25.1/23.9 | 11.6/ 19.3 | 22.5/19.5 | 16.8/18.1 | 19.0/ 20.2 |
| OPT-2.7B $\rightarrow$ OPT-125M | $\alpha$ - $\beta$ Div | 24.9/23.6 | 11.2/16.5 | 21.7/18.3 | 16.7/13.9 | 18.6/18.1 |
| OPT-2.7B $\rightarrow$ OPT-125M | CaRE-Div (Ours) | 24.3/20.5 | 12.0 /18.7 | 23.9/21.4 | 16.7/13.5 | 19.3 /18.5 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | FKL | 18.4/19.6 | 10.4/22.0 | 19.2/15.3 | 17.1/16.0 | 16.3/18.2 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | RKL | 17.5/20.2 | 10.0/21.8 | 19.7 /19.1 | 20.1/ 21.9 | 16.8/20.7 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | SRKL | 19.7 /24.9 | 10.8/20.8 | 16.5/18.3 | 18.9/18.2 | 16.5/20.6 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | $\alpha$ - $\beta$ Div | 18.9/ 25.6 | 10.7/21.6 | 17.5/17.2 | 20.0/20.2 | 16.8/21.1 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | CaRE-Div (Ours) | 17.0/23.4 | 11.5/22.0 | 19.4/ 19.9 | 20.3 /21.0 | 17.1/21.6 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | FKL | 24.3/32.2 | 17.6/25.7 | 30.5/31.3 | 16.9/22.3 | 22.3/27.9 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | RKL | 29.1 /40.0 | 20.7/ 31.7 | 35.7/ 39.4 | 20.0 /27.8 | 26.4 /34.7 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | SRKL | 29.0/ 40.9 | 20.8/31.2 | 36.6/37.8 | 19.2/ 31.1 | 26.4/35.3 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | $\alpha$ - $\beta$ Div | 28.3/38.1 | 19.3/29.7 | 33.5/35.2 | 18.6/29.9 | 24.9/33.2 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | CaRE-Div (Ours) | 27.7/38.1 | 20.9 /29.8 | 37.3 /38.6 | 18.9/28.6 | 26.2/33.8 |

### 원문 Table 2 — Revival 추가에 따른 ROUGE-L 차이

| 학생 | 손실 | Dolly | Self-Inst | Super-Nat | Vicuna | 평균 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT2-base | FKL | 0.22 | 0.14 | -0.64 | 0.09 | -0.05 |
| GPT2-base | RKL | 0.32 | 0.61 | 3.51 | 0.12 | 1.14 |
| GPT2-base | SRKL | 0.41 | 0.54 | 1.64 | 0.30 | 0.72 |
| GPT2-base | $\alpha$ - $\beta$ Div | 1.11 | 0.80 | 0.94 | 0.67 | 0.88 |
| GPT2-base | CaRE | 0.35 | 0.94 | 1.26 | 1.04 | 0.90 |
| GPT2-large | FKL | -0.36 | -0.69 | -0.22 | -0.26 | -0.38 |
| GPT2-large | RKL | 0.35 | -1.78 | -1.44 | 1.75 | -0.28 |
| GPT2-large | SRKL | 0.43 | 0.70 | 1.09 | -0.56 | 0.41 |
| GPT2-large | $\alpha$ - $\beta$ Div | -0.44 | -0.35 | 0.70 | 0.68 | 0.15 |
| GPT2-large | CaRE | -0.04 | 0.71 | 1.06 | 0.10 | 0.46 |
| OPT-125M | FKL | -0.23 | -0.18 | -0.01 | -0.29 | -0.18 |
| OPT-125M | RKL | 0.29 | 0.57 | 0.75 | 0.71 | 0.58 |
| OPT-125M | SRKL | 0.27 | 0.61 | 1.98 | -0.76 | 0.52 |
| OPT-125M | $\alpha$ - $\beta$ Div | -0.42 | 0.35 | 0.05 | -0.43 | -0.11 |
| OPT-125M | CaRE | 0.55 | 1.12 | 3.62 | 0.38 | 1.42 |
| Gemma-2-2B-IT | FKL | -4.02 | -3.68 | -7.31 | -0.66 | -3.91 |
| Gemma-2-2B-IT | RKL | -0.48 | -2.68 | 1.54 | 0.4 | -0.3 |
| Gemma-2-2B-IT | SRKL | 1.48 | 1.89 | 1.56 | 0.99 | 1.48 |
| Gemma-2-2B-IT | $\alpha$ - $\beta$ Div | 1.37 | 2.05 | 10.17 | -1.26 | 3.08 |
| Gemma-2-2B-IT | CaRE | 3.6 | 2.61 | 5.79 | -0.4 | 2.90 |
| OLLaMA2-3B | FKL | 0.18 | -0.31 | -0.96 | 0.09 | -0.25 |
| OLLaMA2-3B | RKL | 0.02 | 0.18 | -0.29 | -0.03 | -0.04 |
| OLLaMA2-3B | SRKL | -0.15 | 0.67 | -1.48 | -0.10 | -0.26 |
| OLLaMA2-3B | $\alpha$ - $\beta$ Div | 0.24 | 1.14 | 2.16 | 0.43 | 0.99 |
| OLLaMA2-3B | CaRE | 0.27 | 0.26 | 1.82 | 0.57 | 0.73 |

### 원문 Table 3 — chat·코드·수학의 결과

| 방법 | Alpaca (R/LLM) | Evol (R/LLM) | Ultra (R/LLM) | HumanEval pass@1 | MBPP pass@1 | GSM8k | CollegeMath |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Student | 8.81/27.3 | 14.9/31.2 | 10.8/30.5 | 32.3 | 58.5 | 69.9 | 37.1 |
| Distillm2 | 16.3/43.3 | 26.5/52.1 | 22.1/ 49.3 | 43.3 | 60.3 | 72.2 | 44.2 |
| ABKD | 16.4 / 44.8 | 26.7/51.9 | 22.3/48.4 | 42.7 | 59.8 | 71.0 | 44.3 |
| GKD | – | – | – | 43.1 | 60.2 | 71.4 | 43.9 |
| Distillm | – | – | – | 42.9 | 60.0 | 71.7 | 44.1 |
| CaRE-KD (Ours) | 16.4 /44.4 | 26.8/53.0 | 22.6 / 49.3 | 43.3 | 62.4 | 73.9 | 46.1 |

### 원문 Table 4 — MC dropout 추정의 비퇴화 검사

| 교사 | N | 평균 BALD | BALD 범위 | 비영 BALD | 평균 top-1 확률 변화 |
| --- | --- | --- | --- | --- | --- |
| GPT2-XL | 3 | 0.049 | 0.013–0.107 | 500/500 (100%) | 0.074 |
| OPT-2.7B | 3 | 0.152 | 0.057–0.239 | 500/500 (100%) | 0.079 |

### 원문 Table 5 — non-SGO의 ROUGE-L / LLM 사실성

| 모델 조합 | 손실 | Dolly | Self-Inst | Super-Nat | Vicuna | 평균 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | FKL | 23.0/20.4 | 9.7/14.2 | 16.8/14.9 | 14.9/13.0 | 16.1/15.6 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | RKL | 23.0/20.1 | 10.1/14.9 | 17.7/15.3 | 15.5/13.7 | 16.6/16.0 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | SRKL | 23.5/21.7 | 11.1/16.2 | 21.5/18.8 | 14.8/14.2 | 17.7/17.7 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | $\alpha$ - $\beta$ Div | 23.0/21.6 | 10.3/15.3 | 19.7/17.0 | 14.3/13.8 | 16.8/16.9 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-base (0.1B) | CaRE-Div | 25.2/23.5 | 12.5/18.6 | 23.7/21.8 | 16.3/14.8 | 19.5/19.7 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | FKL | 22.5/19.5 | 10.1/13.5 | 14.8/12.9 | 14.4/12.8 | 15.4/14.7 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | RKL | 18.2/13.2 | 7.8/12.6 | 11.0/10.8 | 13.8/11.5 | 12.7/12.0 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | SRKL | 22.1/19.1 | 10.3/13.8 | 19.0/16.3 | 14.0/12.9 | 16.3/15.5 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | $\alpha$ - $\beta$ Div | 21.6/18.5 | 8.6/12.9 | 15.9/14.2 | 14.6/13.0 | 15.2/14.7 |
| GPT2-XL (1.5B) $\rightarrow$ GPT2-large (0.8B) | CaRE-Div | 25.0/24.1 | 11.5/15.6 | 19.6/16.9 | 15.4/13.7 | 17.9/17.6 |
| OPT-2.7B $\rightarrow$ OPT-125M | FKL | 22.0/19.8 | 9.4/13.8 | 17.0/14.7 | 15.3/13.3 | 15.9/15.4 |
| OPT-2.7B $\rightarrow$ OPT-125M | RKL | 25.1/23.0 | 11.5/15.7 | 22.3/18.5 | 17.0/16.1 | 19.0/18.3 |
| OPT-2.7B $\rightarrow$ OPT-125M | SRKL | 25.6/23.7 | 12.3/18.9 | 23.6/20.4 | 15.9/15.1 | 19.4/19.5 |
| OPT-2.7B $\rightarrow$ OPT-125M | $\alpha$ - $\beta$ Div | 24.5/22.8 | 12.1/17.3 | 21.5/18.0 | 16.7/14.0 | 18.7/18.0 |
| OPT-2.7B $\rightarrow$ OPT-125M | CaRE-Div | 23.9/20.6 | 11.3/17.0 | 22.3/19.2 | 16.4/13.7 | 18.5/17.6 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | FKL | 17.7/18.9 | 11.0/21.3 | 18.3/14.7 | 16.5/15.4 | 15.9/17.6 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | RKL | 16.9/19.5 | 9.2/21.0 | 18.9 /18.4 | 19.2/ 21.0 | 16.0/20.0 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | SRKL | 18.8 /23.6 | 9.9/20.2 | 15.6/17.5 | 17.8/17.3 | 15.5/19.7 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | $\alpha$ - $\beta$ Div | 18.1/ 24.2 | 9.8/20.9 | 16.7/16.5 | 19.1/19.5 | 15.9/20.3 |
| Gemma-2-9B-IT $\rightarrow$ Gemma-2-2B-IT | CaRE-Div | 16.3/22.1 | 10.7/ 21.3 | 18.6/ 19.0 | 19.5 /20.3 | 16.3/20.7 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | FKL | 23.9/31.4 | 17.8/25.1 | 29.2/30.7 | 17.1/22.1 | 22.0/27.3 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | RKL | 26.5/36.5 | 19.0/29.8 | 34.2/37.6 | 20.0 /27.3 | 24.9/32.8 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | SRKL | 28.2/39.4 | 20.0/ 30.7 | 35.6/ 38.2 | 19.1/ 30.0 | 25.7/34.6 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | $\alpha$ - $\beta$ Div | 26.8/36.8 | 18.6/29.2 | 32.6/35.1 | 18.8/29.3 | 24.2/32.6 |
| OpenLLaMA2-7B $\rightarrow$ OpenLLaMA2-3B | CaRE-Div | 27.4/36.9 | 20.2 /29.5 | 36.2 /38.0 | 18.5/28.3 | 25.6/33.2 |

### 원문 Table 6 — GPT2-XL → GPT2-base; 괄호는 원문 seed SD

| 방법 | Dolly | Self-Inst | Super-Nat | Vicuna | 평균 |
| --- | --- | --- | --- | --- | --- |
| SFT | 23.33 (0.22) | 10.56 (0.54) | 15.12 (0.47) | 17.08 (0.29) | 16.52 |
| KD [ 13 ] | 23.52 (0.19) | 11.23 (0.41) | 15.92 (0.37) | 20.68 (0.14) | 17.84 |
| SeqKD [ 1 ] | 23.38 (0.37) | 10.18 (0.18) | 15.01 (0.28) | 15.08 (0.12) | 15.91 |
| ImitKD [ 62 ] | 21.63 (0.51) | 10.85 (0.38) | 14.70 (0.29) | 17.94 (0.10) | 16.28 |
| MiniLLM [ 5 ] | 23.84 (0.26) | 12.44 (0.28) | 18.29 (0.36) | 22.62 (0.26) | 19.30 |
| GKD [ 2 ] | 23.75 (0.15) | 12.73 (0.24) | 16.64 (0.24) | 23.05 (0.23) | 19.04 |
| Distillm [ 6 ] | 24.86 (0.16) | 12.11 (0.27) | 16.52 (0.53) | 23.62 (0.42) | 19.27 |
| ABKD [ 7 ] | 23.93 (0.53) | 11.50 (0.30) | 16.11 (0.57) | 21.89 (0.33) | 18.36 |
| CaRE-KD (Ours) | 25.55 (0.24) | 12.88 (0.09) | 17.33 (0.57) | 22.80 (0.23) | 19.64 |

### 원문 Table 7 — OPT-2.7B → OPT-125M; 괄호는 원문 seed SD

| 방법 | Dolly | Self-Inst | Super-Nat | Vicuna | 평균 |
| --- | --- | --- | --- | --- | --- |
| SFT | 21.78 (0.19) | 8.09 (0.39) | 14.40 (0.17) | 13.45 (0.20) | 14.43 |
| KD [ 13 ] | 20.54 (0.38) | 9.16 (0.29) | 14.65 (0.47) | 15.79 (0.26) | 15.04 |
| SeqKD [ 1 ] | 20.72 (0.59) | 8.94 (0.37) | 13.56 (0.33) | 16.80 (0.36) | 15.01 |
| ImitKD [ 62 ] | 20.16 (0.19) | 8.95 (0.50) | 15.05 (0.52) | 14.72 (0.21) | 14.72 |
| MiniLLM [ 5 ] | 22.24 (0.32) | 9.92 (0.47) | 16.97 (0.49) | 16.58 (0.19) | 16.43 |
| GKD [ 2 ] | 22.46 (0.27) | 10.59 (0.36) | 16.25 (0.65) | 19.33 (0.20) | 17.16 |
| Distillm [ 6 ] | 25.07 (0.55) | 11.60 (0.23) | 16.75 (0.23) | 22.45 (0.29) | 18.97 |
| ABKD [ 7 ] | 24.88 (0.45) | 11.18 (0.77) | 16.72 (0.24) | 21.73 (0.13) | 18.63 |
| CaRE-KD (Ours) | 24.33 (0.19) | 12.03 (0.20) | 16.73 (0.29) | 23.93 (0.19) | 19.25 |

### 원문 Table 8 — OpenLLaMA2-7B → 3B; 괄호는 원문 seed SD

| 방법 | Dolly | Self-Inst | Super-Nat | Vicuna | 평균 |
| --- | --- | --- | --- | --- | --- |
| SFT | 25.11 (0.58) | 16.52 (0.56) | 16.33 (0.39) | 29.28 (0.45) | 21.81 |
| KD [ 13 ] | 20.95 (0.49) | 16.12 (0.80) | 15.39 (0.39) | 27.93 (0.26) | 20.10 |
| SeqKD [ 1 ] | 24.67 (0.47) | 15.83 (0.62) | 17.06 (0.47) | 29.06 (0.28) | 21.66 |
| ImitKD [ 62 ] | 24.53 (0.26) | 17.80 (0.80) | 17.36 (0.22) | 31.50 (0.12) | 22.80 |
| MiniLLM [ 5 ] | 27.88 (0.28) | 19.94 (0.58) | 20.50 (0.41) | 36.91 (0.28) | 26.31 |
| GKD [ 2 ] | 26.30 (0.31) | 19.56 (1.03) | 18.66 (0.34) | 35.71 (0.20) | 25.06 |
| Distillm [ 6 ] | 28.97 (0.36) | 20.84 (0.54) | 19.19 (0.36) | 36.64 (0.15) | 26.41 |
| ABKD [ 7 ] | 28.25 (0.35) | 19.25 (0.78) | 18.64 (0.26) | 33.53 (0.23) | 24.92 |
| CaRE-KD (Ours) | 27.74 (0.19) | 20.93 (0.90) | 18.94 (0.55) | 37.34 (0.34) | 26.23 |

### 원문 Table 9 — Dolly/GPT2의 confidence와 ECE

| 목적함수 | 평균 confidence | ECE |
| --- | --- | --- |
| Forward KL | 0.67 | 0.22 |
| CaRE-KD (softest configuration) | 0.68 | 0.28 |
| $\alpha$ – $\beta$ divergence | 0.75 | 0.28 |
| Reverse KL | 0.77 | 0.31 |
| Skewed-RKL | 0.79 | 0.31 |
| CaRE-KD (default temperature) | 0.79 | 0.34 |

### 원문 Table 10 — OpenLLaMA-Dolly의 선택 편향 검사

| LLM 평가 항목 | Revival 활성 영역 | 반대 영역 | Mann–Whitney p |
| --- | --- | --- | --- |
| Question hardness | 2.87 | 2.98 | 0.37 |
| Response diversity | 3.96 | 3.99 | 0.63 |

### 원문 참고문헌 — 영문 서지 유지

- [1] [1] Y. Kim and A. M. Rush, “Sequence-level knowledge distillation,” arXiv preprint arXiv:1606.07947 , 2016.

- [2] [2] R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem, “On-policy distillation of language models: Learning from self-generated mistakes,” in The Twelfth International Conference on Learning Representations , 2024.

- [3] [3] A. T. Kalai, O. Nachum, S. S. Vempala, and E. Zhang, “Why language models hallucinate,” arXiv preprint arXiv:2509.04664 , 2025.

- [4] [4] F. Sun, N. Li, K. Wang, and L. Goette, “Large language models are overconfident and amplify human bias,” arXiv preprint arXiv:2505.02151 , 2025.

- [5] [5] Y. Gu, L. Dong, F. Wei, and M. Huang, “Minillm: Knowledge distillation of large language models,” in The Twelfth International Conference on Learning Representations , 2024.

- [6] [6] J. Ko, S. Kim, T. Chen, and S.-Y. Yun, “Distillm: Towards streamlined distillation for large language models,” arXiv preprint arXiv:2402.03898 , 2024.

- [7] [7] G. Wang, Z. Yang, Z. Wang, S. Wang, Q. Xu, and Q. Huang, “ABKD: Pursuing a proper allocation of the probability mass in knowledge distillation via $\alpha$-$\beta$-divergence,” in Forty-second International Conference on Machine Learning , 2025. [Online]. Available: https://openreview.net/forum?id=vt65VjJakt

- [8] [8] S. Stanton, P. Izmailov, P. Kirichenko, A. A. Alemi, and A. G. Wilson, “Does knowledge distillation really work?” Advances in Neural Information Processing Systems , vol. 34, pp. 6906–6919, 2021.

- [9] [9] S. K. Ramesh, A. Sengupta, and T. Chakraborty, “On the generalization vs fidelity paradox in knowledge distillation,” in Findings of the Association for Computational Linguistics: ACL 2025 , W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 17 930–17 951. [Online]. Available: https://aclanthology.org/2025.findings-acl.923/

- [10] [10] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, S. Johnston, S. El-Showk, A. Jones, N. Elhage, T. Hume, A. Chen, Y. Bai, S. Bowman, S. Fort, D. Ganguli, D. Hernandez, J. Jacobson, J. Kernion, S. Kravec, L. Lovitt, K. Ndousse, C. Olsson, S. Ringer, D. Amodei, T. Brown, J. Clark, N. Joseph, B. Mann, S. McCandlish, C. Olah, and J. Kaplan, “Language models (mostly) know what they know,” 2022. [Online]. Available: https://arxiv.org/abs/2207.05221

- [11] [11] G. Prato, J. Huang, P. Parthasarathi, S. Sodhani, and S. Chandar, “Do large language models know how much they know?” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing , Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 6054–6070. [Online]. Available: https://aclanthology.org/2024.emnlp-main.348/

- [12] [12] N. Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel, “Bayesian active learning for classification and preference learning,” arXiv preprint arXiv:1112.5745 , 2011.

- [13] [13] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531 , 2015.

- [14] [14] A. Sengupta, S. Dixit, M. S. Akhtar, and T. Chakraborty, “A good learner can teach better: Teacher-student collaborative knowledge distillation,” in The Twelfth International Conference on Learning Representations , 2024. [Online]. Available: https://openreview.net/forum?id=Ixi4j6LtdX

- [15] [15] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Ding, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Chen, J. Yuan, J. Tu, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. You, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Zhou, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang, “Deepseek-r1 incentivizes reasoning in llms through reinforcement learning,” Nature , vol. 645, no. 8081, p. 633–638, Sep. 2025. [Online]. Available: http://dx.doi.org/10.1038/s41586-025-09422-z

- [16] [16] L. Fang, Y. Chen, W. Zhong, and P. Ma, “Bayesian knowledge distillation: A bayesian perspective of distillation with uncertainty quantification,” in Forty-first International Conference on Machine Learning , 2024. [Online]. Available: https://openreview.net/forum?id=knZ4NYzGUd

- [17] [17] C. He, Y. Ding, J. Guo, R. Gong, H. Qin, and X. Liu, “DA-KD: Difficulty-aware knowledge distillation for efficient large language models,” in Forty-second International Conference on Machine Learning , 2025. [Online]. Available: https://openreview.net/forum?id=NCYBdRCpw1

- [18] [18] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in International Conference on Machine Learning . PMLR, 2016, pp. 1050–1059.

- [19] [19] J. Ko, T. Chen, S. Kim, T. Ding, L. Liang, I. Zharkov, and S.-Y. Yun, “DistiLLM-2: A contrastive approach boosts the distillation of LLMs,” in Forty-second International Conference on Machine Learning , 2025. [Online]. Available: https://openreview.net/forum?id=rc65N9xIrY

- [20] [20] Y. Pawitan and C. Holmes, “Confidence in the reasoning of large language models,” Harvard Data Science Review , vol. 7, no. 1, pp. 2644–2353, 2025.

- [21] [21] M. Conover, M. Hayes, A. Mathur, J. Xie, J. Wan, S. Shah, A. Ghodsi, P. Wendell, M. Zaharia, and R. Xin, “Free dolly: Introducing the world’s first truly open instructiontuned llm,” 2023.

- [22] [22] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language models are unsupervised multitask learners,” 2019.

- [23] [23] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin et al. , “Opt: Open pre-trained transformer language models,” arXiv preprint arXiv:2205.01068 , 2022.

- [24] [24] G. Team, M. Riviere, S. Pathak, P. G. Sessa, C. Hardin, S. Bhupatiraju, L. Hussenot, T. Mesnard, B. Shahriari, A. Ramé et al. , “Gemma 2: Improving open language models at a practical size,” arXiv preprint arXiv:2408.00118 , 2024.

- [25] [25] X. Geng and H. Liu, “Openllama: An open reproduction of llama,” May 2023. [Online]. Available: https://github.com/openlm-research/open_llama

- [26] [26] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-instruct: Aligning language models with self-generated instructions,” in Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers) , 2023, pp. 13 484–13 508.

- [27] [27] Y. Wang, S. Mishra, P. Alipoormolabashi, Y. Kordi, A. Mirzaei, A. Naik, A. Ashok, A. S. Dhanasekaran, A. Arunkumar, D. Stap, E. Pathak, G. Karamanolakis, H. Lai, I. Purohit, I. Mondal, J. Anderson, K. Kuznia, K. Doshi, K. K. Pal, M. Patel, M. Moradshahi, M. Parmar, M. Purohit, N. Varshney, P. R. Kaza, P. Verma, R. S. Puri, R. Karia, S. Doshi, S. K. Sampat, S. Mishra, S. Reddy A, S. Patro, T. Dixit, and X. Shen, “Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 5085–5109. [Online]. Available: https://aclanthology.org/2022.emnlp-main.340/

- [28] [28] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Xu, J. Zhou, J. Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang, X. Wei, X. Ren, Y. Fan, Y. Yao, Y. Zhang, Y. Wan, Y. Chu, Y. Liu, Z. Cui, Z. Zhang, and Z. Fan, “Qwen2 technical report,” arXiv preprint arXiv:2407.10671 , 2024.

- [29] [29] N. Ding, Y. Chen, B. Xu, Y. Qin, S. Hu, Z. Liu, M. Sun, and B. Zhou, “Enhancing chat language models by scaling high-quality instructional conversations,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , 2023, pp. 3029–3051.

- [30] [30] X. Li, T. Zhang, Y. Dubois, R. Taori, I. Gulrajani, C. Guestrin, P. Liang, and T. B. Hashimoto, “Alpacaeval: An automatic evaluator of instruction-following models,” 2023.

- [31] [31] C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, Q. Lin, and D. Jiang, “Wizardlm: Empowering large pre-trained language models to follow complex instructions,” in The Twelfth International Conference on Learning Representations , 2024.

- [32] [32] G. Cui, L. Yuan, N. Ding, G. Yao, B. He, W. Zhu, Y. Ni, G. Xie, R. Xie, Y. Lin, Z. Liu, and M. Sun, “Ultrafeedback: boosting language models with scaled ai feedback,” in Proceedings of the 41st International Conference on Machine Learning , ser. ICML’24. JMLR.org, 2024.

- [33] [33] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, F. Luo, Y. Xiong, and W. Liang, “Deepseek-coder: When the large language model meets programming – the rise of code intelligence,” 2024. [Online]. Available: https://arxiv.org/abs/2401.14196

- [34] [34] Z. Luo, C. Xu, P. Zhao, Q. Sun, X. Geng, W. Hu, C. Tao, J. Ma, Q. Lin, and D. Jiang, “Wizardcoder: Empowering code large language models with evol-instruct,” in The Twelfth International Conference on Learning Representations , 2024. [Online]. Available: https://openreview.net/forum?id=UnUwSIgK5W

- [35] [35] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba, “Evaluating large language models trained on code,” 2021. [Online]. Available: https://arxiv.org/abs/2107.03374

- [36] [36] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le et al. , “Program synthesis with large language models,” arXiv preprint arXiv:2108.07732 , 2021.

- [37] [37] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu, “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115 , 2024.

- [38] [38] L. Yu, W. Jiang, H. Shi, J. YU, Z. Liu, Y. Zhang, J. Kwok, Z. Li, A. Weller, and W. Liu, “Metamath: Bootstrap your own mathematical questions for large language models,” in The Twelfth International Conference on Learning Representations .

- [39] [39] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano et al. , “Training verifiers to solve math word problems,” arXiv preprint arXiv:2110.14168 , 2021.

- [40] [40] Z. Tang, X. Zhang, B. Wang, and F. Wei, “Mathscale: Scaling instruction tuning for mathematical reasoning,” arXiv preprint arXiv:2403.02884 , 2024.

- [41] [41] S. Sun, Y. Cheng, Z. Gan, and J. Liu, “Patient knowledge distillation for bert model compression,” arXiv preprint arXiv:1908.09355 , 2019.

- [42] [42] C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and T. Pfister, “Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes,” in Findings of the Association for Computational Linguistics: ACL 2023 , 2023, pp. 8003–8017.

- [43] [43] Q. Zhong, L. Ding, L. Shen, J. Liu, B. Du, and D. Tao, “Revisiting knowledge distillation for autoregressive language models,” arXiv preprint arXiv:2402.11890 , 2024.

- [44] [44] M. Li, F. Zhou, and X. Song, “Bild: Bi-directional logits difference loss for large language model distillation,” in Proceedings of the 31st International Conference on Computational Linguistics , 2025, pp. 1168–1182.

- [45] [45] K. Saadi and D. Wang, “TASKD-LLM: Task-aware selective knowledge distillation for LLMs,” in ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy , 2025. [Online]. Available: https://openreview.net/forum?id=QQBfoVJWY2

- [46] [46] X.-C. Li, W.-S. Fan, S. Song, Y. Li, S. Yunfeng, D.-C. Zhan et al. , “Asymmetric temperature scaling makes larger networks teach well again,” Advances in neural information processing systems , vol. 35, pp. 3830–3842, 2022.

- [47] [47] K. Zheng and E.-H. Yang, “Knowledge distillation based on transformed teacher matching,” arXiv preprint arXiv:2402.11148 , 2024.

- [48] [48] S. Sun, W. Ren, J. Li, R. Wang, and X. Cao, “Logit standardization in knowledge distillation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2024, pp. 15 731–15 740.

- [49] [49] Y. Gu, L. Dong, F. Wei, and M. Huang, “Minillm: Knowledge distillation of large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2306.08543

- [50] [50] G. Kim, D. Jang, and E. Yang, “Promptkd: Distilling student-friendly knowledge for generative language models via prompt tuning,” in Findings of the Association for Computational Linguistics: EMNLP 2024 , 2024, pp. 6266–6282.

- [51] [51] T. Huang, S. You, F. Wang, C. Qian, and C. Xu, “Knowledge distillation from a stronger teacher,” Advances in Neural Information Processing Systems , vol. 35, pp. 33 716–33 727, 2022.

- [52] [52] Y. Wen, Z. Li, W. Du, and L. Mou, “F-divergence minimization for sequence-level knowledge distillation,” arXiv preprint arXiv:2307.15190 , 2023.

- [53] [53] W. F. Huang, J. Tao, C. Deng, M. Fan, W. Wan, Q. Xiong, and G. Piao, “Rényi divergence deep mutual learning,” in Machine Learning and Knowledge Discovery in Databases: Research Track , D. Koutra, C. Plant, M. Gomez Rodriguez, E. Baralis, and F. Bonchi, Eds. Cham: Springer Nature Switzerland, 2023, pp. 156–172.

- [54] [54] J. Lv, H. Yang, and P. Li, “Wasserstein distance rivals kullback-leibler divergence for knowledge distillation,” Advances in Neural Information Processing Systems , vol. 37, pp. 65 445–65 475, 2024.

- [55] [55] C. Guo, S. Zhong, X. Liu, Q. Feng, and Y. Ma, “Why does knowledge distillation work? rethink its attention and fidelity mechanism,” Expert Systems with Applications , vol. 262, p. 125579, 2025.

- [56] [56] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. Dillon, B. Lakshminarayanan, and J. Snoek, “Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift,” Advances in neural information processing systems , vol. 32, 2019.

- [57] [57] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality,” March 2023. [Online]. Available: https://lmsys.org/blog/2023-03-30-vicuna/

- [58] [58] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations , 2022. [Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9

- [59] [59] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou, “The language model evaluation harness,” 07 2024. [Online]. Available: https://zenodo.org/records/12608602

- [60] [60] J. Liu, C. S. Xia, Y. Wang, and L. Zhang, “Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation,” Advances in Neural Information Processing Systems , vol. 36, pp. 21 558–21 572, 2023.

- [61] [61] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles , 2023.

- [62] [62] A. Lin, J. Wohlwend, H. Chen, and T. Lei, “Autoregressive knowledge distillation through imitation learning,” 2020. [Online]. Available: https://arxiv.org/abs/2009.07253

- [63] [63] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in International conference on machine learning . PMLR, 2017, pp. 1321–1330.

- [64] [64] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” Advances in neural information processing systems , vol. 30, 2017.
