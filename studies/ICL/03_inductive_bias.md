# 03. Inductive Bias와 Feature Bias

> 상태: 초안 — ACL 2023 논문 1–2절 기반

## 1. 연구 질문

하나의 Demonstration 집합이 두 분류 규칙과 모두 일치할 때, LLM은 어느 규칙에 따라 Query를 분류하는가?

논문: Si et al., *Measuring Inductive Biases of In-Context Learning with Underspecified Demonstrations* (ACL 2023)

## 2. Inductive Bias의 개념적 정의

### 2.1 데이터만으로는 일반화 규칙이 유일하게 결정되지 않는다

학습 데이터셋을 다음과 같이 정의합니다.

\[
D=\{(x_i,y_i)\}_{i=1}^{N}
\]

입력과 출력을 매핑하는 가설 함수는 \(y=h(x)\)로 나타낼 수 있습니다. **유한한 데이터셋에 동일하게 부합하는 함수 \(h\)는 여러 개 존재할 수 있습니다.** 학습 데이터를 모두 정확히 설명하는 함수들의 집합은 다음과 같이 표현할 수 있습니다.

\[
\mathcal H_D=\left\{h\mid h(x_i)=y_i,\ \forall i=1,\ldots,N\right\}
\]

예를 들어,

\[
D=\{(1,2),(2,4),(3,6)\}
\]

에 대해

\[
h_1(x)=2x,\qquad
h_2(x)=2x+(x-1)(x-2)(x-3)
\]

는 모든 학습 입력에서 같은 출력을 만들지만 새로운 입력에서는

\[
h_1(4)=8,\qquad h_2(4)=14
\]

처럼 서로 다른 결과를 낼 수 있습니다. **관측된 데이터에 잘 부합한다는 사실만으로 관측하지 않은 입력에서의 규칙이 보장되지는 않습니다.**

### 2.2 모델의 구조는 어떤 함수가 가능한지 제한하거나 선호하게 만든다

학습 모델의 구조는 가설 공간(Hypothesis Space) \(\mathcal H\)에 영향을 줍니다.

- **Linear Regression:** \(\mathcal H_{\mathrm{linear}}=\{h(x)=ax+b\}\) — 선형 형태로 제한하는 *Restriction Bias*의 예.
- **Polynomial Regression:** \(\mathcal H_{\mathrm{poly}}=\{h(x)=ax^3+bx^2+cx+d\}\) — 더 넓은 종류의 함수를 고려할 수 있음.
- **CNN:** 국소적 연결과 Weight Sharing 등을 통해 영상의 공간적 구조에 관련된 *Architectural Inductive Bias*를 가짐.

따라서 동일한 \(D\)라도 **모델 구조가 다르면 표현 가능한 함수의 범위나 선호하는 일반화 방식이 달라질 수 있습니다.** 하지만 서로 다른 모델이 반드시 서로 다른 함수를 선택하는 것은 아닙니다. 예를 들어 선형회귀와 충분한 표현력을 가진 MLP 모두 \(h(x)=2x\)를 학습할 수 있습니다.

### 2.3 Inductive Bias는 모델 구조만으로 결정되지 않는다

최종적으로 학습되는 함수는 구조 외에도 Loss, Regularization, Optimizer, 초기화, 사전학습 등 여러 요소의 영향을 받습니다. 이를 개념적으로

\[
h^*=A(D;\mathcal H,\mathcal L,\mathcal O,\ldots)
\]

라고 표현할 수 있습니다.

- \(A\): 전체 학습 절차
- \(D\): 데이터셋
- \(\mathcal H\): 모델이 고려하는 가설 공간
- \(\mathcal L\): 손실 함수
- \(\mathcal O\): 최적화 방식
- \(h^*\): 학습 결과 얻어진 함수

이 표현은 특정 학습 알고리즘의 정리가 아니라 **어떤 요인들이 함수 선택에 영향을 주는지 나타낸 개념적 표기**입니다.

Inductive Bias의 대표적인 형태는 다음과 같습니다.

| 유형 | 핵심 의미 | 예시 |
|---|---|---|
| Restriction Bias | 허용하는 가설 자체를 제한 | 선형 함수만 고려 |
| Preference Bias | 가능한 가설 중 일부를 더 선호 | 복잡도 페널티를 통한 단순한 함수 선호 |
| Architectural Bias | 구조에 특정 데이터 처리 특성을 반영 | CNN의 지역적 연결과 Weight Sharing |
| Optimization/Pretraining에서 비롯되는 Bias | 같은 구조에서도 학습 이력과 절차에 따라 다른 일반화 경향 | 초기화·최적화·사전학습의 영향 |

**주의:** 'Inductive Bias = 단순한 규칙을 선호함'은 올바른 정의가 아닙니다. 단순성에 대한 선호는 여러 Inductive Bias 중 한 사례일 뿐입니다. 또한 **선택된 함수 \(h^*\) 자체가 Inductive Bias인 것이 아니라, 그 선택과 일반화에 작용한 가정이나 선호가 Inductive Bias**입니다.

### 2.4 ICL 논문과의 연결: Feature Bias

ICL에서는 추론 중 모델의 파라미터를 업데이트하지 않더라도, 사전학습된 모델이 동일한 Demonstration을 서로 다르게 해석할 수 있습니다.

\[
f_{\theta_1}(x\mid D_{\mathrm{demo}})
\ne
f_{\theta_2}(x\mid D_{\mathrm{demo}})
\]

여기서 \(\theta_1,\theta_2\)는 서로 다른 사전학습 상태의 모델 파라미터를 뜻하며, 위 부등식은 **가능한 현상**을 나타냅니다(언제나 다르다는 뜻은 아님).

이번 논문은 특히 동일한 Demonstration을 **Sentiment**와 **Topic**이라는 서로 다른 Feature로 모두 설명할 수 있을 때, 모델이 어느 Feature를 따라 일반화하는지를 측정합니다. 즉, 여러 형태의 Inductive Bias 중 **Feature Bias**를 관찰하는 연구입니다.

**한 문장 정의:** 학습 데이터만으로 일반화 규칙이 유일하게 정해지지 않을 때, 학습 모델의 구조와 학습 과정 등에 내재한 가정·선호가 어떤 가설을 선택하고 새로운 입력에 일반화할지 영향을 주는데, 이를 Inductive Bias라고 합니다.

---

## 3. 두 Feature Function

\[
h_1,h_2:\mathcal X\to\{0,1\}
\]

예시:
- \(h_1(x)\): 감정 분류 (긍정 1 / 부정 0)
- \(h_2(x)\): 주제 분류 (영화 1 / 음식 0)

Demonstration에서는

\[
y=h_1(x)=h_2(x)
\]

를 만족하게 구성합니다. 따라서 어느 규칙을 따라야 하는지 Context만으로는 특정하기 어렵습니다.

반면 평가용 입력은

\[
h_1(x)\ne h_2(x)
\]

인 경우로 구성해 모델이 선호하는 규칙을 구별합니다.

## 4. 평가 지표

\[
\operatorname{h\text{-}accuracy}(h)
=\frac{1}{|D_{\mathrm{test}}|}
\sum_{x\in D_{\mathrm{test}}}\mathbf{1}[f(x)=h(x)]
\]

여기서 \(f\)는 주어진 Prompt를 조건으로 한 모델의 분류 함수입니다.

## 5. 살펴볼 Intervention

- Instruction
- Semantic verbalizer
- Explanation
- Disambiguating examples

## 참고

- [논문 원문](https://aclanthology.org/2023.acl-long.632/)
- [강의 19:10부터](https://www.youtube.com/watch?v=9liSz8PDAp8&t=1150s)

실험별 수치와 해석은 논문 Figure/Table을 함께 검증하면서 추가합니다.
