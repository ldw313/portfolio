# 03. Inductive Bias와 Feature Bias

> 상태: 초안 — ACL 2023 논문 1–2절 기반

## 1. 연구 질문

하나의 Demonstration 집합이 두 분류 규칙과 모두 일치할 때, LLM은 어느 규칙에 따라 Query를 분류하는가?

논문: Si et al., *Measuring Inductive Biases of In-Context Learning with Underspecified Demonstrations* (ACL 2023)

## 2. 두 Feature Function

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

## 3. 평가 지표

\[
\operatorname{h\text{-}accuracy}(h)
=\frac{1}{|D_{\mathrm{test}}|}
\sum_{x\in D_{\mathrm{test}}}\mathbf{1}[f(x)=h(x)]
\]

여기서 \(f\)는 주어진 Prompt를 조건으로 한 모델의 분류 함수입니다.

## 4. 살펴볼 Intervention

- Instruction
- Semantic verbalizer
- Explanation
- Disambiguating examples

## 참고

- [논문 원문](https://aclanthology.org/2023.acl-long.632/)
- [강의 19:10부터](https://www.youtube.com/watch?v=9liSz8PDAp8&t=1150s)

실험별 수치와 해석은 논문 Figure/Table을 함께 검증하면서 추가합니다.
