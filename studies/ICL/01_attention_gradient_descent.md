# 01. Attention과 Gradient Descent의 관계

> 상태: 학습 예정 — 강의 04:27–14:05

## 학습 목표

Gradient Descent로 학습된 Linear Layer의 예측을 과거 학습 샘플에 대한 Dot-Product 연산으로 전개하고, Attention의 Query/Key/Value와 대응시킵니다.

## 1. 시작 수식: Linear Layer

\[
f_W(x)=Wx
\]

일정한 학습률 \(\eta\)와 단계별 Gradient \(\nabla_W\mathcal{L}_i\)에 대해,

\[
W_T=W_0-\eta\sum_{i=0}^{T-1}\nabla_W\mathcal{L}_i
\]

입니다.

## 2. 다음에 유도할 내용

- \(\nabla_W\mathcal L_i\)의 외적 구조
- 테스트 입력 \(x\)에서 \(W_Tx\)의 전개
- 과거 입력을 Key, 테스트 입력을 Query, 학습 신호를 Value로 해석하는 조건
- Linear Attention과 Softmax Attention의 차이
- Meta-Optimizer 해석의 가정과 한계

**주의:** 위 내용은 전체 유도가 아니라 출발점입니다. 실제 Transformer가 추론 중 가중치를 업데이트한다는 의미가 아닙니다.

## 참고

- [강의 05:54부터](https://www.youtube.com/watch?v=9liSz8PDAp8&t=354s)
- [The Dual Form of Neural Networks Revisited](https://proceedings.mlr.press/v162/irie22a.html)
- [Why Can GPT Learn In-Context?](https://arxiv.org/abs/2212.10559)
