# 00. Transformer 기반 ICL 기초

> 상태: 초안 — 개념 정의 중심

## 1. In-Context Learning 정의

학습된 모델의 파라미터 \(\theta\)를 추론 시 고정한 채, 앞선 시연(Context)을 조건으로 새로운 Query에 대한 예측을 수행하는 방식입니다.

\[
p_\theta(y_q\mid C,x_q)
\]

- \(C=\{(x_i,y_i)\}_{i=1}^K\): Demonstration 또는 Context
- \(x_q\): Query 입력
- \(y_q\): Query의 목표 출력
- \(\theta\): 사전학습된 모델 파라미터

추론 시 \(\Delta\theta=0\)이지만, Context에 따라 활성화와 예측 분포는 바뀔 수 있습니다.

## 2. Task 기호 \(\tau\)

\(\tau\)는 특정 데이터 생성 규칙이나 과제를 나타내는 **기호**입니다. \(\tau\sim p_{\mathrm{train}}(\tau)\)는 학습에 사용하는 Task 분포에서 하나를 샘플링한다는 뜻이며, 모든 일반 LLM이 반드시 명시적 Task 샘플링으로 학습된다는 뜻은 아닙니다.

예: \(y=ax\)에서 \(a=2\), \(a=3\)은 서로 다른 Task로 간주할 수 있습니다.

## 3. Decoder-only Transformer에서의 처리

프롬프트는 Context와 Query를 이어 붙인 Token Sequence로 구성됩니다.

\[
T=\operatorname{Tokenizer}([x_1,y_1,\dots,x_K,y_K,x_q])
\]

Causal Self-Attention은 각 Token이 자신과 이전 Token의 정보를 활용하도록 합니다.

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
\]

여기서 \(M\)은 미래 위치를 가리는 Causal Mask입니다.

**주의:** Context Encoder는 ICL의 필수 모듈이 아닙니다. 단어 Token을 함께 처리하는 Transformer와 별도의 Context Encoder를 쓰는 모델은 서로 다른 구현 방식입니다.

## 4. 다음 학습 질문

- Gradient Descent를 적용한 Linear Layer의 예측을 어떻게 과거 샘플에 대한 합으로 나타내는가?
- 그 합과 Dot-Product Attention의 대응은 어떤 조건에서 성립하는가?
- 이 대응이 실제 파라미터 업데이트를 의미하지 않는 이유는 무엇인가?
