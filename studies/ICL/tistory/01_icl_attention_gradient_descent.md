# [ICL 스터디 1] Transformer는 어떻게 Context에서 학습할까? — Gradient Descent와 Attention의 Dual Form

> **학습 기록** | In-Context Learning · Transformer · Gradient Descent · Attention  
> **범위:** ICL 기초, Dual Form 수학적 유도, Meta-Optimizer 해석  
> **태그:** In-Context Learning, ICL, Transformer, Attention, Gradient Descent, Meta-Learning, Deep Learning

## 1. In-Context Learning이란?

일반적인 Fine-tuning은 특정 Task의 데이터를 이용해 모델 파라미터를 업데이트한다. 반면 **In-Context Learning(ICL)**은 이미 학습된 모델에 몇 개의 Demonstration(입력–정답 예시)을 제공하고, 파라미터를 변경하지 않은 채 새로운 Query에 응답하는 방식을 뜻한다.

예를 들어 다음과 같은 Prompt를 생각하자.

\`\`\`text
English: cat
Korean: 고양이

English: dog
Korean: 개

English: apple
Korean:
\`\`\`

모델은 앞선 예시를 Context로 이용해 다음 출력을 생성할 수 있다. 여기서 중요한 질문은 **파라미터 업데이트가 없는데도 어떻게 Context에 적응하는가?**이다.

수학적으로 Context와 Query를 다음과 같이 정의하자.

$$
C=\{(x_i,y_i)\}_{i=1}^{K}, \qquad x_q=\text{Query}.
$$

ICL의 목표는 다음 조건부 예측을 수행하는 것이다.

$$
\boxed{p_\theta(y_q\mid C,x_q)}
$$

- $C$: Demonstration으로 구성된 Context
- $x_q$: 새 Query 입력
- $y_q$: 예측할 출력
- $\theta$: 사전 학습된 모델 파라미터

추론 중 모델 파라미터는 고정된다.

$$
\Delta\theta=0.
$$

단, Context가 바뀌면 Transformer 내부 Hidden State와 Attention 값은 달라질 수 있다. **모델 파라미터가 변하지 않는 것과 모델의 중간 계산 결과가 변하지 않는 것은 서로 다른 이야기**다.

### Task를 나타내는 $\tau$는 무엇인가?

ICL을 설명할 때 자주 등장하는 식이 있다.

$$
\tau\sim p_{\mathrm{train}}(\tau).
$$

여기서 $\tau$(tau)는 하나의 Task, $p_{\mathrm{train}}$은 학습용 Task 분포, $\sim$은 그 분포에서 샘플링한다는 뜻이다.

예를 들어 $y=ax$에서 $a=2$와 $a=3$을 서로 다른 Task로 설정할 수 있다. 이처럼 Task를 명시적으로 샘플링하는 것은 **ICL 학습을 설명하기 위한 한 가지 방법**이지, 모든 LLM의 일반적인 사전학습이 이런 형식을 사용한다는 의미는 아니다.

## 2. Transformer에서는 Context가 어떻게 전달될까?

Decoder-only Transformer는 Context와 Query를 하나의 Token Sequence로 처리한다.

$$
S=[x_1,y_1,\ldots,x_K,y_K,x_q].
$$

각 Token 표현에서 Query, Key, Value를 계산한다.

$$
Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V.
$$

Causal Self-Attention은 다음과 같다.

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V.
$$

$M$은 미래 Token을 참조하지 못하게 하는 Causal Mask다. Query 위치의 Hidden State는 앞선 Demonstration Token의 정보를 Attention을 통해 반영할 수 있다.

이제 더 흥미로운 질문으로 넘어가자. **Attention의 계산이 Gradient Descent와 연결될 수 있을까?**

## 3. Linear Layer에서 Gradient Descent 유도

여기부터는 Irie et al.의 *The Dual Form of Neural Networks Revisited*에서 다루는 수학적 관계를 공부한다.

먼저 Bias가 없는 Linear Layer를 정의하자.

$$
\hat y_t=W_{t-1}x_t.
$$

기호는 다음과 같다.

| 기호 | 의미 |
|---|---|
| $t$ | 학습 업데이트 단계 |
| $x_t\in\mathbb R^{d_{\mathrm{in}}}$ | $t$번째 학습 입력 |
| $y_t^*$ | $t$번째 목표 출력 |
| $\hat y_t\in\mathbb R^{d_{\mathrm{out}}}$ | $t$번째 예측 출력 |
| $W_{t-1}\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$ | 업데이트 전 Weight |
| $\mathcal L_t$ | Loss |
| $g_t=\nabla_{\hat y_t}\mathcal L_t$ | **출력에 대한** Loss Gradient |
| $\eta_t$ | Learning Rate |
| $e_t=-\eta_tg_t$ | 부호 및 학습률을 포함한 출력 측 업데이트 신호 |
| $x$ | 학습 후 새롭게 예측하려는 **테스트 입력** |

### 3.1 출력 Gradient와 Weight Gradient의 차이

Gradient Descent는 Weight에 대한 미분을 이용한다.

$$
W_t=W_{t-1}-\eta_t\nabla_W\mathcal L_t.
$$

그런데 우리가 직접 계산하기 쉬운 것은 출력에 대한 Gradient다.

$$
g_t=\frac{\partial\mathcal L_t}{\partial\hat y_t}.
$$

**왜 $g_t$에 입력 $x_t$를 곱할까?** Chain Rule 때문이다.

Linear Layer의 $k$번째 출력은

$$
\hat y_{t,k}=\sum_j W_{kj}x_{t,j}.
$$

따라서 Weight 원소 $W_{ij}$에 대해 미분하면

$$
\frac{\partial\hat y_{t,k}}{\partial W_{ij}}
=\delta_{ki}x_{t,j}
$$

이다. 여기서 $\delta_{ki}$는 $k=i$일 때 1, 아니면 0인 Kronecker Delta다.

Chain Rule을 적용하면

$$
\begin{aligned}
\frac{\partial\mathcal L_t}{\partial W_{ij}}
&=\sum_k
\frac{\partial\mathcal L_t}{\partial\hat y_{t,k}}
\frac{\partial\hat y_{t,k}}{\partial W_{ij}}\\
&=\sum_k g_{t,k}\delta_{ki}x_{t,j}\\
&=g_{t,i}x_{t,j}.
\end{aligned}
$$

이를 행렬로 묶으면

$$
\boxed{\nabla_W\mathcal L_t=g_tx_t^\top}.
$$

이는 **출력 Gradient 벡터와 입력 벡터의 Outer Product(외적)**이다.

$$
\underbrace{g_t}_{d_{\mathrm{out}}\times1}
\underbrace{x_t^\top}_{1\times d_{\mathrm{in}}}
=
\underbrace{\nabla_W\mathcal L_t}_{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
$$

### 3.2 Error Signal $e_t$ 정의

앞의 결과를 Gradient Descent에 대입한다.

$$
W_t=W_{t-1}-\eta_tg_tx_t^\top.
$$

표기를 간단히 하기 위해

$$
\boxed{e_t\triangleq-\eta_tg_t}
$$

라고 **정의**한다. $\triangleq$는 새로운 수학적 정의를 나타내는 기호다. 따라서

$$
\boxed{W_t=W_{t-1}+e_tx_t^\top}.
$$

가 된다. $e_t$는 별도로 학습하는 변수가 아니라 기존 출력 Gradient에 음의 학습률을 곱한 값이다.

### 3.3 여러 Gradient Step 전개

첫 번째 업데이트:

$$
W_1=W_0+e_1x_1^\top.
$$

두 번째 업데이트:

$$
W_2=W_0+e_1x_1^\top+e_2x_2^\top.
$$

총 $T$회 업데이트를 전개하면,

$$
\boxed{
W_T=W_0+\sum_{t=1}^{T}e_tx_t^\top
}
$$

를 얻는다.

## 4. 학습된 모델의 예측을 Dual Form으로 바꾸기

학습이 끝난 뒤 새로운 테스트 입력 $x$를 예측한다고 하자.

여기서 **$x_t$는 과거 학습 단계의 입력**, **$x$는 현재 테스트 입력**이다.

학습된 Linear Layer의 출력은

$$
y=W_Tx.
$$

여기에 앞서 구한 $W_T$를 대입한다.

$$
\begin{aligned}
y
&=\left(W_0+\sum_{t=1}^{T}e_tx_t^\top\right)x\\
&=W_0x+\sum_{t=1}^{T}e_t(x_t^\top x)\\
&=\boxed{W_0x+\sum_{t=1}^{T}(x_t^\top x)e_t}.
\end{aligned}
$$

이 결과가 중요하다. 학습된 Weight를 한 번 곱하는 계산이 **초기 Weight의 출력과, 과거 학습 입력들에 대한 내적 기반 합산**으로 바뀌었다.

### 4.1 Attention의 Q/K/V와 대응시키기

Softmax가 없는 **Unnormalised Dot Attention**을 다음과 같이 정의하자.

$$
\operatorname{Attention}(q,K,V)
=\sum_{t=1}^{T}(k_t^\top q)v_t.
$$

위 Dual Form에서 대응되는 항목은 다음과 같다.

| Attention 요소 | Gradient Descent Dual Form | 역할 |
|---|---|---|
| Query $q$ | 테스트 입력 $x$ | 현재 예측할 대상 |
| Key $k_t$ | 학습 입력 $x_t$ | 과거 학습 입력 |
| Value $v_t$ | 업데이트 신호 $e_t$ | 해당 학습 단계의 기여 |
| Score $k_t^\top q$ | $x_t^\top x$ | 과거와 현재 입력의 내적 |

따라서

$$
\boxed{
W_Tx=W_0x+\operatorname{Attention}(x,K,V)
}
$$

가 된다. 여기서 $K=[x_1,\ldots,x_T]$와 $V=[e_1,\ldots,e_T]$이며, 열벡터를 나란히 쌓았다.

이것이 Gradient Descent로 학습된 Linear Layer의 **Primal Form과 Attention 기반 Dual Form의 동치**다.

### 4.2 수치 예제

$$
W_0=[0\quad0],\quad
x_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
e_1=2
$$

$$
x_2=\begin{bmatrix}0\\1\end{bmatrix},\quad
e_2=3,\quad
x=\begin{bmatrix}4\\5\end{bmatrix}.
$$

학습된 Weight는

$$
W_2=W_0+e_1x_1^\top+e_2x_2^\top=[2\quad3].
$$

Primal Form:

$$
y=W_2x=2\cdot4+3\cdot5=23.
$$

Dual Form:

$$
y=W_0x+(x_1^\top x)e_1+(x_2^\top x)e_2
=0+4\cdot2+5\cdot3=23.
$$

두 방식은 동일하다. 이 예제에서 $e_1,e_2$는 식의 구조를 보여주기 위한 값으로, 특정 Loss의 실제 Gradient에서 계산된 값은 아니다.

## 5. Transformer ICL을 Meta-Optimizer로 해석하기

다음은 Dai et al.의 *Why Can GPT Learn In-Context?*에서 논의하는 관점이다.

Gradient Descent의 Dual Form에서 Attention 구조를 확인했다면, 반대로 **Transformer의 Attention을 유효 Weight 업데이트처럼 해석할 수 있는지** 검토할 수 있다.

### 5.1 Demonstration과 현재 Query를 나누기

현재 Attention 입력의 Token을 두 집합으로 나누자.

- $X'$: Demonstration Token 표현의 행렬
- $X$: 현재 Query 측 Token 표현의 행렬
- $q$: 현재 위치의 Query 벡터

설명을 위해 각 Token을 열벡터로 배치한다. Transformer의 Projection Matrix를 $W_K,W_V$라고 하면 일반적인 Softmax Attention은

$$
F_{\mathrm{ICL}}(q)=
W_V[X';X]\,
\operatorname{softmax}\left(
\frac{(W_K[X';X])^\top q}{\sqrt{d_k}}
\right).
$$

### 5.2 Softmax Attention을 Dot Attention으로 단순화

여기서부터는 **근사적·이상화된 분석**이다.

$$
\widetilde F_{\mathrm{ICL}}(q)
=
W_V[X';X](W_K[X';X])^\top q.
$$

Softmax와 Scaling을 제거하면 원래 Attention과 일반적으로 같은 수치가 나오지 않는다. 따라서 다음은 변경된 **Unnormalised Dot Attention 모델 내부에서의 정확한 전개**다.

행렬을 나누면,

$$
\begin{aligned}
\widetilde F_{\mathrm{ICL}}(q)
&=W_VX(W_KX)^\top q\\
&\quad+W_VX'(W_KX')^\top q.
\end{aligned}
$$

첫 번째 항은 Demonstration 없이도 존재하는 **ZSL(Zero-Shot Learning)에 대응하는 항**, 두 번째 항은 Demonstration에서 추가되는 부분으로 해석한다.

$$
W_{\mathrm{ZSL}}\triangleq W_VX(W_KX)^\top,
$$

$$
\Delta W_{\mathrm{ICL}}\triangleq W_VX'(W_KX')^\top.
$$

따라서

$$
\boxed{
\widetilde F_{\mathrm{ICL}}(q)=
(W_{\mathrm{ZSL}}+\Delta W_{\mathrm{ICL}})q.
}
$$

### 5.3 왜 Zero-Shot Learning을 도입할까?

기준 상태를 설정하기 위해서다.

- **ZSL:** Demonstration 없이 Query로 예측
- **ICL:** Demonstration을 함께 제공하여 예측

위에서 단순화한 Attention에 대해서는

$$
\boxed{
\widetilde F_{\mathrm{ICL}}(q)-
\widetilde F_{\mathrm{ZSL}}(q)
=
\Delta W_{\mathrm{ICL}}q
}
$$

가 성립한다. 이는 **Demonstration에 의한 추가 효과를 유효 Weight의 변화로 표현할 수 있다**는 뜻이다.

그러나 이는 실제 모델 파라미터 $W_Q,W_K,W_V$가 추론 중 업데이트된다는 뜻이 아니다. 더구나 실제 Softmax Attention은 Token 추가에 따라 정규화 분모와 가중치가 바뀌므로 위와 같이 단순 분리할 수 없다. 여러 Transformer Layer에서도 Demonstration 유무에 따라 뒤쪽 Hidden State 자체가 달라질 수 있으므로, 이 비교는 단순화된 단일 Attention 연산의 해석으로 이해해야 한다.

### 5.4 Gradient Descent와의 구조적 대응

| Gradient Descent | ICL 근사식 |
|---|---|
| 초기 Weight $W_0$ | 유효 Weight $W_{\mathrm{ZSL}}$ |
| 업데이트 $\sum_t e_tx_t^\top$ | $\Delta W_{\mathrm{ICL}}=W_VX'(W_KX')^\top$ |
| 업데이트 후 예측 $W_Tx$ | $(W_{\mathrm{ZSL}}+\Delta W_{\mathrm{ICL}})q$ |
| 실제 Loss 기반 Error Signal $e_t$ | Demonstration Value 표현 $W_Vx'_t$ |

**가장 중요한 구분:** $W_Vx'_t$가 실제 Backpropagation Gradient라는 뜻은 아니다. 두 연산이 **외적의 합**이라는 같은 형태를 가지므로 Meta-Gradient/Meta-Optimizer라는 관점으로 해석한다.

## 6. 지금까지의 핵심 정리

1. Linear Layer의 Weight Gradient는 Chain Rule에 따라 $g_tx_t^\top$라는 Outer Product로 계산된다.
2. 여러 Gradient Step을 전개하면 학습된 Weight를 초기 Weight와 단계별 업데이트의 합으로 표현할 수 있다.
3. 테스트 입력에 대한 예측을 전개하면 $\sum_t(x_t^\top x)e_t$라는 Dot-Product Attention 형태가 나타난다.
4. Transformer ICL도 Softmax를 제거한 단순화된 Attention에서는 Demonstration 기여를 유효 Weight 변화로 묶을 수 있다.
5. **Dual Form의 정확한 동치**와 **일반적인 GPT의 ICL을 Meta-Optimization으로 해석하는 근사·가설**은 구분해야 한다.

## 7. 다음 스터디: Demonstration과 Inductive Bias

이번 글에서는 Attention의 수학적 구조를 공부했다. 하지만 실제 ICL에서 Demonstration이 항상 정확한 입출력 규칙을 가르치는 것은 아닐 수 있다. 후속 연구에서는 다음을 실험적으로 검토한다.

- Demonstration의 정답 Label이 틀리면 성능이 얼마나 달라지는가?
- 입력 분포, Label Space, Input–Label Format의 역할은 무엇인가?
- Context에 여러 해석 가능한 규칙이 있다면 모델은 어느 Feature를 선호하는가?

이는 로봇 시연을 활용하는 In-Context Imitation Learning을 이해하는 데도 중요한 질문이다.

---

## 참고 문헌 및 강의

1. Irie, K., Csordás, R., & Schmidhuber, J. (2022). **The Dual Form of Neural Networks Revisited: Connecting Test Time Predictions to Training Patterns via Spotlights of Attention.** *ICML 2022*. https://proceedings.mlr.press/v162/irie22a.html
2. Dai et al. (2023). **Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers.** *Findings of ACL 2023*. https://aclanthology.org/2023.findings-acl.247/
3. Min et al. (2022). **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?** *EMNLP 2022*. https://aclanthology.org/2022.emnlp-main.759/
4. Si et al. (2023). **Measuring Inductive Biases of In-Context Learning with Underspecified Demonstrations.** *ACL 2023*. https://aclanthology.org/2023.acl-long.632/
5. **ICL 세미나 영상:** https://www.youtube.com/watch?v=9liSz8PDAp8

> 본문은 강의와 논문을 참고하여 작성한 개인 학습 정리다. Chain Rule의 세부 전개, 간단한 숫자 예제와 기호 설명은 이해를 돕기 위해 별도로 추가했다.
