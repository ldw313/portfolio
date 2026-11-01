# 01. Attention과 Gradient Descent의 관계

> 상태: 1차 수학적 유도 완료 · 강의 04:27–14:05 중 Dual Form 기초  
> 출처: Irie, Csordás, Schmidhuber, **The Dual Form of Neural Networks Revisited: Connecting Test Time Predictions to Training Patterns via Spotlights of Attention**, ICML 2022, Section 2–3.  
> 원문: https://proceedings.mlr.press/v162/irie22a.html · PDF: https://proceedings.mlr.press/v162/irie22a/irie22a.pdf

## 학습 목표

Gradient Descent(GD)로 학습된 Linear Layer를 *Unnormalised Dot Attention*과 초기 Weight로 구성된 Key–Value Memory 형태로 **정확히 다시 표현**한다. 이를 통해 논문의 Eq. (11)과 Eq. (13)를 도출한다. 이 결과를 일반 Transformer Softmax Attention과 혼동하지 않는다.

## 1. 기호와 가정

Bias를 생략한 Linear Layer:

$$
\hat y=Wx,\quad
W\in\mathbb R^{d_{\rm out}\times d_{\rm in}},\
x\in\mathbb R^{d_{\rm in}},\
\hat y\in\mathbb R^{d_{\rm out}}.
$$

- $W_0$: 초기 Weight Matrix.
- $t=1,\ldots,T$: 순차 학습 업데이트 인덱스.
- $x_t$: 해당 학습 단계에서 Linear Layer에 실제로 입력된 벡터.
- $y_t^*$: 목표 출력(학습 설명을 위한 표기).
- $\hat y_t=W_{t-1}x_t$: 업데이트 전 모델 출력.
- $\mathcal L_t=\mathcal L(\hat y_t,y_t^*)$: 미분 가능한 Loss.
- $\eta_t$: 단계별 Learning Rate.
- $g_t=\nabla_{\hat y_t}\mathcal L_t\in\mathbb R^{d_{\rm out}}$: 해당 층 출력에 대한 Gradient.
- $e_t=-\eta_tg_t$: 부호와 Learning Rate를 포함한 Error Signal.

**인덱스 참고:** 여기서는 업데이트 전 Weight를 $W_{t-1}$로 정의한다. 원문 각주에서는 해당 단계의 Weight를 $W_t$로 표기하지만, 실제 의미는 같다.

## 2. [수학적 유도] Linear Layer의 Gradient가 Outer Product인 이유

출력의 $i$번째 성분:

$$
\hat y_{t,i}=\sum_{j=1}^{d_{\rm in}}W_{ij}x_{t,j}.
$$

Chain Rule:

$$
\frac{\partial\mathcal L_t}{\partial W_{ij}}
=\sum_k
\frac{\partial\mathcal L_t}{\partial\hat y_{t,k}}
\frac{\partial\hat y_{t,k}}{\partial W_{ij}}.
$$

$\partial\hat y_{t,k}/\partial W_{ij}=\mathbf 1[k=i]x_{t,j}$ 이므로,

$$
\frac{\partial\mathcal L_t}{\partial W_{ij}}
=g_{t,i}x_{t,j}.
$$

원소를 행렬로 모으면:

$$
\boxed{\nabla_W\mathcal L_t=g_tx_t^\top}.
$$

차원 검증:

$$
(d_{\rm out}\times1)(1\times d_{\rm in})
=d_{\rm out}\times d_{\rm in}.
$$

이것이 외적(Outer Product)이다. 이 유도는 논문의 핵심 결과를 풀어서 설명하기 위해 추가한 단계다.

## 3. [수학적 유도] Gradient Descent를 학습 이력의 합으로 표현

GD 업데이트:

$$
W_t=W_{t-1}-\eta_t\nabla_W\mathcal L_t
=W_{t-1}-\eta_tg_tx_t^\top
=W_{t-1}+e_tx_t^\top.
$$

반복 전개하면,

$$
W_1=W_0+e_1x_1^\top,
$$

$$
W_2=W_0+e_1x_1^\top+e_2x_2^\top,
$$

따라서

$$
\boxed{W_T=W_0+\sum_{t=1}^{T}e_tx_t^\top.}
\tag{11}
$$

**[논문]** Section 3, Corollary 1의 Eq. (11)과 같은 표현. $e_t$는 학습 중 실제 Backpropagation으로 생성된 값이므로 일반적으로 이전 학습 단계들에 의존한다. 임의의 고정된 레이블이라고 가정하면 안 된다.

## 4. [수학적 유도] 새로운 테스트 입력에 대한 예측

새 입력 $x$를 학습된 층에 넣으면,

$$
\begin{aligned}
y
&=W_Tx\\
&=\left(W_0+\sum_{t=1}^{T}e_tx_t^\top\right)x\\
&=W_0x+\sum_{t=1}^{T}e_t(x_t^\top x).
\end{aligned}
$$

즉,

$$
\boxed{y=W_0x+\sum_{t=1}^{T}(x_t^\top x)e_t.}
$$

$\alpha_t=x_t^\top x$가 스칼라이므로 $e_t(x_t^\top x)=(x_t^\top x)e_t$이다.

**해석:** (1) 초기 Weight가 만든 출력 $W_0x$, (2) 각 학습 입력 $x_t$와 테스트 입력 $x$의 내적에 의해 가중된 업데이트 신호 $e_t$를 더한 결과.

## 5. [논문] Unnormalised Dot Attention의 정의 (Eq. 1–5)

Key와 Value를 열벡터로 저장한다:

$$
K=[k_1,\ldots,k_T]\in\mathbb R^{d_{\rm in}\times T},\quad
V=[v_1,\ldots,v_T]\in\mathbb R^{d_{\rm out}\times T}.
$$

Query $q\in\mathbb R^{d_{\rm in}}$에 대해 논문 Definition 2.1은 다음을 정의한다:

$$
\operatorname{Attention}(K,V,q)
=\sum_{t=1}^{T}(k_t^\top q)v_t
=VK^\top q.
\tag{1–2}
$$

$\sum_tv_tk_t^\top=VK^\top$이기 때문이다. 이 Attention은 **Softmax 정규화가 없고**, 내적 점수가 음수일 수도 있다.

논문 Proposition 1(Eq. 7–10)은 $W=\sum_tv_tk_t^\top$인 Linear Layer와 이 Dot Attention 시스템이 모든 $q$에 대해 같은 출력을 생성한다고 증명한다.

## 6. [논문] GD의 Dual Form (Corollary 1, Eq. 13)

다음과 같이 대응시킨다:

| Attention | GD Linear Layer |
|---|---|
| Query $q$ | 테스트 입력 $x$ |
| Key $k_t$ | 학습 입력 $x_t$ |
| Value $v_t$ | Backpropagation Error Signal $e_t=-\eta_tg_t$ |
| Weight $k_t^\top q$ | 내적 $x_t^\top x$ |

$$
X=[x_1,\ldots,x_T],\qquad E=[e_1,\ldots,e_T].
$$

그러면:

$$
\boxed{
W_Tx
=W_0x+EX^\top x
=W_0x+\operatorname{Attention}(X,E,x).
}
\tag{13}
$$

이것은 **Primal Form과 Dual Form의 수학적 동치**이다. 논문 Figure 1은 $W_T$ 하나로 추론하는 Primal 구조, Figure 2는 초기 Weight 및 학습 입력·오차 신호를 저장하는 Key–Value Memory 구조를 보여준다.

## 7. [추가 설명] 수치 예제

입력 2차원, 출력 1차원이며 다음과 같이 가정하자.

$$
W_0=[0\ \ 0],\quad x_1=[1\ \ 0]^\top,\quad e_1=2,
\quad x_2=[0\ \ 1]^\top,\quad e_2=3.
$$

두 번 업데이트 후:

$$
W_2=W_0+e_1x_1^\top+e_2x_2^\top=[2\ \ 3].
$$

테스트 입력 $x=[4\ \ 5]^\top$이면 Primal Form은

$$
y=W_2x=2\cdot4+3\cdot5=23.
$$

Dual Form도

$$
y=W_0x+(x_1^\top x)e_1+(x_2^\top x)e_2
=0+4\cdot2+5\cdot3=23.
$$

동일한 결과이다. **참고:** 이 예시의 $e_1,e_2$는 항등식을 보여주기 위해 설정한 값이며, 특정 Loss에서 실제 구한 Gradient가 아니다.

## 8. 적용 조건과 주의점

1. **Attention 종류:** 논문의 정확한 동치는 Unnormalised Dot Attention 기준이다. Softmax Attention을 그대로 대입하면 일반적으로 성립하지 않는다.
2. **학습 과정:** SGD 스타일로 한 학습 입력에 대한 Gradient를 외적으로 표현한 것이다. Mini-batch에서는 개별 샘플 기여를 전개할 수 있으며, Momentum/Adam/Weight Decay와 같은 다른 업데이트는 별도 분석이 필요하다.
3. **모델 깊이:** 각 Linear Layer의 입력 $x_t$는 학습 도중 그 층에 들어온 활성화이며, 반드시 원본 입력 데이터 자체는 아니다.
4. **추론 파라미터:** 이미 학습된 Weight로 계산한 출력을 **재표현**한 것이지, 추론할 때 실제 파라미터를 업데이트한다는 뜻이 아니다.
5. **복잡도:** Primal Form은 고정된 Weight Matrix를 사용하지만 Dual Form은 학습 단계 수 $T$에 비례하는 메모리 및 Attention 계산이 필요하다.
6. **ICL과의 관계:** 이 논문만으로 일반적인 GPT의 ICL이 Gradient Descent를 수행한다고 단정할 수 없다. 다음 학습에서 *Why Can GPT Learn In-Context?*의 추가 가정과 근사를 별도로 검토한다.

## 9. 다음 학습 질문

- $W_Tx=W_0x+\operatorname{Attention}(X,E,x)$를 Prompt의 Demonstration과 Query에 연결하려면 무엇이 더 필요한가?
- Transformer의 $W_Q,W_K,W_V$와 위에서 정의한 $X,E,x$의 관계는 무엇이고, 어디까지 같은가?
- Softmax를 생략하거나 근사할 때 무엇을 잃는가?

## 참고 문헌

- Irie, K., Csordás, R., & Schmidhuber, J. (2022). [The Dual Form of Neural Networks Revisited](https://proceedings.mlr.press/v162/irie22a.html). *ICML*, Section 2–3, Eq. (1–13), Figure 1–2.
- [강의 05:54부터](https://www.youtube.com/watch?v=9liSz8PDAp8&t=354s).
- [Why Can GPT Learn In-Context?](https://arxiv.org/abs/2212.10559) — 후속 학습 자료.
