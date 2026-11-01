# 04. Instant Policy와 Graph 기반 ICIL

> 상태: 학습 예정 — ICL 개념을 로봇 정책에 연결

## 목표

언어 Token의 시연 기반 조건부 예측과, 로봇 관측/행동 Graph를 사용하는 In-Context Imitation Learning을 비교합니다.

## 비교할 항목

| 항목 | 언어 모델 ICL | Instant Policy |
|---|---|---|
| Context | 입력-출력 예시 Token | 과거 로봇 시연 |
| Query | 새로운 텍스트 입력 | 현재 로봇 관측 |
| 정보 표현 | Token Sequence | Graph Node / Edge |
| 예측 목표 | 출력 Token/Label | 미래 로봇 Action |
| 적응 | 추론 중 Context 조건화 | 추론 중 시연 조건화 |

## 확인할 질문

1. 서로 다른 시연은 미래 Action 분포에 어떤 영향을 주는가?
2. Graph Transformer는 시연과 현재 관측의 어떤 관계를 처리하는가?
3. Diffusion 과정과 Context 조건화는 어떻게 연결되는가?
4. Demonstration의 Feature Bias가 로봇 환경에서도 나타날 수 있는가?

## 참고

- *Instant Policy: In-Context Imitation Learning via Graph Diffusion*
- https://github.com/vv19/instant_policy

실험 결과나 구현 세부사항은 해당 논문과 코드를 재검토한 후 추가합니다.
