# In-Context Learning (ICL) Study

Transformer 기반 In-Context Learning을 수학적 원리부터 실험 분석, 로봇 모방학습 응용까지 단계적으로 공부하는 문서 모음입니다.

## 학습 순서

| 순서 | 문서 | 상태 |
|---|---|---|
| 00 | [기초 개념 및 표기](00_foundations.md) | 초안 |
| 01 | [Attention과 Gradient Descent의 관계](01_attention_gradient_descent.md) | 학습 예정 |
| 02 | [Demonstration의 역할과 경험적 분석](02_demonstrations.md) | 학습 예정 |
| 03 | [Inductive Bias와 Feature Bias](03_inductive_bias.md) | 학습 예정 |
| 04 | [Instant Policy와 Graph ICIL](04_instant_policy.md) | 학습 예정 |

## 참고 자료

- [ICL 세미나 영상](https://www.youtube.com/watch?v=9liSz8PDAp8&t=663s) — 강의 자막은 자동 인식 오류가 있으므로 원문 논문과 대조
- Irie et al., *The Dual Form of Neural Networks Revisited: Connecting Test Time Predictions to Training Patterns via Spotlights of Attention*, ICML 2022 — https://proceedings.mlr.press/v162/irie22a.html
- *Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers* — https://arxiv.org/abs/2212.10559
- *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*
- Si et al., *Measuring Inductive Biases of In-Context Learning with Underspecified Demonstrations*, ACL 2023 — https://aclanthology.org/2023.acl-long.632/

## 문서 작성 원칙

1. 수식은 기호 정의 → 가정 → 유도 → 해석 순서로 작성합니다.
2. **논문에서 확인된 주장**, **강의에서 소개한 해석**, **스터디 중 추가한 설명**을 분리합니다.
3. Softmax Attention과 Linear Attention의 차이 및 근사 조건을 생략하지 않습니다.
4. 추론 중 모델 파라미터 업데이트와 Context에 따른 Hidden State 변화는 구분합니다.
5. 미완료 항목은 '학습 예정' 또는 '검증 필요'로 표시합니다.

## 수정 방법

- ChatGPT: 이 저장소의 `studies/ICL/` 경로를 지정해 문서 읽기/수정을 요청합니다.
- Codex: 저장소를 로컬에 Clone한 후, `AGENTS.md` 지침을 따라 Markdown을 수정합니다.
- 변경 전 `git pull`, 변경 후 Commit/Push를 수행하고, 동일 파일의 동시 수정을 피합니다.
- Tistory 발행은 별도 단계로 진행하며, 이 Markdown을 원본으로 유지합니다.
