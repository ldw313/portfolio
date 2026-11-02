# Tistory 게시 및 수식 표시 안내

## 저장된 원고

- [01. Transformer ICL — Gradient Descent와 Attention의 Dual Form](01_icl_attention_gradient_descent.md)

## GitHub Markdown의 수식

GitHub는 인라인 수식 \`$...$\`와 블록 수식 \`$$...$$\`를 지원한다. 원고는 이 구문으로 저장한다.

## Tistory에서 LaTeX 수식을 표시하는 방법

Tistory의 일반 편집/스킨 환경에서는 LaTeX 구문을 자동으로 해석하지 않을 수 있으므로 MathJax를 설정한다.

1. Tistory **관리 → 꾸미기 → 스킨 편집 → HTML 편집** 이동.
2. 사용 중인 스킨을 백업한다.
3. \`<head>\` 내부에서 \`</head>\` 바로 앞에 아래 코드를 추가한다.
4. 적용 후 테스트 글에서 \`$x^2$\` 및 \`$$W_Tx=W_0x+\sum_t e_t(x_t^\top x)$$\`가 수식으로 표시되는지 확인한다.
5. 모바일 페이지, 스킨 변경 후에도 테스트한다.

\`\`\`html
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    },
    options: {
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
\`\`\`

**주의:** GitHub Markdown 원고를 Tistory 기본 에디터에 그대로 붙여 넣는다고 Markdown 문법 전체가 자동 변환되는 것은 아니다. 마크다운을 HTML로 변환하거나, Tistory에서 제공하는 편집 기능에 맞춰 붙여 넣어야 한다. 게시 전 수식, 코드 블록, 표를 미리보기/실제 페이지에서 검증한다. 이 스크립트는 **GitHub 문서 본문이 아니라 Tistory 스킨 HTML에 추가**하는 코드다.

## 이력

- 2026-11-02: 원고의 잘못 이스케이프된 코드 펜스(\`\\\`\`\`\` 계열)를 정상 코드 펜스로 수정; 수식 구분자 \`$$\`의 개수(74개, 37쌍) 및 코드 펜스 수(2개, 1쌍)를 점검함. 브라우저별 실제 표시까지 검증한 것은 아님.
