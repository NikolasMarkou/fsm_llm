import { renderMarkdown } from '../static/utils/markdown.js';
const r = renderMarkdown;
const code = r('```py\nx = a*b*c\n__init__\n\nz\n```');
const out = {
    fence_keeps_code: code.includes('x = a*b*c\n__init__\n\nz</code>') && !code.includes('<em>') && !code.includes('<br>'),
    inline_keeps_code: r('use `snake_case_a` and `a*b*c` now'),
    ordered: r('Steps:\n1. first\n2. second'),
    unordered: r('- a\n- b'),
    nul_forged: r('a\u00000\u0000b `c`'),
    xss_fence: r('```\n<img src=x onerror=alert(1)>\n```'),
    xss_inline: r('`<script>alert(1)</script>`'),
};
console.log(JSON.stringify(out));
