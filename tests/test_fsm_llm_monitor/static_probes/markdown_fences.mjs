import { renderMarkdown as r } from '../static/utils/markdown.js';
// Plan 9283c1fd step 7 (B6a, B6b): a fence with any info string is a code block (the body is never
// run through the inline passes and the info string is never emitted), and runs of `_italic_` all render.
// markdown.mjs and markdown_blocks.mjs stay byte for byte; the new rows live here.
const body = '*a* **b** _c_ __d__';
const infos = ['c++', 'c#', 'ts-node', 'objective-c', 'js  ', 'js', '', '<script>alert(1)</script>', '<img src=x onerror=alert(1)>'];
const fence = Object.fromEntries(infos.map(i => [i, r('```' + i + '\n' + body + '\n```')]));
const out = {
    fence,
    fence_crlf_info: r('```c++ \r\nint x;\r\n```'),
    fence_after_text: r('before\n```c++\nint *p;\n```\nafter'),
    backtick_in_info: r('```a`b\ncode\n```'),   // a backtick ends the info string: no block
    nested_unchanged: r('`a ```\nx<b>\n``` b`'),
    italic_pair: r('_a_ _b_'),
    italic_three: r('_a_ _b_ _c_'),
    italic_lines: r('_a_\n_b_'),
    italic_punct: r('(_a_), _b_.'),
    identifiers: ['snake_case_here', 'foo_bar_baz', 'http://x/a_b_c', 'a_b_ c'].map(t => r(t)),
};
console.log(JSON.stringify(out));
