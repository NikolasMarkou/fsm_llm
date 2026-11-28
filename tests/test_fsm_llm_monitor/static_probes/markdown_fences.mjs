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
    // Plan c9d908d4 step 2 (C1): a star opens emphasis only before a non-space and closes after one, and a rule
    // between two lists is a rule (the rule pass runs before the list passes, which eat the newline after an item run).
    star_item: r('* item with *emph*\n* two'),
    star_spaced: r('2 * 3 * 4'),
    star_inside: r('a*b*c'),
    star_pair: r('*a* *b*'),
    star_flank: [r('x * a* y'), r('x *a * b')],
    star_bold: r('* **b** x'),
    rule_dash_list: r('- a\n---\n- b'),
    rule_star_list: r('* a\n***\n* b'),
    rule_ordered: r('1. a\n---\n2. b'),
    star_hostile: ['* <script>alert(1)</script> *x*', '*<img src=x onerror=alert(1)>*', '- *a*<b>\n---\n- *"c"*'].map(t => r(t)),
};
// A seeded fuzz (an LCG, no clock): every output keeps its em, strong, ul, ol and li tags paired and holds no tag of the input.
let seed = 12345;
const rnd = n => Math.floor(((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x80000000) * n);
const alpha = ['*', '-', ' ', 'a', '\n', '_', '1. ', '`', '#', '**', '---', '***', '<b>', '&'];
const open = (h, t) => (h.match(new RegExp(`<${t}[ >]`, 'g')) || []).length;
const shut = (h, t) => (h.match(new RegExp(`</${t}>`, 'g')) || []).length;
let unpaired = 0, rawTag = 0;
for (let i = 0; i < 2000; i++) {
    let s = '';
    for (let n = 1 + rnd(9); n > 0; n--) s += alpha[rnd(alpha.length)];
    const h = r(s);
    if (['em', 'strong', 'ul', 'ol', 'li'].some(t => open(h, t) !== shut(h, t))) unpaired++;
    if (h.includes('<b>')) rawTag++;
}
out.fuzz = { unpaired, rawTag };
console.log(JSON.stringify(out));
