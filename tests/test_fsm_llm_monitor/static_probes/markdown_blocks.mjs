import { renderMarkdown as r } from '../static/utils/markdown.js';
const balanced = html => (html.match(/<p>/g) || []).length === (html.match(/<\/p>/g) || []).length;
const cases = {
    list_then_text: '- a\n- b\n\nAfter the list',
    ordered_then_text: '1. x\n2. y\nrest',
    code_then_text: '```\ncode\n```\nafter',
    rule_then_text: '---\nfoo',
    plain_paragraphs: 'Para one\n\nPara two',
    list_only: '- a',
    list_gap_text: '- a\n\n\nAfter',
    block_run: '```\nx\n```\n- a\n\n1. b\n\n---\n```\ny\n```',
};
const html = Object.fromEntries(Object.entries(cases).map(([k, v]) => [k, r(v)]));
console.log(JSON.stringify({ html, balanced: Object.fromEntries(Object.entries(html).map(([k, v]) => [k, balanced(v)])) }));
