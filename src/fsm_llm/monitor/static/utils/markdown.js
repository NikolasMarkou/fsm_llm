// FSM-LLM Monitor — Lightweight Markdown Renderer
// Converts a safe subset of Markdown to HTML. XSS-safe (escapes first).

import { esc } from './dom.js';

export function renderMarkdown(text) {
    if (!text) return '';

    // Escape first (XSS). Raw NULs are dropped so a placeholder below can only be ours.
    let s = esc(text).replace(/\u0000/g, '');

    // Code is stashed behind NUL placeholders (already escaped) so the bold, italic,
    // list and line-break passes below never touch it; restored before the clean-up.
    const stash = [];
    const hold = html => `\u0000${stash.push(html) - 1}\u0000`;

    // Fenced code blocks
    s = s.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, _lang, code) =>
        hold(`<pre class="md-code-block"><code>${code.trim()}</code></pre>`));

    // Inline code
    s = s.replace(/`([^`\n]+)`/g, (_m, code) => hold(`<code class="md-code-inline">${code}</code>`));

    // Headers (only at line start)
    s = s.replace(/^### (.+)$/gm, '<strong class="md-h3">$1</strong>');
    s = s.replace(/^## (.+)$/gm, '<strong class="md-h2">$1</strong>');
    s = s.replace(/^# (.+)$/gm, '<strong class="md-h1">$1</strong>');

    // Bold
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Italic
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    s = s.replace(/(^|[^a-zA-Z0-9])_([^_\n]+)_([^a-zA-Z0-9]|$)/g, '$1<em>$2</em>$3');

    // Unordered lists
    s = s.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
    s = s.replace(/((?:<li>.*<\/li>\n?)+)/g, m => `<ul>${m.replace(/\n/g, '')}</ul>`);

    // Ordered lists (a distinct item marker keeps them out of the <ul> pass above)
    s = s.replace(/^\d+\. (.+)$/gm, '<oli>$1</oli>');
    s = s.replace(/((?:<oli>.*<\/oli>\n?)+)/g, m => `<ol>${m.replace(/\n/g, '').replace(/<(\/?)oli>/g, '<$1li>')}</ol>`);

    // Horizontal rule
    s = s.replace(/^(?:---|\*\*\*)$/gm, '<hr class="md-hr">');

    // Line breaks
    s = s.replace(/\n\n+/g, '</p><p>');
    s = s.replace(/\n/g, '<br>');

    // Wrap in paragraph
    s = `<p>${s}</p>`;

    // Restore the stashed code
    s = s.replace(/\u0000(\d+)\u0000/g, (_m, i) => stash[Number(i)]);

    // Clean up
    s = s.replace(/<p>\s*<\/p>/g, '');
    s = s.replace(/<p>((?:<pre|<ul|<ol|<hr))/g, '$1');
    s = s.replace(/((?:<\/pre>|<\/ul>|<\/ol>|<hr[^>]*>))<\/p>/g, '$1');
    s = s.replace(/<br>\s*(<pre|<ul|<ol|<\/p>)/g, '$1');
    s = s.replace(/(<\/pre>|<\/ul>|<\/ol>)\s*<br>/g, '$1');

    return s;
}
