// FSM-LLM Monitor — Lightweight Markdown Renderer
// Converts a safe subset of Markdown to HTML. XSS-safe (escapes first).

import { esc } from './dom.js';

export function renderMarkdown(text) {
    if (!text) return '';

    let s = esc(text);

    // Fenced code blocks
    s = s.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, _lang, code) =>
        `<pre class="md-code-block"><code>${code.trim()}</code></pre>`);

    // Inline code
    s = s.replace(/`([^`\n]+)`/g, '<code class="md-code-inline">$1</code>');

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
    s = s.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Ordered lists
    s = s.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Horizontal rule
    s = s.replace(/^(?:---|\*\*\*)$/gm, '<hr class="md-hr">');

    // Line breaks
    s = s.replace(/\n\n+/g, '</p><p>');
    s = s.replace(/\n/g, '<br>');

    // Wrap in paragraph
    s = `<p>${s}</p>`;

    // Clean up
    s = s.replace(/<p>\s*<\/p>/g, '');
    s = s.replace(/<p>((?:<pre|<ul|<hr))/g, '$1');
    s = s.replace(/((?:<\/pre>|<\/ul>|<hr[^>]*>))<\/p>/g, '$1');
    s = s.replace(/<br>\s*(<pre|<ul|<\/p>)/g, '$1');
    s = s.replace(/(<\/pre>|<\/ul>)\s*<br>/g, '$1');

    return s;
}
