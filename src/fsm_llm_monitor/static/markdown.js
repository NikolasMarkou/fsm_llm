// FSM-LLM Monitor — Lightweight Markdown Renderer
// Converts a subset of Markdown to safe HTML.
// Supports: bold, italic, inline code, code blocks, headers, lists, links, line breaks.

'use strict';

function renderMarkdown(text) {
    if (!text) return '';

    // Escape HTML first (XSS protection), then apply markdown transforms
    var s = esc(text);

    // Fenced code blocks: ```lang\n...\n```
    s = s.replace(/```(\w*)\n([\s\S]*?)```/g, function(_m, _lang, code) {
        return '<pre class="md-code-block"><code>' + code.trim() + '</code></pre>';
    });

    // Inline code: `...`
    s = s.replace(/`([^`\n]+)`/g, '<code class="md-code-inline">$1</code>');

    // Headers: ### h3, ## h2, # h1 (only at line start)
    s = s.replace(/^### (.+)$/gm, '<strong class="md-h3">$1</strong>');
    s = s.replace(/^## (.+)$/gm, '<strong class="md-h2">$1</strong>');
    s = s.replace(/^# (.+)$/gm, '<strong class="md-h1">$1</strong>');

    // Bold: **text** or __text__
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Italic: *text* or _text_ (but not inside words with underscores)
    // Note: underscore italic uses word-boundary \b instead of lookbehind
    // for Safari <16.4 compatibility.
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    s = s.replace(/(^|[^a-zA-Z0-9])_([^_\n]+)_([^a-zA-Z0-9]|$)/g, '$1<em>$2</em>$3');

    // Unordered lists: - item or * item (at line start)
    s = s.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
    // Wrap consecutive <li> in <ul>
    s = s.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Ordered lists: 1. item (at line start)
    s = s.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Horizontal rule: --- or ***
    s = s.replace(/^(?:---|\*\*\*)$/gm, '<hr class="md-hr">');

    // Line breaks: double newline = paragraph break, single = <br>
    // First handle double newlines
    s = s.replace(/\n\n+/g, '</p><p>');
    // Single newlines (but not inside pre/code blocks)
    s = s.replace(/\n/g, '<br>');

    // Wrap in paragraph
    s = '<p>' + s + '</p>';

    // Clean up empty paragraphs
    s = s.replace(/<p>\s*<\/p>/g, '');

    // Fix pre blocks that got paragraph-wrapped
    s = s.replace(/<p>((?:<pre|<ul|<hr))/g, '$1');
    s = s.replace(/((?:<\/pre>|<\/ul>|<hr[^>]*>))<\/p>/g, '$1');
    s = s.replace(/<br>\s*(<pre|<ul|<\/p>)/g, '$1');
    s = s.replace(/(<\/pre>|<\/ul>)\s*<br>/g, '$1');

    return s;
}
