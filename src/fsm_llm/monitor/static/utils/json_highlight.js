// FSM-LLM Monitor: JSON syntax colouring (plan 51fde95b D-040).
// The text stays the source of truth: the coloured nodes hold exactly the input, byte for byte, so selecting
// and copying gives plain JSON. Nodes are built with textContent only, never innerHTML, so a string holding
// `<script>` stays text. No imports.

/** Class of an element whose text is JSON (set by `jsonViewHtml` in dom.js, found by `highlightJsonIn`). */
export const JSON_VIEW_CLASS = 'json-view';
/** A longer text is shown plain: the page must not freeze on a huge payload. */
export const JSON_HIGHLIGHT_MAX_CHARS = 200000;

const _WS = ' \t\n\r';
const _PUNCT = _WS + '{}[],:';
const _NUMBER = '+-.0123456789eE';

/**
 * Split valid JSON text into `{type, text}` tokens (type: key, string, number, bool, null, punct), one linear
 * scan with no recursion and no regex, so a deeply nested array costs no stack. Whitespace belongs to the
 * punct run beside it and no character is ever dropped: `tokens.map(t => t.text).join('')` equals `text`.
 * Returns null for a non-string, a text over JSON_HIGHLIGHT_MAX_CHARS, or a text that does not parse.
 */
export function tokenizeJson(text) {
    if (typeof text !== 'string' || text.length > JSON_HIGHLIGHT_MAX_CHARS) return null;
    try { JSON.parse(text); } catch { return null; }
    const tokens = [];
    const n = text.length;
    let i = 0;
    const take = (type, end) => { tokens.push({ type, text: text.slice(i, end) }); i = end; };
    while (i < n) {
        const c = text[i];
        let j = i + 1;
        if (c === '"') {
            while (j < n && text[j] !== '"') j += text[j] === '\\' ? 2 : 1;
            j = Math.min(j + 1, n);
            let k = j;
            while (k < n && _WS.includes(text[k])) k++;
            take(text[k] === ':' ? 'key' : 'string', j);
        } else if (c === '-' || (c >= '0' && c <= '9')) {
            while (j < n && _NUMBER.includes(text[j])) j++;
            take('number', j);
        } else if (c === 't' || c === 'f' || c === 'n') {
            while (j < n && text[j] >= 'a' && text[j] <= 'z') j++;
            const word = text.slice(i, j);
            take(word === 'null' ? 'null' : word === 'true' || word === 'false' ? 'bool' : 'punct', j);
        } else {
            while (j < n && _PUNCT.includes(text[j])) j++;
            take('punct', j);
        }
    }
    return tokens;
}

/**
 * Fill `el` with `text`: one `span.json-<type>` per token when `text` is valid JSON within the size cap,
 * else the text as it is. Either way `el.textContent === text`. Returns true when it coloured.
 */
export function renderJson(el, text) {
    const plain = String(text ?? '');
    const tokens = tokenizeJson(plain);
    if (!tokens) {
        el.textContent = plain;
        return false;
    }
    const frag = document.createDocumentFragment();
    for (const t of tokens) {
        const span = document.createElement('span');
        span.className = `json-${t.type}`;
        span.textContent = t.text;
        frag.appendChild(span);
    }
    el.replaceChildren(frag);
    return true;
}

/** Colour every `.json-view` under `root`; call it after `root.innerHTML = html` built with `jsonViewHtml`. */
export function highlightJsonIn(root) {
    if (!root) return;
    for (const el of root.querySelectorAll(`.${JSON_VIEW_CLASS}`)) renderJson(el, el.textContent);
}
