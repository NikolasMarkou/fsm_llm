// JSON colouring (plan 51fde95b D-040): tokens, byte equality, escaping, fallback, the size cap, deep nesting.
import { El, out } from './stub.mjs';
import { tokenizeJson, renderJson, highlightJsonIn, JSON_HIGHLIGHT_MAX_CHARS } from '../static/utils/json_highlight.js';

// Record every created tag and every innerHTML write: the renderer may build `span` nodes and nothing else.
const tags = [], writes = [];
const watch = e => { Object.defineProperty(e, 'innerHTML', { get: () => '', set: v => writes.push(v) }); return e; };
document.createElement = tag => { tags.push(tag); const e = watch(new El('created')); e.tag = tag; return e; };
const shown = text => { const el = watch(new El('t')); const coloured = renderJson(el, text); return { el, coloured }; };
const types = text => (tokenizeJson(text) ?? []).map(t => `${t.type}:${t.text}`);
const equal = text => { const t = tokenizeJson(text); return t !== null && t.map(x => x.text).join('') === text; };

const corpus = [
    '{}', '[]', '{"a": [1, 2, {"b": null}], "c": "d"}', '{\n  "a": 1,\n  "b": [\n    true,\n    false\n  ]\n}\n', ' \t\r\n[1]\t ',
    '"just a string"', '123', 'true', 'null', '-0', '{"é": "😀 \\u00e9 \\ud83d\\ude00 日本"}', '{"p": "C:\\\\", "q": "say \\"hi\\"", "r": "\\\\\\""}',
    '{"a" : 1 , "b":2}', '[1,[2,[3,[4]]]]', '{"<script>": "<script>alert(1)</script>", "x": "&amp; </pre><img src=x onerror=1>"}',
];
const flat = corpus.map(text => {
    const { el, coloured } = shown(text);
    return [equal(text), el.textContent === text, coloured];
});

const bad = ['{a:1}', '{"a":1,}', '', 'plain text', "{'a': 1}", '[1 2]', '{"a": undefined}', '{"a": NaN}', '"unterminated'];
const fallback = bad.map(text => {
    const { el, coloured } = shown(text);
    return [tokenizeJson(text) === null, el.textContent === text, coloured, (el.children ?? []).length];
});
const nonString = [undefined, null, 5, {}].map(v => tokenizeJson(v) === null);
const nullShown = shown(null).el.textContent;

const one = '{"s": "x", "n": -1.5e+10, "t": true, "f": false, "z": null, "arr": [1, "k"]}';
const keyed = '{"k":"v","s":["k"],"sp" \n : "x", "c": "b:c", "n": 1}';
const oneTokens = types(one), keyedTokens = types(keyed);
const nums = types('[-1, 0, 1.5, -2.5e-3, 1E+10, 10, -0, 0.25e2]').filter(t => t.startsWith('number'));
const esc = types('["a\\"b", "\\\\", "C:\\\\", "\\u00e9é😀"]').filter(t => t.startsWith('string'));

const capText = n => '"' + 'a'.repeat(n - 2) + '"';
const atCap = capText(JSON_HIGHLIGHT_MAX_CHARS), overCap = capText(JSON_HIGHLIGHT_MAX_CHARS + 1);
const capShown = shown(overCap);
const capRow = [tokenizeJson(atCap) !== null, tokenizeJson(overCap) === null, capShown.el.textContent === overCap, capShown.coloured];

// Deep nesting: one scan, no recursion. Arrays collapse into one punct run, so objects give a token per level too.
const deep = { array: '['.repeat(90000) + ']'.repeat(90000), object: '{"a":'.repeat(30000) + '1' + '}'.repeat(30000) };
const t0 = Date.now();
const deepRow = Object.values(deep).map(text => {
    const tokens = tokenizeJson(text);
    return [tokens !== null, tokens && tokens.length, shown(text).el.textContent === text];
});
const deepFast = Date.now() - t0 < 10000;

const tagsBefore = tags.length;
const x = shown('{"<script>": "<script>alert(1)</script>", "k": "&lt;b&gt;"}');
const created = tags.slice(tagsBefore);
const xss = { text: x.el.textContent === '{"<script>": "<script>alert(1)</script>", "k": "&lt;b&gt;"}', spansOnly: created.length > 0 && created.every(t => t === 'span'), innerHTMLWrites: writes.length };
const classes = (x.el.children ?? []).map(c => c.className);

// highlightJsonIn: only the .json-view elements it is given, each by its own text; an invalid one stays text.
const a = watch(new El('a')), b = watch(new El('b'));
a.textContent = '{"a": 1}'; b.textContent = 'not json';
let asked;
highlightJsonIn({ querySelectorAll: sel => { asked = sel; return [a, b]; } });
highlightJsonIn(null);
const into = { asked, a: [a.textContent, (a.children ?? []).length], b: [b.textContent, (b.children ?? []).length] };

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const keyedOk = ['key:"k"', 'key:"sp"', 'string:"b:c"'].every(t => keyedTokens.includes(t)) && keyedTokens.filter(t => t === 'string:"k"').length === 1 && !keyedTokens.includes('key:"b:c"');
const checks = {
    everyByteKept: flat.every(r => r.every(Boolean)),
    types: same(oneTokens.filter(t => /^(bool|null)/.test(t)), ['bool:true', 'bool:false', 'null:null']) && oneTokens[1] === 'key:"s"' && oneTokens[3] === 'string:"x"' && oneTokens[7] === 'number:-1.5e+10',
    keyVersusString: keyedOk,
    numbers: same(nums, ['-1', '0', '1.5', '-2.5e-3', '1E+10', '10', '-0', '0.25e2'].map(n => `number:${n}`)),
    stringEscapes: same(esc.slice(0, 3), ['string:"a\\"b"', 'string:"\\\\"', 'string:"C:\\\\"']),
    invalidStaysPlain: fallback.every(r => same(r, [true, true, false, 0])) && nonString.every(Boolean) && nullShown === '',
    sizeCap: JSON_HIGHLIGHT_MAX_CHARS === 200000 && same(capRow, [true, true, true, false]),
    deepNesting: same(deepRow, [[true, 1, true], [true, 60003, true]]) && deepFast,
    htmlStaysText: xss.text && xss.spansOnly && xss.innerHTMLWrites === 0 && writes.length === 0,
    highlightIn: into.asked === '.json-view' && same(into.a, ['{"a": 1}', 5]) && same(into.b, ['not json', 0]),
};
out(checks);
