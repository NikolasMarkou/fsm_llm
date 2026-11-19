# utils

Small browser helpers for the FSM-LLM Monitor web dashboard, in `src/fsm_llm/monitor/static/utils`. They cover safe HTML building, time and number formatting, a tiny Markdown renderer, JSON syntax colouring, and an SVG graph drawer.

## What it is for

The FSM-LLM Monitor (command `fsm-llm-monitor`) is a web dashboard for watching and driving FSM conversations, agents, and workflows. Its frontend is plain JavaScript modules with no build step, served as-is by the monitor's FastAPI server under `/static/`. The pages build HTML as strings. These helpers keep that safe (values are escaped before they go into HTML) and consistent (the same time format, badges, toasts, and graph look on every page). None of them talk to the server.

## How it works

- `dom.js` escapes text, looks up elements, shows errors, status lines, toasts, and badges, opens and closes dialogs with focus handling, and parses form values.
- `format.js` turns timestamps and numbers into short readable strings.
- `markdown.js` escapes the input first, then converts a small subset of Markdown into HTML with regular expressions. Because escaping comes first, raw HTML in the text is shown as text, not run.
- `json_highlight.js` colours JSON text: one linear scan splits it into keys, strings, numbers, booleans, `null` and punctuation, and one `span` per token is built with `textContent`, never `innerHTML`, so a string holding `<script>` stays text and selecting or copying gives the plain JSON. Text that is not valid JSON, or longer than 200000 characters, is shown as it is.
- `graph.js` places states or steps in layers using breadth-first search (BFS, visiting nodes level by level) from the initial node, then draws boxes and arrows into an `<svg>` element.

`markdown.js` and `graph.js` import `esc` from `dom.js`; `dom.js` imports the class name from `json_highlight.js`. `format.js` and `json_highlight.js` import nothing.

## Files

- `dom.js` - escaping, element lookup (`$`), toasts, status and error lines, badges, safe CSS class tokens (`safeClass`, `levelClass`), dialog open/close with focus restore, form value parsing (`numVal`, `intVal`), clipboard copy, search highlighting, instance-list hashing, and two small renderers (`renderResultBanner`, `renderLLMData`).
- `format.js` - `formatTime`, `relativeTime`, `formatNumber`.
- `markdown.js` - `renderMarkdown(text)`.
- `json_highlight.js` - `tokenizeJson`, `renderJson(el, text)`, `highlightJsonIn(root)`.
- `graph.js` - `renderGraph(svgId, data, opts)`.

## How to use it

From a module in `static/pages/`:

```js
import { esc, showToast, statusBadge } from '../utils/dom.js';
import { formatNumber, relativeTime } from '../utils/format.js';
import { renderMarkdown } from '../utils/markdown.js';
import { renderGraph } from '../utils/graph.js';

el.innerHTML = `${statusBadge('running')} ${esc(userText)}`;
formatNumber(12345);           // "12.3K"
relativeTime(new Date());      // "just now"
chat.innerHTML = renderMarkdown('**bold** and `code`');
showToast('Saved', 'success');
renderGraph('viz-svg', {
  nodes: [{ id: 'start', is_initial: true }, { id: 'end', is_terminal: true }],
  edges: [{ from: 'start', to: 'end', label: 'done' }],
});
```

## Things to know

- `renderMarkdown` supports fenced and inline code, `#`/`##`/`###` headings (rendered as bold text, not real headings), bold (`**x**` or `__x__`), italic (`*x*` or `_x_`), `-`/`*` bullet lists, numbered lists (`<ol>`), and horizontal rules. The inside of a code block or inline code is left as typed. Links and tables are not supported.
- `renderGraph(svgId, data, opts)`: nodes take `id`, optional `label`, `is_initial`, `is_terminal`, `step_type` and `description`; edges take `from`, `to` and optional `label`. `opts` may set `colorVar`, `arrowColor`, `rx` and `nodeClass`. The initial node (else the first one) starts the layout, and layouts of more than 5 layers wrap into rows. It changes the node objects you pass in: it writes `x` and `y` onto each one.
- Nodes the layout cannot reach from the initial node all go in one extra layer at the end.
- `showToast` shows a success style only when the type is exactly `'success'`; any other type gets the error style. It also copies the text into the `#toast-live` element so screen readers announce it.
- `copyToClipboard` falls back to the old `document.execCommand('copy')` path and always resolves to `true`.
- `markdown.js` and `json_highlight.js` have node probes, and the pages that show JSON are probed through `json_surfaces.mjs`; the colours themselves (`--json-*` in `style.css`) have a Python contrast test. The pages build HTML as strings, so a page writes its JSON with `jsonViewHtml(text)` (the text is escaped there) and calls `highlightJsonIn(container)` after setting `innerHTML`; `renderJson(el, text)` is for an element that is filled by `textContent`. Other files here have no node probe (`tests/test_fsm_llm_monitor/test_static_js.py`); for the other files Python tests only check that they exist and are served.
