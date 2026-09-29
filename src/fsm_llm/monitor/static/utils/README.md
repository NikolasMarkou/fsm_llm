# utils

Small browser helpers for the FSM-LLM Monitor web dashboard, in `src/fsm_llm/monitor/static/utils`. They cover safe HTML building, time and number formatting, a tiny Markdown renderer, and an SVG graph drawer.

## What it is for

The FSM-LLM Monitor (command `fsm-llm-monitor`) is a web dashboard for watching and driving FSM conversations, agents, and workflows. Its frontend is plain JavaScript modules with no build step, served as-is by the monitor's FastAPI server under `/static/`. The pages build HTML as strings. These helpers keep that safe (values are escaped before they go into HTML) and consistent (the same time format, badges, toasts, and graph look on every page). None of them talk to the server.

## How it works

- `dom.js` escapes text, looks up elements, shows errors, status lines, toasts, and badges, opens and closes dialogs with focus handling, and parses form values.
- `format.js` turns timestamps and numbers into short readable strings.
- `markdown.js` escapes the input first, then converts a small subset of Markdown into HTML with regular expressions. Because escaping comes first, raw HTML in the text is shown as text, not run.
- `graph.js` places states or steps in layers using breadth-first search (BFS, visiting nodes level by level) from the initial node, then draws boxes and arrows into an `<svg>` element.

`markdown.js` and `graph.js` import `esc` from `dom.js`. `format.js` imports nothing.

## Files

- `dom.js` - escaping, element lookup, toasts, status and error lines, badges, safe CSS class tokens, dialog open/close with focus restore, form value parsing, clipboard copy, search highlighting, instance-list hashing.
- `format.js` - `formatTime`, `relativeTime`, `formatNumber`.
- `markdown.js` - `renderMarkdown(text)`.
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

- `renderMarkdown` supports fenced and inline code, `#`/`##`/`###` headings (rendered as bold text, not real headings), bold, italic, `-`/`*` bullet lists, numbered items, and horizontal rules. Numbered items become `<li>` elements but are not wrapped in an `<ol>`. Links and tables are not supported.
- `renderGraph` changes the node objects you pass in: it writes `x` and `y` onto each one.
- Nodes the layout cannot reach from the initial node all go in one extra layer at the end.
- `showToast` shows a success style only when the type is exactly `'success'`; any other type gets the error style. It also copies the text into the `#toast-live` element so screen readers announce it.
- `copyToClipboard` falls back to the old `document.execCommand('copy')` path and always resolves to `true`.
- There are no JavaScript unit tests. Python tests only check that the four files exist and are served.
