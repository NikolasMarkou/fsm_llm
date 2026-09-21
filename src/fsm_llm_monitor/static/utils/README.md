# utils

Small helper functions for the FSM-LLM Monitor web dashboard: safe HTML building, number and time formatting, a tiny Markdown renderer, and an SVG graph drawer.

## What it is for

The FSM-LLM Monitor is a browser dashboard for watching and driving FSM conversations, agents, and workflows. Its pages build HTML as strings. These helpers keep that safe (every value is escaped before insertion) and consistent (the same time format, badge style, and graph look on every page). None of them talk to the server.

## How it works

- `dom.js` escapes text, looks up elements, and shows errors, status lines, toasts, and badges.
- `format.js` turns timestamps and numbers into short readable strings.
- `markdown.js` escapes the input first, then converts a small subset of Markdown into HTML. Because escaping comes first, raw HTML in the text is shown as text, not run.
- `graph.js` lays out states or steps in layers using breadth-first search from the initial node, then draws boxes and arrows into an `<svg>` element.

## Files

- `dom.js` - escaping, element lookup, toasts, status and error lines, badges, form value parsing, clipboard copy, search highlighting, instance-list hashing.
- `format.js` - `formatTime`, `relativeTime`, `formatNumber`.
- `markdown.js` - `renderMarkdown(text)`.
- `graph.js` - `renderGraph(svgId, data, opts)`.

## How to use it

```js
import { esc, showToast, statusBadge } from './utils/dom.js';
import { formatNumber, relativeTime } from './utils/format.js';
import { renderMarkdown } from './utils/markdown.js';
import { renderGraph } from './utils/graph.js';

el.innerHTML = `${statusBadge('running')} ${esc(userText)}`;
formatNumber(12345);           // "12.3K"
relativeTime(new Date());      // "just now"
chat.innerHTML = renderMarkdown('**bold** and `code`');
renderGraph('viz-svg', {
  nodes: [{ id: 'start', is_initial: true }, { id: 'end', is_terminal: true }],
  edges: [{ from: 'start', to: 'end', label: 'done' }],
});
```

## Things to know

- `renderMarkdown` supports fenced and inline code, `#`/`##`/`###` headings (rendered as bold, not real headings), bold, italic, `-`/`*` bullet lists, numbered items, and horizontal rules. Numbered items become `<li>` elements but are not wrapped in an `<ol>`. Links and tables are not supported.
- `renderGraph` mutates the node objects you pass in: it writes `x` and `y` onto each one.
- Nodes the layout cannot reach from the initial node are put in one extra column at the end.
- `copyToClipboard` falls back to the old `document.execCommand('copy')` path and always resolves to `true`.
