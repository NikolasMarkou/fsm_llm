# utils

Path: `src/fsm_llm_monitor/static/utils`
Purpose: Pure browser helpers for the FSM-LLM Monitor frontend: escaping and DOM helpers, formatting, Markdown-to-HTML, SVG graph layout.

## Scope

Stateless ES modules (except `graph.js`, which mutates passed-in nodes). No network calls, no shared state. Consumed by `static/pages/*.js`, `static/services/ws.js`, and `static/app.js`. The monitor frontend has no build step or framework; files are served as-is by the `fsm_llm_monitor` FastAPI app.

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `dom.js` | Escaping, DOM, UI feedback | Base of the XSS model: `esc()` |
| `format.js` | Time and number formatting | No imports |
| `markdown.js` | Safe Markdown subset renderer | Escapes first, then regex transforms |
| `graph.js` | BFS layered layout + SVG string render | Node box 180x60 |

## Public interface

`dom.js`
- `esc(s): string` - escapes `& < > " '`; `null`/`undefined` -> `''`.
- `$(id): HTMLElement | null` - `document.getElementById`.
- `showError(elementId, msg)` - sets `<span class="error-message">`.
- `showStatus(elementId, msg, color)` - `<span class="status-msg status-<color>">`; empty `msg` clears.
- `showToast(msg, type = 'error')` - removes existing `.toast`, shows new one for 4 s (+300 ms fade).
- `statusBadge(status): string` - `<span class="badge badge-<status>">STATUS</span>`. `status` is not escaped in the class attribute.
- `renderResultBanner(success: boolean): string`.
- `renderLLMData(obj): string` - key/value HTML; skips null values; objects as pretty JSON in `<pre>`.
- `highlightText(text, query): string` - escapes both, regex-escapes query, wraps matches in `.search-highlight`.
- `hashInstances(items): number` - 32-bit rolling hash over `instance_id:status` of each item plus length. Used for change detection only.
- `numVal(id, fallback)`, `intVal(id, fallback)` - parse input value; non-finite -> fallback.
- `copyToClipboard(text): Promise<true>` - Clipboard API, falls back to hidden textarea + `execCommand('copy')`.

`format.js`
- `formatTime(ts)` - `HH:MM:SS` via `toLocaleTimeString('en-US', {hour12: false})`; invalid date -> `String(ts).substring(11, 19)`.
- `relativeTime(dateStr)` - `just now` (<5 s), `Ns ago`, `Nm ago`, `Nh ago`, `Nd ago`.
- `formatNumber(n)` - `>=1e9` B, `>=1e6` M, `>=1e4` K (one decimal, trailing `.0` stripped), `>=1000` locale commas, else plain; null/NaN -> `'0'`.

`markdown.js`
- `renderMarkdown(text): string` - order: `esc` -> fenced code -> inline code -> `###/##/#` to `<strong class="md-h3|h2|h1">` -> `**`/`__` bold -> `*`/`_` italic -> `-`/`*` list items wrapped in `<ul>` -> `1.` items as bare `<li>` -> `---`/`***` to `<hr class="md-hr">` -> blank lines to `</p><p>`, newlines to `<br>` -> wrap in `<p>` and clean up empty/misnested tags.

`graph.js`
- `renderGraph(svgId, {nodes, edges}, opts = {})` - no-op if the SVG element is missing.
  - `opts`: `colorVar` (default `var(--primary-dim)`), `arrowColor` (default `colorVar`), `rx` (default 4), `nodeClass` (default `'fsm'`).
  - Node fields read: `id`, `label`, `is_initial`, `is_terminal`, `step_type`, `description` (first 24 chars as subtitle when no `step_type`).
  - Edge fields read: `from`, `to`, `label`.

## Invariants and constraints

- XSS: every interpolated value must pass through `esc()` or an escaping helper. `renderMarkdown` depends on escaping first; do not add transforms that reintroduce raw input.
- `graph.js` layout: start = first node with `is_initial`, else `nodes[0]`. BFS layers; unreachable nodes go to layer `max+1`. Layers wrap into rows of at most 5 columns in a serpentine order (odd rows reversed). Constants `XPAD=120`, `YPAD=100`, `PAD=60`.
- Edges: self-loops drawn as a cubic arc above the node; a pair of nodes with more than one edge between them gets quadratic curves offset +/-12 px; otherwise straight lines. Arrow marker id is `arrow-<svgId>`, so several graphs can share a page.
- The viewBox is the larger of content size and container size; content is centered when the container is larger. `minWidth/minHeight` is set only when content exceeds the container.
- `renderGraph` writes `x`/`y` onto the caller's node objects.

## Dependencies

- `markdown.js` and `graph.js` import `esc` from `dom.js`. No other internal or external dependencies.
- CSS classes referenced (`toast`, `badge-*`, `node-rect`, `edge-line`, `edge-label`, `node-label`, `node-subtitle`, `md-*`, `search-highlight`, `llm-kv`) are defined in `static/style.css`.

## Failure modes

- `highlightText` builds a `RegExp` from escaped input with metacharacters escaped, so it should not throw on user input.
- `copyToClipboard` never rejects and always returns `true`, even if the fallback copy fails.
- `formatTime` on an unparseable non-ISO string returns a substring that may be meaningless.

## Working here

- Keep helpers pure and synchronous where possible; anything needing the server belongs in `static/services/api.js` callers.
- Adding a Markdown feature: insert the regex step after `esc()` and before paragraph wrapping, and add a cleanup rule if it produces block elements.
- Changing node box size: update `W`/`H` in `graph.js` and the matching CSS.
- No automated JS tests. Check visually in the Visualizer and Builder pages of `fsm-llm-monitor`.
