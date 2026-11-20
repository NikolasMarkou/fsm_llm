# utils

Path: `src/fsm_llm/monitor/static/utils`
Purpose: Browser helpers for the FSM-LLM Monitor frontend: escaping and DOM helpers, formatting, Markdown-to-HTML, JSON colouring, SVG graph layout.

## Scope

The ES modules used by the monitor dashboard (`fsm-llm-monitor`, a FastAPI app in `src/fsm_llm/monitor/server/`). The frontend has no build step or framework; `server/application.py` mounts `static/` at `/static` and a middleware sets `Cache-Control: no-cache, no-store, must-revalidate` on `/static/` paths. No network calls here; server access lives in `static/services/api.js`.

Consumers (import sites): `static/app.js`, `static/services/auth.js`, `static/services/ws.js`, and `static/pages/` `builder.js`, `control.js`, `conversations.js`, `dashboard.js`, `launch.js`, `logs.js`, `settings.js`, `visualizer.js`. `graph.js` is used by `builder.js` and `visualizer.js`; `markdown.js` by `builder.js` and `conversations.js`; `json_highlight.js` by `builder.js`, `conversations.js`, `control.js` and `eventlog.js` (and by `dom.js`, for the class name); `format.js` by `control.js`, `dashboard.js`, `logs.js`.

State: `dom.js` keeps one module-level `Map` (`_returnFocus`) for dialog focus restore. `graph.js` mutates the node objects passed in. Everything else is stateless.

## Architecture

```mermaid
flowchart LR
    dom[dom.js esc] --> md[markdown.js]
    dom --> graph[graph.js]
    fmt[format.js]
    md & graph & dom & fmt --> pages[static/pages/*.js, app.js, services/auth.js, services/ws.js]
```

Callers build HTML as template strings and assign `innerHTML`; these helpers make the interpolated values safe and consistent.

`graph.js` flow: `layoutNodes` (private) builds adjacency ignoring self-loops, picks the start node, runs BFS to assign layers, places layers; then `renderGraph` computes bounds and the viewBox and emits edges followed by nodes as one SVG string assigned to the SVG's `innerHTML`.

`markdown.js` flow: escape the whole input with `esc`, then apply a fixed sequence of regex replacements (listed under Public interface).

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `dom.js` | Escaping, DOM, UI feedback, dialogs | Base of the XSS model: `esc()` |
| `format.js` | Time and number formatting | No imports |
| `markdown.js` | Safe Markdown subset renderer | Imports `esc`; escapes first, then regex transforms |
| `json_highlight.js` | JSON tokenizer and `textContent`-only renderer | Imports nothing; a size cap and a plain fallback |
| `graph.js` | BFS layered layout + SVG string render | Imports `esc`; node box `W=180`, `H=60` |

## Public interface

`dom.js`
- `esc(s): string` - escapes `& < > " '` to entities; `null`/`undefined` -> `''`.
- `$(id): HTMLElement | null` - `document.getElementById`.
- `showError(elementId, msg)` - sets innerHTML to `<span class="error-message">` with escaped msg; no-op if element missing.
- `showStatus(elementId, msg, color)` - `<span class="status-msg status-<color>">`; empty `msg` clears. `color` is interpolated unescaped into the class.
- `showToast(msg, type)` - removes any existing `.toast`, appends a new `div.toast.toast-success|toast-error` (success only when `type === 'success'`), adds `toast-visible` next frame, removes it after 4000 ms, then the node 300 ms later. Uses `textContent`. Also clears `#toast-live` and sets its text after 50 ms (polite live region in `templates/index.html`).
- `safeClass(value, fallback = 'unknown'): string` (module-private, used by `statusBadge`) - lowercases and strips everything outside `[a-z0-9_-]`; empty -> fallback.
- `levelClass(level): string` - lowercased level if in `trace, debug, info, success, warning, error, critical`, else `'info'`.
- `statusBadge(status): string` - `<span class="badge badge-<safeClass(status)>">` with escaped upper-case text; null or `''` -> `unknown`.
- `openDialog(el, display = 'flex', focusTarget = null)` - sets `el.style.display`. Only when the element was hidden: records `document.activeElement` (unless it is `body`) and focuses `focusTarget` or the first visible, enabled focusable descendant.
- `closeDialog(el)` - no-op if hidden; sets `display: none`, restores and forgets the recorded focus if that element is still in the document.
- `renderResultBanner(success): string` - `div.result-banner.success|failure` with fixed text `Agent completed successfully` / `Agent failed`.
- `jsonViewHtml(text, tag = 'span'): string` - `<tag class="json-view">` holding the escaped text, the marker that `highlightJsonIn` finds.
- `renderLLMData(obj): string` - `div.llm-kv` rows; skips null values; objects as pretty JSON in `pre.json-view`; non-object -> `No data`, no rows -> `Empty`.
- `highlightText(text, query): string` - regex-escapes the raw query, splits the raw text on case-insensitive matches, escapes every part and wraps the matches in `span.search-highlight` (so `amp` never matches inside `&amp;`).
- `renderPagination(el, page, totalPages, action): void` - hides `el` for one page, else fills it with Prev/Next buttons whose `data-action` is `<action>-prev` / `<action>-next`.
- `showDrawerDetailView(): void` - shows the Control drawer's detail view and hides the conversation sub-view.
- `hashInstances(items): number` - 32-bit rolling hash seeded with `items.length` over `instance_id:status` of each item. Change detection only.
- `numVal(id, fallback)`, `intVal(id, fallback)` - `parseFloat` / `parseInt(..., 10)` of an input's value; non-finite or missing element -> fallback.
- `copyToClipboard(text): Promise<true>` - `navigator.clipboard.writeText`, on error falls back to hidden textarea + `document.execCommand('copy')`.

`format.js`
- `formatTime(ts)` - falsy -> `''`; `toLocaleTimeString('en-US', { hour12: false })`; invalid date -> `String(ts).substring(11, 19)`.
- `relativeTime(dateStr)` - falsy -> `''`; `just now` (<5 s), `Ns ago`, `Nm ago`, `Nh ago`, `Nd ago`.
- `formatNumber(n)` - null/NaN -> `'0'`; `>=1e9` B, `>=1e6` M, `>=1e4` K (one decimal, trailing `.0` stripped), `>=1000` `en-US` commas, else `String(n)`.

`markdown.js`
- `renderMarkdown(text): string` - falsy -> `''`. Order: `esc` (and NULs dropped) -> fenced code to `pre.md-code-block` and inline code to `code.md-code-inline`, both stashed behind NUL placeholders so no later pass touches them -> `###`/`##`/`#` to `<strong class="md-h3|md-h2|md-h1">` -> `**`/`__` bold -> `*`/`_` italic -> `-`/`*` items wrapped in `<ul>` -> `1.` runs as `<ol>` (a distinct `<oli>` marker keeps them out of the `<ul>` pass) -> `---`/`***` to `<hr class="md-hr">` -> blank lines to `</p><p>`, newlines to `<br>` -> wrap in `<p>` -> restore the stashed code -> cleanup of empty `<p>` and `<p>`/`<br>` around `pre`, `ul`, `ol`, `hr`.

`json_highlight.js`
- `tokenizeJson(text, lenient = false): {type, text}[] | null` - `type` is `key`, `string`, `number`, `bool`, `null` or `punct`; one linear scan, no recursion and no regex, so nesting depth costs no stack. Whitespace belongs to the neighbouring `punct` token and no character is dropped: the token texts joined equal `text`. `null` for a non-string, a text over `JSON_HIGHLIGHT_MAX_CHARS` (200000) or, unless `lenient`, a text `JSON.parse` refuses (`lenient` colours whatever the scan finds, for text that is invalid while it is typed). A string is a `key` when the next non-space character is `:`.
- `renderJson(el, text, lenient = false): boolean` - fills `el` with one `span.json-<type>` per token (a `DocumentFragment`, then `replaceChildren`) and returns true, else sets `el.textContent = text` and returns false. Either way `el.textContent === text`.
- `attachJsonOverlay(textarea): HTMLPreElement | null` - wraps an editable textarea in `div.json-editor` with a `pre.json-overlay` behind it (the textarea text is transparent, its caret and selection stay); the overlay is refilled by `renderJson(..., lenient)` on every `input` event and follows the scroll. A value set from code fires no `input`: dispatch one. Once per textarea, null for none. Used for the Visualizer FSM source (`initVizEditor`).
- `highlightJsonIn(root)` - runs `renderJson` on every `.json-view` under `root` with its own text; call it after `root.innerHTML = html`. Idempotent, null-safe.

`graph.js`
- `renderGraph(svgId, data, opts = {})` - no-op if the SVG element is missing. Replaces the SVG's `innerHTML` and sets its `width`/`height` to `100%`, `viewBox`, and `style.minWidth`/`minHeight`. Returns nothing. Shapes of `data` and `opts` are under Data shapes.

## Data shapes

`renderGraph` input `data`: `{ nodes: Node[], edges: Edge[] }`.

Node (fields read):

| Field | Use |
| --- | --- |
| `id` | Key for edges; label fallback |
| `label` | Main text; falls back to `id` |
| `is_initial` | BFS start (first match wins); adds class `initial` |
| `is_terminal` | Adds class `terminal` (only if not initial) |
| `step_type` | Subtitle text |
| `description` | Subtitle when no `step_type`, first 24 chars |

Node fields written: `x`, `y` (centre coordinates in viewBox space).

Edge: `{ from, to, label? }`. `from`/`to` are node ids; edges whose ends are not in `nodes` are skipped. `from === to` is a self-loop.

`opts`:

| Key | Default | Use |
| --- | --- | --- |
| `colorVar` | `var(--primary-dim)` | Default for `arrowColor` |
| `arrowColor` | `colorVar` | Arrow marker fill |
| `rx` | `4` | Node rect corner radius |
| `nodeClass` | `'fsm'` | Emitted as `node-<nodeClass>` on each rect |

Emitted SVG: `<defs><marker id="arrow-<svgId>">`, then `path|line.edge-line`, `text.edge-label`, then per node `rect.node-rect.node-<nodeClass>[.initial|.terminal]`, `text.node-label`, `text.node-subtitle`.

`hashInstances` items: objects with `instance_id` and `status`.

## Invariants and constraints

- XSS: every interpolated value must go through `esc()`, `safeClass()`, `levelClass()`, or another escaping helper. `renderMarkdown` depends on escaping first; do not add transforms that reintroduce raw input. `showStatus` `color` and `graph.js` `opts` values are not escaped, so pass only trusted constants.
- `graph.js` layout: start is the first node with `is_initial`, else `nodes[0]`. Unreachable nodes go to layer `max+1`. Layers wrap into rows of at most `MAX_COLS = 5` columns in serpentine order (odd rows reversed). Constants `XPAD=120`, `YPAD=100`, outer `PAD=60`, plus 60 px top padding for self-loop arcs.
- Edges: self-loops are a cubic arc above the node; a node pair with more than one edge gets quadratic curves offset +12 then -12 px; other edges are straight lines. Arrow marker id is `arrow-<svgId>`, so several graphs can share a page.
- The viewBox is the larger of content and container size; content is centered when the container is larger. `minWidth`/`minHeight` are set only when content exceeds the container.
- `json_highlight.js` builds nodes with `textContent` only: never `innerHTML` for a token or for the fallback (a probe records every `innerHTML` write and every created tag), so a JSON string holding `<script>` or HTML stays text. The page text is the source of truth and the coloured nodes must hold exactly the same characters. Only object or array values are marked `json-view` (a plain string that happens to read as a number would be coloured as one). The colours are CSS variables in `:root` of `style.css` (the dashboard has one dark theme; a test checks the 4.5:1 contrast and that no second theme appeared).
- `openDialog`/`closeDialog` must be used as a pair on the same element for focus restore to work.

## Dependencies

- Internal: `markdown.js` and `graph.js` import `esc` from `dom.js`; `dom.js` imports `JSON_VIEW_CLASS` from `json_highlight.js`, which imports nothing. Nothing else.
- External: none. Browser APIs only (DOM, `navigator.clipboard`, `requestAnimationFrame`, `getComputedStyle`).
- CSS classes used (`toast`, `toast-visible`, `badge`, `error-message`, `status-msg`, `result-banner`, `llm-kv`, `search-highlight`, `json-key`, `json-string`, `json-number`, `json-bool`, `json-null`, `json-punct`, `node-rect`, `edge-line`, `edge-label`, `node-label`, `node-subtitle`, `md-code-block`, `md-hr`, `text-dim`) are defined in `static/style.css`.
- `showToast` expects `#toast-live` in `templates/index.html`; it skips the mirror if missing.

## Failure modes

- `highlightText` escapes regex metacharacters, so user input should not make `RegExp` throw.
- `copyToClipboard` never rejects and always returns `true`, even if the fallback copy fails.
- `formatTime` on an unparseable non-ISO string returns a substring that may be meaningless.
- `relativeTime` on an unparseable date yields `NaN`, which falls through every comparison to `NaNd ago`.
- `renderGraph` with an empty `nodes` array produces infinite/negative bounds; callers should not pass empty graphs. `layoutNodes` returns early on empty input.
- `hashInstances` throws if `items` is not an array-like with `length`.

## Working here

- Keep helpers free of server calls; anything needing the server belongs with `static/services/api.js` callers.
- Adding a Markdown feature: insert the regex step after `esc()` and before paragraph wrapping, and add a cleanup rule if it produces block elements.
- Changing node box size: update `W`/`H` in `graph.js` and the matching CSS.
- Adding a new file here: `pyproject.toml` package data (`"fsm_llm.monitor" = ["static/**/*", "templates/*"]`) already ships it; `tests/test_packaging.py` checks every static file is covered.
- Renaming or removing a file breaks `tests/test_fsm_llm_monitor/test_app.py`, which lists `utils/dom.js`, `utils/format.js`, `utils/markdown.js`, `utils/json_highlight.js`, `utils/graph.js` for serving and existence checks.
- `markdown.js` and `json_highlight.js` have node probes (`tests/test_fsm_llm_monitor/test_static_js.py`; `json_surfaces.mjs` probes the pages that show JSON). A new page or panel that shows JSON writes it with `jsonViewHtml` and calls `highlightJsonIn` after `innerHTML`, and gets a row in that probe. Check the other files visually in the Visualizer and Builder pages of `fsm-llm-monitor` (default `http://127.0.0.1:8420`).
- Style: 4-space indent, ES module named exports, JSDoc one-liners.
