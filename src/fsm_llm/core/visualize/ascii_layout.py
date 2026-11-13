"""ASCII drawing constants, box geometry, state ordering and the state boxes.

Layer ``core`` (plan d454fd62, step 76): split out of the old ``fsm_llm.visualizer``.
The glyph tables (``BOX_STYLES``, ``ARROW_STYLES``, ``ICONS``), the row-fitting helpers,
the graph walks (depths, longest path, loops, ordering) and ``build_state_boxes``.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from typing import Any

from .shapes import _is_terminal, _key_list, _transitions_of

# ASCII drawing characters for different box styles
BOX_STYLES = {
    "default": {
        "topleft": "┌",
        "topright": "┐",
        "bottomleft": "└",
        "bottomright": "┘",
        "horizontal": "─",
        "vertical": "│",
        "title_sep": "─",
    },
    "initial": {
        "topleft": "╔",
        "topright": "╗",
        "bottomleft": "╚",
        "bottomright": "╝",
        "horizontal": "═",
        "vertical": "║",
        "title_sep": "═",
    },
    "terminal": {
        "topleft": "┏",
        "topright": "┓",
        "bottomleft": "┗",
        "bottomright": "┛",
        "horizontal": "━",
        "vertical": "┃",
        "title_sep": "━",
    },
    "both": {  # For states that are both initial and terminal
        "topleft": "╔",
        "topright": "╗",
        "bottomleft": "┗",
        "bottomright": "┛",
        "horizontal": "═",
        "vertical": "║",
        "title_sep": "═",
    },
}


# Arrow styles for transitions
ARROW_STYLES = {
    "forward": "↓",
    "backward": "↑",
    "self": "⟲",
    "connector": "→",
    "bidirectional": "↔",
}


# Icons for different state attributes
ICONS = {
    "input": "*",  # States that require user input
    "branching": "+",  # States with multiple outbound transitions
    "merge": "o",  # States with multiple inbound transitions
    "key": "#",  # Used for required keys
}


# The ASCII layouts of `visualize_fsm_ascii` (CLI `--style`).
ASCII_STYLES: tuple[str, ...] = ("full", "compact", "minimal")


# The compact diagram's per-state prefix, keyed like `BOX_STYLES`.
_COMPACT_PREFIXES = {
    "both": "╔═╤═╝ ",
    "initial": "╔═══> ",
    "terminal": "┗━━━> ",
    "default": "────> ",
}


# Width, in characters, of the content area BETWEEN the two vertical glyphs of a
# per-state mini box in the STATES section. Every content line must be padded (and
# truncated) to exactly this, or the box's right border stops lining up.
_STATES_BOX_INNER_WIDTH = 56


# The marker left behind when a row is shortened. Exactly ONE character wide, so
# every `len()`-based alignment guarantee in this module is unaffected.
_ELLIPSIS = "…"


def _fit(content: str, width: int) -> str:
    """Shorten ``content`` to ``width``, VISIBLY and without losing the tail.

    Args:
        content: the row's logical text.
        width: the exact number of characters available.

    Returns:
        ``content`` unchanged when it fits, otherwise ``head + "…" + tail`` of
        exactly ``width`` characters.
    """
    # DECISION plan-2026-07-19T191147-4b664252/D-033 [STALE]
    # D-028 fixed the ALIGNMENT by slicing every over-long row to the box width. That
    # was correct and is not being undone — but a plain head slice is SILENT and
    # LOSSY-AT-THE-TAIL, and for rows that carry a state id those two properties
    # combine into something worse than a ragged box: two DISTINCT states sharing a
    # long prefix render byte-identically. Measured on the TRANSITIONS graph,
    # `checkout_payment_authorization_pending_manual_review_ALPHA` and `..._BRAVO`
    # both rendered as `│ From: checkout_payment_authorization_pending_manual_review │`
    # with no marker of any kind, and a 100-char id survived nowhere in the `full` or
    # `detailed` render. A diagram whose whole purpose is showing which state goes
    # where must not silently draw two different states as the same node.
    # MIDDLE elision fixes both halves at once: the `…` makes the loss visible, and
    # keeping the TAIL is what keeps same-prefix ids distinguishable — a head-only
    # truncation plus an ellipsis would still render ALPHA and BRAVO identically.
    # Do NOT "simplify" this back to `content[:width]`, and do NOT drop the tail half.
    # See decisions.md D-033.
    if len(content) <= width:
        return content
    if width <= 1:
        return content[:width]
    keep = width - 1
    head = (keep + 1) // 2
    return content[:head] + _ELLIPSIS + content[len(content) - (keep - head) :]


def _states_box_row(vertical: str, content: str) -> str:
    """Render one content row of a STATES-section mini box.

    Args:
        vertical: the box style's vertical glyph (``BOX_STYLES[...]["vertical"]``).
        content: the row's text WITHOUT either border glyph, and without any
            padding — this function owns both.

    Returns:
        A row of exactly ``_STATES_BOX_INNER_WIDTH + 6`` characters, so that every
        row of every box lines up with the section's outer border.
    """
    # DECISION plan-2026-07-19T191147-4b664252/D-022 [STALE]
    # The `[:width]` slice before `.ljust(width)` is load-bearing, not defensive.
    # `ljust` NEVER shortens, so before this helper existed a long state id or a long
    # required-keys list pushed the right-hand border out of alignment for the rest of
    # the box. This mirrors `build_state_boxes`' `[: box_width - 1].ljust(box_width - 1)`
    # (see the `[REUSE]` note in findings/cli-exports-coherence.md). Do NOT drop the
    # slice, and do NOT let callers pre-pad `content`. See decisions.md D-022.
    # D-033: `_fit` replaces the bare `[:width]` slice so an over-long id is
    # shortened VISIBLY and keeps its tail. The padding contract is unchanged.
    # D-002 (plan-a8e47b88): `build_state_boxes` now routes through `_fit` too.
    # D-007 (plan-a8e47b88): the sibling icon-budget branch in
    # `build_states_section` (line ~569, at the time of this note) also fixed.
    width = _STATES_BOX_INNER_WIDTH
    return "│ " + vertical + _fit(content, width).ljust(width) + vertical + " │"


# Width, in characters, of the content area between the outer "│ " and " │" of any
# top-level section box (METADATA, PERSONA, STATES, TRANSITIONS). The box border is
# "┌" + 60 * "─" + "┐" = 62 columns, so a content row is 1 + 1 + 58 + 1 + 1 = 62.
_SECTION_BOX_INNER_WIDTH = 58


def _section_box_row(content: str) -> str:
    """Render one content row of a top-level section box.

    Args:
        content: the row's text WITHOUT either border glyph and without padding —
            this function owns both. Callers pass the full logical line
            (``f"Initial State: {state_id}"``), not a pre-``ljust``-ed fragment.

    Returns:
        A row of exactly ``_SECTION_BOX_INNER_WIDTH + 4`` == 62 characters, matching
        the section's own border, for ANY input including one longer than the box.
    """
    # DECISION plan-2026-07-19T191147-4b664252/D-028 [STALE]
    # SC-18 says "a 100-char state id does not break box alignment" — unscoped. Step 14
    # fixed only `build_states_section` (via `_states_box_row`) and the pinning test
    # measured only that section, so the suite stayed green while METADATA rendered a
    # 119-char row, TRANSITIONS a 127-char row and PERSONA a stray 58-char row against
    # the same 62-char border. That is a scope-narrowed test standing in for a broader
    # claim. The cause was uniform: every row was built as `"│ " + text.ljust(N)` with a
    # HAND-COUNTED N per row, and `ljust` never shortens. Hand-counted widths also drifted
    # on their own — several METADATA rows were 63-69 chars with no long id involved.
    # Do NOT go back to per-row `.ljust(N)` literals in any section builder: route every
    # bordered row through this helper so a new row cannot be added at the wrong width,
    # and so truncation is automatic rather than per-call. The `[:width]` slice is
    # load-bearing for the same reason it is in `_states_box_row`.
    # See decisions.md D-028.
    # D-033: `_fit` replaces the bare `[:width]` slice — see `_fit` for why a
    # silent head-only truncation made the TRANSITIONS graph ambiguous.
    width = _SECTION_BOX_INNER_WIDTH
    return "│ " + _fit(content, width).ljust(width) + " │"


def _box_style_name(is_initial: bool, is_terminal: bool) -> str:
    """The ``BOX_STYLES`` key of a state: ``both``, ``initial``, ``terminal`` or
    ``default``."""
    if is_initial and is_terminal:
        return "both"
    if is_initial:
        return "initial"
    return "terminal" if is_terminal else "default"


def _state_type_label(is_initial: bool, is_terminal: bool) -> str:
    """`` (INITIAL)``, `` (TERMINAL)``, `` (INITIAL, TERMINAL)`` or ``""``."""
    kinds = [k for k, on in (("INITIAL", is_initial), ("TERMINAL", is_terminal)) if on]
    return f" ({', '.join(kinds)})" if kinds else ""


def _state_icons(metrics: dict[str, Any]) -> list[str]:
    """The icons of a state: user input, branching, merge (in that order)."""
    return [
        icon
        for icon, on in (
            (ICONS["input"], metrics.get("required_keys")),
            (ICONS["branching"], metrics.get("outbound", 0) > 1),
            (ICONS["merge"], metrics.get("inbound", 0) > 1),
        )
        if on
    ]


def build_graph_representation(
    states: dict[str, Any], initial_state: str | None = None
) -> tuple[dict[str, list], dict[str, dict[str, Any]]]:
    """Build a representation of the graph structure and analyze state metrics."""
    graph = {}
    state_metrics: dict[str, dict[str, Any]] = {}

    # First pass: build the basic graph
    for state_id, state in states.items():
        targets = []
        for transition in _transitions_of(state):
            target = transition.get("target_state", "")
            desc = transition.get("description", "")
            # Extract required context keys if available
            required_keys: list[str] = []
            for condition in transition.get("conditions") or []:
                required_keys.extend(
                    _key_list(
                        condition.get("requires_context_keys"),
                        f"'requires_context_keys' of a condition of state {state_id!r}",
                    )
                )

            # Add this transition to the targets list
            targets.append((target, desc, required_keys))

        graph[state_id] = targets

        # Initialize metrics
        state_metrics[state_id] = {
            "outbound": len(targets),
            "inbound": 0,
            "required_keys": _key_list(
                state.get("required_context_keys"),
                f"'required_context_keys' of state {state_id!r}",
            ),
            "is_terminal": _is_terminal(state),
            "depth": 0,  # Will be calculated in the next pass
        }

    # Second pass: calculate inbound transitions and depths
    for _state_id, targets in graph.items():
        for target, _, _ in targets:
            if target in state_metrics:
                state_metrics[target]["inbound"] += 1

    # Calculate depths (distance from initial state)
    calculate_depths(graph, state_metrics, initial_state)

    return graph, state_metrics


def calculate_depths(
    graph: dict[str, list],
    state_metrics: dict[str, dict[str, Any]],
    initial_state: str | None = None,
) -> None:
    """Calculate the depth of each state from the initial state."""
    # Find the initial state if not provided
    if initial_state is None:
        # Assume the state with no inbound transitions is the initial state
        for state_id, metrics in state_metrics.items():
            if metrics["inbound"] == 0:
                initial_state = state_id
                break

    if not initial_state:
        return  # Can't determine depths without an initial state

    # Start with the initial state at depth 0
    visited = {initial_state}

    # Use breadth-first search to calculate depths
    current_depth = 0
    current_frontier = [initial_state]

    while current_frontier:
        next_frontier = []
        current_depth += 1

        for state_id in current_frontier:
            for target, _, _ in graph.get(state_id, []):
                if target not in visited:
                    visited.add(target)
                    state_metrics[target]["depth"] = current_depth
                    next_frontier.append(target)

        current_frontier = next_frontier


def find_longest_path(graph: dict[str, list], initial_state: str) -> int:
    """The number of states on the longest path from ``initial_state``.

    A transition back to a state already on the current path (a loop) is not
    followed, so every state is visited once. Exact for a graph without loops.
    With loops it is a LOWER BOUND of the longest simple path (never above it):
    a state's best length is memoised under the path that first reached it and
    reused under another, so the answer can depend on the order of the
    transitions. The exact value is NP-hard in general, and a diagram label
    does not need it.
    """
    # best[s]: states on the longest path that starts at s (s included)
    best: dict[str, int] = {}
    on_path = {initial_state}
    stack = [(initial_state, iter(graph.get(initial_state, [])), 0)]
    while stack:
        state_id, targets, below = stack.pop()
        for target, _desc, _keys in targets:
            if target in best:
                below = max(below, best[target])
            elif target not in on_path:
                on_path.add(target)
                stack.append((state_id, targets, below))
                stack.append((target, iter(graph.get(target, [])), 0))
                break
        else:
            on_path.discard(state_id)
            best[state_id] = below + 1
            if stack:
                parent, parent_targets, parent_below = stack.pop()
                stack.append(
                    (parent, parent_targets, max(parent_below, best[state_id]))
                )
    return best[initial_state]


def sort_states_logically(
    states: dict[str, Any],
    initial_state: str,
    terminal_states: set[str],
    state_metrics: dict[str, dict[str, Any]],
) -> list[str]:
    """Sort states in a logical order for display."""
    # Start with initial state
    sorted_states = [initial_state]
    visited = {initial_state}

    # Sort non-terminal states by depth
    non_terminal_states = [
        (state_id, state_metrics.get(state_id, {}).get("depth", 0))
        for state_id in states
        if state_id != initial_state and state_id not in terminal_states
    ]

    sorted_non_terminals = [
        s[0] for s in sorted(non_terminal_states, key=lambda x: x[1])
    ]

    # Add sorted non-terminals
    for state_id in sorted_non_terminals:
        if state_id not in visited:
            sorted_states.append(state_id)
            visited.add(state_id)

    # End with terminal states
    # DECISION plan-2026-07-18T162030-a02151fe/D-010 [STALE]
    # `sorted()` here is load-bearing, not cosmetic. Do NOT "simplify" this back to
    # `list(terminal_states)`: `terminal_states` is a set of str, so its iteration order
    # varies with PYTHONHASHSEED and the rendered diagram silently differs run-to-run
    # (measured: 8 distinct orderings across 8 seeds, for both style="full" and
    # style="compact"). Alphabetical is arbitrary but deterministic — that is the point.
    # See decisions.md D-010.
    terminal_states_list = sorted(terminal_states)
    # Don't add initial state again if it's also terminal
    if initial_state in terminal_states_list:
        terminal_states_list.remove(initial_state)

    for ts in terminal_states_list:
        sorted_states.append(ts)
        visited.add(ts)

    # Add any remaining states
    for state_id in states:
        if state_id not in visited:
            sorted_states.append(state_id)
            visited.add(state_id)

    return sorted_states


def sort_states_by_depth(
    initial_state: str, state_metrics: dict[str, dict[str, Any]]
) -> list[str]:
    """Sort states by their depth from the initial state for better layout."""
    depth_groups = defaultdict(list)

    # Group states by depth
    for state_id, metrics in state_metrics.items():
        depth = metrics.get("depth", 0)
        depth_groups[depth].append(state_id)

    # Order by depth, with initial state first
    ordered_states = []
    for depth in sorted(depth_groups.keys()):
        states_at_depth = depth_groups[depth]
        # Make sure initial state comes first
        if initial_state in states_at_depth:
            ordered_states.append(initial_state)
            states_at_depth.remove(initial_state)
        # Add other states at this depth
        ordered_states.extend(states_at_depth)

    return ordered_states


def _box_row(vertical: str, content: str, box_width: int) -> str:
    """One row of a STATE DIAGRAM box: ``vertical``, a space, ``content``,
    padded or elided (``_fit``, head and tail kept) to ``box_width`` columns with
    the closing ``vertical`` last. ``content`` is the text without borders."""
    return _fit(f"{vertical} {content}", box_width - 1).ljust(box_width - 1) + vertical


def build_state_boxes(
    ordered_states: list[str],
    initial_state: str,
    terminal_states: set[str],
    states: dict[str, Any],
) -> dict[str, list[str]]:
    """Create enhanced boxes for each state."""
    state_boxes = {}

    for state_id in ordered_states:
        is_initial = state_id == initial_state
        is_terminal = state_id in terminal_states

        # Choose appropriate box style
        style = BOX_STYLES[_box_style_name(is_initial, is_terminal)]
        vertical = style["vertical"]

        # Get state details
        state_data = states.get(state_id, {})
        description = state_data.get("description", "")
        purpose = state_data.get("purpose", "")

        # Determine box width based on content
        content_width = max(
            len(state_id) + 8,  # Allow for padding and state type
            len(description) + 4 if description else 0,
            len(purpose) + 10 if purpose else 0,
            40,  # Minimum width for readability
        )

        # Limit width for readability
        box_width = min(content_width, 60)

        # Create the box
        box = []

        # Box header
        box.append(
            f"{style['topleft']}{style['horizontal'] * (box_width - 2)}{style['topright']}"
        )

        # State ID with state type (truncate to fit box width)
        # DECISION plan-2026-09-20T114608-a8e47b88/D-002
        # This was the one sibling call site D-022's own comment (above,
        # `_states_box_row`) named but did not migrate: a bare
        # `[: box_width - 1]` head slice, no ellipsis, no tail preservation. Two
        # TERMINAL/INITIAL states sharing a >60-char prefix rendered
        # byte-identical rows here in the STATE DIAGRAM section of
        # `--style full` output, even though `_fit` had already fixed the
        # exact same defect for `build_states_section`/`build_transitions_section`
        # (D-033). Routing through `_fit` completes that sweep; it does not
        # reverse D-033. Do NOT go back to a bare slice here. See decisions.md D-002.
        # The `_fit` now lives in `_box_row`, which every row of this box uses.
        box.append(
            _box_row(
                vertical,
                f"{state_id}{_state_type_label(is_initial, is_terminal)}",
                box_width,
            )
        )

        # Add state description
        box.append(f"{vertical}{style['title_sep'] * (box_width - 2)}{vertical}")
        if description:
            for line in textwrap.wrap(description, width=box_width - 4):
                box.append(_box_row(vertical, line, box_width))

        # Add empty line for separation
        box.append(_box_row(vertical, "", box_width))

        # Add purpose
        if purpose:
            for line in textwrap.wrap(f"Purpose: {purpose}", width=box_width - 4):
                box.append(_box_row(vertical, line, box_width))

        # Add required context keys
        required_keys = state_data.get("required_context_keys") or []
        if required_keys:
            box.append(_box_row(vertical, "", box_width))
            key_str = ", ".join(required_keys)
            for line in textwrap.wrap(f"Requires: {key_str}", width=box_width - 4):
                box.append(_box_row(vertical, line, box_width))

        # Box footer
        box.append(
            f"{style['bottomleft']}{style['horizontal'] * (box_width - 2)}{style['bottomright']}"
        )

        state_boxes[state_id] = box

    return state_boxes


def detect_loops(
    graph: dict[str, list], ordered_states: list[str]
) -> list[tuple[str, ...]]:
    """Find all loops in the FSM using DFS."""
    found_loops: list[tuple[str, ...]] = []

    # Detect self-loops
    for state_id, targets in graph.items():
        for target, desc, _ in targets:
            if target == state_id:
                found_loops.append((state_id, desc))

    # Detect loops between states
    for i, state_id in enumerate(ordered_states):
        for target, desc, _ in graph.get(state_id, []):
            # Skip self-loops (already handled); every target is a state
            # (`_require_valid_transition_targets`)
            if target != state_id and ordered_states.index(target) < i:
                found_loops.append((state_id, target, desc))  # a backward loop

    return found_loops
