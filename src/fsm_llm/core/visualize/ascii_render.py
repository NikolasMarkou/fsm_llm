"""The ASCII renderer: ``visualize_fsm_ascii`` and its three layouts.

Layer ``core`` (plan d454fd62, step 76): split out of the old ``fsm_llm.visualizer``.
The header, metadata, persona, states and transitions sections, the legend and the full,
compact and minimal diagrams; the geometry they use is in ``ascii_layout``.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from typing import Any

from .ascii_layout import (
    _COMPACT_PREFIXES,
    _SECTION_BOX_INNER_WIDTH,
    _STATES_BOX_INNER_WIDTH,
    ARROW_STYLES,
    ASCII_STYLES,
    BOX_STYLES,
    ICONS,
    _box_style_name,
    _fit,
    _section_box_row,
    _state_icons,
    _state_type_label,
    _states_box_row,
    build_graph_representation,
    build_state_boxes,
    detect_loops,
    find_longest_path,
    sort_states_by_depth,
    sort_states_logically,
)
from .shapes import (
    _DEFAULT_NAME,
    _one_line,
    _require_definition_shapes,
    _require_initial_state,
    _require_valid_transition_targets,
)


def visualize_fsm_ascii(fsm_data: dict[str, Any], style: str = "full") -> str:
    """
    Generate an enhanced ASCII visualization of an FSM.

    Args:
        fsm_data: The FSM definition as a dictionary
        style: Visualization style, one of ``ASCII_STYLES``: "full", "compact"
            or "minimal"

    Returns:
        A string containing the ASCII visualization

    Raises:
        ValueError: if ``style`` is unknown, ``states``, a state or a transition
            is not a mapping, a key list is not a list of strings,
            ``initial_state`` is not present in ``states``, or any transition
            names a ``target_state`` that is not a key of ``states``. Every
            style refuses the same input, so a renderer failure is never
            hidden behind a half diagram.
    """
    if style not in ASCII_STYLES:
        raise ValueError(
            f"unknown style {style!r} (choose from {', '.join(ASCII_STYLES)})"
        )
    states = fsm_data.get("states", {})
    initial_state = fsm_data.get("initial_state", "")
    persona = fsm_data.get("persona", None)

    _require_definition_shapes(states)
    _require_initial_state(states, initial_state)
    _require_valid_transition_targets(states)

    # Build a representation of the graph structure and analyze it
    graph, state_metrics = build_graph_representation(states, initial_state)
    terminal_states = {s for s, m in state_metrics.items() if m["is_terminal"]}

    # Create ASCII visualization
    lines = []

    # Create header based on style
    name = _one_line(fsm_data.get("name"), _DEFAULT_NAME)
    if style != "minimal":
        lines.extend(build_fancy_header(name))
    else:
        # Simple header for minimal style
        lines.append(f"FSM: {name}")
        lines.append("")

    # Add metadata based on style
    if style == "full":
        lines.extend(
            build_metadata_section(fsm_data, initial_state, state_metrics, graph)
        )

        # Add persona section if available
        if persona:
            lines.extend(build_persona_section(persona))

        # Add states section
        lines.extend(
            build_states_section(states, initial_state, terminal_states, state_metrics)
        )

        # Add transitions section
        lines.extend(build_transitions_section(graph))
    elif style == "compact" and persona:
        # In compact mode, just show a simplified persona section
        lines.append("╭" + "─" * 60 + "╮")
        # D-028: this row was `"│ " + ljust(10) + ljust(50) + " │"` == 64 against a
        # 62-char border — ragged on every compact render carrying a persona,
        # independent of state-id length.
        _compact_label = "PERSONA:  "
        lines.append(
            _section_box_row(
                _compact_label
                + textwrap.shorten(
                    persona, width=_SECTION_BOX_INNER_WIDTH - len(_compact_label)
                )
            )
        )
        lines.append("╰" + "─" * 60 + "╯")
        lines.append("")

    # Create diagram based on style
    if style != "minimal":
        lines.append("")
        lines.append("╭" + "─" * 60 + "╮")
        lines.append("│" + " STATE DIAGRAM ".center(60) + "│")
        lines.append("╰" + "─" * 60 + "╯")
        lines.append("")

        # Add legend
        lines.extend(
            build_legend(
                initial_state in terminal_states,
                any(len(m["required_keys"]) > 0 for m in state_metrics.values()),
            )
        )

    lines.append("")

    # Generate diagram based on selected style
    if style == "minimal":
        diagram_lines = generate_minimal_ascii_diagram(
            graph, initial_state, terminal_states, state_metrics
        )
    elif style == "compact":
        diagram_lines = generate_compact_ascii_diagram(
            graph, initial_state, terminal_states, states, state_metrics
        )
    else:  # full
        diagram_lines = generate_enhanced_ascii_diagram(
            graph, initial_state, terminal_states, states, state_metrics
        )

    lines.extend(diagram_lines)
    return "\n".join(lines)


def build_fancy_header(name: str) -> list[str]:
    """Create a fancy header for the FSM visualization."""
    # DECISION plan-2026-09-20T114608-a8e47b88/D-019
    # Width used to grow with the name (`max(60, len(name) + 10)`), while every
    # sibling top-level box (METADATA/STATES/TRANSITIONS/PERSONA, all built via
    # `_section_box_row`) is hard-coded to 60 -- so an FSM name over ~50 chars
    # rendered a header box WIDER than every box below it, and `.center(width)`
    # never shortens, so a name that did not fit was never truncated either. Do
    # NOT let this box's width vary with content again: cap it to 60 like its
    # siblings, and elide an over-long name through `_fit()` (the same helper
    # D-033/D-002/D-007 use for exactly this failure shape) instead of growing
    # the box. See decisions.md D-019.
    width = 60
    display_name = _fit(name, width - 2)
    lines = [
        "╭" + "─" * width + "╮",
        "│" + f" {display_name} ".center(width) + "│",
        "│" + "FINITE STATE MACHINE VISUALIZATION".center(width) + "│",
        "╰" + "─" * width + "╯",
        "",
    ]
    return lines


def build_metadata_section(
    fsm_data: dict[str, Any],
    initial_state: str,
    state_metrics: dict[str, dict[str, Any]],
    graph: dict[str, list],
) -> list[str]:
    """Create the metadata section of the visualization."""
    lines = [
        "┌" + "─" * 60 + "┐",
        "│" + " METADATA ".center(60) + "│",
        "├" + "─" * 60 + "┤",
    ]

    # Format description with word wrapping
    # DECISION plan-2026-07-18T051819-80b0bd4d/D-009 [STALE]: do NOT "simplify" this back to
    # fsm_data.get("description", "No description") + a bare textwrap.wrap(...).
    # `.get(k, default)` returns None for an explicit null, and textwrap.wrap("")
    # / wrap("   ") both return [], so `wrapped_desc[0]` raised out of this
    # public, non-pydantic-validated function. Same guard at :364 and :470.
    description = fsm_data.get("description") or "No description"
    # Wrap to the space left AFTER the "Description: " label, so a normal
    # description never reaches `_section_box_row`'s truncation.
    _desc_label = "Description: "
    wrapped_desc = textwrap.wrap(
        description, width=_SECTION_BOX_INNER_WIDTH - len(_desc_label)
    ) or [""]

    lines.append(_section_box_row(_desc_label + wrapped_desc[0]))
    for line in wrapped_desc[1:]:
        lines.append(_section_box_row(" " * len(_desc_label) + line))

    lines.append(_section_box_row(f"Version: {fsm_data.get('version', 'N/A')}"))
    lines.append(_section_box_row(f"Initial State: {initial_state}"))
    lines.append(_section_box_row(f"Total States: {len(fsm_data.get('states', {}))}"))

    # Add some statistics
    terminal_count = sum(
        1 for m in state_metrics.values() if m.get("is_terminal", False)
    )
    lines.append(_section_box_row(f"Terminal States: {terminal_count}"))

    input_states = sum(1 for m in state_metrics.values() if m.get("required_keys"))
    lines.append(_section_box_row(f"States With User Input: {input_states}"))

    branching_states = sum(
        1 for m in state_metrics.values() if m.get("outbound", 0) > 1
    )
    lines.append(_section_box_row(f"Branching States: {branching_states}"))

    max_path = find_longest_path(graph, initial_state)
    lines.append(_section_box_row(f"Maximum Path Length: {max_path}"))

    lines.append("└" + "─" * 60 + "┘")
    lines.append("")

    return lines


def build_persona_section(persona: str) -> list[str]:
    """Create the persona section of the visualization."""
    # Prepare a more decorative persona section
    lines = [
        "┌" + "─" * 60 + "┐",
        "│" + " PERSONA ".center(60) + "│",
        "├" + "─" * 60 + "┤",
    ]

    # Add personality type label based on persona content
    if "pirate" in persona.lower():
        lines.append("│" + " Captain's Personality ".center(60) + "│")
    elif "story" in persona.lower() or "tale" in persona.lower():
        lines.append("│" + " Storyteller's Personality ".center(60) + "│")
    elif "teacher" in persona.lower() or "professor" in persona.lower():
        lines.append("│" + " Educator's Personality ".center(60) + "│")
    elif "customer" in persona.lower() or "service" in persona.lower():
        lines.append("│" + " Service Professional Personality ".center(60) + "│")
    else:
        lines.append("│" + " Conversation Personality ".center(60) + "│")

    # NOTE: this spacer used to be a hand-typed literal of 56 spaces between two
    # glyphs — a 58-char row inside a 62-char box, ragged on EVERY render with a
    # persona, with or without a long state id. Routed through the helper (D-028).
    lines.append(_section_box_row(""))

    # Format persona with word wrapping
    wrapped_persona = textwrap.wrap(persona, width=_SECTION_BOX_INNER_WIDTH)
    for line in wrapped_persona:
        lines.append(_section_box_row(line))

    lines.append("└" + "─" * 60 + "┘")
    lines.append("")

    return lines


def build_states_section(
    states: dict[str, Any],
    initial_state: str,
    terminal_states: set[str],
    state_metrics: dict[str, dict[str, Any]],
) -> list[str]:
    """Create the states section of the visualization."""
    lines = [
        "┌" + "─" * 60 + "┐",
        "│" + " STATES ".center(60) + "│",
        "├" + "─" * 60 + "┤",
    ]

    # Sort states logically: initial first, then others in order of depth, then terminals
    sorted_states = sort_states_logically(
        states, initial_state, terminal_states, state_metrics
    )

    for state_id in sorted_states:
        state = states[state_id]
        is_initial = state_id == initial_state
        is_terminal = state_id in terminal_states
        state_type_str = _state_type_label(is_initial, is_terminal)
        metrics = state_metrics.get(state_id, {})

        # Create a mini box for each state with appropriate decoration
        style = BOX_STYLES[_box_style_name(is_initial, is_terminal)]

        width = _STATES_BOX_INNER_WIDTH
        vertical = style["vertical"]

        lines.append(
            "│ "
            + style["topleft"]
            + style["horizontal"] * width
            + style["topright"]
            + " │"
        )

        # State ID with type and metrics.
        # DECISION plan-2026-07-19T191147-4b664252/D-022 [STALE]
        # No leading "│ " here: the border glyph is supplied by `_states_box_row` from the
        # box's own style. Baking one in produced the doubled "║│ greeting" glyph on every
        # state box in the default `full` style. See decisions.md D-022.
        state_line = f" {state_id}{state_type_str}"

        # Add icons for state properties
        icons = _state_icons(metrics)

        if icons:
            icon_str = " " + " ".join(icons)
            # Make sure we don't exceed the box width
            body = width - len(icon_str)
            # DECISION plan-2026-09-20T114608-a8e47b88/D-007
            # A bare `state_line[:body]` here truncates BEFORE `_states_box_row`'s
            # own `_fit()` call ever sees this string: by the time `_fit` runs, the
            # content is already exactly `width` chars, so `_fit` is a no-op. That
            # silently reproduces D-033's byte-identical-row bug for any long-
            # shared-prefix state that ALSO carries an icon (required_keys/
            # branching/merge) -- the D-002 fix at `build_state_boxes` did not
            # cover this sibling site. Route through `_fit` here too, reserving
            # `body` chars for the state-id text so the trailing icon still fits.
            # Do NOT go back to a bare slice. See decisions.md D-007.
            state_line = _fit(state_line, body).ljust(body) + icon_str

        lines.append(_states_box_row(vertical, state_line))

        # Add purpose with word wrapping
        purpose = state.get("purpose", "")
        if purpose:
            label = " Purpose: "
            # `if purpose:` admits whitespace-only, which wraps to [] (D-009)
            wrapped_purpose = textwrap.wrap(purpose, width=width - len(label)) or [""]
            lines.append(_states_box_row(vertical, label + wrapped_purpose[0]))
            for line in wrapped_purpose[1:]:
                lines.append(_states_box_row(vertical, " " * len(label) + line))

        # Add required context keys if any
        required_keys = state.get("required_context_keys") or []
        if required_keys:
            key_str = ", ".join(required_keys)
            lines.append(
                _states_box_row(vertical, "  " + ICONS["key"] + " Required: " + key_str)
            )

        lines.append(
            "│ "
            + style["bottomleft"]
            + style["horizontal"] * width
            + style["bottomright"]
            + " │"
        )

    lines.append("└" + "─" * 60 + "┘")
    lines.append("")

    return lines


def build_transitions_section(graph: dict[str, list]) -> list[str]:
    """Create the transitions section of the visualization."""
    lines = [
        "┌" + "─" * 60 + "┐",
        "│" + " TRANSITIONS ".center(60) + "│",
        "├" + "─" * 60 + "┤",
    ]

    # First find states with incoming transitions to create better tree view
    incoming_transitions = defaultdict(list)
    for source, targets in graph.items():
        for target, desc, required_keys in targets:
            incoming_transitions[target].append((source, desc, required_keys))

    # Process transitions in a more tree-like structure
    for state_id, targets in graph.items():
        # Add decorative elements based on state type
        if len(targets) > 1:
            # Branching state
            branches = f" ({len(targets)} branches)"
            lines.append(_section_box_row(f"From: {state_id}{branches}"))
        elif len(incoming_transitions[state_id]) > 1:
            # Merge state
            merges = f" (merges {len(incoming_transitions[state_id])} paths)"
            lines.append(_section_box_row(f"From: {state_id}{merges}"))
        else:
            lines.append(_section_box_row(f"From: {state_id}"))

        if not targets:
            lines.append(_section_box_row("  └─ (No outgoing transitions)"))
            continue

        for i, (target, desc, required_keys) in enumerate(targets):
            if i == len(targets) - 1:
                prefix = "  └─"
                continuation = "     "
            else:
                prefix = "  ├─"
                continuation = "  │  "

            # Add special decorations for certain transition types
            if target == state_id:
                # Self-loop
                lines.append(
                    _section_box_row(
                        f"{prefix} To: {target} {ARROW_STYLES['self']} (Self loop)"
                    )
                )
            else:
                # Normal transition
                lines.append(_section_box_row(f"{prefix} To: {target}"))

            # Add transition description. Wrap to the width left after the
            # continuation indent and the "└─ Why: " lead-in, so ordinary text
            # never reaches `_section_box_row`'s truncation.
            if desc:
                _why_lead = f"{continuation}└─ Why: "
                # `if desc:` admits whitespace-only, which wraps to [] (D-009)
                wrapped_desc = textwrap.wrap(
                    desc, width=_SECTION_BOX_INNER_WIDTH - len(_why_lead)
                ) or [""]
                lines.append(_section_box_row(_why_lead + wrapped_desc[0]))
                for line in wrapped_desc[1:]:
                    lines.append(_section_box_row(f"{continuation}   {line}"))

            # Add required keys with key icon
            if required_keys:
                key_str = ", ".join(required_keys)
                _key_lead = f"{continuation}└─ {ICONS['key']} "
                wrapped_keys = textwrap.wrap(
                    f"Requires: {key_str}",
                    width=_SECTION_BOX_INNER_WIDTH - len(_key_lead),
                )
                for j, line in enumerate(wrapped_keys):
                    if j == 0:
                        lines.append(_section_box_row(_key_lead + line))
                    else:
                        lines.append(_section_box_row(f"{continuation}   {line}"))

    lines.append("└" + "─" * 60 + "┘")
    lines.append("")

    return lines


def build_legend(is_initial_terminal: bool, has_input_states: bool) -> list[str]:
    """Create a legend explaining the diagram symbols."""
    legend = [
        "Legend:",
        f"  {BOX_STYLES['initial']['topleft']}{BOX_STYLES['initial']['horizontal'] * 3}{BOX_STYLES['initial']['topright']}  - Initial state",
        f"  {BOX_STYLES['terminal']['topleft']}{BOX_STYLES['terminal']['horizontal'] * 3}{BOX_STYLES['terminal']['topright']}  - Terminal state",
    ]

    if is_initial_terminal:
        legend.append(
            f"  {BOX_STYLES['both']['topleft']}{BOX_STYLES['both']['horizontal'] * 3}{BOX_STYLES['both']['topright']}  - State that is both initial and terminal"
        )

    legend.extend(
        [
            f"  {ARROW_STYLES['forward']} {ARROW_STYLES['connector']}  - Forward transition",
            f"  {ARROW_STYLES['backward']} {ARROW_STYLES['connector']}  - Backward transition (loop)",
            f"  {ARROW_STYLES['self']}  - Self-loop",
        ]
    )

    if has_input_states:
        legend.extend(
            [
                f"  {ICONS['input']}  - State that collects user input",
                f"  {ICONS['key']}  - Required input data",
            ]
        )

    legend.extend(
        [
            f"  {ICONS['branching']}  - Branching state (multiple outgoing transitions)",
            f"  {ICONS['merge']}  - Merge state (multiple incoming transitions)",
        ]
    )

    return legend


def generate_enhanced_ascii_diagram(
    graph: dict[str, list],
    initial_state: str,
    terminal_states: set[str],
    states: dict[str, Any],
    state_metrics: dict[str, dict[str, Any]],
) -> list[str]:
    """
    Generate an enhanced ASCII diagram of the FSM.

    Args:
        graph: Dictionary mapping state_id to list of (target, desc, keys) tuples
        initial_state: ID of the initial state
        terminal_states: Set of terminal state IDs
        states: Dictionary containing state definitions
        state_metrics: Dictionary of metrics for each state

    Returns:
        list of strings representing the ASCII diagram
    """
    # Organize states in a logical order
    ordered_states = sort_states_logically(
        states, initial_state, terminal_states, state_metrics
    )

    # Create state boxes
    state_boxes = build_state_boxes(
        ordered_states, initial_state, terminal_states, states
    )

    # Generate the diagram layout
    diagram_lines = []

    # First, stack states vertically
    connections = []  # Store connection information for drawing arrows later

    for i, state_id in enumerate(ordered_states):
        box = state_boxes[state_id]

        # Extend diagram with current state box
        diagram_lines.extend(box)

        # Process transitions
        if state_id not in terminal_states:
            targets = graph.get(state_id, [])
            for target, desc, required_keys in targets:
                # Skip self-loops for now (we'll handle them separately)
                if target == state_id:
                    # Add self-loop arrow
                    self_loop_line = f"  {ARROW_STYLES['self']} Self loop: {desc}" + (
                        f" [{ICONS['key']} {', '.join(required_keys)}]"
                        if required_keys
                        else ""
                    )
                    diagram_lines.append(self_loop_line)
                    continue

                # Store connection info for drawing later (every target is a
                # state: `_require_valid_transition_targets`)
                target_idx = ordered_states.index(target)
                direction = "forward" if target_idx > i else "backward"
                connections.append((i, target_idx, direction, desc, required_keys))

            # Add spacing between states
            diagram_lines.append("")

    # Process connections to add transition arrows
    diagram_lines.append("")
    diagram_lines.append("Connections:")
    for from_idx, to_idx, direction, desc, required_keys in connections:
        from_state = ordered_states[from_idx]
        to_state = ordered_states[to_idx]

        arrow = (
            ARROW_STYLES["forward"]
            if direction == "forward"
            else ARROW_STYLES["backward"]
        )
        req_str = (
            f" [{ICONS['key']} {', '.join(required_keys)}]" if required_keys else ""
        )

        diagram_lines.append(
            f"  {from_state} {arrow} {ARROW_STYLES['connector']} {to_state}: {desc}{req_str}"
        )

    # Add loop detection information
    found_loops = detect_loops(graph, ordered_states)
    if found_loops:
        diagram_lines.append("")
        diagram_lines.append("Loops:")
        for loop in found_loops:
            if len(loop) == 2:  # Self-loop
                state_id, desc = loop
                desc_str = f" ({desc})" if desc else ""
                diagram_lines.append(
                    f"  * {state_id} {ARROW_STYLES['self']} {state_id}{desc_str}"
                )
            else:  # Loop between states
                from_state, to_state, desc = loop
                desc_str = f" ({desc})" if desc else ""
                diagram_lines.append(
                    f"  * {from_state} {ARROW_STYLES['bidirectional']} {to_state}{desc_str}"
                )

    return diagram_lines


def generate_compact_ascii_diagram(
    graph: dict[str, list],
    initial_state: str,
    terminal_states: set[str],
    states: dict[str, Any],
    state_metrics: dict[str, dict[str, Any]],
) -> list[str]:
    """Generate a more compact ASCII diagram with simpler state boxes."""
    # Organize states in a logical order
    ordered_states = sort_states_logically(
        states, initial_state, terminal_states, state_metrics
    )

    # Generate the diagram layout
    diagram_lines = []

    # Create a more compact representation
    for _i, state_id in enumerate(ordered_states):
        # Determine the box style based on state type
        prefix = _COMPACT_PREFIXES[
            _box_style_name(state_id == initial_state, state_id in terminal_states)
        ]

        # Add icons for state properties
        icons = _state_icons(state_metrics.get(state_id, {}))

        icon_str = " " + " ".join(icons) if icons else ""

        # Add the state line
        state_line = f"{prefix}{state_id}{icon_str}"

        # Add purpose if available, but shortened
        purpose = states.get(state_id, {}).get("purpose", "")
        if purpose:
            max_purpose_len = 80 - len(state_line) - 5  # Allow for some padding
            if max_purpose_len > 10:  # Only add if there's enough space
                purpose_short = textwrap.shorten(purpose, width=max_purpose_len)
                state_line += f" - {purpose_short}"

        diagram_lines.append(state_line)

        # Add outgoing transitions if not a terminal state
        if state_id not in terminal_states:
            targets = graph.get(state_id, [])
            for j, (target, desc, required_keys) in enumerate(targets):
                # Skip self-loops for compactness
                if target == state_id:
                    continue

                # Determine prefix based on position
                if j == len(targets) - 1:
                    t_prefix = "    └─"
                else:
                    t_prefix = "    ├─"

                # Add transition with shortened description
                arrow = ARROW_STYLES["connector"]
                if required_keys:
                    req_str = f" [{ICONS['key']} {', '.join(required_keys)}]"
                else:
                    req_str = ""

                if desc:
                    max_desc_len = 80 - len(t_prefix) - len(target) - len(req_str) - 5
                    if max_desc_len > 10:
                        desc_short = textwrap.shorten(desc, width=max_desc_len)
                        diagram_lines.append(
                            f"{t_prefix} {arrow} {target}: {desc_short}{req_str}"
                        )
                    else:
                        diagram_lines.append(f"{t_prefix} {arrow} {target}{req_str}")
                else:
                    diagram_lines.append(f"{t_prefix} {arrow} {target}{req_str}")

            # Add spacing between states
            diagram_lines.append("")

    return diagram_lines


def generate_minimal_ascii_diagram(
    graph: dict[str, list],
    initial_state: str,
    terminal_states: set[str],
    state_metrics: dict[str, dict[str, Any]],
) -> list[str]:
    """Generate a minimal ASCII diagram showing just the states and connections."""
    diagram_lines: list[str] = []

    # Sort states by depth for better layout
    ordered_states = sort_states_by_depth(initial_state, state_metrics)

    # Create a mapping of states to their line position in the diagram
    state_positions = {}

    # Draw the states as simple boxes with arrows
    for _i, state_id in enumerate(ordered_states):
        # Format the state label
        if state_id == initial_state:
            label = f"[{state_id}]* "  # Initial state indicator
        elif state_id in terminal_states:
            label = f"[{state_id}]# "  # Terminal state indicator
        else:
            label = f"[{state_id}] "

        # Add icons for required input
        if state_metrics.get(state_id, {}).get("required_keys"):
            label += f"{ICONS['input']} "

        state_positions[state_id] = len(diagram_lines)
        diagram_lines.append(label)

        # Add immediate transitions with simple arrows
        targets = graph.get(state_id, [])
        if not targets:
            continue

        for j, (target, _desc, required_keys) in enumerate(targets):
            # Skip self-loops and backward references for clarity
            if target == state_id or (
                target in state_positions
                and state_positions[target] < len(diagram_lines)
            ):
                continue

            prefix = "  ├→" if j < len(targets) - 1 else "  └→"
            target_label = f"{prefix} {target}"

            # Add minimal info about requirements
            if required_keys:
                target_label += f" {ICONS['key']}"

            diagram_lines.append(target_label)

    # Add a simple legend
    diagram_lines.append("")
    diagram_lines.append(
        "Legend: [State]* = Initial, [State]# = Terminal, "
        + f"{ICONS['input']} = User Input, {ICONS['key']} = Required Data"
    )

    return diagram_lines
