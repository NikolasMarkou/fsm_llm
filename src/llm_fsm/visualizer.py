"""
Enhanced FSM ASCII Visualizer: A beautiful, feature-rich tool to visualize
Finite State Machines using fancy ASCII art.

No external dependencies required - works with standard Python.
"""

import re
import json
import argparse
import textwrap
from collections import defaultdict
from typing import Dict, Any, List, Set, Tuple, Optional

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger

# --------------------------------------------------------------

# ASCII drawing characters for different box styles
BOX_STYLES = {
    "default": {
        "topleft": "┌", "topright": "┐", "bottomleft": "└", "bottomright": "┘",
        "horizontal": "─", "vertical": "│", "title_sep": "─"
    },
    "initial": {
        "topleft": "╔", "topright": "╗", "bottomleft": "╚", "bottomright": "╝",
        "horizontal": "═", "vertical": "║", "title_sep": "═"
    },
    "terminal": {
        "topleft": "┏", "topright": "┓", "bottomleft": "┗", "bottomright": "┛",
        "horizontal": "━", "vertical": "┃", "title_sep": "━"
    },
    "both": {  # For states that are both initial and terminal
        "topleft": "╔", "topright": "╗", "bottomleft": "┗", "bottomright": "┛",
        "horizontal": "═", "vertical": "║", "title_sep": "═"
    },
    "section": {
        "topleft": "╭", "topright": "╮", "bottomleft": "╰", "bottomright": "╯",
        "horizontal": "─", "vertical": "│", "title_sep": "─"
    }
}

# Arrow styles for transitions
ARROW_STYLES = {
    "forward": "↓",
    "backward": "↑",
    "self": "⟲",
    "connector": "→",
    "bidirectional": "↔",
    "down_arrow": "▼",
    "right_arrow": "▶",
    "diamond": "◆"
}

# Icons for different state attributes
ICONS = {
    "input": "✎",      # States that require user input
    "branching": "⎇",   # States with multiple outbound transitions
    "merge": "⊕",       # States with multiple inbound transitions
    "key": "🔑",        # Used for required keys
    "note": "📝"       # For notes and observations
}

def visualize_fsm_ascii(fsm_data: Dict[str, Any], style: str = "full") -> str:
    """
    Generate an enhanced ASCII visualization of an FSM.

    Args:
        fsm_data: The FSM definition as a dictionary
        style: Visualization style - "full", "compact", or "minimal"

    Returns:
        A string containing the ASCII visualization
    """
    states = fsm_data.get("states", {})
    initial_state = fsm_data.get("initial_state", "")
    persona = fsm_data.get("persona", None)

    # Find terminal states (those with no outgoing transitions)
    terminal_states = {
        state_id for state_id, state in states.items()
        if not state.get("transitions", [])
    }

    # Build a representation of the graph structure and analyze it
    graph, state_metrics = build_graph_representation(states)

    # Create ASCII visualization
    lines = []

    # Create header based on style
    if style != "minimal":
        lines.extend(create_fancy_header(fsm_data.get("name", "FSM")))
    else:
        # Simple header for minimal style
        lines.append(f"FSM: {fsm_data.get('name', 'Unnamed FSM')}")
        lines.append("")

    # Add metadata based on style
    if style == "full":
        lines.extend(create_metadata_section(fsm_data, initial_state, state_metrics))

        # Add persona section if available
        if persona:
            lines.extend(create_persona_section(persona))

        # Add states section
        lines.extend(create_states_section(states, initial_state, terminal_states, state_metrics))

        # Add transitions section
        lines.extend(create_transitions_section(graph, states))
    elif style == "compact" and persona:
        # In compact mode, just show a simplified persona section
        lines.append("╭" + "─" * 60 + "╮")
        lines.append("│ " + "PERSONA: ".ljust(10) + textwrap.shorten(persona, width=48).ljust(50) + " │")
        lines.append("╰" + "─" * 60 + "╯")
        lines.append("")

    # Create diagram based on style
    try:
        if style != "minimal":
            lines.append("")
            lines.append("╭" + "─" * 60 + "╮")
            lines.append("│" + " STATE DIAGRAM ".center(60) + "│")
            lines.append("╰" + "─" * 60 + "╯")
            lines.append("")

            # Add legend
            lines.extend(create_legend(
                initial_state in terminal_states,
                any(len(state_metrics.get(s, {}).get("required_keys", [])) > 0 for s in states)
            ))

        lines.append("")

        # Generate diagram based on selected style
        if style == "minimal":
            diagram_lines = generate_minimal_ascii_diagram(
                graph, initial_state, terminal_states, states, state_metrics
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
    except Exception as e:
        lines.append(f"Could not generate diagram: {e}")

    return "\n".join(lines)

def create_fancy_header(name: str) -> List[str]:
    """Create a fancy header for the FSM visualization."""
    width = max(60, len(name) + 10)
    lines = [
        "╭" + "─" * width + "╮",
        "│" + f" {name} ".center(width) + "│",
        "│" + "FINITE STATE MACHINE VISUALIZATION".center(width) + "│",
        "╰" + "─" * width + "╯",
        ""
    ]
    return lines

def create_metadata_section(fsm_data: Dict[str, Any], initial_state: str, state_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    """Create the metadata section of the visualization."""
    lines = [
        "┌" + "─" * 60 + "┐",
        "│" + " METADATA ".center(60) + "│",
        "├" + "─" * 60 + "┤"
    ]

    # Format description with word wrapping
    description = fsm_data.get("description", "No description")
    wrapped_desc = textwrap.wrap(description, width=56)

    lines.append("│ Description: " + wrapped_desc[0].ljust(46) + " │")
    for line in wrapped_desc[1:]:
        lines.append("│ " + " "*12 + line.ljust(48) + " │")

    lines.append("│ Version: " + str(fsm_data.get("version", "N/A")).ljust(51) + " │")
    lines.append("│ Initial State: " + initial_state.ljust(45) + " │")
    lines.append("│ Total States: " + str(len(fsm_data.get("states", {}))).ljust(47) + " │")

    # Add some statistics
    terminal_count = sum(1 for m in state_metrics.values() if m.get("is_terminal", False))
    lines.append("│ Terminal States: " + str(terminal_count).ljust(43) + " │")

    input_states = sum(1 for m in state_metrics.values() if m.get("required_keys"))
    lines.append("│ States With User Input: " + str(input_states).ljust(38) + " │")

    branching_states = sum(1 for m in state_metrics.values() if m.get("outbound", 0) > 1)
    lines.append("│ Branching States: " + str(branching_states).ljust(43) + " │")

    max_path = find_longest_path(state_metrics)
    lines.append("│ Maximum Path Length: " + str(max_path).ljust(40) + " │")

    lines.append("└" + "─" * 60 + "┘")
    lines.append("")

    return lines

def create_persona_section(persona: str) -> List[str]:
    """Create the persona section of the visualization."""
    # Prepare a more decorative persona section
    lines = [
        "┌" + "─" * 60 + "┐",
        "│" + " PERSONA ".center(60) + "│",
        "├" + "─" * 60 + "┤"
    ]

    # Add some decorative elements based on persona content
    if "pirate" in persona.lower():
        lines.append("│        ⚓️  YARRR! Captain's Personality  ☠️         │")
    elif "story" in persona.lower() or "tale" in persona.lower():
        lines.append("│        📚  Storyteller's Personality  📜         │")
    elif "teacher" in persona.lower() or "professor" in persona.lower():
        lines.append("│        🎓  Educator's Personality  📝         │")
    elif "customer" in persona.lower() or "service" in persona.lower():
        lines.append("│        👩‍💼  Service Professional Personality  👨‍💼         │")
    else:
        lines.append("│        ✨  Conversation Personality  ✨         │")

    lines.append("│                                                        │")

    # Format persona with word wrapping
    wrapped_persona = textwrap.wrap(persona, width=56)
    for line in wrapped_persona:
        lines.append("│ " + line.ljust(58) + " │")

    lines.append("└" + "─" * 60 + "┘")
    lines.append("")

    return lines

def create_states_section(
    states: Dict[str, Any],
    initial_state: str,
    terminal_states: Set[str],
    state_metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Create the states section of the visualization."""
    lines = [
        "┌" + "─" * 60 + "┐",
        "│" + " STATES ".center(60) + "│",
        "├" + "─" * 60 + "┤"
    ]

    # Sort states logically: initial first, then others in order of depth, then terminals
    sorted_states = sort_states_logically(states, initial_state, terminal_states, state_metrics)

    for state_id in sorted_states:
        state = states[state_id]
        state_type = []
        if state_id == initial_state:
            state_type.append("INITIAL")
        if state_id in terminal_states:
            state_type.append("TERMINAL")

        state_type_str = f" ({', '.join(state_type)})" if state_type else ""
        metrics = state_metrics.get(state_id, {})

        # Create a mini box for each state with appropriate decoration
        if state_id == initial_state and state_id in terminal_states:
            box_style = "both"
        elif state_id == initial_state:
            box_style = "initial"
        elif state_id in terminal_states:
            box_style = "terminal"
        else:
            box_style = "default"

        style = BOX_STYLES[box_style]

        lines.append("│ " + style["topleft"] + style["horizontal"] * 56 + style["topright"] + " │")

        # State ID with type and metrics
        state_line = f"│ {state_id}{state_type_str}"

        # Add icons for state properties
        icons = []
        if metrics.get("required_keys"):
            icons.append(f"{ICONS['input']}")
        if metrics.get("outbound", 0) > 1:
            icons.append(f"{ICONS['branching']}")
        if metrics.get("inbound", 0) > 1:
            icons.append(f"{ICONS['merge']}")

        if icons:
            icon_str = " " + " ".join(icons)
            # Make sure we don't exceed the box width
            state_line = state_line.ljust(56 - len(icon_str)) + icon_str

        lines.append("│ " + style["vertical"] + state_line.ljust(56) + style["vertical"] + " │")

        # Add purpose with word wrapping
        purpose = state.get("purpose", "")
        if purpose:
            wrapped_purpose = textwrap.wrap(purpose, width=52)
            lines.append("│ " + style["vertical"] + " Purpose: " + wrapped_purpose[0].ljust(47) + style["vertical"] + " │")
            for line in wrapped_purpose[1:]:
                lines.append("│ " + style["vertical"] + " "*9 + line.ljust(47) + style["vertical"] + " │")

        # Add required context keys if any
        required_keys = state.get("required_context_keys", [])
        if required_keys:
            key_str = ", ".join(required_keys)
            lines.append("│ " + style["vertical"] + " "*2 + ICONS["key"] + " Required: " + key_str.ljust(43) + style["vertical"] + " │")

        lines.append("│ " + style["bottomleft"] + style["horizontal"] * 56 + style["bottomright"] + " │")

    lines.append("└" + "─" * 60 + "┘")
    lines.append("")

    return lines

def create_transitions_section(graph: Dict[str, List], states: Dict[str, Any]) -> List[str]:
    """Create the transitions section of the visualization."""
    lines = [
        "┌" + "─" * 60 + "┐",
        "│" + " TRANSITIONS ".center(60) + "│",
        "├" + "─" * 60 + "┤"
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
            lines.append("│ " + f"From: {state_id}{branches}".ljust(58) + " │")
        elif len(incoming_transitions[state_id]) > 1:
            # Merge state
            merges = f" (merges {len(incoming_transitions[state_id])} paths)"
            lines.append("│ " + f"From: {state_id}{merges}".ljust(58) + " │")
        else:
            lines.append("│ " + f"From: {state_id}".ljust(58) + " │")

        if not targets:
            lines.append("│ " + "  └─ (No outgoing transitions)".ljust(58) + " │")
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
                lines.append("│ " + f"{prefix} To: {target} {ARROW_STYLES['self']} (Self loop)".ljust(58) + " │")
            else:
                # Normal transition
                lines.append("│ " + f"{prefix} To: {target}".ljust(58) + " │")

            # Add transition description
            if desc:
                wrapped_desc = textwrap.wrap(desc, width=46)
                lines.append("│ " + f"{continuation}└─ Why: {wrapped_desc[0]}".ljust(58) + " │")
                for line in wrapped_desc[1:]:
                    lines.append("│ " + f"{continuation}   {line}".ljust(58) + " │")

            # Add required keys with key icon
            if required_keys:
                key_str = ", ".join(required_keys)
                wrapped_keys = textwrap.wrap(f"Requires: {key_str}", width=46)
                for j, line in enumerate(wrapped_keys):
                    if j == 0:
                        lines.append("│ " + f"{continuation}└─ {ICONS['key']} {line}".ljust(58) + " │")
                    else:
                        lines.append("│ " + f"{continuation}   {line}".ljust(58) + " │")

    lines.append("└" + "─" * 60 + "┘")
    lines.append("")

    return lines

def build_graph_representation(states: Dict[str, Any]) -> Tuple[Dict[str, List], Dict[str, Dict[str, Any]]]:
    """Build a representation of the graph structure and analyze state metrics."""
    graph = {}
    state_metrics = {}

    # First pass: build the basic graph
    for state_id, state in states.items():
        targets = []
        for transition in state.get("transitions", []):
            target = transition.get("target_state", "")
            desc = transition.get("description", "")
            # Extract required context keys if available
            required_keys = []
            if transition.get("conditions"):
                for condition in transition.get("conditions", []):
                    if condition.get("requires_context_keys"):
                        required_keys.extend(condition.get("requires_context_keys", []))

            # Add this transition to the targets list
            targets.append((target, desc, required_keys))

        graph[state_id] = targets

        # Initialize metrics
        state_metrics[state_id] = {
            "outbound": len(targets),
            "inbound": 0,
            "required_keys": state.get("required_context_keys", []),
            "is_terminal": len(targets) == 0,
            "depth": 0  # Will be calculated in the next pass
        }

    # Second pass: calculate inbound transitions and depths
    for state_id, targets in graph.items():
        for target, _, _ in targets:
            if target in state_metrics:
                state_metrics[target]["inbound"] += 1

    # Calculate depths (distance from initial state)
    calculate_depths(graph, state_metrics)

    return graph, state_metrics

def calculate_depths(graph: Dict[str, List], state_metrics: Dict[str, Dict[str, Any]], initial_state: str = None) -> None:
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
    visited = set([initial_state])
    state_metrics[initial_state]["depth"] = 0

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

def find_longest_path(state_metrics: Dict[str, Dict[str, Any]]) -> int:
    """Find the length of the longest path through the FSM."""
    if not state_metrics:
        return 0

    # The longest path is the maximum depth of any terminal state
    max_depth = 0
    for state_id, metrics in state_metrics.items():
        if metrics.get("is_terminal", False):
            max_depth = max(max_depth, metrics.get("depth", 0))

    # Add 1 to count the states rather than transitions
    return max_depth + 1

def sort_states_logically(
    states: Dict[str, Any],
    initial_state: str,
    terminal_states: Set[str],
    state_metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
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

    sorted_non_terminals = [s[0] for s in sorted(non_terminal_states, key=lambda x: x[1])]

    # Add sorted non-terminals
    for state_id in sorted_non_terminals:
        if state_id not in visited:
            sorted_states.append(state_id)
            visited.add(state_id)

    # End with terminal states
    terminal_states_list = list(terminal_states)
    # Don't add initial state again if it's also terminal
    if initial_state in terminal_states_list:
        terminal_states_list.remove(initial_state)

    sorted_states.extend(terminal_states_list)

    # Add any remaining states
    for state_id in states:
        if state_id not in visited:
            sorted_states.append(state_id)
            visited.add(state_id)

    return sorted_states

def create_legend(is_initial_terminal: bool, has_input_states: bool) -> List[str]:
    """Create a legend explaining the diagram symbols."""
    legend = [
        "Legend:",
        f"  {BOX_STYLES['initial']['topleft']}{BOX_STYLES['initial']['horizontal'] * 3}{BOX_STYLES['initial']['topright']}  - Initial state",
        f"  {BOX_STYLES['terminal']['topleft']}{BOX_STYLES['terminal']['horizontal'] * 3}{BOX_STYLES['terminal']['topright']}  - Terminal state",
    ]

    if is_initial_terminal:
        legend.append(f"  {BOX_STYLES['both']['topleft']}{BOX_STYLES['both']['horizontal'] * 3}{BOX_STYLES['both']['topright']}  - State that is both initial and terminal")

    legend.extend([
        f"  {ARROW_STYLES['forward']} {ARROW_STYLES['connector']}  - Forward transition",
        f"  {ARROW_STYLES['backward']} {ARROW_STYLES['connector']}  - Backward transition (loop)",
        f"  {ARROW_STYLES['self']}  - Self-loop"
    ])

    if has_input_states:
        legend.extend([
            f"  {ICONS['input']}  - State that collects user input",
            f"  {ICONS['key']}  - Required input data"
        ])

    legend.extend([
        f"  {ICONS['branching']}  - Branching state (multiple outgoing transitions)",
        f"  {ICONS['merge']}  - Merge state (multiple incoming transitions)"
    ])

    return legend

def generate_enhanced_ascii_diagram(
    graph: Dict[str, List],
    initial_state: str,
    terminal_states: Set[str],
    states: Dict[str, Any],
    state_metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    """
    Generate an enhanced ASCII diagram of the FSM.

    Args:
        graph: Dictionary mapping state_id to list of (target, desc, keys) tuples
        initial_state: ID of the initial state
        terminal_states: Set of terminal state IDs
        states: Dictionary containing state definitions
        state_metrics: Dictionary of metrics for each state

    Returns:
        List of strings representing the ASCII diagram
    """
    # Organize states in a logical order
    ordered_states = sort_states_logically(states, initial_state, terminal_states, state_metrics)

    # Create state boxes
    state_boxes = create_state_boxes(ordered_states, initial_state, terminal_states, states, state_metrics)

    # Generate the diagram layout
    diagram_lines = []

    # First, stack states vertically
    connections = []  # Store connection information for drawing arrows later

    for i, state_id in enumerate(ordered_states):
        box = state_boxes[state_id]
        box_height = len(box)

        # Extend diagram with current state box
        diagram_lines.extend(box)

        # Process transitions
        if state_id not in terminal_states:
            targets = graph.get(state_id, [])
            for target, desc, required_keys in targets:
                # Skip self-loops for now (we'll handle them separately)
                if target == state_id:
                    # Add self-loop arrow
                    self_loop_line = f"  {ARROW_STYLES['self']} Self loop: {desc}" + (f" [{ICONS['key']} {', '.join(required_keys)}]" if required_keys else "")
                    diagram_lines.append(self_loop_line)
                    continue

                # Store connection info for drawing later
                try:
                    target_idx = ordered_states.index(target)
                    if target_idx > i:  # Forward connection
                        direction = "forward"
                    else:  # Backward connection
                        direction = "backward"

                    connections.append((i, target_idx, direction, desc, required_keys))
                except ValueError:
                    # Target state not in ordered_states (shouldn't happen in valid FSMs)
                    pass

            # Add spacing between states
            diagram_lines.append("")

    # Process connections to add transition arrows
    diagram_lines.append("")
    diagram_lines.append("Connections:")
    for from_idx, to_idx, direction, desc, required_keys in connections:
        from_state = ordered_states[from_idx]
        to_state = ordered_states[to_idx]

        arrow = ARROW_STYLES['forward'] if direction == "forward" else ARROW_STYLES['backward']
        req_str = f" [{ICONS['key']} {', '.join(required_keys)}]" if required_keys else ""

        diagram_lines.append(f"  {from_state} {arrow} {ARROW_STYLES['connector']} {to_state}: {desc}{req_str}")

    # Add loop detection information
    found_loops = detect_loops(graph, ordered_states)
    if found_loops:
        diagram_lines.append("")
        diagram_lines.append("Loops:")
        for loop in found_loops:
            if len(loop) == 2:  # Self-loop
                state_id, desc = loop
                desc_str = f" ({desc})" if desc else ""
                diagram_lines.append(f"  * {state_id} {ARROW_STYLES['self']} {state_id}{desc_str}")
            else:  # Loop between states
                from_state, to_state, desc = loop
                desc_str = f" ({desc})" if desc else ""
                diagram_lines.append(f"  * {from_state} {ARROW_STYLES['bidirectional']} {to_state}{desc_str}")

    return diagram_lines

def generate_compact_ascii_diagram(
    graph: Dict[str, List],
    initial_state: str,
    terminal_states: Set[str],
    states: Dict[str, Any],
    state_metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Generate a more compact ASCII diagram with simpler state boxes."""
    # Organize states in a logical order
    ordered_states = sort_states_logically(states, initial_state, terminal_states, state_metrics)

    # Generate the diagram layout
    diagram_lines = []

    # Create a more compact representation
    for i, state_id in enumerate(ordered_states):
        # Determine the box style based on state type
        if state_id == initial_state and state_id in terminal_states:
            prefix = "╔═╤═╝ "
        elif state_id == initial_state:
            prefix = "╔═══> "
        elif state_id in terminal_states:
            prefix = "┗━━━> "
        else:
            prefix = "────> "

        # Add icons for state properties
        icons = []
        metrics = state_metrics.get(state_id, {})
        if metrics.get("required_keys"):
            icons.append(f"{ICONS['input']}")
        if metrics.get("outbound", 0) > 1:
            icons.append(f"{ICONS['branching']}")
        if metrics.get("inbound", 0) > 1:
            icons.append(f"{ICONS['merge']}")

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
                arrow = ARROW_STYLES['connector']
                if required_keys:
                    req_str = f" [{ICONS['key']} {', '.join(required_keys)}]"
                else:
                    req_str = ""

                if desc:
                    max_desc_len = 80 - len(t_prefix) - len(target) - len(req_str) - 5
                    if max_desc_len > 10:
                        desc_short = textwrap.shorten(desc, width=max_desc_len)
                        diagram_lines.append(f"{t_prefix} {arrow} {target}: {desc_short}{req_str}")
                    else:
                        diagram_lines.append(f"{t_prefix} {arrow} {target}{req_str}")
                else:
                    diagram_lines.append(f"{t_prefix} {arrow} {target}{req_str}")

            # Add spacing between states
            diagram_lines.append("")

    return diagram_lines

def generate_minimal_ascii_diagram(
    graph: Dict[str, List],
    initial_state: str,
    terminal_states: Set[str],
    states: Dict[str, Any],
    state_metrics: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Generate a minimal ASCII diagram showing just the states and connections."""
    diagram_lines = []

    # Sort states by depth for better layout
    ordered_states = sort_states_by_depth(initial_state, graph, state_metrics)

    # Create a mapping of states to their line position in the diagram
    state_positions = {}

    # Draw the states as simple boxes with arrows
    for i, state_id in enumerate(ordered_states):
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

        for j, (target, desc, required_keys) in enumerate(targets):
            # Skip self-loops and backward references for clarity
            if target == state_id or (target in state_positions and state_positions[target] < len(diagram_lines)):
                continue

            prefix = "  ├→" if j < len(targets) - 1 else "  └→"
            target_label = f"{prefix} {target}"

            # Add minimal info about requirements
            if required_keys:
                target_label += f" {ICONS['key']}"

            diagram_lines.append(target_label)

    # Add a simple legend
    diagram_lines.append("")
    diagram_lines.append("Legend: [State]* = Initial, [State]# = Terminal, " +
                         f"{ICONS['input']} = User Input, {ICONS['key']} = Required Data")

    return diagram_lines

def sort_states_by_depth(initial_state: str, graph: Dict[str, List], state_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
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

def create_state_boxes(
    ordered_states: List[str],
    initial_state: str,
    terminal_states: Set[str],
    states: Dict[str, Any],
    state_metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """Create enhanced boxes for each state."""
    state_boxes = {}

    for state_id in ordered_states:
        is_initial = state_id == initial_state
        is_terminal = state_id in terminal_states
        metrics = state_metrics.get(state_id, {})

        # Choose appropriate box style
        if is_initial and is_terminal:
            style = BOX_STYLES["both"]
        elif is_initial:
            style = BOX_STYLES["initial"]
        elif is_terminal:
            style = BOX_STYLES["terminal"]
        else:
            style = BOX_STYLES["default"]

        # Get state details
        state_data = states.get(state_id, {})
        description = state_data.get("description", "")
        purpose = state_data.get("purpose", "")

        # Determine box width based on content
        content_width = max(
            len(state_id) + 8,  # Allow for padding and state type
            len(description) + 4 if description else 0,
            len(purpose) + 10 if purpose else 0,
            40  # Minimum width for readability
        )

        # Limit width for readability
        box_width = min(content_width, 60)

        # Create the box
        box = []

        # State type label
        state_type = []
        if is_initial:
            state_type.append("INITIAL")
        if is_terminal:
            state_type.append("TERMINAL")
        state_type_str = f" ({', '.join(state_type)})" if state_type else ""

        # Box header
        box.append(f"{style['topleft']}{style['horizontal'] * (box_width - 2)}{style['topright']}")

        # State ID with state type
        box.append(f"{style['vertical']} {state_id}{state_type_str}".ljust(box_width - 1) + f"{style['vertical']}")

        # Add state description
        box.append(f"{style['vertical']}{style['title_sep'] * (box_width - 2)}{style['vertical']}")
        if description:
            wrapped_desc = textwrap.wrap(description, width=box_width - 4)
            for line in wrapped_desc:
                box.append(f"{style['vertical']} {line}".ljust(box_width - 1) + f"{style['vertical']}")

        # Add empty line for separation
        box.append(f"{style['vertical']}".ljust(box_width - 1) + f"{style['vertical']}")

        # Add purpose
        if purpose:
            wrapped_purpose = textwrap.wrap(f"Purpose: {purpose}", width=box_width - 4)
            for line in wrapped_purpose:
                box.append(f"{style['vertical']} {line}".ljust(box_width - 1) + f"{style['vertical']}")

        # Add required context keys
        required_keys = state_data.get("required_context_keys", [])
        if required_keys:
            box.append(f"{style['vertical']}".ljust(box_width - 1) + f"{style['vertical']}")
            key_str = ", ".join(required_keys)
            wrapped_keys = textwrap.wrap(f"Requires: {key_str}", width=box_width - 4)
            for line in wrapped_keys:
                box.append(f"{style['vertical']} {line}".ljust(box_width - 1) + f"{style['vertical']}")

        # Box footer
        box.append(f"{style['bottomleft']}{style['horizontal'] * (box_width - 2)}{style['bottomright']}")

        state_boxes[state_id] = box

    return state_boxes

def detect_loops(graph: Dict[str, List], ordered_states: List[str]) -> List:
    """Find all loops in the FSM using DFS."""
    found_loops = []

    # Detect self-loops
    for state_id, targets in graph.items():
        for target, desc, _ in targets:
            if target == state_id:
                found_loops.append((state_id, desc))

    # Detect loops between states
    for i, state_id in enumerate(ordered_states):
        for target, desc, _ in graph.get(state_id, []):
            if target != state_id:  # Skip self-loops (already handled)
                try:
                    target_idx = ordered_states.index(target)
                    if target_idx < i:
                        # Found a backward loop
                        found_loops.append((state_id, target, desc))
                except ValueError:
                    pass

    return found_loops

def visualize_fsm_from_file(json_file: str, style: str = "full") -> str:
    """
    Visualize an FSM definition from a JSON file.

    Args:
        json_file: Path to the JSON file containing the FSM definition
        style: Visualization style - "full", "compact", or "minimal"

    Returns:
        A string containing the ASCII visualization
    """
    try:
        with open(json_file, 'r') as f:
            fsm_data = json.load(f)

        return visualize_fsm_ascii(fsm_data, style)
    except FileNotFoundError:
        return f"Error: File '{json_file}' not found."
    except json.JSONDecodeError:
        return f"Error: '{json_file}' is not a valid JSON file."
    except Exception as e:
        return f"Error: {e}"

def main_cli():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Visualize LLM-FSM definitions using enhanced ASCII art")
    parser.add_argument("--fsm", "-f", type=str, required=True, help="Path to FSM definition JSON file")
    parser.add_argument("--output", "-o", help="Output file (default: print to console)")
    parser.add_argument("--style", "-s", default="full", choices=["full", "compact", "minimal"],
                        help="Visualization style (default: full)")

    args = parser.parse_args()

    ascii_diagram = visualize_fsm_from_file(args.fsm, args.style)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(ascii_diagram)
        logger.info(f"ASCII diagram saved to {args.output}")
    else:
        print(ascii_diagram)

if __name__ == "__main__":
    main_cli()