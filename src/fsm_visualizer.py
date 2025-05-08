"""
FSM ASCII Visualizer: A lightweight, dependency-free tool to visualize
Finite State Machines using ASCII art.

No external dependencies required - works with standard Python.
"""

import json
import argparse
from typing import Dict, Any, List, Set, Tuple

from .logging import logger

def visualize_fsm_ascii(fsm_data: Dict[str, Any]) -> str:
    """
    Generate an ASCII visualization of an FSM.

    Args:
        fsm_data: The FSM definition as a dictionary

    Returns:
        A string containing the ASCII visualization
    """
    states = fsm_data.get("states", {})
    initial_state = fsm_data.get("initial_state", "")

    # Find terminal states (those with no outgoing transitions)
    terminal_states = {
        state_id for state_id, state in states.items()
        if not state.get("transitions", [])
    }

    # Build a representation of the graph structure
    graph = {}
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

    # Create ASCII visualization
    lines = []

    # Add title and info
    lines.append(f"=== {fsm_data.get('name', 'FSM')} ===")
    lines.append(f"Description: {fsm_data.get('description', '')}")
    lines.append(f"Initial State: {initial_state}")
    lines.append("")

    # First, draw a state list
    lines.append("STATES:")
    lines.append("-------")
    for state_id, state in states.items():
        state_type = []
        if state_id == initial_state:
            state_type.append("INITIAL")
        if state_id in terminal_states:
            state_type.append("TERMINAL")

        state_type_str = f" ({', '.join(state_type)})" if state_type else ""
        lines.append(f"* {state_id}{state_type_str}: {state.get('description', '')}")
    lines.append("")

    # Draw transitions
    lines.append("TRANSITIONS:")
    lines.append("-----------")
    for state_id, targets in graph.items():
        if not targets:
            lines.append(f"{state_id} → (no transitions)")
            continue

        for target, desc, required_keys in targets:
            keys_str = f" [Requires: {', '.join(required_keys)}]" if required_keys else ""
            desc_str = f" ({desc})" if desc else ""
            lines.append(f"{state_id} → {target}{desc_str}{keys_str}")
    lines.append("")

    # Draw a simple state diagram
    try:
        lines.append("STATE DIAGRAM:")
        lines.append("-------------")
        lines.extend(generate_ascii_diagram(graph, initial_state, terminal_states))
    except Exception as e:
        lines.append(f"Could not generate diagram: {e}")

    return "\n".join(lines)


def generate_ascii_diagram(graph: Dict[str, List], initial_state: str,
                          terminal_states: Set[str]) -> List[str]:
    """
    Generate a simple ASCII diagram of the FSM.

    Args:
        graph: Dictionary mapping state_id to list of (target, desc, keys) tuples
        initial_state: ID of the initial state
        terminal_states: Set of terminal state IDs

    Returns:
        List of strings representing the ASCII diagram
    """
    lines = []

    # Create a list of all states in a logical order for display
    # Start with initial state, then follow transitions
    ordered_states = []
    visited = set()

    def dfs(state_id):
        if state_id in visited:
            return
        visited.add(state_id)
        ordered_states.append(state_id)
        for target, _, _ in graph.get(state_id, []):
            dfs(target)

    dfs(initial_state)

    # Add any remaining states (for completeness)
    for state_id in graph:
        if state_id not in visited:
            ordered_states.append(state_id)
            visited.add(state_id)

    # Draw a simple box for each state
    state_boxes = {}
    for state_id in ordered_states:
        is_initial = state_id == initial_state
        is_terminal = state_id in terminal_states

        # Create a box for this state
        box_width = max(len(state_id) + 4, 12)

        # Top line with special marker for initial state
        if is_initial:
            box = ["┌" + "─" * (box_width - 2) + "┐ ← Initial"]
        else:
            box = ["┌" + "─" * (box_width - 2) + "┐"]

        # State ID centered in the box
        padding = (box_width - 2 - len(state_id)) // 2
        box.append("│" + " " * padding + state_id + " " * (box_width - 2 - len(state_id) - padding) + "│")

        # Bottom line with special marker for terminal state
        if is_terminal:
            box.append("└" + "─" * (box_width - 2) + "┘ (Terminal)")
        else:
            box.append("└" + "─" * (box_width - 2) + "┘")

        state_boxes[state_id] = box

    # Detect loops and special structures
    found_loops = []
    for state_id, targets in graph.items():
        for target, desc, _ in targets:
            # Self-loop
            if target == state_id:
                found_loops.append((state_id, desc))
            # Loop back to a previous state (excluding direct predecessors)
            else:
                idx1 = ordered_states.index(state_id) if state_id in ordered_states else -1
                idx2 = ordered_states.index(target) if target in ordered_states else -1
                if idx1 > idx2 and idx1 >= 0 and idx2 >= 0:
                    found_loops.append((target, state_id, desc))

    # Create the diagram layout
    # First, just stack states vertically for a simple view
    for i, state_id in enumerate(ordered_states):
        box = state_boxes[state_id]
        lines.extend(box)

        # Add transitions if not a terminal state
        if state_id not in terminal_states:
            targets = graph.get(state_id, [])
            for target, desc, required_keys in targets:
                # Skip self-loops for now
                if target == state_id:
                    continue

                # Add an arrow to each target
                arrow = "↓ " if target in ordered_states[i+1:] else "↑ "
                desc_str = f"({desc})" if desc else ""
                req_str = f"[{', '.join(required_keys)}]" if required_keys else ""
                lines.append(f"{arrow} → {target} {desc_str} {req_str}")

            # Add spacing between states
            lines.append("")

    # Add information about loops
    if found_loops:
        lines.append("\nLOOPS:")
        for loop in found_loops:
            if len(loop) == 2:  # Self-loop
                state_id, desc = loop
                desc_str = f" ({desc})" if desc else ""
                lines.append(f"* {state_id} → {state_id}{desc_str} (Self-loop)")
            else:  # Loop between states
                from_state, to_state, desc = loop
                desc_str = f" ({desc})" if desc else ""
                lines.append(f"* {from_state} ↔ {to_state}{desc_str} (Loop)")

    return lines


def visualize_fsm_from_file(json_file: str) -> str:
    """
    Visualize an FSM definition from a JSON file.

    Args:
        json_file: Path to the JSON file containing the FSM definition

    Returns:
        A string containing the ASCII visualization
    """
    try:
        with open(json_file, 'r') as f:
            fsm_data = json.load(f)

        return visualize_fsm_ascii(fsm_data)
    except FileNotFoundError:
        return f"Error: File '{json_file}' not found."
    except json.JSONDecodeError:
        return f"Error: '{json_file}' is not a valid JSON file."
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LLM-FSM definitions using ASCII art")
    parser.add_argument("json_file", help="Path to the JSON file containing the FSM definition")
    parser.add_argument("--output", "-o", help="Output file (default: print to console)")

    args = parser.parse_args()

    ascii_diagram = visualize_fsm_from_file(args.json_file)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(ascii_diagram)
        logger.info(f"ASCII diagram saved to {args.output}")
    else:
        logger.info(ascii_diagram)

# Example usage:
# python fsm_ascii_visualizer.py my_fsm.json
# python fsm_ascii_visualizer.py my_fsm.json --output fsm_diagram.txt