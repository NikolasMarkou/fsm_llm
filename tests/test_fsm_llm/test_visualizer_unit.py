"""
Unit tests for fsm_llm.visualizer module.

Tests cover:
- visualize_fsm_ascii: basic FSM, multi-state FSM, different styles
- generate_enhanced_ascii_diagram: diagram generation
- sort_states_logically: ordering guarantees
- create_state_boxes: box generation
- Output is always a non-empty string containing state names
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from fsm_llm.visualizer import (
    build_graph_representation,
    create_state_boxes,
    generate_enhanced_ascii_diagram,
    sort_states_logically,
    visualize_fsm_ascii,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _linear_fsm_data():
    """Simple linear FSM: start -> end."""
    return {
        "name": "LinearFSM",
        "description": "A two-state linear FSM",
        "initial_state": "start",
        "states": {
            "start": {
                "description": "Start state",
                "purpose": "Begin the conversation",
                "transitions": [{"target_state": "end", "description": "Finish"}],
            },
            "end": {
                "description": "End state",
                "purpose": "Terminate",
                "transitions": [],
            },
        },
    }


def _unsorted_terminal_fsm_data():
    """FSM whose terminal states are inserted in NON-alphabetical order.

    Insertion order (zulu, mike, alpha, delta, echo, bravo) deliberately differs from
    alphabetical order so an ordering assertion cannot pass vacuously. All six terminals
    sit at the same depth (1), so nothing but the terminal-tail ordering rule decides
    their relative position.
    """
    terminals = ["zulu", "mike", "alpha", "delta", "echo", "bravo"]
    states = {
        "init": {
            "id": "init",
            "description": "Initialization",
            "purpose": "Fan out",
            "transitions": [
                {"target_state": t, "description": f"Go to {t}"} for t in terminals
            ],
        }
    }
    for t in terminals:
        states[t] = {
            "id": t,
            "description": f"Terminal {t}",
            "purpose": "Finish",
            "transitions": [],
        }
    return {
        "name": "UnsortedTerminalFSM",
        "description": "FSM with several equal-depth terminal states",
        "initial_state": "init",
        "states": states,
    }


def _multi_state_fsm_data():
    """FSM with several states: init -> collect -> process -> done."""
    return {
        "name": "MultiFSM",
        "description": "Multi-state FSM for testing",
        "initial_state": "init",
        "states": {
            "init": {
                "description": "Initialization",
                "purpose": "Set up",
                "transitions": [
                    {"target_state": "collect", "description": "Collect data"}
                ],
            },
            "collect": {
                "description": "Data collection",
                "purpose": "Gather info",
                "required_context_keys": ["user_name"],
                "transitions": [
                    {"target_state": "process", "description": "Process data"}
                ],
            },
            "process": {
                "description": "Processing",
                "purpose": "Crunch numbers",
                "transitions": [
                    {"target_state": "done", "description": "Complete"},
                    {"target_state": "collect", "description": "Need more data"},
                ],
            },
            "done": {
                "description": "Finished",
                "purpose": "Show results",
                "transitions": [],
            },
        },
    }


# ==================================================================
# visualize_fsm_ascii
# ==================================================================


class TestVisualizeFsmAscii:
    def test_basic_fsm_returns_string(self):
        output = visualize_fsm_ascii(_linear_fsm_data())
        assert isinstance(output, str)
        assert len(output) > 0

    def test_output_contains_state_names(self):
        output = visualize_fsm_ascii(_linear_fsm_data())
        assert "start" in output
        assert "end" in output

    def test_output_contains_fsm_name(self):
        output = visualize_fsm_ascii(_linear_fsm_data())
        assert "LinearFSM" in output

    def test_multi_state_fsm(self):
        output = visualize_fsm_ascii(_multi_state_fsm_data())
        for state in ("init", "collect", "process", "done"):
            assert state in output

    def test_compact_style(self):
        output = visualize_fsm_ascii(_linear_fsm_data(), style="compact")
        assert isinstance(output, str)
        assert "start" in output

    def test_minimal_style(self):
        output = visualize_fsm_ascii(_linear_fsm_data(), style="minimal")
        assert isinstance(output, str)
        assert "start" in output


# ==================================================================
# generate_enhanced_ascii_diagram
# ==================================================================


class TestGenerateEnhancedAsciiDiagram:
    def test_returns_list_of_strings(self):
        data = _linear_fsm_data()
        graph, metrics = build_graph_representation(
            data["states"], data["initial_state"]
        )
        terminal = {"end"}

        lines = generate_enhanced_ascii_diagram(
            graph, "start", terminal, data["states"], metrics
        )
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)
        assert len(lines) > 0

    def test_diagram_contains_connection_info(self):
        data = _multi_state_fsm_data()
        graph, metrics = build_graph_representation(
            data["states"], data["initial_state"]
        )
        terminal = {"done"}

        lines = generate_enhanced_ascii_diagram(
            graph, "init", terminal, data["states"], metrics
        )
        text = "\n".join(lines)
        assert "Connections:" in text


# ==================================================================
# sort_states_logically
# ==================================================================


class TestSortStatesLogically:
    def test_initial_state_first(self):
        data = _multi_state_fsm_data()
        _graph, metrics = build_graph_representation(
            data["states"], data["initial_state"]
        )
        terminal = {"done"}

        ordered = sort_states_logically(data["states"], "init", terminal, metrics)
        assert ordered[0] == "init"

    def test_terminal_state_last(self):
        data = _multi_state_fsm_data()
        _graph, metrics = build_graph_representation(
            data["states"], data["initial_state"]
        )
        terminal = {"done"}

        ordered = sort_states_logically(data["states"], "init", terminal, metrics)
        assert ordered[-1] == "done"

    def test_all_states_present(self):
        data = _multi_state_fsm_data()
        _graph, metrics = build_graph_representation(
            data["states"], data["initial_state"]
        )
        terminal = {"done"}

        ordered = sort_states_logically(data["states"], "init", terminal, metrics)
        assert set(ordered) == set(data["states"].keys())


# ==================================================================
# create_state_boxes
# ==================================================================


class TestCreateStateBoxes:
    def test_returns_box_for_each_state(self):
        data = _linear_fsm_data()
        _graph, metrics = build_graph_representation(
            data["states"], data["initial_state"]
        )
        terminal = {"end"}
        ordered = sort_states_logically(data["states"], "start", terminal, metrics)

        boxes = create_state_boxes(ordered, "start", terminal, data["states"], metrics)
        assert "start" in boxes
        assert "end" in boxes
        # Each box is a list of strings
        for box_lines in boxes.values():
            assert isinstance(box_lines, list)
            assert len(box_lines) > 0

    def test_box_contains_state_id(self):
        data = _linear_fsm_data()
        _graph, metrics = build_graph_representation(
            data["states"], data["initial_state"]
        )
        terminal = {"end"}
        ordered = sort_states_logically(data["states"], "start", terminal, metrics)

        boxes = create_state_boxes(ordered, "start", terminal, data["states"], metrics)
        start_text = "\n".join(boxes["start"])
        assert "start" in start_text


# ------------------------------------------------------------------
# Regression: empty / None / whitespace-only text fields (S3)
# ------------------------------------------------------------------


def _reproducer_fsm_data():
    """The exact reproducer dict from findings/prompts-and-tooling.md #2.

    Deliberately a RAW dict: `visualize_fsm_ascii` is an exported public
    function whose input is never routed through the `FSMDefinition` pydantic
    model, so `description`/`purpose` carry no min_length guarantee here.
    """
    return {
        "name": "Test",
        "description": "",
        "initial_state": "s1",
        "states": {"s1": {"id": "s1", "purpose": "p", "transitions": []}},
    }


class TestVisualizeFSMAsciiEmptyTextFields:
    """`visualize_fsm_ascii` must never raise out of the public API.

    These call the exported `visualize_fsm_ascii` directly, NOT the CLI wrapper
    `visualize_fsm_from_file` -- the wrapper is shielded by its own
    try/except, so asserting through it would be verification theatre.

    `style="full"` is the default and the only style that renders the metadata
    section; compact/minimal skip it entirely.
    """

    def test_empty_description(self):
        out = visualize_fsm_ascii(_reproducer_fsm_data(), style="full")
        assert isinstance(out, str)
        assert out

    def test_none_description(self):
        # `.get(key, default)` returns None for an explicit null -- it does NOT
        # fall back to the default. This shape raised AttributeError, not
        # IndexError, so the guard must handle both.
        data = _reproducer_fsm_data()
        data["description"] = None
        out = visualize_fsm_ascii(data, style="full")
        assert isinstance(out, str)
        assert out

    def test_whitespace_only_description(self):
        # Truthy, so a `if description:` guard would not catch it, yet
        # textwrap.wrap("   ") == [].
        data = _reproducer_fsm_data()
        data["description"] = "   "
        out = visualize_fsm_ascii(data, style="full")
        assert isinstance(out, str)
        assert out

    def test_missing_description_key(self):
        data = _reproducer_fsm_data()
        del data["description"]
        out = visualize_fsm_ascii(data, style="full")
        assert isinstance(out, str)
        assert out

    def test_whitespace_only_state_purpose(self):
        # Sibling of the same defect class: create_states_section guards with
        # `if purpose:`, which whitespace-only passes, then indexes [0].
        data = _reproducer_fsm_data()
        data["states"]["s1"]["purpose"] = "   "
        out = visualize_fsm_ascii(data, style="full")
        assert isinstance(out, str)
        assert out

    def test_whitespace_only_transition_description(self):
        # Sibling of the same defect class in create_transitions_section.
        data = _reproducer_fsm_data()
        data["states"]["s2"] = {"id": "s2", "purpose": "q", "transitions": []}
        data["states"]["s1"]["transitions"] = [
            {"target_state": "s2", "description": "   "}
        ]
        out = visualize_fsm_ascii(data, style="full")
        assert isinstance(out, str)
        assert out

    def test_all_empty_text_fields_at_once(self):
        data = _reproducer_fsm_data()
        data["description"] = None
        data["states"]["s2"] = {"id": "s2", "purpose": "", "transitions": []}
        data["states"]["s1"]["purpose"] = "   "
        data["states"]["s1"]["transitions"] = [
            {"target_state": "s2", "description": "   "}
        ]
        out = visualize_fsm_ascii(data, style="full")
        assert isinstance(out, str)
        assert out

    def test_well_formed_fsm_still_renders_its_description(self):
        # Non-regression: the guard must not alter the happy path.
        data = _reproducer_fsm_data()
        data["description"] = "A real description"
        out = visualize_fsm_ascii(data, style="full")
        assert "A real description" in out


# ==================================================================
# PYTHONHASHSEED determinism (B7 / SC-03)
# ==================================================================

# Rendered in a SUBPROCESS on purpose: PYTHONHASHSEED is read once at interpreter
# startup, so set iteration order cannot be perturbed in-process. The probe imports the
# fixture helper from this very module rather than restating it, so the two can't drift.
_RENDER_PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from test_visualizer_unit import _unsorted_terminal_fsm_data
from fsm_llm.visualizer import visualize_fsm_ascii

data = _unsorted_terminal_fsm_data()
sys.stdout.write(
    json.dumps({s: visualize_fsm_ascii(data, style=s) for s in ("full", "compact")})
)
"""


def _render_under_hashseed(seed):
    """Render the unsorted-terminal FSM in a fresh interpreter at a given hash seed."""
    env = dict(os.environ, PYTHONHASHSEED=seed)
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_PROBE, str(Path(__file__).parent)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


class TestVisualizationIsHashSeedIndependent:
    """Regression: terminal states were ordered by set iteration order (unstable).

    Seeds 0 and 1 were empirically confirmed to produce DIFFERENT output for both
    affected styles before the `sorted()` fix, so this pair is pinned deliberately.
    `style="minimal"` is not covered: it routes through `sort_states_by_depth` and was
    never exposed.
    """

    def test_full_and_compact_are_byte_identical_across_hash_seeds(self):
        first = _render_under_hashseed("0")
        second = _render_under_hashseed("1")

        assert first["full"] == second["full"]
        assert first["compact"] == second["compact"]

    def test_terminal_states_are_emitted_in_sorted_order(self):
        data = _unsorted_terminal_fsm_data()
        _graph, metrics = build_graph_representation(data["states"], "init")
        terminal = {
            state_id
            for state_id, state in data["states"].items()
            if not state["transitions"]
        }

        ordered = sort_states_logically(data["states"], "init", terminal, metrics)

        assert ordered[0] == "init"
        assert ordered[1:] == sorted(terminal)
