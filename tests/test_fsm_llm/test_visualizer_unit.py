"""
Unit tests for fsm_llm.visualizer module.

Tests cover:
- visualize_fsm_ascii: basic FSM, multi-state FSM, different styles
- generate_enhanced_ascii_diagram: diagram generation
- sort_states_logically: ordering guarantees
- create_state_boxes: box generation
- Output is always a non-empty string containing state names
"""

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
