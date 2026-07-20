"""
Unit tests for fsm_llm.visualizer module.

Tests cover:
- visualize_fsm_ascii: basic FSM, multi-state FSM, different styles
- generate_enhanced_ascii_diagram: diagram generation
- sort_states_logically: ordering guarantees
- create_state_boxes: box generation
- Output is always a non-empty string containing state names
"""

import io
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

from fsm_llm.visualizer import (
    build_graph_representation,
    create_state_boxes,
    generate_enhanced_ascii_diagram,
    main_cli,
    sort_states_logically,
    visualize_fsm_ascii,
    visualize_fsm_from_file,
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


# ------------------------------------------------------------------
# F-04 / SC-16: the CLI must actually PRINT the diagram
# ------------------------------------------------------------------


@contextmanager
def _cli_capture():
    """Run a console-script body from the library's real default state.

    `logging.py` calls `logger.disable("fsm_llm")` at import; this restores
    that state first, so the capture measures whether `main()` opts BACK IN.
    A sink is attached rather than using capsys because loguru binds
    `sys.stderr` at handler-add time and never sees pytest's replacement.
    """
    buffer = io.StringIO()
    logger.disable("fsm_llm")
    sink_id = logger.add(buffer, format="{message}", level="DEBUG")
    try:
        yield buffer
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")


class TestVisualizeCliEmitsOutput:
    """F-04. `fsm-llm-visualize` exists to print a diagram and printed nothing
    at all -- on success AND on failure -- while still exiting 0/1 "correctly".
    Asserting the exit code alone is satisfied by that strictly worse outcome,
    so these assert that the DIAGRAM itself reaches the user.
    """

    def test_diagram_actually_reaches_the_user(self, tmp_path):
        path = tmp_path / "linear.json"
        path.write_text(json.dumps(_linear_fsm_data()), encoding="utf-8")

        with _cli_capture() as buffer:
            with patch.object(sys, "argv", ["fsm-llm-visualize", "--fsm", str(path)]):
                with pytest.raises(SystemExit) as exc_info:
                    main_cli()

        output = buffer.getvalue()
        assert exc_info.value.code == 0, "exit code must be unchanged by the fix"
        # Not merely non-empty: the FSM's own name and both state ids must be
        # present, so a stub that logged "done" would not satisfy this.
        assert "LinearFSM" in output
        assert "start" in output and "end" in output
        assert "─" in output, f"no box drawing in the output: {output!r}"

    def test_missing_file_names_the_missing_path(self, tmp_path):
        missing = tmp_path / "definitely_absent.json"

        with _cli_capture() as buffer:
            with patch.object(
                sys, "argv", ["fsm-llm-visualize", "--fsm", str(missing)]
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main_cli()

        output = buffer.getvalue()
        assert exc_info.value.code == 1
        assert str(missing) in output, (
            f"the missing path must be named, got: {output!r}"
        )
        assert "not found" in output


# ------------------------------------------------------------------
# F-17 / SC-18: the STATES section must render an intact box
# ------------------------------------------------------------------

_SECTION_END = "└" + "─" * 60 + "┘"


def _states_section_lines(output):
    """Return the STATES section's lines, header and footer included.

    Args:
        output: a full `style="full"` render.

    Returns:
        The contiguous block from the "STATES" title row through the
        section's closing border.
    """
    lines = output.split("\n")
    start = next(i for i, line in enumerate(lines) if " STATES " in line)
    end = next(i for i, line in enumerate(lines[start:], start) if line == _SECTION_END)
    return lines[start : end + 1]


# Glyphs that OPEN and CLOSE a box at the start of a line. Nested per-state boxes
# inside the STATES section start with the section's own "│", so they are never
# mistaken for a top-level box here.
_BOX_OPENERS = ("┌", "╭", "┏", "╔")
_BOX_CLOSERS = ("└", "╰", "┗", "╚")
# Horizontal run glyphs that may appear BETWEEN a border line's two corners.
_BOX_HORIZONTALS = set("─═━")
# Width of every top-level SECTION box: "┌" + 60 * "─" + "┐".
_SECTION_WIDTH = 62


def _is_border_line(line, corners):
    """True if `line` is a pure box border: a corner, a horizontal run, a corner.

    The flow diagram draws arrows with the SAME glyphs that open boxes
    (`╔═══> state`, `┗━━━> tail`), so "starts with ╔" is not sufficient — a
    border line must contain nothing but box-drawing characters.
    """
    return (
        len(line) >= 2
        and line.startswith(corners)
        and set(line[1:-1]) <= _BOX_HORIZONTALS
        and not line[-1].isalnum()
    )


def _bordered_boxes(output):
    """Split a render into every complete top-level box it contains.

    Unlike `_states_section_lines`, this does NOT privilege one section — it
    returns every box in the document so an alignment claim can be checked
    against the whole render rather than against the one box a fix touched.

    Args:
        output: a full render at any style.

    Returns:
        List of boxes, each a list of lines from its opening border through its
        closing border inclusive.
    """
    boxes = []
    current = None
    for line in output.split("\n"):
        if current is None:
            if _is_border_line(line, _BOX_OPENERS):
                current = [line]
        else:
            current.append(line)
            if _is_border_line(line, _BOX_CLOSERS):
                boxes.append(current)
                current = None
    return boxes


def _long_id_fsm_data():
    """FSM whose ids, purpose and required keys all overflow the box."""
    long_id = "overflowing_state_identifier_" + "x" * 71  # exactly 100 chars
    assert len(long_id) == 100
    return long_id, {
        "name": "OverflowFSM",
        "initial_state": long_id,
        "states": {
            long_id: {
                "id": long_id,
                "description": "d " * 60,
                "purpose": "p " * 80,
                "required_context_keys": ["k" * 90, "another_very_long_key" * 3],
                "transitions": [{"target_state": "tail", "description": "go"}],
            },
            "tail": {"id": "tail", "description": "end", "transitions": []},
        },
    }


class TestStatesSectionBoxIntegrity:
    """F-17. `create_states_section` baked a literal "│ " into the state-id line
    and then wrapped that line in the box border AGAIN, so every state box in the
    default `full` style rendered a doubled glyph ("║│ greeting"). It also never
    truncated its content, unlike `create_state_boxes`, so a long id, purpose or
    required-keys list pushed the right border out of alignment.
    """

    def test_no_doubled_border_glyph_in_any_full_render(self):
        output = visualize_fsm_ascii(_multi_state_fsm_data(), style="full")

        for doubled in ("║│", "┃│", "││"):
            assert doubled not in output, (
                f"doubled border glyph {doubled!r} in the render:\n{output}"
            )

    def test_the_render_actually_contains_the_boxes_being_asserted_on(self):
        """Vacuity guard: the glyph assertions above are trivially satisfied by an
        empty render, so pin that all three box styles are genuinely present."""
        output = visualize_fsm_ascii(_multi_state_fsm_data(), style="full")
        section = "\n".join(_states_section_lines(output))

        assert "║" in section, "no INITIAL (double-line) box was rendered"
        assert "┃" in section, "no TERMINAL (heavy-line) box was rendered"
        assert "│ │" in section, "no default (light-line) box was rendered"

    def test_state_id_and_type_survive_the_truncation(self):
        """Over-correction guard. A fix that truncated the content to nothing, or
        that dropped the leading pad, would satisfy the glyph test above."""
        output = visualize_fsm_ascii(_multi_state_fsm_data(), style="full")
        section = _states_section_lines(output)

        assert any("║ init (INITIAL)" in line for line in section), (
            f"the initial state's own header row is gone:\n{chr(10).join(section)}"
        )

    def test_every_states_section_line_is_the_same_width(self):
        output = visualize_fsm_ascii(_multi_state_fsm_data(), style="full")
        section = _states_section_lines(output)

        widths = {len(line) for line in section}
        assert widths == {62}, (
            f"ragged STATES section, widths={sorted(widths)}:\n"
            + "\n".join(f"{len(line):>3} {line}" for line in section)
        )

    def test_a_100_char_state_id_does_not_break_alignment(self):
        long_id, data = _long_id_fsm_data()
        output = visualize_fsm_ascii(data, style="full")
        section = _states_section_lines(output)

        widths = {len(line) for line in section}
        assert widths == {62}, (
            f"a {len(long_id)}-char id made the box ragged, widths={sorted(widths)}:\n"
            + "\n".join(f"{len(line):>3} {line}" for line in section)
        )


class TestWholeRenderBoxAlignment:
    """SC-18 says "a 100-char state id does not break box alignment" — about the
    RENDER, not about one section. The original pinning test measured only the
    STATES section (the box step 14 touched) via `_states_section_lines`, so the
    suite stayed green while `create_metadata_section` emitted a 119-char row,
    `create_transitions_section` a 127-char row, and `create_persona_section` a
    stray 58-char spacer, all against the same 62-char border. These tests
    measure EVERY bordered row of the WHOLE document instead, so a fix applied to
    one section builder and not its twins cannot pass.
    """

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_every_box_in_the_render_is_internally_uniform(self, style):
        _, data = _long_id_fsm_data()
        output = visualize_fsm_ascii(data, style=style)

        for box in _bordered_boxes(output):
            widths = {len(line) for line in box}
            assert len(widths) == 1, (
                f"ragged box in style={style!r}, widths={sorted(widths)}:\n"
                + "\n".join(f"{len(line):>4} {line}" for line in box)
            )

    @pytest.mark.parametrize("style", ["full", "compact"])
    def test_every_section_box_is_exactly_the_border_width(self, style):
        """Per-state mini boxes are sized to their own content, so uniformity
        alone would be satisfied by a section box that is uniformly WRONG. Pin
        the top-level section boxes to the 62-char border specifically."""
        _, data = _long_id_fsm_data()
        output = visualize_fsm_ascii(data, style=style)

        section_boxes = [
            box for box in _bordered_boxes(output) if len(box[0]) == _SECTION_WIDTH
        ]
        assert section_boxes, f"no section box found in style={style!r}"

        for box in section_boxes:
            for line in box:
                assert len(line) == _SECTION_WIDTH, (
                    f"row is {len(line)} chars against a {_SECTION_WIDTH}-char "
                    f"border in style={style!r}:\n{line!r}\nfull box:\n"
                    + "\n".join(f"{len(x):>4} {x}" for x in box)
                )

    def test_the_probe_actually_sees_the_sections_that_were_broken(self):
        """Vacuity guard. The two tests above are trivially satisfied if
        `_bordered_boxes` returns nothing, or returns only the STATES box that
        was already correct. Pin that METADATA, PERSONA and TRANSITIONS — the
        three sections that were ragged — are genuinely among the boxes measured.
        """
        _, data = _long_id_fsm_data()
        data["persona"] = "A persona, whose section carried the 58-char spacer."
        output = visualize_fsm_ascii(data, style="full")

        boxes = _bordered_boxes(output)
        measured = "\n".join("\n".join(box) for box in boxes)

        for title in (" METADATA ", " PERSONA ", " STATES ", " TRANSITIONS "):
            assert title in measured, (
                f"{title!r} is not inside any box returned by `_bordered_boxes`, "
                "so the whole-render assertions never look at it"
            )

    def test_a_long_id_is_truncated_rather_than_dropped(self):
        """Over-correction guard: padding every row to 62 by emitting an empty
        row would pass the alignment tests. The id must still be legible."""
        long_id, data = _long_id_fsm_data()
        output = visualize_fsm_ascii(data, style="full")

        assert long_id[:40] in output, (
            "the long state id vanished from the render entirely"
        )

    def test_an_overflowing_required_keys_list_does_not_break_alignment(self):
        """The required-keys row was `key_str.ljust(43)`, which never shortens --
        the same defect as the id row and reachable without any long id."""
        data = {
            "name": "KeysFSM",
            "initial_state": "only",
            "states": {
                "only": {
                    "id": "only",
                    "description": "d",
                    "required_context_keys": ["k" * 200],
                    "transitions": [],
                }
            },
        }
        section = _states_section_lines(visualize_fsm_ascii(data, style="full"))

        assert {len(line) for line in section} == {62}, "\n".join(
            f"{len(line):>3} {line}" for line in section
        )


# ------------------------------------------------------------------
# F-18 / SC-19: a missing `initial_state` must say so
# ------------------------------------------------------------------


class TestMissingInitialStateIsDiagnosable:
    """F-18. An `initial_state` absent from `states` surfaced as a bare
    `KeyError('start')`, which `visualize_fsm_from_file`'s broad `except Exception`
    rendered as the useless `"Error: 'start'"` -- a single-quoted key name with no
    hint of which field was wrong.
    """

    def test_empty_states_message_names_both_fields(self, tmp_path):
        path = tmp_path / "empty_states.json"
        path.write_text(
            json.dumps({"name": "Empty", "initial_state": "start", "states": {}}),
            encoding="utf-8",
        )

        message = visualize_fsm_from_file(str(path))

        assert message != "Error: 'start'", (
            "the bare KeyError message is still surfacing"
        )
        assert "initial_state" in message, message
        assert "states" in message, message
        assert "start" in message, "the offending id must still be named: " + message

    def test_message_lists_the_states_that_do_exist(self):
        data = {
            "name": "Ghost",
            "initial_state": "ghost",
            "states": {
                "a": {"id": "a", "transitions": []},
                "b": {"id": "b", "transitions": []},
            },
        }

        with pytest.raises(ValueError) as exc_info:
            visualize_fsm_ascii(data, style="full")

        message = str(exc_info.value)
        assert "'a'" in message and "'b'" in message, message

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_every_style_gets_the_contextual_error(self, style):
        """The first raise is in `calculate_depths`, which runs for EVERY style --
        not in the STATES section, which only `full` builds. A guard placed in
        `create_states_section` would leave these two styles still raising KeyError.
        """
        data = {"name": "Ghost", "initial_state": "ghost", "states": {}}

        with pytest.raises(ValueError, match="initial_state"):
            visualize_fsm_ascii(data, style=style)

    def test_a_well_formed_fsm_is_unaffected(self):
        """Over-correction guard: the new check must not reject valid input."""
        for style in ("full", "compact", "minimal"):
            output = visualize_fsm_ascii(_multi_state_fsm_data(), style=style)
            assert "init" in output
