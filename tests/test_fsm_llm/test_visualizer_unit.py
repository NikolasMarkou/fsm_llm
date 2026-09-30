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
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

from fsm_llm.visualizer import (
    ICONS,
    build_graph_representation,
    create_fancy_header,
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
        # Step-14 C8 re-baseline: the primary payload (report / diagram) now
        # goes to stdout via print() and diagnostics stay on the logger, so
        # both streams land in the one buffer "the user sees". The stdout
        # routing itself is pinned by test_audit_2026_09_21 test_c8_*.
        with redirect_stdout(buffer):
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
        row would pass the alignment tests. The id must still be legible.

        D-033 strengthened this from `long_id[:40] in output`, which the
        free-standing state diagram satisfied on its own and which said nothing
        about the section boxes. Both ENDS must survive now.
        """
        long_id, data = _long_id_fsm_data()
        output = visualize_fsm_ascii(data, style="full")

        assert long_id[:20] in output, (
            "the head of the long state id vanished from the render entirely"
        )
        assert long_id[-20:] in output, (
            "the TAIL of the long state id appears nowhere in the render -- a "
            "head-only truncation put the information beyond recovery"
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
# D-019 / fresh-audit-sweep.md finding #12: create_fancy_header computed its
# own width from `max(60, len(name) + 10)`, growing wider than every sibling
# 60-col section box (METADATA/STATES/TRANSITIONS/PERSONA) for a name over
# ~50 chars -- and never truncated a name that still did not fit.
# ------------------------------------------------------------------


class TestFancyHeaderBoxWidth:
    def test_header_box_width_matches_section_boxes_for_a_long_name(self):
        """The exact scenario plan.md names: a 54-char FSM name."""
        name = "x" * 54
        assert len(name) == 54
        data = {
            "name": name,
            "initial_state": "only",
            "states": {"only": {"id": "only", "description": "d", "transitions": []}},
        }
        output = visualize_fsm_ascii(data, style="full")

        boxes = _bordered_boxes(output)
        header_box = next((box for box in boxes if box[0].startswith("╭")), None)
        assert header_box is not None, "no header box found in the render"

        header_width = len(header_box[0])
        assert header_width == _SECTION_WIDTH, (
            f"header box is {header_width} chars wide against sibling section "
            f"boxes' {_SECTION_WIDTH}-char border:\n"
            + "\n".join(f"{len(x):>4} {x}" for x in header_box)
        )
        for line in header_box:
            assert len(line) == _SECTION_WIDTH, (
                f"ragged header box, widths={sorted({len(x) for x in header_box})}:\n"
                + "\n".join(f"{len(x):>4} {x}" for x in header_box)
            )

        # Vacuity guard: confirm a METADATA-shaped box (also 62 chars) is
        # present too, so this test is genuinely comparing two boxes, not
        # trivially passing because only the header box exists.
        section_boxes = [
            b for b in boxes if len(b[0]) == _SECTION_WIDTH and b is not header_box
        ]
        assert section_boxes, "no sibling section box found to compare against"

    def test_header_elides_a_name_too_long_to_fit_rather_than_growing(self):
        """A DIFFERENT code path than the width-cap alone: a name still too
        long for the capped 60-col box (unlike the 54-char case above, which
        fits inside 58 once padding is subtracted) must be shortened via
        `_fit()`, not silently overflow `.center()` (which never truncates).
        """
        long_name = "y" * 200
        lines = create_fancy_header(long_name)

        assert len(lines[0]) == 62, f"top border is {len(lines[0])} chars, not 62"
        assert len(lines[1]) == 62, f"name row is {len(lines[1])} chars, not 62"
        assert "…" in lines[1], (
            f"a 200-char name was not visibly shortened: {lines[1]!r}"
        )
        assert long_name not in lines[1], (
            "the full 200-char name still appears verbatim -- not elided at all"
        )

    def test_a_short_name_is_unaffected(self):
        """Over-correction guard: a normal-length name must not be elided or
        otherwise changed by the width cap."""
        lines = create_fancy_header("MyFSM")
        assert len(lines[0]) == 62
        assert "MyFSM" in lines[1]
        assert "…" not in lines[1]


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


# ------------------------------------------------------------------
# G-18 / SC-17: a dangling `target_state` must say so
# ------------------------------------------------------------------


def _dangling_target_fsm_data():
    return {
        "name": "Dangling",
        "initial_state": "a",
        "states": {
            "a": {
                "id": "a",
                "description": "first",
                "transitions": [
                    {"target_state": "nowhere", "description": "goes away"}
                ],
            },
        },
    }


class TestDanglingTransitionTargetIsDiagnosable:
    """G-18. A `target_state` absent from `states` surfaced as a bare
    `KeyError('nowhere')` from `calculate_depths`, which
    `visualize_fsm_from_file`'s broad `except Exception` rendered as the useless
    `"Error: 'nowhere'"`.

    The motivating framing ("the validator passes input the visualizer crashes on")
    is a GHOST -- both `FSMValidator` and pydantic Stage 0 reject this shape as an
    ERROR. The real gap is that the visualizer is a separate entry point that never
    consults either. See decisions.md D-013.
    """

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_every_style_names_both_the_source_and_the_target(self, style):
        """The first raise is `calculate_depths`' `state_metrics[target]["depth"]`,
        which runs for EVERY style -- confirmed by probe, not by reading.
        """
        with pytest.raises(ValueError) as exc_info:
            visualize_fsm_ascii(_dangling_target_fsm_data(), style=style)

        message = str(exc_info.value)
        assert "'a'" in message, "the source state must be named: " + message
        assert "'nowhere'" in message, "the missing target must be named: " + message

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_it_is_a_value_error_not_a_key_error(self, style):
        """`KeyError` is a subclass of `LookupError`, not of `ValueError`, so this
        assertion is genuinely falsified by the un-fixed source.
        """
        with pytest.raises(ValueError):
            visualize_fsm_ascii(_dangling_target_fsm_data(), style=style)

        with pytest.raises(Exception) as exc_info:
            visualize_fsm_ascii(_dangling_target_fsm_data(), style=style)
        assert not isinstance(exc_info.value, KeyError), (
            "the bare KeyError is still escaping"
        )

    def test_cli_path_gets_the_contextual_message(self, tmp_path):
        """`visualize_fsm_from_file` swallows everything into `f"Error: {e}"`, so the
        CLI user is who this fix is for.
        """
        path = tmp_path / "dangling.json"
        path.write_text(json.dumps(_dangling_target_fsm_data()), encoding="utf-8")

        message = visualize_fsm_from_file(str(path))

        assert message != "Error: 'nowhere'", (
            "the bare KeyError message is still surfacing"
        )
        assert "'a'" in message and "'nowhere'" in message, message

    def test_a_dangling_target_on_an_unreachable_state_is_also_rejected(self):
        """Deliberate tightening (D-013). `calculate_depths` never visits an
        unreachable state, so this shape rendered an arrow to a state that does not
        exist instead of crashing. The guard sweeps every state, not the reachable set.
        """
        data = {
            "name": "Orphan",
            "initial_state": "a",
            "states": {
                "a": {"id": "a", "description": "first", "transitions": []},
                "orphan": {
                    "id": "orphan",
                    "description": "unreachable",
                    "transitions": [{"target_state": "nowhere"}],
                },
            },
        }

        with pytest.raises(ValueError) as exc_info:
            visualize_fsm_ascii(data, style="full")

        message = str(exc_info.value)
        assert "'orphan'" in message and "'nowhere'" in message, message

    def test_a_transition_with_no_target_state_at_all_is_named(self):
        """`build_graph_representation` defaults a missing `target_state` to `""`,
        which produced the maximally unhelpful `KeyError('')` / `"Error: ''"`.
        """
        data = {
            "name": "NoTarget",
            "initial_state": "a",
            "states": {
                "a": {
                    "id": "a",
                    "description": "first",
                    "transitions": [{"description": "target_state key is missing"}],
                },
            },
        }

        with pytest.raises(ValueError) as exc_info:
            visualize_fsm_ascii(data, style="full")

        assert "'a'" in str(exc_info.value), str(exc_info.value)

    def test_a_well_formed_fsm_is_unaffected(self):
        """Over-correction guard: the new check must not reject valid input, including
        the self-loops and back-edges the other fixtures in this file rely on.
        """
        for fixture in (
            _multi_state_fsm_data,
            _linear_fsm_data,
            _unsorted_terminal_fsm_data,
        ):
            for style in ("full", "compact", "minimal"):
                assert visualize_fsm_ascii(fixture(), style=style)


class TestTruncationIsVisibleAndUnambiguous:
    """D-033. D-028 bought alignment with a silent head-only slice, and for rows
    carrying a state id that is worse than a ragged box: two DISTINCT states
    sharing a long prefix rendered byte-identically in the TRANSITIONS graph,
    with no marker that anything had been cut. Reading which state goes where is
    the entire purpose of that section.
    """

    _A = "checkout_payment_authorization_pending_manual_review_ALPHA"
    _B = "checkout_payment_authorization_pending_manual_review_BRAVO"

    def _same_prefix_fsm(self):
        return {
            "name": "AmbiguityFSM",
            "initial_state": "start",
            "states": {
                "start": {
                    "id": "start",
                    "description": "d",
                    "purpose": "p",
                    "transitions": [
                        {"target_state": self._A, "description": "to alpha"},
                        {"target_state": self._B, "description": "to bravo"},
                    ],
                },
                self._A: {
                    "id": self._A,
                    "description": "d",
                    "purpose": "p",
                    "transitions": [],
                },
                self._B: {
                    "id": self._B,
                    "description": "d",
                    "purpose": "p",
                    "transitions": [],
                },
            },
        }

    @pytest.mark.parametrize("style", ["full", "compact", "detailed"])
    def test_two_same_prefix_states_do_not_render_identically(self, style):
        """The defect, stated directly. Under a head-only slice both ids became
        `...pending_manual_review` and the diagram showed one node where the FSM
        has two."""
        output = visualize_fsm_ascii(self._same_prefix_fsm(), style=style)

        rows = [
            line
            for line in output.splitlines()
            if line.startswith("│ ") and "review" in line
        ]
        duplicates = {row for row in rows if rows.count(row) > 1}
        assert not duplicates, (
            f"style={style!r}: distinct states render as identical rows, so the "
            f"diagram is ambiguous:\n" + "\n".join(sorted(duplicates))
        )

    @pytest.mark.parametrize("style", ["full", "compact", "detailed"])
    def test_both_distinguishing_suffixes_survive(self, style):
        """Stronger than 'the rows differ': the part that actually tells the two
        states apart must be present, not merely some incidental difference."""
        output = visualize_fsm_ascii(self._same_prefix_fsm(), style=style)

        for marker in ("ALPHA", "BRAVO"):
            assert marker in output, (
                f"style={style!r}: {marker} appears nowhere, so the reader "
                "cannot tell the two states apart at all"
            )

    def test_shortening_leaves_a_visible_marker(self):
        """A truncation the reader cannot see is a truncation the reader will
        mistake for the whole id."""
        output = visualize_fsm_ascii(self._same_prefix_fsm(), style="full")
        assert "…" in output, "content was shortened with no ellipsis marker"

    def test_the_marker_costs_exactly_one_character_of_width(self):
        """Vacuity/regression guard: the ellipsis must not reintroduce the
        raggedness D-028 fixed. It is one character in `len()` terms."""
        output = visualize_fsm_ascii(self._same_prefix_fsm(), style="full")

        for box in _bordered_boxes(output):
            widths = {len(line) for line in box}
            assert len(widths) == 1, (
                f"the ellipsis made a box ragged, widths={sorted(widths)}:\n"
                + "\n".join(f"{len(x):>4} {x}" for x in box)
            )

    def test_fit_is_exact_and_keeps_both_ends(self):
        """Unit-level pin on the helper, including the degenerate widths."""
        from fsm_llm.visualizer import _fit

        assert _fit("short", 10) == "short"
        assert _fit("exactfit!!", 10) == "exactfit!!"
        assert len(_fit("a" * 100, 10)) == 10
        # width 5 -> 4 content chars + the marker: 2 from the head, 2 from the tail
        assert _fit("abcdefghij", 5) == "ab…ij"
        assert _fit("abcdefghij", 1) == "a"
        assert _fit("abcdefghij", 0) == ""

        fitted = _fit("HEAD" + "x" * 60 + "TAIL", 20)
        assert fitted.startswith("HEAD")
        assert fitted.endswith("TAIL")
        assert "…" in fitted


# ==================================================================
# D-002 / fresh-audit-sweep.md finding #1: create_state_boxes never got the
# D-033 ellipsis fix
# ==================================================================

# A common prefix long enough that BOTH the "┃ " lead-in and the trailing
# " (TERMINAL)" label survive a bare `[: box_width - 1]` == 59-character head
# slice before the divergent tail (ALPHA/BRAVO) is ever reached. box_width is
# capped at 60 (visualizer.py:1218) no matter how long the state id itself is.
_STATE_DIAGRAM_PREFIX = (
    "checkout_payment_authorization_pending_manual_review_workflow_stage_"
)
_STATE_DIAGRAM_A = _STATE_DIAGRAM_PREFIX + "ALPHA"
_STATE_DIAGRAM_B = _STATE_DIAGRAM_PREFIX + "BRAVO"


def _state_diagram_prefix_collision_fsm_data():
    """Two TERMINAL states sharing a >60-char prefix, diverging only past the
    old bare ``[: box_width - 1]`` cutoff -- the exact shape
    ``findings/fresh-audit-sweep.md`` finding #1 describes for
    ``create_state_boxes``.
    """
    return {
        "name": "StateDiagramPrefixCollisionFSM",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "d",
                "purpose": "p",
                "transitions": [
                    {"target_state": _STATE_DIAGRAM_A, "description": "to alpha"},
                    {"target_state": _STATE_DIAGRAM_B, "description": "to bravo"},
                ],
            },
            _STATE_DIAGRAM_A: {
                "id": _STATE_DIAGRAM_A,
                "description": "d",
                "purpose": "p",
                "transitions": [],
            },
            _STATE_DIAGRAM_B: {
                "id": _STATE_DIAGRAM_B,
                "description": "d",
                "purpose": "p",
                "transitions": [],
            },
        },
    }


def _state_diagram_rows(output):
    """Return the two states' own id rows from the STATE DIAGRAM section.

    Both A and B are TERMINAL (no outgoing transitions), so their box rows
    start with the terminal style's vertical glyph (``"┃ "``), not the
    default ``"│ "`` -- the STATES-section content filter
    ``TestTruncationIsVisibleAndUnambiguous`` uses would not select these
    rows at all; this is a genuinely different renderer (``create_state_boxes``,
    not ``create_states_section``).
    """
    return [
        line
        for line in output.splitlines()
        if line.startswith("┃ ") and "checkout_payment" in line
    ]


class TestStateDiagramBoxTruncationIsVisibleAndUnambiguous:
    """D-002 / fresh-audit-sweep.md finding #1. ``create_state_boxes`` (used
    only by the STATE DIAGRAM section of ``--style full`` output, via
    ``generate_enhanced_ascii_diagram``) never received D-033's ``_fit()``
    truncation fix, even though the D-022 comment at ``visualizer.py:147``
    names it as the mirror site. Drives the PUBLIC ``fsm-llm-visualize`` CLI
    entry point (``main_cli``) end to end -- not a private helper -- per the
    plan's requirement that this reproduce through the real console script.
    """

    def _render_full_style(self, tmp_path):
        path = tmp_path / "prefix_collision.json"
        path.write_text(
            json.dumps(_state_diagram_prefix_collision_fsm_data()), encoding="utf-8"
        )
        with _cli_capture() as buffer:
            with patch.object(
                sys,
                "argv",
                ["fsm-llm-visualize", "--fsm", str(path), "--style", "full"],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main_cli()
        assert exc_info.value.code == 0
        return buffer.getvalue()

    def test_two_same_prefix_states_do_not_render_identically(self, tmp_path):
        """The defect, stated directly. Under the old bare head slice both ids
        became the identical row ``"┃ checkout_payment_...ow_stage_work┃"``
        and the STATE DIAGRAM showed one node where the FSM has two."""
        output = self._render_full_style(tmp_path)
        rows = _state_diagram_rows(output)
        assert len(rows) == 2, (
            "expected exactly 2 state-id rows in the STATE DIAGRAM section, "
            f"got {len(rows)}:\n" + "\n".join(output.splitlines())
        )
        assert rows[0] != rows[1], (
            "the two states' STATE DIAGRAM rows are byte-identical -- the "
            f"reader cannot tell them apart:\n{rows[0]!r}\n{rows[1]!r}"
        )

    def test_both_distinguishing_suffixes_survive(self, tmp_path):
        """Stronger than 'the rows differ': the part that actually tells the
        two states apart must be present IN THE ROW ITSELF, not merely
        somewhere else in the document (e.g. the untruncated "Connections:"
        list, which always carries the full id and would pass vacuously)."""
        output = self._render_full_style(tmp_path)
        rows = _state_diagram_rows(output)
        assert len(rows) == 2
        assert any("ALPHA" in row for row in rows), (
            "ALPHA appears in neither STATE DIAGRAM row:\n" + "\n".join(rows)
        )
        assert any("BRAVO" in row for row in rows), (
            "BRAVO appears in neither STATE DIAGRAM row:\n" + "\n".join(rows)
        )

    def test_shortening_leaves_a_visible_marker(self, tmp_path):
        """A truncation the reader cannot see is a truncation the reader will
        mistake for the whole id."""
        output = self._render_full_style(tmp_path)
        rows = _state_diagram_rows(output)
        assert any("…" in row for row in rows), (
            "content was shortened with no ellipsis marker in the STATE "
            "DIAGRAM section:\n" + "\n".join(rows)
        )


# ==================================================================
# D-007 / review-iter-1.md WARNING 4: create_states_section's icon-budget
# branch (visualizer.py ~569) never got the D-033/D-002 _fit() fix either.
# ==================================================================


class TestStatesSectionIconBudgetTruncationIsVisible:
    """The icon-budget branch of ``create_states_section`` pre-sliced
    ``state_line`` with a bare ``state_line[:body]`` BEFORE ``_states_box_row``'s
    own ``_fit()`` call ever saw it -- by the time ``_fit`` ran, the content was
    already exactly ``width`` chars long, so ``_fit`` was a no-op. This branch
    only fires when a state carries an icon (``required_context_keys`` triggers
    the ``*`` input icon here), so ``TestTruncationIsVisibleAndUnambiguous``'s
    icon-less fixture never exercised it, and Success Criterion 2 was not met
    for every state shape.
    """

    _A = "checkout_payment_authorization_pending_manual_review_ALPHA"
    _B = "checkout_payment_authorization_pending_manual_review_BRAVO"

    def _same_prefix_fsm_with_icons(self):
        return {
            "name": "IconBudgetAmbiguityFSM",
            "initial_state": "start",
            "states": {
                "start": {
                    "id": "start",
                    "description": "d",
                    "purpose": "p",
                    "transitions": [
                        {"target_state": self._A, "description": "to alpha"},
                        {"target_state": self._B, "description": "to bravo"},
                    ],
                },
                self._A: {
                    "id": self._A,
                    "description": "d",
                    "purpose": "p",
                    "required_context_keys": ["some_key"],
                    "transitions": [],
                },
                self._B: {
                    "id": self._B,
                    "description": "d",
                    "purpose": "p",
                    "required_context_keys": ["some_key"],
                    "transitions": [],
                },
            },
        }

    def _states_section_rows(self, output):
        """The STATES-section content rows for A/B. A/B have no outgoing
        transitions, so they render as TERMINAL states whose box vertical
        glyph is ``"┃"`` nested inside the section's own ``"│ "`` border --
        distinct from the TRANSITIONS section's plain ``"│ "`` rows and from
        ``create_state_boxes``' STATE DIAGRAM rows (which start with a bare
        ``"┃ "``, no section border, and are a different renderer)."""
        return [
            line
            for line in output.splitlines()
            if line.startswith("│ ┃") and "review" in line
        ]

    def test_two_same_prefix_icon_carrying_states_do_not_render_identically(self):
        """The defect, stated directly: under the bare pre-slice both ids
        became the identical row ``"checkout_..._review_ *"`` -- the icon
        survived but BOTH the distinguishing suffix and the ellipsis did not."""
        output = visualize_fsm_ascii(self._same_prefix_fsm_with_icons(), style="full")
        rows = self._states_section_rows(output)
        assert len(rows) == 2, (
            "expected exactly 2 STATES-section id rows, got "
            f"{len(rows)}:\n" + "\n".join(output.splitlines())
        )
        assert rows[0] != rows[1], (
            "the two icon-carrying states' STATES-section rows are byte-"
            f"identical -- the reader cannot tell them apart:\n"
            f"{rows[0]!r}\n{rows[1]!r}"
        )

    def test_both_distinguishing_suffixes_survive(self):
        """Stronger than 'the rows differ': the part that actually tells the
        two states apart must be present in the row itself."""
        output = visualize_fsm_ascii(self._same_prefix_fsm_with_icons(), style="full")
        rows = self._states_section_rows(output)
        assert len(rows) == 2
        assert any("ALPHA" in row for row in rows), (
            "ALPHA appears in neither STATES-section row:\n" + "\n".join(rows)
        )
        assert any("BRAVO" in row for row in rows), (
            "BRAVO appears in neither STATES-section row:\n" + "\n".join(rows)
        )

    def test_the_icon_still_renders(self):
        """Regression guard: the fix reserves room for the icon and must not
        drop it while making the distinguishing suffix visible."""
        output = visualize_fsm_ascii(self._same_prefix_fsm_with_icons(), style="full")
        rows = self._states_section_rows(output)
        assert len(rows) == 2
        for row in rows:
            assert ICONS["input"] in row, f"input icon missing from row:\n{row!r}"

    def test_shortening_leaves_a_visible_marker(self):
        """A truncation the reader cannot see is a truncation the reader will
        mistake for the whole id."""
        output = visualize_fsm_ascii(self._same_prefix_fsm_with_icons(), style="full")
        rows = self._states_section_rows(output)
        assert any("…" in row for row in rows), (
            "content was shortened with no ellipsis marker in the STATES "
            "section icon-budget branch:\n" + "\n".join(rows)
        )


# ------------------------------------------------------------------
# Graph data, Mermaid and DOT export, `--format`
# ------------------------------------------------------------------

import re

import pydantic

import fsm_llm
from fsm_llm.definitions import FSMDefinition
from fsm_llm.visualizer import (
    OUTPUT_FORMATS,
    FSMGraph,
    FSMGraphEdge,
    FSMGraphNode,
    build_fsm_graph,
    to_dot,
    to_mermaid,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _claude_md_example_fsm() -> dict:
    """The FSM definition example of the root ``CLAUDE.md``, read from the file."""
    text = (_REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    blocks = [
        block
        for block in re.findall(r"```json\n(.*?)```", text, re.DOTALL)
        if '"initial_state"' in block
    ]
    assert len(blocks) == 1, "root CLAUDE.md must hold exactly one FSM example"
    return json.loads(blocks[0])


def _tricky_fsm_data() -> dict:
    """Four states in one FSM: a self-loop, two terminal states, a state id
    with a space and a hyphen, one that is a Mermaid keyword (``end``), one
    that equals a generated alias (``state_1``), a label with quotes, a
    newline and ``: ; # < >``, a label with a backslash, and a transition
    with neither description nor priority."""
    return {
        "name": 'Tricky "FSM"',
        "description": "Export edge cases",
        "initial_state": "intake",
        "states": {
            "intake": {
                "description": "Collect the request",
                "purpose": "Route it",
                "transitions": [
                    {
                        "target_state": "my state-1",
                        "description": 'Say "hi":\n next; #1 <b>',
                        "priority": 5,
                    },
                    {
                        "target_state": "intake",
                        "description": "again",
                        "priority": 20,
                    },
                    {"target_state": "end"},
                    {
                        "target_state": "state_1",
                        "description": "back\\slash",
                        "priority": 0,
                    },
                ],
            },
            "my state-1": {
                "description": "Needs quoting",
                "purpose": "Review",
                "transitions": [{"target_state": "end", "description": "done"}],
            },
            "end": {"description": "Done", "purpose": "Stop", "transitions": []},
            "state_1": {"description": "Rejected", "purpose": "Stop"},
        },
    }


_DOT_STRING = r'"(?:[^"\\\n]|\\.)*"'
_DOT_STATEMENT = re.compile(
    rf"    (?:rankdir=LR"
    rf"|{_DOT_STRING}(?: \[shape=(?:point|doublecircle)\])?"
    rf"|{_DOT_STRING} -> {_DOT_STRING}(?: \[label={_DOT_STRING}\])?);\Z"
)
_MERMAID_STATEMENT = re.compile(
    r"    (?:state \"(?P<display>[^\"\n]*)\" as \w+"
    r"|(?:\[\*\]|\w+) --> (?:\[\*\]|\w+)(?: : (?P<label>.+))?)\Z"
)


def _assert_well_formed_dot(text: str) -> None:
    lines = text.split("\n")
    assert re.fullmatch(rf"digraph {_DOT_STRING} \{{", lines[0]), lines[0]
    assert lines[-1] == "}"
    for line in lines[1:-1]:
        assert _DOT_STATEMENT.match(line), f"not a DOT statement: {line!r}"


def _assert_well_formed_mermaid(text: str) -> None:
    lines = text.split("\n")
    assert lines[0] == "stateDiagram-v2"
    for line in lines[1:]:
        match = _MERMAID_STATEMENT.match(line)
        assert match, f"not a state-diagram statement: {line!r}"
        free_text = match.group("display") or match.group("label") or ""
        leftover = re.sub(r"#\w+;", "", free_text)
        assert not set(leftover) & set('"#;:<>'), f"unescaped text in {line!r}"


class TestFsmGraph:
    """`build_fsm_graph`: the one graph of an FSM definition (D-018)."""

    def test_claude_md_example(self):
        graph = build_fsm_graph(_claude_md_example_fsm())

        assert graph.name == "MyBot"
        assert graph.initial_state == "start"
        assert graph.nodes == (
            FSMGraphNode(
                id="start",
                description="Brief state description",
                purpose="What should be accomplished",
                is_initial=True,
                is_terminal=False,
            ),
            FSMGraphNode(
                id="next",
                description="Terminal state",
                purpose="Wrap up the conversation",
                is_initial=False,
                is_terminal=True,
            ),
        )
        assert graph.edges == (
            FSMGraphEdge(
                source="start",
                target="next",
                description="When this transition should fire",
                priority=100,
            ),
        )

    def test_definition_and_its_dict_give_the_same_graph(self):
        data = _claude_md_example_fsm()
        assert build_fsm_graph(FSMDefinition(**data)) == build_fsm_graph(data)

    def test_order_priority_self_loop_and_terminals(self):
        graph = build_fsm_graph(_tricky_fsm_data())

        assert [n.id for n in graph.nodes] == ["intake", "my state-1", "end", "state_1"]
        assert [n.id for n in graph.nodes if n.is_initial] == ["intake"]
        assert [n.id for n in graph.nodes if n.is_terminal] == ["end", "state_1"]
        assert [(e.source, e.target, e.priority) for e in graph.edges] == [
            ("intake", "my state-1", 5),
            ("intake", "intake", 20),
            ("intake", "end", 100),  # no priority: the Transition default
            ("intake", "state_1", 0),  # an explicit 0 is kept, not defaulted
            ("my state-1", "end", 100),
        ]
        assert graph.edges[2].description == ""

    def test_descriptions_are_not_shortened(self):
        data = _linear_fsm_data()
        long_text = "word " * 60 + 'and the "last" word\nis on a second line'
        data["states"]["start"]["transitions"][0]["description"] = long_text

        assert build_fsm_graph(data).edges[0].description == long_text

    def test_null_transitions_is_a_terminal_state(self):
        data = _linear_fsm_data()
        data["states"]["end"]["transitions"] = None

        assert build_fsm_graph(data).nodes[1].is_terminal is True

    def test_a_state_that_is_initial_and_terminal(self):
        data = {
            "name": "One",
            "initial_state": "only",
            "states": {"only": {"description": "d", "purpose": "p"}},
        }
        graph = build_fsm_graph(data)

        assert (graph.nodes[0].is_initial, graph.nodes[0].is_terminal) == (True, True)
        assert graph.edges == ()
        assert to_mermaid(graph).split("\n")[1:] == [
            "    [*] --> only",
            "    only --> [*]",
        ]

    def test_missing_initial_state_is_refused(self):
        data = _linear_fsm_data()
        data["initial_state"] = "nowhere"

        with pytest.raises(ValueError, match="initial_state 'nowhere'"):
            build_fsm_graph(data)

    def test_dangling_target_is_refused(self):
        data = _linear_fsm_data()
        data["states"]["start"]["transitions"][0]["target_state"] = "ghost"

        with pytest.raises(ValueError, match="non-existent state 'ghost'"):
            build_fsm_graph(data)

    def test_a_non_integer_priority_is_refused(self):
        data = _linear_fsm_data()
        data["states"]["start"]["transitions"][0]["priority"] = "high"

        with pytest.raises(ValueError, match="priority"):
            build_fsm_graph(data)

    def test_graph_is_frozen(self):
        graph = build_fsm_graph(_linear_fsm_data())

        with pytest.raises(pydantic.ValidationError):
            graph.name = "other"
        with pytest.raises(pydantic.ValidationError):
            graph.nodes[0].is_terminal = True
        with pytest.raises(pydantic.ValidationError):
            graph.edges[0].priority = 1

    def test_input_is_not_mutated(self):
        data = _tricky_fsm_data()
        before = json.dumps(data, sort_keys=True)
        build_fsm_graph(data)

        assert json.dumps(data, sort_keys=True) == before

    def test_public_names_are_exported(self):
        for name in (
            "FSMGraph",
            "FSMGraphNode",
            "FSMGraphEdge",
            "build_fsm_graph",
            "to_mermaid",
            "to_dot",
        ):
            assert name in fsm_llm.__all__
            assert getattr(fsm_llm, name) is getattr(fsm_llm.visualizer, name)


class TestMermaidExport:
    def test_claude_md_example(self):
        assert to_mermaid(build_fsm_graph(_claude_md_example_fsm())) == (
            "stateDiagram-v2\n"
            "    [*] --> start\n"
            "    start --> next : P100 When this transition should fire\n"
            "    next --> [*]"
        )

    def test_escaping_aliases_self_loop_and_two_terminals(self):
        text = to_mermaid(build_fsm_graph(_tricky_fsm_data()))

        assert text == (
            "stateDiagram-v2\n"
            '    state "my state-1" as state_1_\n'
            '    state "end" as state_2\n'
            "    [*] --> intake\n"
            "    intake --> state_1_ : "
            "P5 Say #quot;hi#quot;#58; next#59; #35;1 #lt;b#gt;\n"
            "    intake --> intake : P20 again\n"
            "    intake --> state_2 : P100\n"
            "    intake --> state_1 : P0 back\\slash\n"
            "    state_1_ --> state_2 : P100 done\n"
            "    state_2 --> [*]\n"
            "    state_1 --> [*]"
        )
        _assert_well_formed_mermaid(text)

    def test_a_state_id_with_quotes_and_a_newline_is_escaped_in_its_label(self):
        data = _linear_fsm_data()
        data["states"]['the "end"\n#2'] = data["states"].pop("end")
        data["states"]["start"]["transitions"][0]["target_state"] = 'the "end"\n#2'
        text = to_mermaid(build_fsm_graph(data))

        assert text == (
            "stateDiagram-v2\n"
            '    state "the #quot;end#quot; #35;2" as state_1\n'
            "    [*] --> start\n"
            "    start --> state_1 : P100 Finish\n"
            "    state_1 --> [*]"
        )
        _assert_well_formed_mermaid(text)

    @pytest.mark.parametrize("keyword", ["end", "End", "state", "note", "direction"])
    def test_a_keyword_state_id_is_never_a_bare_id(self, keyword):
        data = _linear_fsm_data()
        data["states"][keyword] = data["states"].pop("end")
        data["states"]["start"]["transitions"][0]["target_state"] = keyword
        lines = to_mermaid(build_fsm_graph(data)).split("\n")

        assert f'    state "{keyword}" as state_1' in lines
        assert "    start --> state_1 : P100 Finish" in lines
        edges = [ln.split(" : ")[0].split() for ln in lines if "-->" in ln]
        assert edges == [
            ["[*]", "-->", "start"],
            ["start", "-->", "state_1"],
            ["state_1", "-->", "[*]"],
        ]

    def test_renderer_reads_only_the_graph(self):
        graph = FSMGraph(
            name="Hand built",
            initial_state="a",
            nodes=(
                FSMGraphNode(id="a", is_initial=True),
                FSMGraphNode(id="b", is_terminal=True),
            ),
            edges=(FSMGraphEdge(source="a", target="b", priority=7),),
        )

        assert to_mermaid(graph) == (
            "stateDiagram-v2\n    [*] --> a\n    a --> b : P7\n    b --> [*]"
        )
        assert to_mermaid(graph) == to_mermaid(graph)


class TestDotExport:
    def test_claude_md_example(self):
        text = to_dot(build_fsm_graph(_claude_md_example_fsm()))

        assert text == (
            'digraph "MyBot" {\n'
            "    rankdir=LR;\n"
            '    "__start__" [shape=point];\n'
            '    "start";\n'
            '    "next" [shape=doublecircle];\n'
            '    "__start__" -> "start";\n'
            '    "start" -> "next" [label="P100 When this transition should fire"];\n'
            "}"
        )
        _assert_well_formed_dot(text)

    def test_escaping_quoted_ids_self_loop_and_two_terminals(self):
        text = to_dot(build_fsm_graph(_tricky_fsm_data()))

        assert text == (
            'digraph "Tricky \\"FSM\\"" {\n'
            "    rankdir=LR;\n"
            '    "__start__" [shape=point];\n'
            '    "intake";\n'
            '    "my state-1";\n'
            '    "end" [shape=doublecircle];\n'
            '    "state_1" [shape=doublecircle];\n'
            '    "__start__" -> "intake";\n'
            '    "intake" -> "my state-1" '
            '[label="P5 Say \\"hi\\": next; #1 <b>"];\n'
            '    "intake" -> "intake" [label="P20 again"];\n'
            '    "intake" -> "end" [label="P100"];\n'
            '    "intake" -> "state_1" [label="P0 back\\\\slash"];\n'
            '    "my state-1" -> "end" [label="P100 done"];\n'
            "}"
        )
        _assert_well_formed_dot(text)

    def test_a_state_id_with_a_quote_a_backslash_and_a_newline(self):
        odd = 'a"b\\c\nd'
        data = _linear_fsm_data()
        data["states"][odd] = data["states"].pop("end")
        data["states"]["start"]["transitions"][0]["target_state"] = odd
        text = to_dot(build_fsm_graph(data))

        assert '    "a\\"b\\\\c\\nd" [shape=doublecircle];' in text.split("\n")
        assert '    "start" -> "a\\"b\\\\c\\nd" [label="P100 Finish"];' in text
        _assert_well_formed_dot(text)

    def test_the_entry_point_never_reuses_a_state_id(self):
        data = _linear_fsm_data()
        data["states"]["__start__"] = data["states"].pop("start")
        data["initial_state"] = "__start__"
        lines = to_dot(build_fsm_graph(data)).split("\n")

        assert '    "__start___" [shape=point];' in lines
        assert '    "__start___" -> "__start__";' in lines
        assert '    "__start__";' in lines

    def test_renderer_reads_only_the_graph(self):
        graph = FSMGraph(
            name="G",
            initial_state="a",
            nodes=(FSMGraphNode(id="a", is_initial=True, is_terminal=True),),
            edges=(),
        )

        assert to_dot(graph) == (
            'digraph "G" {\n'
            "    rankdir=LR;\n"
            '    "__start__" [shape=point];\n'
            '    "a" [shape=doublecircle];\n'
            '    "__start__" -> "a";\n'
            "}"
        )


class TestVisualizeFormatFlag:
    """`fsm-llm-visualize --format ascii|mermaid|dot` through `main_cli`."""

    @staticmethod
    def _run(tmp_path, data, *flags):
        path = tmp_path / "fsm.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with _cli_capture() as buffer:
            with patch.object(
                sys, "argv", ["fsm-llm-visualize", "--fsm", str(path), *flags]
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main_cli()
        return exc_info.value.code, buffer.getvalue()

    def test_formats_are_ascii_mermaid_dot(self):
        assert OUTPUT_FORMATS == ("ascii", "mermaid", "dot")

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_default_output_is_the_ascii_diagram_unchanged(self, tmp_path, style):
        data = _claude_md_example_fsm()
        expected = visualize_fsm_ascii(data, style) + "\n"

        assert self._run(tmp_path, data, "--style", style) == (0, expected)
        assert self._run(tmp_path, data, "--style", style, "--format", "ascii") == (
            0,
            expected,
        )

    def test_no_flag_at_all_is_the_full_ascii_diagram(self, tmp_path):
        data = _claude_md_example_fsm()

        assert self._run(tmp_path, data) == (
            0,
            visualize_fsm_ascii(data, "full") + "\n",
        )

    def test_format_mermaid(self, tmp_path):
        data = _claude_md_example_fsm()
        code, output = self._run(tmp_path, data, "--format", "mermaid")

        assert code == 0
        assert output == to_mermaid(build_fsm_graph(data)) + "\n"
        assert output.startswith("stateDiagram-v2\n    [*] --> start\n")
        assert "    next --> [*]\n" in output
        assert [ln for ln in output.split("\n") if "P100" in ln] == [
            "    start --> next : P100 When this transition should fire"
        ]

    def test_format_dot(self, tmp_path):
        data = _tricky_fsm_data()
        code, output = self._run(tmp_path, data, "--format", "dot")

        assert code == 0
        assert output == to_dot(build_fsm_graph(data)) + "\n"
        _assert_well_formed_dot(output.rstrip("\n"))

    def test_style_does_not_change_a_graph_export(self, tmp_path):
        data = _claude_md_example_fsm()

        assert self._run(
            tmp_path, data, "--format", "mermaid", "--style", "minimal"
        ) == (self._run(tmp_path, data, "--format", "mermaid"))

    @pytest.mark.parametrize("flag", ["--format", "--style"])
    def test_an_invalid_choice_is_an_argparse_usage_error(self, tmp_path, flag, capsys):
        """An unknown `--format` exits as an unknown `--style` always has:
        argparse's usage error (code 2), with nothing on stdout."""
        code, output = self._run(tmp_path, _linear_fsm_data(), flag, "svg")

        assert code == 2
        assert output == ""
        assert "invalid choice: 'svg'" in capsys.readouterr().err

    @pytest.mark.parametrize("output_format", ["mermaid", "dot"])
    def test_an_invalid_definition_exits_1_with_the_reason(
        self, tmp_path, output_format
    ):
        data = _linear_fsm_data()
        data["initial_state"] = "nowhere"
        code, output = self._run(tmp_path, data, "--format", output_format)

        assert code == 1
        assert output.startswith("Error: initial_state 'nowhere' is not present")
        assert "stateDiagram" not in output and "digraph" not in output

    def test_from_file_refuses_an_unknown_format(self, tmp_path):
        path = tmp_path / "fsm.json"
        path.write_text(json.dumps(_linear_fsm_data()), encoding="utf-8")

        assert visualize_fsm_from_file(str(path), output_format="svg") == (
            "Error: unknown output format 'svg' (choose from ascii, mermaid, dot)"
        )
        assert visualize_fsm_from_file(str(path), output_format="dot") == to_dot(
            build_fsm_graph(_linear_fsm_data())
        )
