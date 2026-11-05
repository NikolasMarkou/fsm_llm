"""Step 8 of plan 5980fbb7: handlers and visualizer findings (F-HV-*).

Each class pins one finding and failed on the parent commit (8bd6cba).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsm_llm import visualizer
from fsm_llm.handlers import HandlerTiming, LambdaHandler
from fsm_llm.visualizer import (
    main,
    visualize_fsm_ascii,
    visualize_fsm_from_file,
)


def _state(sid: str, targets: tuple[str, ...] = (), **extra) -> dict:
    state = {"id": sid, "description": f"d {sid}", "purpose": f"p {sid}"}
    if targets:
        state["transitions"] = [
            {"target_state": t, "description": f"to {t}", "priority": 100}
            for t in targets
        ]
    state.update(extra)
    return state


def _fsm(initial: str, **states: tuple[str, ...]) -> dict:
    return {
        "name": "T",
        "description": "d",
        "initial_state": initial,
        "states": {sid: _state(sid, tg) for sid, tg in states.items()},
    }


def _write(tmp_path: Path, data, name: str = "f.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestMaximumPathLength:
    """F-HV-02: the METADATA row is the longest path, not the shallowest depth
    of the deepest terminal."""

    @staticmethod
    def _row(fsm: dict) -> str:
        out = visualize_fsm_ascii(fsm, "full")
        (row,) = [ln for ln in out.splitlines() if "Maximum Path Length" in ln]
        return row.strip("│ ").strip()

    def test_shortcut_edge_to_the_terminal_does_not_hide_the_long_path(self):
        fsm = _fsm("a", a=("b", "end"), b=("c",), c=("end",), end=())
        assert self._row(fsm) == "Maximum Path Length: 4"

    def test_a_loop_is_not_followed_twice(self):
        fsm = _fsm("a", a=("b",), b=("a", "end"), end=())
        assert self._row(fsm) == "Maximum Path Length: 3"

    def test_a_chain_and_a_single_state(self):
        assert self._row(_fsm("a", a=("b",), b=("c",), c=())) == (
            "Maximum Path Length: 3"
        )
        assert self._row(_fsm("a", a=())) == "Maximum Path Length: 1"

    def test_an_unreachable_terminal_is_not_counted(self):
        fsm = _fsm("a", a=("a",), orphan=("z",), z=())
        assert self._row(fsm) == "Maximum Path Length: 1"


class TestStyleIsValidated:
    """F-HV-06: ``style`` is refused like ``output_format``."""

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="unknown style 'bogus'"):
            visualize_fsm_ascii(_fsm("a", a=()), "bogus")

    def test_from_file_reports_it(self, tmp_path):
        path = _write(tmp_path, _fsm("a", a=()))
        out = visualize_fsm_from_file(path, "bogus")
        assert out.startswith("Error:")
        assert "unknown style 'bogus'" in out

    def test_main_exits_with_failure(self, tmp_path):
        assert main(_write(tmp_path, _fsm("a", a=())), "bogus") == 1

    def test_the_three_styles_are_one_tuple(self):
        assert visualizer.ASCII_STYLES == ("full", "compact", "minimal")


class TestMalformedInputHasOneExitContract:
    """F-HV-07: the same malformed input fails in every style."""

    @staticmethod
    def _bad_condition_keys() -> dict:
        fsm = _fsm("a", a=("b",), b=())
        fsm["states"]["a"]["transitions"][0]["conditions"] = [
            {"description": "c", "requires_context_keys": [1]}
        ]
        return fsm

    @staticmethod
    def _bad_state_keys() -> dict:
        fsm = _fsm("a", a=("b",), b=())
        fsm["states"]["a"]["required_context_keys"] = [1, 2]
        return fsm

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    @pytest.mark.parametrize("make", ["_bad_condition_keys", "_bad_state_keys"])
    def test_every_style_fails(self, tmp_path, style, make):
        path = _write(tmp_path, getattr(self, make)())
        out = visualize_fsm_from_file(path, style)
        assert out.startswith("Error:")
        assert "Could not generate diagram" not in out
        assert main(path, style) == 1

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_a_good_definition_never_prints_the_fallback_line(self, style):
        out = visualize_fsm_ascii(_fsm("a", a=("b",), b=()), style)
        assert "Could not generate diagram" not in out


class TestOneRawFsmJsonReader:
    """F-HV-08: one reader (UTF-8, object check) for loader, validator, visualizer."""

    def test_read_json_object_returns_the_object(self, tmp_path):
        from fsm_llm.utilities import read_json_object

        path = tmp_path / "u.json"
        path.write_text(json.dumps({"name": "é"}, ensure_ascii=False), "utf-8")
        assert read_json_object(str(path)) == {"name": "é"}

    def test_read_json_object_refuses_a_list(self, tmp_path):
        from fsm_llm.utilities import read_json_object

        with pytest.raises(ValueError, match="must be a JSON object"):
            read_json_object(_write(tmp_path, [1, 2]))

    @pytest.mark.parametrize("fmt", ["ascii", "mermaid", "dot"])
    def test_visualizer_says_it_is_not_an_object(self, tmp_path, fmt):
        out = visualize_fsm_from_file(_write(tmp_path, [1, 2]), output_format=fmt)
        assert out == "Error: FSM definition must be a JSON object"


class TestLambdaHandlerNormalisesBareStrings:
    """F-HV-09: a bare ``str`` is one name for all six set arguments."""

    @staticmethod
    def _handler(**sets) -> LambdaHandler:
        args = {
            "states": set(),
            "target_states": set(),
            "required_keys": set(),
            "updated_keys": set(),
        }
        args.update(sets)
        return LambdaHandler(
            name="h",
            condition_lambdas=[],
            execution_lambda=lambda ctx: None,
            timings=None,
            **args,
        )

    def test_states_is_not_a_substring_test(self):
        h = self._handler(states="abc")
        t = HandlerTiming.PRE_PROCESSING
        assert h.should_execute(t, "abc", None, {}) is True
        assert h.should_execute(t, "b", None, {}) is False

    def test_target_states_is_not_a_substring_test(self):
        h = self._handler(target_states="abc")
        t = HandlerTiming.PRE_TRANSITION
        assert h.should_execute(t, "x", "abc", {}) is True
        assert h.should_execute(t, "x", "b", {}) is False

    def test_required_keys_is_not_a_character_set(self):
        h = self._handler(required_keys="name")
        t = HandlerTiming.PRE_PROCESSING
        assert h.should_execute(t, "x", None, {"name": 1}) is True
        assert h.should_execute(t, "x", None, {"n": 1, "a": 1, "m": 1, "e": 1}) is False

    def test_updated_keys_is_not_a_character_set(self):
        h = self._handler(updated_keys="name")
        t = HandlerTiming.CONTEXT_UPDATE
        assert h.should_execute(t, "x", None, {}, {"name"}) is True
        assert h.should_execute(t, "x", None, {}, {"n"}) is False

    def test_the_two_negative_sets_still_take_a_bare_string(self):
        h = self._handler(not_states="abc", not_target_states="abc")
        assert h.not_states == {"abc"}
        assert h.not_target_states == {"abc"}
