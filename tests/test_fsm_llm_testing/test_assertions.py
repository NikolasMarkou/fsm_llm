"""Contract of `assert_state`, `assert_context`, `assert_path`, `assert_result` and `PathRecorder`.

The conversations run offline on the `simple_greeting` example FSM (read only,
loaded with `API.from_file`) and on small inline FSMs, with `ScriptedLLM` as the
model. The failure-message tests pin the first-divergence wording; the
`python -O` test proves the assertions are explicit `raise` statements.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
import fsm_llm.testing as public
from fsm_llm import API
from fsm_llm.codes import CompletionKind, Reason, Status
from fsm_llm.definitions import CompletionResponse
from fsm_llm.handlers import BaseHandler, HandlerTiming
from fsm_llm.synthesis.runner import _result as synthesis_result
from fsm_llm.testing import (
    ConversationRun,
    PathRecorder,
    ScriptedLLM,
    ScriptedUser,
    advance_until_terminal,
    assert_context,
    assert_path,
    assert_result,
    assert_state,
    final_reply,
    tool_call_reply,
)
from fsm_llm.testing import assertions as assertions_module

_ROOT = Path(fsm_llm.__file__).resolve().parent.parent
_GREETER = _ROOT / "examples" / "basic" / "simple_greeting" / "fsm.json"
_GREETER_REPLIES = {
    "greeting": "Hi! How are you?",
    "conversation": "Glad to hear it.",
    "farewell": "Goodbye!",
}
_GREETER_DATA = {"mood": "happy", "intent": "chat"}
_GREETER_PATH = ("greeting", "conversation", "farewell")


def _greeter(*, handlers: list[Any] | None = None) -> tuple[API, ScriptedLLM]:
    llm = ScriptedLLM(extraction_data=dict(_GREETER_DATA), responses=_GREETER_REPLIES)
    return API.from_file(str(_GREETER), llm_interface=llm, handlers=handlers), llm


def _driven() -> ConversationRun:
    api, _ = _greeter()
    return ScriptedUser(["hello", "good, thanks"]).run(api)


def _state(
    state_id: str, *, say: str | None, to: str | None = None, **extra: Any
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "id": state_id,
        "description": state_id,
        "purpose": f"be {state_id}",
    }
    if say is not None:
        state["response_instructions"] = say
    if to is not None:
        state["transitions"] = [
            {"target_state": to, "description": f"go to {to}", "priority": 100}
        ]
    state.update(extra)
    return state


def _chain() -> dict[str, Any]:
    """a -> b (silent) -> c (terminal): no user message needed."""
    return {
        "name": "Chain",
        "description": "three states in a row",
        "initial_state": "a",
        "persona": "a test persona",
        "states": {
            "a": _state("a", say="Say A", to="b"),
            "b": _state("b", say="", to="c"),
            "c": _state("c", say="Say done"),
        },
    }


def _stuck() -> dict[str, Any]:
    """`a` waits for a key nobody extracts, so every turn stays in `a`."""
    gate = {
        "target_state": "b",
        "description": "when go is set",
        "priority": 100,
        "conditions": [
            {
                "description": "go is set",
                "requires_context_keys": ["go"],
                "logic": {"==": [{"var": "go"}, True]},
            }
        ],
    }
    return {
        "name": "Stuck",
        "description": "never leaves a",
        "initial_state": "a",
        "persona": "a test persona",
        "states": {
            "a": _state(
                "a",
                say="Still here.",
                extraction_instructions="Extract go.",
                required_context_keys=["go"],
                transitions=[gate],
            ),
            "b": _state("b", say="Left."),
        },
    }


def _inline(definition: dict[str, Any], *, handlers: list[Any] | None = None) -> API:
    """An API over an inline FSM; every state answers `ok` (nothing else is scripted)."""
    llm = ScriptedLLM(extraction_data={}, default_response="ok")
    return API.from_definition(definition, llm_interface=llm, handlers=handlers)


def _fails(call: Any, *args: Any, **kwargs: Any) -> str:
    """The message of the `AssertionError` that `call(*args, **kwargs)` raises."""
    with pytest.raises(AssertionError) as raised:
        call(*args, **kwargs)
    return str(raised.value)


class TestAssertState:
    def test_the_state_of_a_run_passes(self) -> None:
        assert_state(_driven(), "farewell")

    def test_an_api_and_conversation_id_pair_is_a_source(self) -> None:
        run = _driven()

        assert_state((run.api, run.conversation_id), "farewell")

    def test_a_wrong_state_names_expected_and_actual_and_the_path(self) -> None:
        message = _fails(assert_state, _driven(), "conversation")

        assert message == (
            "state: expected 'conversation', got 'farewell' "
            "(recorded path: greeting -> conversation -> farewell)"
        )

    def test_a_pair_has_no_recorded_path_to_show(self) -> None:
        run = _driven()

        message = _fails(assert_state, (run.api, run.conversation_id), "greeting")

        assert message == "state: expected 'greeting', got 'farewell'"

    def test_the_state_is_read_live_not_taken_from_the_record(self) -> None:
        api, _ = _greeter()
        run = ScriptedUser(["hello"]).run(api)
        assert_state(run, "conversation")

        api.converse("good, thanks", run.conversation_id)

        assert_state(run, "farewell")
        assert run.final_state == "conversation"  # the record is not rewritten

    def test_a_conversation_closed_by_the_run_reads_its_snapshot(self) -> None:
        api, _ = _greeter()
        run = ScriptedUser(["hello"]).run(api, close=True)

        assert run.ended is False  # still open when the run stopped
        assert_state(run, "conversation")
        message = _fails(assert_state, run, "farewell")
        assert message.startswith("state: expected 'farewell', got 'conversation'")

    def test_a_conversation_the_api_never_started_fails_with_a_clear_line(self) -> None:
        api, _ = _greeter()

        message = _fails(assert_state, (api, "no-such-conversation"), "greeting")

        assert message.startswith(
            "assert_state: cannot read the state of conversation "
            "'no-such-conversation': "
        )
        assert "snapshot" in message

    @pytest.mark.parametrize("source", [None, "run", 3, ("only-one",), (object(), "x")])
    def test_a_bad_source_is_a_type_error(self, source: Any) -> None:
        with pytest.raises(TypeError, match="ConversationRun or an"):
            assert_state(source, "a")

    def test_a_bare_api_is_not_a_source(self) -> None:
        api, _ = _greeter()

        with pytest.raises(TypeError, match="got API"):
            assert_state(api, "greeting")

    def test_a_state_id_must_be_a_string(self) -> None:
        with pytest.raises(TypeError, match="state_id is a string"):
            assert_state(_driven(), ["farewell"])  # type: ignore[arg-type]

    def test_state_id_may_be_passed_by_keyword(self) -> None:
        assert_state(_driven(), state_id="farewell")


class TestAssertContext:
    def test_present_keys_with_equal_values_pass(self) -> None:
        assert_context(_driven(), mood="happy", intent="chat")

    def test_other_keys_are_ignored(self) -> None:
        assert_context(_driven(), mood="happy")

    def test_a_pair_is_a_source(self) -> None:
        run = _driven()

        assert_context((run.api, run.conversation_id), mood="happy")

    def test_a_wrong_value_names_key_expected_and_actual(self) -> None:
        message = _fails(assert_context, _driven(), mood="sad")

        assert message == "context['mood']: expected 'sad', got 'happy'"

    def test_a_missing_key_lists_the_keys_present(self) -> None:
        message = _fails(assert_context, _driven(), nope=1)

        assert message == "context: key 'nope' is missing (keys: ['intent', 'mood'])"

    def test_the_first_divergence_in_the_order_given_is_reported(self) -> None:
        message = _fails(assert_context, _driven(), mood="happy", intent="x", nope=1)

        assert message == "context['intent']: expected 'x', got 'chat'"

    def test_an_internal_key_is_invisible_so_it_is_missing(self) -> None:
        message = _fails(assert_context, _driven(), _conversation_id="x")

        assert message.startswith("context: key '_conversation_id' is missing")

    def test_a_long_value_is_cut_in_the_message(self) -> None:
        api, _ = _greeter()
        run = ScriptedUser(["hello"]).run(api, initial_context={"blob": "x" * 5000})

        message = _fails(assert_context, run, blob="y" * 5000)

        assert len(message) < 600
        assert message.count("...") == 2

    def test_a_key_may_be_named_like_a_parameter(self) -> None:
        api, _ = _greeter()
        run = ScriptedUser(["hello"]).run(
            api, initial_context={"source": 1, "state_id": 2}
        )

        assert_context(run, source=1, state_id=2)

    def test_naming_no_key_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one key=value"):
            assert_context(_driven())

    def test_a_bad_source_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="ConversationRun or an"):
            assert_context(None, mood="happy")  # type: ignore[arg-type]

    def test_an_unreadable_conversation_fails_with_a_clear_line(self) -> None:
        api, _ = _greeter()

        message = _fails(assert_context, (api, "ghost"), mood="happy")

        assert message.startswith(
            "assert_context: cannot read the data of conversation 'ghost': "
        )


class TestAssertPath:
    def test_an_equal_path_passes_for_a_run(self) -> None:
        assert_path(_driven(), _GREETER_PATH)

    def test_a_list_and_a_tuple_of_state_ids_are_sources(self) -> None:
        assert_path(list(_GREETER_PATH), ["greeting", "conversation", "farewell"])
        assert_path(_GREETER_PATH, _GREETER_PATH)

    def test_the_first_difference_is_named_with_its_index(self) -> None:
        message = _fails(assert_path, ["a", "b", "b"], ["a", "b", "c"])

        assert message == (
            "path: first difference at index 2: expected 'c', got 'b'\n"
            "  expected: a -> b -> c\n"
            "  actual:   a -> b -> b"
        )

    def test_a_run_that_stopped_early_says_which_state_it_missed(self) -> None:
        message = _fails(assert_path, ["a", "b"], ["a", "b", "c"])

        assert message == (
            "path: stopped after 'b' (index 1), expected 'c' next\n"
            "  expected: a -> b -> c\n"
            "  actual:   a -> b"
        )

    def test_a_run_that_went_on_says_which_state_it_added(self) -> None:
        message = _fails(assert_path, ["a", "b", "c"], ["a", "b"])

        assert message == (
            "path: expected it to end at 'b' (index 1), but it went on to 'c'\n"
            "  expected: a -> b\n"
            "  actual:   a -> b -> c"
        )

    def test_an_empty_actual_path_fails_before_the_first_state(self) -> None:
        message = _fails(assert_path, [], ["a"])

        assert message.startswith("path: stopped before the first state")
        assert "actual:   (empty)" in message

    def test_an_empty_expected_path_against_a_longer_one_fails(self) -> None:
        message = _fails(assert_path, ["a"], [])

        assert message.startswith("path: expected an empty path, but it started at 'a'")

    def test_an_empty_path_equals_an_empty_path(self) -> None:
        assert_path([], [])

    def test_a_run_of_the_example_reports_the_difference_in_its_states(self) -> None:
        message = _fails(assert_path, _driven(), ["greeting", "farewell"])

        assert message.startswith(
            "path: first difference at index 1: expected 'farewell', got 'conversation'"
        )

    def test_a_stay_repeats_in_a_runs_path(self) -> None:
        api = _inline(_stuck())
        run = ScriptedUser(["x"]).run(api)

        assert_path(run, ["a", "a"])
        message = _fails(assert_path, run, ["a"])
        assert message.startswith("path: expected it to end at 'a' (index 0)")

    def test_advance_results_are_a_source(self) -> None:
        api = _inline(_chain())
        conversation_id, _ = api.start_conversation()
        steps = api.run_until_terminal(conversation_id, max_steps=5)

        assert_path(steps, ["a", "b", "c"])
        assert_path(list(steps), ["a", "b", "c"])

    def test_the_path_of_a_message_free_run_passes(self) -> None:
        api = _inline(_chain())

        assert_path(advance_until_terminal(api, max_steps=5), ("a", "b", "c"))

    def test_no_steps_is_the_empty_path_not_a_crash(self) -> None:
        assert_path((), [])

    @pytest.mark.parametrize(
        "expected",
        [["greeting", "farewell"], ["conversation"], ["greeting"], []],
    )
    def test_a_subsequence_passes_when_not_exact(self, expected: list[str]) -> None:
        assert_path(_driven(), expected, exact=False)

    def test_a_subsequence_must_keep_the_order(self) -> None:
        message = _fails(assert_path, _driven(), ["farewell", "greeting"], exact=False)

        assert message.startswith(
            "path: expected state 'greeting' (item 1 of the expected path) was not "
            "visited after 'farewell' (index 2)"
        )

    def test_a_missing_first_state_is_reported_in_the_path(self) -> None:
        message = _fails(assert_path, ["a", "b"], ["zzz"], exact=False)

        assert message.startswith(
            "path: expected state 'zzz' (item 0 of the expected path) was not "
            "visited in the path"
        )

    def test_a_repeat_is_matched_once_per_expected_item(self) -> None:
        assert_path(["a", "a", "b"], ["a", "a"], exact=False)
        message = _fails(assert_path, ["a", "b"], ["a", "a"], exact=False)
        assert "item 1" in message

    @pytest.mark.parametrize("expected", ["abc", b"abc", 3, None])
    def test_expected_must_be_a_sequence_of_ids_not_a_string(
        self, expected: Any
    ) -> None:
        with pytest.raises(TypeError, match="expected is a sequence of state ids"):
            assert_path(["a"], expected)

    def test_an_id_in_expected_must_be_a_string(self) -> None:
        with pytest.raises(TypeError, match="a state id is a string"):
            assert_path(["a"], ["a", 1])  # type: ignore[list-item]

    @pytest.mark.parametrize("source", ["abc", None, 7, ["a", 1], [object()]])
    def test_a_bad_source_is_a_type_error(self, source: Any) -> None:
        with pytest.raises(TypeError, match="assert_path"):
            assert_path(source, ["a"])

    def test_exact_is_keyword_only(self) -> None:
        assert inspect.signature(assert_path).parameters["exact"].kind is (
            inspect.Parameter.KEYWORD_ONLY
        )


class TestAssertResult:
    def test_a_completion_result_mapping_passes(self) -> None:
        assert_result({"kind": "final", "text": "ok", "calls": []}, kind="final")
        assert_result(
            {"kind": "calls", "text": "", "calls": [object()]},
            kind=CompletionKind.CALLS,
        )

    def test_a_completion_response_has_a_kind(self) -> None:
        assert_result(final_reply("done"), kind="final")
        assert_result(tool_call_reply(("search", {"q": "x"})), kind="calls")

    def test_a_synthesis_result_passes_status_and_reason(self) -> None:
        result = synthesis_result("failed", "budget_spent")

        assert_result(result, status="failed", reason="budget_spent")
        assert_result(result, status=Status.FAILED, reason=Reason.BUDGET_SPENT)

    def test_the_real_synthesis_result_shape_has_the_keys_the_docs_name(self) -> None:
        assert set(synthesis_result("ok")) == {
            "status",
            "procedure_id",
            "version",
            "reason",
        }

    def test_reason_none_means_no_reason(self) -> None:
        assert_result(synthesis_result("ok"), status="ok", reason=None)
        message = _fails(
            assert_result, synthesis_result("failed", "budget_spent"), reason=None
        )

        assert message == "reason: expected None, got 'budget_spent'"

    def test_a_result_without_a_reason_key_has_no_reason(self) -> None:
        assert_result({"status": "ok"}, status="ok", reason=None)

    def test_reason_left_out_is_not_checked(self) -> None:
        assert_result(synthesis_result("failed", "budget_spent"), status="failed")

    def test_the_default_of_reason_is_a_deterministic_marker_not_none(self) -> None:
        default = inspect.signature(assert_result).parameters["reason"].default

        assert default is not None
        assert repr(default) == "<unset>"

    def test_a_wrong_kind_names_expected_and_actual(self) -> None:
        message = _fails(
            assert_result, {"kind": "calls", "text": "", "calls": []}, kind="final"
        )

        assert message == "kind: expected 'final', got 'calls'"

    def test_a_wrong_status_and_a_wrong_reason_are_reported(self) -> None:
        result = synthesis_result("failed", "budget_spent")

        assert _fails(assert_result, result, status="ok") == (
            "status: expected 'ok', got 'failed'"
        )
        assert _fails(assert_result, result, status="failed", reason="push_failed") == (
            "reason: expected 'push_failed', got 'budget_spent'"
        )

    def test_kind_is_checked_before_status_before_reason(self) -> None:
        result = {"kind": "final", "status": "failed", "reason": "budget_spent"}

        message = _fails(
            assert_result, result, kind="calls", status="ok", reason="push_failed"
        )

        assert message.startswith("kind:")
        message = _fails(assert_result, result, status="ok", reason="push_failed")
        assert message.startswith("status:")

    @pytest.mark.parametrize(
        "field, typo, vocabulary",
        [
            ("kind", "finale", "['calls', 'final', 'malformed']"),
            ("status", "okay", "['ok', 'failed', 'aborted', 'running', 'abstained']"),
            ("reason", "budget", "'budget_spent'"),
        ],
    )
    def test_a_typo_raises_value_error_listing_the_vocabulary(
        self, field: str, typo: str, vocabulary: str
    ) -> None:
        with pytest.raises(ValueError) as raised:
            assert_result({"kind": "final"}, **{field: typo})

        assert typo in str(raised.value)
        assert vocabulary in str(raised.value)

    def test_a_typo_raises_even_when_the_result_would_fail_anyway(self) -> None:
        with pytest.raises(ValueError, match="Invalid completion kind"):
            assert_result(None, kind="finale")  # type: ignore[arg-type]

    def test_a_typo_in_reason_is_caught_when_the_kind_would_pass(self) -> None:
        with pytest.raises(ValueError, match="Invalid reason"):
            assert_result({"kind": "final"}, kind="final", reason="nope")

    def test_naming_nothing_to_check_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one of kind, status or reason"):
            assert_result({"kind": "final"})

    def test_an_actual_code_outside_the_vocabulary_says_so(self) -> None:
        message = _fails(assert_result, {"kind": "weird"}, kind="final")

        assert message == (
            "kind: expected 'final', got 'weird' "
            "(not in the kind vocabulary ['calls', 'final', 'malformed'])"
        )

    def test_a_result_that_was_never_written_fails(self) -> None:
        message = _fails(assert_result, None, status="ok")

        assert message == (
            "result: expected a result mapping, got None (was the result key written?)"
        )

    def test_a_missing_field_names_the_keys_present(self) -> None:
        message = _fails(assert_result, {"kind": "final"}, status="ok")

        assert message == (
            "status: expected 'ok', but the result has no 'status' (keys: ['kind'])"
        )

    def test_a_completion_response_has_no_status_or_reason(self) -> None:
        response = final_reply("done")

        assert _fails(assert_result, response, status="ok") == (
            "status: expected 'ok', but a CompletionResponse has no status"
        )
        assert _fails(assert_result, response, reason="budget_spent") == (
            "reason: expected 'budget_spent', but a CompletionResponse has no reason"
        )
        assert_result(response, kind="final", reason=None)

    def test_a_real_completion_response_object_is_checked_by_kind(self) -> None:
        response = CompletionResponse(kind="final", text="x")

        assert_result(response, kind=CompletionKind.FINAL)
        assert _fails(assert_result, response, kind="calls") == (
            "kind: expected 'calls', got 'final'"
        )

    @pytest.mark.parametrize("result", [[("kind", "final")], "final", 3])
    def test_neither_a_mapping_nor_a_response_is_a_type_error(
        self, result: Any
    ) -> None:
        with pytest.raises(TypeError, match="result mapping"):
            assert_result(result, kind="final")

    def test_kind_status_and_reason_are_keyword_only(self) -> None:
        parameters = inspect.signature(assert_result).parameters
        for name in ("kind", "status", "reason"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


class TestFailuresAreExplicit:
    """No `assert` statement, `__tracebackhide__` everywhere, and `-O` changes nothing."""

    def _tree(self) -> ast.Module:
        return ast.parse(Path(inspect.getfile(assertions_module)).read_text("utf-8"))

    def test_the_module_has_no_assert_statement(self) -> None:
        asserts = [
            n.lineno for n in ast.walk(self._tree()) if isinstance(n, ast.Assert)
        ]

        assert asserts == []

    def test_the_module_does_not_import_pytest(self) -> None:
        names = {
            alias.name.split(".")[0]
            for node in ast.walk(self._tree())
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(self._tree())
            if isinstance(node, ast.ImportFrom)
        }

        assert "pytest" not in names

    def test_every_function_that_raises_an_assertion_error_hides_its_frame(
        self,
    ) -> None:
        tree = self._tree()
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            raises = any(
                isinstance(inner, ast.Raise)
                and "AssertionError" in ast.dump(inner.exc or ast.Pass())
                for inner in ast.walk(node)
            )
            hides = any(
                isinstance(inner, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__tracebackhide__"
                    for t in inner.targets
                )
                for inner in ast.walk(node)
            )
            if raises and not hides:
                offenders.append(node.name)

        assert offenders == []

    @pytest.mark.parametrize(
        "call",
        [
            lambda: assert_state(_driven(), "greeting"),
            lambda: assert_context(_driven(), mood="sad"),
            lambda: assert_path(_driven(), ["greeting"]),
            lambda: assert_path(["a"], ["b"], exact=False),
            lambda: assert_result({"kind": "final"}, kind="calls"),
            lambda: assert_state(
                (_greeter()[0], "ghost"), "greeting"
            ),  # raised inside the reader helper
        ],
        ids=["state", "context", "path", "subsequence", "result", "unreadable"],
    )
    def test_every_frame_in_the_module_carries_the_hide_flag(self, call: Any) -> None:
        with pytest.raises(AssertionError) as raised:
            call()

        module_file = inspect.getfile(assertions_module)
        frames = []
        tb = raised.value.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == module_file:
                frames.append(tb.tb_frame.f_locals.get("__tracebackhide__"))
            tb = tb.tb_next
        assert frames, "the error was not raised inside the module"
        assert all(flag is True for flag in frames)

    def test_the_assertions_still_fail_under_python_dash_O(self) -> None:
        script = (
            "import sys\n"
            "assert_ran = False\n"
            "try:\n"
            "    assert False\n"
            "except AssertionError:\n"
            "    print('assert-statement-active')\n"
            "else:\n"
            "    print('assert-statement-stripped')\n"
            "from fsm_llm import API\n"
            "from fsm_llm.testing import (ScriptedLLM, ScriptedUser, assert_context,\n"
            "    assert_path, assert_result, assert_state)\n"
            "definition = {'name': 'T', 'description': 'd', 'initial_state': 'a',\n"
            "  'persona': 'p', 'states': {'a': {'id': 'a', 'description': 'a',\n"
            "  'purpose': 'p', 'response_instructions': 'say a'}}}\n"
            "api = API.from_definition(definition, llm_interface=ScriptedLLM(\n"
            "    extraction_data={}, default_response='ok'))\n"
            "run = ScriptedUser([]).run(api)\n"
            "calls = [lambda: assert_state(run, 'zzz'),\n"
            "         lambda: assert_context(run, nope=1),\n"
            "         lambda: assert_path(run, ['zzz']),\n"
            "         lambda: assert_result({'kind': 'final'}, kind='calls')]\n"
            "for call in calls:\n"
            "    try:\n"
            "        call()\n"
            "    except AssertionError as error:\n"
            "        print('raised:' + str(error).split(':')[0])\n"
            "    else:\n"
            "        print('SILENT')\n"
        )
        env = {
            **os.environ,
            "OLLAMA_API_BASE": "http://127.0.0.1:9",
            "OLLAMA_HOST": "http://127.0.0.1:9",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
        env.pop("LLM_MODEL", None)

        done = subprocess.run(
            [sys.executable, "-O", "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=_ROOT,
            env=env,
        )

        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == [
            "assert-statement-stripped",
            "raised:state",
            "raised:context",
            "raised:path",
            "raised:kind",
        ]


class _CopyProbe:
    """A context value that counts how many times the pipeline deep-copies it."""

    def __init__(self) -> None:
        self.copies = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> _CopyProbe:
        self.copies += 1
        return self


class TestPathRecorder:
    def test_it_records_the_path_of_the_example_fsm(self) -> None:
        recorder = PathRecorder()
        api, _ = _greeter(handlers=[recorder])

        run = ScriptedUser(["hello", "good, thanks"]).run(api)

        assert recorder.path(run.conversation_id) == _GREETER_PATH
        assert recorder.frames == (run.conversation_id,)
        assert_path(recorder.path(run.conversation_id), run.states)

    def test_it_can_be_registered_after_the_api_was_built(self) -> None:
        recorder = PathRecorder()
        api, _ = _greeter()
        api.register_handler(recorder)

        run = ScriptedUser(["hello"]).run(api)

        assert recorder.path(run.conversation_id) == ("greeting", "conversation")

    def test_a_message_free_run_is_recorded_too(self) -> None:
        recorder = PathRecorder()
        api = _inline(_chain(), handlers=[recorder])

        run = advance_until_terminal(api, max_steps=5)

        assert_path(recorder.path(run.conversation_id), run.states)

    def test_a_stay_is_not_a_transition_so_it_is_not_recorded(self) -> None:
        recorder = PathRecorder()
        api = _inline(_stuck(), handlers=[recorder])

        run = ScriptedUser(["x", "y"]).run(api)

        assert run.states == ("a", "a", "a")  # the run repeats a stay
        assert recorder.path(run.conversation_id) == ("a",)  # the recorder does not

    def test_two_conversations_are_two_frames_in_first_seen_order(self) -> None:
        recorder = PathRecorder()
        api, _ = _greeter(handlers=[recorder])

        first = ScriptedUser(["hello"]).run(api)
        second = ScriptedUser(["hello", "good, thanks"]).run(api)

        assert recorder.frames == (first.conversation_id, second.conversation_id)
        assert len(recorder.path(first.conversation_id)) == 2
        assert len(recorder.path(second.conversation_id)) == 3

    def test_an_unknown_conversation_lists_what_was_recorded(self) -> None:
        recorder = PathRecorder()
        api, _ = _greeter(handlers=[recorder])
        run = ScriptedUser(["hello"]).run(api)

        with pytest.raises(ValueError) as raised:
            recorder.path("typo")

        assert "no path recorded for conversation 'typo'" in str(raised.value)
        assert run.conversation_id in str(raised.value)

    def test_clear_forgets_everything(self) -> None:
        recorder = PathRecorder()
        api, _ = _greeter(handlers=[recorder])
        run = ScriptedUser(["hello"]).run(api)

        recorder.clear()

        assert recorder.frames == ()
        with pytest.raises(ValueError):
            recorder.path(run.conversation_id)

    def test_it_changes_nothing_about_the_conversation(self) -> None:
        plain_api, plain_llm = _greeter()
        recorded_api, recorded_llm = _greeter(handlers=[PathRecorder()])
        messages = ["hello", "good, thanks"]

        plain = ScriptedUser(messages).run(plain_api)
        recorded = ScriptedUser(messages).run(recorded_api)

        assert (recorded.greeting, recorded.replies, recorded.states) == (
            plain.greeting,
            plain.replies,
            plain.states,
        )
        assert recorded_api.get_data(recorded.conversation_id) == plain_api.get_data(
            plain.conversation_id
        )
        assert len(recorded_llm.requests) == len(plain_llm.requests)

    @pytest.mark.parametrize("timing", list(HandlerTiming))
    def test_it_never_asks_to_execute(self, timing: HandlerTiming) -> None:
        recorder = PathRecorder()

        assert recorder.should_execute(timing, "a", "b", {}) is False
        assert recorder.execute({"k": 1}) == {}

    def test_it_is_a_base_handler_that_runs_after_the_tests_own_handlers(self) -> None:
        recorder = PathRecorder(name="mine", priority=5)

        assert isinstance(recorder, BaseHandler)
        assert recorder.name == "mine"
        assert recorder.priority == 5
        assert PathRecorder().priority == 10_000
        assert PathRecorder().name == "PathRecorder"
        assert PathRecorder().critical is False

    def test_a_context_without_a_conversation_id_is_recorded_under_an_empty_id(
        self,
    ) -> None:
        recorder = PathRecorder()

        recorder.should_execute(HandlerTiming.START_CONVERSATION, "a", "a", {})
        recorder.should_execute(HandlerTiming.POST_TRANSITION, "b", "b", {})

        assert recorder.frames == ("",)
        assert recorder.path("") == ("a", "b")

    def test_registering_a_handler_deep_copies_the_context_more_often(self) -> None:
        """The documented side effect: with zero handlers the pipeline skips the copy."""

        def copies(handlers: list[Any] | None) -> int:
            probe = _CopyProbe()
            api, _ = _greeter(handlers=handlers)
            run = ScriptedUser(["hello", "good, thanks"]).run(
                api, initial_context={"probe": probe}
            )
            assert run.states == _GREETER_PATH  # same conversation either way
            return probe.copies

        without = copies(None)
        with_recorder = copies([PathRecorder()])

        assert with_recorder > without

    def test_the_docstring_states_the_deep_copy_side_effect(self) -> None:
        doc = inspect.getdoc(PathRecorder) or ""

        assert "deep-copies the context" in doc
        assert "D-022" in doc
        assert "Interface contract" in doc

    def test_a_non_copyable_context_value_breaks_only_with_a_handler(self) -> None:
        class Locked:
            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                raise TypeError("cannot copy")

        api, _ = _greeter()
        run = ScriptedUser([]).run(api, initial_context={"locked": Locked()})
        assert run.states == ("greeting",)  # fine: no handler, no copy at the timings

        recorded_api, _ = _greeter(handlers=[PathRecorder()])
        # The first deep copy of the context happens at START_CONVERSATION.
        with pytest.raises(Exception) as raised:
            ScriptedUser([]).run(recorded_api, initial_context={"locked": Locked()})
        assert "cannot copy" in str(raised.value) or "Locked" in repr(raised.value)

    def test_a_pushed_fsm_is_its_own_frame(self) -> None:
        recorder = PathRecorder()
        api = _inline(_stuck(), handlers=[recorder])
        conversation_id, _ = api.start_conversation()
        api.push_fsm(conversation_id, _chain())
        child_id = api.get_sub_conversation_id(conversation_id)

        api.advance(conversation_id)  # a step of the pushed FSM: a -> b

        assert child_id != conversation_id
        assert recorder.frames == (conversation_id, child_id)
        assert recorder.path(conversation_id) == ("a",)
        assert recorder.path(child_id) == ("a", "b")


class TestExports:
    @pytest.mark.parametrize(
        "name",
        [
            "assert_state",
            "assert_context",
            "assert_path",
            "assert_result",
            "PathRecorder",
        ],
    )
    def test_public_with_an_interface_contract(self, name: str) -> None:
        assert name in public.__all__
        item = getattr(public, name)
        assert item.__module__ == "fsm_llm.testing.assertions"
        assert "Interface contract" in (inspect.getdoc(item) or "")

    def test_the_source_is_positional_only_everywhere(self) -> None:
        for function in (assert_state, assert_context, assert_path):
            first = next(iter(inspect.signature(function).parameters.values()))
            assert first.name == "source"
            assert first.kind is inspect.Parameter.POSITIONAL_ONLY

    def test_the_sentinel_stays_private(self) -> None:
        for name in ("_UNSET", "_Unset"):
            assert name not in public.__all__
        assert not hasattr(public, "_UNSET")
