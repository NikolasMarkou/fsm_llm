"""Tests for ``fsm_llm_harness.hardening``.

This module is the harness's small-model survival layer, and every one of its
four jobs exists because a MEASURED core behaviour is unavailable or wrong for
``ollama_chat/qwen3.5:4b``.  The tests are written against those measurements,
not against the code's shape:

* :func:`strip_model_noise` strips ``<think>`` and fences UNCONDITIONALLY
  because core's response-generation parser (``llm.py:728-763``) does not,
  while its field-extraction parser (``llm.py:879-888``) does -- the M6
  asymmetry.  So the tests drive the two D-027 traps directly: a TRUNCATED
  ``<think>`` whose payload sits AFTER the unclosed tag must survive, and a
  ``}`` inside a quoted string must not be trimmed away.
* :func:`parse_json_payload` must never raise and must never accept a
  non-object, because its callers turn failure into ``success=False`` and a
  BLOCKED gate rather than into an exception.
* The coercers are the load-bearing half of invariant I8 against a
  soft-comparing JsonLogic gate.  ``test_no_garbage_opens_a_real_gate`` drives
  the REAL ``harness._WORKER_WRITABLE`` table -- not a hand-written stand-in --
  with the exact values that soft comparison lets through (``"3"``, ``3.0``,
  ``True``), so a new gate key added later is covered without anyone
  remembering to extend this file.
* :func:`retry` exists because ``LiteLLMInterface(retries=N)`` is a measured
  no-op for ``ollama_chat/*``.  Its ``sleep`` is injected, so the backoff
  SCHEDULE is asserted as recorded numbers -- a test that merely counts calls
  would pass with the backoff deleted.

Fixture note (``plans/LESSONS.md`` [I:5]): the recorded-sleep and
counting-callable helpers below are deliberately not "a mock that always
succeeds on attempt 2".  ``_Flaky`` is parameterised by how many times it
fails, including "always", so exhaustion is a real case rather than an
unreachable one.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import pytest

from fsm_llm.definitions import LLMResponseError
from fsm_llm_harness import hardening
from fsm_llm_harness.constants import Defaults
from fsm_llm_harness.hardening import (
    RETRYABLE_EXCEPTIONS,
    RoleOutput,
    as_int,
    coerce_worker_output,
    parse_json_payload,
    parse_role_output,
    retry,
    strip_model_noise,
    type_matches,
)
from fsm_llm_harness.harness import _WORKER_WRITABLE

# ---------------------------------------------------------------------------
# Shared hostile inputs
# ---------------------------------------------------------------------------

#: Values that a soft-comparing JsonLogic gate treats as satisfying ``>= 3``
#: (see ``hardening.py``'s D-028 block) but that are NOT the exact type the
#: protocol declares.  Every one of these must be refused by the coercers.
SOFT_TRUE_FOR_THREE: tuple[Any, ...] = ("3", 3.0, True, "99", 99.9)

#: Every non-``str`` shape a 4B reply has actually arrived as, plus the ones a
#: caller could pass by accident.  Used to pin "never raises".
NON_STRINGS: tuple[Any, ...] = (None, 0, 1, 3.5, True, False, [], {}, object(), b"{}")


class _Flaky:
    """A zero-argument callable that fails *failures* times, then succeeds.

    Interface contract (several call sites in ``TestRetry``):
        - ``failures=None`` means fail FOREVER, which is what makes the
          exhaustion path a real case rather than an unreachable one.
        - ``exc_factory`` builds a DISTINCT exception per attempt, so a test
          can assert which one propagated (the LAST, per the contract).
        - ``calls`` counts every invocation, including the failing ones.
    """

    def __init__(
        self,
        failures: int | None,
        *,
        value: Any = "ok",
        exc_type: type[BaseException] = LLMResponseError,
    ) -> None:
        self.failures = failures
        self.value = value
        self.exc_type = exc_type
        self.calls = 0
        self.raised: list[BaseException] = []

    def __call__(self) -> Any:
        self.calls += 1
        if self.failures is None or self.calls <= self.failures:
            error = self.exc_type(f"transient #{self.calls}")
            self.raised.append(error)
            raise error
        return self.value


class _RecordedSleep:
    """A ``sleep`` stand-in recording the exact delays asked for."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# ---------------------------------------------------------------------------
# strip_model_noise
# ---------------------------------------------------------------------------


class TestStripModelNoise:
    """Noise removal is unconditional, lossless for payloads, and total."""

    @pytest.mark.parametrize(
        "raw",
        [
            "<think>reasoning</think>{'x': 1}",
            "<thinking>reasoning</thinking>{'x': 1}",
            "<THINK>reasoning</THINK>{'x': 1}",
            "<think >reasoning</think >{'x': 1}",
            "<think signature=\"abc\">reasoning</think>{'x': 1}",
        ],
    )
    def test_balanced_blocks_are_removed_with_their_content(self, raw: str) -> None:
        """A closed reasoning block is reasoning: tags AND body both go."""
        cleaned = strip_model_noise(raw)
        assert "reasoning" not in cleaned
        assert "think" not in cleaned.lower()
        assert "{'x': 1}" in cleaned

    def test_truncated_think_keeps_the_payload_after_it(self) -> None:
        """D-027 trap 1: a 4B reply routinely truncates BEFORE ``</think>``.

        ``re.sub(r"<think>.*", "", ...)`` -- the obvious simplification -- would
        delete the JSON entirely.  Only the markup may be removed.
        """
        raw = '<think>I should answer\n{"findings_count": 3}'
        cleaned = strip_model_noise(raw)
        assert '{"findings_count": 3}' in cleaned
        assert "<think" not in cleaned

    def test_orphan_closing_tag_keeps_the_payload_before_it(self) -> None:
        """The mirror case: content ahead of a lone ``</think>`` survives."""
        cleaned = strip_model_noise('{"findings_count": 3}</think>')
        assert '{"findings_count": 3}' in cleaned
        assert "think" not in cleaned.lower()

    def test_no_prose_trimming_so_a_braced_string_survives(self) -> None:
        """D-027 trap 2: this function must NOT locate the JSON span.

        A naive "strip everything before the first ``{``/after the last ``}``"
        mis-handles a ``}`` inside a quoted string.  Locating the span is
        ``extract_json_from_text``'s job (the repo's only string-aware brace
        scanner), so the prose must still be here afterwards.
        """
        raw = 'Here is my answer: {"note": "a } brace"} -- hope that helps'
        cleaned = strip_model_noise(raw)
        assert cleaned.startswith("Here is my answer:")
        assert cleaned.endswith("hope that helps")
        assert '{"note": "a } brace"}' in cleaned

    @pytest.mark.parametrize(
        "fence", ["```", "```json", "```JSON", "```python", "```j_s-on1"]
    )
    def test_fence_markers_are_removed(self, fence: str) -> None:
        cleaned = strip_model_noise(f'{fence}\n{{"a": 1}}\n```')
        assert "```" not in cleaned
        assert cleaned == '{"a": 1}'

    def test_fence_language_hint_does_not_eat_the_payload(self) -> None:
        """The language hint is bounded; a fence butted against JSON is safe."""
        assert strip_model_noise('```json{"a": 1}```') == '{"a": 1}'

    def test_result_is_stripped(self) -> None:
        assert strip_model_noise('   \n {"a": 1} \n  ') == '{"a": 1}'

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "plain text",
            '<think>a</think>```json\n{"a": 1}\n```',
            "<think>unclosed",
            "</think>",
            '{"note": "<think>quoted</think>"}',
            "```" * 5,
        ],
    )
    def test_is_idempotent(self, raw: str) -> None:
        once = strip_model_noise(raw)
        assert strip_model_noise(once) == once

    @pytest.mark.parametrize("value", NON_STRINGS)
    def test_non_string_yields_empty_string(self, value: Any) -> None:
        assert strip_model_noise(value) == ""

    def test_leaves_clean_json_untouched(self) -> None:
        assert strip_model_noise('{"a": 1}') == '{"a": 1}'

    def test_strip_is_lossy_inside_quoted_strings_and_that_is_known(self) -> None:
        """PINNED, not endorsed: removal is textual, so quoted markup dies.

        A payload whose own VALUE contains ``<think>`` or a fence comes back
        with that value blanked.  This is a real cost of stripping before
        parsing, and it is pinned here so a future change to it is a decision
        rather than an accident.  It cannot open a gate: every gate key is an
        ``int`` or ``bool`` and this only damages ``str`` values.
        """
        assert strip_model_noise('{"note": "<think>x</think>"}') == '{"note": " "}'
        assert strip_model_noise('{"note": "```json"}') == '{"note": " "}'

    @pytest.mark.parametrize(
        "raw",
        [
            "<" * 200,
            "<think>" * 50,
            "```" * 50 + "{",
            "\x00\x01<think>\x02</think>",
            "<think>\n" + "x" * 5000 + "\n</think>{}",
        ],
    )
    def test_never_raises_on_hostile_input(self, raw: str) -> None:
        assert isinstance(strip_model_noise(raw), str)


# ---------------------------------------------------------------------------
# parse_json_payload
# ---------------------------------------------------------------------------


class TestParseJsonPayload:
    """Recovery is delegated; the CONTRACT (object-or-None, never raise) is here."""

    @pytest.mark.parametrize(
        "raw",
        [
            '{"findings_count": 3}',
            '```json\n{"findings_count": 3}\n```',
            '<think>let me think</think>{"findings_count": 3}',
            '<think>truncated\n{"findings_count": 3}',
            'Here you go: {"findings_count": 3} -- done.',
            '```\n{"findings_count": 3}\n```\nHope that helps!',
            '<thinking>x</thinking>\n```json\n{"findings_count": 3}\n```\n',
        ],
    )
    def test_recovers_the_object_from_every_measured_noise_shape(
        self, raw: str
    ) -> None:
        assert parse_json_payload(raw) == {"findings_count": 3}

    def test_nested_object_with_a_brace_in_a_string(self) -> None:
        """Strategy 3's string-aware scanner is why this works at all."""
        assert parse_json_payload('{"a": {"b": "}"}}') == {"a": {"b": "}"}}

    def test_first_object_wins_on_a_multi_object_reply(self) -> None:
        """A 4B model that emits a draft then a final answer loses the final.

        This is the D-031 failure the ``roles.py`` observer WARNS about; the
        parser's own behaviour is first-wins and is pinned here so the warning
        and the behaviour cannot drift apart.
        """
        assert parse_json_payload('{"a": 1} {"b": 2}') == {"a": 1}

    def test_a_real_payload_beats_one_inside_a_reasoning_block(self) -> None:
        """Cleaned text is tried FIRST, so reasoning never shadows the answer."""
        raw = '<think>{"draft": 1}</think>{"final": 2}'
        assert parse_json_payload(raw) == {"final": 2}

    def test_reasoning_block_json_is_the_last_resort_only(self) -> None:
        """PINNED: with NO payload outside it, the block's JSON is recovered.

        ``parse_json_payload`` retries the RAW text when the cleaned text has
        no object, so a reply that put its only JSON inside ``<think>`` is
        still usable.  The preceding test proves this never outranks a real
        answer.
        """
        assert parse_json_payload('<think>{"findings_count": 3}</think>') == {
            "findings_count": 3
        }

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   \n\t ",
            "<think>only reasoning, no json</think>",
            "```json\n```",
            "I could not complete the task.",
            '{"a": 1',
            "{'a': 1,",
            "null",
        ],
    )
    def test_unrecoverable_input_yields_none(self, raw: str) -> None:
        assert parse_json_payload(raw) is None

    @pytest.mark.parametrize("raw", ["[1, 2, 3]", '["a"]', '"a string"', "42", "true"])
    def test_a_non_object_top_level_is_not_a_payload(self, raw: str) -> None:
        """The protocol only ever exchanges objects; a list is not one."""
        assert parse_json_payload(raw) is None

    @pytest.mark.parametrize("value", NON_STRINGS)
    def test_non_string_yields_none(self, value: Any) -> None:
        assert parse_json_payload(value) is None

    def test_empty_object_is_a_payload_not_a_failure(self) -> None:
        """``{}`` parsed fine; it simply satisfies no expected key."""
        assert parse_json_payload("{}") == {}

    @pytest.mark.parametrize(
        "raw",
        [
            "{" * 200,
            "}" * 200,
            '{"a": "' + "x" * 10000,
            "\\" * 100,
            '{"a": "\\u"}',
            "<think>{</think>}",
        ],
    )
    def test_never_raises_on_hostile_input(self, raw: str) -> None:
        result = parse_json_payload(raw)
        assert result is None or isinstance(result, dict)

    def test_a_ladder_exception_is_swallowed_not_propagated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parse failure is a VALUE for every caller, even if core regresses.

        The guard around ``extract_json_from_text`` is unreachable today, which
        is exactly why it needs a test: a future core change that starts
        raising must not turn a gate decision into a crashed run.
        """

        def _boom(_text: str) -> Any:
            raise RuntimeError("core regressed")

        monkeypatch.setattr(hardening, "extract_json_from_text", _boom)
        assert parse_json_payload('{"a": 1}') is None

    @pytest.mark.parametrize("raw", ["", "   ", "\n\t\r "])
    def test_blank_input_short_circuits_before_the_ladder(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty reply is answered here, not by four recovery strategies.

        The result is the same either way today, so this pins the DECISION
        rather than the value: the commonest 4B failure (a timed-out, empty
        reply) must not cost a walk down core's recovery ladder, and it must
        not depend on how core happens to treat ``""``.
        """
        calls: list[str] = []
        monkeypatch.setattr(
            hardening,
            "extract_json_from_text",
            lambda text: calls.append(text) or None,
        )

        assert parse_json_payload(raw) is None
        assert calls == []


# ---------------------------------------------------------------------------
# parse_role_output
# ---------------------------------------------------------------------------


class TestParseRoleOutput:
    """The worker seam: every failure is typed, and missing keys fail CLOSED."""

    def test_none_is_an_empty_reply(self) -> None:
        out = parse_role_output(None)
        assert out.success is False
        assert out.failure_reason == "empty-reply"
        assert out.payload == {}
        assert out.missing_keys == ()

    @pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
    def test_blank_text_is_an_empty_reply(self, raw: str) -> None:
        assert parse_role_output(raw).failure_reason == "empty-reply"

    def test_unparseable_text_is_typed_as_such(self) -> None:
        out = parse_role_output("I am sorry, I cannot help with that.")
        assert out.success is False
        assert out.failure_reason == "unparseable"
        assert out.payload == {}

    def test_a_mapping_reply_bypasses_the_parser(self) -> None:
        """An already-structured worker result is taken as-is."""
        out = parse_role_output({"findings_count": 3}, expected_keys=["findings_count"])
        assert out.success is True
        assert out.payload == {"findings_count": 3}

    def test_a_mapping_reply_is_copied_not_aliased(self) -> None:
        source = {"findings_count": 3}
        out = parse_role_output(source)
        source["findings_count"] = 99
        assert out.payload == {"findings_count": 3}

    def test_an_empty_mapping_is_not_an_empty_reply(self) -> None:
        """``{}`` is a structured answer that happens to satisfy nothing."""
        out = parse_role_output({}, expected_keys=["findings_count"])
        assert out.failure_reason == "missing-keys:findings_count"

    def test_a_non_string_non_mapping_is_stringified_as_a_last_resort(self) -> None:
        class _Weird:
            def __str__(self) -> str:
                return '{"findings_count": 3}'

        out = parse_role_output(_Weird(), expected_keys=["findings_count"])
        assert out.success is True
        assert out.payload == {"findings_count": 3}

    def test_a_stringified_object_with_no_json_is_unparseable(self) -> None:
        assert parse_role_output(object()).failure_reason == "unparseable"

    def test_missing_keys_fail_closed_and_are_named(self) -> None:
        out = parse_role_output(
            '{"findings_count": 3}',
            expected_keys=["findings_count", "needs_explore", "message"],
        )
        assert out.success is False
        assert out.missing_keys == ("needs_explore", "message")
        assert out.failure_reason == "missing-keys:needs_explore,message"
        assert out.payload == {"findings_count": 3}, "the payload is still reported"

    def test_missing_key_order_follows_expected_keys(self) -> None:
        out = parse_role_output("{}", expected_keys=["b", "a", "c"])
        assert out.missing_keys == ("b", "a", "c")

    def test_no_expected_keys_means_any_object_succeeds(self) -> None:
        out = parse_role_output('{"whatever": 1}')
        assert out.success is True
        assert out.failure_reason is None

    def test_a_present_but_null_key_counts_as_present(self) -> None:
        """PINNED layering: presence is this layer's job, TYPE is the next one.

        ``{"findings_count": null}`` passes here and is then DROPPED by
        :func:`coerce_worker_output`, leaving the gate closed.  The assertion
        below drives BOTH layers so "success here" can never be mistaken for
        "the gate opens".
        """
        out = parse_role_output(
            '{"findings_count": null}', expected_keys=["findings_count"]
        )
        assert out.success is True
        assert coerce_worker_output(out.payload, {"findings_count": int}) == {}

    def test_expected_keys_may_be_any_iterable(self) -> None:
        out = parse_role_output('{"a": 1}', expected_keys=(k for k in ("a", "b")))
        assert out.missing_keys == ("b",)

    @pytest.mark.parametrize("raw", [None, "", "junk", {}, 0, 3.5, [], object()])
    def test_never_raises(self, raw: Any) -> None:
        assert isinstance(parse_role_output(raw, expected_keys=["a"]), RoleOutput)

    def test_role_output_is_frozen(self) -> None:
        out = RoleOutput()
        with pytest.raises(Exception):
            out.success = True  # type: ignore[misc]

    def test_role_output_defaults_are_a_failure(self) -> None:
        """The zero value must be the SAFE value, not an open gate."""
        out = RoleOutput()
        assert out.success is False
        assert out.payload == {}
        assert out.failure_reason is None
        assert out.missing_keys == ()


# ---------------------------------------------------------------------------
# type_matches / as_int
# ---------------------------------------------------------------------------


class TestTypeMatches:
    """Exactness, in both directions -- the two traps D-028 names."""

    def test_bool_does_not_satisfy_int(self) -> None:
        """``isinstance(True, int)`` is True in Python; this must not be."""
        assert type_matches(True, int) is False
        assert type_matches(False, int) is False

    def test_int_does_not_satisfy_bool(self) -> None:
        assert type_matches(1, bool) is False
        assert type_matches(0, bool) is False

    def test_bool_satisfies_bool_and_int_satisfies_int(self) -> None:
        assert type_matches(True, bool) is True
        assert type_matches(3, int) is True

    @pytest.mark.parametrize("value", ["3", 3.0, None, [], {}, object()])
    def test_lookalikes_do_not_satisfy_int(self, value: Any) -> None:
        assert type_matches(value, int) is False

    @pytest.mark.parametrize("value", ["true", 1, 0.0, None, "yes"])
    def test_lookalikes_do_not_satisfy_bool(self, value: Any) -> None:
        assert type_matches(value, bool) is False

    def test_str_is_ordinary_isinstance(self) -> None:
        assert type_matches("x", str) is True
        assert type_matches(3, str) is False

    def test_subclasses_of_other_types_are_accepted(self) -> None:
        class MyStr(str):
            pass

        assert type_matches(MyStr("x"), str) is True

    def test_float_is_exact_too(self) -> None:
        assert type_matches(3.0, float) is True
        assert type_matches(3, float) is False

    def test_a_non_class_expected_raises(self) -> None:
        """A programmer error, not model noise -- it must NOT be swallowed."""
        with pytest.raises(TypeError):
            type_matches(3, "int")  # type: ignore[arg-type]


class TestAsInt:
    """The counter coercer: a miss returns the caller's default, never a guess."""

    @pytest.mark.parametrize("value", [0, 3, -7, 10**18])
    def test_exact_ints_pass_through(self, value: int) -> None:
        assert as_int(value, -1) == value

    @pytest.mark.parametrize(
        "value", [*SOFT_TRUE_FOR_THREE, None, [], {}, "", object()]
    )
    def test_everything_else_returns_the_default(self, value: Any) -> None:
        assert as_int(value, -1) == -1

    def test_the_default_is_returned_identically(self) -> None:
        sentinel = 4242
        assert as_int("3", sentinel) is sentinel

    def test_bool_is_refused_even_though_it_is_an_int(self) -> None:
        assert as_int(True, 0) == 0
        assert as_int(False, 5) == 5


# ---------------------------------------------------------------------------
# coerce_worker_output
# ---------------------------------------------------------------------------


class TestCoerceWorkerOutput:
    """The allowlist filter: drop on ANY doubt, so the gate stays closed."""

    def test_allowlisted_keys_of_the_right_type_pass(self) -> None:
        accepted = coerce_worker_output(
            {"findings_count": 3, "needs_explore": False},
            {"findings_count": int, "needs_explore": bool},
        )
        assert accepted == {"findings_count": 3, "needs_explore": False}

    def test_unknown_keys_are_dropped(self) -> None:
        accepted = coerce_worker_output(
            {"findings_count": 3, "plan_dir": "/etc", "all_criteria_pass": True},
            {"findings_count": int},
        )
        assert accepted == {"findings_count": 3}

    def test_absent_keys_are_simply_absent(self) -> None:
        accepted = coerce_worker_output({}, {"findings_count": int})
        assert accepted == {}

    @pytest.mark.parametrize("value", SOFT_TRUE_FOR_THREE)
    def test_soft_comparable_lookalikes_are_dropped(self, value: Any) -> None:
        """These all satisfy ``>= 3`` in ``fsm_llm.expressions``; not here."""
        accepted = coerce_worker_output(
            {"findings_count": value}, {"findings_count": int}
        )
        assert accepted == {}

    def test_a_null_value_is_dropped_not_written_as_none(self) -> None:
        accepted = coerce_worker_output(
            {"findings_count": None}, {"findings_count": int}
        )
        assert accepted == {}

    def test_an_empty_allowlist_accepts_nothing(self) -> None:
        """EXECUTE's table is empty on purpose: its worker writes no gate flag."""
        assert coerce_worker_output({"findings_count": 3}, {}) == {}

    @pytest.mark.parametrize("payload", [None, "text", 3, [], object()])
    def test_a_non_mapping_payload_yields_empty(self, payload: Any) -> None:
        assert coerce_worker_output(payload, {"findings_count": int}) == {}

    def test_the_result_is_a_new_plain_dict(self) -> None:
        payload = {"findings_count": 3}
        accepted = coerce_worker_output(payload, {"findings_count": int})
        assert accepted is not payload
        assert type(accepted) is dict
        accepted["findings_count"] = 99
        assert payload["findings_count"] == 3

    def test_a_drop_is_logged_with_the_caller_label(self, captured_logs: Any) -> None:
        coerce_worker_output(
            {"findings_count": "3"}, {"findings_count": int}, where="explorer"
        )
        drops = [line for line in captured_logs if "Dropping it" in line]
        assert len(drops) == 1
        assert "explorer" in drops[0]
        assert "findings_count='3'" in drops[0]
        assert "str" in drops[0] and "int" in drops[0]

    def test_an_accepted_key_is_not_logged(self, captured_logs: Any) -> None:
        coerce_worker_output({"findings_count": 3}, {"findings_count": int})
        assert not [line for line in captured_logs if "Dropping it" in line]

    def test_one_bad_key_does_not_discard_the_good_ones(self) -> None:
        accepted = coerce_worker_output(
            {"findings_count": "3", "needs_explore": True},
            {"findings_count": int, "needs_explore": bool},
        )
        assert accepted == {"needs_explore": True}

    def test_a_mapping_subclass_payload_is_accepted(self) -> None:
        from types import MappingProxyType

        accepted = coerce_worker_output(
            MappingProxyType({"findings_count": 3}), {"findings_count": int}
        )
        assert accepted == {"findings_count": 3}


class TestNoGarbageOpensARealGate:
    """Driven by the REAL ``harness._WORKER_WRITABLE`` table, not a stand-in.

    Deriving the cases from the shipping table is what makes a gate key added
    by a later step covered here automatically -- the hand-listed copy is the
    drift that let review C1 (nine keys, diagnosed as one) through.
    """

    def test_the_table_is_not_vacuous(self) -> None:
        """Guard against the whole class passing because the table went empty."""
        keys = {key for scope in _WORKER_WRITABLE.values() for key in scope}
        assert len(keys) >= 8
        assert any(t is int for s in _WORKER_WRITABLE.values() for t in s.values())
        assert any(t is bool for s in _WORKER_WRITABLE.values() for t in s.values())

    @pytest.mark.parametrize("state", sorted(_WORKER_WRITABLE))
    @pytest.mark.parametrize("garbage", [*SOFT_TRUE_FOR_THREE, None, [], {}, "yes"])
    def test_no_garbage_survives_for_any_state(self, state: str, garbage: Any) -> None:
        allowlist: Mapping[str, type] = _WORKER_WRITABLE[state]
        payload = {key: garbage for key in allowlist}
        accepted = coerce_worker_output(payload, allowlist, where=state)
        for key, expected in allowlist.items():
            if key in accepted:
                assert type_matches(accepted[key], expected), (
                    f"{state}.{key} accepted {garbage!r} for {expected.__name__}"
                )

    @pytest.mark.parametrize("state", sorted(_WORKER_WRITABLE))
    def test_an_honest_payload_still_gets_through(self, state: str) -> None:
        """The mirror of the above: this class must not pass by rejecting all."""
        allowlist: Mapping[str, type] = _WORKER_WRITABLE[state]
        honest = {key: expected() for key, expected in allowlist.items()}
        accepted = coerce_worker_output(honest, allowlist, where=state)
        assert set(accepted) == set(allowlist)


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


class TestRetry:
    """Backoff, exhaustion, and the strict allowlist that keeps 4B runs short."""

    def test_a_first_attempt_success_never_sleeps(self) -> None:
        fn = _Flaky(0)
        sleep = _RecordedSleep()
        assert retry(fn, sleep=sleep) == "ok"
        assert fn.calls == 1
        assert sleep.delays == []

    def test_it_retries_until_it_succeeds(self) -> None:
        fn = _Flaky(2)
        sleep = _RecordedSleep()
        assert retry(fn, attempts=3, sleep=sleep) == "ok"
        assert fn.calls == 3
        assert len(sleep.delays) == 2

    def test_the_backoff_schedule_is_exponential_and_recorded(self) -> None:
        """The DELAYS are asserted, not just their count.

        A test that only counted sleeps would still pass with the backoff
        deleted and ``sleep(0)`` in its place.
        """
        fn = _Flaky(None)
        sleep = _RecordedSleep()
        with pytest.raises(LLMResponseError):
            retry(
                fn,
                attempts=5,
                base_delay=1.0,
                backoff_factor=2.0,
                max_delay=30.0,
                sleep=sleep,
            )
        assert sleep.delays == [1.0, 2.0, 4.0, 8.0]

    def test_the_delay_is_capped_at_max_delay(self) -> None:
        fn = _Flaky(None)
        sleep = _RecordedSleep()
        with pytest.raises(LLMResponseError):
            retry(
                fn,
                attempts=6,
                base_delay=1.0,
                backoff_factor=10.0,
                max_delay=5.0,
                sleep=sleep,
            )
        assert sleep.delays == [1.0, 5.0, 5.0, 5.0, 5.0]

    def test_a_non_default_base_delay_scales_every_step(self) -> None:
        fn = _Flaky(None)
        sleep = _RecordedSleep()
        with pytest.raises(LLMResponseError):
            retry(fn, attempts=4, base_delay=0.5, backoff_factor=3.0, sleep=sleep)
        assert sleep.delays == [0.5, 1.5, 4.5]

    def test_exhaustion_reraises_the_LAST_exception(self) -> None:
        fn = _Flaky(None)
        with pytest.raises(LLMResponseError) as excinfo:
            retry(fn, attempts=3, sleep=lambda _s: None)
        assert fn.calls == 3
        assert excinfo.value is fn.raised[-1]
        assert "transient #3" in str(excinfo.value)

    def test_the_last_attempt_is_not_followed_by_a_sleep(self) -> None:
        fn = _Flaky(None)
        sleep = _RecordedSleep()
        with pytest.raises(LLMResponseError):
            retry(fn, attempts=3, sleep=sleep)
        assert len(sleep.delays) == 2, "attempts-1 sleeps, never attempts"

    def test_attempts_of_one_disables_retrying(self) -> None:
        fn = _Flaky(None)
        sleep = _RecordedSleep()
        with pytest.raises(LLMResponseError):
            retry(fn, attempts=1, sleep=sleep)
        assert fn.calls == 1
        assert sleep.delays == []

    @pytest.mark.parametrize("attempts", [0, -1, -100])
    def test_a_non_positive_attempts_is_a_programmer_error(self, attempts: int) -> None:
        with pytest.raises(ValueError, match="attempts must be >= 1"):
            retry(_Flaky(0), attempts=attempts, sleep=lambda _s: None)

    def test_a_non_retryable_exception_propagates_from_the_first_attempt(self) -> None:
        """A deterministic fault cannot succeed on attempt 2; do not burn 3x.

        On a 4B model each attempt is tens of seconds, so this is a wall-clock
        decision, not a stylistic one.
        """
        fn = _Flaky(None, exc_type=ValueError)
        sleep = _RecordedSleep()
        with pytest.raises(ValueError):
            retry(fn, attempts=5, sleep=sleep)
        assert fn.calls == 1
        assert sleep.delays == []

    @pytest.mark.parametrize("exc_type", list(RETRYABLE_EXCEPTIONS))
    def test_every_default_retryable_type_is_actually_retried(
        self, exc_type: type[BaseException]
    ) -> None:
        fn = _Flaky(1, exc_type=exc_type)
        assert retry(fn, attempts=2, sleep=lambda _s: None) == "ok"
        assert fn.calls == 2

    def test_the_retryable_set_is_the_transport_boundary(self) -> None:
        assert RETRYABLE_EXCEPTIONS == (
            LLMResponseError,
            TimeoutError,
            ConnectionError,
        )

    def test_retry_on_is_a_strict_allowlist(self) -> None:
        fn = _Flaky(1, exc_type=KeyError)
        assert (
            retry(fn, attempts=2, retry_on=(KeyError,), sleep=lambda _s: None) == "ok"
        )
        narrowed = _Flaky(None, exc_type=LLMResponseError)
        with pytest.raises(LLMResponseError):
            retry(narrowed, attempts=3, retry_on=(KeyError,), sleep=lambda _s: None)
        assert narrowed.calls == 1, "narrowing retry_on must EXCLUDE the default set"

    def test_a_subclass_of_a_retryable_type_is_retried(self) -> None:
        class Flaky502(ConnectionError):
            pass

        fn = _Flaky(1, exc_type=Flaky502)
        assert retry(fn, attempts=2, sleep=lambda _s: None) == "ok"

    def test_the_return_value_is_passed_through_unchanged(self) -> None:
        marker = object()
        assert retry(_Flaky(1, value=marker), attempts=2, sleep=lambda _s: None) is (
            marker
        )

    def test_a_falsy_return_value_is_not_mistaken_for_a_failure(self) -> None:
        for value in (None, 0, "", False, []):
            assert retry(_Flaky(0, value=value), sleep=lambda _s: None) == value

    def test_the_description_labels_both_the_warning_and_the_error(
        self, captured_logs: Any
    ) -> None:
        with pytest.raises(LLMResponseError):
            retry(
                _Flaky(None),
                attempts=2,
                sleep=lambda _s: None,
                description="explorer dispatch",
            )
        warnings = [line for line in captured_logs if line.startswith("WARNING|")]
        errors = [line for line in captured_logs if line.startswith("ERROR|")]
        assert any("explorer dispatch failed on attempt 1/2" in w for w in warnings)
        assert any("retrying in" in w for w in warnings)
        assert any("explorer dispatch failed after 2 attempts" in e for e in errors)

    def test_the_defaults_come_from_the_package_constants(self) -> None:
        """No second copy of the retry policy lives in this module."""
        fn = _Flaky(None)
        sleep = _RecordedSleep()
        with pytest.raises(LLMResponseError):
            retry(fn, sleep=sleep)
        assert fn.calls == Defaults.RETRY_ATTEMPTS
        expected = [
            min(
                Defaults.RETRY_BASE_DELAY * Defaults.RETRY_BACKOFF_FACTOR ** (n - 1),
                Defaults.RETRY_MAX_DELAY,
            )
            for n in range(1, Defaults.RETRY_ATTEMPTS)
        ]
        assert sleep.delays == expected

    def test_the_default_sleep_is_real_time_sleep(self) -> None:
        """``sleep`` is injected FOR TESTS; production must still wait."""
        import time as _time

        assert retry.__kwdefaults__["sleep"] is _time.sleep


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    """What this module deliberately does NOT own (D-027 / D-029 / D-030 / D-059)."""

    def test_all_is_sorted_and_matches_the_public_names(self) -> None:
        assert hardening.__all__ == sorted(hardening.__all__)
        public = {
            name
            for name, value in vars(hardening).items()
            if not name.startswith("_")
            and getattr(value, "__module__", None) == hardening.__name__
            and not isinstance(value, TypeVar)
        }
        assert public == set(hardening.__all__) - {"RETRYABLE_EXCEPTIONS"}
        assert "RETRYABLE_EXCEPTIONS" in vars(hardening)

    @pytest.mark.parametrize(
        "name",
        [
            "build_response_format",
            "coerce_int",
            "coerce_bool",
            "coerce_str",
            "prepare_ollama_call",
            "extract_json",
        ],
    )
    def test_the_deleted_helpers_stay_deleted(self, name: str) -> None:
        """Each of these was a call-site-free duplicate; D-030/D-059 removed them.

        ``build_response_format`` in particular asserted two call sites in its
        own docstring and had zero -- ``AgentConfig.output_schema`` makes
        ``BaseAgent._init_context`` build the identical envelope.
        """
        assert not hasattr(hardening, name)

    def test_no_second_brace_scanner_lives_here(self) -> None:
        """The repo's only string-aware scanner is ``_match_brace_partners``.

        A ``{``-indexing loop appearing in this module is the Complexity-Budget
        BREACH condition D-027 names, so it is checked mechanically -- against
        the CODE, not the prose, so the D-027 comment explaining the ban does
        not itself trip the check.
        """
        calls = _module_code_names()
        assert not {"index", "find", "rfind"} & calls
        assert "json" not in calls, "the recovery ladder is core's, not ours"

    def test_the_recovery_ladder_is_delegated_to_core(self) -> None:
        assert "extract_json_from_text" in _module_code_names()

    def test_no_second_ollama_thinking_suppressor_lives_here(self) -> None:
        """D-030: ``LiteLLMInterface._make_llm_call`` already does this."""
        names = _module_code_names()
        assert "reasoning_effort" not in names
        assert "prepare_ollama_messages" not in names
        assert "apply_ollama_params" not in names
        assert not [s for s in _module_code_strings() if "nothink" in s]


def _module_ast() -> ast.Module:
    """``hardening.py`` parsed from its own ``__file__``.

    Interface contract (2 call sites below): reads the SHIPPING file so the
    assertions cannot drift onto a stale copy.  Comments are dropped by the
    parser, which is the point -- these checks are about code, and the module's
    D-027/D-030 comments legitimately quote the very things they forbid.
    """
    return ast.parse(Path(hardening.__file__).read_text(encoding="utf-8"))


def _module_code_names() -> set[str]:
    """Every identifier, attribute and keyword-argument name in the module."""
    names: set[str] = set()
    for node in ast.walk(_module_ast()):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.name for alias in node.names)
    return names


def _module_code_strings() -> list[str]:
    """Every string literal in the module EXCEPT the docstrings.

    One parse, reused for both walks: two parses would give two node
    identities, and the docstring exclusion would silently never match.
    """
    tree = _module_ast()
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]
