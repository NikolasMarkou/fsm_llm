"""The harness FSM must cost ZERO core Pass-1 LLM calls per turn (D-041).

Why this file exists
--------------------
``fsm_llm/pipeline.py::_execute_data_extraction`` calls
``_bulk_extract_from_instructions`` -- exactly one
``LiteLLMInterface._make_llm_call`` -- from **two** sites:

* the *early* fallback, when a state has ``extraction_instructions`` and NO
  field or classification configs;
* the *additive* pass, when a state has ``extraction_instructions`` and SOME
  configs.  This one is gated on ``has_extraction_instructions and
  (has_field_configs or has_classification_configs)`` and on nothing else --
  in particular NOT on any field being missing.

Between them the two sites cover every state that has ``extraction_instructions``,
so the trigger reduces to ``bool(state.extraction_instructions)`` and the only
package-local way to make it unreachable is for that to be falsy.  Measured on
``ollama_chat/qwen3.5:4b`` before the change: **2.000 core calls per turn**, with
``data_extraction`` on 15/15 turns; after: **1.000**, on 0/15.

Every test here derives the trigger from core's OWN predicate over the REAL FSM
rather than restating it, because a hand-copied predicate is exactly the drift
that would let the call back in silently.  ``test_the_control_arm_fires`` is the
anti-vacuity half: with instructions put back, the same machinery records a real
``_make_llm_call``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.definitions import FSMContext, FSMDefinition, FSMInstance
from fsm_llm.handlers import HandlerSystem
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.transition_evaluator import TransitionEvaluator
from fsm_llm_harness import build_harness_fsm
from fsm_llm_harness.constants import DRIVER_OWNED_SEEDS, ContextKeys, HarnessStates
from tests.conftest import MockLLM2Interface

# ---------------------------------------------------------------------------
# Instrumented LLM: a mock that OWNS `_make_llm_call`, so the bulk fallback
# gets a working method instead of an AttributeError swallowed at
# pipeline.py's `except Exception`.  A mock without it would make every
# assertion below pass for the wrong reason.
# ---------------------------------------------------------------------------


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class CountingLLM(MockLLM2Interface):
    """``MockLLM2Interface`` plus a recording ``_make_llm_call``.

    Interface contract (used by every test in this module):
        - ``raw_calls`` holds one ``call_type`` string per ``_make_llm_call``.
        - the returned payload is a well-formed bulk-extraction reply, so a
          fired fallback SUCCEEDS and merges data -- the failure mode is
          visible as data, not only as a count.
        - ``extract_field`` behaviour is inherited unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_calls: list[str] = []

    def _make_llm_call(
        self,
        messages: list[dict[str, str]],
        call_type: str = "general",
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        self.raw_calls.append(call_type)
        return _Response('{"extracted_data": {"invented_key": "invented_value"}}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fsm(**overrides: Any) -> FSMDefinition:
    """The real harness FSM, optionally with state fields patched in."""
    raw = build_harness_fsm("exercise the harness protocol")
    for state_id, patch in overrides.items():
        raw["states"][state_id].update(patch)
    return FSMDefinition(**raw)


def _pipeline(fsm_def: FSMDefinition, llm: CountingLLM) -> MessagePipeline:
    """A real ``MessagePipeline`` over *fsm_def*, wired to *llm*."""
    return MessagePipeline(
        llm_interface=llm,
        data_extraction_prompt_builder=DataExtractionPromptBuilder(),
        response_generation_prompt_builder=ResponseGenerationPromptBuilder(),
        transition_evaluator=TransitionEvaluator(),
        handler_system=HandlerSystem(),
        fsm_resolver=lambda _fsm_id: fsm_def,
        field_extraction_prompt_builder=FieldExtractionPromptBuilder(),
    )


def _instance(state: str, context: dict[str, Any] | None = None) -> FSMInstance:
    """An ``FSMInstance`` parked in *state* with the driver's own seeds."""
    data = {**DRIVER_OWNED_SEEDS, ContextKeys.GOAL: "a goal"}
    data.update(context or {})
    return FSMInstance(
        fsm_id="iterative_planner_harness",
        current_state=state,
        context=FSMContext(data=data),
    )


def _trigger_terms(fsm_def: FSMDefinition, state_id: str) -> tuple[bool, bool, bool]:
    """Core's own three trigger terms for *state_id*.

    Interface contract (3 call sites in this module):
        - the configs come from ``MessagePipeline._build_field_configs_from_state``
          itself, so ``has_field_configs`` is whatever core would compute --
          including the configs core synthesises from ``required_context_keys``
          and from every transition condition's ``requires_context_keys``.
        - returns ``(has_extraction_instructions, has_field_configs,
          has_classification_configs)``.
        - no I/O, no LLM.
    """
    state = fsm_def.states[state_id]
    configs = MessagePipeline._build_field_configs_from_state(state)
    return (
        bool(state.extraction_instructions),
        bool(configs),
        bool(state.classification_extractions),
    )


# ---------------------------------------------------------------------------
# The trigger, derived from core rather than restated
# ---------------------------------------------------------------------------


class TestBulkExtractionIsUnreachable:
    """Neither bulk-extraction site can fire for any of the six states.

    Both assertions are written as core writes them.  Do NOT "simplify" them to
    ``assert not has_extraction_instructions``: the point is that the two
    branch conditions are transcribed from ``_execute_data_extraction`` and
    evaluated against the real FSM, so if core ever changes a branch this file
    is where the divergence shows up.
    """

    @pytest.mark.parametrize("state_id", HarnessStates.ALL)
    def test_the_additive_site_cannot_fire(self, state_id: str) -> None:
        """``has_extraction_instructions and (configs or classifications)``.

        This is the site D-008 assumed could be defused by making the field
        configs exhaustive.  It cannot: it is not gated on anything being
        missing, so MORE configs make it MORE likely to fire, not less.
        """
        instructions, fields, classifications = _trigger_terms(_fsm(), state_id)
        assert not (instructions and (fields or classifications))

    @pytest.mark.parametrize("state_id", HarnessStates.ALL)
    def test_the_early_site_cannot_fire(self, state_id: str) -> None:
        """``not configs and not classifications`` then ``instructions``.

        CLOSE is the state that reaches this branch (terminal, so no gated
        keys, so no synthesised configs); the other five never do.  Both are
        covered because the parametrisation is over ``HarnessStates.ALL``.
        """
        instructions, fields, classifications = _trigger_terms(_fsm(), state_id)
        assert not (not fields and not classifications and instructions)

    def test_close_is_the_state_that_reaches_the_early_branch(self) -> None:
        """Pins WHICH branch each state takes, so the two tests above are not
        silently testing the same thing six times over.

        If a future edge gives CLOSE a gated key, or removes every gated key
        from another state, this fails and the branch map above is re-read
        rather than assumed.
        """
        fsm_def = _fsm()
        early = {
            state_id
            for state_id in HarnessStates.ALL
            if _trigger_terms(fsm_def, state_id)[1:] == (False, False)
        }
        assert early == {HarnessStates.CLOSE}


class TestNoExtractionCallIsIssued:
    """The behavioural half: run core's real extraction and count the calls."""

    @pytest.mark.parametrize("state_id", HarnessStates.ALL)
    def test_a_turn_in_any_state_issues_no_raw_llm_call(self, state_id: str) -> None:
        """``_execute_data_extraction`` is the whole of Pass 1, and it is free.

        Both cost channels are asserted: ``raw_calls`` (the bulk fallback) and
        ``call_history`` (per-field ``extract_field``).  The second is zero
        because every gated key is in ``DRIVER_OWNED_SEEDS`` and core skips a
        field already present in context -- that is D-044's mechanism, checked
        here for its COST rather than for its security.
        """
        llm = CountingLLM()
        pipeline = _pipeline(_fsm(), llm)

        response = pipeline._execute_data_extraction(
            _instance(state_id), "continue", "conv-1"
        )

        assert llm.raw_calls == []
        assert [name for name, _ in llm.call_history] == []
        assert response.extracted_data == {}

    def test_no_invented_key_can_enter_context(self) -> None:
        """The bulk prompt asked for "descriptive snake_case key names" and
        merged whatever came back.  With the call gone, nothing can arrive.

        The control below shows this same assertion failing when the
        instructions are restored, so it is not vacuous.
        """
        llm = CountingLLM()
        pipeline = _pipeline(_fsm(), llm)

        for state_id in HarnessStates.ALL:
            response = pipeline._execute_data_extraction(
                _instance(state_id), "continue", "conv-1"
            )
            assert "invented_key" not in response.extracted_data

    @pytest.mark.parametrize(
        ("state_id", "expected_site"),
        [
            (HarnessStates.EXPLORE, "additive"),
            (HarnessStates.CLOSE, "early"),
        ],
    )
    def test_the_control_arm_fires(self, state_id: str, expected_site: str) -> None:
        """ANTI-VACUITY. Put the instructions back and the call comes back.

        Both branch sites are exercised: EXPLORE has gated keys (the additive
        site), CLOSE has none (the early site).  Without this test every
        assertion above would pass just as well against a pipeline that never
        extracts at all, which is the false-green shape review C4 punished.
        """
        llm = CountingLLM()
        fsm_def = _fsm(**{state_id: {"extraction_instructions": "Extract a thing."}})
        pipeline = _pipeline(fsm_def, llm)

        response = pipeline._execute_data_extraction(
            _instance(state_id), "continue", "conv-1"
        )

        assert llm.raw_calls == ["data_extraction"], expected_site
        assert response.extracted_data["invented_key"] == "invented_value"


class TestTheHarnessDriverIssuesNoExtractionCall:
    """End-to-end through the REAL driver, not a hand-built pipeline.

    The class above proves the property for a pipeline this file constructed.
    This one proves it for the pipeline ``HarnessAgent.run`` actually builds,
    so a future harness that stops going through ``build_harness_fsm`` (or that
    injects state fields at runtime) cannot pass the unit tests and still pay
    for an extraction call.
    """

    def test_a_whole_run_issues_zero_bulk_extraction_calls(
        self, make_harness, monkeypatch
    ) -> None:
        calls: list[str] = []
        real = MessagePipeline._bulk_extract_from_instructions

        def _spy(self, instance, user_message, state, conversation_id):  # type: ignore[no-untyped-def]
            calls.append(state.id)
            return real(self, instance, user_message, state, conversation_id)

        monkeypatch.setattr(MessagePipeline, "_bulk_extract_from_instructions", _spy)

        harness = make_harness()
        harness.run()

        assert calls == []

    def test_the_spy_would_have_seen_it(self, make_harness, monkeypatch) -> None:
        """ANTI-VACUITY for the run above: the spy is wired correctly.

        ``build_harness_fsm`` is patched at the driver's import site to hand
        every state an ``extraction_instructions`` string, which is the ONLY
        thing the real one no longer does.  The same run then trips the spy.
        """
        import fsm_llm_harness.harness as harness_module

        real_builder = harness_module.build_harness_fsm

        def _instrumented(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raw = real_builder(*args, **kwargs)
            for state in raw["states"].values():
                state["extraction_instructions"] = "Extract a thing."
            return raw

        monkeypatch.setattr(harness_module, "build_harness_fsm", _instrumented)

        calls: list[str] = []
        real = MessagePipeline._bulk_extract_from_instructions

        def _spy(self, instance, user_message, state, conversation_id):  # type: ignore[no-untyped-def]
            calls.append(state.id)
            return real(self, instance, user_message, state, conversation_id)

        monkeypatch.setattr(MessagePipeline, "_bulk_extract_from_instructions", _spy)

        harness = make_harness()
        harness.run()

        assert calls, "the spy never fired: it is not observing the real path"
