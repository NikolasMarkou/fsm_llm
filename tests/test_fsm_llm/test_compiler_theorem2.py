"""
R6.4 — Theorem-2 invariant test for the FSM cohort.

For cohort terminal states emitted by ``compile_fsm`` under
``FSM_LLM_COHORT_EMISSION=1`` (R6.2), the executor's oracle-call count must
equal the planner's predicted-calls value strictly. Each cohort state's body
is exactly one ``Leaf`` after Case dispatch, so the contract simplifies to:

    Executor.run(term, env).oracle_calls == 1 == plan(PlanInputs(n=1, tau=1, ...)).predicted_calls

Non-cohort exclusions (documented):
- States with transitions, field_extractions, classification_extractions,
  required_context_keys, or extraction_instructions stay on the legacy
  host-callback path. Their oracle calls (via classifier, field-extractor,
  etc.) are NOT counted by ``Executor.oracle_calls`` because they fire
  outside the Leaf path. Theorem-2 strict equality is reserved for cohort.
"""

from __future__ import annotations

import pytest

from fsm_llm.dialog.compile_fsm import (
    CB_RESPOND,
    COHORT_RESPONSE_PROMPT_VAR,
    VAR_CONV_ID,
    VAR_INSTANCE,
    VAR_MESSAGE,
    VAR_STATE_ID,
    compile_fsm,
)
from fsm_llm.dialog.definitions import FSMDefinition, State
from fsm_llm.runtime.executor import Executor
from fsm_llm.runtime.planner import PlanInputs, plan


class _ScriptedOracle:
    """Deterministic mock oracle. Each invoke returns the next scripted reply."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.invocations: list[tuple[str, type | None]] = []

    def invoke(self, prompt, schema=None, *, model_override=None, env=None):
        self.invocations.append((prompt, schema))
        if not self._replies:
            raise RuntimeError("scripted oracle exhausted")
        return self._replies.pop(0)

    def tokenize(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def cohort_emission_on(monkeypatch):
    monkeypatch.setenv("FSM_LLM_COHORT_EMISSION", "1")
    from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

    _compile_fsm_by_id.cache_clear()
    yield
    _compile_fsm_by_id.cache_clear()


def _make_cohort_fsm(state_id: str, *, persona: str | None = None) -> FSMDefinition:
    state = State(
        id=state_id,
        description=f"{state_id} desc",
        purpose=f"Purpose of {state_id}",
        response_instructions="Respond appropriately.",
    )
    kwargs = {
        "name": "T2FSM",
        "description": "T2 fixture",
        "initial_state": state_id,
        "states": {state_id: state},
    }
    if persona is not None:
        kwargs["persona"] = persona
    return FSMDefinition(**kwargs)


@pytest.fixture(
    params=[
        ("cohort_min", None, "minimal cohort"),
        ("cohort_persona", "A friendly bot.", "cohort with persona"),
        ("cohort_named", "A serious bot.", "cohort with named state + persona"),
    ],
    ids=lambda p: p[2],
)
def cohort_fsm_fixture(request, cohort_emission_on):
    """Parametrized over ≥3 cohort fixtures (per plan SC4)."""
    state_id, persona, _description = request.param
    return _make_cohort_fsm(state_id, persona=persona)


class TestTheorem2InvariantCohort:
    """`Executor.oracle_calls == plan(...).predicted_calls` for cohort states."""

    def test_oracle_calls_equals_predicted_calls_strict(self, cohort_fsm_fixture):
        defn = cohort_fsm_fixture
        term = compile_fsm(defn)

        # Strip the 4 Abs layers — same pattern as MessagePipeline.process_compiled.
        case_body = term
        for _ in range(4):
            case_body = case_body.body

        from fsm_llm.dialog.definitions import FSMContext, FSMInstance

        instance = FSMInstance(
            fsm_id="t2",
            current_state=defn.initial_state,
            context=FSMContext(),
        )
        env = {
            VAR_STATE_ID: instance.current_state,
            VAR_MESSAGE: "hello",
            VAR_CONV_ID: "conv-1",
            VAR_INSTANCE: instance,
            CB_RESPOND: lambda _i: "would-not-fire-for-cohort",
            COHORT_RESPONSE_PROMPT_VAR: "<rendered>",
        }

        oracle = _ScriptedOracle(["Hello back!"])
        executor = Executor(oracle=oracle)
        result = executor.run(case_body, env)

        assert result == "Hello back!"
        # The cohort Leaf fires exactly once.
        assert executor.oracle_calls == 1
        # Theorem-2: the simplest plan (n=1, tau=1) predicts 1 leaf call.
        predicted = plan(PlanInputs(n=1, tau=1, K=8192))
        assert predicted.predicted_calls == 1
        assert executor.oracle_calls == predicted.predicted_calls

    def test_oracle_invoked_with_substituted_prompt(self, cohort_fsm_fixture):
        """The cohort Leaf substitutes the env binding into its template."""
        defn = cohort_fsm_fixture
        term = compile_fsm(defn)
        case_body = term
        for _ in range(4):
            case_body = case_body.body

        from fsm_llm.dialog.definitions import FSMContext, FSMInstance

        instance = FSMInstance(
            fsm_id="t2",
            current_state=defn.initial_state,
            context=FSMContext(),
        )
        rendered_prompt = "<RENDERED PROMPT FOR COHORT TEST>"
        env = {
            VAR_STATE_ID: instance.current_state,
            VAR_MESSAGE: "hi",
            VAR_CONV_ID: "conv-2",
            VAR_INSTANCE: instance,
            CB_RESPOND: lambda _i: "would-not-fire",
            COHORT_RESPONSE_PROMPT_VAR: rendered_prompt,
        }

        oracle = _ScriptedOracle(["OK"])
        executor = Executor(oracle=oracle)
        executor.run(case_body, env)

        # The oracle saw exactly the substituted prompt.
        assert oracle.invocations == [(rendered_prompt, None)]


class TestTheorem2NonCohortDocumentedExclusion:
    """Non-cohort states do NOT satisfy strict T2 — host callbacks are invisible.

    This isn't a contract violation; it's the documented coverage boundary.
    Non-cohort states keep the legacy host-callback path, and host-side LLM
    calls are NOT counted by Executor.oracle_calls (only Leaf invocations
    increment it). Strict T2 strict equality is reserved for cohort.
    """

    def test_non_cohort_state_does_not_emit_leaf(self, cohort_emission_on):
        from fsm_llm.dialog.definitions import Transition
        from fsm_llm.runtime.ast import App, Let

        sa = State(
            id="a",
            description="a",
            purpose="p",
            transitions=[Transition(target_state="b", description="g")],
        )
        sb = State(id="b", description="b", purpose="p")
        defn = FSMDefinition(
            name="NC",
            description="d",
            initial_state="a",
            states={"a": sa, "b": sb},
        )
        term = compile_fsm(defn)
        body = term
        for _ in range(4):
            body = body.body
        # Non-cohort 'a' → Let (legacy shape).
        assert isinstance(body.branches["a"], Let)
        # Cohort 'b' → Leaf (R6.2 shape).
        from fsm_llm.runtime.ast import Leaf

        assert isinstance(body.branches["b"], Leaf)
        # 'a' doesn't reach a Leaf via its dispatch; its CB_RESPOND in legacy
        # shape would be App, not Leaf — confirming non-cohort is excluded.
        let_value = body.branches["a"].value
        # value should be App over CB_EVAL_TRANSIT (legacy transition path).
        assert isinstance(let_value, App)
