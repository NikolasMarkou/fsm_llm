"""Shape-equivalence + smoke tests for stdlib/agents/lam_factories.py (M3 slice 1).

Each factory must produce a λ-term whose structural skeleton matches the
inline ``build_term()`` from the corresponding M4 reference example.
Prompt-string content and env-var literals are factory parameters, so we
intentionally compare *structure* (kind tree + let-binding names +
var-name references + leaf input_vars), not text.

The smoke tests are opt-in via ``TEST_REAL_LLM=1`` + ``LLM_MODEL`` and run
on ``ollama_chat/qwen3.5:4b`` (per project policy).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from fsm_llm.runtime import Term, app, leaf, let, var
from fsm_llm.stdlib.agents import (
    memory_term,
    react_term,
    reflexion_term,
    rewoo_term,
)

# ---------------------------------------------------------------------------
# Shape comparator
# ---------------------------------------------------------------------------


def _term_shape_eq(a: Term, b: Term) -> bool:
    """Compare two Terms by structural skeleton.

    Compares: ``kind`` recursively; ``Let.name``; ``Var.name``;
    ``Leaf.input_vars`` (as set — order-insensitive). Does NOT compare
    leaf templates, schema_refs, or model_overrides — those are
    factory parameters, not part of the shape.
    """
    if a.kind != b.kind:
        return False
    if a.kind == "Leaf":
        # Compare input_vars only — the *role* of the leaf in the term.
        return tuple(a.input_vars) == tuple(b.input_vars)
    if a.kind == "Var":
        return a.name == b.name
    if a.kind == "Let":
        return (
            a.name == b.name
            and _term_shape_eq(a.value, b.value)
            and _term_shape_eq(a.body, b.body)
        )
    if a.kind == "App":
        return _term_shape_eq(a.fn, b.fn) and _term_shape_eq(a.arg, b.arg)
    if a.kind == "Abs":
        return a.param == b.param and _term_shape_eq(a.body, b.body)
    if a.kind == "Fix":
        return _term_shape_eq(a.body, b.body)
    if a.kind == "Case":
        if not _term_shape_eq(a.scrutinee, b.scrutinee):
            return False
        if set(a.branches.keys()) != set(b.branches.keys()):
            return False
        for k in a.branches:
            if not _term_shape_eq(a.branches[k], b.branches[k]):
                return False
        if (a.default is None) != (b.default is None):
            return False
        if a.default is not None:
            return _term_shape_eq(a.default, b.default)
        return True
    if a.kind == "Combinator":
        if a.op != b.op:
            return False
        if len(a.args) != len(b.args):
            return False
        return all(_term_shape_eq(x, y) for x, y in zip(a.args, b.args, strict=True))
    raise AssertionError(f"unknown kind {a.kind!r}")


# ---------------------------------------------------------------------------
# Reference inline builders (verbatim from M4 examples — text simplified, but
# the structural skeleton is identical)
# ---------------------------------------------------------------------------


def _ref_react() -> Term:
    decide = leaf(
        template="decide template task={task}",
        input_vars=("task",),
        schema_ref="examples.pipeline.react_search.schemas.ToolDecision",
    )
    synth = leaf(
        template="synth template t={task} d={decision} o={observation}",
        input_vars=("task", "decision", "observation"),
        schema_ref="examples.pipeline.react_search.schemas.FinalAnswer",
    )
    return let(
        "decision",
        decide,
        let("observation", app(var("tool_dispatch"), var("decision")), synth),
    )


def _ref_rewoo() -> Term:
    plan_l = leaf(
        template="plan template t={task}",
        input_vars=("task",),
    )
    synth = leaf(
        template="synth template t={task} p={plan} e={evidence}",
        input_vars=("task", "plan", "evidence"),
    )
    return let(
        "plan",
        plan_l,
        let("evidence", app(var("plan_exec"), var("plan")), synth),
    )


def _ref_reflexion() -> Term:
    solve = leaf(template="solve {task}", input_vars=("task",))
    evaluate = leaf(template="eval {task} {attempt1}", input_vars=("task", "attempt1"))
    reflect_l = leaf(
        template="reflect {task} {attempt1} {evaluation}",
        input_vars=("task", "attempt1", "evaluation"),
    )
    re_solve = leaf(
        template="resolve {task} {reflection}",
        input_vars=("task", "reflection"),
    )
    return let(
        "attempt1",
        solve,
        let(
            "evaluation",
            evaluate,
            let("reflection", reflect_l, re_solve),
        ),
    )


def _ref_memory() -> Term:
    ctx = leaf(template="ctx {task}", input_vars=("task",))
    ans = leaf(template="ans {task} {context}", input_vars=("task", "context"))
    return let("context", ctx, ans)


# ---------------------------------------------------------------------------
# Shape-equivalence tests (always-on)
# ---------------------------------------------------------------------------


class TestShapeEquivalence:
    def test_react_term_shape_equivalence(self):
        factory_out = react_term(
            decide_prompt="decide {task}",
            synth_prompt="synth {task} {decision} {observation}",
        )
        assert _term_shape_eq(factory_out, _ref_react())

    def test_rewoo_term_shape_equivalence(self):
        factory_out = rewoo_term(
            plan_prompt="plan {task}",
            synth_prompt="synth {task} {plan} {evidence}",
        )
        assert _term_shape_eq(factory_out, _ref_rewoo())

    def test_reflexion_term_shape_equivalence(self):
        factory_out = reflexion_term(
            solve_prompt="solve {task}",
            eval_prompt="eval {task} {attempt1}",
            reflect_prompt="reflect {task} {attempt1} {evaluation}",
            resolve_prompt="resolve {task} {reflection}",
        )
        assert _term_shape_eq(factory_out, _ref_reflexion())

    def test_memory_term_shape_equivalence(self):
        factory_out = memory_term(
            context_prompt="ctx {task}",
            answer_prompt="ans {task} {context}",
        )
        assert _term_shape_eq(factory_out, _ref_memory())

    def test_factory_purity_no_imports_outside_lam(self):
        """STOP IF guard from PLAN step 1: lam_factories must import only fsm_llm.lam."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[2]
            / "src/fsm_llm/stdlib/agents/lam_factories.py"
        ).read_text()
        # Allow only the single canonical import line.
        forbidden = ("from fsm_llm.llm", "from fsm_llm.fsm", "from fsm_llm.pipeline")
        for needle in forbidden:
            assert needle not in src, (
                f"purity violation: {needle!r} in lam_factories.py"
            )

    def test_react_term_custom_tool_dispatch_var(self):
        """Custom env-var name flows through to the App's Var name."""
        t = react_term(
            decide_prompt="d {task}",
            synth_prompt="s {task} {decision} {observation}",
            tool_dispatch_var="my_dispatcher",
        )
        # Walk: Let("decision", ..., Let("observation", App(Var("my_dispatcher"), ...), _))
        assert t.kind == "Let" and t.name == "decision"
        inner = t.body
        assert inner.kind == "Let" and inner.name == "observation"
        app_node = inner.value
        assert app_node.kind == "App"
        assert app_node.fn.kind == "Var"
        assert app_node.fn.name == "my_dispatcher"


# ---------------------------------------------------------------------------
# Smoke tests (opt-in)
# ---------------------------------------------------------------------------


def _real_llm_env() -> tuple[str, str]:
    model = os.environ.get("LLM_MODEL", "ollama_chat/qwen3.5:4b")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return model, api_key


def _make_oracle_executor():
    from fsm_llm.runtime import Executor, LiteLLMOracle
    from fsm_llm.runtime._litellm import LiteLLMInterface

    model, _ = _real_llm_env()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    return Executor(oracle=oracle)


def _mock_tool_dispatch(decision: Any) -> str:
    """Trivial dispatcher that ignores the decision payload — for smoke only."""
    return "France has approximately 68 million inhabitants."


def _mock_plan_exec(plan: Any) -> dict:
    return {"#E1": "France pop = 68M", "#E2": "68M / 2 = 34M"}


@pytest.mark.real_llm
@pytest.mark.slow
class TestSmokeRuns:
    def test_react_term_smoke_run(self):
        ex = _make_oracle_executor()
        term = react_term(
            decide_prompt=(
                "Pick one tool. tool_name in [search, calculate, lookup]. "
                "query is the argument string.\nTask: {task}"
            ),
            synth_prompt=(
                "Write the final answer using the tool result.\n"
                "Task: {task}\nDecision: {decision}\nObservation: {observation}"
            ),
        )
        env = {
            "task": "What is the population of France?",
            "tool_dispatch": _mock_tool_dispatch,
        }
        result = ex.run(term, env)
        assert ex.oracle_calls == 2, f"expected 2 oracle calls, got {ex.oracle_calls}"
        assert result is not None

    def test_rewoo_term_smoke_run(self):
        ex = _make_oracle_executor()
        term = rewoo_term(
            plan_prompt=(
                "Produce a 2-step plan. Each step has tool_name "
                "(search|calculate|lookup), args (one string), label (#E1, #E2).\n"
                "Task: {task}"
            ),
            synth_prompt=(
                "Synthesize the final answer using the plan and evidence.\n"
                "Task: {task}\nPlan: {plan}\nEvidence: {evidence}"
            ),
        )
        env = {
            "task": "What is the population of France divided by 2?",
            "plan_exec": _mock_plan_exec,
        }
        result = ex.run(term, env)
        assert ex.oracle_calls == 2, f"expected 2 oracle calls, got {ex.oracle_calls}"
        assert result is not None

    def test_reflexion_term_smoke_run(self):
        ex = _make_oracle_executor()
        term = reflexion_term(
            solve_prompt="Solve. Provide answer + brief rationale.\nTask: {task}",
            eval_prompt=(
                "Evaluate the attempt. quality_score 0..1, feedback, "
                "passed=true iff quality >= 0.7.\nTask: {task}\nAttempt: {attempt1}"
            ),
            reflect_prompt=(
                "Reflect on the attempt and feedback. Provide lesson and strategy.\n"
                "Task: {task}\nAttempt: {attempt1}\nEvaluation: {evaluation}"
            ),
            resolve_prompt=(
                "Solve again, applying the lesson and strategy.\n"
                "Task: {task}\nReflection: {reflection}"
            ),
        )
        env = {"task": "What is the population density of France?"}
        result = ex.run(term, env)
        assert ex.oracle_calls == 4, f"expected 4 oracle calls, got {ex.oracle_calls}"
        assert result is not None

    def test_memory_term_smoke_run(self):
        ex = _make_oracle_executor()
        term = memory_term(
            context_prompt=(
                "Build relevant context. Provide summary + 2-3 key points "
                "as a single string.\nTask: {task}"
            ),
            answer_prompt=(
                "Provide the final answer using the context.\n"
                "Task: {task}\nContext: {context}"
            ),
        )
        env = {"task": "Recall and answer: what did we discuss about climate models?"}
        result = ex.run(term, env)
        assert ex.oracle_calls == 2, f"expected 2 oracle calls, got {ex.oracle_calls}"
        assert result is not None
