"""Live (real-LLM) smoke tests for ``fsm_llm.stdlib.workflows.lam_factories``.

Opt-in via ``TEST_REAL_LLM=1``; default-skipped in CI. Runs on
``ollama_chat/qwen3.5:4b`` (override with ``LLM_MODEL``).

Each test verifies Theorem-2 (strict for linear/parallel; runtime
upper-bound for branch/switch/retry — only the taken arm fires)::

    Executor.run(term, env).oracle_calls
        == leaves_in_taken_arm  (branch/switch)
        == sum(leaves(b))        (linear/parallel)
        == 0                     (retry — body is host-callable, no Leaf)
"""

from __future__ import annotations

import os

import pytest

from fsm_llm.runtime import leaf
from fsm_llm.stdlib.workflows.lam_factories import (
    branch_term,
    linear_term,
    parallel_term,
    retry_term,
    switch_term,
)


def _real_llm_env() -> tuple[str, str]:
    model = os.environ.get("LLM_MODEL", "ollama_chat/qwen3.5:4b")
    return model, os.environ.get("OPENAI_API_KEY", "")


def _make_oracle_executor():
    from fsm_llm.runtime import Executor, LiteLLMOracle
    from fsm_llm.runtime._litellm import LiteLLMInterface

    model, _ = _real_llm_env()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    return Executor(oracle=oracle)


@pytest.mark.real_llm
@pytest.mark.slow
class TestSmokeRuns:
    def test_linear_term_smoke(self) -> None:
        """3 leaves chained → 3 oracle calls (strict)."""
        ex = _make_oracle_executor()
        a = leaf(template="Stage A. Topic: {input}", input_vars=("input",))
        b = leaf(template="Stage B. Build on A.\nA: {a}", input_vars=("a",))
        c = leaf(template="Stage C. Synthesize.\nB: {b}", input_vars=("b",))
        term = linear_term(("a", a), ("b", b), ("c", c))
        result = ex.run(term, {"input": "ocean tides"})
        assert ex.oracle_calls == 3, (
            f"linear: expected 3 oracle calls, got {ex.oracle_calls}"
        )
        assert result is not None

    def test_branch_term_smoke_then_branch(self) -> None:
        """Boolean branch — 'then' arm has 1 leaf, 'else' has 1 leaf;
        host-callable picks 'true' → 1 oracle call (then-only)."""
        ex = _make_oracle_executor()
        t = leaf(template="Then-arm: {input}", input_vars=("input",))
        e = leaf(template="Else-arm: {input}", input_vars=("input",))
        term = branch_term("is_question", t, e)

        def cond(payload: str) -> str:
            return "true" if "?" in payload else "false"

        result = ex.run(
            term,
            {"input": "What is the capital of France?", "is_question": cond},
        )
        assert ex.oracle_calls == 1, (
            f"branch (then): expected 1 oracle call, got {ex.oracle_calls}"
        )
        assert result is not None

    def test_switch_term_smoke(self) -> None:
        """3-way switch — host-callable picks one of 3 arms → 1 oracle call."""
        ex = _make_oracle_executor()
        a = leaf(template="Arm-A: {input}", input_vars=("input",))
        b = leaf(template="Arm-B: {input}", input_vars=("input",))
        c = leaf(template="Arm-C: {input}", input_vars=("input",))
        d = leaf(template="Default: {input}", input_vars=("input",))
        term = switch_term("classify", {"a": a, "b": b, "c": c}, d)

        def classify(payload: str) -> str:
            # Always pick "b" — deterministic.
            return "b"

        result = ex.run(
            term,
            {"input": "test payload", "classify": classify},
        )
        assert ex.oracle_calls == 1, (
            f"switch: expected 1 oracle call, got {ex.oracle_calls}"
        )
        assert result is not None

    def test_parallel_term_smoke(self) -> None:
        """3-branch parallel + reduce — 3 oracle calls (strict)."""
        from fsm_llm.runtime import ReduceOp

        ex = _make_oracle_executor()
        b1 = leaf(template="Branch-1 view of: {input}", input_vars=("input",))
        b2 = leaf(template="Branch-2 view of: {input}", input_vars=("input",))
        b3 = leaf(template="Branch-3 view of: {input}", input_vars=("input",))
        term = parallel_term(
            [("b1", b1), ("b2", b2), ("b3", b3)],
            reduce_op_name="my_concat",
        )

        # Need closure to capture env-bound branch results.
        # Strategy: bind a callable that captures the current env via
        # the executor's let-bound branches. But Executor passes only
        # the last branch's value as the App arg. So our list-builder
        # ignores the arg and returns a fixed sentinel list — which
        # breaks Theorem-2 (oracle_calls counts ONLY actual Leaf evals,
        # which is 3 regardless of what happens after). The reduce
        # then folds the list. We'll bind a no-op concatenator.
        def build_branch_list(_last: object) -> list:
            # Workaround: returning a list with 1 entry is cheapest.
            # The reduce+fmap path doesn't add oracle calls (identity).
            return [_last]

        def identity(x):
            return x

        my_concat = ReduceOp(
            name="my_concat",
            fn=lambda a, b: f"{a}|{b}",
            associative=True,
            unit="",
        )

        result = ex.run(
            term,
            {
                "input": "rain forecast",
                "build_branch_list": build_branch_list,
                "identity": identity,
                "my_concat": my_concat,
            },
        )
        assert ex.oracle_calls == 3, (
            f"parallel: expected 3 oracle calls, got {ex.oracle_calls}"
        )
        assert result is not None

    def test_retry_term_smoke_succeeds_on_first(self) -> None:
        """Retry — body is host-callable (no Leaf); success on attempt 1.
        Expected oracle_calls == 0."""
        ex = _make_oracle_executor()
        term = retry_term("body", "success", max_attempts=3)

        attempts = {"count": 0}

        def body(x: str) -> str:
            attempts["count"] += 1
            return f"attempt-{attempts['count']}-of-{x}"

        def success(_attempt: str) -> str:
            return "true"  # always succeed first try

        result = ex.run(
            term,
            {"input": "task-A", "body": body, "success": success},
        )
        assert ex.oracle_calls == 0, (
            f"retry (first-success): expected 0 oracle calls, got {ex.oracle_calls}"
        )
        assert attempts["count"] == 1, (
            f"retry (first-success): body should fire once, got {attempts['count']}"
        )
        assert result is not None
