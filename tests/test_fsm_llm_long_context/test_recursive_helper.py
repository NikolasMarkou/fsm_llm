from __future__ import annotations

"""Tests for fsm_llm.stdlib.long_context._recursive — M5 slice 3 (D-S2-001).

Proves the private ``_recursive_long_context`` helper produces an
executor-equivalent term to an inline-built term with the same shape and
arguments. Also covers the ``extra_input_vars`` threading hook used by
``multi_hop`` and the helper's argument guards.
"""

from typing import Any

import pytest

from fsm_llm.lam import (
    Executor,
    abs_,
    app,
    case_,
    fix,
    fmap,
    leaf,
    reduce_,
    split,
)
from fsm_llm.stdlib.long_context import best_answer_op, make_size_bucket

# Private helper — imported deliberately for white-box equivalence tests.
from fsm_llm.stdlib.long_context._recursive import _recursive_long_context


class _ScriptedOracle:
    """Minimal Oracle: pops scripted responses, records prompts.

    Copy-paste of the test_niah / test_multi_hop helper. D-S2-001
    follow-up (conftest extraction) is deferred — see decisions.md.
    """

    def __init__(self, responses: list[Any], K: int = 10_000) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self._K = K

    def invoke(
        self,
        prompt: str,
        schema: Any = None,
        *,
        model_override: str | None = None,
    ) -> Any:
        self.calls.append(prompt)
        return self._responses.pop(0)

    def tokenize(self, text: str) -> int:
        return max(1, len(text))

    def context_window(self) -> int:
        return self._K


def _inline_recursive_term(
    leaf_prompt: str,
    *,
    k: int,
    reduce_op_name: str,
    input_var: str,
) -> Any:
    """Hand-written term mirroring the helper's body shape (no extras)."""
    body = abs_(
        "self",
        abs_(
            "P",
            case_(
                app("size_bucket", "P"),
                {"small": leaf(leaf_prompt, ("P",))},
                default=reduce_(reduce_op_name, fmap("self", split("P", k))),
            ),
        ),
    )
    return app(fix(body), input_var)


def test_t1_helper_equivalent_to_inline_term() -> None:
    """T1 — helper-built term and inline term produce identical executor traces."""
    doc = "AAAAXAAA"  # n=8, τ=1, k=2
    responses = ["NOT_FOUND" if ch != "X" else "X" for ch in doc]
    env_factory = lambda: {  # noqa: E731 — small lambda for test brevity
        "document": doc,
        "size_bucket": make_size_bucket(1),
        "best": best_answer_op(),
    }

    helper_term = _recursive_long_context(
        "Q: {P}", tau=1, k=2, reduce_op_name="best", input_var="document"
    )
    inline_term = _inline_recursive_term(
        "Q: {P}", k=2, reduce_op_name="best", input_var="document"
    )

    o_helper = _ScriptedOracle(responses=list(responses))
    ex_helper = Executor(oracle=o_helper)
    r_helper = ex_helper.run(helper_term, env_factory())

    o_inline = _ScriptedOracle(responses=list(responses))
    ex_inline = Executor(oracle=o_inline)
    r_inline = ex_inline.run(inline_term, env_factory())

    assert r_helper == r_inline == "X"
    assert ex_helper.oracle_calls == ex_inline.oracle_calls == 8
    assert o_helper.calls == o_inline.calls


def test_t2_extra_input_vars_threading() -> None:
    """T2 — extra_input_vars binding is threaded into the leaf prompt at runtime."""
    doc = "AB"
    term = _recursive_long_context(
        "ctx={ctx}|P={P}",
        tau=1,
        k=2,
        reduce_op_name="best",
        input_var="document",
        extra_input_vars=("ctx",),
    )

    oracle = _ScriptedOracle(responses=["r1", "r2"])
    ex = Executor(oracle=oracle)
    result = ex.run(
        term,
        {
            "document": doc,
            "ctx": "EXTRA",
            "size_bucket": make_size_bucket(1),
            "best": best_answer_op(),
        },
    )

    assert result in {"r1", "r2"}
    assert ex.oracle_calls == 2
    assert all("EXTRA" in p for p in oracle.calls), oracle.calls
    assert all("ctx=EXTRA" in p for p in oracle.calls), oracle.calls


def test_t3_helper_guards() -> None:
    """T3 — tau<1 and k<2 raise ValueError."""
    with pytest.raises(ValueError, match="tau"):
        _recursive_long_context(
            "p", tau=0, k=2, reduce_op_name="best", input_var="document"
        )
    with pytest.raises(ValueError, match="k"):
        _recursive_long_context(
            "p", tau=1, k=1, reduce_op_name="best", input_var="document"
        )


def test_t4_degenerate_single_call() -> None:
    """T4 — tau >= len(doc): single oracle call, no recursion."""
    term = _recursive_long_context(
        "Q: {P}", tau=100, k=2, reduce_op_name="best", input_var="document"
    )
    oracle = _ScriptedOracle(responses=["only-answer"])
    ex = Executor(oracle=oracle)
    result = ex.run(
        term,
        {
            "document": "tiny",
            "size_bucket": make_size_bucket(100),
            "best": best_answer_op(),
        },
    )
    assert result == "only-answer"
    assert ex.oracle_calls == 1
