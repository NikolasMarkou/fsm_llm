from __future__ import annotations

"""
Needle-in-haystack (NIAH) factory.

Builds a λ-term that recursively splits an input document of arbitrary
size into ``k``-ary chunks until each chunk fits the leaf threshold ``τ``
(in characters), invokes the oracle on each chunk to extract a candidate
answer, and reduces with a caller-supplied "best" op to select the final
answer.

Semantics::

    fix(λself. λP.
       case size_bucket(P) of
         "small" → leaf(<question prompt>, P)
         _      → reduce_(best, fmap(self, split(P, k))))

The factory closes over no Python state. All dynamic values are bound by
the caller in ``env``:

- ``<input_var>`` — the document string (default name: ``"document"``).
- ``size_bucket`` — a callable ``str → {"small","big"}`` deciding base case.
- ``<reduce_op_name>`` — a ``ReduceOp`` (or bare callable) for combining
  answers (default name: ``"best"``).

Convenience helpers ``make_size_bucket(tau)`` and ``best_answer_op()``
build the standard env values; using them is optional.
"""

from collections.abc import Callable
from typing import Any

from fsm_llm.runtime import Term
from fsm_llm.runtime.combinators import ReduceOp

from ._recursive import _recursive_long_context

_NIAH_PROMPT_TEMPLATE = (
    "You are searching a portion of a long document for the answer to a "
    "question.\n\n"
    "Question: {question}\n\n"
    "Text:\n{{P}}\n\n"
    "If the text contains the answer, output it verbatim and nothing else. "
    "If the text does NOT contain the answer, output exactly: NOT_FOUND"
)


def niah(
    question: str,
    *,
    tau: int = 512,
    k: int = 2,
    reduce_op_name: str = "best",
    input_var: str = "document",
) -> Term:
    """Build a needle-in-haystack λ-term.

    Parameters
    ----------
    question:
        The question to ask each leaf-level chunk. Baked into the leaf
        prompt template at factory-build time.
    tau:
        Leaf-size threshold (characters). Inputs of length ≤ τ go to a
        single oracle call; larger inputs are SPLIT into ``k`` pieces and
        recursed. The caller's ``size_bucket`` env binding is responsible
        for honouring τ — the standard helper is ``make_size_bucket(tau)``.
    k:
        Branching factor for SPLIT. Default 2 (paper Theorem 4 optimum
        under linear cost). The planner default ``max_k`` is also 2 — if
        you change ``k`` here, supply a matching ``max_k`` to ``plan()``.
    reduce_op_name:
        Name of the REDUCE op to look up in env. Caller must bind a
        ``ReduceOp`` (or bare callable) under this name. Default ``"best"``.
    input_var:
        Name of the env variable that holds the document string. Default
        ``"document"``.

    Returns
    -------
    A closed λ-term ready for ``Executor.run(program, env)``. The caller's
    env must bind ``input_var`` (the doc), ``"size_bucket"`` (callable),
    and ``reduce_op_name`` (ReduceOp).

    Notes
    -----
    The planner-executor cost equality (``ex.oracle_calls ==
    plan(...).predicted_calls``) holds when ``len(document) == τ · k^d``
    for some integer ``d ≥ 0``. For non-aligned inputs ``split_impl``
    still produces a balanced k-ary tree but the leaf count may diverge
    from ``k^d``; document accordingly to callers.
    """
    leaf_prompt = _NIAH_PROMPT_TEMPLATE.format(question=question)

    # DECISION D-S2-001 (resolved slice 3): term construction delegated to
    # the shared private helper. Guards (tau >= 1, k >= 2) live in the helper.
    return _recursive_long_context(
        leaf_prompt,
        tau=tau,
        k=k,
        reduce_op_name=reduce_op_name,
        input_var=input_var,
    )


def make_size_bucket(tau: int) -> Callable[[Any], str]:
    """Build the standard size-bucket callable for a given τ.

    Returns ``"small"`` when ``len(p) <= tau``, ``"big"`` otherwise.
    Non-sized values are treated as ``"small"`` (single leaf call).
    """

    def _size_bucket(p: Any) -> str:
        try:
            return "small" if len(p) <= tau else "big"
        except TypeError:
            return "small"

    return _size_bucket


def best_answer_op(sentinel: str = "NOT_FOUND") -> ReduceOp:
    """Build the standard "best" ReduceOp for NIAH.

    Strategy: discard ``sentinel`` / empty values; among the remainder
    pick the longer string (proxy for "more informative answer"). Empty
    fold returns ``sentinel`` as the unit.
    """

    def _pick(a: Any, b: Any) -> Any:
        a_bad = a is None or a == "" or a == sentinel
        b_bad = b is None or b == "" or b == sentinel
        if a_bad and b_bad:
            return sentinel
        if a_bad:
            return b
        if b_bad:
            return a
        return a if len(str(a)) >= len(str(b)) else b

    return ReduceOp(name="best", fn=_pick, associative=True, unit=sentinel)


__all__ = ["niah", "make_size_bucket", "best_answer_op"]
