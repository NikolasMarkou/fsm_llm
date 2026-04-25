from __future__ import annotations

"""
Aggregate factory — synthesise an answer across ALL chunks of a long
document (distinct from ``niah``, which picks the BEST single chunk's
answer).

Builds a λ-term that recursively splits the input into ``k``-ary chunks,
calls the oracle on each chunk to extract per-chunk findings, and
reduces with a caller-supplied joiner op (default: ``aggregate_op()``,
a bullet-joining ReduceOp). The reduce step is pure Python — NO extra
oracle calls — so Theorem 2 (predicted_calls = k^d) holds identically
to ``niah``.

Semantics::

    fix(λself. λP.
       case size_bucket(P) of
         "small" → leaf(<summarise this chunk prompt>, P)
         _      → reduce_(merge, fmap(self, split(P, k))))

Per D-001 (slice-2 plan): the term construction body is duplicated from
``niah.py``, NOT extracted into a shared helper. Slice 3 (when a 3rd
factory lands) is the natural extraction point.
"""

from collections.abc import Callable
from typing import Any

from fsm_llm.lam import Term
from fsm_llm.lam.combinators import ReduceOp

from ._recursive import _recursive_long_context

_AGGREGATE_PROMPT_TEMPLATE = (
    "You are reading one chunk of a longer document. Extract any "
    "information from this chunk that is relevant to the question.\n\n"
    "Question: {question}\n\n"
    "Chunk:\n{{P}}\n\n"
    "Output a single concise bullet (one line) with the key fact(s) from "
    "this chunk that bear on the question. If this chunk contains no "
    "relevant information, output exactly: NOT_FOUND"
)


def aggregate(
    question: str,
    *,
    tau: int = 512,
    k: int = 2,
    reduce_op_name: str = "merge",
    input_var: str = "document",
) -> Term:
    """Build an aggregate-across-chunks λ-term.

    Parameters
    ----------
    question:
        The question whose answer should be synthesised across all chunks.
        Baked into each leaf prompt at factory-build time.
    tau:
        Leaf-size threshold (characters). See ``niah`` for full semantics.
    k:
        Branching factor for SPLIT. Default 2.
    reduce_op_name:
        Name of the joiner op to look up in env. Caller binds an
        ``aggregate_op()`` (or any associative ReduceOp). Default ``"merge"``.
    input_var:
        Name of the env variable holding the document. Default ``"document"``.

    Returns
    -------
    A closed λ-term ready for ``Executor.run(program, env)``. Caller's env
    must bind ``input_var`` (the doc), ``"size_bucket"`` (callable), and
    ``reduce_op_name`` (ReduceOp).

    Notes
    -----
    Same Theorem-2 cost equality as ``niah`` (``ex.oracle_calls ==
    plan(...).predicted_calls`` for τ·k^d-aligned inputs). The reduce step
    runs in pure Python and adds zero oracle calls.

    Output is free-form synthesis. Verification of correctness is a
    weak heuristic at best (length, sentinel-rejection); slice 2 ships
    no labelled-benchmark scoring.
    """
    leaf_prompt = _AGGREGATE_PROMPT_TEMPLATE.format(question=question)

    # DECISION D-S2-001 (resolved slice 3): term construction delegated to
    # the shared private helper. Guards (tau >= 1, k >= 2) live in the helper.
    return _recursive_long_context(
        leaf_prompt,
        tau=tau,
        k=k,
        reduce_op_name=reduce_op_name,
        input_var=input_var,
    )


def aggregate_op(
    separator: str = "\n- ",
    sentinel: str = "NOT_FOUND",
) -> ReduceOp:
    """Build the standard joiner ReduceOp for ``aggregate``.

    Strategy: drop ``sentinel`` and empty values; join the remainder with
    ``separator``. Empty fold returns ``sentinel`` as the unit.

    Associativity: textual concatenation is associative; sentinel filtering
    is order-independent. Both invariants hold so ``associative=True`` is
    sound.
    """

    def _is_bad(x: Any) -> bool:
        return x is None or x == "" or x == sentinel

    def _join(a: Any, b: Any) -> Any:
        a_bad = _is_bad(a)
        b_bad = _is_bad(b)
        if a_bad and b_bad:
            return sentinel
        if a_bad:
            return str(b)
        if b_bad:
            return str(a)
        return f"{a}{separator}{b}"

    return ReduceOp(name="merge", fn=_join, associative=True, unit=sentinel)


def make_size_bucket(tau: int) -> Callable[[Any], str]:
    """Re-export of ``niah.make_size_bucket`` for symmetry. Use either."""
    from .niah import make_size_bucket as _mk

    return _mk(tau)


__all__ = ["aggregate", "aggregate_op", "make_size_bucket"]
