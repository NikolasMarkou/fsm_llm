from __future__ import annotations

"""
Pairwise factory — pick the segment of a long document most relevant to a
given question, by recursive k-ary tournament reduction.

Builds a λ-term that recursively splits the input document into ``k``-ary
chunks, asks the oracle (at each leaf) to return the segment text most
relevant to the caller's question (or a sentinel if neither is relevant),
and reduces with a caller-supplied "compare" op that picks the longer
non-sentinel candidate at each merge step.

Semantics::

    fix(λself. λP.
       case size_bucket(P) of
         "small" → leaf(<pairwise prompt>, P)
         _      → reduce_(compare, fmap(self, split(P, k))))

The factory closes over no Python state. All dynamic values are bound by
the caller in ``env``:

- ``<input_var>`` — the document string (default name: ``"document"``).
- ``size_bucket`` — a callable ``str → {"small","big"}`` deciding base case.
- ``<reduce_op_name>`` — a ``ReduceOp`` (or bare callable) for combining
  candidate segments (default name: ``"compare"``). Use ``compare_op()`` for
  the standard op.

Convenience helper ``compare_op(sentinel)`` builds the standard env value;
using it is optional.

Slice 3 limitation (D-S3-001)
-----------------------------

The slice-3 ``compare_op`` is functionally equivalent to ``best_answer_op``
(longer-non-sentinel-wins). Pairwise's differentiation from ``niah`` lives
in the *leaf prompt template* (it asks the oracle to pick between two
segments) and *demo content*, not in the op math. A true oracle-mediated
pairwise comparison op (where each reduce step asks the oracle to pick a
winner of each pair) is non-trivially associative under non-deterministic
oracle responses and is deferred to slice 4.

Theorem-2 cost equality (``ex.oracle_calls == plan(...).predicted_calls``)
holds when ``len(document) == τ · k^d`` for some integer ``d ≥ 0``, just as
for ``niah`` and ``aggregate``. See ``docs/lambda.md`` §13 (M5 slice 3).
"""

from typing import Any

from fsm_llm.lam import Term
from fsm_llm.lam.combinators import ReduceOp

from ._recursive import _recursive_long_context

_PAIRWISE_PROMPT_TEMPLATE = (
    "You are evaluating a passage from a long document for relevance to a "
    "question. The passage may itself contain multiple distinct segments.\n\n"
    "Question: {question}\n\n"
    "Passage:\n{{P}}\n\n"
    "Identify the single segment of the passage above that is MOST relevant "
    "to the question and output it verbatim and nothing else. If NO segment "
    "of this passage is relevant to the question, output exactly: NOT_FOUND"
)


def pairwise(
    question: str,
    *,
    tau: int = 512,
    k: int = 2,
    reduce_op_name: str = "compare",
    input_var: str = "document",
) -> Term:
    """Build a pairwise-tournament λ-term over a long document.

    Parameters
    ----------
    question:
        The question used to judge relevance at each leaf. Baked into the
        leaf prompt template at factory-build time.
    tau:
        Leaf-size threshold (characters). Inputs of length ≤ τ go to a
        single oracle call; larger inputs are SPLIT into ``k`` pieces and
        recursed. The caller's ``size_bucket`` env binding is responsible
        for honouring τ — use ``make_size_bucket(tau)`` from
        ``stdlib.long_context.niah``.
    k:
        Branching factor for SPLIT. Default 2 (paper Theorem 4 optimum
        under linear cost).
    reduce_op_name:
        Name of the REDUCE op to look up in env. Caller must bind a
        ``ReduceOp`` (or bare callable) under this name. Default
        ``"compare"`` — bind via ``compare_op()``.
    input_var:
        Name of the env variable that holds the document string. Default
        ``"document"``.

    Returns
    -------
    A closed λ-term ready for ``Executor.run(program, env)``. The caller's
    env must bind ``input_var`` (the doc), ``"size_bucket"`` (callable),
    and ``reduce_op_name`` (ReduceOp).

    Raises
    ------
    ValueError
        If ``tau < 1`` or ``k < 2`` (enforced inside
        ``_recursive_long_context``).

    Notes
    -----
    Cost equality (``ex.oracle_calls == plan(...).predicted_calls``) holds
    when ``len(document) == τ · k^d`` for some integer ``d ≥ 0``. For
    non-aligned inputs, the leaf count may diverge from ``k^d`` — document
    accordingly to callers.
    """
    leaf_prompt = _PAIRWISE_PROMPT_TEMPLATE.format(question=question)

    return _recursive_long_context(
        leaf_prompt,
        tau=tau,
        k=k,
        reduce_op_name=reduce_op_name,
        input_var=input_var,
    )


def compare_op(sentinel: str = "NOT_FOUND") -> ReduceOp:
    # DECISION D-S3-001: slice-3 op is "longer-non-sentinel-wins" — same as
    # best_answer_op. Oracle-mediated comparison deferred to slice 4. See
    # decisions.md.
    """Build the standard "compare" ReduceOp for ``pairwise``.

    Strategy: discard ``sentinel`` / empty values; among the remainder
    pick the longer string (proxy for "more informative segment"). Empty
    fold returns ``sentinel`` as the unit.

    Per D-S3-001 this is functionally identical to ``best_answer_op`` —
    pairwise differentiation lives in the leaf prompt template, not here.
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

    return ReduceOp(name="compare", fn=_pick, associative=True, unit=sentinel)


__all__ = ["pairwise", "compare_op"]
