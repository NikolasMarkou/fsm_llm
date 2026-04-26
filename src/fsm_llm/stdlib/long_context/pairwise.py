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

Slice 3 default op (D-S3-001 — historical)
------------------------------------------

The slice-3 ``compare_op`` is functionally equivalent to ``best_answer_op``
(longer-non-sentinel-wins). Pairwise's differentiation from ``niah`` lives
in the *leaf prompt template* (it asks the oracle to pick between two
segments) and *demo content*, not in the op math. The oracle-mediated
variant — where each reduce step asks the oracle to pick a winner of each
pair — shipped as ``oracle_compare_op`` in M5 slice 5 (see below).

Theorem-2 cost equality:

- With ``compare_op()`` (slice-3 default): ``ex.oracle_calls ==
  plan(...).predicted_calls = k^d`` (REDUCE is pure; no oracle in fold).
- With ``oracle_compare_op(question, executor)`` (slice-5): each pair
  comparison invokes the oracle, so ``ex.oracle_calls = k^d + (k^d - 1) =
  2·k^d - 1``. Match by passing ``reduce_calls_per_node=1`` to
  ``PlanInputs``.

Both variants require ``len(document) == τ · k^d`` for strict equality.
See ``docs/lambda.md`` §13 (M5 slice 3 + slice 5).
"""

from typing import Any

from fsm_llm.lam import Term
from fsm_llm.lam.combinators import ReduceOp
from fsm_llm.logging import logger

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
    # DECISION D-S3-001 (historical): slice-3 op is "longer-non-sentinel-
    # wins" — algorithmically identical to best_answer_op. Oracle-mediated
    # variant shipped slice 5 — see ``oracle_compare_op`` below. This op
    # is retained as the default (back-compat for slice-3 callers and the
    # eval harness baseline).
    """Build the standard "compare" ReduceOp for ``pairwise``.

    Strategy: discard ``sentinel`` / empty values; among the remainder
    pick the longer string (proxy for "more informative segment"). Empty
    fold returns ``sentinel`` as the unit.

    Per D-S3-001 this is functionally identical to ``best_answer_op`` —
    pairwise differentiation lives in the leaf prompt template, not here.
    For a true oracle-mediated tournament, use ``oracle_compare_op``.
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


# --------------------------------------------------------------------------
# M5 slice 5: oracle-mediated pairwise comparison op
# --------------------------------------------------------------------------

_ORACLE_COMPARE_PROMPT_TEMPLATE = (
    "You are picking the segment more relevant to a question.\n\n"
    "Question: {question}\n\n"
    "Segment A:\n{a}\n\n"
    "Segment B:\n{b}\n\n"
    "Reply with exactly one character: A or B. No other words."
)


def _parse_compare_winner(response: Any) -> str | None:
    """Parse an oracle response into 'A', 'B', or None (unparseable).

    Tolerant of whitespace, case, and a few common verbose patterns
    (``"A"``, ``"a"``, ``"first"``, ``"1"``, etc.). Returns ``None`` on
    failure so the caller can fall back to a length-tiebreak.
    """
    if response is None:
        return None
    text = str(response).strip().lower()
    if not text:
        return None
    # Cheapest path: leading character.
    head = text[0]
    if head == "a" or head == "1":
        return "A"
    if head == "b" or head == "2":
        return "B"
    # Verbose patterns.
    if "first" in text and "second" not in text:
        return "A"
    if "second" in text and "first" not in text:
        return "B"
    if "segment a" in text and "segment b" not in text:
        return "A"
    if "segment b" in text and "segment a" not in text:
        return "B"
    return None


def oracle_compare_op(
    question: str,
    executor: Any,
    *,
    sentinel: str = "NOT_FOUND",
    model_override: str | None = None,
) -> ReduceOp:
    # DECISION D-S5-001: oracle-mediated pairwise comparison. The op
    # closes over the user's Executor and increments
    # ``executor._oracle_calls`` on every successful compare invocation,
    # so Theorem-2 cost equality holds via the existing ``oracle_calls``
    # counter. Trade-off: tight coupling to a private Executor attribute,
    # documented at the call site. Required so a single counter governs
    # both leaf and reduce-side oracle calls (matches ``lam/CLAUDE.md``).
    """Build an *oracle-mediated* pairwise comparison ReduceOp.

    At each pair-fold step, the oracle is asked to pick the more relevant
    of two candidate segments against ``question``. The op closes over the
    supplied ``executor`` and uses its ``oracle`` for invocations,
    incrementing ``executor._oracle_calls`` per call. Pair the result with
    ``PlanInputs(reduce_calls_per_node=1)`` to satisfy Theorem-2:
    ``ex.oracle_calls == predicted_calls = leaf_calls + reduce_calls
    = k^d + (k^d - 1) = 2·k^d - 1`` on aligned inputs.

    Parameters
    ----------
    question:
        The original question, baked into the compare prompt for context.
    executor:
        The ``fsm_llm.lam.Executor`` whose ``oracle`` will be invoked and
        whose ``_oracle_calls`` counter will be incremented per compare.
        Must have a non-None ``oracle`` attribute. Typed as ``Any`` here
        to preserve the I-PURITY invariant (``pairwise.py`` imports only
        from ``fsm_llm.lam`` for type names; ``Executor`` is duck-typed).
    sentinel:
        Value treated as "no relevant content" — short-circuits without
        an oracle call. Default ``"NOT_FOUND"`` (matches ``compare_op``).
    model_override:
        Optional litellm model string forwarded to ``oracle.invoke``.

    Returns
    -------
    A ``ReduceOp`` named ``"oracle_compare"``, associative=True (best-
    effort: oracle determinism is not guaranteed; same caveat as
    ``compare_op``), unit=``sentinel``.

    Notes
    -----
    Sentinel short-circuit: if either arm is sentinel/empty/None, the
    other arm is returned without an oracle call. This means strict
    Theorem-2 equality requires every reduce input to have at least one
    non-sentinel arm at the leaf level (relax to upper bound otherwise).

    Parse fallback: if the oracle response cannot be parsed as A/B, the
    op falls back to length-tiebreak (longer wins). The oracle call
    still counts toward ``oracle_calls`` (it was made successfully).
    """

    def _pick(a: Any, b: Any) -> Any:
        a_bad = a is None or a == "" or a == sentinel
        b_bad = b is None or b == "" or b == sentinel
        # Sentinel short-circuit: no oracle call.
        if a_bad and b_bad:
            return sentinel
        if a_bad:
            return b
        if b_bad:
            return a

        # Both arms real → invoke the oracle.
        if executor.oracle is None:
            # Defensive: caller built the op against an oracle-less
            # Executor. Fall back to length-tiebreak silently.
            return a if len(str(a)) >= len(str(b)) else b

        prompt = _ORACLE_COMPARE_PROMPT_TEMPLATE.format(
            question=question, a=a, b=b
        )
        try:
            response = executor.oracle.invoke(
                prompt, schema=None, model_override=model_override
            )
        except Exception as e:
            # Oracle errors fall back to length-tiebreak (no counter tick
            # — call did not succeed).
            logger.debug(f"oracle_compare_op: oracle.invoke raised {e!r}; "
                         "falling back to length-tiebreak")
            return a if len(str(a)) >= len(str(b)) else b

        # Successful invocation → tick the Executor's counter (D-S5-001).
        executor._oracle_calls += 1

        winner = _parse_compare_winner(response)
        if winner == "A":
            return a
        if winner == "B":
            return b
        # Unparseable response → length-tiebreak, log at debug.
        logger.debug(
            f"oracle_compare_op: could not parse winner from response "
            f"{response!r}; falling back to length-tiebreak"
        )
        return a if len(str(a)) >= len(str(b)) else b

    return ReduceOp(
        name="oracle_compare", fn=_pick, associative=True, unit=sentinel
    )


__all__ = ["pairwise", "compare_op", "oracle_compare_op"]
