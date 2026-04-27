from __future__ import annotations

"""
λ-term factories for the 9 ReasoningType strategies + classifier + outer
orchestrator (M3 slice 2).

This module exposes named factory functions that return closed
``fsm_llm.lam`` λ-terms ready for ``Executor.run``. Each factory captures
one of the cognitive shapes described in
``fsm_llm.stdlib.reasoning.reasoning_modes`` as a fixed-depth let-chain:

- ``analytical_term``  — decompose → analyze → integrate           (3 leaves)
- ``deductive_term``   — premises → infer → conclude               (3 leaves)
- ``inductive_term``   — examples → pattern → generalize           (3 leaves)
- ``abductive_term``   — observe → hypothesize → select_best       (3 leaves)
- ``analogical_term``  — source → mapping → target_inference       (3 leaves)
- ``creative_term``    — diverge → combine → refine                (3 leaves)
- ``critical_term``    — examine → evaluate → verdict              (3 leaves)
- ``hybrid_term``      — facets → strategies → execute → integrate (4 leaves)
- ``calculator_term``  — parse → compute                           (2 leaves)
- ``classifier_term``  — domain → structure → needs → recommend    (4 leaves)
- ``solve_term``       — strategy → solution → validation → final  (4 leaves; 2 host-callable Apps)

**Purity invariant** — this module imports ONLY from ``fsm_llm.lam``. No
imports of ``fsm_llm.llm``, ``fsm_llm.fsm``, ``fsm_llm.pipeline``, or
``fsm_llm.stdlib.reasoning.engine``. The factories close over no Python
state; all dynamic values (problem strings, host-callable dispatchers)
are bound by the caller in ``env`` when invoking
``Executor.run(term, env)``.

See ``docs/lambda.md`` §11 for design rationale and §13 M3 row for slice
context. Slice 1 (``fsm_llm.stdlib.agents.lam_factories``) is the
load-bearing precedent.
"""

from fsm_llm.lam import Term, app, leaf, let_, var

__all__: list[str] = []


def _chain(*pairs: tuple[str, Term]) -> Term:
    """Fold ``[(name1, leaf1), (name2, leaf2), ..., (nameN, leafN)]`` into
    a right-nested ``let_`` chain.

    Given pairs ``[(n1, l1), (n2, l2), (n3, l3)]`` returns::

        let_(n1, l1, let_(n2, l2, l3))

    The **last** pair's term is the body — its name is unused but kept
    in the tuple for documentation symmetry. The result has exactly
    ``len(pairs)`` leaves (assuming each ``Term`` is itself a single
    Leaf — which is how the factories below use it).

    This is a private helper. Callers should use the named ``*_term``
    factories rather than constructing chains manually.

    Raises
    ------
    ValueError
        If ``pairs`` has fewer than 2 entries (a 1-leaf "chain" is just
        the leaf itself; callers should use that directly).
    """
    if len(pairs) < 2:
        raise ValueError(
            f"_chain requires at least 2 pairs, got {len(pairs)}"
        )
    # Build right-nested let_ from the back.
    # Last pair's leaf is the innermost body.
    name_last, body = pairs[-1]
    # Walk from second-to-last down to first, wrapping body in let_.
    for name, leaf_term in reversed(pairs[:-1]):
        body = let_(name, leaf_term, body)
    return body
