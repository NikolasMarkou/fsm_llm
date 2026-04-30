from __future__ import annotations

"""
λ-term factories for the 9 ReasoningType strategies + classifier + outer
orchestrator (M3 slice 2).

This module exposes named factory functions that return closed
``fsm_llm.runtime`` λ-terms ready for ``Executor.run``. Each factory captures
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

**Purity invariant** — this module imports ONLY from ``fsm_llm.runtime``. No
imports of ``fsm_llm.llm``, ``fsm_llm.fsm``, ``fsm_llm.pipeline``, or
``fsm_llm.stdlib.reasoning.engine``. The factories close over no Python
state; all dynamic values (problem strings, host-callable dispatchers)
are bound by the caller in ``env`` when invoking
``Executor.run(term, env)``.

See ``docs/lambda.md`` §11 for design rationale and §13 M3 row for slice
context. Slice 1 (``fsm_llm.stdlib.agents.lam_factories``) is the
load-bearing precedent.
"""

from fsm_llm.runtime import Term, app, leaf, let_, var

__all__ = [
    "analytical_term",
    "deductive_term",
    "inductive_term",
    "abductive_term",
    "analogical_term",
    "creative_term",
    "critical_term",
    "hybrid_term",
    "calculator_term",
    "classifier_term",
    "solve_term",
]


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
        raise ValueError(f"_chain requires at least 2 pairs, got {len(pairs)}")
    # Build right-nested let_ from the back.
    # Last pair's leaf is the innermost body.
    _name_last, body = pairs[-1]
    # Walk from second-to-last down to first, wrapping body in let_.
    for name, leaf_term in reversed(pairs[:-1]):
        body = let_(name, leaf_term, body)
    return body


# ---------------------------------------------------------------------------
# 3-leaf strategy factories
#
# Each takes three descriptive prompt names plus per-stage input_vars and
# optional schema_refs. The descriptive prefixes match each factory's
# bind_names. Default input_vars assume a "problem" env var as the universal
# seed and use the let-bound name from the previous stage.
#
# The internal _three_leaf helper keeps generic positional names — it is a
# private implementation detail.
# ---------------------------------------------------------------------------


def _three_leaf(
    prompt_a: str,
    prompt_b: str,
    prompt_c: str,
    *,
    bind_names: tuple[str, str, str],
    input_vars_a: tuple[str, ...],
    input_vars_b: tuple[str, ...],
    input_vars_c: tuple[str, ...],
    schema_ref_a: str | None,
    schema_ref_b: str | None,
    schema_ref_c: str | None,
) -> Term:
    a = leaf(template=prompt_a, input_vars=input_vars_a, schema_ref=schema_ref_a)
    b = leaf(template=prompt_b, input_vars=input_vars_b, schema_ref=schema_ref_b)
    c = leaf(template=prompt_c, input_vars=input_vars_c, schema_ref=schema_ref_c)
    return _chain((bind_names[0], a), (bind_names[1], b), (bind_names[2], c))


def analytical_term(
    decomposition_prompt: str,
    analysis_prompt: str,
    integration_prompt: str,
    *,
    decomposition_input_vars: tuple[str, ...] = ("problem",),
    analysis_input_vars: tuple[str, ...] = ("problem", "decomposition"),
    integration_input_vars: tuple[str, ...] = ("problem", "analysis"),
    decomposition_schema_ref: str | None = None,
    analysis_schema_ref: str | None = None,
    integration_schema_ref: str | None = None,
) -> Term:
    """Analytical reasoning — decompose → analyze → integrate.

    Mirrors ``analytical_fsm`` (3-state) from
    ``fsm_llm.stdlib.reasoning.reasoning_modes``. 3 oracle calls per
    ``Executor.run``.
    """
    return _three_leaf(
        decomposition_prompt,
        analysis_prompt,
        integration_prompt,
        bind_names=("decomposition", "analysis", "integration"),
        input_vars_a=decomposition_input_vars,
        input_vars_b=analysis_input_vars,
        input_vars_c=integration_input_vars,
        schema_ref_a=decomposition_schema_ref,
        schema_ref_b=analysis_schema_ref,
        schema_ref_c=integration_schema_ref,
    )


def deductive_term(
    premises_prompt: str,
    inference_prompt: str,
    conclusion_prompt: str,
    *,
    premises_input_vars: tuple[str, ...] = ("problem",),
    inference_input_vars: tuple[str, ...] = ("problem", "premises"),
    conclusion_input_vars: tuple[str, ...] = ("problem", "inference"),
    premises_schema_ref: str | None = None,
    inference_schema_ref: str | None = None,
    conclusion_schema_ref: str | None = None,
) -> Term:
    """Deductive reasoning — premises → infer → conclude. 3 oracle calls."""
    return _three_leaf(
        premises_prompt,
        inference_prompt,
        conclusion_prompt,
        bind_names=("premises", "inference", "conclusion"),
        input_vars_a=premises_input_vars,
        input_vars_b=inference_input_vars,
        input_vars_c=conclusion_input_vars,
        schema_ref_a=premises_schema_ref,
        schema_ref_b=inference_schema_ref,
        schema_ref_c=conclusion_schema_ref,
    )


def inductive_term(
    examples_prompt: str,
    pattern_prompt: str,
    generalization_prompt: str,
    *,
    examples_input_vars: tuple[str, ...] = ("problem",),
    pattern_input_vars: tuple[str, ...] = ("problem", "examples"),
    generalization_input_vars: tuple[str, ...] = ("problem", "pattern"),
    examples_schema_ref: str | None = None,
    pattern_schema_ref: str | None = None,
    generalization_schema_ref: str | None = None,
) -> Term:
    """Inductive reasoning — examples → pattern → generalize. 3 oracle calls."""
    return _three_leaf(
        examples_prompt,
        pattern_prompt,
        generalization_prompt,
        bind_names=("examples", "pattern", "generalization"),
        input_vars_a=examples_input_vars,
        input_vars_b=pattern_input_vars,
        input_vars_c=generalization_input_vars,
        schema_ref_a=examples_schema_ref,
        schema_ref_b=pattern_schema_ref,
        schema_ref_c=generalization_schema_ref,
    )


def abductive_term(
    observation_prompt: str,
    hypothesis_prompt: str,
    selection_prompt: str,
    *,
    observation_input_vars: tuple[str, ...] = ("problem",),
    hypothesis_input_vars: tuple[str, ...] = ("problem", "observation"),
    selection_input_vars: tuple[str, ...] = ("problem", "hypothesis"),
    observation_schema_ref: str | None = None,
    hypothesis_schema_ref: str | None = None,
    selection_schema_ref: str | None = None,
) -> Term:
    """Abductive reasoning — observe → hypothesize → select_best. 3 oracle calls."""
    return _three_leaf(
        observation_prompt,
        hypothesis_prompt,
        selection_prompt,
        bind_names=("observation", "hypothesis", "best_explanation"),
        input_vars_a=observation_input_vars,
        input_vars_b=hypothesis_input_vars,
        input_vars_c=selection_input_vars,
        schema_ref_a=observation_schema_ref,
        schema_ref_b=hypothesis_schema_ref,
        schema_ref_c=selection_schema_ref,
    )


def analogical_term(
    source_prompt: str,
    mapping_prompt: str,
    target_inference_prompt: str,
    *,
    source_input_vars: tuple[str, ...] = ("problem",),
    mapping_input_vars: tuple[str, ...] = ("problem", "source_domain"),
    target_inference_input_vars: tuple[str, ...] = ("problem", "mapping"),
    source_schema_ref: str | None = None,
    mapping_schema_ref: str | None = None,
    target_inference_schema_ref: str | None = None,
) -> Term:
    """Analogical reasoning — source → mapping → target_inference. 3 oracle calls."""
    return _three_leaf(
        source_prompt,
        mapping_prompt,
        target_inference_prompt,
        bind_names=("source_domain", "mapping", "target_inference"),
        input_vars_a=source_input_vars,
        input_vars_b=mapping_input_vars,
        input_vars_c=target_inference_input_vars,
        schema_ref_a=source_schema_ref,
        schema_ref_b=mapping_schema_ref,
        schema_ref_c=target_inference_schema_ref,
    )


def creative_term(
    divergence_prompt: str,
    combination_prompt: str,
    refinement_prompt: str,
    *,
    divergence_input_vars: tuple[str, ...] = ("problem",),
    combination_input_vars: tuple[str, ...] = ("problem", "divergence"),
    refinement_input_vars: tuple[str, ...] = ("problem", "combination"),
    divergence_schema_ref: str | None = None,
    combination_schema_ref: str | None = None,
    refinement_schema_ref: str | None = None,
) -> Term:
    """Creative reasoning — diverge → combine → refine. 3 oracle calls."""
    return _three_leaf(
        divergence_prompt,
        combination_prompt,
        refinement_prompt,
        bind_names=("divergence", "combination", "refinement"),
        input_vars_a=divergence_input_vars,
        input_vars_b=combination_input_vars,
        input_vars_c=refinement_input_vars,
        schema_ref_a=divergence_schema_ref,
        schema_ref_b=combination_schema_ref,
        schema_ref_c=refinement_schema_ref,
    )


def critical_term(
    examination_prompt: str,
    evaluation_prompt: str,
    verdict_prompt: str,
    *,
    examination_input_vars: tuple[str, ...] = ("problem",),
    evaluation_input_vars: tuple[str, ...] = ("problem", "examination"),
    verdict_input_vars: tuple[str, ...] = ("problem", "evaluation"),
    examination_schema_ref: str | None = None,
    evaluation_schema_ref: str | None = None,
    verdict_schema_ref: str | None = None,
) -> Term:
    """Critical reasoning — examine → evaluate → verdict. 3 oracle calls."""
    return _three_leaf(
        examination_prompt,
        evaluation_prompt,
        verdict_prompt,
        bind_names=("examination", "evaluation", "verdict"),
        input_vars_a=examination_input_vars,
        input_vars_b=evaluation_input_vars,
        input_vars_c=verdict_input_vars,
        schema_ref_a=examination_schema_ref,
        schema_ref_b=evaluation_schema_ref,
        schema_ref_c=verdict_schema_ref,
    )


# ---------------------------------------------------------------------------
# hybrid_term (4 leaves) and calculator_term (2 leaves) — distinct shapes
# ---------------------------------------------------------------------------


def hybrid_term(
    facets_prompt: str,
    strategies_prompt: str,
    execution_prompt: str,
    integration_prompt: str,
    *,
    facets_input_vars: tuple[str, ...] = ("problem",),
    strategies_input_vars: tuple[str, ...] = ("problem", "facets"),
    execution_input_vars: tuple[str, ...] = ("problem", "strategies"),
    integration_input_vars: tuple[str, ...] = ("problem", "execution"),
    facets_schema_ref: str | None = None,
    strategies_schema_ref: str | None = None,
    execution_schema_ref: str | None = None,
    integration_schema_ref: str | None = None,
) -> Term:
    """Hybrid reasoning — facets → strategies → execute → integrate.

    4-leaf chain (mirrors ``hybrid_fsm``). 4 oracle calls per
    ``Executor.run``.
    """
    f = leaf(
        template=facets_prompt,
        input_vars=facets_input_vars,
        schema_ref=facets_schema_ref,
    )
    s = leaf(
        template=strategies_prompt,
        input_vars=strategies_input_vars,
        schema_ref=strategies_schema_ref,
    )
    e = leaf(
        template=execution_prompt,
        input_vars=execution_input_vars,
        schema_ref=execution_schema_ref,
    )
    i = leaf(
        template=integration_prompt,
        input_vars=integration_input_vars,
        schema_ref=integration_schema_ref,
    )
    return _chain(
        ("facets", f),
        ("strategies", s),
        ("execution", e),
        ("integration", i),
    )


def calculator_term(
    parse_prompt: str,
    compute_prompt: str,
    *,
    parse_input_vars: tuple[str, ...] = ("problem",),
    compute_input_vars: tuple[str, ...] = ("problem", "parsed"),
    parse_schema_ref: str | None = None,
    compute_schema_ref: str | None = None,
) -> Term:
    """Simple calculator — parse_expression → compute.

    2-leaf chain (mirrors ``simple_calculator_fsm`` collapsed to its
    LLM-driven extraction + evaluation steps). 2 oracle calls.
    """
    p = leaf(
        template=parse_prompt, input_vars=parse_input_vars, schema_ref=parse_schema_ref
    )
    c = leaf(
        template=compute_prompt,
        input_vars=compute_input_vars,
        schema_ref=compute_schema_ref,
    )
    return _chain(("parsed", p), ("computed", c))


# ---------------------------------------------------------------------------
# classifier_term (4 leaves) — analyze_domain → analyze_structure →
# identify_needs → recommend_strategy
# ---------------------------------------------------------------------------


def classifier_term(
    domain_prompt: str,
    structure_prompt: str,
    needs_prompt: str,
    recommendation_prompt: str,
    *,
    domain_input_vars: tuple[str, ...] = ("problem",),
    structure_input_vars: tuple[str, ...] = ("problem", "domain"),
    needs_input_vars: tuple[str, ...] = ("problem", "structure"),
    recommendation_input_vars: tuple[str, ...] = ("problem", "needs"),
    domain_schema_ref: str | None = None,
    structure_schema_ref: str | None = None,
    needs_schema_ref: str | None = None,
    recommendation_schema_ref: str | None = None,
) -> Term:
    """Reasoning-strategy classifier — domain → structure → needs → recommend.

    4-leaf let-chain. The last leaf returns a recommended
    ``ReasoningType`` string; callers can feed that into ``solve_term``'s
    ``classify_var`` (or use ``solve_term``'s outer App which expects a
    direct host-callable instead).

    4 oracle calls per ``Executor.run``.
    """
    d = leaf(
        template=domain_prompt,
        input_vars=domain_input_vars,
        schema_ref=domain_schema_ref,
    )
    s = leaf(
        template=structure_prompt,
        input_vars=structure_input_vars,
        schema_ref=structure_schema_ref,
    )
    n = leaf(
        template=needs_prompt, input_vars=needs_input_vars, schema_ref=needs_schema_ref
    )
    r = leaf(
        template=recommendation_prompt,
        input_vars=recommendation_input_vars,
        schema_ref=recommendation_schema_ref,
    )
    return _chain(
        ("domain", d),
        ("structure", s),
        ("needs", n),
        ("recommendation", r),
    )


# ---------------------------------------------------------------------------
# solve_term — outer orchestrator with 2 host-callable Apps
# ---------------------------------------------------------------------------


def solve_term(
    validation_prompt: str,
    final_prompt: str,
    *,
    classify_var: str = "classify",
    dispatch_var: str = "dispatch",
    problem_var: str = "problem",
    validation_input_vars: tuple[str, ...] = ("problem", "solution"),
    final_input_vars: tuple[str, ...] = ("problem", "validation"),
    validation_schema_ref: str | None = None,
    final_schema_ref: str | None = None,
) -> Term:
    """Outer orchestrator — classify → dispatch → validate → final.

    Shape::

        let_("strategy",   app(var(<classify_var>), var(<problem_var>)),
            let_("solution",   app(var(<dispatch_var>), var("strategy")),
                let_("validation", validation_leaf,
                    final_leaf)))

    The caller's ``env`` must bind:

    - ``problem_var`` (default ``"problem"``) — the problem string;
    - ``classify_var`` (default ``"classify"``) — callable
      ``problem -> strategy_name``; can be a Python wrapper around
      ``classifier_term``'s last let-binding, or a direct rules-based
      classifier;
    - ``dispatch_var`` (default ``"dispatch"``) — callable
      ``strategy_name -> solution``; typically wraps a registry of
      strategy terms (e.g. ``{"analytical": analytical_term(...), ...}``)
      and runs ``Executor.run(picked_term, env)``.

    This factory mirrors the slice-1 pattern ``tool_dispatch_var`` /
    ``plan_exec_var``: host-callables are env-bound rather than baked
    in. Theorem-2 oracle-call count = 2 (validate + final). The
    classify/dispatch Apps execute Python code, not the oracle.

    Returns a 2-leaf let-chain with 2 outer host-callable Apps.
    """
    v = leaf(
        template=validation_prompt,
        input_vars=validation_input_vars,
        schema_ref=validation_schema_ref,
    )
    f = leaf(
        template=final_prompt, input_vars=final_input_vars, schema_ref=final_schema_ref
    )
    return let_(
        "strategy",
        app(var(classify_var), var(problem_var)),
        let_(
            "solution",
            app(var(dispatch_var), var("strategy")),
            let_("validation", v, f),
        ),
    )
