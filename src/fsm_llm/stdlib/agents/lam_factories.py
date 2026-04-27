from __future__ import annotations

"""
λ-term factories for the four canonical M4-verified agent shapes.

This module exposes named factory functions that return closed
``fsm_llm.lam`` λ-terms ready for ``Executor.run``. Each factory
captures one of the four shapes proven across 46 pipeline examples
on ``ollama_chat/qwen3.5:4b``:

- ``react_term``     — S1 ReAct        (decide → tool_dispatch → synth, 2 oracle calls)
- ``rewoo_term``     — S2 REWOO/PE     (plan → plan_exec → synth,       2 oracle calls)
- ``reflexion_term`` — S3 Reflexion    (solve → eval → reflect → solve, 4 oracle calls)
- ``memory_term``    — S4 memory/orch  (context → answer,                2 oracle calls)

**Purity invariant** — this module imports ONLY from ``fsm_llm.lam``.
No imports of ``fsm_llm.llm``, ``fsm_llm.fsm``, or ``fsm_llm.pipeline``.
The factories close over no Python state; all dynamic values
(tools, plan executors, task strings) are bound by the caller in ``env``
when invoking ``Executor.run(term, env)``.

See ``docs/lambda.md`` §11 for the design rationale and ``plans/LESSONS.md``
for the M4 evidence corpus that anchors the four shapes.
"""

from fsm_llm.lam import Term, app, leaf, let_, var

__all__ = ["react_term", "rewoo_term", "reflexion_term", "memory_term"]


def react_term(
    decide_prompt: str,
    synth_prompt: str,
    *,
    decide_input_vars: tuple[str, ...] = ("task",),
    synth_input_vars: tuple[str, ...] = ("task", "decision", "observation"),
    decide_schema_ref: str | None = None,
    synth_schema_ref: str | None = None,
    tool_dispatch_var: str = "tool_dispatch",
) -> Term:
    """Build an S1 ReAct λ-term.

    Shape::

        let_("decision", decide_leaf,
            let_("observation", app(var(<tool_dispatch_var>), var("decision")),
                synth_leaf))

    The caller's ``env`` must bind:
    - every name listed in ``decide_input_vars`` and ``synth_input_vars``
      (typically ``"task"``);
    - ``tool_dispatch_var`` (default ``"tool_dispatch"``) to a callable
      ``decision -> observation``. See
      ``examples/pipeline/_helpers.make_tool_dispatcher`` for the
      canonical construction.

    Parameters
    ----------
    decide_prompt:
        Format string for the decide leaf — must contain placeholders for
        each ``decide_input_vars`` entry.
    synth_prompt:
        Format string for the synth leaf — placeholders for each
        ``synth_input_vars`` entry (``decision`` + ``observation`` come
        from the let-bindings above).
    decide_input_vars / synth_input_vars:
        Env-var names referenced by the leaf templates.
    decide_schema_ref / synth_schema_ref:
        Optional dotted Pydantic model paths for structured output.
    tool_dispatch_var:
        Env-var name holding the dispatcher callable.

    Returns
    -------
    Closed ``Term`` — 2 oracle calls per ``Executor.run``.
    """
    decide = leaf(
        template=decide_prompt,
        input_vars=decide_input_vars,
        schema_ref=decide_schema_ref,
    )
    synth = leaf(
        template=synth_prompt,
        input_vars=synth_input_vars,
        schema_ref=synth_schema_ref,
    )
    return let_(
        "decision",
        decide,
        let_("observation", app(var(tool_dispatch_var), var("decision")), synth),
    )


def rewoo_term(
    plan_prompt: str,
    synth_prompt: str,
    *,
    plan_input_vars: tuple[str, ...] = ("task",),
    synth_input_vars: tuple[str, ...] = ("task", "plan", "evidence"),
    plan_schema_ref: str | None = None,
    synth_schema_ref: str | None = None,
    plan_exec_var: str = "plan_exec",
) -> Term:
    """Build an S2 REWOO / plan-execute λ-term.

    Shape::

        let_("plan", plan_leaf,
            let_("evidence", app(var(<plan_exec_var>), var("plan")),
                synth_leaf))

    The caller's ``env`` must bind:
    - every name listed in ``plan_input_vars`` and ``synth_input_vars``;
    - ``plan_exec_var`` (default ``"plan_exec"``) to a callable
      ``plan -> evidence``. See
      ``examples/pipeline/_helpers.make_plan_executor``.

    Returns 2 oracle calls per ``Executor.run``.
    """
    plan_l = leaf(
        template=plan_prompt,
        input_vars=plan_input_vars,
        schema_ref=plan_schema_ref,
    )
    synth = leaf(
        template=synth_prompt,
        input_vars=synth_input_vars,
        schema_ref=synth_schema_ref,
    )
    return let_(
        "plan",
        plan_l,
        let_("evidence", app(var(plan_exec_var), var("plan")), synth),
    )


def reflexion_term(
    solve_prompt: str,
    eval_prompt: str,
    reflect_prompt: str,
    resolve_prompt: str,
    *,
    solve_input_vars: tuple[str, ...] = ("task",),
    eval_input_vars: tuple[str, ...] = ("task", "attempt1"),
    reflect_input_vars: tuple[str, ...] = ("task", "attempt1", "evaluation"),
    resolve_input_vars: tuple[str, ...] = ("task", "reflection"),
    solve_schema_ref: str | None = None,
    eval_schema_ref: str | None = None,
    reflect_schema_ref: str | None = None,
    resolve_schema_ref: str | None = None,
) -> Term:
    """Build an S3 Reflexion λ-term.

    Shape::

        let_("attempt1", solve,
            let_("evaluation", evaluate,
                let_("reflection", reflect,
                    re_solve)))

    Four leaves chained via three nested let-bindings. The caller's ``env``
    must bind every name listed in the ``*_input_vars`` tuples (typically
    just ``"task"``; the rest come from the let-bindings).

    Returns 4 oracle calls per ``Executor.run``.
    """
    solve = leaf(
        template=solve_prompt,
        input_vars=solve_input_vars,
        schema_ref=solve_schema_ref,
    )
    evaluate = leaf(
        template=eval_prompt,
        input_vars=eval_input_vars,
        schema_ref=eval_schema_ref,
    )
    reflect = leaf(
        template=reflect_prompt,
        input_vars=reflect_input_vars,
        schema_ref=reflect_schema_ref,
    )
    re_solve = leaf(
        template=resolve_prompt,
        input_vars=resolve_input_vars,
        schema_ref=resolve_schema_ref,
    )
    return let_(
        "attempt1",
        solve,
        let_(
            "evaluation",
            evaluate,
            let_("reflection", reflect, re_solve),
        ),
    )


def memory_term(
    context_prompt: str,
    answer_prompt: str,
    *,
    context_input_vars: tuple[str, ...] = ("task",),
    answer_input_vars: tuple[str, ...] = ("task", "context"),
    context_schema_ref: str | None = None,
    answer_schema_ref: str | None = None,
) -> Term:
    """Build an S4 memory / orchestrator λ-term.

    Shape::

        let_("context", ctx_leaf, ans_leaf)

    The simplest of the four canonical shapes — a 2-leaf chain. The
    caller's ``env`` must bind every name listed in
    ``context_input_vars`` (typically ``"task"``); ``"context"`` comes
    from the let-binding.

    Returns 2 oracle calls per ``Executor.run``.
    """
    ctx = leaf(
        template=context_prompt,
        input_vars=context_input_vars,
        schema_ref=context_schema_ref,
    )
    ans = leaf(
        template=answer_prompt,
        input_vars=answer_input_vars,
        schema_ref=answer_schema_ref,
    )
    return let_("context", ctx, ans)
