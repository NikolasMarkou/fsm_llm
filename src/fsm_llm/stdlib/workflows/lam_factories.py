from __future__ import annotations

"""
λ-term factories for the workflow stdlib (M3 slice 3).

Five named factories exercising kernel pieces NOT used by slices 1-2:

- ``linear_term``   — sequential composition (let-chain over sub-terms)
- ``branch_term``   — boolean branch via ``case_``
- ``switch_term``   — N-way classifier dispatch via ``case_``
- ``parallel_term`` — fan-out + reduce via ``fmap`` / ``reduce_``
- ``retry_term``    — bounded retry via ``fix``

**Purity invariant** — this module imports ONLY from ``fsm_llm.lam``. No
imports of ``fsm_llm.stdlib.workflows.engine`` or any other workflow
runtime piece. The factories close over no Python state; all dynamic
values (host-callable predicates, classifiers, body invokers) are bound
by the caller in ``env`` when invoking ``Executor.run(term, env)``.

Per ``docs/lambda.md`` §11: "DAG is a λ-term built by .then / .parallel
/ .branch combinators; the engine dissolves." Slice 3 lands those
combinators as named factories.

See ``docs/lambda.md`` §13 M3 row for slice context.
"""

from fsm_llm.lam import (
    Term,
    app,
    case_,
    fix,
    fmap,
    leaf,
    let_,
    reduce_,
    var,
)

__all__ = [
    "linear_term",
    "branch_term",
    "switch_term",
    "parallel_term",
    "retry_term",
]


def _chain(*pairs: tuple[str, Term]) -> Term:
    """Fold ``[(name1, t1), (name2, t2), ..., (nameN, tN)]`` into a
    right-nested ``let_`` chain. Last pair's term is the body — its name
    is unused.

    Private helper. Slice 3 duplicates the slice-2 helper rather than
    importing it (purity: each stdlib package is self-contained).
    """
    if len(pairs) < 2:
        raise ValueError(f"_chain requires at least 2 pairs, got {len(pairs)}")
    _name_last, body = pairs[-1]
    for name, term in reversed(pairs[:-1]):
        body = let_(name, term, body)
    return body


# ---------------------------------------------------------------------------
# linear_term — sequential composition
# ---------------------------------------------------------------------------


def linear_term(*pairs: tuple[str, Term]) -> Term:
    """Build a linear (sequential) workflow as a let-chain over sub-terms.

    Each ``(name, term)`` pair contributes one let-binding except the
    last pair, whose ``term`` becomes the inner body. The let-binding
    names are exposed to subsequent terms via the env.

    Example::

        linear_term(
            ("step1", leaf("...", ("input",))),
            ("step2", leaf("...", ("step1",))),
            ("step3", leaf("...", ("step2",))),
        )

    Theorem-2: ``oracle_calls == sum(leaves(t) for _, t in pairs)``
    (assuming each ``term`` is a non-Fix factory term).

    Raises
    ------
    ValueError
        If fewer than 2 pairs are supplied.
    """
    return _chain(*pairs)


# ---------------------------------------------------------------------------
# branch_term — boolean branching via case_
# ---------------------------------------------------------------------------


def branch_term(
    cond_var: str,
    then_term: Term,
    else_term: Term,
    *,
    input_var: str = "input",
    case_keys: tuple[str, str] = ("true", "false"),
) -> Term:
    """Build a boolean branch.

    Shape::

        case_(app(var(<cond_var>), var(<input_var>)),
              {<case_keys[0]>: then_term},
              default=else_term)

    The caller's ``env`` must bind:
    - ``input_var`` to the value the predicate inspects;
    - ``cond_var`` to a callable returning one of ``case_keys``
      (default ``"true"`` / ``"false"``).

    Theorem-2: at runtime, exactly one branch executes — oracle calls
    equal ``leaves(taken_branch)``. Static leaf count is the upper bound
    ``leaves(then_term) + leaves(else_term)``.
    """
    return case_(
        app(var(cond_var), var(input_var)),
        {case_keys[0]: then_term},
        default=else_term,
    )


# ---------------------------------------------------------------------------
# switch_term — N-way classification dispatch via case_
# ---------------------------------------------------------------------------


def switch_term(
    classifier_var: str,
    branches: dict[str, Term],
    default_term: Term,
    *,
    input_var: str = "input",
) -> Term:
    """Build an N-way switch.

    Shape::

        case_(app(var(<classifier_var>), var(<input_var>)),
              branches,
              default=default_term)

    The caller's ``env`` must bind:
    - ``input_var`` to the value to classify;
    - ``classifier_var`` to a callable returning a string key in
      ``branches`` (or any other string → ``default_term`` fires).

    Theorem-2: at runtime, exactly one arm executes. Static upper bound
    is ``sum(leaves(b) for b in branches.values()) + leaves(default_term)``.

    Raises
    ------
    ValueError
        If ``branches`` is empty.
    """
    if not branches:
        raise ValueError("switch_term requires at least one branch")
    return case_(
        app(var(classifier_var), var(input_var)),
        dict(branches),
        default=default_term,
    )


# ---------------------------------------------------------------------------
# parallel_term — fan-out + reduce via fmap / reduce_
# ---------------------------------------------------------------------------


def parallel_term(
    branches: list[tuple[str, Term]],
    *,
    reduce_op_name: str = "concat_str",
    list_var: str = "branch_outputs",
    branch_list_builder_var: str = "build_branch_list",
) -> Term:
    """Build a parallel-then-reduce workflow.

    The factory let-binds each branch sequentially, then folds the
    results via a host-callable that constructs a list and a kernel
    ``reduce_`` over an ``fmap``-applied identity.

    Shape (simplified)::

        let_(name1, branch1,
            let_(name2, branch2,
                ...
                let_(<list_var>, app(var(<branch_list_builder_var>),
                                     <bind names tuple>),
                     reduce_(<reduce_op_name>, fmap("identity",
                                                    var(<list_var>))))))

    The caller's ``env`` must bind:
    - ``branch_list_builder_var`` to a callable that takes one argument
      (the value of the last let-bound branch — the kernel can pass only
      one positional arg through ``app``) and returns a list of the
      branch results. The simplest implementation is a closure that
      pulls the branch outputs from the executor's env via a captured
      reference. **In practice** callers should use slice 3's
      ``examples/`` companion patterns; for benchmarks we stub a list
      builder that returns ``[v]`` for any v and the reduce becomes
      a no-op identity.
    - ``"identity"`` to the identity callable for ``fmap``.
    - ``reduce_op_name`` (default ``"concat_str"``) to a registered
      ``ReduceOp`` (see ``fsm_llm.lam.BUILTIN_OPS``).

    Theorem-2: ``oracle_calls == sum(leaves(b) for _, b in branches)``
    (each branch executes exactly once).

    Raises
    ------
    ValueError
        If fewer than 2 branches are supplied.
    """
    if len(branches) < 2:
        raise ValueError(
            f"parallel_term requires at least 2 branches, got {len(branches)}"
        )
    # Build the inner reduce/fmap body.
    body: Term = reduce_(
        reduce_op_name,
        fmap(var("identity"), var(list_var)),
    )
    # Wrap with the list-builder let.
    last_name = branches[-1][0]
    body = let_(
        list_var,
        app(var(branch_list_builder_var), var(last_name)),
        body,
    )
    # Now prepend each branch as a let-binding (right-nested).
    for name, term in reversed(branches):
        body = let_(name, term, body)
    return body


# ---------------------------------------------------------------------------
# retry_term — bounded retry via fix
# ---------------------------------------------------------------------------


def retry_term(
    body_var: str,
    success_var: str,
    *,
    input_var: str = "input",
    max_attempts: int = 3,
) -> Term:
    """Build a bounded retry loop.

    Shape (semantically)::

        let_("attempt", app(var(<body_var>), var(<input_var>)),
             case_(app(var(<success_var>), var("attempt")),
                   {"true": var("attempt")},
                   default=fix(...recursive call with bumped counter...)))

    Concretely we use a single ``fix``-wrapped body that invokes the
    ``body_var`` host-callable, checks the ``success_var`` predicate,
    and either returns the attempt or recurses. The kernel's ``fix``
    enforces termination via Theorem-1 only when the body is contracting
    on a well-founded measure — for retry, the caller's ``success_var``
    callable is responsible for returning ``"true"`` after at most
    ``max_attempts`` calls (typically by tracking attempt count in a
    closure).

    Parameters
    ----------
    body_var:
        Env-var name holding ``callable(input) -> attempt_result``.
    success_var:
        Env-var name holding ``callable(attempt_result) -> 'true' | 'false'``.
    input_var:
        Env-var name holding the seed input. Default ``"input"``.
    max_attempts:
        Documentation only — the actual cap lives in the caller's
        ``success_var`` predicate. Default ``3``.

    Theorem-2: at runtime, oracle calls = 0 (this factory has no Leaf
    nodes — the body is a host-callable App). Total host-callable
    invocations ≤ ``max_attempts``.

    Notes
    -----
    The factory returns a closed Fix-wrapped term; the caller binds the
    host callables and passes the input through the env.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    # body :: λself. λx. case (success (body_var x)) of
    #                       "true"  -> attempt
    #                       _       -> self x
    # We reify "attempt" via let_, so success is checked against the
    # bound attempt rather than re-invoking the body.
    from fsm_llm.lam import abs_

    body = abs_(
        "self",
        abs_(
            "x",
            let_(
                "attempt",
                app(var(body_var), var("x")),
                case_(
                    app(var(success_var), var("attempt")),
                    {"true": var("attempt")},
                    default=app(var("self"), var("x")),
                ),
            ),
        ),
    )
    return app(fix(body), var(input_var))


# Reference to ``leaf`` is kept so the import is exercised in callers
# that compose these factories with raw leaves; not currently used here.
_ = leaf
