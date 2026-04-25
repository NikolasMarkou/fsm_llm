from __future__ import annotations

"""
Private helper: shared recursive λ-term construction body for the
``long_context`` stdlib factories (``niah``, ``aggregate``, and slice-3
additions ``pairwise`` / ``multi_hop``).

This module is intentionally private (leading underscore). It is NOT
re-exported from ``stdlib.long_context.__init__`` and is not part of any
public API. Callers outside ``stdlib/long_context/`` should not import it.

Per D-S2-001: the ~12-line term construction body was duplicated across
``niah.py`` and ``aggregate.py`` until a 3rd factory landed (slice 3),
at which point extraction became net-positive. This module is the
extraction. Public factory signatures are unchanged; the AST produced by
delegating factories is byte-identical to the pre-refactor inline form.
"""

from fsm_llm.lam import (
    Term,
    abs_,
    app,
    case_,
    fix,
    fmap,
    leaf,
    reduce_,
    split,
)


def _recursive_long_context(
    leaf_prompt: str,
    *,
    tau: int,
    k: int,
    reduce_op_name: str,
    input_var: str = "document",
    extra_input_vars: tuple[str, ...] = (),
) -> Term:
    """Build the canonical recursive long-context λ-term.

    Semantics::

        fix(λself. λP.
           case size_bucket(P) of
             "small" → leaf(<leaf_prompt>, P)
             _      → reduce_(<reduce_op_name>, fmap(self, split(P, k))))

    Parameters
    ----------
    leaf_prompt:
        Fully-formatted leaf prompt template (any ``{question}`` or other
        caller-specific placeholders must already be substituted; the
        remaining ``{P}`` placeholder is filled by the executor at leaf
        invocation time).
    tau:
        Leaf-size threshold. Must be ``>= 1``. (Note: τ itself does not
        appear in the term; the caller's ``size_bucket`` env binding
        honours it. The guard here matches the pre-refactor factory
        contract.)
    k:
        Branching factor for SPLIT. Must be ``>= 2`` for non-degenerate
        recursion.
    reduce_op_name:
        Name of the REDUCE op to look up in env. Caller must bind a
        ``ReduceOp`` (or bare callable) under this name.
    input_var:
        Name of the env variable that holds the document string. Default
        ``"document"``.
    extra_input_vars:
        Extra env variable names that the leaf prompt template references
        in addition to the chunk variable ``P``. Each name must be bound
        in the executor env at run time. Default ``()`` (leaf consumes
        only ``P``). Used by ``multi_hop`` to thread previous-hop results
        into the leaf prompt via ``Let``-bound env bindings; ``niah`` and
        ``aggregate`` leave this empty.

    Returns
    -------
    A closed λ-term ready for ``Executor.run(program, env)``.

    Raises
    ------
    ValueError
        If ``tau < 1`` or ``k < 2``.
    """
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}")
    if k < 2:
        raise ValueError(f"k must be >= 2 for non-degenerate recursion, got {k}")

    body = abs_(
        "self",
        abs_(
            "P",
            case_(
                app("size_bucket", "P"),
                {"small": leaf(leaf_prompt, ("P", *tuple(extra_input_vars)))},
                default=reduce_(reduce_op_name, fmap("self", split("P", k))),
            ),
        ),
    )
    return app(fix(body), input_var)


__all__ = ["_recursive_long_context"]
