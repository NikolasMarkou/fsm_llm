from __future__ import annotations

"""
Python builder DSL for the λ-AST.

Every helper returns an AST node from ``ast.py``. There is no string
λ-calculus parser in M1 (per ``findings/lambda-md-m1-spec.md``).

Combinator builder names follow the paper ("reduce_", "fmap", "ffilter"
with the trailing underscore or doubled-f to avoid colliding with Python
builtins ``reduce``, ``map``, ``filter``). ``split``, ``peek``, ``concat``,
``cross`` keep their paper names.

The builders accept either AST nodes or — for convenience in nested
expressions — they wrap raw strings via ``Var(name=...)`` when used in a
position that unambiguously expects a variable reference. All other raw
Python values must be wrapped explicitly by the caller; the DSL is
deliberately strict to keep the AST layer free of implicit conversions.
"""

from typing import Any

from .ast import (
    Abs,
    App,
    Case,
    Combinator,
    CombinatorOp,
    Fix,
    Leaf,
    Let,
    Term,
    Var,
    is_term,
)
from .errors import ASTConstructionError


def _as_term(x: Any, *, context: str) -> Term:
    """Coerce a Var-name string to ``Var``; pass terms through; reject else."""
    if isinstance(x, str):
        return Var(name=x)
    if is_term(x):
        return x  # type: ignore[return-value]
    raise ASTConstructionError(
        f"{context}: expected a Term or variable-name str, got {type(x).__name__}"
    )


# --------------------------------------------------------------
# Core λ forms
# --------------------------------------------------------------


def var(name: str) -> Var:
    """Variable reference."""
    return Var(name=name)


def abs_(param: str, body: Any) -> Abs:
    """λ-abstraction. Trailing underscore avoids the ``abs`` builtin."""
    return Abs(param=param, body=_as_term(body, context="abs_.body"))


def app(fn: Any, arg: Any) -> App:
    """Application."""
    return App(
        fn=_as_term(fn, context="app.fn"),
        arg=_as_term(arg, context="app.arg"),
    )


def let_(name: str, value: Any, body: Any) -> Let:
    """``let name = value in body``. Trailing underscore avoids Python's
    ``let`` being a keyword-adjacent name in readers' minds."""
    return Let(
        name=name,
        value=_as_term(value, context="let_.value"),
        body=_as_term(body, context="let_.body"),
    )


def case_(
    scrutinee: Any,
    branches: dict[str, Any],
    default: Any | None = None,
) -> Case:
    """Finite discrimination."""
    typed_branches = {
        k: _as_term(v, context=f"case_.branches[{k!r}]")
        for k, v in branches.items()
    }
    d = _as_term(default, context="case_.default") if default is not None else None
    return Case(
        scrutinee=_as_term(scrutinee, context="case_.scrutinee"),
        branches=typed_branches,
        default=d,
    )


def fix(body: Any) -> Fix:
    """Bounded recursion. ``body`` must be an ``Abs`` (we validate here
    rather than letting it fail opaquely deep inside the executor)."""
    term = _as_term(body, context="fix.body")
    if not isinstance(term, Abs):
        raise ASTConstructionError(
            f"fix: body must be an Abs (λ), got {type(term).__name__}"
        )
    return Fix(body=term)


def leaf(
    template: str,
    input_vars: tuple[str, ...] | list[str] = (),
    schema_ref: str | None = None,
    model_override: str | None = None,
) -> Leaf:
    """The ONLY oracle-calling node (I1).

    ``template`` is a format string whose placeholders must match
    ``input_vars``; the executor substitutes env bindings at call time.
    ``schema_ref`` is a dotted import path to a pydantic ``BaseModel``
    used for structured output. ``None`` → unstructured generation.
    """
    return Leaf(
        template=template,
        input_vars=tuple(input_vars),
        schema_ref=schema_ref,
        model_override=model_override,
    )


# --------------------------------------------------------------
# Combinator builders (paper names)
# --------------------------------------------------------------


def split(p: Any, k: Any) -> Combinator:
    """SPLIT p into ≤ k pieces. ``k`` can be a literal int (wrapped as
    ``Var(name=str(k))`` is wrong here — we want it as a Term literal);
    we wrap ints into a special lambda-side constant by routing through
    ``Var`` with a sentinel name... not a good path. Instead, ints are
    encoded positionally as a second arg of type Combinator-free via a
    ``Leaf`` is also wrong. The cleanest M1 encoding: literal ints live
    in the env as named bindings, so callers pass a var name or the
    python int directly which we wrap as a Var("_k_{int}"). We don't
    want that either.

    Chosen encoding: if ``k`` is an int, we store it as a synthetic
    ``Var(name=f"_const_{k}")`` and the executor's SPLIT dispatcher
    resolves the ``_const_`` prefix specially. This keeps the AST purely
    terms-of-terms without adding a separate Lit node (deferred to M2).
    """
    if isinstance(k, int):
        k_term: Term = Var(name=f"_const_{k}")
    else:
        k_term = _as_term(k, context="split.k")
    return Combinator(
        op=CombinatorOp.SPLIT,
        args=(_as_term(p, context="split.p"), k_term),
    )


def peek(p: Any, size: Any) -> Combinator:
    """PEEK: take a bounded prefix of ``p`` of length ``size``."""
    if isinstance(size, int):
        size_term: Term = Var(name=f"_const_{size}")
    else:
        size_term = _as_term(size, context="peek.size")
    return Combinator(
        op=CombinatorOp.PEEK,
        args=(_as_term(p, context="peek.p"), size_term),
    )


def fmap(f: Any, xs: Any) -> Combinator:
    """MAP: apply ``f`` to each element of ``xs``."""
    return Combinator(
        op=CombinatorOp.MAP,
        args=(_as_term(f, context="fmap.f"), _as_term(xs, context="fmap.xs")),
    )


def ffilter(pred: Any, xs: Any) -> Combinator:
    """FILTER: keep elements of ``xs`` for which ``pred`` returns true."""
    return Combinator(
        op=CombinatorOp.FILTER,
        args=(
            _as_term(pred, context="ffilter.pred"),
            _as_term(xs, context="ffilter.xs"),
        ),
    )


def reduce_(op: Any, xs: Any) -> Combinator:
    """REDUCE: fold ``xs`` using associative op ``op``.

    ``op`` may be either a Term (resolved to a callable at runtime via
    the env — typically a Var whose binding is a ``ReduceOp`` instance)
    or a string that gets wrapped as a Var."""
    return Combinator(
        op=CombinatorOp.REDUCE,
        args=(_as_term(op, context="reduce_.op"), _as_term(xs, context="reduce_.xs")),
    )


def concat(*xs: Any) -> Combinator:
    """CONCAT: flatten a sequence of lists."""
    return Combinator(
        op=CombinatorOp.CONCAT,
        args=tuple(_as_term(x, context=f"concat.args[{i}]") for i, x in enumerate(xs)),
    )


def cross(xs: Any, ys: Any) -> Combinator:
    """CROSS: cartesian product."""
    return Combinator(
        op=CombinatorOp.CROSS,
        args=(_as_term(xs, context="cross.xs"), _as_term(ys, context="cross.ys")),
    )


__all__ = [
    "var",
    "abs_",
    "app",
    "let_",
    "case_",
    "fix",
    "leaf",
    "split",
    "peek",
    "fmap",
    "ffilter",
    "reduce_",
    "concat",
    "cross",
]
