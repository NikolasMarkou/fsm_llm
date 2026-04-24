from __future__ import annotations

"""
Pure runtime implementations of the 7 combinators (ℒ ∖ {𝓜}).

All impls are:

- **Total** — every input in the declared domain maps to an output. No
  unchecked exceptions. ``TerminationError`` is the only exception these
  impls raise, and only when SPLIT fails to strictly reduce rank (a
  structural violation, not a runtime partiality).
- **Deterministic** — no randomness, no I/O, no clock reads.
- **Side-effect-free** — no mutation of inputs.

Inputs to combinators are ordinary Python values (str, list, tuple, int),
NOT AST terms. The executor is responsible for evaluating AST args to
values before calling the impls here.

The ``ReduceOp`` registry carries associativity flags per named operator.
REDUCE uses ``functools.reduce`` internally; we don't need associativity
to execute, but we record it so tests and future parallel-reduce backends
can rely on it.
"""

import functools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import TerminationError


# --------------------------------------------------------------
# ReduceOp registry
# --------------------------------------------------------------


@dataclass(frozen=True)
class ReduceOp:
    """A named binary operation for REDUCE.

    ``associative`` must be truthful — REDUCE relies on it for correctness
    when the backend parallelises the fold (M1 doesn't, M5 likely will).
    ``unit`` is the identity element used when reducing an empty list;
    ``None`` means an empty input raises ``TerminationError``.
    """

    name: str
    fn: Callable[[Any, Any], Any]
    associative: bool = True
    unit: Any = None


# A minimal set of built-in ops; users can construct more.
BUILTIN_OPS: dict[str, ReduceOp] = {
    "sum": ReduceOp(name="sum", fn=lambda a, b: a + b, associative=True, unit=0),
    "max": ReduceOp(name="max", fn=max, associative=True, unit=None),
    "min": ReduceOp(name="min", fn=min, associative=True, unit=None),
    "concat_str": ReduceOp(
        name="concat_str", fn=lambda a, b: a + b, associative=True, unit=""
    ),
    "concat_list": ReduceOp(
        name="concat_list",
        fn=lambda a, b: list(a) + list(b),
        associative=True,
        unit=[],
    ),
}


# --------------------------------------------------------------
# Combinator implementations
# --------------------------------------------------------------


def split_impl(p: Any, k: int) -> list[Any]:
    """SPLIT: partition ``p`` into ``≤ k`` pieces of roughly equal rank.

    - For ``str``: character-wise slicing into ``k`` near-equal chunks.
    - For list/tuple: element-wise chunking.
    - For other types: returns ``[p]`` (identity).

    Invariant: for any returned piece ``q``, ``|q| < |p|`` unless ``|p|``
    is already ≤ 1 (base case — the caller's Fix planner should have
    detected this and taken the τ-threshold branch). If the invariant
    is violated, raise ``TerminationError``.
    """
    if not isinstance(k, int) or k < 1:
        raise TerminationError(
            f"split: k must be a positive int, got {k!r} ({type(k).__name__})"
        )

    if isinstance(p, (str, list, tuple)):
        n = len(p)
        if n <= 1 or k == 1:
            # E1: identity. Rank not strictly reduced — caller must NOT
            # recurse without a base-case guard. Return [p] so fmap/reduce
            # degrade gracefully.
            return [p]
        # Near-equal chunks: ceiling division.
        chunk_size = (n + k - 1) // k
        pieces = [p[i : i + chunk_size] for i in range(0, n, chunk_size)]
        # Rank-reduction check — every piece strictly smaller than the whole.
        for q in pieces:
            if len(q) >= n:
                raise TerminationError(
                    "split: produced a piece with rank >= parent "
                    f"(|q|={len(q)}, |p|={n}); T1 termination invariant violated"
                )
        return pieces

    # Non-sized types: identity.
    return [p]


def peek_impl(p: Any, size: int) -> Any:
    """PEEK: return a bounded prefix of length ``size`` (or less)."""
    if not isinstance(size, int) or size < 0:
        raise TerminationError(f"peek: size must be a non-negative int, got {size!r}")
    if isinstance(p, (str, list, tuple)):
        return p[:size]
    return p


def map_impl(f: Callable[[Any], Any], xs: Iterable[Any]) -> list[Any]:
    """MAP: apply ``f`` to each element of ``xs``. Returns a list."""
    return [f(x) for x in xs]


def filter_impl(pred: Callable[[Any], bool], xs: Iterable[Any]) -> list[Any]:
    """FILTER: keep elements where ``pred`` is truthy."""
    return [x for x in xs if pred(x)]


def reduce_impl(op: ReduceOp | Callable[[Any, Any], Any], xs: Sequence[Any]) -> Any:
    """REDUCE: fold ``xs`` using ``op``.

    ``op`` may be a ``ReduceOp`` (preferred — carries associativity +
    unit) or a bare callable ``(a, b) -> c``. Empty input with a
    ``ReduceOp`` whose ``unit`` is set returns the unit; else raises
    ``TerminationError`` (E2).
    """
    xs_list = list(xs)
    if isinstance(op, ReduceOp):
        if not xs_list:
            if op.unit is None:
                raise TerminationError(
                    f"reduce: empty input with op {op.name!r} has no unit"
                )
            return op.unit
        return functools.reduce(op.fn, xs_list)
    # Bare callable
    if not xs_list:
        raise TerminationError("reduce: empty input with bare callable op has no unit")
    return functools.reduce(op, xs_list)


def concat_impl(*xs: Iterable[Any]) -> list[Any]:
    """CONCAT: flatten a variadic sequence of iterables into a list."""
    out: list[Any] = []
    for x in xs:
        out.extend(x)
    return out


def cross_impl(xs: Iterable[Any], ys: Iterable[Any]) -> list[tuple[Any, Any]]:
    """CROSS: cartesian product as a list of tuples. Materialised — M1
    does not lazy-iterate here so ``len(cross(xs, ys)) = |xs| * |ys|``
    for immediate use in downstream combinators."""
    ys_list = list(ys)
    return [(x, y) for x in xs for y in ys_list]


__all__ = [
    "ReduceOp",
    "BUILTIN_OPS",
    "split_impl",
    "peek_impl",
    "map_impl",
    "filter_impl",
    "reduce_impl",
    "concat_impl",
    "cross_impl",
]
