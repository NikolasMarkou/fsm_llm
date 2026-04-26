from __future__ import annotations

"""
Padded needle-in-haystack (NIAH) factory for non-τ·k^d-aligned inputs.

The canonical ``niah`` factory's planner-executor cost equality
(``ex.oracle_calls == plan(...).predicted_calls``) holds *exactly* only
when ``len(document) == τ · k^d`` for some integer ``d ≥ 0``. For
arbitrary user-supplied document sizes the leaf count drifts off
``k^d`` by however much the ceiling-divisions in ``split_impl``
truncate the rightmost chunks.

``niah_padded`` removes that constraint: it pre-pads the raw document
to ``N* = aligned_size(n, τ, k) = τ · k^ceil(log_k(n/τ))`` *before*
recursion begins, then delegates to the canonical recursive body. The
caller binds the *raw* document under ``raw_document`` (or any custom
``input_var``); the factory's term resolves the inner ``document`` via
a ``Let`` whose value is ``app(<pad callable>, <raw>)``.

Pad-budget contract
-------------------
Padding is bounded above by a factor of ``k`` in the worst case
(``n = τ·k^d + 1``); average-case overhead is ~1.5× depending on
distribution. Callers paying for cost-predictability accept this in
exchange for a deterministic ``predicted_calls = k^d`` against the
*padded* size. See ``decisions.md`` D-1 in the slice-4 plan.

Public surface
--------------
- ``niah_padded(question, *, tau, k, ...) → Term`` — the factory.
- ``aligned_size(n, tau, k) → int`` — pure math: returns the next
  ``τ·k^d`` boundary at or above ``n``.
- ``pad_to_aligned(doc, tau, k, *, pad_char=" ") → str`` — the direct
  string-padding helper (callers may use it ahead of plain ``niah``).
- ``make_pad_callable(tau, k, *, pad_char=" ") → Callable[[str], str]`` —
  closure used to construct the env binding consumed by the factory's
  ``Let``-bound term.

Purity
------
This module imports only from ``fsm_llm.lam`` and the sibling
``.niah`` module (for the canonical leaf prompt template). NO
``fsm_llm.{llm,fsm,pipeline}`` — same invariant as the rest of
``stdlib/long_context``.
"""

from collections.abc import Callable
from math import ceil, log

from fsm_llm.lam import Term, app, let_, var

from ._recursive import _recursive_long_context
from .niah import _NIAH_PROMPT_TEMPLATE


def aligned_size(n: int, tau: int, k: int) -> int:
    """Compute ``N* = τ · k^ceil(log_k(n/τ))``.

    The smallest value ``N* ≥ n`` of the form ``τ · k^d`` for some
    integer ``d ≥ 0``. When ``n ≤ τ`` the function returns ``n`` itself
    (the leaf branch produces a single oracle call regardless of
    padding, so no padding is required to keep cost-equality).

    Parameters
    ----------
    n:
        Raw document length (must be ``>= 0``).
    tau:
        Leaf-size threshold (must be ``>= 1``).
    k:
        Branching factor (must be ``>= 2``).

    Raises
    ------
    ValueError
        If any argument is out of range.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if n <= tau:
        return n
    # Smallest integer d with τ·k^d ≥ n  ⇒  d = ceil(log_k(n/τ)).
    # Use float log + ceil; correct integer-comparison fixup below to
    # guard against floating-point shaving (e.g. log_2(8/1) returning
    # 2.9999999...).
    d = max(0, ceil(log(n / tau) / log(k)))
    n_star = tau * (k**d)
    while n_star < n:
        d += 1
        n_star = tau * (k**d)
    return n_star


def pad_to_aligned(
    doc: str,
    tau: int,
    k: int,
    *,
    pad_char: str = " ",
) -> str:
    """Pad ``doc`` with ``pad_char`` until its length equals
    ``aligned_size(len(doc), tau, k)``.

    The returned string is the concatenation of ``doc`` and exactly
    ``N* - len(doc)`` copies of ``pad_char``. ``pad_char`` must be a
    single character (length 1) to keep the per-character padding
    semantics simple and the resulting length exact.

    For ``len(doc) <= tau`` this is a no-op (returns ``doc``
    unchanged), matching ``aligned_size`` semantics.
    """
    if not isinstance(pad_char, str) or len(pad_char) != 1:
        raise ValueError(
            f"pad_char must be a single character, got {pad_char!r}"
        )
    n_star = aligned_size(len(doc), tau, k)
    if n_star <= len(doc):
        return doc
    return doc + pad_char * (n_star - len(doc))


def make_pad_callable(
    tau: int,
    k: int,
    *,
    pad_char: str = " ",
) -> Callable[[str], str]:
    """Build a unary pad callable closed over ``(tau, k, pad_char)``.

    The returned function takes a raw document string and returns the
    padded version. Used as the env binding looked up by
    ``niah_padded``'s internal ``Let`` term.
    """
    if not isinstance(pad_char, str) or len(pad_char) != 1:
        raise ValueError(
            f"pad_char must be a single character, got {pad_char!r}"
        )
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    def _pad(doc: str) -> str:
        return pad_to_aligned(doc, tau, k, pad_char=pad_char)

    return _pad


def niah_padded(
    question: str,
    *,
    tau: int = 512,
    k: int = 2,
    reduce_op_name: str = "best",
    input_var: str = "raw_document",
    pad_callable_var: str = "pad_to_aligned",
    inner_input_var: str = "document",
) -> Term:
    """Build a padded NIAH λ-term for non-aligned input lengths.

    Semantics::

        let document = pad_to_aligned(raw_document) in
          fix(λself. λP.
            case size_bucket(P) of
              "small" → leaf(<niah prompt>, P)
              _      → reduce_(best, fmap(self, split(P, k))))

    The caller's env must bind:

    - ``input_var`` (default ``"raw_document"``) — the raw, unaligned
      document string.
    - ``pad_callable_var`` (default ``"pad_to_aligned"``) — a unary
      ``str → str`` pad callable (use ``make_pad_callable(tau, k)``).
    - ``"size_bucket"`` — same callable consumed by ``niah``.
    - ``reduce_op_name`` — the ``ReduceOp`` (default name ``"best"``).

    Theorem-2 contract
    ------------------
    For arbitrary raw length ``n``, ``ex.oracle_calls`` equals
    ``plan(PlanInputs(n=N*, ..., max_k=k)).predicted_calls`` where
    ``N* = aligned_size(n, tau, k)``. Note: callers must call ``plan``
    with ``n=N*`` (use ``aligned_size``) — passing the raw ``n`` would
    measure the wrong tree.

    Parameters
    ----------
    question:
        Question baked into the leaf prompt template.
    tau:
        Leaf-size threshold (chars).
    k:
        Branching factor.
    reduce_op_name:
        Env name of the ``ReduceOp`` (default ``"best"``).
    input_var:
        Env name of the raw document (default ``"raw_document"``).
    pad_callable_var:
        Env name of the pad callable (default ``"pad_to_aligned"``).
    inner_input_var:
        Internal name used by the inner recursive body for the padded
        document. Default ``"document"``. Renaming is rarely needed;
        the ``Let`` shadows any outer binding for the same name.

    Returns
    -------
    A closed λ-term ready for ``Executor.run(program, env)``.

    Raises
    ------
    ValueError
        If ``tau < 1`` or ``k < 2`` (delegated to the inner term
        constructor).
    """
    leaf_prompt = _NIAH_PROMPT_TEMPLATE.format(question=question)

    inner_term = _recursive_long_context(
        leaf_prompt,
        tau=tau,
        k=k,
        reduce_op_name=reduce_op_name,
        input_var=inner_input_var,
    )

    # let inner_input_var = pad_callable(input_var) in inner_term
    return let_(
        inner_input_var,
        app(var(pad_callable_var), var(input_var)),
        inner_term,
    )


__all__ = [
    "niah_padded",
    "aligned_size",
    "pad_to_aligned",
    "make_pad_callable",
]
