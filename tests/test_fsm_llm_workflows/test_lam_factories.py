"""Shape-equivalence + purity tests for ``fsm_llm.stdlib.workflows.lam_factories``.

No LLM calls. Structural assertions on the AST returned by each factory.
Live smokes live in ``test_lam_factories_live.py``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from fsm_llm.runtime import App, Case, Combinator, Leaf, Let, leaf
from fsm_llm.stdlib.workflows import lam_factories
from fsm_llm.stdlib.workflows.lam_factories import (
    _chain,
    branch_term,
    linear_term,
    parallel_term,
    retry_term,
    switch_term,
)

# --- _chain helper --------------------------------------------------------


def test_chain_two_pairs() -> None:
    a = leaf(template="A: {x}", input_vars=("x",))
    b = leaf(template="B: {y}", input_vars=("y",))
    term = _chain(("first", a), ("second", b))
    assert isinstance(term, Let)
    assert term.name == "first"
    assert term.body is b


def test_chain_rejects_single_pair() -> None:
    a = leaf(template="A", input_vars=())
    with pytest.raises(ValueError):
        _chain(("only", a))


# --- purity ---------------------------------------------------------------


def test_purity_imports_only_runtime() -> None:
    """AST-walk: every Import / ImportFrom is either ``__future__`` or
    originates in ``fsm_llm.runtime``.

    ``fsm_llm.lam`` was removed in 0.6.0 (R13 epoch); ``fsm_llm.runtime`` is
    the canonical and only path for the λ-kernel.
    """
    path = Path(inspect.getfile(lam_factories))
    tree = ast.parse(path.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name != "__future__" and not n.name.startswith("fsm_llm.runtime"):
                    offenders.append(f"import {n.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "__future__":
                continue
            if mod != "fsm_llm.runtime":
                offenders.append(f"from {mod}")
    assert offenders == [], (
        f"Purity violation: workflows lam_factories must import only "
        f"from fsm_llm.runtime — found: {offenders}"
    )


# --- inventory ------------------------------------------------------------


def test_all_factories_in_module_all() -> None:
    expected = [
        "linear_term",
        "branch_term",
        "switch_term",
        "parallel_term",
        "retry_term",
    ]
    assert lam_factories.__all__ == expected


# --- shape: linear_term ---------------------------------------------------


def test_linear_term_shape() -> None:
    a = leaf(template="A: {input}", input_vars=("input",))
    b = leaf(template="B: {a}", input_vars=("a",))
    c = leaf(template="C: {b}", input_vars=("b",))
    term = linear_term(("a", a), ("b", b), ("c", c))
    assert isinstance(term, Let)
    assert term.name == "a"
    assert term.value is a
    inner = term.body
    assert isinstance(inner, Let) and inner.name == "b"
    assert inner.value is b
    assert inner.body is c  # last pair is body, name unused


def test_linear_term_rejects_one_pair() -> None:
    a = leaf(template="A", input_vars=())
    with pytest.raises(ValueError):
        linear_term(("a", a))


# --- shape: branch_term ---------------------------------------------------


def test_branch_term_shape() -> None:
    t = leaf(template="THEN: {input}", input_vars=("input",))
    e = leaf(template="ELSE: {input}", input_vars=("input",))
    term = branch_term("is_positive", t, e)
    assert isinstance(term, Case)
    # Scrutinee is App(Var("is_positive"), Var("input"))
    assert isinstance(term.scrutinee, App)
    # Branches contain "true" only, default = else
    assert "true" in term.branches
    assert term.branches["true"] is t
    assert term.default is e


def test_branch_term_custom_keys() -> None:
    t = leaf(template="T", input_vars=())
    e = leaf(template="E", input_vars=())
    term = branch_term("p", t, e, case_keys=("yes", "no"))
    assert isinstance(term, Case)
    assert "yes" in term.branches
    # default still set
    assert term.default is e


# --- shape: switch_term ---------------------------------------------------


def test_switch_term_shape() -> None:
    a = leaf(template="A", input_vars=())
    b = leaf(template="B", input_vars=())
    c = leaf(template="C", input_vars=())
    d = leaf(template="DEF", input_vars=())
    term = switch_term("classify", {"alpha": a, "beta": b, "gamma": c}, d)
    assert isinstance(term, Case)
    assert set(term.branches.keys()) == {"alpha", "beta", "gamma"}
    assert term.default is d


def test_switch_term_rejects_empty_branches() -> None:
    d = leaf(template="DEF", input_vars=())
    with pytest.raises(ValueError):
        switch_term("classify", {}, d)


# --- shape: parallel_term -------------------------------------------------


def test_parallel_term_shape() -> None:
    b1 = leaf(template="B1: {input}", input_vars=("input",))
    b2 = leaf(template="B2: {input}", input_vars=("input",))
    b3 = leaf(template="B3: {input}", input_vars=("input",))
    term = parallel_term([("b1", b1), ("b2", b2), ("b3", b3)])
    # Outer is the first let-binding (b1)
    assert isinstance(term, Let)
    assert term.name == "b1"
    assert term.value is b1
    # Walk through to find the eventual reduce_ Combinator at the bottom.
    cur = term
    seen_branches: list[str] = []
    while isinstance(cur, Let):
        seen_branches.append(cur.name)
        cur = cur.body
    # We expect to traverse through b1, b2, b3, branch_outputs (4 lets).
    assert seen_branches == ["b1", "b2", "b3", "branch_outputs"]
    # The innermost body is a Combinator (reduce_)
    assert isinstance(cur, Combinator)


def test_parallel_term_rejects_single_branch() -> None:
    b1 = leaf(template="B1", input_vars=())
    with pytest.raises(ValueError):
        parallel_term([("b1", b1)])


# --- shape: retry_term ----------------------------------------------------


def test_retry_term_shape() -> None:
    """retry_term returns App(Fix(...), Var(input)). No leaves are
    embedded — body and success are host-callables."""
    term = retry_term("body", "success")
    # Outer is App(Fix, Var(input))
    assert isinstance(term, App)
    # Walk: term.fn is Fix(Abs("self", Abs("x", Let("attempt", App(...), Case(...)))))
    fix_node = term.fn
    # Don't depend on Fix's exact attr name beyond the kind discriminator.
    assert fix_node.kind == "Fix"
    # No Leaf nodes anywhere.
    leaf_count = _count_leaves(term)
    assert leaf_count == 0, f"retry_term should have 0 leaves, got {leaf_count}"


def test_retry_term_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError):
        retry_term("body", "success", max_attempts=0)


# --- helper ---------------------------------------------------------------


def _count_leaves(term) -> int:
    if isinstance(term, Leaf):
        return 1
    if isinstance(term, Let):
        return _count_leaves(term.value) + _count_leaves(term.body)
    if isinstance(term, App):
        return _count_leaves(term.fn) + _count_leaves(term.arg)
    if isinstance(term, Case):
        n = _count_leaves(term.scrutinee)
        for v in term.branches.values():
            n += _count_leaves(v)
        if term.default is not None:
            n += _count_leaves(term.default)
        return n
    return 0
