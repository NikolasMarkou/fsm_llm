"""Shape-equivalence + purity tests for ``fsm_llm.stdlib.reasoning.lam_factories``.

No LLM calls. Pure structural assertions on the AST returned by each
factory. Live smokes live in ``test_lam_factories_live.py``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from fsm_llm.lam import App, Leaf, Let, Term, Var, leaf
from fsm_llm.stdlib.reasoning import lam_factories
from fsm_llm.stdlib.reasoning.lam_factories import _chain

# --- _chain helper --------------------------------------------------------


class TestChainHelper:
    def test_chain_two_leaves(self) -> None:
        a = leaf(template="A: {x}", input_vars=("x",))
        b = leaf(template="B: {y}", input_vars=("y",))
        term = _chain(("first", a), ("second", b))
        assert isinstance(term, Let)
        assert term.name == "first"
        assert term.value is a
        assert term.body is b  # second's name is unused; b is the body

    def test_chain_four_leaves(self) -> None:
        leaves = [leaf(template=f"L{i}: {{x}}", input_vars=("x",)) for i in range(4)]
        pairs = [(f"step{i}", leaves[i]) for i in range(4)]
        term = _chain(*pairs)
        # Outer let
        assert isinstance(term, Let) and term.name == "step0"
        assert term.value is leaves[0]
        # Inner let1
        inner1 = term.body
        assert isinstance(inner1, Let) and inner1.name == "step1"
        assert inner1.value is leaves[1]
        # Inner let2
        inner2 = inner1.body
        assert isinstance(inner2, Let) and inner2.name == "step2"
        assert inner2.value is leaves[2]
        # Innermost body is the last leaf (step3 name is unused)
        assert inner2.body is leaves[3]

    def test_chain_rejects_single_pair(self) -> None:
        a = leaf(template="A", input_vars=())
        with pytest.raises(ValueError):
            _chain(("only", a))


# --- helpers --------------------------------------------------------------


def _count_leaves(term: Term) -> int:
    """Count Leaf nodes reachable through Let bodies (the only structure
    these factories produce). For solve_term we also walk through App
    nodes (host-callables don't contribute leaves)."""
    if isinstance(term, Leaf):
        return 1
    if isinstance(term, Let):
        return _count_leaves(term.value) + _count_leaves(term.body)
    if isinstance(term, App):
        # App's func/arg are typically Var refs — no leaves.
        return _count_leaves(term.fn) + _count_leaves(term.arg)
    if isinstance(term, Var):
        return 0
    # Anything else — be conservative.
    return 0


def _walk_let_names(term: Term) -> list[str]:
    out: list[str] = []
    cur = term
    while isinstance(cur, Let):
        out.append(cur.name)
        cur = cur.body
    return out


# --- purity ---------------------------------------------------------------


class TestPurity:
    def test_purity_imports_only_lam(self) -> None:
        """AST-walk the module and assert every Import / ImportFrom is
        either ``__future__`` or originates in ``fsm_llm.lam``."""
        path = Path(inspect.getfile(lam_factories))
        tree = ast.parse(path.read_text())
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name != "__future__" and not n.name.startswith("fsm_llm.lam"):
                        offenders.append(f"import {n.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "__future__":
                    continue
                if mod != "fsm_llm.lam":
                    offenders.append(f"from {mod}")
        assert offenders == [], (
            f"Purity violation: lam_factories must import only from "
            f"fsm_llm.lam — found: {offenders}"
        )


# --- 11-factory inventory -------------------------------------------------

EXPECTED_NAMES_AND_LEAVES = [
    ("analytical_term", 3),
    ("deductive_term", 3),
    ("inductive_term", 3),
    ("abductive_term", 3),
    ("analogical_term", 3),
    ("creative_term", 3),
    ("critical_term", 3),
    ("hybrid_term", 4),
    ("calculator_term", 2),
    ("classifier_term", 4),
    ("solve_term", 4),
]


class TestFactoryInventory:
    def test_all_factories_in_module_all(self) -> None:
        names = [n for n, _ in EXPECTED_NAMES_AND_LEAVES]
        assert lam_factories.__all__ == names, (
            f"__all__ mismatch. Expected {names}, got {lam_factories.__all__}"
        )


# --- 3-leaf strategy factories --------------------------------------------


THREE_LEAF_BIND_NAMES = {
    "analytical_term": ("decomposition", "analysis", "integration"),
    "deductive_term": ("premises", "inference", "conclusion"),
    "inductive_term": ("examples", "pattern", "generalization"),
    "abductive_term": ("observation", "hypothesis", "best_explanation"),
    "analogical_term": ("source_domain", "mapping", "target_inference"),
    "creative_term": ("divergence", "combination", "refinement"),
    "critical_term": ("examination", "evaluation", "verdict"),
}


@pytest.mark.parametrize(
    "fname",
    list(THREE_LEAF_BIND_NAMES.keys()),
)
def test_three_leaf_factory_shape(fname: str) -> None:
    fac = getattr(lam_factories, fname)
    term = fac(
        prompt_a="A: {problem}",
        prompt_b="B: {problem} {" + THREE_LEAF_BIND_NAMES[fname][0] + "}",
        prompt_c="C: {problem} {" + THREE_LEAF_BIND_NAMES[fname][1] + "}",
    )
    assert isinstance(term, Let)
    assert _count_leaves(term) == 3
    let_names = _walk_let_names(term)
    # Two outer let names — the 3rd leaf is the inner body, no let-name.
    expected = THREE_LEAF_BIND_NAMES[fname]
    assert let_names == [expected[0], expected[1]], (
        f"{fname}: let-binding names mismatch. Expected "
        f"{[expected[0], expected[1]]}, got {let_names}"
    )


# --- hybrid (4-leaf) ------------------------------------------------------


def test_hybrid_term_shape() -> None:
    term = lam_factories.hybrid_term(
        facets_prompt="F: {problem}",
        strategies_prompt="S: {problem} {facets}",
        execute_prompt="E: {problem} {strategies}",
        integrate_prompt="I: {problem} {execution}",
    )
    assert isinstance(term, Let)
    assert _count_leaves(term) == 4
    assert _walk_let_names(term) == ["facets", "strategies", "execution"]


# --- calculator (2-leaf) --------------------------------------------------


def test_calculator_term_shape() -> None:
    term = lam_factories.calculator_term(
        parse_prompt="Parse: {problem}",
        compute_prompt="Compute: {problem} {parsed}",
    )
    assert isinstance(term, Let)
    assert _count_leaves(term) == 2
    assert _walk_let_names(term) == ["parsed"]


# --- classifier (4-leaf) --------------------------------------------------


def test_classifier_term_shape() -> None:
    term = lam_factories.classifier_term(
        domain_prompt="D: {problem}",
        structure_prompt="S: {problem} {domain}",
        needs_prompt="N: {problem} {structure}",
        recommend_prompt="R: {problem} {needs}",
    )
    assert isinstance(term, Let)
    assert _count_leaves(term) == 4
    assert _walk_let_names(term) == ["domain", "structure", "needs"]


# --- solve_term (4-leaf, 2 host-callable Apps) ----------------------------


def test_solve_term_shape() -> None:
    term = lam_factories.solve_term(
        validate_prompt="V: {solution}",
        final_prompt="F: {problem} {validation}",
    )
    # Outer let binds "strategy" via App(classify_var, problem)
    assert isinstance(term, Let)
    assert term.name == "strategy"
    # term.value is App(classify_var, problem)
    assert isinstance(term.value, App)
    # Innermost has 2 real leaves (validate, final). Total leaves = 2.
    # _count_leaves walks Apps recursively; their children are Vars (0).
    assert _count_leaves(term) == 2


def test_solve_term_uses_app_for_dispatch() -> None:
    """Solve_term must use App nodes for both classify and dispatch
    host-callables (slice-1 ``tool_dispatch_var`` pattern)."""
    term = lam_factories.solve_term(
        validate_prompt="V: {solution}",
        final_prompt="F: {problem} {validation}",
    )
    # term: Let("strategy", App(Var("classify"), Var("problem")), inner1)
    assert isinstance(term, Let) and term.name == "strategy"
    assert isinstance(term.value, App)
    assert isinstance(term.value.fn, Var) and term.value.fn.name == "classify"

    # inner1: Let("solution", App(Var("dispatch"), Var("strategy")), inner2)
    inner1 = term.body
    assert isinstance(inner1, Let) and inner1.name == "solution"
    assert isinstance(inner1.value, App)
    assert isinstance(inner1.value.fn, Var)
    assert inner1.value.fn.name == "dispatch"
