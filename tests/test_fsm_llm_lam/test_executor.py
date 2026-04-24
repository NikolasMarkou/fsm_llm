from __future__ import annotations

"""Tests for fsm_llm.lam.executor — β-reduction + Fix trampoline + I1/I5."""

from typing import Any
from unittest.mock import Mock

import pytest

from fsm_llm.lam.ast import (
    Abs,
    App,
    Case,
    Combinator,
    CombinatorOp,
    Fix,
    Leaf,
    Let,
    Var,
)
from fsm_llm.lam.combinators import BUILTIN_OPS
from fsm_llm.lam.dsl import (
    abs_,
    app,
    case_,
    concat,
    cross,
    ffilter,
    fix,
    fmap,
    leaf,
    let_,
    peek,
    reduce_,
    split,
    var,
)
from fsm_llm.lam.errors import ASTConstructionError, TerminationError
from fsm_llm.lam.executor import Executor
from fsm_llm.lam.oracle import Oracle


class _MockOracle:
    """Deterministic scripted oracle for executor tests."""

    def __init__(self, responses: list[Any] | None = None, K: int = 10_000) -> None:
        self.responses = list(responses or [])
        self.calls: list[str] = []
        self._K = K

    def invoke(
        self,
        prompt: str,
        schema: Any = None,
        *,
        model_override: str | None = None,
    ) -> Any:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return f"answer({prompt})"

    def tokenize(self, text: str) -> int:
        return max(1, len(text))

    def context_window(self) -> int:
        return self._K


# --------------------------------------------------------------
# Basic dispatch
# --------------------------------------------------------------


class TestVar:
    def test_var_from_env(self) -> None:
        ex = Executor()
        assert ex.run(Var(name="x"), {"x": 42}) == 42

    def test_var_unbound_raises(self) -> None:
        ex = Executor()
        with pytest.raises(ASTConstructionError, match="unbound"):
            ex.run(Var(name="x"), {})

    def test_var_const_int(self) -> None:
        ex = Executor()
        assert ex.run(Var(name="_const_42"), {}) == 42


class TestAbsApp:
    def test_identity(self) -> None:
        ex = Executor()
        # (λx. x) 7 = 7
        term = App(fn=Abs(param="x", body=Var(name="x")), arg=Var(name="y"))
        assert ex.run(term, {"y": 7}) == 7

    def test_constant_fn(self) -> None:
        ex = Executor()
        term = App(
            fn=Abs(param="x", body=Var(name="c")),
            arg=Var(name="x"),
        )
        assert ex.run(term, {"x": 1, "c": 99}) == 99


class TestLet:
    def test_let(self) -> None:
        ex = Executor()
        # let y = 5 in y
        term = Let(name="y", value=Var(name="_const_5"), body=Var(name="y"))
        assert ex.run(term, {}) == 5


class TestCase:
    def test_case_hit(self) -> None:
        ex = Executor()
        term = Case(
            scrutinee=Var(name="tag"),
            branches={"a": Var(name="_const_1"), "b": Var(name="_const_2")},
        )
        assert ex.run(term, {"tag": "a"}) == 1
        assert ex.run(term, {"tag": "b"}) == 2

    def test_case_default(self) -> None:
        ex = Executor()
        term = Case(
            scrutinee=Var(name="tag"),
            branches={"a": Var(name="_const_1")},
            default=Var(name="_const_99"),
        )
        assert ex.run(term, {"tag": "unknown"}) == 99

    def test_case_no_branch_no_default_raises(self) -> None:
        ex = Executor()
        term = Case(
            scrutinee=Var(name="tag"),
            branches={"a": Var(name="_const_1")},
        )
        with pytest.raises(ASTConstructionError):
            ex.run(term, {"tag": "unknown"})


class TestCombinators:
    def test_split(self) -> None:
        ex = Executor()
        # split("abcd", 2) → chunks summing to "abcd"
        result = ex.run(split("p", 2), {"p": "abcd"})
        assert "".join(result) == "abcd"
        assert len(result) <= 2

    def test_peek(self) -> None:
        ex = Executor()
        assert ex.run(peek("p", 3), {"p": "abcdef"}) == "abc"

    def test_map_with_closure(self) -> None:
        ex = Executor()
        # fmap(λx.x, xs) → xs
        doubler = abs_("x", "x")
        assert ex.run(fmap(doubler, "xs"), {"xs": [1, 2, 3]}) == [1, 2, 3]

    def test_map_with_python_callable(self) -> None:
        ex = Executor()
        # Inject a Python callable via env.
        assert ex.run(fmap("f", "xs"), {"f": lambda x: x * 10, "xs": [1, 2]}) == [
            10,
            20,
        ]

    def test_filter(self) -> None:
        ex = Executor()
        assert ex.run(
            ffilter("p", "xs"), {"p": lambda x: x > 1, "xs": [0, 1, 2, 3]}
        ) == [2, 3]

    def test_reduce(self) -> None:
        ex = Executor()
        assert ex.run(
            reduce_("op", "xs"), {"op": BUILTIN_OPS["sum"], "xs": [1, 2, 3]}
        ) == 6

    def test_concat(self) -> None:
        ex = Executor()
        assert ex.run(concat("a", "b"), {"a": [1, 2], "b": [3]}) == [1, 2, 3]

    def test_cross(self) -> None:
        ex = Executor()
        assert ex.run(cross("xs", "ys"), {"xs": [1], "ys": ["a", "b"]}) == [
            (1, "a"),
            (1, "b"),
        ]


# --------------------------------------------------------------
# Leaf dispatch + I1
# --------------------------------------------------------------


class TestLeafDispatch:
    def test_leaf_unstructured(self) -> None:
        oracle = _MockOracle(responses=["bonjour"])
        ex = Executor(oracle=oracle)
        l = leaf("translate {msg}", ("msg",))
        assert ex.run(l, {"msg": "hello"}) == "bonjour"
        assert oracle.calls == ["translate hello"]
        assert ex.oracle_calls == 1

    def test_leaf_unbound_var_raises(self) -> None:
        ex = Executor(oracle=_MockOracle())
        l = leaf("q {x}", ("x",))
        with pytest.raises(ASTConstructionError):
            ex.run(l, {})

    def test_leaf_without_oracle_raises(self) -> None:
        ex = Executor(oracle=None)
        with pytest.raises(ASTConstructionError, match="no oracle"):
            ex.run(leaf("t"), {})


class TestLeafOnlyOracleInvariant:
    """SC4 — executor with no Leaf subtree makes zero oracle calls."""

    def test_no_leaf_zero_oracle_calls(self) -> None:
        oracle = _MockOracle()
        ex = Executor(oracle=oracle)

        # Complex AST with every node kind EXCEPT Leaf.
        term = let_(
            "xs",
            "raw",
            reduce_(
                "op",
                fmap(
                    abs_("x", "x"),
                    ffilter(
                        "pred",
                        concat("xs", "xs"),
                    ),
                ),
            ),
        )
        result = ex.run(
            term,
            {
                "raw": [1, 2, 3],
                "op": BUILTIN_OPS["sum"],
                "pred": lambda x: x > 0,
            },
        )
        assert result == 1 + 2 + 3 + 1 + 2 + 3
        assert ex.oracle_calls == 0
        assert oracle.calls == []


# --------------------------------------------------------------
# Fix + I5 (max_depth)
# --------------------------------------------------------------


class TestFixBoundedDepth:
    def test_fix_base_case_immediate(self) -> None:
        """Fix whose body returns without self-recursion."""
        ex = Executor(oracle=_MockOracle())
        # fix(λself. λP. leaf(...) )   applied to "data"
        # The self is never referenced, so it's a d=0 case.
        term = app(fix(abs_("self", abs_("P", leaf("answer {P}", ("P",))))), "input")
        result = ex.run(term, {"input": "data"})
        assert result == "answer(answer data)"  # mock default echoes prompt
        assert ex.oracle_calls == 1

    def test_fix_one_level_recursion(self) -> None:
        """Fix that calls self once on a smaller input, then stops."""
        ex = Executor(oracle=_MockOracle(responses=["LEAF_A", "LEAF_B"]))

        # body: λself. λP. case size(P) of
        #                       "small" → leaf(answer P)
        #                       _       → self(peek P 1)
        # Env supplies: size(P) maps len-5 → "big", len-1 → "small".
        def size_of(p: Any) -> str:
            return "small" if len(p) <= 1 else "big"

        body = abs_(
            "self",
            abs_(
                "P",
                case_(
                    app("size", "P"),
                    {"small": leaf("answer {P}", ("P",))},
                    default=app("self", peek("P", 1)),
                ),
            ),
        )
        term = app(fix(body), "input")
        result = ex.run(term, {"input": "hello", "size": size_of})
        # Exactly one leaf call total (the small-case).
        assert ex.oracle_calls == 1
        assert result == "LEAF_A"

    def test_fix_max_depth_cap(self) -> None:
        """SC5: pathological non-rank-reducing self-call hits max_depth."""
        ex = Executor(oracle=_MockOracle(), max_depth=5)
        # body: λself. λP. self(P)   — never decreases, never returns
        body = abs_("self", abs_("P", app("self", "P")))
        term = app(fix(body), "anything")
        with pytest.raises(TerminationError, match="max_depth"):
            ex.run(term, {"anything": "x"})

    def test_fix_records_plan(self) -> None:
        ex = Executor(oracle=_MockOracle())
        term = app(fix(abs_("self", abs_("P", leaf("q {P}", ("P",))))), "input")
        ex.run(term, {"input": "data"})
        # One plan recorded at top-level Fix entry.
        assert len(ex.plans) == 1
        assert ex.plans[0].k_star >= 1


# --------------------------------------------------------------
# Cost accumulation
# --------------------------------------------------------------


class TestCostAccumulation:
    def test_cost_recorded_per_leaf(self) -> None:
        ex = Executor(oracle=_MockOracle())
        l = leaf("q {x}", ("x",))
        ex.run(l, {"x": "hi"})
        assert ex.cost_accumulator.total_calls == 1

    def test_cost_isolated_between_runs(self) -> None:
        ex = Executor(oracle=_MockOracle())
        l = leaf("q {x}", ("x",))
        ex.run(l, {"x": "hi"})
        ex.cost_accumulator.reset()
        ex.run(l, {"x": "hi"})
        assert ex.cost_accumulator.total_calls == 1


# --------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------


class TestMockOracleProtocol:
    def test_conforms(self) -> None:
        assert isinstance(_MockOracle(), Oracle)
