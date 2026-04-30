from __future__ import annotations

"""Tests for fsm_llm.lam.dsl — builder → AST equivalence."""

import pytest

from fsm_llm.runtime.ast import (
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
from fsm_llm.runtime.dsl import (
    abs_,
    app,
    case_,
    concat,
    cross,
    ffilter,
    fix,
    fmap,
    leaf,
    let,
    peek,
    reduce,
    split,
    var,
)
from fsm_llm.runtime.errors import ASTConstructionError


class TestCoreForms:
    def test_var(self) -> None:
        assert var("x") == Var(name="x")

    def test_abs_body_str_becomes_var(self) -> None:
        assert abs_("x", "x") == Abs(param="x", body=Var(name="x"))

    def test_abs_body_term(self) -> None:
        inner = leaf("t", ("x",))
        assert abs_("x", inner) == Abs(param="x", body=inner)

    def test_app(self) -> None:
        assert app("f", "x") == App(fn=Var(name="f"), arg=Var(name="x"))

    def test_let(self) -> None:
        assert let("y", "x", "y") == Let(
            name="y", value=Var(name="x"), body=Var(name="y")
        )

    def test_case_no_default(self) -> None:
        c = case_("t", {"a": "x", "b": "y"})
        assert c == Case(
            scrutinee=Var(name="t"),
            branches={"a": Var(name="x"), "b": Var(name="y")},
        )
        assert c.default is None

    def test_case_with_default(self) -> None:
        c = case_("t", {"a": "x"}, default="z")
        assert c.default == Var(name="z")

    def test_fix_requires_abs(self) -> None:
        assert fix(abs_("self", "self")) == Fix(
            body=Abs(param="self", body=Var(name="self"))
        )

    def test_fix_rejects_non_abs(self) -> None:
        with pytest.raises(ASTConstructionError):
            fix("x")  # Var, not Abs

    def test_leaf_defaults(self) -> None:
        lf = leaf("hello {x}", ("x",))
        assert isinstance(lf, Leaf)
        assert lf.input_vars == ("x",)
        assert lf.schema_ref is None

    def test_leaf_all_fields(self) -> None:
        lf = leaf("t", ["x"], schema_ref="mod.Cls", model_override="gpt-4o")
        assert lf.input_vars == ("x",)
        assert lf.schema_ref == "mod.Cls"
        assert lf.model_override == "gpt-4o"


class TestCombinatorBuilders:
    def test_split_with_int_k(self) -> None:
        s = split("P", 2)
        assert s == Combinator(
            op=CombinatorOp.SPLIT,
            args=(Var(name="P"), Var(name="_const_2")),
        )

    def test_split_with_var_k(self) -> None:
        s = split("P", "k")
        assert s.args[1] == Var(name="k")

    def test_peek(self) -> None:
        p = peek("P", 100)
        assert p.op == CombinatorOp.PEEK
        assert p.args[1] == Var(name="_const_100")

    def test_fmap(self) -> None:
        m = fmap("f", "xs")
        assert m == Combinator(
            op=CombinatorOp.MAP, args=(Var(name="f"), Var(name="xs"))
        )

    def test_ffilter(self) -> None:
        f = ffilter("pred", "xs")
        assert f.op == CombinatorOp.FILTER

    def test_reduce(self) -> None:
        r = reduce("best", "xs")
        assert r.op == CombinatorOp.REDUCE
        assert r.args == (Var(name="best"), Var(name="xs"))

    def test_concat_varargs(self) -> None:
        c = concat("a", "b", "c")
        assert c.op == CombinatorOp.CONCAT
        assert c.args == (Var(name="a"), Var(name="b"), Var(name="c"))

    def test_cross(self) -> None:
        x = cross("xs", "ys")
        assert x.op == CombinatorOp.CROSS


class TestNestedComposition:
    def test_paper_style_fix_body(self) -> None:
        # fix(λself. λP. case size(P) of small → leaf_answer(P)
        #                                 _   → reduce(best, fmap(self, split(P, 2))))
        term = fix(
            abs_(
                "self",
                abs_(
                    "P",
                    case_(
                        "size_bucket",
                        {"small": leaf("answer {P}", ("P",))},
                        default=reduce("best", fmap("self", split("P", 2))),
                    ),
                ),
            )
        )
        assert isinstance(term, Fix)
        # Round-trips cleanly (inherits from AST tests; quick sanity here)
        assert Fix.model_validate(term.model_dump()) == term


class TestStrictCoercion:
    def test_raw_int_in_non_k_position_rejected(self) -> None:
        with pytest.raises(ASTConstructionError):
            app("f", 42)

    def test_raw_none_rejected(self) -> None:
        with pytest.raises(ASTConstructionError):
            abs_("x", None)
