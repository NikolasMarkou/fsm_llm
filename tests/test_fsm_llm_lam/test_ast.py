from __future__ import annotations

"""Tests for fsm_llm.lam.ast — construction, frozenness, round-trip."""

import pytest
from pydantic import ValidationError

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
    is_term,
)


class TestConstruction:
    def test_var(self) -> None:
        v = Var(name="x")
        assert v.name == "x"
        assert v.kind == "Var"

    def test_abs_app(self) -> None:
        a = Abs(param="x", body=Var(name="x"))
        app = App(fn=a, arg=Var(name="y"))
        assert app.fn is a
        assert isinstance(app.arg, Var)

    def test_let(self) -> None:
        l_ = Let(name="y", value=Var(name="x"), body=Var(name="y"))
        assert l_.name == "y"

    def test_case(self) -> None:
        c = Case(
            scrutinee=Var(name="t"),
            branches={"a": Var(name="x"), "b": Var(name="y")},
            default=Var(name="z"),
        )
        assert set(c.branches) == {"a", "b"}
        assert c.default is not None

    def test_case_no_default(self) -> None:
        c = Case(scrutinee=Var(name="t"), branches={"a": Var(name="x")})
        assert c.default is None

    def test_combinator_all_ops(self) -> None:
        for op in CombinatorOp:
            k = Combinator(op=op, args=(Var(name="x"),))
            assert k.op == op
            assert k.args == (Var(name="x"),)

    def test_fix(self) -> None:
        f = Fix(body=Abs(param="self", body=Var(name="self")))
        assert isinstance(f.body, Abs)

    def test_leaf_unstructured(self) -> None:
        leaf = Leaf(template="summarise {p}", input_vars=("p",))
        assert leaf.schema_ref is None
        assert leaf.input_vars == ("p",)

    def test_leaf_structured(self) -> None:
        leaf = Leaf(
            template="extract {p}",
            input_vars=("p",),
            schema_ref="my.module.MySchema",
            model_override="gpt-4o",
        )
        assert leaf.schema_ref == "my.module.MySchema"
        assert leaf.model_override == "gpt-4o"


class TestFrozen:
    def test_var_is_frozen(self) -> None:
        v = Var(name="x")
        with pytest.raises(ValidationError):
            v.name = "y"  # type: ignore[misc]

    def test_abs_is_frozen(self) -> None:
        a = Abs(param="x", body=Var(name="x"))
        with pytest.raises(ValidationError):
            a.param = "y"  # type: ignore[misc]

    def test_leaf_is_frozen(self) -> None:
        leaf = Leaf(template="t", input_vars=("x",))
        with pytest.raises(ValidationError):
            leaf.template = "u"  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Var(name="x", extra_field=1)  # type: ignore[call-arg]


class TestEquality:
    def test_structural_equality(self) -> None:
        a = Abs(param="x", body=Var(name="x"))
        b = Abs(param="x", body=Var(name="x"))
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality_on_structure(self) -> None:
        assert Var(name="x") != Var(name="y")


class TestRoundtrip:
    def test_simple_roundtrip(self) -> None:
        app = App(
            fn=Abs(param="x", body=Var(name="x")),
            arg=Var(name="y"),
        )
        dumped = app.model_dump()
        restored = App.model_validate(dumped)
        assert restored == app

    def test_deep_roundtrip(self) -> None:
        term = Fix(
            body=Abs(
                param="self",
                body=Case(
                    scrutinee=Var(name="p"),
                    branches={
                        "small": Leaf(template="solve {p}", input_vars=("p",)),
                    },
                    default=Combinator(
                        op=CombinatorOp.REDUCE,
                        args=(
                            Var(name="best"),
                            Combinator(
                                op=CombinatorOp.MAP,
                                args=(
                                    Var(name="self"),
                                    Combinator(
                                        op=CombinatorOp.SPLIT,
                                        args=(Var(name="p"),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        dumped = term.model_dump()
        restored = Fix.model_validate(dumped)
        assert restored == term


class TestHelpers:
    def test_is_term_true(self) -> None:
        assert is_term(Var(name="x"))
        assert is_term(Leaf(template="t"))

    def test_is_term_false(self) -> None:
        assert not is_term("x")
        assert not is_term(42)
        assert not is_term(None)
