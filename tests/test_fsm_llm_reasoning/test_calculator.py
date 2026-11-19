"""The calculator's evaluator is total: a number or a ``ValueError``, never anything else."""

from __future__ import annotations

import ast
import inspect
import math

import pytest

from fsm_llm.reasoning import calculator
from fsm_llm.reasoning.calculator import (
    MAX_EXPONENT,
    MAX_EXPRESSION_CHARS,
    MAX_EXPRESSION_NODES,
    calculate,
    evaluate_expression,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2 + 3", 5),
        ("0.15 * 240", 36),  # an integer-valued float is the int, not "36.0"
        ("10 / 2", 5),
        ("7 / 2", 3.5),
        ("2 ** 3 ** 2", 512),
        ("-2 ** 2", -4),
        ("(3 + 4) * 5 / 2", 17.5),
        ("-7 // 2", -4),
        ("-7 % 3", 2),
        ("1_000 + 1", 1001),
        ("0.1 + 0.2", 0.1 + 0.2),  # the IEEE answer, unrounded
        ("5 - 5", 0),  # zero is an answer
        ("2 ** -1", 0.5),
        ("2 ** 1000", 2**1000),
        ("123456789 * 987654321", 123456789 * 987654321),
        ("+ 4", 4),
        (" 1 + " * 24 + "1 ", 25),  # a long ordinary sum
        ("2 ** 52 * 1.0", 2**52),  # an integer-valued float below 2**53: the int
        ("2 ** 53 * 1.0", 2.0**53),  # from 2**53 up it stays a float
        ("1e23", 1e23),
    ],
)
def test_evaluates(text, expected):
    value = evaluate_expression(text)
    assert value == expected
    assert type(value) is type(expected)


# Every input ends in a value or a ValueError (``;`` separates the plain ones).
_PLAIN = """
9**9**9;(9**1000)**1000;10**1000 * 10**1000;2**1000 ** 1000;1e308*10;(-8)**0.5;0**-1;0.0**-1
1//0;1%0;1/0;1.0//0;True+1;0x10;0b11;1j;x;abs(1);(1).real;(lambda: 1)();f"{1}";(x:=1)
[i for i in range(3)];not 1;1<2;1 if 1 else 2;-True;+True;"a";"a"*3;b"a";None;...;1 and 2
2^3;6&3;6|3;~1;1<<5;2@3;1,2;[1];1e999;1e-999;.5;5.;3 4;1 +;;15% of 240;1,000+1;import os
__import__('os');nan;1e200**1000;10.0**1000;10**400*1.5;10**400/1;10**400//1.5
((9**100)**100)**100;(9**100)**100;2**1001;(-2)**1000;2**-1000;2**-1001;2**0.5;(-1)**0.5
(-1)**2.5;0**0;9**999 * 9**999;2**2047+2**2047;2**2048*2;2**1000*2**1000*2**1000
1e308 % 1e-308;1e308 // 1e-308;1e308 / 1e-308;1.7e308+1.7e308;-0.0;5.5//2;1/3;2/3*3
10**-1001;2**1000.5;2**-1e9;2**1e999;1.0000001**1000;10**616;10**617;3**1000;7**1000
(2**1000)**2;1e308+1e308-1e308;1e308*10-1e308*10;1e16;2.0**1000**1;1 + 1 +1+1
"""
_TEXTS = [t for line in _PLAIN.strip().splitlines() for t in line.split(";")] + [
    "9" * 4300,
    "9" * 4301,
    "9" * 100_000 + " + 1",
    "9" * 190 + " ** 1000",
    "0x" + "f" * 100_000,
    "0x" + "f" * 190 + " ** 1000",
    "(" * 99 + "1" + ")" * 99,
    "(" * 201 + "1" + ")" * 201,
    "(" * 100_000 + "1" + ")" * 100_000,
    "-" * 199 + "1",
    "-" * 1_000_000 + "1",
    "1 + " * 500_000 + "1",
    "2 ** " * 66 + "2",
    "2 ** " * 300_000 + "2",
    " " * 1_000_000 + "1",
    "1 + " * 99 + "1",
    "1\n+ 2",
    "1 + \x00 1",
    "١ + ٢",
    "5 − 3",
    "5 × 3",
]


@pytest.mark.parametrize("text", _TEXTS, ids=lambda text: repr(text[:24]))
def test_the_evaluator_is_total(text):
    try:
        value = evaluate_expression(text)
    except ValueError:
        return
    assert type(value) in (int, float)
    assert isinstance(value, int) or math.isfinite(value)
    str(value)  # never trips the int-to-str limit


_REFUSED = """
15% of 240;1 +;x;abs(1);(1).real;1 < 2;'a' * 3;True + 1;-True;None;1j;2 ^ 3;1 << 5
(-8) ** 0.5;1 / 0;1 // 0;1 % 0;0 ** -1;1e999;10.0 ** 1000;10 ** 400 * 1.5;2 ** 1001
9 ** 9 ** 9;(9 ** 1000) ** 1000;10 ** 617
"""


@pytest.mark.parametrize("text", [*_REFUSED.replace("\n", ";").split(";"), "1\n+ 2"])
def test_refusals_are_value_errors(text):
    with pytest.raises(ValueError):
        evaluate_expression(text)


@pytest.mark.parametrize(
    "value", [None, 5, 5.0, ["1+1"], {"a": 1}, b"1+1", True, "", "  "]
)
def test_a_non_text_value_is_no_expression(value):
    with pytest.raises(ValueError, match="no expression"):
        evaluate_expression(value)
    assert calculate(value) == (None, "no expression")


def test_the_answer_or_the_reason():
    assert calculate("(1 + 2) * 3") == (9, None)
    assert calculate("5 - 5") == (0, None)
    assert calculate("10 / 0") == (None, "division by zero")


def test_messages_are_fixed_text_not_python_text():
    """CPython words ZeroDivisionError and OverflowError differently per version (3.14 changed them)."""
    for text, message in [
        ("1 / 0", "division by zero"),
        ("1 // 0", "division by zero"),
        ("0 ** -1", "division by zero"),
        ("10.0 ** 1000", "result out of range"),
        ("10 ** 400 * 1.5", "result out of range"),
        ("2 ** 1001", "exponent too large"),
        ("2 ** 1000 ** 2", "exponent too large"),
        ("1+" * 70 + "1", "expression too large"),
        ("1" * 201, "expression too long"),
        ("2 ^ 3", "not an arithmetic expression"),
        ("(-8) ** 0.5", "result is not a real number"),
    ]:
        with pytest.raises(ValueError) as caught:
            evaluate_expression(text)
        assert str(caught.value) == message, text
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ or caught.value.__context__ is None


def test_the_text_bound_holds_before_the_parser_runs(monkeypatch):
    assert evaluate_expression("1" * MAX_EXPRESSION_CHARS) > 0  # the cap is inclusive

    def explode(*args, **kwargs):
        raise AssertionError("parsed an over-long expression")

    monkeypatch.setattr(calculator.ast, "parse", explode)
    with pytest.raises(ValueError, match="too long"):
        evaluate_expression("1" * (MAX_EXPRESSION_CHARS + 1))


def test_the_node_budget_is_inclusive():
    terms = (MAX_EXPRESSION_NODES + 1) // 2
    assert evaluate_expression("+".join(["1"] * terms)) == terms
    with pytest.raises(ValueError, match="too large"):
        evaluate_expression("+".join(["1"] * (terms + 1)))


def test_a_power_that_cannot_fit_is_refused_before_it_is_computed(monkeypatch):
    """Computing ``(2 ** 1000) ** 1000`` takes ~0.2 s and 1 Mbit; the bound refuses it first."""
    real = calculator._OPERATORS[ast.Pow]

    def guarded(left, right):
        assert not (isinstance(left, int) and left.bit_length() * right > 4096)
        return real(left, right)

    monkeypatch.setitem(calculator._OPERATORS, ast.Pow, guarded)
    with pytest.raises(ValueError, match="out of range"):
        evaluate_expression("(2 ** 1000) ** 1000")


def test_the_exponent_bound_is_inclusive():
    assert evaluate_expression(f"1 ** {MAX_EXPONENT}") == 1
    with pytest.raises(ValueError, match="exponent"):
        evaluate_expression(f"1 ** {MAX_EXPONENT + 1}")


def test_the_largest_integer_still_prints():
    assert (
        len(str(evaluate_expression("10 ** 616"))) == 617
    )  # 10 ** 617 is refused above


def test_the_module_never_evaluates_text():
    """No eval, exec, compile or ``__import__`` call: the walk of the AST is the whole evaluator."""
    called = {
        node.func.id
        for node in ast.walk(ast.parse(inspect.getsource(calculator)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"eval", "exec", "compile", "__import__", "getattr"}
