from __future__ import annotations

"""Tests for the R5 ``HOST_CALL`` combinator op.

``HOST_CALL`` is the host-callable bridge: a ``Combinator(op=HOST_CALL,
args=(Var(name), *call_args))`` resolves ``Var(name)`` against the env to
a Python callable, evaluates the remaining args, and invokes it. NOT an
oracle call — used by R5 to splice handler hooks into the compiled term
and by R6 to keep the streaming response path on the host side.

Five tests:

1. Basic invocation — env-bound callable is called with evaluated args.
2. No oracle bookkeeping — ``ex.oracle_calls`` stays at 0 even when
   HOST_CALL is invoked many times.
3. Composes with other AST nodes — HOST_CALL inside Let bindings.
4. Error: env name not callable → ``ASTConstructionError``.
5. Error: first arg not a ``Var`` → ``ASTConstructionError``.
"""

from typing import Any

import pytest

from fsm_llm.runtime.ast import Combinator, CombinatorOp
from fsm_llm.runtime.dsl import host_call, leaf, let_, var
from fsm_llm.runtime.errors import ASTConstructionError
from fsm_llm.runtime.executor import Executor


class _MockOracle:
    """Tiny scripted oracle, only used in test 2 to ensure HOST_CALL does
    NOT consume oracle calls when invoked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(
        self,
        prompt: str,
        schema: Any = None,
        *,
        model_override: str | None = None,
    ) -> Any:
        self.calls.append(prompt)
        return f"answer({prompt})"

    def tokenize(self, text: str) -> int:
        return max(1, len(text))


def test_host_call_basic_invocation() -> None:
    """Bound callable receives evaluated arg values; result returned."""
    received: list[tuple[Any, ...]] = []

    def my_hook(*args: Any) -> str:
        received.append(args)
        return "ok"

    term = host_call("my_hook", var("x"), var("y"))
    ex = Executor()
    out = ex.run(term, env={"my_hook": my_hook, "x": 1, "y": "two"})
    assert out == "ok"
    assert received == [(1, "two")]


def test_host_call_no_oracle_bookkeeping() -> None:
    """HOST_CALL is host-side glue — never increments oracle_calls."""

    def double(n: int) -> int:
        return n * 2

    term = host_call("double", var("x"))
    oracle = _MockOracle()
    ex = Executor(oracle=oracle)  # type: ignore[arg-type]
    # Run repeatedly — each HOST_CALL must not bump the counter.
    for _ in range(5):
        ex.run(term, env={"double": double, "x": 3})
    assert ex.oracle_calls == 0
    assert oracle.calls == []


def test_host_call_composes_inside_let() -> None:
    """HOST_CALL inside a Let body — value flows through normally."""

    def upper(s: str) -> str:
        return s.upper()

    # let r = host_call(upper, x) in r
    term = let_("r", host_call("upper", var("x")), var("r"))
    ex = Executor()
    assert ex.run(term, env={"upper": upper, "x": "hello"}) == "HELLO"


def test_host_call_callable_name_must_resolve_to_callable() -> None:
    """Env binding must be callable — non-callable raises clearly."""
    term = host_call("not_callable", var("x"))
    ex = Executor()
    with pytest.raises(ASTConstructionError, match="not callable"):
        ex.run(term, env={"not_callable": 42, "x": 1})


def test_host_call_first_arg_must_be_var() -> None:
    """Hand-built AST with a non-Var head raises ASTConstructionError."""
    # We construct directly via Combinator/Leaf to bypass the dsl.host_call
    # builder which enforces Var-headedness statically.
    bad = Combinator(
        op=CombinatorOp.HOST_CALL,
        args=(
            leaf(template="not a var", input_vars=()),
        ),
    )
    ex = Executor()
    with pytest.raises(ASTConstructionError, match="first arg must be Var"):
        ex.run(bad, env={})


def test_host_call_dsl_rejects_empty_callable_name() -> None:
    """Builder-level validation: empty/None name rejected at construction."""
    with pytest.raises(ASTConstructionError, match="non-empty str"):
        host_call("")
    with pytest.raises(ASTConstructionError, match="non-empty str"):
        host_call(None)  # type: ignore[arg-type]


def test_host_call_zero_args_invokes_thunk() -> None:
    """HOST_CALL with only the callable (no extra args) works as a thunk."""
    calls: list[int] = []

    def thunk() -> str:
        calls.append(1)
        return "fired"

    term = host_call("thunk")
    ex = Executor()
    assert ex.run(term, env={"thunk": thunk}) == "fired"
    assert calls == [1]
