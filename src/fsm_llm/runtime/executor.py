# ruff: noqa: RUF002, RUF003
from __future__ import annotations

"""
β-reduction executor for the λ-kernel.

Strategy-pattern dispatch over AST node types. A single public entry
point, :py:meth:`Executor.run`, walks an AST term under an environment
and returns a Python value.

Invariant wiring (enforced structurally):

- **I1 (Leaf-only oracle)**: the ``Oracle`` reference is a private
  attribute on ``Executor`` and is read ONLY inside ``_eval_leaf``. No
  other dispatch method touches it. A static audit (see
  ``test_executor.py::test_no_leaf_zero_oracle_calls``) verifies that
  an AST with no Leaf subtree produces zero oracle calls.
- **I5 (bounded Fix)**: the ``Fix`` trampoline is guarded by
  ``max_depth``. Exceeding the cap raises ``TerminationError`` BEFORE
  the next Leaf call is issued.
- **I6 (immutable AST)**: the executor never mutates an AST node; it
  only builds fresh environments.

Variable encoding notes:

- ``Var(name="_const_<int>")`` resolves to the integer ``<int>``. This
  is the M1 encoding for integer literals embedded by the DSL's
  ``split(k=2)`` and ``peek(size=100)`` helpers. The prefix is reserved.
- All other Var names must be bound in the environment. Unbound refs
  raise ``ASTConstructionError``.

The planner is invoked at each Fix entry. It's a pure function, so
repeated calls are cheap and deterministic. Per-Fix plans are recorded
on the executor for post-run inspection.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
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
)
from .combinators import (
    concat_impl,
    cross_impl,
    filter_impl,
    map_impl,
    peek_impl,
    reduce_impl,
    split_impl,
)
from .constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_DEPTH, DEFAULT_TAU
from .cost import CostAccumulator
from .errors import ASTConstructionError, OracleError, TerminationError
from .oracle import Oracle, StreamingOracle
from .planner import Plan, PlanInputs
from .planner import plan as plan_fn

# --------------------------------------------------------------
# Closure type — a λ-abstraction captured with its defining env.
# --------------------------------------------------------------


@dataclass(frozen=True)
class _Closure:
    """Runtime value for an ``Abs``: the body + captured environment."""

    param: str
    body: Term
    env: dict[str, Any]


# A recursive Fix produces a callable that re-applies the executor; we
# represent it as a regular Python callable so downstream combinators
# (like MAP) can invoke it uniformly via ``_apply``.


Env = dict[str, Any]


# --------------------------------------------------------------
# Executor
# --------------------------------------------------------------


@dataclass
class Executor:
    """β-reduction interpreter.

    Construct with an ``Oracle`` to enable Leaf evaluation; pass
    ``oracle=None`` for pure-ℒ∖{𝓜} runs (useful for tests that want to
    prove no oracle call was made — see SC4).
    """

    oracle: Oracle | None = None
    max_depth: int = DEFAULT_MAX_DEPTH
    context_window: int = DEFAULT_CONTEXT_WINDOW
    default_tau: int = DEFAULT_TAU
    cost_accumulator: CostAccumulator = field(default_factory=CostAccumulator)
    # Per-run plan log — one entry per Fix invocation, for audit.
    plans: list[Plan] = field(default_factory=list)
    # Internal call counter; tests assert this against plan.predicted_calls.
    _oracle_calls: int = 0
    # A.D4(b) — caller's streaming intent for the current run. Set by ``run``
    # at entry, read by ``_eval_leaf``. Default False preserves byte-equivalent
    # behaviour for every existing caller. A streaming Leaf only routes
    # through ``oracle.invoke_stream`` when BOTH this flag AND ``Leaf.streaming``
    # are True AND the bound oracle satisfies ``StreamingOracle``.
    _stream: bool = False

    # ----- public entry point -----

    def run(self, term: Term, env: Env | None = None, *, stream: bool = False) -> Any:
        """Evaluate ``term`` under ``env`` and return the resulting value.

        When ``stream=True``, any ``Leaf`` node with ``streaming=True`` whose
        bound oracle implements ``StreamingOracle`` will dispatch through
        ``oracle.invoke_stream`` and return an ``Iterator[str]`` as its value.
        Non-streaming Leaves (``streaming=False``) and non-streaming oracles
        continue to use ``oracle.invoke`` regardless of ``stream``. This allows
        a single compiled term to mix streaming response Leaves with
        non-streaming extraction Leaves under one ``Executor.run`` call.
        """
        self.plans = []
        self._oracle_calls = 0
        self._stream = stream
        return self._eval(term, env or {}, _fix_depth=0)

    @property
    def oracle_calls(self) -> int:
        """Count of successful ``Oracle.invoke`` calls in the last run."""
        return self._oracle_calls

    # ----- dispatch -----

    def _eval(self, term: Term, env: Env, *, _fix_depth: int) -> Any:
        # Strategy-pattern dispatch — each branch handles exactly one
        # node kind. No fall-through; unknown nodes raise structurally
        # (pydantic's tagged-union validation guarantees we never see
        # one in practice).
        if isinstance(term, Var):
            return self._eval_var(term, env)
        if isinstance(term, Abs):
            return _Closure(param=term.param, body=term.body, env=dict(env))
        if isinstance(term, App):
            return self._eval_app(term, env, _fix_depth=_fix_depth)
        if isinstance(term, Let):
            # Let x = v in b  ≡  App(Abs(x, b), v)
            v = self._eval(term.value, env, _fix_depth=_fix_depth)
            new_env = {**env, term.name: v}
            return self._eval(term.body, new_env, _fix_depth=_fix_depth)
        if isinstance(term, Case):
            return self._eval_case(term, env, _fix_depth=_fix_depth)
        if isinstance(term, Combinator):
            return self._eval_combinator(term, env, _fix_depth=_fix_depth)
        if isinstance(term, Fix):
            return self._eval_fix(term, env, _fix_depth=_fix_depth)
        if isinstance(term, Leaf):
            return self._eval_leaf(term, env)
        raise ASTConstructionError(
            f"unknown term type {type(term).__name__}"
        )  # pragma: no cover

    # ----- leaf dispatch (THE ONLY PLACE 𝓜 IS CALLED — I1) -----

    def _eval_leaf(self, term: Leaf, env: Env) -> Any:
        if self.oracle is None:
            raise ASTConstructionError(
                "Executor has no oracle but AST contains a Leaf node"
            )
        # Substitute env bindings into the template.
        try:
            subs = {v: env[v] for v in term.input_vars}
        except KeyError as e:
            raise ASTConstructionError(
                f"Leaf references unbound variable {e.args[0]!r}; "
                f"env keys: {sorted(env.keys())}"
            ) from e
        try:
            prompt = term.template.format(**subs)
        except (KeyError, IndexError) as e:
            raise ASTConstructionError(f"Leaf template formatting failed: {e}") from e

        # Resolve optional structured-output schema.
        schema: Any = None
        if term.schema_ref is not None:
            # Local import to avoid circular import at module load time.
            from .oracle import _resolve_schema

            schema = _resolve_schema(term.schema_ref)

        # A.D4(b) — streaming branch. All four conditions must hold:
        # caller asked for streaming, this Leaf is streaming-capable, the
        # bound oracle implements StreamingOracle, and (defensive) schema is
        # None (mid-stream schema enforcement is unreliable per
        # runtime/oracle.py:120-128 and is gated out at the compiler per
        # plan_2026-04-28_ca542489 D-005). On all-True, route through
        # invoke_stream and return Iterator[str]; otherwise fall through to
        # the standard invoke path.
        if self._stream and term.streaming:
            if not isinstance(self.oracle, StreamingOracle):
                raise OracleError(
                    "streaming Leaf requires a StreamingOracle, but the bound "
                    f"oracle {type(self.oracle).__name__} does not implement "
                    "invoke_stream"
                )
            stream_iter = self.oracle.invoke_stream(
                prompt, schema=schema, model_override=term.model_override
            )
            self._oracle_calls += 1
            # Cost telemetry deliberately skipped: tokenizing an iterator
            # before consumption would force exhaustion. The dialog wrapper
            # (CB_APPEND_HISTORY post-D2) records cumulative tokens at
            # iterator-exhaustion time. Mirrors the legacy streaming path
            # which carried no per-chunk cost telemetry either.
            return stream_iter

        # Invoke the oracle.
        result = self.oracle.invoke(
            prompt, schema=schema, model_override=term.model_override
        )
        self._oracle_calls += 1

        # Record cost. Token counts approximated from prompt + str(result).
        try:
            tokens_in = self.oracle.tokenize(prompt)
            out_text = result if isinstance(result, str) else str(result)
            tokens_out = self.oracle.tokenize(out_text)
        except (TypeError, ValueError, AttributeError):
            # Tokeniser probe failed — provider doesn't expose a tokenizer
            # or the text triggered an encoder edge case. Skip the cost
            # record rather than fail the leaf evaluation.
            tokens_in = tokens_out = 0
        self.cost_accumulator.record(
            leaf_id=term.template,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=0.0,
        )
        return result

    # ----- other dispatchers (NO oracle access) -----

    def _eval_var(self, term: Var, env: Env) -> Any:
        # Integer-literal sentinel injected by the DSL.
        if term.name.startswith("_const_"):
            suffix = term.name[len("_const_") :]
            try:
                return int(suffix)
            except ValueError as e:
                raise ASTConstructionError(
                    f"malformed _const_ literal: {term.name!r}"
                ) from e
        if term.name not in env:
            raise ASTConstructionError(
                f"unbound variable {term.name!r}; env keys: {sorted(env.keys())}"
            )
        return env[term.name]

    def _eval_app(self, term: App, env: Env, *, _fix_depth: int) -> Any:
        fn = self._eval(term.fn, env, _fix_depth=_fix_depth)
        arg = self._eval(term.arg, env, _fix_depth=_fix_depth)
        return self._apply(fn, arg, _fix_depth=_fix_depth)

    def _apply(self, fn: Any, arg: Any, *, _fix_depth: int) -> Any:
        if isinstance(fn, _Closure):
            new_env = {**fn.env, fn.param: arg}
            return self._eval(fn.body, new_env, _fix_depth=_fix_depth)
        if callable(fn):
            return fn(arg)
        raise ASTConstructionError(
            f"cannot apply non-callable value of type {type(fn).__name__}"
        )

    def _eval_case(self, term: Case, env: Env, *, _fix_depth: int) -> Any:
        scrut_val = self._eval(term.scrutinee, env, _fix_depth=_fix_depth)
        key = str(scrut_val)
        if key in term.branches:
            return self._eval(term.branches[key], env, _fix_depth=_fix_depth)
        if term.default is not None:
            return self._eval(term.default, env, _fix_depth=_fix_depth)
        raise ASTConstructionError(
            f"case: scrutinee {scrut_val!r} not in branches {sorted(term.branches)} "
            "and no default provided"
        )

    # ----- combinator dispatch -----

    def _eval_combinator(self, term: Combinator, env: Env, *, _fix_depth: int) -> Any:
        # HOST_CALL has bespoke arg evaluation: the FIRST arg must be a Var
        # whose name resolves to a Python callable in env; we resolve it via
        # _eval (so _const_ literals etc. cannot be smuggled in as
        # callables), then evaluate the remaining args under the same env
        # and invoke. No oracle bookkeeping — HOST_CALL is host-side glue.
        if term.op is CombinatorOp.HOST_CALL:
            if not term.args:
                raise ASTConstructionError(
                    "HOST_CALL expects at least 1 arg (callable Var)"
                )
            head = term.args[0]
            if not isinstance(head, Var):
                raise ASTConstructionError(
                    f"HOST_CALL: first arg must be Var (callable name); "
                    f"got {type(head).__name__}"
                )
            fn = self._eval(head, env, _fix_depth=_fix_depth)
            if not callable(fn):
                raise ASTConstructionError(
                    f"HOST_CALL: env binding {head.name!r} is not callable "
                    f"(got {type(fn).__name__})"
                )
            call_args = [
                self._eval(a, env, _fix_depth=_fix_depth) for a in term.args[1:]
            ]
            return fn(*call_args)

        # Evaluate each arg under the current env.
        vals = [self._eval(a, env, _fix_depth=_fix_depth) for a in term.args]

        if term.op is CombinatorOp.SPLIT:
            if len(vals) != 2:
                raise ASTConstructionError(
                    f"SPLIT expects 2 args (p, k), got {len(vals)}"
                )
            return split_impl(vals[0], vals[1])
        if term.op is CombinatorOp.PEEK:
            if len(vals) != 2:
                raise ASTConstructionError(
                    f"PEEK expects 2 args (p, size), got {len(vals)}"
                )
            return peek_impl(vals[0], vals[1])
        if term.op is CombinatorOp.MAP:
            if len(vals) != 2:
                raise ASTConstructionError(
                    f"MAP expects 2 args (f, xs), got {len(vals)}"
                )
            fn_val, xs_val = vals
            fn_wrap = self._wrap_callable(fn_val, _fix_depth=_fix_depth)
            return map_impl(fn_wrap, xs_val)
        if term.op is CombinatorOp.FILTER:
            if len(vals) != 2:
                raise ASTConstructionError(
                    f"FILTER expects 2 args (pred, xs), got {len(vals)}"
                )
            pred_val, xs_val = vals
            pred_wrap = self._wrap_callable(pred_val, _fix_depth=_fix_depth)
            return filter_impl(pred_wrap, xs_val)
        if term.op is CombinatorOp.REDUCE:
            if len(vals) != 2:
                raise ASTConstructionError(
                    f"REDUCE expects 2 args (op, xs), got {len(vals)}"
                )
            op_val, xs_val = vals
            return reduce_impl(op_val, xs_val)
        if term.op is CombinatorOp.CONCAT:
            return concat_impl(*vals)
        if term.op is CombinatorOp.CROSS:
            if len(vals) != 2:
                raise ASTConstructionError(
                    f"CROSS expects 2 args (xs, ys), got {len(vals)}"
                )
            return cross_impl(vals[0], vals[1])
        raise ASTConstructionError(
            f"unknown combinator op: {term.op!r}"
        )  # pragma: no cover

    def _wrap_callable(self, value: Any, *, _fix_depth: int) -> Callable[[Any], Any]:
        """Turn a Closure / Python callable into a unary callable that
        invokes the executor on its result. Ensures that recursive Fix
        applications and lambda bodies re-enter ``_eval`` rather than
        being treated as opaque values."""
        if isinstance(value, _Closure):

            def _invoke(x: Any, cl: _Closure = value) -> Any:
                return self._apply(cl, x, _fix_depth=_fix_depth)

            return _invoke
        if callable(value):
            return value  # type: ignore[no-any-return]
        raise ASTConstructionError(
            f"expected a callable for MAP/FILTER arg, got {type(value).__name__}"
        )

    # ----- Fix trampoline (I5 enforced) -----

    def _eval_fix(self, term: Fix, env: Env, *, _fix_depth: int) -> Any:
        if not isinstance(term.body, Abs):
            raise ASTConstructionError(
                f"Fix.body must be an Abs, got {type(term.body).__name__}"
            )

        abs_body = term.body  # Abs(self, inner)
        self_name = abs_body.param
        inner = abs_body.body  # the body of the lambda that takes `self`

        # The Fix encoding: ``fix (λself. inner) = inner[self := self_ref]``
        # where self_ref, when applied to a value, re-enters the same
        # recursion at depth+1. We build a Python closure ``make_self``
        # that produces such a self-reference at any given depth.

        captured_env = dict(env)
        top_depth = _fix_depth

        def make_self(depth: int) -> Callable[[Any], Any]:
            def _self_ref(x: Any) -> Any:
                # I5: hard cap on recursion depth.
                if depth >= self.max_depth:
                    raise TerminationError(
                        f"Fix exceeded max_depth={self.max_depth} at "
                        f"depth={depth}; SPLIT may not be strictly "
                        "rank-reducing (T1 violation)."
                    )
                # Log a plan the first time the top-level Fix is applied.
                if depth == top_depth:
                    try:
                        size = len(x) if hasattr(x, "__len__") else 1
                        p = plan_fn(
                            PlanInputs(
                                n=size,
                                K=self.context_window,
                                tau=self.default_tau,
                            )
                        )
                        self.plans.append(p)
                    except Exception:
                        pass
                # Bind ``self_name`` to a fresh self-ref whose calls
                # advance depth by 1. Evaluate inner, then apply to x.
                inner_env = {**captured_env, self_name: make_self(depth + 1)}
                body_val = self._eval(inner, inner_env, _fix_depth=depth + 1)
                return self._apply(body_val, x, _fix_depth=depth + 1)

            return _self_ref

        # Return a top-level callable. ``App(Fix(...), arg)`` will invoke
        # this at depth=top_depth; nested ``self(arg)`` calls advance.
        return make_self(top_depth)


__all__ = ["Executor"]
