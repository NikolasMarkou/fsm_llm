from __future__ import annotations

"""End-to-end tests for fsm_llm.lam — SC1 (Category B) and SC2 (Category C).

Category B: stateless pipeline (no Fix). Pattern: evaluator → optimizer.
Category C: long-context recursive reduction. Pattern: count needle in
haystack by SPLIT → FMAP(self) → REDUCE(sum).

Both use a scripted mock oracle so the tests are deterministic and
verify ``predicted_calls`` matches the actual leaf-call count.
"""

from typing import Any

from fsm_llm.lam import (
    Executor,
    Oracle,
    PlanInputs,
    abs_,
    app,
    case_,
    fix,
    fmap,
    leaf,
    let_,
    plan,
    reduce_,
    split,
)


class _ScriptedOracle:
    """Responds with a fixed sequence; records every prompt."""

    def __init__(self, responses: list[Any], K: int = 10_000) -> None:
        self._responses = list(responses)
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
        return self._responses.pop(0)

    def tokenize(self, text: str) -> int:
        return max(1, len(text))

    def context_window(self) -> int:
        return self._K


# --------------------------------------------------------------
# SC1 — Category B: evaluator → optimizer pipeline (no Fix)
# --------------------------------------------------------------


class TestCategoryBPipeline:
    def test_category_b_pipeline(self) -> None:
        """A two-stage pipeline: draft answer, then refine it.

        Semantics: ``let draft = leaf(draft) in leaf(refine draft)``.
        No Fix, no recursion, no SPLIT — pure Category B.
        """
        oracle = _ScriptedOracle(responses=["draft_v1", "refined_v1"])
        ex = Executor(oracle=oracle)

        # Ensure Oracle protocol conformance.
        assert isinstance(oracle, Oracle)

        program = let_(
            "draft",
            leaf("draft {topic}", ("topic",)),
            leaf("refine {draft}", ("draft",)),
        )

        result = ex.run(program, {"topic": "FSMs"})

        assert result == "refined_v1"
        assert oracle.calls == [
            "draft FSMs",
            "refine draft_v1",
        ]
        assert ex.oracle_calls == 2


# --------------------------------------------------------------
# SC2 — Category C: recursive long-string reduction
# --------------------------------------------------------------


class TestCategoryCRecursiveBoundedCalls:
    def test_category_c_recursive_bounded_calls(self) -> None:
        """Count the occurrences of a needle in a haystack by recursive
        SPLIT→FMAP(self)→REDUCE(sum).

        Small-case (size ≤ τ): a Leaf call returns a count. Large-case:
        split into k pieces, map self over them, sum results.

        Verifies SC2: exact Leaf-call count == ``predicted_calls`` from
        the planner for the realised (k, d).
        """
        # We choose an input whose rank splits cleanly: 8 characters,
        # k=2 → two halves of 4 → each splits again into 2s → each 2
        # split into 1s (base case). Depth d = 3, leaves = 2^3 = 8.
        haystack = "aXaXaXaX"  # 4 X's
        tau = 1  # base case: rank-1 input → Leaf

        # Scripted oracle: every Leaf sees a single-char input and
        # returns "1" if it's the needle 'X' else "0".
        def _script(n_leaves: int) -> list[Any]:
            return ["1" if i % 2 == 1 else "0" for i in range(n_leaves)]

        oracle = _ScriptedOracle(responses=_script(8))
        ex = Executor(oracle=oracle, default_tau=tau)

        # Build the term:
        #   fix(λself. λP.
        #      case size_bucket(P) of
        #        "small" → leaf(count_char {P})  -- treats P as 1 char
        #        _      → reduce_(sum, fmap(self, split(P, 2))))
        program = app(
            fix(
                abs_(
                    "self",
                    abs_(
                        "P",
                        case_(
                            app("size_bucket", "P"),
                            {
                                "small": leaf("count {P}", ("P",)),
                            },
                            default=reduce_(
                                "sum_op",
                                fmap("self", split("P", 2)),
                            ),
                        ),
                    ),
                )
            ),
            "input",
        )

        def size_bucket(p: Any) -> str:
            return "small" if len(p) <= tau else "big"

        # REDUCE receives strings ("0"/"1" from leaves, then coerced
        # into ints by the op). We wrap BUILTIN_OPS["sum"] to coerce
        # child values to int on first fold step.
        from fsm_llm.lam.combinators import ReduceOp

        coerce_sum = ReduceOp(
            name="sum_coerce",
            fn=lambda a, b: int(a) + int(b),
            associative=True,
            unit=0,
        )

        result = ex.run(
            program,
            {
                "input": haystack,
                "size_bucket": size_bucket,
                "sum_op": coerce_sum,
            },
        )

        # 4 needles in "aXaXaXaX".
        assert result == 4

        # SC2: actual Leaf call count matches planner prediction for the
        # realised (k, d). With n=8, tau=1, k=2: d = log2(8/1) = 3,
        # predicted_calls = k^d = 2^3 = 8.
        predicted = plan(
            PlanInputs(n=len(haystack), K=10_000, tau=tau, alpha=1.0, max_k=2)
        )
        assert predicted.k_star == 2
        assert predicted.d == 3
        assert predicted.predicted_calls == 8

        assert ex.oracle_calls == predicted.predicted_calls == 8
        assert len(oracle.calls) == 8

    def test_category_c_degenerate_small_input(self) -> None:
        """Input already ≤ τ: single Leaf call, depth 0."""
        oracle = _ScriptedOracle(responses=["done"])
        ex = Executor(oracle=oracle, default_tau=100)

        program = app(
            fix(
                abs_(
                    "self",
                    abs_(
                        "P",
                        case_(
                            app("size_bucket", "P"),
                            {"small": leaf("solve {P}", ("P",))},
                            default=leaf("split {P}", ("P",)),  # unreachable
                        ),
                    ),
                )
            ),
            "input",
        )

        def size_bucket(p: Any) -> str:
            return "small"

        result = ex.run(program, {"input": "hi", "size_bucket": size_bucket})
        assert result == "done"
        assert ex.oracle_calls == 1


# --------------------------------------------------------------
# Package-level smoke: `from fsm_llm.lam import *` works
# --------------------------------------------------------------


class TestPublicAPI:
    def test_import_surface(self) -> None:
        import fsm_llm.lam as m

        # Essential names present
        for name in [
            "Executor",
            "Oracle",
            "LiteLLMOracle",
            "Plan",
            "PlanInputs",
            "plan",
            "CostAccumulator",
            "fix",
            "leaf",
            "split",
            "fmap",
            "reduce_",
            "LambdaError",
        ]:
            assert hasattr(m, name), f"fsm_llm.lam missing {name}"
        # __all__ is populated
        assert len(m.__all__) >= 30

    def test_top_level_fsm_llm_not_modified(self) -> None:
        """D-004: no top-level exports in M1."""
        import fsm_llm

        # None of the lam names should be re-exported at the top level.
        for name in ["Executor", "plan", "LambdaError", "Oracle"]:
            assert name not in getattr(fsm_llm, "__all__", ()), (
                f"{name} must not appear in top-level fsm_llm.__all__ during M1"
            )
