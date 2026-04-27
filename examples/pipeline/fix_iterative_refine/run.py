"""
Fix Iterative Refine — inline ``fix`` for an iterative draft→critique loop
==============================================================================

Demonstrates the kernel ``fix`` primitive at the inline-DSL level for a
bounded iterative-refinement loop:

    fix(λself. λx.
      let_("draft", leaf_draft,
        let_("critique", leaf_critique,
          case_(success_check(critique),
                {"true": var("critique")},
                default=app(self, x)))))

A host-callable ``success_check`` returns ``"true"`` once a host-side
attempt counter reaches a target threshold (here: iteration 2). The
loop emits 2 leaves per iteration; with 2 iterations until success,
total oracle calls = 4.

This is the inline-DSL equivalent of ``retry_term`` but with leaves
inside the body (retry_term's body is host-callable only).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline  # noqa: E402
from fsm_llm.lam import abs_, app, case_, fix, leaf, let_, var  # noqa: E402

SCHEMA_DRAFT = "examples.pipeline.fix_iterative_refine.schemas.Draft"
SCHEMA_CRITIQUE = "examples.pipeline.fix_iterative_refine.schemas.Critique"

TARGET_ITERATIONS = 2  # success_check returns "true" on this iteration

DRAFT_TEMPLATE = (
    "You are a writer. Topic: {topic}\n"
    "Produce a 1-sentence draft. Return JSON with:\n"
    "- text: the draft sentence\n"
    "- iteration: integer (0 if first draft)\n"
)

CRITIQUE_TEMPLATE = (
    "You are a critic. Critique this draft.\n"
    "Topic: {topic}\nDraft (JSON): {draft}\n\n"
    "Return JSON with:\n"
    "- text: the critique text\n"
    "- score: 0..1 quality\n"
    "- issues: list of issue strings (may be empty)\n"
)

TOPIC = "Why curiosity helps engineers debug faster."

# Mutable iteration counter — closed over by `success_check`.
_LOOP = {"iter": 0}


def success_check(critique: Any) -> str:
    """Return 'true' when the host-side iteration counter reaches the target."""
    _LOOP["iter"] += 1
    return "true" if _LOOP["iter"] >= TARGET_ITERATIONS else "false"


def build_term():
    leaf_draft = leaf(
        template=DRAFT_TEMPLATE,
        input_vars=("topic",),
        schema_ref=SCHEMA_DRAFT,
    )
    leaf_critique = leaf(
        template=CRITIQUE_TEMPLATE,
        input_vars=("topic", "draft"),
        schema_ref=SCHEMA_CRITIQUE,
    )
    # body :: λself. λx. let_("draft", ..., let_("critique", ...,
    #   case_(success_check(critique), {"true": critique}, default=self(x))))
    body = abs_(
        "self",
        abs_(
            "x",
            let_(
                "draft",
                leaf_draft,
                let_(
                    "critique",
                    leaf_critique,
                    case_(
                        app(var("success_check"), var("critique")),
                        {"true": var("critique")},
                        default=app(var("self"), var("x")),
                    ),
                ),
            ),
        ),
    )
    return app(fix(body), var("topic"))


def checks(result, error, oracle_calls):
    has_text = isinstance(result, dict) and "text" in result
    iters_correct = _LOOP["iter"] == TARGET_ITERATIONS
    leaves_count_ok = oracle_calls == 2 * TARGET_ITERATIONS  # 2 leaves * iters
    return {
        "result_is_dict": isinstance(result, dict),
        "critique_has_text": has_text,
        "loop_ran_target_iters": iters_correct,
        "oracle_calls_match_iter_count": leaves_count_ok,
        "no_error": error is None,
    }


def main() -> int:
    # Reset counter on each run.
    _LOOP["iter"] = 0
    return run_pipeline(
        build_term(),
        env={"topic": TOPIC, "success_check": success_check},
        checks_fn=checks,
        title="Fix Iterative Refine (inline kernel fix + leaves)",
    )


if __name__ == "__main__":
    sys.exit(main())
