"""
Shared runtime helpers for examples/pipeline/* (M4 slice 4+).

Three small helpers cover all S1/S2/S3/S4 patterns:

- ``run_pipeline(term, env, *, checks_fn) -> int``: boilerplate eliminator.
  Detects model + api_key, builds Oracle + Executor, runs the term, and
  prints a VERIFICATION block parseable by scripts/eval.py.
- ``make_tool_dispatcher(tools)``: turns ``{"tool_name", "args"}`` Pydantic
  dicts into observations by dispatching to mock tool callables.
- ``make_plan_executor(tools)``: walks REWOO-style ``Plan.steps`` and
  returns an evidence dict.

These helpers are NOT term factories (D-001 forbids stdlib factories
under src/). They are runtime glue invoked through env-bound App-chains
(host-callable escape hatch, LESSONS.md M2 S5).

Each example's ``run.py`` should:

    from examples.pipeline._helpers import run_pipeline, make_tool_dispatcher

    def build_term(): return let_(...)  # inline λ-DSL

    def main():
        tools = {"search": search_fn, ...}
        env = {"task": ..., "tool_dispatch": make_tool_dispatcher(tools)}
        run_pipeline(build_term(), env, checks_fn=my_checks)
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure project root is importable so `examples.pipeline.<name>.schemas`
# resolves regardless of cwd (D-006 sys.path shim, applied once here).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def run_pipeline(
    term: Any,
    env: dict[str, Any],
    *,
    checks_fn: Callable[[Any, Exception | None, int], dict[str, bool]],
    title: str = "Pipeline",
    context_window_tokens: int = 8192,
) -> int:
    """Run an inline λ-term and emit a VERIFICATION block.

    ``checks_fn`` receives ``(result, error, oracle_calls)`` and returns a
    dict of named boolean checks. It MUST return at least one check.

    Returns exit code: 0 if any checks pass, else 1.
    """
    # Local import to avoid loading lam at module import time.
    from fsm_llm.llm import LiteLLMInterface
    from fsm_llm.runtime import Executor, LiteLLMOracle

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Model: {model}")
    print("-" * 60)

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=context_window_tokens)
    ex = Executor(oracle=oracle)

    error: Exception | None = None
    result: Any = None
    try:
        result = ex.run(term, dict(env))
    except Exception as e:
        error = e
        print(f"Error: {e}")

    print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = checks_fn(result, error, ex.oracle_calls)
    extracted = sum(1 for v in checks.values() if v)
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    pct = 100 * extracted // max(1, len(checks))
    print(f"\nExtraction rate: {extracted}/{len(checks)} ({pct}%)")
    return 0 if extracted >= 1 else 1


def make_tool_dispatcher(
    tools: dict[str, Callable[[dict[str, Any]], str]],
) -> Callable[[Any], dict[str, Any]]:
    """Return a callable ``decision -> {"observation": str, "tool_used": str}``.

    ``decision`` is the output of a ``decide_leaf`` (Pydantic dict-like)
    with keys ``tool_name`` and ``args`` (or "input"/"query" — we are
    forgiving). Unknown tools yield an "unknown tool" observation.
    """

    def _dispatch(decision: Any) -> dict[str, Any]:
        d = decision if isinstance(decision, dict) else {}
        name = str(d.get("tool_name", d.get("tool", ""))).strip()
        # Accept flat ``query`` (preferred for Ollama-friendly schemas) OR
        # a nested ``args`` dict.
        if "query" in d and "args" not in d:
            args_raw: Any = {"query": d["query"]}
        else:
            args_raw = d.get("args", d.get("input", d.get("params", {})))
        if isinstance(args_raw, str):
            # Some Ollama models return the args as a JSON string; pass
            # through as {"query": ...} for the common case.
            args = {"query": args_raw}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        if name in tools:
            try:
                obs = tools[name](args)
            except Exception as e:
                obs = f"Tool {name} raised: {e}"
            return {"observation": str(obs), "tool_used": name}
        return {
            "observation": f"Unknown tool {name!r}; available: {list(tools)}",
            "tool_used": name or "(none)",
        }

    return _dispatch


def make_plan_executor(
    tools: dict[str, Callable[[dict[str, Any]], str]],
) -> Callable[[Any], dict[str, Any]]:
    """Return a callable ``plan -> {"evidence": dict, "trace": list}``.

    ``plan`` is the output of a ``plan_leaf`` Pydantic dict with a
    ``steps`` list, each step having ``tool_name``, ``args``, optionally
    a label like ``#E1``. Evidence labels default to ``#E{i}``.
    """

    def _exec(plan: Any) -> dict[str, Any]:
        p = plan if isinstance(plan, dict) else {}
        steps = p.get("steps") or []
        if not isinstance(steps, list):
            steps = []
        evidence: dict[str, str] = {}
        trace: list[dict[str, Any]] = []
        for i, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            label = step.get("label") or step.get("name") or f"#E{i}"
            name = str(step.get("tool_name", step.get("tool", "")))
            args_raw = step.get("args", step.get("input", {}))
            if isinstance(args_raw, str):
                args = {"query": args_raw}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            if name in tools:
                try:
                    obs = str(tools[name](args))
                except Exception as e:
                    obs = f"Tool {name} raised: {e}"
            else:
                obs = f"Unknown tool {name!r}"
            evidence[str(label)] = obs
            trace.append({"label": label, "tool": name, "obs": obs})
        return {"evidence": evidence, "trace": trace, "n_steps": len(steps)}

    return _exec


__all__ = ["run_pipeline", "make_tool_dispatcher", "make_plan_executor"]
