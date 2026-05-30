#!/usr/bin/env python
"""Long multi-turn conversation harness for stress-testing fsm_llm ReAct agents.

The agent layer is single-shot (`agent.run(task)` builds a fresh FSM each call), so
*this driver IS the multi-turn loop*. It runs a sequence of user turns against one
agent, carrying long-term memory across turns via a shared ``SemanticMemoryStore``
(for the ``auto_memory`` agent), and records per-turn behaviour to surface bugs:

  * tool under/over-calling (the known qwen3.5 4B/9B loop fragility),
  * ``BudgetExhaustedError`` / ``AgentTimeoutError`` (the 9B over-call death),
  * cross-turn long-term-memory recall failures,
  * decomposition agents that "synthesize" without executing planned tools.

It is a *diagnostic tool*, not a test: every turn's exceptions are caught and logged
so one bad turn never aborts the run. Pair the JSON transcript with a manual read of
the per-turn records (the heuristic ``flags`` field is a hint, not a verdict).

Usage::

    .venv/bin/python scripts/agent_chat_harness.py \
        --agent auto_memory --scenario mixed \
        --model ollama_chat/qwen3.5:4b \
        --persist-path /tmp/agent_mem.json --out /tmp/harness.json

    # reproduce the 9B budget-exhaustion bug (slow):
    .venv/bin/python scripts/agent_chat_harness.py \
        --agent react --scenario tools --model ollama_chat/qwen3.5:9b --out /tmp/9b.json
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import sys
import time
import traceback
from typing import Any

# Allow running from a source checkout without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fsm_llm_agents import AgentConfig, ToolRegistry, create_agent, tool
from fsm_llm_agents.auto_memory import AutoMemoryReactAgent
from fsm_llm_agents.exceptions import (
    AgentTimeoutError,
    BudgetExhaustedError,
)
from fsm_llm_agents.native_fc import NativeFunctionCallingReactAgent
from fsm_llm_agents.semantic_memory import SemanticMemoryStore

# --------------------------------------------------------------------------- #
# Tools — small, deterministic, deliberately mixed (one flaky) for bug-hunting #
# --------------------------------------------------------------------------- #

_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '17 * 23' or '(7+3)**2'."""
    return str(_safe_eval(ast.parse(expression, mode="eval")))


@tool
def get_time(timezone: str = "UTC") -> str:
    """Return the current time. Always use this for any time/date question."""
    import datetime

    return f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} ({timezone})"


_FAKE_KB = {
    "python": "Python is a high-level programming language created by Guido van Rossum.",
    "ollama": "Ollama runs open LLMs locally; qwen3.5 is available in 4b/9b/27b sizes.",
    "fsm": "An FSM (finite state machine) has states and transitions between them.",
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
}


@tool
def web_search(query: str) -> str:
    """Search a small knowledge base for a topic. Use for factual lookups."""
    q = query.lower().strip()
    for key, val in _FAKE_KB.items():
        if key in q or q in key:
            return val
    return f"No results for '{query}'. Try a different query."


_FLAKY_STATE = {"calls": 0}


@tool
def flaky_lookup(key: str) -> str:
    """Look up a record by key. (Backend is unreliable and may fail transiently.)"""
    _FLAKY_STATE["calls"] += 1
    if _FLAKY_STATE["calls"] % 2 == 1:
        raise RuntimeError("transient backend error (retry)")
    return f"Record[{key}] = OK (resolved after {_FLAKY_STATE['calls']} calls)"


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for fn in (calculate, get_time, web_search, flaky_lookup):
        registry.register(fn._tool_definition)
    return registry


# --------------------------------------------------------------------------- #
# Scenarios — each turn is (text, needs_tool) where needs_tool flags a turn    #
# that should provoke at least one tool call (for the UNDER_CALL heuristic).   #
# --------------------------------------------------------------------------- #

_SCENARIOS: dict[str, list[tuple[str, bool]]] = {
    "tools": [
        ("What is 17 times 23?", True),
        ("Now raise that result to the power of 2.", True),
        ("What is the current UTC time?", True),
        ("Look up the record with key 'alpha'.", True),
        ("What is 144 divided by 12, minus 5?", True),
    ],
    "memory": [
        ("My name is Nikolas and my favorite language is Python. Please remember this.", False),
        ("I work on a framework called fsm_llm. Remember that too.", False),
        ("What is my name?", False),
        ("What is my favorite programming language?", False),
        ("What framework do I work on?", False),
    ],
    "decompose": [
        ("Compare the capitals of France and Japan, then tell me which name is longer.", True),
        ("Search for what Python is, search for what Ollama is, then summarize both in one sentence.", True),
        ("Calculate 12*12, then calculate 13*13, then tell me the difference between the two.", True),
    ],
    "mixed": [
        ("My name is Nikolas; remember it.", False),
        ("What is 25 * 4?", True),
        ("What's my name?", False),
        ("Look up 'fsm' and tell me what it means.", True),
        ("What is the current time, and what is 2 to the power of 10?", True),
        ("Earlier I told you my name — what was it?", False),
        ("Compare the capital of France with the capital of Japan.", True),
    ],
}


def _load_turns(args: argparse.Namespace) -> list[tuple[str, bool]]:
    if args.turns_file:
        with open(args.turns_file, encoding="utf-8") as fh:
            data = json.load(fh)
        # Accept either ["text", ...] or [["text", true], ...] or [{"text":..,"needs_tool":..}]
        turns: list[tuple[str, bool]] = []
        for item in data:
            if isinstance(item, str):
                turns.append((item, False))
            elif isinstance(item, (list, tuple)):
                turns.append((item[0], bool(item[1]) if len(item) > 1 else False))
            elif isinstance(item, dict):
                turns.append((item["text"], bool(item.get("needs_tool", False))))
        return turns
    return _SCENARIOS[args.scenario]


# --------------------------------------------------------------------------- #
# Agent construction                                                          #
# --------------------------------------------------------------------------- #


def _build_agent(args: argparse.Namespace, registry: ToolRegistry) -> tuple[Any, Any]:
    """Return (agent, memory_store_or_None)."""
    config = AgentConfig(
        model=args.model,
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        max_history_size=8,
        auto_summarize_after=8,
    )
    if args.agent == "auto_memory":
        store = SemanticMemoryStore(
            embedding_model=args.embedding_model,
            persist_path=args.persist_path,  # None => ephemeral (a known gap, B10)
        )
        agent = AutoMemoryReactAgent(
            tools=registry, config=config, memory=store, recall_k=args.recall_k
        )
        return agent, store
    if args.agent == "native_fc":
        return NativeFunctionCallingReactAgent(tools=registry, config=config), None
    # react / plan_execute / orchestrator / adapt via the factory
    agent = create_agent(pattern=args.agent, tools=registry, config=config)
    return agent, None


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def _flags(rec: dict[str, Any], needs_tool: bool) -> list[str]:
    flags: list[str] = []
    if rec.get("error"):
        flags.append(f"ERROR:{rec['error']}")
    if needs_tool and not rec.get("tools_used"):
        flags.append("UNDER_CALL")  # a tool-needing turn that called no tool
    if rec.get("success") is False and not rec.get("error"):
        flags.append("NOT_SUCCESS")
    return flags


def run_harness(args: argparse.Namespace) -> dict[str, Any]:
    registry = _build_registry()
    agent, store = _build_agent(args, registry)
    turns = _load_turns(args)

    print(
        f"[harness] agent={args.agent} model={args.model} "
        f"scenario={args.scenario} turns={len(turns)} "
        f"persist={args.persist_path or 'EPHEMERAL'}"
    )

    log: list[dict[str, Any]] = []
    for i, (text, needs_tool) in enumerate(turns, 1):
        t0 = time.time()
        rec: dict[str, Any] = {"turn": i, "input": text, "needs_tool": needs_tool}
        try:
            result = agent.run(text)
            rec.update(
                answer=(result.answer or "").strip(),
                success=bool(result.success),
                tools_used=list(result.tools_used or []),
                iterations=result.iterations_used,
            )
        except (BudgetExhaustedError, AgentTimeoutError) as exc:
            rec.update(error=type(exc).__name__, msg=str(exc))
        except Exception as exc:
            rec.update(
                error=type(exc).__name__,
                msg=str(exc),
                tb=traceback.format_exc(limit=4),
            )
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["flags"] = _flags(rec, needs_tool)
        if store is not None:
            rec["mem_size"] = len(store)
        log.append(rec)
        _print_turn(rec)

    summary = _summarize(args, log, store)
    out = {"args": vars(args), "summary": summary, "turns": log}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"[harness] transcript -> {args.out}")
    return out


def _print_turn(rec: dict[str, Any]) -> None:
    flags = (" " + " ".join(rec["flags"])) if rec.get("flags") else ""
    if rec.get("error"):
        print(
            f"  turn {rec['turn']:>2} [{rec['elapsed_s']:>5}s] "
            f"!! {rec['error']}: {rec.get('msg', '')[:80]}{flags}"
        )
    else:
        ans = (rec.get("answer") or "")[:90].replace("\n", " ")
        print(
            f"  turn {rec['turn']:>2} [{rec['elapsed_s']:>5}s] "
            f"ok={rec['success']} tools={rec['tools_used']} "
            f"iters={rec['iterations']}{flags}\n        -> {ans}"
        )


def _summarize(
    args: argparse.Namespace, log: list[dict[str, Any]], store: Any
) -> dict[str, Any]:
    errors = [r for r in log if r.get("error")]
    under = [r for r in log if "UNDER_CALL" in r.get("flags", [])]
    budget = [r for r in log if r.get("error") == "BudgetExhaustedError"]
    summary = {
        "turns": len(log),
        "errors": len(errors),
        "budget_exhausted": len(budget),
        "under_calls": len(under),
        "success_turns": sum(1 for r in log if r.get("success")),
        "total_elapsed_s": round(sum(r.get("elapsed_s", 0) for r in log), 1),
    }
    if store is not None:
        summary["final_mem_size"] = len(store)
    print(
        f"[harness] SUMMARY: {summary['success_turns']}/{summary['turns']} ok, "
        f"{summary['errors']} errors ({summary['budget_exhausted']} budget), "
        f"{summary['under_calls']} under-calls, {summary['total_elapsed_s']}s total"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--agent",
        default="auto_memory",
        choices=["auto_memory", "react", "native_fc", "plan_execute", "orchestrator", "adapt"],
    )
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "ollama_chat/qwen3.5:4b"))
    p.add_argument("--scenario", default="mixed", choices=list(_SCENARIOS))
    p.add_argument("--turns-file", default=None, help="JSON list of turns (overrides --scenario)")
    p.add_argument("--out", default=None, help="write transcript JSON here")
    p.add_argument("--max-iterations", type=int, default=6)
    p.add_argument("--timeout-seconds", type=float, default=180.0)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--persist-path", default=None, help="SemanticMemoryStore JSON path (auto_memory)")
    p.add_argument("--embedding-model", default="ollama/qwen3-embedding:0.6b")
    p.add_argument("--recall-k", type=int, default=3)
    args = p.parse_args(argv)

    out = run_harness(args)
    # Non-zero exit only on a hard harness failure, not on agent bugs (those are data).
    return 0 if out["summary"]["turns"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
