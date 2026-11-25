"""The cells of the differential proof of plan 659c07c3 (X3b), written once from the PARENT.

Not collected (no ``test_`` prefix). ``tests/test_fsm_llm_agents/test_agent_preview_golden.py`` holds
the rows. Run it from a ``git archive`` of the parent commit to write the golden::

    PYTHONPATH=<archive>/src python tests/test_fsm_llm_agents/agent_preview_cells.py --write PATH

and with ``--self-test`` to print the difference the comparer finds under each planted mutant.
The golden ``tests/fixtures/agent_preview_golden.json`` is a fixture like
``kind_equality_baseline.json``: written on the parent, never regenerated to silence a failure.

Cells (doubles only: a ``ScriptedLLM`` and stub tools, no provider call, no network):

- G1 per id (the 18 ``create_agent`` ids and the 3 shipped presets): the sorted nodes
  ``(id, is_initial, is_terminal)`` and edges ``(from, to, priority)`` of the drawing route (a
  pattern file through ``instantiate_body`` with ``<name>`` placeholders and ``GENERATORS.get``, or
  the dict builder of a code-backed id, a one-name stub tool where the builder needs a registry), the
  placeholder names found in the text, and the FSM of the REAL agent (built by ``create_agent`` with
  one stub tool, its FSM captured where the run would make its ``API``) for the ids that have one.
- G2 per old ``_AGENT_CLASSES`` class name: the class and the digest of the FSM the old way built
  (the class called directly) against ``create_agent(<id>)``.
- G3 the ``/api/fsm/visualize`` bytes of one fixed FSM through the TestClient (``_fsm_viz_or_400``).

Interface contract: ``compute() -> dict`` (keys ``g1``, ``g2``, ``g3``; ``g2`` holds the ``old`` and
the ``new`` way), ``compare(golden, actual) -> list[str]`` (one line per difference, empty when equal),
``MUTANTS`` (name to a callable returning a mutated copy of the actual document).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from unittest import mock

#: old ``_AGENT_CLASSES`` name -> executable id (the 7 types of the old launch table)
OLD_TYPES = {
    "ReactAgent": "react",
    "ReflexionAgent": "reflexion",
    "PlanExecuteAgent": "plan_execute",
    "REWOOAgent": "rewoo",
    "ADaPTAgent": "adapt",
    "DebateAgent": "debate",
    "SelfConsistencyAgent": "self_consistency",
}
#: the old ``_TOOL_BASED_AGENTS``: these were built with a registry
OLD_TOOL_BASED = frozenset(
    {"ReactAgent", "ReflexionAgent", "PlanExecuteAgent", "REWOOAgent", "ADaPTAgent"}
)
CREATE_AGENT_IDS = (
    "adapt auto_memory debate evaluator_optimizer maker_checker meta_builder native_fc "
    "orchestrator parallel_react plan_execute prompt_chain react reasoning_react reflexion "
    "rewoo self_consistency swarm verified_react"
).split()
PRESET_IDS = ["code-review", "data-extraction", "summarize"]
TASK = "Summarize the quarterly report."
STUB_TOOL = "lookup"
PLACEHOLDER = "<{}>"


class _Captured(BaseException):
    """Raised where a run would make its API: the FSM was seen, nothing runs."""


def stub_registry() -> Any:
    from fsm_llm.tools import ToolRegistry

    def stub(**kwargs: Any) -> str:
        return "stub"

    return ToolRegistry().register_function(stub, name=STUB_TOOL, description="A stub.")


def graph_of(definition: dict[str, Any]) -> dict[str, Any]:
    """Sorted nodes and edges of an FSM dict, read through core's one graph builder."""
    from fsm_llm import build_fsm_graph

    graph = build_fsm_graph(definition)
    return {
        "nodes": sorted(
            [n.id, bool(n.is_initial), bool(n.is_terminal)] for n in graph.nodes
        ),
        "edges": sorted([e.source, e.target, e.priority] for e in graph.edges),
    }


def digest(definition: dict[str, Any]) -> str:
    text = json.dumps(definition, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def placeholder_names(definition: dict[str, Any], names: list[str]) -> list[str]:
    text = json.dumps(definition, sort_keys=True, default=str)
    return sorted(n for n in names if PLACEHOLDER.format(n) in text)


def file_definition(
    pattern: str, placeholder: str = PLACEHOLDER
) -> tuple[dict, list[str]]:
    """The drawing route of a pattern file: placeholders for its variables, the shipped generators."""
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.core.definitions.executable_instance import instantiate_body
    from fsm_llm.engine.executables.namespaces import GENERATORS

    envelope = load_envelope(pattern)
    names = sorted(envelope.variables)
    definition = instantiate_body(
        envelope,
        None,
        variables={n: placeholder.format(n) for n in names},
        generators=GENERATORS.get,
    )
    return definition, names


def builder_definition(pattern: str) -> dict[str, Any] | None:
    """The dict builder of a code-backed id (no arguments; a one-name stub tool where it needs one)."""
    from fsm_llm.agents import fsm_definitions as fsmd

    if pattern == "adapt":
        return fsmd.build_adapt_fsm()
    if pattern == "self_consistency":
        return fsmd.build_self_consistency_fsm()
    if pattern == "meta_builder":
        return fsmd.build_meta_builder_fsm()
    if pattern == "parallel_react":
        return fsmd.build_parallel_react_fsm(stub_registry())
    if pattern == "reasoning_react":
        return fsmd.build_react_fsm(stub_registry())
    return None  # swarm: its members own their FSMs


def _has_file(pattern: str) -> bool:
    from fsm_llm.agents.pattern_files import PATTERN_ORDER

    return pattern in PATTERN_ORDER


def real_agent(pattern: str, llm: Any) -> Any:
    """The real agent of an id, built by ``create_agent`` with the objects it needs and one stub tool."""
    from fsm_llm.agents import create_agent
    from fsm_llm.agents.base.agent import accepts_tools
    from fsm_llm.agents.definitions import AgentConfig, ChainStep, EvaluationResult
    from fsm_llm.engine.executables.namespaces import PATTERN_AGENTS

    kwargs: dict[str, Any] = {
        "config": AgentConfig(model="stub/double", max_iterations=4),
        "llm_interface": llm,
    }
    if pattern == "evaluator_optimizer":
        kwargs["evaluation_fn"] = lambda _o, _c: EvaluationResult(
            passed=True, score=1.0, feedback="ok"
        )
    if pattern == "maker_checker":
        kwargs["maker_instructions"] = "Write a memo."
        kwargs["checker_instructions"] = "Check the memo."
    if pattern == "prompt_chain":
        kwargs["chain"] = [
            ChainStep(
                step_id=f"s{i}",
                name=f"Stage {i}",
                extraction_instructions="Extract it.",
                response_instructions="Reply.",
            )
            for i in range(2)
        ]
    tools = stub_registry() if accepts_tools(PATTERN_AGENTS.get(pattern)) else None
    return create_agent(pattern, tools, **kwargs)


def captured_fsm(agent: Any) -> dict[str, Any] | None:
    """The FSM dict a run of *agent* hands to its API (the run stops there), or ``None``."""
    from fsm_llm.agents.base.runs import RunsMixin

    seen: list[dict[str, Any]] = []

    def grab(self: Any, fsm_def: dict[str, Any]) -> Any:
        seen.append(copy.deepcopy(fsm_def))
        raise _Captured

    with mock.patch.object(RunsMixin, "_create_api", grab):
        try:
            agent.run(TASK)
        except (_Captured, Exception):
            pass
    return seen[0] if seen else None


def _llm() -> Any:
    from fsm_llm.testing import ScriptedLLM

    return ScriptedLLM()


def compute_g1(placeholder: str = PLACEHOLDER) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for pattern in [*CREATE_AGENT_IDS, *PRESET_IDS]:
        cell: dict[str, Any] = {
            "route": "none",
            "graph": None,
            "placeholders": [],
            "real": None,
        }
        if _has_file(pattern):
            definition, names = file_definition(pattern, placeholder)
            cell.update(
                route="file",
                graph=graph_of(definition),
                placeholders=placeholder_names(definition, names),
            )
        elif (definition := builder_definition(pattern)) is not None:
            cell.update(route="builder", graph=graph_of(definition))
        if (
            pattern in CREATE_AGENT_IDS
            and pattern != "swarm"
            and pattern != "meta_builder"
        ):
            fsm = captured_fsm(real_agent(pattern, _llm()))
            if fsm is not None:
                cell["real"] = {"states": len(fsm["states"]), "graph": graph_of(fsm)}
        cells[pattern] = cell
    return cells


def _old_class(name: str) -> type:
    """The class of an old ``_AGENT_CLASSES`` name, from the public package (never from the monitor)."""
    import fsm_llm.agents as agents

    return getattr(agents, name)


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _g2_cell(agent: Any) -> dict[str, str]:
    fsm = captured_fsm(agent)
    return {
        "class": _qualified(type(agent)),
        "fsm_digest": digest(fsm) if fsm is not None else "none",
    }


def compute_g2() -> dict[str, Any]:
    """The old way (the class called directly, as the launch table did) and the new way
    (``create_agent(<id>)``), per old class name, with the stub tool for the five tool-based types."""
    from fsm_llm.agents import create_agent
    from fsm_llm.agents.definitions import AgentConfig

    def config() -> Any:
        return AgentConfig(model="stub/double", max_iterations=4)

    old: dict[str, Any] = {}
    new: dict[str, Any] = {}
    for name, pattern in OLD_TYPES.items():
        tools = name in OLD_TOOL_BASED
        kwargs = {"config": config(), "llm_interface": _llm()}
        if tools:
            kwargs["tools"] = stub_registry()
        old[name] = {"id": pattern, **_g2_cell(_old_class(name)(**kwargs))}
        kwargs = {"config": config(), "llm_interface": _llm()}
        # create_agent routes tools by the constructor: plan_execute and adapt take them, the rest of
        # the table's tool-based types too; debate and self_consistency get none, as the old launch
        new[name] = {
            "id": pattern,
            **_g2_cell(
                create_agent(pattern, stub_registry() if tools else None, **kwargs)
            ),
        }
    return {"old": old, "new": new}


#: a fixed FSM for G3: three states, an edge description longer than the 30 character label cut
G3_FSM = {
    "name": "G3",
    "description": "Fixed FSM of the drawing path",
    "initial_state": "ask",
    "persona": "A tester",
    "states": {
        "ask": {
            "id": "ask",
            "description": "Ask",
            "purpose": "Ask the question",
            "response_instructions": "Ask it",
            "transitions": [
                {
                    "target_state": "ask",
                    "description": "No answer yet",
                    "priority": 100,
                },
                {
                    "target_state": "done",
                    "description": "The user gave a complete and usable answer to the question",
                    "priority": 50,
                    "conditions": [
                        {
                            "description": "answer given",
                            "requires_context_keys": ["answer"],
                            "logic": {"==": [{"var": "answer"}, "yes"]},
                        }
                    ],
                },
            ],
        },
        "done": {
            "id": "done",
            "description": "Done",
            "purpose": "Wrap up",
            "response_instructions": "Say goodbye",
        },
    },
}


def compute_g3(label_chars: int | None = None) -> dict[str, Any]:
    """The bytes of ``POST /api/fsm/visualize`` for ``G3_FSM`` through the TestClient."""
    from fastapi.testclient import TestClient

    from fsm_llm.monitor.instance_manager import InstanceManager
    from fsm_llm.monitor.server import app, configure, routes_visualize

    os.environ.pop("FSM_LLM_MONITOR_API_KEY", None)
    configure(manager=InstanceManager())
    with _label_cut(routes_visualize, label_chars):
        response = TestClient(app).post("/api/fsm/visualize", json=G3_FSM)
    body = response.content.decode("utf-8")
    return {
        "status": response.status_code,
        "body": body,
        "body_sha256": hashlib.sha256(response.content).hexdigest(),
    }


@contextmanager
def _label_cut(module: Any, chars: int | None) -> Any:
    if chars is None:
        yield
        return
    with mock.patch.object(module, "VIZ_EDGE_LABEL_CHARS", chars):
        yield


def compute() -> dict[str, Any]:
    return {"g1": compute_g1(), "g2": compute_g2(), "g3": compute_g3()}


def compare(golden: Any, actual: Any, path: str = "") -> list[str]:
    """One line per difference between two JSON-shaped values (empty when equal)."""
    if isinstance(golden, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted({*golden, *actual}, key=str):
            where = f"{path}/{key}"
            if key not in actual:
                out.append(f"{where}: missing in actual")
            elif key not in golden:
                out.append(f"{where}: not in golden")
            else:
                out += compare(golden[key], actual[key], where)
        return out
    if isinstance(golden, list) and isinstance(actual, list):
        if len(golden) != len(actual):
            return [f"{path}: length {len(golden)} != {len(actual)}"]
        out = []
        for i, (g, a) in enumerate(zip(golden, actual, strict=True)):
            out += compare(g, a, f"{path}[{i}]")
        return out
    if golden != actual or type(golden) is not type(actual):
        return [f"{path}: {_around(golden, actual)}"]
    return []


def _around(golden: Any, actual: Any) -> str:
    """Both values cut around the first character they differ at (a long string shows its change)."""
    g, a = str(golden), str(actual)
    at = next((i for i, (x, y) in enumerate(zip(g, a, strict=False)) if x != y), 0)
    lo = max(at - 20, 0)
    return f"{g[lo : at + 30]!r} != {a[lo : at + 30]!r} (type {type(golden).__name__} vs {type(actual).__name__})"


def golden_view(document: dict[str, Any]) -> dict[str, Any]:
    """What is written to the golden: G1 and G3 as computed, G2 as the OLD way (the parent's)."""
    return {"g1": document["g1"], "g2": document["g2"]["old"], "g3": document["g3"]}


def tree_view(document: dict[str, Any]) -> dict[str, Any]:
    """What a tree must equal: G1 and G3 as computed, G2 as the NEW way (``create_agent(<id>)``)."""
    return {"g1": document["g1"], "g2": document["g2"]["new"], "g3": document["g3"]}


def _mut_edge_dropped(doc: dict[str, Any]) -> dict[str, Any]:
    doc["g1"]["react"]["graph"]["edges"].pop(0)
    return doc


def _mut_ids_swapped(doc: dict[str, Any]) -> dict[str, Any]:
    g1 = doc["g1"]
    g1["react"]["graph"], g1["rewoo"]["graph"] = (
        g1["rewoo"]["graph"],
        g1["react"]["graph"],
    )
    return doc


def _mut_native_fc_three(doc: dict[str, Any]) -> dict[str, Any]:
    graph = doc["g1"]["native_fc"]["graph"]
    keep = {n[0] for n in graph["nodes"][:3]}
    graph["nodes"] = graph["nodes"][:3]
    graph["edges"] = [e for e in graph["edges"] if e[0] in keep and e[1] in keep]
    return doc


def _mut_class_swapped(doc: dict[str, Any]) -> dict[str, Any]:
    new = doc["g2"]["new"]
    new["ReactAgent"]["class"] = new["ReflexionAgent"]["class"]
    return doc


def _mut_placeholder_changed() -> dict[str, Any]:
    doc = compute()
    doc["g1"] = compute_g1(placeholder="[{}]")
    return doc


def _mut_label_cut_changed() -> dict[str, Any]:
    doc = compute()
    doc["g3"] = compute_g3(label_chars=29)
    return doc


def _on(
    mutate: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda doc: mutate(copy.deepcopy(doc))


#: planted mutants of the ACTUAL document: each must make ``compare`` print a difference
MUTANTS: dict[str, Callable[..., dict[str, Any]]] = {
    "edge dropped from one preview": _on(_mut_edge_dropped),
    "two ids swapped": _on(_mut_ids_swapped),
    "native_fc given three states": _on(_mut_native_fc_three),
    "a class of the new way swapped": _on(_mut_class_swapped),
    "placeholder format changed": lambda doc: _mut_placeholder_changed(),
    "label cut changed (G3)": lambda doc: _mut_label_cut_changed(),
}


def main(argv: list[str]) -> int:
    import fsm_llm

    here = os.path.dirname(os.path.abspath(fsm_llm.__file__))
    print("fsm_llm from", here, file=sys.stderr)
    document = compute()
    if "--write" in argv:
        path = argv[argv.index("--write") + 1]
        if os.path.exists(os.path.join(here, "..", "..", ".git")):
            print(
                "refused: the golden is written from a git archive of the PARENT",
                file=sys.stderr,
            )
            return 1
        golden = {
            "meta": {
                "written_from": "git archive of the parent commit ab2b4499",
                "task": TASK,
                "stub_tool": STUB_TOOL,
                "placeholder": PLACEHOLDER,
                "no_real_fsm": {
                    "swarm": "no FSM: its members own theirs",
                    "meta_builder": "builds its API in its own session, not through a run",
                    "code-review, data-extraction, summarize": "presets: no agent class",
                },
            },
            **golden_view(document),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(golden, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("wrote", path)
    if "--self-test" in argv or "--write" in argv:
        golden_doc = golden_view(document)
        print(
            "parity (tree vs golden):",
            compare(golden_doc, tree_view(document)) or "equal",
        )
        for name, mutate in MUTANTS.items():
            found = compare(golden_doc, tree_view(mutate(document)))
            print(f"mutant {name!r}: {len(found)} difference(s); first: {found[:1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
