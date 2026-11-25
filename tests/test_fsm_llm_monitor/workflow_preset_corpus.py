"""The corpus and the snapshot of differential G2 (plan 7d9c0955 step 1). Not collected.

``snapshot()`` runs the two launchable workflow presets (``demo_linear``, ``demo_branching``)
through a ``WorkflowEngine`` for eight values of ``amount`` and records, per run: the final status,
the error text, the public context, the history ``(step_id, status, message)`` and every lifecycle
hook. It also records the monitor surface: ``get_workflow_presets()``, the ``launch_workflow``
refusal texts, the bytes of ``GET /api/workflow/presets``, the launch route's error bodies and, per
preset and initial context, the instance info, status, instance list and the monitor events of one
launched run (ids and timestamps scrubbed).

``tests/fixtures/workflow_preset_golden.json`` was written from a ``git archive`` of the parent
commit ``a8ad5d31`` by a script outside the repository (its path is in the plan's progress file).
It holds ``engine_runs`` and ``monitor`` (the presets as they are at BASE, their last step a
``_TerminalStep``) and ``document_runs`` and ``document_monitor`` (the same presets with the last
step an ``AutoStep(next_step_id="")``, which is what a preset document builds). ``named_deltas``
lists what differs between the two, and nothing else may: step 7 of the plan moves the presets
into documents and must equal the ``document_*`` halves.

``preset_definitions()`` is the one place that names where the preset definitions come from. At
step 1 it read ``_WORKFLOW_PRESETS`` (the fixture was written by that form against the archive of
``a8ad5d31``); step 7 retargeted it to the registry of preset documents. The fixture is never
written again from the tree.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from fsm_llm.workflows import WorkflowDefinition, WorkflowEngine, auto_step

ABSENT = object()
AMOUNTS: list[tuple[str, Any]] = [
    ("absent", ABSENT),
    ("0", 0),
    ("999", 999),
    ("1000", 1000),
    ("str_1500", "1500"),
    ("none", None),
    ("str_abc", "abc"),
    ("true", True),
]
LAUNCH_CONTEXTS: list[tuple[str, dict[str, Any] | None]] = [
    ("no_context", None),
    ("amount_5000", {"amount": 5000}),
    ("amount_10", {"amount": 10}),
]
REFUSED_LAUNCHES: list[dict[str, Any]] = [
    {"preset_id": "nope"},
    {"preset_id": None},
    {"definition_json": {"a": 1}},
    {"preset_id": " demo_linear"},
    {"preset_id": "demo_linear "},
    {"preset_id": "Demo_Linear"},
    {"preset_id": "DEMO_LINEAR"},
    {"preset_id": ""},
]
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{3,4}(?:-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?")


def preset_definitions() -> dict[str, Callable[[], WorkflowDefinition]]:
    """The preset id to a builder of a fresh definition, read from the preset documents."""
    from fsm_llm.monitor.instance_manager.workflow_presets import (
        build_preset,
        load_preset_registry,
    )

    registry = load_preset_registry()
    return {
        e.id: (lambda i=e.id: build_preset(registry, i))
        for e in registry.list(kind="workflow")
    }


def document_equivalent(build: Callable[[], WorkflowDefinition]) -> WorkflowDefinition:
    """The preset with ``finish`` as a document builds it: ``AutoStep(next_step_id="")``."""
    definition = build()
    definition.steps["finish"] = auto_step("finish", "Finish", next_step_id="")
    return definition


def scrub(value: Any) -> Any:
    """Replace every generated id in strings, keys and values by ``<id>``."""
    if isinstance(value, str):
        return _ID.sub("<id>", value)
    if isinstance(value, dict):
        return {scrub(str(k)): scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return (
        value if value is None or isinstance(value, (bool, int, float)) else repr(value)
    )


async def _engine_run(definition: WorkflowDefinition, amount: Any) -> dict[str, Any]:
    engine = WorkflowEngine()
    engine.register_workflow(definition)
    hooks: list[list[Any]] = []
    engine.add_hook(
        lambda name, instance, data: hooks.append([name, scrub(dict(data))])
    )
    context = {} if amount is ABSENT else {"amount": amount}
    instance_id = await engine.start(definition.workflow_id, initial_context=context)
    instance = engine.workflow_instances[instance_id]
    return {
        "status": instance.status.value,
        "error": scrub(instance.error),
        "context": scrub(
            {k: v for k, v in instance.context.items() if not k.startswith("_")}
        ),
        "history": [
            [h.step_id, getattr(h.status, "value", str(h.status)), scrub(h.message)]
            for h in instance.history
        ],
        "hooks": hooks,
    }


async def _runs(equivalent: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for preset_id, build in sorted(preset_definitions().items()):
        out[preset_id] = {}
        for label, amount in AMOUNTS:
            definition = document_equivalent(build) if equivalent else build()
            out[preset_id][label] = await _engine_run(definition, amount)
    return out


def engine_runs() -> dict[str, Any]:
    """The presets as the tree builds them."""
    return asyncio.run(_runs(False))


def document_runs() -> dict[str, Any]:
    """The presets with the last step an ``AutoStep(next_step_id="")``."""
    return asyncio.run(_runs(True))


def _launch_error(manager: Any, **kwargs: Any) -> list[str]:
    try:
        manager.launch_workflow(**kwargs)
    except Exception as error:
        return [type(error).__name__, str(error)]
    return ["launched"]


async def _launch_and_read(
    manager: Any, preset_id: str, context: Any
) -> dict[str, Any]:
    managed = manager.launch_workflow(preset_id=preset_id, label="L")
    wf_id = await manager.start_workflow_instance(
        managed.instance_id, managed.workflow_id, context
    )
    status = manager.get_workflow_status(managed.instance_id, wf_id)
    status["history"] = [
        {k: v for k, v in h.items() if k != "timestamp"} for h in status["history"]
    ]
    status.pop("created_at", None)
    status.pop("updated_at", None)
    instances = manager.get_workflow_instances(managed.instance_id)
    for entry in instances:
        entry["history"] = [
            {k: v for k, v in h.items() if k != "timestamp"}
            for h in entry.get("history", [])
        ]
        entry.pop("created_at", None)
        entry.pop("updated_at", None)
    events = [
        [e.event_type, e.level, scrub(e.data)]
        for e in reversed(manager.global_collector.get_events())
        if e.event_type.startswith("workflow_")
        and e.data.get("instance_id") == managed.instance_id
    ]
    info = managed.to_info().model_dump()
    for key in ("instance_id", "created_at"):
        info.pop(key, None)
    return scrub(
        {"info": info, "status": status, "instances": instances, "events": events}
    )


def monitor_surface() -> dict[str, Any]:
    """The monitor reads of the presets: manager calls and the HTTP bytes."""
    from fastapi.testclient import TestClient

    from fsm_llm.monitor.definitions import MonitorConfig
    from fsm_llm.monitor.instance_manager import InstanceManager
    from fsm_llm.monitor.server import app, configure

    manager = InstanceManager(config=MonitorConfig())
    manager.global_collector.cleanup()
    surface: dict[str, Any] = {
        "presets": manager.get_workflow_presets(),
        "refused": [
            [scrub(kwargs), _launch_error(manager, **kwargs)]
            for kwargs in REFUSED_LAUNCHES
        ],
    }

    async def launches() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for preset in surface["presets"]:
            for label, context in LAUNCH_CONTEXTS:
                out[f"{preset['id']}/{label}"] = await _launch_and_read(
                    manager, preset["id"], context
                )
        return out

    surface["launches"] = asyncio.run(launches())
    configure(manager=manager)
    try:
        client = TestClient(app)
        presets = client.get("/api/workflow/presets")
        surface["http_presets"] = [presets.status_code, presets.content.decode("utf-8")]
        surface["http_refused"] = []
        for body in ({}, {"preset_id": "nope"}, {"definition_json": {"a": 1}}):
            response = client.post("/api/workflow/launch", json=body)
            surface["http_refused"].append(
                [body, response.status_code, response.content.decode("utf-8")]
            )
        launched = client.post(
            "/api/workflow/launch",
            json={"preset_id": "demo_linear", "label": "L", "initial_context": {}},
        )
        info = launched.json()
        for key in ("instance_id", "created_at", "workflow_instance_id"):
            info.pop(key, None)
        surface["http_launch_linear"] = [launched.status_code, scrub(info)]
    finally:
        manager.close()
        configure()
    return surface


def snapshot() -> dict[str, Any]:
    """Everything the golden holds, read from the checkout that is imported."""
    return {
        "engine_runs": engine_runs(),
        "document_runs": document_runs(),
        "monitor": monitor_surface(),
    }
