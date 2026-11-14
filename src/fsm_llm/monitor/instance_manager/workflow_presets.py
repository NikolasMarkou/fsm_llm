"""The built-in workflow presets of the dashboard (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``_TerminalStep`` (the no-op last step), the two preset builders and ``_WORKFLOW_PRESETS``, the registry the
launch route and ``get_workflow_presets`` read. Presets use only pure steps (auto and condition), so they run
with no model and no API. Each registry call builds a fresh ``WorkflowDefinition``. Imports no other module of
the package.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.workflows import (
    WorkflowDefinition,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    condition_step,
    create_workflow,
)


# DECISION plan_2026-05-29_0c00a594/D-002 [STALE]: the dashboard exposes a small set of
# built-in, DSL-built workflow presets rather than accepting arbitrary
# `definition_json`. WorkflowDefinition.steps are WorkflowStep ABC subclasses
# carrying Python callables; there is no JSON->workflow loader, so arbitrary
# user JSON cannot be turned into a runnable workflow. Presets use only pure
# steps (auto/condition) so they run deterministically with no external LLM/API.
class _TerminalStep(WorkflowStep):
    """No-op terminal step (no outgoing transition) for monitor demo presets.

    A step whose result carries no ``next_step_id`` and which references no
    other state is treated as terminal by ``WorkflowDefinition`` and drives
    the instance to COMPLETED.
    """

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.success_result(
            message=f"Reached terminal step '{self.step_id}'"
        )


def _preset_linear() -> WorkflowDefinition:
    """Linear pipeline: ingest -> process -> finish (auto-completes on launch)."""
    wf = create_workflow(
        "demo_linear",
        "Demo: Linear Pipeline",
        "Two automatic processing steps followed by a terminal step.",
    )
    wf.with_initial_step(
        auto_step(
            "ingest",
            "Ingest",
            next_step_id="process",
            action=lambda ctx: {"ingested": True},
        )
    )
    wf.with_step(
        auto_step(
            "process",
            "Process",
            next_step_id="finish",
            action=lambda ctx: {"processed": True},
        )
    )
    wf.with_step(_TerminalStep(step_id="finish", name="Finish"))
    return wf


def _preset_branching() -> WorkflowDefinition:
    """Conditional branch on the initial-context ``amount`` value."""
    wf = create_workflow(
        "demo_branching",
        "Demo: Conditional Branch",
        "Routes to a high/low path based on the 'amount' context value.",
    )
    wf.with_initial_step(
        condition_step(
            "route",
            "Route by amount",
            condition=lambda ctx: float(ctx.get("amount", 0) or 0) >= 1000,
            true_step_id="high_path",
            false_step_id="low_path",
        )
    )
    wf.with_step(
        auto_step(
            "high_path",
            "High Path",
            next_step_id="finish",
            action=lambda ctx: {"tier": "high"},
        )
    )
    wf.with_step(
        auto_step(
            "low_path",
            "Low Path",
            next_step_id="finish",
            action=lambda ctx: {"tier": "low"},
        )
    )
    wf.with_step(_TerminalStep(step_id="finish", name="Finish"))
    return wf


# Registry of preset builders. Each call returns a fresh WorkflowDefinition so
# concurrent launches never share mutable instance state.
_WORKFLOW_PRESETS: dict[str, Any] = {
    "demo_linear": _preset_linear,
    "demo_branching": _preset_branching,
}
