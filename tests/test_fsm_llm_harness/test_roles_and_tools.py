"""Falsifying tests for ``fsm_llm_harness.roles`` and ``fsm_llm_harness.tools``.

These two modules had **zero** test references before step 7d: a ``grep`` over
``tests/`` for ``tool_scope`` / ``READ_ONLY_TOOLS`` / ``WRITE_TOOLS`` /
``build_role_prompt`` / ``COMMAND_ALLOWLIST`` returned nothing at all.  That is
why review C2 -- five of the six roles ordered by their operative rules to write
artifacts they held no write tool for -- survived 139 green tests: nothing
tested the surface it lived on.  ``run_command`` had never been executed either,
live or in test.

===============================  ==========================================
Class                            Decision / defect it pins
===============================  ==========================================
``TestToolScopeMatchesOwnership``  D-047, review C2 (scope DERIVED from
                                   ownership, checked over all six roles)
``TestPlanMemoryOwnership``        D-048, invariant I7 (the refusal itself)
``TestRootsCannotCross``           D-032, D-047 (two roots, one chokepoint)
``TestAbsolutePathRepair``         D-006 of plan-2026-07-21-bf7ffe24 (the
                                   ``/workspace/x.py`` shape lands; every
                                   escape shape still refuses)
``TestRolePromptNamesHeldTools``   review C2's other direction
``TestShellAllowlist``             D-050, review W5
===============================  ==========================================

Deterministic and offline: every filesystem test runs under ``tmp_path``, and
the one subprocess test executes ``cat`` on a file it just wrote.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from fsm_llm.definitions import StateNotFoundError
from fsm_llm.llm import _GENERIC_FALLBACK_MESSAGE, LiteLLMInterface
from fsm_llm_agents.definitions import AgentConfig, AgentResult, AgentTrace, ToolCall
from fsm_llm_agents.native_fc import NativeFunctionCallingReactAgent
from fsm_llm_agents.react import ReactAgent
from fsm_llm_harness import build_harness_fsm
from fsm_llm_harness.constants import (
    ArtifactNames,
    ContextKeys,
    Defaults,
    HarnessStates,
    PlanSchema,
    Role,
)
from fsm_llm_harness.exceptions import (
    HarnessConfinementError,
    HarnessError,
    HarnessOwnershipError,
)
from fsm_llm_harness.hardening import coerce_worker_output, parse_role_output
from fsm_llm_harness.harness import _WORKER_WRITABLE, RoleRequest
from fsm_llm_harness.roles import (
    ROLE_SPECS,
    _default_agent_builder,
    build_default_worker_factory,
    build_role_prompt,
    build_role_system_prompt,
    build_role_task_prompt,
    count_top_level_json_objects,
    get_role_spec,
    held_tools,
)
from fsm_llm_harness.rules import (
    EXPLORE_TOPICS,
    OWNERSHIP,
    ROLE_BY_STATE,
    artifacts_writable_by,
    explore_topic,
    explore_topics,
    get_rules,
)
from fsm_llm_harness.tools import (
    _COUNTERPART_TOOL,
    _PER_PLAN_DIRS,
    COMMAND_ALLOWLIST,
    DISK_DERIVED_COUNTS,
    MAX_GREP_FILE_BYTES,
    MAX_GREP_HITS,
    MAX_LIST_ENTRIES,
    PLAN_READ_TOOLS,
    PLAN_WRITE_TOOLS,
    READ_ONLY_TOOLS,
    SHELL_TOOLS,
    VERIFICATION_COMMANDS,
    WRITE_TOOLS,
    PlanMemory,
    PlanTools,
    Workspace,
    WorkspaceTools,
    _addresses_plan_memory,
    _gate_state,
    _missing_target_hint,
    _owned_empty_directory,
    _relocated_hint,
    _routing_hint,
    build_plan_tools,
    build_workspace_tools,
    count_gate_files,
    derive_disk_counts,
    gate_files,
)

# DECISION plan-2026-07-21T125237-191b2eb2/D-057
# THIS FILE IS THE ANSWER TO "why did 139 green tests coexist with review C2".
# They did not test `roles.py` or `tools.py` AT ALL: a grep over `tests/` for
# `tool_scope` / `READ_ONLY_TOOLS` / `WRITE_TOOLS` / `build_role_prompt` /
# `COMMAND_ALLOWLIST` returned zero hits, and line coverage was 43% / 28%.
# Do NOT fold these classes back into `test_harness_agent.py`: the defect class
# is per-MODULE (an untested module, not a degenerate fixture), and keeping the
# file named after the modules is what makes the next such gap visible from the
# directory listing. Do NOT drop the parametrisation over `HarnessStates.ALL` /
# `Role.WORKERS` in favour of spot-checking one role either -- C2 was five of
# six roles wrong and one right, so a spot check would have passed.
# See decisions.md D-057.
#: Executables that run code the EXECUTE role can author inside the workspace.
#:
#: ``tools.py``'s own comment claims the default allowlist "executes nothing
#: that lives inside the workspace"; review W5 showed the claim was false while
#: ``make`` / ``pytest`` / ``git`` were on it.  This set is the claim, written
#: as an assertion.  A ``Makefile``, a ``conftest.py`` and ``.git/hooks/*`` are
#: all workspace files, and an interpreter needs no file at all.
_CODE_EXECUTING = frozenset(
    {
        "awk",
        "bash",
        "cargo",
        "dash",
        "env",
        "find",
        "git",
        "go",
        "gradle",
        "java",
        "make",
        "mypy",
        "node",
        "npm",
        "npx",
        "perl",
        "php",
        "pytest",
        "python",
        "python3",
        "ruby",
        "ruff",
        "sed",
        "sh",
        "tox",
        "xargs",
        "zsh",
    }
)


#: The states whose role owns at least one artifact, and its complement.
#:
#: Computed rather than skipped-over inside the tests: a ``pytest.skip`` here
#: would make "no state owns anything" and "every state owns something" both
#: look like a clean run.  ``test_the_two_scope_families_partition_the_states``
#: below asserts the partition is real.
_OWNING_STATES = tuple(
    state for state in HarnessStates.ALL if ROLE_SPECS[state].owned_artifacts
)
_NON_OWNING_STATES = tuple(
    state for state in HarnessStates.ALL if not ROLE_SPECS[state].owned_artifacts
)


#: One schema-valid payload per state, DERIVED from the state's own schema so a
#: new writable key cannot leave these samples silently stale.  ``message`` is
#: excluded on purpose -- the tests below add it (or not) themselves.
_TYPE_SAMPLE: dict[type, Any] = {int: 3, bool: True, str: "sample"}
_SCHEMA_SAMPLE: dict[str, dict[str, Any]] = {
    state: {
        name: _TYPE_SAMPLE[info.annotation]
        for name, info in ROLE_SPECS[state].output_schema.model_fields.items()
        if name != "message"
    }
    for state in HarnessStates.ALL
}


def _parse_as_core_would(content: str) -> Any:
    """Run *content* through core's REAL response-generation parser.

    Interface contract (2 call sites, both in
    ``TestEveryRoleSchemaCarriesMessage``):
        - Parameter: the exact text a role would return.
        - Returns core's ``ResponseGenerationResponse``.
        - Calls the real private method on an uninitialised interface (no
          network, no config): a reimplementation here would assert against a
          copy of the guard rather than against the guard.
    """
    llm = LiteLLMInterface.__new__(LiteLLMInterface)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return llm._parse_response_generation_response(response)


def _role_request(
    state: str,
    *,
    plan_dir: Path | None,
    workspace_root: Path | None = None,
    context: dict[str, Any] | None = None,
    assigned_topic: str | None = None,
    assigned_write_target: str | None = None,
) -> RoleRequest:
    """Build the ``RoleRequest`` the driver would hand this state's worker.

    Interface contract (several call sites below):
        - Uses the REAL frozen rules for *state*, not placeholder prose, so a
          prompt assertion is made against the string a live dispatch renders.
        - ``plan_dir=None`` reproduces the production degrade shape: no plan
          directory means the role holds no plan-file tool at all.
    """
    from fsm_llm_harness.rules import get_rules

    rules = get_rules(state)
    return RoleRequest(
        role=ROLE_BY_STATE[state],
        state=state,
        goal="exercise the harness protocol",
        operative_rules=rules.operative_rules,
        gate_summary=rules.gate_summary,
        iteration=1,
        step_number=1,
        total_steps=1,
        fix_attempts=0,
        context=(
            {ContextKeys.GOAL: "exercise the harness protocol"}
            if context is None
            else context
        ),
        plan_dir=None if plan_dir is None else str(plan_dir),
        workspace_root=None if workspace_root is None else str(workspace_root),
        assigned_topic=assigned_topic,
        assigned_write_target=assigned_write_target,
    )


# ---------------------------------------------------------------------------
# Tool scope vs the ownership table (review C2, D-047)
# ---------------------------------------------------------------------------


class TestToolScopeMatchesOwnership:
    """Every role's tool scope is a superset of what ``OWNERSHIP`` requires.

    This is the check whose ABSENCE let C2 through.  It is asserted
    programmatically over all six roles rather than by inspecting a table,
    because the defect was a hand-maintained scope table drifting from the
    ownership table it was supposed to encode.
    """

    def test_every_worker_role_has_exactly_one_spec(self) -> None:
        """The six specs and the six dispatchable roles are the same set."""
        assert len(ROLE_SPECS) == len(HarnessStates.ALL)
        assert {spec.role for spec in ROLE_SPECS.values()} == set(Role.WORKERS)

    def test_the_two_scope_families_partition_the_states(self) -> None:
        """Both parametrised families below are non-empty and cover all 6 states.

        Without this, deleting every ``OWNERSHIP`` entry would empty one
        family, silently collect zero tests from it, and still report green.
        """
        assert _OWNING_STATES and _NON_OWNING_STATES
        assert set(_OWNING_STATES) | set(_NON_OWNING_STATES) == set(HarnessStates.ALL)
        assert not set(_OWNING_STATES) & set(_NON_OWNING_STATES)

    @pytest.mark.parametrize("state", _OWNING_STATES)
    def test_a_role_that_owns_an_artifact_holds_a_plan_write_tool(
        self, state: str
    ) -> None:
        """C2 in one assertion: ordered to write, and able to.

        ``rules.py`` orders the explorer to write ``findings/<topic>.md``, the
        plan-writer to seed ``verification.md`` and the archivist to rewrite six
        cross-plan files.  Before D-047 only EXECUTE held any write tool, so
        five of those instructions were unexecutable -- which is a mechanical
        explanation for the live spike's "workspace byte-identical" result that
        D-036/D-037 attributed to model capability.
        """
        spec = get_role_spec(state)
        assert set(PLAN_WRITE_TOOLS) <= set(spec.plan_tool_scope), (
            f"{spec.role} is granted {spec.owned_artifacts} by OWNERSHIP and "
            f"holds no tool that can write them"
        )

    @pytest.mark.parametrize("state", _NON_OWNING_STATES)
    def test_a_role_that_owns_nothing_holds_no_plan_write_tool(
        self, state: str
    ) -> None:
        """The other direction: scope is a SUBSET of what ownership permits."""
        spec = get_role_spec(state)
        assert not set(PLAN_WRITE_TOOLS) & set(spec.plan_tool_scope)

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_owned_artifacts_are_exactly_the_ownership_projection(
        self, state: str
    ) -> None:
        """``owned_artifacts`` is read from ``OWNERSHIP``, never restated."""
        spec = get_role_spec(state)
        assert spec.owned_artifacts == artifacts_writable_by(spec.role)

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_every_role_can_read_protocol_memory(self, state: str) -> None:
        """ "Read the artifacts before acting" is the first rule almost everywhere."""
        spec = get_role_spec(state)
        assert set(PLAN_READ_TOOLS) <= set(spec.plan_tool_scope)

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_plan_tool_scope_names_only_registrable_tools(
        self, state: str, plan_dir: Path
    ) -> None:
        """A misspelt tool name is a silently missing capability, not an error."""
        spec = get_role_spec(state)
        assert set(spec.plan_tool_scope) <= set(PLAN_READ_TOOLS + PLAN_WRITE_TOOLS)
        # Registration is the proof: `build_plan_tools` raises on an unknown name.
        registry = build_plan_tools(
            PlanMemory(plan_dir, role=spec.role), allowed=spec.plan_tool_scope
        )
        assert set(registry.tool_names) == set(spec.plan_tool_scope)

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_every_role_holds_the_read_only_workspace_tools(self, state: str) -> None:
        """A role that can read nothing cannot do its job."""
        assert set(READ_ONLY_TOOLS) <= set(get_role_spec(state).tool_scope)

    def test_only_execute_holds_workspace_write_tools(self) -> None:
        """Invariant I7 at the workspace root: mutation belongs to EXECUTE."""
        holders = {
            state
            for state, spec in ROLE_SPECS.items()
            if set(spec.tool_scope) & set(WRITE_TOOLS)
        }
        assert holders == {HarnessStates.EXECUTE}

    def test_only_reflect_holds_the_shell_tool(self) -> None:
        """Verification is the only job that needs to run a command."""
        holders = {
            state
            for state, spec in ROLE_SPECS.items()
            if set(spec.tool_scope) & set(SHELL_TOOLS)
        }
        assert holders == {HarnessStates.REFLECT}

    @pytest.mark.parametrize("role", Role.WORKERS)
    def test_every_owned_artifact_is_authorised_by_that_roles_plan_memory(
        self, tmp_path: Path, role: str
    ) -> None:
        """The tool-layer grant and the ownership check agree, per role.

        A coarse ``PLAN_WRITE_TOOLS`` grant that ``PlanMemory.authorise`` then
        refuses would be C2 again in a subtler form: the role holds a tool it
        can never successfully call.
        """
        memory = PlanMemory(tmp_path / "plans" / "plan-under-test", role=role)
        for artifact in artifacts_writable_by(role):
            path = (
                f"{artifact}/probe.md"
                if artifact
                in (ArtifactNames.FINDINGS_DIR, ArtifactNames.CHECKPOINTS_DIR)
                else artifact
            )
            assert memory.authorise(path)

    @pytest.mark.parametrize("role", Role.WORKERS)
    def test_every_unowned_artifact_is_refused_by_that_roles_plan_memory(
        self, tmp_path: Path, role: str
    ) -> None:
        """And the complement: nothing outside the grant is writable."""
        memory = PlanMemory(tmp_path / "plans" / "plan-under-test", role=role)
        owned = set(artifacts_writable_by(role))
        for artifact in OWNERSHIP:
            if artifact in owned:
                continue
            path = (
                f"{artifact}/probe.md"
                if artifact
                in (ArtifactNames.FINDINGS_DIR, ArtifactNames.CHECKPOINTS_DIR)
                else artifact
            )
            with pytest.raises(HarnessOwnershipError):
                memory.authorise(path)


# ---------------------------------------------------------------------------
# The ownership refusal itself (invariant I7)
# ---------------------------------------------------------------------------


class TestPlanMemoryOwnership:
    """``HarnessOwnershipError`` was raised nowhere in the codebase before 7b."""

    def test_an_explorer_writes_a_finding(self, tmp_path: Path) -> None:
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.EXPLORER)
        written = memory.write_text("findings/tool-scope.md", "evidence\n")

        assert (memory.plan_dir / "findings" / "tool-scope.md").read_text() == (
            "evidence\n"
        )
        assert written.endswith("findings/tool-scope.md")

    def test_an_explorer_cannot_write_the_plan(self, tmp_path: Path) -> None:
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.EXPLORER)

        with pytest.raises(HarnessOwnershipError) as excinfo:
            memory.write_text(ArtifactNames.PLAN, "a plan the explorer invented")

        assert ArtifactNames.PLAN in str(excinfo.value)
        assert not (memory.plan_dir / ArtifactNames.PLAN).exists()

    def test_a_refused_write_touches_nothing(self, tmp_path: Path) -> None:
        """Authorisation happens BEFORE any I/O, so a refusal cannot truncate."""
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.PLAN_WRITER)
        memory.write_text(ArtifactNames.PLAN, "the real plan\n")

        archivist = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.ARCHIVIST)
        with pytest.raises(HarnessOwnershipError):
            archivist.write_text(ArtifactNames.PLAN, "")

        assert (memory.plan_dir / ArtifactNames.PLAN).read_text() == "the real plan\n"

    def test_a_path_that_is_not_a_protocol_artifact_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The ownership table is the WHOLE of what may be written."""
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.ARCHIVIST)
        with pytest.raises(HarnessOwnershipError):
            memory.write_text("notes/scratch.md", "not an artifact")

    # DECISION plan-2026-07-21T125237-191b2eb2/D-058
    # These two tests pin `PlanMemory.locate`'s addressing rule AS IT BEHAVES,
    # not as `_classify`'s "Another plan's directory" comment reads. A bare
    # `plan-old/summary.md` is namespaced under THIS plan, so the sibling tier
    # is reachable only through `<my-plan-id>/../plan-old/...`. Do NOT "fix"
    # `locate` to special-case sibling plan ids to make the first test go away:
    # the cross-plan sliding window that would consume it has no implementation
    # yet (step 9/11), so a friendlier rule now is designed against no caller --
    # and this file is a test-only step. The half that IS load-bearing today,
    # asserted below, is that the WRITE is refused however the path is spelled.
    # See decisions.md D-058.
    def test_a_bare_relative_path_is_namespaced_under_this_plan(
        self, tmp_path: Path
    ) -> None:
        """``locate`` prefixes anything that is not this plan or a cross-plan file.

        So ``plan-old/summary.md`` addresses ``<my-plan>/plan-old/summary.md``,
        NOT the sibling plan directory -- a role cannot reach another plan by
        guessing its name.
        """
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.ARCHIVIST)
        assert memory.locate(f"plan-old/{ArtifactNames.SUMMARY}") == (
            f"plan-x/plan-old/{ArtifactNames.SUMMARY}"
        )

    def test_another_plans_directory_is_readable_but_never_writable(
        self, tmp_path: Path
    ) -> None:
        """The cross-plan sliding window needs the read; nothing needs the write.

        Reaching a sibling plan takes an explicit ``<my-plan-id>/../<other>``
        path, because ``locate`` namespaces every bare relative path under this
        plan (see the test above).  FOUND at step 7d and deliberately NOT
        "fixed" here: the sliding window that would consume this has no
        implementation yet (step 9/11), so a friendlier addressing rule now
        would be designed against no caller.  What matters today is the half
        that is load-bearing -- the write is refused however the path is
        spelled.
        """
        (tmp_path / "plans" / "plan-old").mkdir(parents=True)
        (tmp_path / "plans" / "plan-old" / ArtifactNames.SUMMARY).write_text("old\n")

        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.ARCHIVIST)
        sibling = f"plan-x/../plan-old/{ArtifactNames.SUMMARY}"

        assert memory.read_text(sibling) == "old\n"
        with pytest.raises(HarnessOwnershipError):
            memory.write_text(sibling, "rewritten")
        assert (
            tmp_path / "plans" / "plan-old" / ArtifactNames.SUMMARY
        ).read_text() == "old\n"

    def test_the_archivist_reaches_the_cross_plan_tier(self, tmp_path: Path) -> None:
        """Those files live BESIDE the plan directories, not inside one (D-047)."""
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.ARCHIVIST)
        memory.write_text(ArtifactNames.LESSONS, "- a lesson [I:5]\n")

        assert (tmp_path / "plans" / ArtifactNames.LESSONS).exists()


# ---------------------------------------------------------------------------
# Two roots, one confinement chokepoint (D-032, D-047)
# ---------------------------------------------------------------------------


class TestRootsCannotCross:
    """A role must never reach one root through the other's tool."""

    def test_the_two_tool_name_sets_are_disjoint(self) -> None:
        """A shared name would make the two roots reachable interchangeably."""
        assert not set(READ_ONLY_TOOLS + SHELL_TOOLS) & set(
            PLAN_READ_TOOLS + PLAN_WRITE_TOOLS
        )

    def test_a_workspace_path_cannot_reach_the_plan_directory(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "plans" / "plan-x").mkdir(parents=True)
        (tmp_path / "plans" / "plan-x" / ArtifactNames.PLAN).write_text("secret\n")
        ws = Workspace(tmp_path / "workspace")

        with pytest.raises(HarnessConfinementError):
            ws.read_text(f"../plans/plan-x/{ArtifactNames.PLAN}")

    def test_a_plan_path_cannot_reach_the_workspace(self, tmp_path: Path) -> None:
        (tmp_path / "workspace").mkdir()
        (tmp_path / "workspace" / "app.py").write_text("print('hi')\n")
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.EXECUTOR)

        with pytest.raises(HarnessConfinementError):
            memory.read_text("../../workspace/app.py")

    @pytest.mark.parametrize(
        "escape",
        ["/etc/passwd", "../outside.txt", "a/../../outside.txt", "bad\x00name", "  "],
    )
    def test_workspace_confinement_rejects_every_escape_shape(
        self, tmp_path: Path, escape: str
    ) -> None:
        ws = Workspace(tmp_path / "workspace")
        with pytest.raises(HarnessConfinementError):
            ws.resolve(escape)

    def test_a_symlink_out_of_the_workspace_is_rejected(self, tmp_path: Path) -> None:
        """Resolve FIRST, compare SECOND -- a lexical check passes this (D-032)."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret\n")
        ws = Workspace(tmp_path / "workspace")
        (ws.root / "link").symlink_to(outside)

        with pytest.raises(HarnessConfinementError):
            ws.read_text("link/secret.txt")

    def test_a_sibling_root_with_a_shared_prefix_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """``/tmp/ws-evil`` startswith ``/tmp/ws``; the check is on components."""
        (tmp_path / "ws-evil").mkdir()
        (tmp_path / "ws-evil" / "loot.txt").write_text("loot\n")
        ws = Workspace(tmp_path / "ws")

        with pytest.raises(HarnessConfinementError):
            ws.read_text("../ws-evil/loot.txt")


# ---------------------------------------------------------------------------
# The absolute-path repair, and what it still refuses (D-006)
# ---------------------------------------------------------------------------


class TestAbsolutePathRepair:
    """``:4b`` emits ``/workspace/uploader.py``; that shape now lands.

    Nothing else does.  The repair drops **one** leading sentinel component and
    then hands the remainder to D-032's untouched resolve-and-compare, so every
    escape shape that was refused before this class existed is still refused --
    including the ones that try to ride the sentinel out (``..``-chaining, a
    symlink, a sibling root with a shared prefix).
    """

    def test_the_measured_failure_shape_now_writes_bytes(self, tmp_path: Path) -> None:
        """The exact string ``:4b`` emitted, end to end through write_text."""
        ws = Workspace(tmp_path / "ws")

        ws.write_text("/workspace/uploader.py", "print('up')\n")

        assert (ws.root / "uploader.py").read_text() == "print('up')\n"

    def test_the_roots_own_basename_is_a_sentinel(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws")

        assert ws.resolve("/ws/x.py") == ws.root / "x.py"

    def test_a_bare_sentinel_maps_to_the_workspace_root(self, tmp_path: Path) -> None:
        """`/workspace` with nothing after it is the root, not a refusal.

        The live `list_dir("/workspace")` the model emits means the root; the
        bare sentinel now takes the same resolve-and-compare as `/workspace/x`
        (D-004).  `/` alone is still refused -- see the escape parametrize.
        """
        ws = Workspace(tmp_path / "ws")

        assert ws.resolve("/workspace") == ws.root
        assert ws.resolve("/ws") == ws.root

    def test_an_absolute_path_already_inside_the_root_is_accepted(
        self, tmp_path: Path
    ) -> None:
        """Branch (a): no repair needed, and the same compare still decides."""
        ws = Workspace(tmp_path / "ws")
        (ws.root / "pkg").mkdir()

        assert ws.resolve(str(ws.root / "pkg" / "mod.py")) == ws.root / "pkg" / "mod.py"

    @pytest.mark.parametrize(
        "escape",
        [
            "/workspace/../../etc/passwd",
            "/ws/../../etc/passwd",
            "/etc/workspace/passwd",
            "/etc/passwd",
            "/",
        ],
    )
    def test_the_repair_cannot_be_ridden_out_of_the_root(
        self, tmp_path: Path, escape: str
    ) -> None:
        """Sentinel-then-resolve: ``..`` after a sentinel still climbs into a
        refusal, and a sentinel that is not the FIRST component is not one."""
        ws = Workspace(tmp_path / "ws")

        with pytest.raises(HarnessConfinementError):
            ws.resolve(escape)

    def test_a_symlink_reached_through_the_sentinel_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """The repair is lexical; only the unchanged resolve sees the link."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret\n")
        ws = Workspace(tmp_path / "ws")
        (ws.root / "link").symlink_to(outside)

        with pytest.raises(HarnessConfinementError):
            ws.read_text("/workspace/link/secret.txt")

    def test_a_shared_prefix_sibling_reached_through_the_sentinel_is_rejected(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "ws-evil").mkdir()
        (tmp_path / "ws-evil" / "loot.txt").write_text("loot\n")
        ws = Workspace(tmp_path / "ws")

        with pytest.raises(HarnessConfinementError):
            ws.read_text("/ws/../ws-evil/loot.txt")

    @pytest.mark.parametrize("prefix", ["/plan", "/plan-x", "/plans/plan-x"])
    def test_plan_memory_repairs_into_the_plan_directory(
        self, tmp_path: Path, prefix: str
    ) -> None:
        """Not into the memory root -- the composed workspace is rooted one
        level above the plan directory, so the strip must happen in locate."""
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.PLAN_WRITER)

        memory.write_text(f"{prefix}/{ArtifactNames.PLAN}", "# plan\n")

        assert (
            tmp_path / "plans" / "plan-x" / ArtifactNames.PLAN
        ).read_text() == "# plan\n"

    @pytest.mark.parametrize(
        "escape",
        ["/etc/passwd", "/plan/../../../etc/passwd", "/etc/plan/passwd"],
    )
    def test_plan_memory_still_refuses_every_escape(
        self, tmp_path: Path, escape: str
    ) -> None:
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.PLAN_WRITER)

        with pytest.raises(HarnessConfinementError):
            memory.read_text(escape)

    def test_a_repaired_path_gains_no_write_into_another_plan(
        self, tmp_path: Path
    ) -> None:
        """``/plan/../plan-y/plan.md`` stays inside the memory root, so it is
        confinement-legal and ownership-illegal -- exactly as the unrepaired
        ``../plan-y/plan.md`` already was.  The repair reaches the ownership
        layer; it must not slip past it."""
        (tmp_path / "plans" / "plan-y").mkdir(parents=True)
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.PLAN_WRITER)

        with pytest.raises(HarnessOwnershipError):
            memory.write_text(f"/plan/../plan-y/{ArtifactNames.PLAN}", "stolen\n")

        assert not (tmp_path / "plans" / "plan-y" / ArtifactNames.PLAN).exists()


# ---------------------------------------------------------------------------
# The prompt names exactly the tools the dispatch holds (review C2, other way)
# ---------------------------------------------------------------------------


class TestRolePromptNamesHeldTools:
    """A prompt naming a tool the role does not hold is just as unexecutable."""

    @staticmethod
    def _named_tools(prompt: str) -> tuple[str, ...]:
        """Parse the prompt's ``TOOLS:`` line back into tool names.

        Split on the rendered separator rather than substring-searching for
        each name: ``path_exists`` is a substring of ``plan_path_exists``, so a
        containment check would pass for a role that holds neither.
        """
        line = next(line for line in prompt.splitlines() if line.startswith("TOOLS: "))
        return tuple(line[len("TOOLS: ") :].split(".")[0].split(", "))

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_prompt_names_exactly_the_held_tools(
        self, state: str, plan_dir: Path, workspace: Path
    ) -> None:
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=plan_dir, workspace_root=workspace)

        prompt = build_role_prompt(request, spec)

        assert self._named_tools(prompt) == held_tools(request, spec)

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_a_dispatch_without_a_plan_directory_names_no_plan_tool(
        self, state: str
    ) -> None:
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=None)

        prompt = build_role_prompt(request, spec)

        assert self._named_tools(prompt) == tuple(spec.tool_scope)
        assert not set(self._named_tools(prompt)) & set(
            PLAN_READ_TOOLS + PLAN_WRITE_TOOLS
        )
        assert "YOU MAY WRITE no protocol file" in prompt

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_verb_is_derived_from_the_tools_actually_registered(
        self, state: str, plan_dir: Path
    ) -> None:
        """Hardcoding "inspect and change" is what told five roles to write."""
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=plan_dir)
        names = held_tools(request, spec)
        can_write = bool(set(names) & set(PLAN_WRITE_TOOLS + WRITE_TOOLS))

        prompt = build_role_prompt(request, spec)

        assert ("inspect and change real files" in prompt) is can_write

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_prompt_lists_every_artifact_the_role_may_write(
        self, state: str, plan_dir: Path
    ) -> None:
        spec = get_role_spec(state)
        prompt = build_role_prompt(_role_request(state, plan_dir=plan_dir), spec)

        for artifact in spec.owned_artifacts:
            assert artifact in prompt

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_registry_a_dispatch_gets_holds_exactly_those_tools(
        self, state: str, plan_dir: Path, workspace: Path
    ) -> None:
        """The prompt and the registry are one function's output (``held_tools``)."""
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=plan_dir, workspace_root=workspace)

        registry = build_workspace_tools(Workspace(workspace), allowed=spec.tool_scope)
        build_plan_tools(
            PlanMemory(plan_dir, role=spec.role),
            allowed=spec.plan_tool_scope,
            registry=registry,
        )

        assert set(registry.tool_names) == set(held_tools(request, spec))


# ---------------------------------------------------------------------------
# Every role schema carries `message` (D-004 of plan-2026-07-21-bf7ffe24)
# ---------------------------------------------------------------------------


class TestEveryRoleSchemaCarriesMessage:
    """``message`` is prose for core's rescue rung, never a protocol key.

    The pairing below is the whole point: the field must be visible to the
    model (schema + prompt) and invisible to the gate (not writable, not
    required).  ``summary`` established that rule at D-035; this asserts
    ``message`` obeys it for all six roles rather than for the one that was
    spot-checked.
    """

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_schema_requires_a_string_message(self, state: str) -> None:
        info = get_role_spec(state).output_schema.model_fields["message"]

        assert info.annotation is str
        assert info.is_required()

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_message_is_not_writable_into_context(self, state: str) -> None:
        assert "message" not in _WORKER_WRITABLE[state]
        assert "message" not in get_role_spec(state).writable_keys

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_coerce_drops_message_even_when_the_worker_sends_it(
        self, state: str
    ) -> None:
        spec = get_role_spec(state)
        payload = {"message": "smuggled prose", **_SCHEMA_SAMPLE[state]}

        accepted = coerce_worker_output(payload, spec.writable_keys, where=spec.role)

        assert "message" not in accepted
        assert set(accepted) == set(spec.writable_keys)

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_parse_role_output_does_not_require_message(self, state: str) -> None:
        spec = get_role_spec(state)

        output = parse_role_output(
            dict(_SCHEMA_SAMPLE[state]), expected_keys=spec.expected_keys
        )

        assert "message" not in spec.expected_keys
        assert output.success is True
        assert output.failure_reason is None

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_prompt_asks_for_the_message_field(
        self, state: str, plan_dir: Path
    ) -> None:
        """The prompt shape is read off the built schema, so it cannot drift."""
        spec = get_role_spec(state)

        prompt = build_role_prompt(_role_request(state, plan_dir=plan_dir), spec)

        assert '"message": <string>' in prompt

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_a_bare_envelope_carrying_message_survives_cores_parser(
        self, state: str
    ) -> None:
        """The RESCUE path, asserted -- not the guard D-022 keeps permanent."""
        payload = {**_SCHEMA_SAMPLE[state], "message": "Indexed three findings."}

        parsed = _parse_as_core_would(json.dumps(payload))

        assert parsed.message == "Indexed three findings."
        assert parsed.message != _GENERIC_FALLBACK_MESSAGE

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_same_envelope_without_message_is_still_destroyed(
        self, state: str
    ) -> None:
        """The control: core's guard is untouched, so the field is what saves it."""
        parsed = _parse_as_core_would(json.dumps(_SCHEMA_SAMPLE[state]))

        assert parsed.message == _GENERIC_FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# PLAN's 11-section response_format schema (D-001 of this plan)
# ---------------------------------------------------------------------------


class TestPlanOutputSchemaCarriesElevenSectionFields:
    """PLAN's ``output_schema`` authors the 11 ``PlanSchema.SECTIONS`` as
    required string content fields, DERIVED from the section tuple (never
    hand-written), and those fields stay OUT of the writable-key allowlist.

    This is Success Criterion 1: the model authors the sections under
    ``response_format`` (schema-visible), the driver RENDERS them into
    ``plan.md``, and they never reach FSM context as gate keys (not writable).
    It is the ``summary`` special-case (D-035) applied to eleven fields at once
    -- so a regression that either drops a section from the schema OR promotes a
    section body into ``_WORKER_WRITABLE`` fails here.
    """

    def test_the_field_set_is_the_slugs_plus_the_three_gate_and_prose_keys(
        self,
    ) -> None:
        """Field set == 11 SECTIONS slugs plus needs_explore/total_steps/message.

        Derived, not enumerated: the expected set is built FROM
        ``PlanSchema.SECTION_SLUGS`` so adding a 12th section would have to add
        its field here too, and a hand-typed literal cannot silently drift.
        """
        spec = get_role_spec(HarnessStates.PLAN)

        fields = set(spec.output_schema.model_fields)

        expected = set(PlanSchema.SECTION_SLUGS) | {
            ContextKeys.NEEDS_EXPLORE,
            ContextKeys.TOTAL_STEPS,
            "message",
        }
        assert fields == expected

    def test_all_eleven_content_fields_are_required_strings(self) -> None:
        """A reply missing any section fails schema validation (fail closed)."""
        model_fields = get_role_spec(HarnessStates.PLAN).output_schema.model_fields

        for slug in PlanSchema.SECTION_SLUGS:
            info = model_fields[slug]
            assert info.annotation is str, slug
            assert info.is_required(), slug

    def test_the_eleven_slugs_are_absent_from_worker_writable(self) -> None:
        """The section bodies are CONTENT, never gate keys: the driver renders
        them, and only ``needs_explore``/``total_steps`` may reach context."""
        spec = get_role_spec(HarnessStates.PLAN)

        for slug in PlanSchema.SECTION_SLUGS:
            assert slug not in _WORKER_WRITABLE[HarnessStates.PLAN], slug
            assert slug not in spec.writable_keys, slug

        assert set(spec.writable_keys) == {
            ContextKeys.NEEDS_EXPLORE,
            ContextKeys.TOTAL_STEPS,
        }

    def test_the_slug_maps_are_consistent_and_section_ordered(self) -> None:
        """The section↔slug maps are exact inverses over the 11 unique slugs,
        and ``SECTION_SLUGS`` is in ``SECTIONS`` order -- the invariant the
        renderer relies on to place each field under ITS heading."""
        assert set(PlanSchema.SLUG_BY_SECTION.keys()) == set(PlanSchema.SECTIONS)
        assert PlanSchema.SECTION_BY_SLUG == {
            slug: section for section, slug in PlanSchema.SLUG_BY_SECTION.items()
        }
        assert len(set(PlanSchema.SECTION_SLUGS)) == len(PlanSchema.SECTIONS) == 11
        assert PlanSchema.SECTION_SLUGS == tuple(
            PlanSchema.SLUG_BY_SECTION[section] for section in PlanSchema.SECTIONS
        )


# ---------------------------------------------------------------------------
# Turn budget + stopping condition (D-013 of plan-2026-07-21-bf7ffe24)
# ---------------------------------------------------------------------------

#: The EXPLORE budget that was MEASURED to be unusable: 5/5 live dispatches on
#: ``ollama_chat/qwen3.5:4b`` spent all 8 turns on read tools and wrote zero
#: bytes.  Named here so a future "tightening" back to it fails a test instead
#: of quietly re-creating the run that could not have succeeded.
_MEASURED_INSUFFICIENT_TURNS = 8

#: The floor every role budget must clear: a few reads, a write per owned
#: artifact, and one turn left to stop and answer.
_MIN_TURNS_PER_ROLE = 10


class TestTurnBudgetAndStoppingCondition:
    """A role that cannot fit "read, write, stop" cannot write -- structurally."""

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_every_role_budget_clears_the_floor(self, state: str) -> None:
        assert get_role_spec(state).max_iterations >= _MIN_TURNS_PER_ROLE

    def test_the_explore_budget_is_above_the_measured_insufficient_one(self) -> None:
        explore = get_role_spec(HarnessStates.EXPLORE)

        assert explore.max_iterations > _MEASURED_INSUFFICIENT_TURNS

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_prompt_states_the_turn_budget_and_the_stop_rule(
        self, state: str, plan_dir: Path
    ) -> None:
        spec = get_role_spec(state)

        prompt = build_role_prompt(_role_request(state, plan_dir=plan_dir), spec)

        assert f"HOW TO FINISH: you have at most {spec.max_iterations} turns" in prompt
        assert "stop calling tools and answer" in prompt

    @pytest.mark.parametrize("state", _OWNING_STATES)
    def test_a_writing_role_is_told_the_write_is_the_deliverable(
        self, state: str, plan_dir: Path
    ) -> None:
        """The measured failure was reading until the budget ran out."""
        spec = get_role_spec(state)

        prompt = build_role_prompt(_role_request(state, plan_dir=plan_dir), spec)

        assert "WRITE -- the write is the deliverable" in prompt
        assert "Never state that you wrote a file unless a write tool" in prompt

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_write_obligation_is_derived_from_the_tools_held(
        self, state: str
    ) -> None:
        """Ordering a role to write with no write tool is the review-C2 defect.

        Without a plan directory only EXECUTE keeps a write tool (the workspace
        one), so the obligation must survive for exactly that state and vanish
        for the other five.
        """
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=None)
        can_write = bool(
            set(held_tools(request, spec)) & set(WRITE_TOOLS + PLAN_WRITE_TOOLS)
        )

        prompt = build_role_prompt(request, spec)

        assert ("the write is the deliverable" in prompt) is can_write
        assert "stop calling tools and answer" in prompt

    @pytest.mark.parametrize("state", _OWNING_STATES)
    def test_the_prompt_names_the_write_tool_it_holds(
        self, state: str, plan_dir: Path
    ) -> None:
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=plan_dir)

        prompt = build_role_prompt(request, spec)

        assert "write_plan_file" in prompt
        assert "write_plan_file" in held_tools(request, spec)

    @pytest.mark.parametrize("state", _OWNING_STATES)
    def test_every_path_shape_the_prompt_offers_is_one_the_tool_accepts(
        self, state: str, plan_dir: Path
    ) -> None:
        """The prompt's promise, executed against the REAL ownership check.

        Measured: told it could write ``findings``, ``:4b`` checked whether that
        folder existed and gave up.  ``findings/<topic>.md`` is the shape
        ``PlanMemory`` actually authorises, so the prompt must offer that one --
        and this asserts the offer by making the write.
        """
        spec = get_role_spec(state)
        memory = PlanMemory(plan_dir, role=spec.role)

        prompt = build_role_prompt(_role_request(state, plan_dir=plan_dir), spec)

        for artifact in spec.owned_artifacts:
            is_dir = artifact in _PER_PLAN_DIRS
            offered = f"{artifact}/<topic>.md" if is_dir else artifact
            assert offered in prompt
            written = memory.write_text(offered.replace("<topic>", "sample"), "x\n")
            assert Path(written).name


# ---------------------------------------------------------------------------
# The shell allowlist (D-050, review W5)
# ---------------------------------------------------------------------------


class TestShellAllowlist:
    """``run_command`` had never been executed, live or in test, before 7d."""

    def test_no_default_command_executes_workspace_authored_code(self) -> None:
        """The allowlist's own comment, asserted rather than believed.

        ``make`` runs a ``Makefile``, ``pytest`` imports ``conftest.py``,
        ``git`` honours ``.git/hooks/*`` and a repo-local ``core.pager`` -- all
        of them files the EXECUTE role can write.
        """
        assert not set(COMMAND_ALLOWLIST) & _CODE_EXECUTING

    def test_the_verification_commands_are_opt_in_by_name(self) -> None:
        assert not set(COMMAND_ALLOWLIST) & set(VERIFICATION_COMMANDS)
        assert set(VERIFICATION_COMMANDS) <= _CODE_EXECUTING

    def test_shell_access_is_off_by_default(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "workspace")
        assert ws.allow_shell is False
        with pytest.raises(Exception, match="disabled"):
            ws.run_command(["cat", "anything"])

    def test_an_allowlisted_command_runs_inside_the_root(self, tmp_path: Path) -> None:
        """The control case: the allowlist is not simply refusing everything."""
        ws = Workspace(tmp_path / "workspace", allow_shell=True)
        ws.write_text("notes.txt", "hello from the workspace\n")

        result = ws.run_command(["cat", "notes.txt"])

        assert result.success is True
        assert "hello from the workspace" in result.result

    def test_a_non_allowlisted_executable_is_refused_even_with_shell_on(
        self, tmp_path: Path
    ) -> None:
        ws = Workspace(tmp_path / "workspace", allow_shell=True)
        with pytest.raises(Exception, match="not allowlisted"):
            ws.run_command(["python", "-c", "print(1)"])

    def test_a_path_qualified_executable_is_refused(self, tmp_path: Path) -> None:
        """``/bin/sh`` must not slip past a basename allowlist."""
        ws = Workspace(tmp_path / "workspace", allow_shell=True)
        with pytest.raises(Exception, match="bare executable name"):
            ws.run_command(["/bin/cat", "notes.txt"])

    def test_the_subprocess_environment_carries_no_parent_secrets(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An API key in the parent process is not the workspace's to hand on."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-cross")
        ws = Workspace(tmp_path / "workspace", allow_shell=True)

        env = ws._command_env()

        assert "OPENAI_API_KEY" not in env
        assert set(env) == {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"}


# ---------------------------------------------------------------------------
# Mechanical verification: disk beats the claim (D-015, D-016)
# ---------------------------------------------------------------------------

#: The exact EXECUTE reply the write-tool RCA measured 5/5: a confident,
#: well-formed completion claim from a run that called NO write tool at all.
_FABRICATED_EXECUTE_ANSWER = (
    "Implemented retry-with-backoff in uploader.py using exponential backoff. "
    '{"summary": "added retry with backoff to uploader.py", '
    '"message": "The uploader now retries failed requests with exponential backoff."}'
)


class _ScriptedAgent:
    """An agent that runs a SCRIPT of tool calls through the REAL registry.

    Interface contract (test double; several call sites below):
        - ``registry``: the registry the factory built for this dispatch, so a
          scripted ``write_plan_file`` really writes bytes and a scripted write
          the role does not own is really REFUSED.  A double that fabricated
          both the trace and the bytes would make every assertion below
          vacuous, which is exactly the fixture defect ``plans/LESSONS.md``
          [I:5] records.
        - ``calls``: ``(tool_name, parameters)`` pairs, executed in order.
        - Returns an ``AgentResult`` whose ``trace`` holds those calls and whose
          ``structured_output`` is the scripted payload.
    """

    def __init__(
        self,
        registry: Any,
        calls: tuple[tuple[str, dict[str, Any]], ...],
        answer: str,
        structured: Any,
    ) -> None:
        self._registry = registry
        self._calls = calls
        self._answer = answer
        self._structured = structured

    def run(self, task: str) -> AgentResult:
        made: list[ToolCall] = []
        for name, params in self._calls:
            call = ToolCall(tool_name=name, parameters=dict(params))
            self._registry.execute(call)  # real tool: real bytes, or a real refusal
            made.append(call)
        return AgentResult(
            answer=self._answer,
            success=True,
            trace=AgentTrace(tool_calls=made, total_iterations=len(made) or 1),
            final_context={},
            structured_output=self._structured,
        )


def _dispatch(
    state: str,
    *,
    workspace_root: Path,
    plan_dir: Path | None,
    payload: dict[str, Any],
    calls: tuple[tuple[str, dict[str, Any]], ...] = (),
    answer: str | None = None,
) -> AgentResult:
    """Run one dispatch through the REAL default worker factory.

    The factory builds the real tool registry, the real ``PlanMemory`` and the
    real prompt; only the LLM is replaced, by :class:`_ScriptedAgent`.
    """
    spec = get_role_spec(state)
    structured = spec.output_schema(**payload)
    factory = build_default_worker_factory(
        Workspace(workspace_root),
        agent_builder=lambda spec_, registry, config: _ScriptedAgent(
            registry,
            calls,
            answer if answer is not None else json.dumps(payload),
            structured,
        ),
    )
    return factory(
        _role_request(state, plan_dir=plan_dir, workspace_root=workspace_root)
    )


def _explore_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        ContextKeys.FINDINGS_COUNT: 3,
        ContextKeys.NEEDS_EXPLORE: False,
        "message": "three findings indexed",
    }
    payload.update(overrides)
    return payload


class TestFindingsCountComesFromDisk:
    """The EXPLORE gate reads the filesystem, never the worker's integer.

    Review C1 reproduced the fail-open: a dispatch that made one read call,
    answered in prose and wrote **0 bytes** returned
    ``structured_output.findings_count == 3``, which satisfied the EXPLORE ->
    PLAN hard gate.  Review C3 showed the number is not even reportable -- the
    explorer is forbidden to touch the ``findings.md`` index its own extraction
    instructions defined the count against.
    """

    def test_a_claimed_count_cannot_open_the_gate_over_an_empty_findings_dir(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        result = _dispatch(
            HarnessStates.EXPLORE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload=_explore_payload(),
            calls=(("read_plan_file", {"path": "state.md"}),),
        )

        assert result.final_context.get(ContextKeys.FINDINGS_COUNT, 0) == 0
        assert result.success is False

    def test_the_gate_opens_on_three_real_findings_files(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """The control: with the files really on disk, the count is really 3."""
        (plan_dir / ArtifactNames.FINDINGS_DIR).mkdir(parents=True, exist_ok=True)
        for topic in ("a", "b"):
            (plan_dir / ArtifactNames.FINDINGS_DIR / f"{topic}.md").write_text("x\n")

        result = _dispatch(
            HarnessStates.EXPLORE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload=_explore_payload(),
            calls=(
                (
                    "write_plan_file",
                    {
                        "path": f"{ArtifactNames.FINDINGS_DIR}/c.md",
                        "content": "found\n",
                    },
                ),
            ),
        )

        assert result.final_context[ContextKeys.FINDINGS_COUNT] == 3
        assert result.success is True

    def test_a_claim_that_contradicts_disk_loses_in_both_directions(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """Over-claim and under-claim both resolve to what the filesystem says."""
        (plan_dir / ArtifactNames.FINDINGS_DIR).mkdir(parents=True, exist_ok=True)
        (plan_dir / ArtifactNames.FINDINGS_DIR / "a.md").write_text("x\n")
        write = (
            "write_plan_file",
            {"path": f"{ArtifactNames.FINDINGS_DIR}/b.md", "content": "found\n"},
        )

        over = _dispatch(
            HarnessStates.EXPLORE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload=_explore_payload(**{ContextKeys.FINDINGS_COUNT: 99}),
            calls=(write,),
        )
        under = _dispatch(
            HarnessStates.EXPLORE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload=_explore_payload(**{ContextKeys.FINDINGS_COUNT: 0}),
            calls=(write,),
        )

        assert over.final_context[ContextKeys.FINDINGS_COUNT] == 2
        assert under.final_context[ContextKeys.FINDINGS_COUNT] == 2

    def test_the_structured_payload_is_reconciled_too_not_just_final_context(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """``harness._apply_role_result`` reads BOTH channels; both must agree.

        Relying on ``final_context`` winning the dict merge would make the fix
        an ordering accident.
        """
        result = _dispatch(
            HarnessStates.EXPLORE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload=_explore_payload(),
            calls=(
                (
                    "write_plan_file",
                    {"path": f"{ArtifactNames.FINDINGS_DIR}/a.md", "content": "f\n"},
                ),
            ),
        )

        assert getattr(result.structured_output, ContextKeys.FINDINGS_COUNT) == 1

    def test_an_empty_findings_file_is_not_a_finding(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """Otherwise ``touch findings/{a,b,c}.md`` opens a HARD gate."""
        (plan_dir / ArtifactNames.FINDINGS_DIR).mkdir(parents=True, exist_ok=True)
        for topic in ("a", "b", "c"):
            (plan_dir / ArtifactNames.FINDINGS_DIR / f"{topic}.md").write_text("")

        result = _dispatch(
            HarnessStates.EXPLORE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload=_explore_payload(),
            calls=(("list_plan_dir", {"path": ArtifactNames.FINDINGS_DIR}),),
        )

        assert result.final_context.get(ContextKeys.FINDINGS_COUNT, 0) == 0

    def test_without_a_plan_directory_the_claim_is_dropped_not_believed(
        self, tmp_path: Path
    ) -> None:
        """Nothing to count against means "not verified", never "assume 3"."""
        result = _dispatch(
            HarnessStates.EXPLORE,
            workspace_root=tmp_path / "ws",
            plan_dir=None,
            payload=_explore_payload(),
        )

        assert ContextKeys.FINDINGS_COUNT not in result.final_context

    def test_the_rules_define_the_count_against_files_the_explorer_may_write(
        self,
    ) -> None:
        """Review C3: the old text defined it against the index it may not touch."""
        explore = get_rules(HarnessStates.EXPLORE)
        explorer = ROLE_BY_STATE[HarnessStates.EXPLORE]

        # MOVED at step 11 (D-041): the definition used to live in this state's
        # `extraction_instructions`, which no longer exists -- a non-empty value
        # there costs one LLM call per turn.  `gate_summary` is where the count
        # is defined now, and it is the string the role prompt actually carries,
        # so the C3 assertion is stronger here than it was before.
        assert ArtifactNames.FINDINGS_INDEX not in explore.gate_summary
        assert f"{ArtifactNames.FINDINGS_DIR}/" in explore.gate_summary
        assert ArtifactNames.FINDINGS_DIR in artifacts_writable_by(explorer)


class TestWriteClaimsNeedToolEvidence:
    """A completion claim with no verified write is not a success (D-016).

    The harness has carried the instruction "never state that you wrote a file
    unless a write tool reported success" since D-013.  It was in the prompt on
    all 5 fabricating runs.  Wording does not carry this; the trace does.
    """

    def test_the_measured_fabricated_execute_reply_is_rejected(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        result = _dispatch(
            HarnessStates.EXECUTE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload={"summary": "added retry with backoff", "message": "done"},
            calls=(("read_file", {"path": "uploader.py"}),),
            answer=_FABRICATED_EXECUTE_ANSWER,
        )

        assert result.success is False

    def test_the_same_payload_with_a_real_write_is_accepted(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """The control: only the trace differs between this and the case above."""
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)

        result = _dispatch(
            HarnessStates.EXECUTE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload={"summary": "added retry with backoff", "message": "done"},
            calls=(
                ("read_file", {"path": "uploader.py"}),
                ("write_file", {"path": "uploader.py", "content": "retry\n"}),
            ),
            answer=_FABRICATED_EXECUTE_ANSWER,
        )

        assert result.success is True
        assert (tmp_path / "ws" / "uploader.py").read_text() == "retry\n"

    def test_a_write_call_that_left_no_bytes_is_not_evidence(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """Tool-name presence is the MINIMUM bar; bytes are the real one.

        ``changelog.md`` is an EXECUTOR-owned artifact, but ``plan.md`` is not,
        so the real ``PlanMemory`` refuses this call and nothing lands.
        """
        result = _dispatch(
            HarnessStates.EXECUTE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload={"summary": "wrote the plan", "message": "done"},
            calls=(
                (
                    "write_plan_file",
                    {"path": ArtifactNames.PLAN, "content": "# plan\n"},
                ),
            ),
        )

        assert result.success is False
        assert not (plan_dir / ArtifactNames.PLAN).exists()

    def test_deleting_a_file_is_not_write_evidence(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ws" / "stale.py").write_text("gone\n")

        result = _dispatch(
            HarnessStates.EXECUTE,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload={"summary": "removed the stale module", "message": "done"},
            calls=(("delete_file", {"path": "stale.py"}),),
        )

        assert result.success is False

    def test_a_role_holding_no_write_tool_is_not_asked_for_write_evidence(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """REFLECT's verifier writes nothing by design -- the driver merges its
        reply.  A check that failed it would be measuring the wrong thing."""
        spec = get_role_spec(HarnessStates.REFLECT)
        request = _role_request(HarnessStates.REFLECT, plan_dir=plan_dir)
        assert not set(held_tools(request, spec)) & set(WRITE_TOOLS + PLAN_WRITE_TOOLS)

        result = _dispatch(
            HarnessStates.REFLECT,
            workspace_root=tmp_path / "ws",
            plan_dir=plan_dir,
            payload={
                ContextKeys.ALL_CRITERIA_PASS: True,
                ContextKeys.NEEDS_PIVOT: False,
                ContextKeys.COMPLETION_FIX: False,
                ContextKeys.NEEDS_EXPLORE: False,
                ContextKeys.CRITERIA_PASS_COUNT: 2,
                ContextKeys.CRITERIA_TOTAL: 2,
                "message": "all criteria pass",
            },
            calls=(("read_plan_file", {"path": "plan.md"}),),
        )

        assert result.success is True

    def test_the_observation_records_what_the_check_caught(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """The live bench reads these; an unrecorded catch cannot be counted."""
        seen: list[dict[str, Any]] = []
        spec = get_role_spec(HarnessStates.EXPLORE)
        payload = _explore_payload()
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry, (), "no tools called", spec.output_schema(**payload)
            ),
        )

        factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert seen[-1]["write_evidence"] == 0
        assert seen[-1]["claimed_findings_count"] == 3
        assert seen[-1]["derived_findings_count"] == 0
        assert seen[-1]["failure_reason"] == "unverified-write"


# ---------------------------------------------------------------------------
# `/plan/*` is not a workspace path (review W4)
# ---------------------------------------------------------------------------


class TestWorkspaceRefusesPlanPaths:
    """The ``plan`` sentinel belongs to ``PlanMemory``, not to ``Workspace``.

    With ``plan`` in the workspace's own sentinel set, an EXECUTE role emitting
    ``/plan/findings/x.md`` silently wrote a protocol artifact into the user's
    SOURCE TREE -- confined, but into the wrong root, and past every ownership
    check.  Before the D-006 repair that shape raised.
    """

    @pytest.mark.parametrize(
        "path", ["/plan/x.py", "/plan/findings/x.md", "/plan/state.md"]
    )
    def test_a_plan_path_handed_to_the_workspace_is_refused(
        self, tmp_path: Path, path: str
    ) -> None:
        ws = Workspace(tmp_path / "ws")

        with pytest.raises(HarnessConfinementError):
            ws.resolve(path)
        assert not (ws.root / "x.py").exists()

    def test_the_workspace_sentinel_still_repairs(self, tmp_path: Path) -> None:
        """The control: narrowing the set did not disable the measured repair."""
        ws = Workspace(tmp_path / "ws")

        assert ws.resolve("/workspace/uploader.py") == ws.root / "uploader.py"

    @pytest.mark.parametrize("path", ["/plan/x.md", "/plan/findings/x.md"])
    def test_plan_memory_still_repairs_the_same_shape(
        self, tmp_path: Path, path: str
    ) -> None:
        """The other direction of the same pin: the sentinel moved, not vanished."""
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.EXPLORER)

        assert memory.locate(path).startswith("plan-x/")

    @pytest.mark.parametrize("bare", ["/plan", "/workspace"])
    def test_plan_memory_maps_a_bare_sentinel_to_the_plan_directory(
        self, tmp_path: Path, bare: str
    ) -> None:
        """The bare-sentinel twin of Defect A on the plan-memory call site.

        `_PLAN_SENTINELS` carries both `plan` and `workspace`, so the ONE
        helper fix (D-004) closes bare `/plan` AND bare `/workspace` here just
        as it closes bare `/workspace` in `Workspace.resolve`.  A bare sentinel
        addresses the plan directory itself, not a refusal.
        """
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.EXPLORER)

        assert memory.locate(bare).startswith("plan-x")
        assert memory.locate_path(bare) == memory.plan_dir


# ---------------------------------------------------------------------------
# The output line asks for the shape core actually delivers (review W6/W9)
# ---------------------------------------------------------------------------


class TestOutputLineAsksForABareObject:
    """D-004 supplied the ``message`` key, which made the old warning FALSE.

    ``_output_line`` still told every role that "a reply that is nothing but a
    JSON object is discarded before this protocol sees it".  Since step 4 that
    is not true, and the ``response_format`` repair turn structurally cannot
    emit prose first.
    """

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_false_discard_claim_is_gone(self, state: str, plan_dir: Path) -> None:
        prompt = build_role_prompt(
            _role_request(state, plan_dir=plan_dir), spec=get_role_spec(state)
        )

        assert "is discarded before this protocol sees it" not in prompt
        assert "The sentence is required" not in prompt

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_prompt_asks_for_exactly_one_json_object(
        self, state: str, plan_dir: Path
    ) -> None:
        prompt = build_role_prompt(
            _role_request(state, plan_dir=plan_dir), spec=get_role_spec(state)
        )

        assert "exactly ONE JSON object" in prompt
        assert "Do not show drafts" in prompt

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_a_bare_object_reply_survives_cores_real_parser(self, state: str) -> None:
        """Why the claim is false, asserted against core rather than believed."""
        spec = get_role_spec(state)
        payload = {name: "x" for name in spec.output_schema.model_fields}
        payload["message"] = "the role's own sentence"

        parsed = _parse_as_core_would(json.dumps(payload))

        assert parsed.message == "the role's own sentence"
        assert parsed.message != _GENERIC_FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# Standing policy vs this dispatch's task (decisions.md D-021)
# ---------------------------------------------------------------------------


class _PolicyAgent(_ScriptedAgent):
    """A scripted agent that CAN hold a system policy, as ``native_fc`` can.

    Interface contract (test double, 2 call sites below):
        - Declares ``system_policy`` so the factory's duck-typed check finds it,
          exactly as :class:`NativeFunctionCallingReactAgent` does.
        - Records every task string it is run with, so a test can assert WHICH
          half of the prompt reached the user turn.
        - Everything else -- the real registry, the real tool calls, the real
          bytes -- is inherited unchanged from :class:`_ScriptedAgent`.
    """

    system_policy: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tasks: list[str] = []

    def run(self, task: str) -> AgentResult:
        self.tasks.append(task)
        return super().run(task)


class TestStandingPolicyIsSplitFromTheTask:
    """WHERE the standing half is delivered was measured to decide behaviour.

    Live ``ollama_chat/qwen3.5:4b``, EXECUTE role, n=5 per arm, bytes stat'd on
    disk: the whole prompt in the user turn wrote 0/5; the identical text with
    every standing block moved into the system message wrote 4/5 (workspace) and
    4/5 (plan directory).  These tests pin the SPLIT -- that it loses nothing,
    that each half carries what it should, and that an agent which cannot hold a
    policy still receives the complete prompt.
    """

    @staticmethod
    def _blocks(text: str) -> list[str]:
        return [block for block in text.split("\n\n") if block]

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_two_halves_partition_the_whole_prompt(
        self, state: str, plan_dir: Path
    ) -> None:
        """Nothing is dropped and nothing is said twice."""
        spec = get_role_spec(state)
        request = _role_request(
            state, plan_dir=plan_dir, context={ContextKeys.TOTAL_STEPS: 4}
        )

        whole = self._blocks(build_role_prompt(request, spec))
        standing = self._blocks(build_role_system_prompt(request, spec))
        task = self._blocks(build_role_task_prompt(request, spec))

        assert sorted(standing + task) == sorted(whole)
        assert not set(standing) & set(task)

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_system_half_carries_every_standing_instruction(
        self, state: str, plan_dir: Path
    ) -> None:
        spec = get_role_spec(state)
        system = build_role_system_prompt(_role_request(state, plan_dir=plan_dir), spec)

        assert system.startswith(f"You are the {ROLE_BY_STATE[state]}")
        for marker in (
            "EXIT GATE:",
            "RULES:",
            "TOOLS:",
            "HOW TO FINISH:",
            "exactly ONE JSON object",
            "YOU MAY WRITE",
        ):
            assert marker in system

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_task_half_carries_only_what_differs_per_dispatch(
        self, state: str, plan_dir: Path
    ) -> None:
        spec = get_role_spec(state)
        task = build_role_task_prompt(_role_request(state, plan_dir=plan_dir), spec)

        assert task.startswith("GOAL: ")
        assert "POSITION: iteration 1" in task
        for marker in ("RULES:", "TOOLS:", "HOW TO FINISH:", "YOU MAY WRITE"):
            assert marker not in task

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_context_snapshot_travels_with_the_task(
        self, state: str, plan_dir: Path
    ) -> None:
        """The snapshot is this dispatch's state, so it is not standing policy."""
        request = _role_request(
            state, plan_dir=plan_dir, context={ContextKeys.TOTAL_STEPS: 4}
        )
        spec = get_role_spec(state)

        assert "CURRENT STATE:" in build_role_task_prompt(request, spec)
        assert "CURRENT STATE:" not in build_role_system_prompt(request, spec)

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_system_half_still_names_exactly_the_held_tools(
        self, state: str, plan_dir: Path, workspace: Path
    ) -> None:
        """Review C2's invariant moved with the block; it did not stay behind."""
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=plan_dir, workspace_root=workspace)

        system = build_role_system_prompt(request, spec)

        assert TestRolePromptNamesHeldTools._named_tools(system) == held_tools(
            request, spec
        )

    def test_an_agent_that_can_hold_a_policy_gets_the_split_prompt(
        self, tmp_path: Path, plan_dir: Path, workspace: Path
    ) -> None:
        """Through the REAL factory: the standing half never reaches the task."""
        state = HarnessStates.EXECUTE
        spec = get_role_spec(state)
        built: list[_PolicyAgent] = []

        def builder(spec_: Any, registry: Any, config: Any) -> _PolicyAgent:
            agent = _PolicyAgent(
                registry, (), "done", spec.output_schema(summary="s", message="m")
            )
            built.append(agent)
            return agent

        factory = build_default_worker_factory(
            Workspace(workspace), agent_builder=builder
        )
        request = _role_request(state, plan_dir=plan_dir, workspace_root=workspace)
        factory(request)

        agent = built[0]
        assert agent.system_policy == build_role_system_prompt(request, spec)
        assert agent.tasks == [build_role_task_prompt(request, spec)]

    def test_an_agent_that_cannot_hold_a_policy_still_gets_everything(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The fallback is no-loss: `create_agent("react")` is unchanged by this."""
        state = HarnessStates.EXECUTE
        spec = get_role_spec(state)
        seen: list[str] = []

        class _NoPolicyAgent(_ScriptedAgent):
            def run(self_inner, task: str) -> AgentResult:
                seen.append(task)
                return super().run(task)

        factory = build_default_worker_factory(
            Workspace(workspace),
            agent_builder=lambda spec_, registry, config: _NoPolicyAgent(
                registry, (), "done", spec.output_schema(summary="s", message="m")
            ),
        )
        request = _role_request(state, plan_dir=plan_dir, workspace_root=workspace)
        factory(request)

        assert seen == [build_role_prompt(request, spec)]
        assert "RULES:" in seen[0]


# ---------------------------------------------------------------------------
# The write tool's RESULT is the feedback channel (D-026)
# ---------------------------------------------------------------------------


def _plan_registry(plan_dir: Path, workspace_root: Path, role: str = Role.EXPLORER):
    """A real registry spanning both roots, exactly as a dispatch builds one."""
    workspace_root.mkdir(parents=True, exist_ok=True)
    memory = PlanMemory(plan_dir, role=role)
    registry = build_workspace_tools(Workspace(workspace_root))
    build_plan_tools(memory, registry=registry)
    return registry, memory


def _call(registry: Any, name: str, **params: Any) -> Any:
    """Execute one tool the way the agent loop does, returning the ToolResult."""
    return registry.execute(ToolCall(tool_name=name, parameters=params))


class TestWriteResultsReportGateState:
    """A repeat write must be visibly different from a new one (D-026 part A).

    The step-5 spike measured the explorer calling ``write_plan_file`` with the
    SAME path 3-11 times in one dispatch and never naming a second topic: 25
    redundant repeat-writes across 10 runs, 3 distinct findings files in 0 of
    them.  The tool answered every one of those calls with a bare
    ``wrote <path>``, so from inside the loop a repeat was indistinguishable
    from progress.  These tests pin the observation, not an instruction.
    """

    def test_a_repeat_write_reads_differently_from_a_new_one(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)
        findings = ArtifactNames.FINDINGS_DIR

        first = _call(
            registry, "write_plan_file", path=f"{findings}/a.md", content="one\n"
        )
        repeat = _call(
            registry, "write_plan_file", path=f"{findings}/a.md", content="one\n"
        )

        assert first.success and repeat.success
        assert first.result != repeat.result
        assert "NEW file" in first.result
        assert "OVERWROTE an existing file" in repeat.result

    def test_the_repeat_states_that_the_count_did_not_move(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The exact signal the model lacked: a rewrite is not a second topic."""
        registry, _ = _plan_registry(plan_dir, workspace)
        findings = ArtifactNames.FINDINGS_DIR

        _call(registry, "write_plan_file", path=f"{findings}/a.md", content="one\n")
        repeat = _call(
            registry, "write_plan_file", path=f"{findings}/a.md", content="again\n"
        )
        second = _call(
            registry, "write_plan_file", path=f"{findings}/b.md", content="two\n"
        )

        assert f"{findings}/ still holds 1 of the" in repeat.result
        assert f"{findings}/ now holds 2 of the" in second.result

    def test_the_reported_count_is_the_count_the_gate_reads(
        self, tmp_path: Path, plan_dir: Path, workspace: Path
    ) -> None:
        """One derivation, two readers.  A second count could disagree (D-015)."""
        registry, memory = _plan_registry(plan_dir, workspace)
        findings = ArtifactNames.FINDINGS_DIR
        for topic in ("a", "b"):
            _call(
                registry,
                "write_plan_file",
                path=f"{findings}/{topic}.md",
                content="x\n",
            )

        told = _call(
            registry, "write_plan_file", path=f"{findings}/c.md", content="x\n"
        ).result
        gate = _dispatch(
            HarnessStates.EXPLORE,
            workspace_root=workspace,
            plan_dir=plan_dir,
            payload=_explore_payload(),
            calls=(
                ("write_plan_file", {"path": f"{findings}/c.md", "content": "x\n"}),
            ),
        )

        assert count_gate_files(memory, findings) == 3
        assert f"holds 3 of the {Defaults.FINDINGS_THRESHOLD} " in told
        assert gate.final_context[ContextKeys.FINDINGS_COUNT] == 3

    def test_the_threshold_is_the_protocols_own_number(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """A literal 3 here would drift the moment the gate's threshold moved."""
        registry, _ = _plan_registry(plan_dir, workspace)

        told = _call(
            registry,
            "write_plan_file",
            path=f"{ArtifactNames.FINDINGS_DIR}/a.md",
            content="x\n",
        ).result

        assert (
            ContextKeys.FINDINGS_COUNT,
            (ArtifactNames.FINDINGS_DIR, Defaults.FINDINGS_THRESHOLD),
        ) in DISK_DERIVED_COUNTS.items()
        assert f"of the {Defaults.FINDINGS_THRESHOLD} distinct" in told

    def test_an_empty_write_does_not_move_the_reported_count(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The message counts what the gate counts: NON-empty files only."""
        registry, _ = _plan_registry(plan_dir, workspace)

        told = _call(
            registry,
            "write_plan_file",
            path=f"{ArtifactNames.FINDINGS_DIR}/a.md",
            content="   \n",
        ).result

        assert "holds 0 of the" in told

    def test_an_append_that_creates_the_file_counts_it(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)
        findings = ArtifactNames.FINDINGS_DIR

        created = _call(
            registry, "append_plan_file", path=f"{findings}/a.md", content="x\n"
        )
        extended = _call(
            registry, "append_plan_file", path=f"{findings}/a.md", content="y\n"
        )

        assert "NEW file" in created.result
        assert f"{findings}/ now holds 1 of the" in created.result
        assert "extended an existing file" in extended.result
        assert f"{findings}/ still holds 1 of the" in extended.result

    def test_a_non_counted_artifact_gets_no_gate_clause(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """Only a key in ``DISK_DERIVED_COUNTS`` has a count worth reporting."""
        registry, _ = _plan_registry(plan_dir, workspace, role=Role.PLAN_WRITER)

        told = _call(
            registry, "write_plan_file", path=ArtifactNames.PLAN, content="# Plan\n"
        ).result

        assert "NEW file" in told
        assert "exit gate requires" not in told

    def test_workspace_writes_report_novelty_and_no_gate_clause(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)

        first = _call(registry, "write_file", path="uploader.py", content="a = 1\n")
        repeat = _call(registry, "write_file", path="uploader.py", content="a = 2\n")

        assert "NEW file" in first.result
        assert "OVERWROTE an existing file" in repeat.result
        assert "exit gate requires" not in repeat.result


class TestWrongRootFailuresNameTheRightTool:
    """An 18% wrong-root call rate is a teachable failure (D-026 part B).

    The step-5 spike recorded **zero** hallucinated tool NAMES in 298 calls and
    54 calls (18%) aiming a workspace tool at a plan artifact or the reverse.
    The model knows the tools; it mis-picks the ROOT.  The refusal stays a
    refusal -- only the message becomes corrective.
    """

    def test_a_workspace_read_of_a_protocol_artifact_names_the_plan_tool(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(registry, "read_file", path=ArtifactNames.STATE)

        assert failed.success is False
        assert "plan directory" in failed.error
        assert "`read_plan_file`" in failed.error

    def test_a_plan_write_of_a_source_file_names_the_workspace_tool(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(registry, "write_plan_file", path="uploader.py", content="x\n")

        assert failed.success is False
        assert "not a protocol artifact" in failed.error
        assert "`write_file`" in failed.error
        assert not (plan_dir / "uploader.py").exists()
        assert not (workspace / "uploader.py").exists()

    def test_the_hint_never_re_routes_the_call(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """Naming the tool is advice, not a retry: no bytes land anywhere."""
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(
            registry, "write_file", path="/plan/findings/x.md", content="loot\n"
        )

        assert failed.success is False
        assert "`write_plan_file`" in failed.error
        assert list(workspace.rglob("*.md")) == []
        assert not (plan_dir / ArtifactNames.FINDINGS_DIR / "x.md").exists()

    def test_the_confinement_refusal_keeps_its_class_and_attributes(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """Enriching a message must not downgrade the refusal it enriches."""
        registry, _ = _plan_registry(plan_dir, workspace)
        write_file = registry.get(WorkspaceTools.WRITE_FILE).execute_fn

        with pytest.raises(HarnessConfinementError) as caught:
            write_file(path="/plan/findings/x.md", content="loot\n")

        assert caught.value.path == "/plan/findings/x.md"
        assert caught.value.root == str(Workspace(workspace).root)
        assert "`write_plan_file`" in str(caught.value)

    def test_the_ownership_refusal_keeps_its_class_and_attributes(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)
        write_plan_file = registry.get(PlanTools.WRITE_PLAN_FILE).execute_fn

        with pytest.raises(HarnessOwnershipError) as caught:
            write_plan_file(path="uploader.py", content="x\n")

        assert caught.value.role == Role.EXPLORER
        assert "`write_file`" in str(caught.value)

    def test_an_owned_artifact_the_role_may_not_write_is_refused_unchanged(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """``plan.md`` IS a protocol artifact: the answer is "not yours", full stop."""
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(
            registry, "write_plan_file", path=ArtifactNames.PLAN, content="x"
        )

        assert failed.success is False
        assert "may not write" in failed.error
        assert "`write_file`" not in failed.error
        assert not (plan_dir / ArtifactNames.PLAN).exists()

    def test_a_genuinely_missing_workspace_file_gets_no_hint(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The control: a hint on every ENOENT would be noise, not a signal."""
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(registry, "read_file", path="src/missing.py")

        assert failed.success is False
        assert "plan directory" not in failed.error

    def test_a_missing_findings_file_in_the_right_root_gets_no_hint(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(
            registry, "read_plan_file", path=f"{ArtifactNames.FINDINGS_DIR}/nope.md"
        )

        assert failed.success is False
        assert "workspace" not in failed.error

    def test_every_hinted_tool_names_a_tool_that_exists(self) -> None:
        """A hint naming a non-existent tool would be worse than silence."""
        ws_names = {v for k, v in vars(WorkspaceTools).items() if not k.startswith("_")}
        plan_names = {v for k, v in vars(PlanTools).items() if not k.startswith("_")}

        for source, counterpart in _COUNTERPART_TOOL.items():
            assert source in ws_names | plan_names
            assert counterpart in ws_names | plan_names
            # The pair always crosses the root boundary -- that is the point.
            assert (source in plan_names) != (counterpart in plan_names)


class TestAnOwnedDirectoryReportsEmptyNotMissing:
    """``findings/`` is an artifact name, so "not created yet" means EMPTY.

    Step-22 attempt 1 measured the chain directly: the routing hint correctly
    moved the explorer off ``list_dir("findings/")`` and onto
    ``list_plan_dir("findings/")``, which then answered ENOENT six times in a
    row, and that run wrote nothing at all.  ``gate_files`` has always answered
    "none, not unknown" for the same directory; this makes the tool agree with
    the derivation the gate reads.
    """

    def test_a_missing_owned_directory_lists_as_empty_with_its_count(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        bare = tmp_path / "plans" / "plan-2026-07-21T000000-bare"
        registry, _ = _plan_registry(bare, workspace)

        listed = _call(registry, "list_plan_dir", path=ArtifactNames.FINDINGS_DIR)

        assert listed.success is True
        assert "(empty)" in listed.result
        assert f"{ArtifactNames.FINDINGS_DIR}/ holds 0 of the" in listed.result

    def test_a_missing_path_that_is_not_an_artifact_still_fails(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        """The control: this is not "answer (empty) for anything missing"."""
        bare = tmp_path / "plans" / "plan-2026-07-21T000000-bare"
        registry, _ = _plan_registry(bare, workspace)

        assert _call(registry, "list_plan_dir", path="src").success is False
        assert (
            _call(registry, "read_plan_file", path=ArtifactNames.PLAN).success is False
        )

    def test_a_listing_names_what_the_gate_counts(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, memory = _plan_registry(plan_dir, workspace)
        findings = ArtifactNames.FINDINGS_DIR
        (plan_dir / findings).mkdir(parents=True, exist_ok=True)
        (plan_dir / findings / "a.md").write_text("x\n")
        (plan_dir / findings / "empty.md").write_text("")

        listed = _call(registry, "list_plan_dir", path=findings).result

        assert gate_files(memory, findings) == ("a.md",)
        assert f"{findings}/ holds 1 of the" in listed
        assert listed.rstrip().endswith("requires: a.md.")

    def test_the_write_result_names_the_topics_already_covered(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """A bare count leaves the model to remember which topics it wrote."""
        registry, _ = _plan_registry(plan_dir, workspace)
        findings = ArtifactNames.FINDINGS_DIR
        _call(registry, "write_plan_file", path=f"{findings}/a.md", content="x\n")

        second = _call(
            registry, "write_plan_file", path=f"{findings}/b.md", content="y\n"
        ).result

        assert second.rstrip().endswith("requires: a.md, b.md.")


# ---------------------------------------------------------------------------
# What earlier dispatches already covered (D-028, the re-dispatch half)
# ---------------------------------------------------------------------------


class TestCoverageLineTellsARedispatchWhatExists:
    """Bounded re-dispatch sends N explorers; this stops them writing N copies.

    The step-5 spike measured one dispatch writing ``findings/uploader.md`` up
    to eleven times and never naming a second topic.  Re-dispatching alone would
    reproduce that ACROSS dispatches -- each one starts with no memory of the
    last -- so every dispatch after the first is told, by NAME, what is already
    on disk.  The names come from ``gate_files``: the ONE derivation the gate
    value and the write tools' own feedback also read (D-015, D-027).
    """

    @staticmethod
    def _payload(spec: Any) -> Any:
        """A schema-valid payload for ANY role, derived from its own schema.

        Interface contract (4 call sites in this class): every field gets a
        falsy value of its declared type, so a role whose schema gains a field
        does not silently turn these tests into construction errors.
        """
        return spec.output_schema(
            **{
                name: (
                    "s"
                    if field.annotation is str
                    else (False if field.annotation is bool else 0)
                )
                for name, field in spec.output_schema.model_fields.items()
            }
        )

    def _task_prompt(
        self, request: RoleRequest, plan_dir: Path, workspace: Path
    ) -> str:
        """The task text a REAL dispatch would send, through the real factory."""
        built: list[_PolicyAgent] = []
        spec = get_role_spec(request.state)

        def builder(spec_: Any, registry: Any, config: Any) -> _PolicyAgent:
            agent = _PolicyAgent(registry, (), "done", self._payload(spec))
            built.append(agent)
            return agent

        factory = build_default_worker_factory(
            Workspace(workspace), agent_builder=builder
        )
        factory(request)
        return built[0].tasks[0]

    def test_the_first_dispatch_is_told_nothing_extra(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """An empty ``findings/`` adds NOTHING: dispatch 1's prompt is unchanged.

        This is the no-regression property -- criterion (a) and every first
        dispatch receive byte-for-byte the prompt they received before D-028.
        """
        request = _role_request(
            HarnessStates.EXPLORE, plan_dir=plan_dir, workspace_root=workspace
        )
        spec = get_role_spec(HarnessStates.EXPLORE)

        assert self._task_prompt(request, plan_dir, workspace) == (
            build_role_task_prompt(request, spec)
        )

    def test_a_later_dispatch_is_told_what_is_already_covered(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        findings = ArtifactNames.FINDINGS_DIR
        (plan_dir / findings).mkdir(parents=True, exist_ok=True)
        (plan_dir / findings / "uploader.md").write_text("notes\n")
        (plan_dir / findings / "config.md").write_text("notes\n")
        request = _role_request(
            HarnessStates.EXPLORE, plan_dir=plan_dir, workspace_root=workspace
        )

        task = self._task_prompt(request, plan_dir, workspace)

        assert "ALREADY ON DISK" in task
        assert "config.md, uploader.md" in task
        assert f"holds 2 of the {Defaults.FINDINGS_THRESHOLD} distinct" in task
        assert "Write ONE file this list does not have" in task

    def test_it_counts_exactly_what_the_gate_counts(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """An empty file is not a finding -- for the gate OR for this line.

        A second count here would be a gate and a progress report that can
        disagree, which is the fail-open defect D-015 closed.
        """
        findings = ArtifactNames.FINDINGS_DIR
        (plan_dir / findings).mkdir(parents=True, exist_ok=True)
        (plan_dir / findings / "real.md").write_text("notes\n")
        (plan_dir / findings / "blank.md").write_text("   \n")
        (plan_dir / findings / "notes.txt").write_text("not markdown\n")
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)
        request = _role_request(
            HarnessStates.EXPLORE, plan_dir=plan_dir, workspace_root=workspace
        )

        task = self._task_prompt(request, plan_dir, workspace)

        assert gate_files(memory, findings) == ("real.md",)
        assert "holds 1 of the" in task
        assert "requires: real.md." in task
        assert "blank.md" not in task
        assert "notes.txt" not in task

    def test_a_role_that_owns_no_gate_count_is_untouched(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """EXECUTE's deliverable is a code edit; its prompt gains nothing here."""
        findings = ArtifactNames.FINDINGS_DIR
        (plan_dir / findings).mkdir(parents=True, exist_ok=True)
        (plan_dir / findings / "uploader.md").write_text("notes\n")
        request = _role_request(
            HarnessStates.EXECUTE, plan_dir=plan_dir, workspace_root=workspace
        )
        spec = get_role_spec(HarnessStates.EXECUTE)

        assert self._task_prompt(request, plan_dir, workspace) == (
            build_role_task_prompt(request, spec)
        )

    def test_no_plan_directory_means_no_coverage_line(self, workspace: Path) -> None:
        """Nothing to read means nothing to say -- never an invented count."""
        request = _role_request(
            HarnessStates.EXPLORE, plan_dir=None, workspace_root=workspace
        )
        spec = get_role_spec(HarnessStates.EXPLORE)

        assert self._task_prompt(request, Path("/nonexistent"), workspace) == (
            build_role_task_prompt(request, spec)
        )

    def test_it_travels_with_the_task_not_the_standing_policy(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """What is already covered differs per dispatch, so it is not policy."""
        findings = ArtifactNames.FINDINGS_DIR
        (plan_dir / findings).mkdir(parents=True, exist_ok=True)
        (plan_dir / findings / "uploader.md").write_text("notes\n")
        built: list[_PolicyAgent] = []
        spec = get_role_spec(HarnessStates.EXPLORE)

        def builder(spec_: Any, registry: Any, config: Any) -> _PolicyAgent:
            agent = _PolicyAgent(registry, (), "done", self._payload(spec))
            built.append(agent)
            return agent

        request = _role_request(
            HarnessStates.EXPLORE, plan_dir=plan_dir, workspace_root=workspace
        )
        build_default_worker_factory(Workspace(workspace), agent_builder=builder)(
            request
        )

        assert "ALREADY ON DISK" in built[0].tasks[0]
        assert "ALREADY ON DISK" not in (built[0].system_policy or "")
        assert built[0].system_policy == build_role_system_prompt(request, spec)

    def test_an_agent_without_a_policy_slot_gets_it_too(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The no-loss fallback keeps the whole prompt AND the coverage line."""
        findings = ArtifactNames.FINDINGS_DIR
        (plan_dir / findings).mkdir(parents=True, exist_ok=True)
        (plan_dir / findings / "uploader.md").write_text("notes\n")
        seen: list[str] = []
        spec = get_role_spec(HarnessStates.EXPLORE)

        class _NoPolicy(_ScriptedAgent):
            def run(self_inner, task: str) -> AgentResult:
                seen.append(task)
                return super().run(task)

        request = _role_request(
            HarnessStates.EXPLORE, plan_dir=plan_dir, workspace_root=workspace
        )
        build_default_worker_factory(
            Workspace(workspace),
            agent_builder=lambda spec_, registry, config: _NoPolicy(
                registry, (), "done", self._payload(spec)
            ),
        )(request)

        assert seen[0].startswith(build_role_prompt(request, spec))
        assert "ALREADY ON DISK" in seen[0]


class TestDeriveDiskCountsIsTheOneDerivation:
    """``derive_disk_counts`` is what the factory AND the driver both call.

    Before D-032 the worker factory owned the only loop over
    ``DISK_DERIVED_COUNTS``.  The driver now derives the same numbers itself,
    after every attempted dispatch, so the loop had to become one function: two
    copies would be a gate value and a re-dispatch condition that can disagree,
    which is the fail-open shape D-015 closed.
    """

    def test_it_counts_what_the_gate_counts(self, plan_dir: Path) -> None:
        findings = ArtifactNames.FINDINGS_DIR
        (plan_dir / findings).mkdir(parents=True, exist_ok=True)
        (plan_dir / findings / "a.md").write_text("x\n")
        (plan_dir / findings / "b.md").write_text("y\n")
        (plan_dir / findings / "blank.md").write_text("   \n")
        (plan_dir / findings / "notes.txt").write_text("not markdown\n")
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        derived = derive_disk_counts(memory, [ContextKeys.FINDINGS_COUNT])

        assert derived == {ContextKeys.FINDINGS_COUNT: 2}
        assert derived[ContextKeys.FINDINGS_COUNT] == count_gate_files(memory, findings)

    def test_a_key_the_caller_does_not_own_is_not_counted_for_it(
        self, plan_dir: Path
    ) -> None:
        """A role (or state) that owns no disk-derived key gets ``{}``."""
        findings = ArtifactNames.FINDINGS_DIR
        (plan_dir / findings).mkdir(parents=True, exist_ok=True)
        (plan_dir / findings / "a.md").write_text("x\n")
        memory = PlanMemory(plan_dir, role=Role.EXECUTOR)

        assert derive_disk_counts(memory, []) == {}
        assert derive_disk_counts(memory, ["total_steps", "needs_explore"]) == {}

    def test_a_missing_directory_counts_zero_and_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """Zero is "none", never "unknown" -- and never an exception."""
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.EXPLORER)

        assert derive_disk_counts(memory, [ContextKeys.FINDINGS_COUNT]) == {
            ContextKeys.FINDINGS_COUNT: 0
        }

    def test_every_disk_derived_key_is_reachable_through_it(
        self, plan_dir: Path
    ) -> None:
        """Derived from the table, so a key added later is covered here too."""
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        derived = derive_disk_counts(memory, list(DISK_DERIVED_COUNTS))

        assert set(derived) == set(DISK_DERIVED_COUNTS)
        assert all(isinstance(value, int) for value in derived.values())


# ---------------------------------------------------------------------------
# Driver-assigned topic slugs (D-035)
# ---------------------------------------------------------------------------


class TestExploreTopicTable:
    """The decomposition itself: distinct, kebab-case, and enough of them.

    The harness used to let the model pick its own topics, which the source
    protocol never does -- ``agents/ip-orchestrator.md``'s EXPLORE dispatch
    assigns "a distinct kebab-case ``findings/{topic-slug}.md`` slug" per
    explorer.  These pin the properties the driver's assignment relies on.
    """

    def test_the_slugs_are_distinct(self) -> None:
        slugs = [topic.slug for topic in EXPLORE_TOPICS]

        assert len(set(slugs)) == len(slugs)

    @pytest.mark.parametrize("topic", EXPLORE_TOPICS, ids=lambda t: t.slug)
    def test_every_slug_is_kebab_case(self, topic: Any) -> None:
        """A slug becomes a FILE NAME; anything else is a confinement problem."""
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", topic.slug), topic.slug
        assert topic.label and topic.brief

    def test_there_are_at_least_as_many_topics_as_the_gate_requires(self) -> None:
        """Fewer topics than the threshold is a gate that can never open."""
        assert len(explore_topics()) >= Defaults.FINDINGS_THRESHOLD

    @pytest.mark.parametrize("threshold", [1, 3, 4, 7])
    def test_a_raised_threshold_is_extended_not_truncated(self, threshold: int) -> None:
        topics = explore_topics(threshold)
        slugs = [topic.slug for topic in topics]

        assert len(topics) >= threshold
        assert len(topics) >= len(EXPLORE_TOPICS)
        assert len(set(slugs)) == len(slugs)

    def test_the_purpose_sentence_is_derived_from_the_table(self) -> None:
        """The rendered text is PINNED, so the DRY refactor changed no prompt.

        ``_EXPLORE.purpose`` reads its coverage axes off ``EXPLORE_TOPICS``
        instead of restating them.  This is the control: the string a live run
        renders is byte-for-byte the hand-written one it replaced, so the live
        measurement of the assignment is not confounded by a prompt edit
        nobody intended.
        """
        assert get_rules(HarnessStates.EXPLORE).purpose == (
            "Build enough grounded context to plan: at least 3 indexed "
            "findings covering problem scope, affected files, and the existing "
            "patterns or constraints that any solution must respect."
        )

    @pytest.mark.parametrize("topic", EXPLORE_TOPICS, ids=lambda t: t.slug)
    def test_a_known_slug_resolves_to_its_own_entry(self, topic: Any) -> None:
        assert explore_topic(topic.slug) is topic

    def test_an_unknown_slug_renders_rather_than_raising(self) -> None:
        """A custom driver may assign a slug this module never heard of."""
        topic = explore_topic("some-other-thing")

        assert topic.slug == "some-other-thing"
        assert "some other thing" in topic.label


class TestTheAssignedTopicNamesOneTargetPath:
    """One dispatch, one topic, one named file (D-035).

    Four mechanisms tried to make ONE dispatch produce THREE findings files and
    all four missed the bar (decisions.md D-027, D-031).  This is the shape the
    source protocol actually specifies and the shape the measured ``:4b``
    envelope supports: write one named file.  The prompt half is pinned here;
    the driver half (which slug, and that assigning writes nothing) is in
    ``test_harness_agent.py``.
    """

    def _task(self, topic: str | None, plan_dir: Path, workspace: Path) -> str:
        request = _role_request(
            HarnessStates.EXPLORE,
            plan_dir=plan_dir,
            workspace_root=workspace,
            assigned_topic=topic,
        )
        return build_role_task_prompt(request, get_role_spec(HarnessStates.EXPLORE))

    @pytest.mark.parametrize("topic", EXPLORE_TOPICS, ids=lambda t: t.slug)
    def test_the_task_prompt_names_exactly_one_target_path(
        self, topic: Any, plan_dir: Path, workspace: Path
    ) -> None:
        """Exactly one ``findings/<slug>.md`` path, and it is the assigned one."""
        task = self._task(topic.slug, plan_dir, workspace)
        paths = set(re.findall(r"findings/[a-z0-9-]+\.md", task))

        assert paths == {f"{ArtifactNames.FINDINGS_DIR}/{topic.slug}.md"}
        assert topic.label in task

    def test_the_other_topics_are_not_named(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """Naming the whole decomposition would re-ask for three files."""
        task = self._task("problem-scope", plan_dir, workspace)

        for other in EXPLORE_TOPICS[1:]:
            assert other.slug not in task

    def test_the_assignment_is_in_the_task_half_not_the_standing_half(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """It differs between two dispatches of the same role, so it is TASK.

        D-021 split the prompt so the system message carries what is true every
        turn.  An assignment installed as standing policy would be re-sent
        unchanged on the next dispatch, telling the second explorer to write
        the file the first one already wrote.
        """
        request = _role_request(
            HarnessStates.EXPLORE,
            plan_dir=plan_dir,
            workspace_root=workspace,
            assigned_topic="affected-files",
        )
        spec = get_role_spec(HarnessStates.EXPLORE)

        assert "affected-files.md" in build_role_task_prompt(request, spec)
        assert "affected-files.md" not in build_role_system_prompt(request, spec)

    def test_no_assignment_leaves_the_prompt_byte_identical(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The no-regression control: every unassigned dispatch is untouched."""
        spec = get_role_spec(HarnessStates.EXPLORE)
        unassigned = _role_request(
            HarnessStates.EXPLORE, plan_dir=plan_dir, workspace_root=workspace
        )
        assigned = _role_request(
            HarnessStates.EXPLORE,
            plan_dir=plan_dir,
            workspace_root=workspace,
            assigned_topic="problem-scope",
        )

        base = build_role_task_prompt(unassigned, spec)
        with_topic = build_role_task_prompt(assigned, spec)

        assert "findings/" not in base
        assert with_topic != base
        # Nothing was REPLACED: the assignment is additive.
        for block in base.split("\n\n"):
            assert block in with_topic

    @pytest.mark.parametrize(
        "state",
        [
            HarnessStates.PLAN,
            HarnessStates.EXECUTE,
            HarnessStates.REFLECT,
            HarnessStates.CLOSE,
        ],
    )
    def test_a_state_the_driver_never_assigns_for_is_unchanged(
        self, state: str, plan_dir: Path, workspace: Path
    ) -> None:
        """Only EXPLORE is assigned a topic; the others must not drift."""
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=plan_dir, workspace_root=workspace)

        assert "YOUR TOPIC THIS DISPATCH" not in build_role_prompt(request, spec)

    def test_the_single_string_prompt_carries_the_assignment_too(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The D-021 fallback (an agent with no ``system_policy``) must see it.

        ``build_role_prompt`` is what ``create_agent("react")`` and every
        injected builder receive; an assignment only the split path carried
        would be silently absent for them.
        """
        request = _role_request(
            HarnessStates.EXPLORE,
            plan_dir=plan_dir,
            workspace_root=workspace,
            assigned_topic="constraints-and-patterns",
        )
        spec = get_role_spec(HarnessStates.EXPLORE)

        assert "findings/constraints-and-patterns.md" in build_role_prompt(
            request, spec
        )

    def test_an_unknown_slug_still_names_a_path(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        task = self._task("cache-invalidation", plan_dir, workspace)

        assert "findings/cache-invalidation.md" in task

    def test_the_assignment_reaches_a_real_dispatch_through_the_factory(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """End of the seam: the text a REAL worker sends, not a rebuilt one."""
        built: list[_PolicyAgent] = []
        spec = get_role_spec(HarnessStates.EXPLORE)

        def builder(spec_: Any, registry: Any, config: Any) -> _PolicyAgent:
            agent = _PolicyAgent(
                registry,
                (),
                "done",
                spec.output_schema(
                    **{
                        name: (
                            "s"
                            if field.annotation is str
                            else (False if field.annotation is bool else 0)
                        )
                        for name, field in spec.output_schema.model_fields.items()
                    }
                ),
            )
            built.append(agent)
            return agent

        factory = build_default_worker_factory(
            Workspace(workspace), agent_builder=builder
        )
        factory(
            _role_request(
                HarnessStates.EXPLORE,
                plan_dir=plan_dir,
                workspace_root=workspace,
                assigned_topic="affected-files",
            )
        )

        assert "findings/affected-files.md" in built[0].tasks[0]


class TestTheAssignedWriteTargetNamesPathToolAndRoot:
    """One EXECUTE dispatch, one driver-assigned workspace edit target (D-010).

    B0 (n=40, scripts/bench_data/l4-execute-write/B0) measured the defect this
    line exists for: the native executor issued a write tool 15/40 and left
    bytes 14/40, but only 2/40 landed the REQUESTED edit on the assigned
    workspace file -- 13 of 15 issued writes missed the target, and the
    measured miss shapes are wrong-ROOT (``write_file('<plan-id>/uploader.py')``,
    ``read_plan_file('<plan-id>/uploader.py')``).  This extends the D-035
    driver-assigned-target pattern (measured 10/10 on EXPLORE) to EXECUTE: the
    prompt names the exact path, the exact tool and the exact root, so the
    model's job shrinks to content.  The driver half (derivation from plan.md's
    Files To Modify) is pinned in ``test_harness_agent.py``.
    """

    _TARGET = "uploader.py"

    def _request(
        self, plan_dir: Path, workspace: Path, target: str | None
    ) -> RoleRequest:
        return _role_request(
            HarnessStates.EXECUTE,
            plan_dir=plan_dir,
            workspace_root=workspace,
            assigned_write_target=target,
        )

    def _line(self) -> str:
        return f"WRITE IT TO: {self._TARGET} USING {WorkspaceTools.WRITE_FILE}"

    def test_the_task_prompt_renders_the_target_exactly_once(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """Exactly one assignment line -- a repeated one is a new ambiguity."""
        spec = get_role_spec(HarnessStates.EXECUTE)
        task = build_role_task_prompt(
            self._request(plan_dir, workspace, self._TARGET), spec
        )

        assert task.count(self._line()) == 1

    def test_the_line_names_the_workspace_root_and_the_write_tool(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The root is stated, not inferred -- the B0 misses are ROOT misses.

        ``_writes_line`` already tells EXECUTE its plan-artifact paths are
        plan-directory-relative; without an explicit counter-label the model
        prefixed the workspace file with the plan id 13/15 times.
        """
        spec = get_role_spec(HarnessStates.EXECUTE)
        task = build_role_task_prompt(
            self._request(plan_dir, workspace, self._TARGET), spec
        )

        assert WorkspaceTools.WRITE_FILE in self._line()
        assert "workspace-relative" in task
        assert "NOT plan-directory-relative" in task
        assert "do not prefix it with the plan directory" in task

    def test_the_assignment_is_in_the_task_half_not_the_standing_half(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The target changes per step, so it is TASK text (D-021, D-035).

        Standing text is re-sent unchanged on the next dispatch -- installed
        there, step 2 would be told to write step 1's file.
        """
        request = self._request(plan_dir, workspace, self._TARGET)
        spec = get_role_spec(HarnessStates.EXECUTE)

        assert self._line() in build_role_task_prompt(request, spec)
        assert self._line() not in build_role_system_prompt(request, spec)

    def test_the_single_string_prompt_carries_the_assignment_too(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """``build_role_prompt`` is what a no-system-policy agent receives."""
        request = self._request(plan_dir, workspace, self._TARGET)

        assert self._line() in build_role_prompt(
            request, get_role_spec(HarnessStates.EXECUTE)
        )

    def test_no_assignment_leaves_the_prompt_byte_identical(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The fail-open-to-status-quo contract: ``None`` is a true no-op.

        The driver assigns nothing when plan.md is missing or unparseable;
        that dispatch must render byte-for-byte the pre-D-010 prompt (verified
        against the pre-change rendering at implementation time) -- additive,
        never a reflow of existing blocks.
        """
        spec = get_role_spec(HarnessStates.EXECUTE)
        base = build_role_prompt(self._request(plan_dir, workspace, None), spec)
        assigned = build_role_prompt(
            self._request(plan_dir, workspace, self._TARGET), spec
        )

        assert "WRITE IT TO:" not in base
        assert assigned != base
        # Nothing was REPLACED: the assignment is additive.
        for block in base.split("\n\n"):
            assert block in assigned

    @pytest.mark.parametrize(
        "state",
        [
            HarnessStates.EXPLORE,
            HarnessStates.PLAN,
            HarnessStates.REFLECT,
            HarnessStates.PIVOT,
            HarnessStates.CLOSE,
        ],
    )
    def test_a_state_the_driver_never_assigns_a_target_for_is_unchanged(
        self, state: str, plan_dir: Path, workspace: Path
    ) -> None:
        """Only EXECUTE dispatches carry a write target; no other state drifts."""
        spec = get_role_spec(state)
        request = _role_request(state, plan_dir=plan_dir, workspace_root=workspace)

        assert "YOUR EDIT THIS STEP" not in build_role_prompt(request, spec)


class TestThePlanDeliverableLineNamesPlanMd:
    """One PLAN dispatch, one static named deliverable: ``plan.md`` (D-001).

    L6 B0 (n=3, ``scripts/bench_data/l6-e2e``) measured the defect this line
    exists for: the one run that reached PLAN got an empty plan-writer reply
    with ``plan_md_bytes=0`` and stalled -- nothing in the prompt had named
    ``plan.md`` as THE deliverable of the dispatch.  This applies the D-035
    driver-names-the-ONE-file pattern -- the structural-over-verbal lever
    measured twice (EXPLORE 10/10; EXECUTE content-match 2/40 -> 40/40, Fisher
    p=1.6e-20) -- to PLAN.  Since plan-2026-07-23 the driver SEEDS a
    HEADERS-ONLY scaffold at PLAN entry, so the line steers the model to FILL it
    by APPENDING with ``append_plan_file`` (measured: 4b picks append 5/5 when
    told the scaffold exists), NOT the overwriting ``write_plan_file`` -- which
    would destroy the seeded headers.  The path is STATIC (``plan.md`` is the
    plan-writer's one obligation per ``rules.OWNERSHIP``), so there is no driver
    derivation and no ``RoleRequest`` field: the line renders on every PLAN
    dispatch that holds a plan directory, and on nothing else.
    """

    _MARKER = "YOUR DELIVERABLE THIS DISPATCH"

    def _line(self) -> str:
        return f"FILL IN {ArtifactNames.PLAN} USING {PlanTools.APPEND_PLAN_FILE}"

    def test_the_plan_task_prompt_renders_the_deliverable_exactly_once(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """Exactly one deliverable line -- a repeated one is a new ambiguity."""
        request = _role_request(
            HarnessStates.PLAN, plan_dir=plan_dir, workspace_root=workspace
        )
        task = build_role_task_prompt(request, get_role_spec(HarnessStates.PLAN))

        assert task.count(self._line()) == 1

    def test_the_line_names_the_append_tool_and_the_scaffold(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The tool is APPEND (never the overwriting write), and the line tells
        the model the scaffold already exists so it FILLS rather than clobbers."""
        request = _role_request(
            HarnessStates.PLAN, plan_dir=plan_dir, workspace_root=workspace
        )
        task = build_role_task_prompt(request, get_role_spec(HarnessStates.PLAN))
        block = next(b for b in task.split("\n\n") if self._MARKER in b)

        assert PlanTools.APPEND_PLAN_FILE in self._line()
        assert "plan-directory-relative" in task
        assert "ALREADY EXISTS" in block
        assert "11 section headers" in block
        # write_plan_file appears only as the tool NOT to use (do NOT rewrite)
        assert f"do NOT rewrite the whole file with {PlanTools.WRITE_PLAN_FILE}" in (
            block
        )

    def test_the_line_does_not_forbid_the_other_owned_artifacts(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """No ``nothing else``: the plan-writer also owns ``decisions.md`` and
        ``verification.md``, and the EXECUTE line's exclusivity clause would
        order it not to write files its own operative rules require."""
        request = _role_request(
            HarnessStates.PLAN, plan_dir=plan_dir, workspace_root=workspace
        )
        task = build_role_task_prompt(request, get_role_spec(HarnessStates.PLAN))
        block = next(b for b in task.split("\n\n") if self._MARKER in b)

        assert "nothing else" not in block

    def test_the_line_is_in_the_task_half_not_the_standing_half(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """Assignment lines are TASK text (D-021, D-035): the sibling topic
        and target lines live there, and the split's system half must stay
        byte-identical so the standing-policy measurement is not re-run."""
        request = _role_request(
            HarnessStates.PLAN, plan_dir=plan_dir, workspace_root=workspace
        )
        spec = get_role_spec(HarnessStates.PLAN)

        assert self._line() in build_role_task_prompt(request, spec)
        assert self._line() not in build_role_system_prompt(request, spec)

    def test_the_single_string_prompt_carries_the_line_too(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """``build_role_prompt`` is what a no-system-policy agent receives."""
        request = _role_request(
            HarnessStates.PLAN, plan_dir=plan_dir, workspace_root=workspace
        )

        assert self._line() in build_role_prompt(
            request, get_role_spec(HarnessStates.PLAN)
        )

    def test_no_plan_directory_renders_no_deliverable_line(
        self, workspace: Path
    ) -> None:
        """Fail-open to the status quo: a PLAN dispatch with no plan directory
        holds no plan-write tool, so naming the deliverable would be an
        unexecutable instruction -- the prompt stays byte-identical instead
        (hash-verified against the pre-change rendering)."""
        request = _role_request(
            HarnessStates.PLAN, plan_dir=None, workspace_root=workspace
        )

        assert self._MARKER not in build_role_prompt(
            request, get_role_spec(HarnessStates.PLAN)
        )

    @pytest.mark.parametrize(
        "state",
        [
            HarnessStates.EXPLORE,
            HarnessStates.EXECUTE,
            HarnessStates.REFLECT,
            HarnessStates.PIVOT,
            HarnessStates.CLOSE,
        ],
    )
    def test_a_state_other_than_plan_never_carries_the_line(
        self, state: str, plan_dir: Path, workspace: Path
    ) -> None:
        """Only the plan-writer's deliverable is ``plan.md``; no other state's
        prompt drifts (hash-verified byte-identical at implementation time)."""
        request = _role_request(state, plan_dir=plan_dir, workspace_root=workspace)

        assert self._MARKER not in build_role_prompt(request, get_role_spec(state))

    def test_the_line_reaches_a_real_dispatch_through_the_factory(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """End of the seam: the text a REAL worker sends, not a rebuilt one."""
        built: list[_PolicyAgent] = []
        spec = get_role_spec(HarnessStates.PLAN)

        def builder(spec_: Any, registry: Any, config: Any) -> _PolicyAgent:
            agent = _PolicyAgent(
                registry,
                (),
                "done",
                spec.output_schema(
                    **{
                        name: (
                            "s"
                            if field.annotation is str
                            else (False if field.annotation is bool else 0)
                        )
                        for name, field in spec.output_schema.model_fields.items()
                    }
                ),
            )
            built.append(agent)
            return agent

        factory = build_default_worker_factory(
            Workspace(workspace), agent_builder=builder
        )
        factory(
            _role_request(
                HarnessStates.PLAN, plan_dir=plan_dir, workspace_root=workspace
            )
        )

        assert self._line() in built[0].tasks[0]


class TestAMissingPlanArtifactSaysSoAndSaysWhatToDo:
    """A read of a file nobody has written yet is a teachable failure (D-036).

    Step 25's live n=10 block measured 124 of 323 failed tool calls reading a
    ``findings/*.md`` that did not exist yet, and in the three runs that missed
    the gate those reads WERE the failure: run 3 called
    ``read_plan_file('findings/constraints-and-patterns.md')`` fifteen times,
    ENOENT every time, and never wrote the file it had been assigned.  A bare
    ``[Errno 2]`` says nothing about what to do next.

    The refusal stays a refusal.  Nothing here repairs, re-routes or invents a
    file; every clause is read off the filesystem.
    """

    def test_a_read_of_an_unwritten_artifact_says_it_can_be_created(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(
            registry,
            "read_plan_file",
            path=f"{ArtifactNames.FINDINGS_DIR}/constraints-and-patterns.md",
        )

        assert failed.success is False
        assert "does not exist yet" in failed.error
        assert "`write_plan_file`" in failed.error

    def test_the_hint_creates_nothing(self, plan_dir: Path, workspace: Path) -> None:
        """Advice, not a side effect: the failed read leaves no file behind."""
        registry, _ = _plan_registry(plan_dir, workspace)

        _call(registry, "read_plan_file", path="findings/problem-scope.md")

        assert list((plan_dir / ArtifactNames.FINDINGS_DIR).iterdir()) == []

    def test_a_read_of_an_artifact_that_exists_is_untouched(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The control: a successful read gets no hint, because nothing failed."""
        registry, _ = _plan_registry(plan_dir, workspace)
        (plan_dir / ArtifactNames.FINDINGS_DIR / "problem-scope.md").write_text("hi\n")

        ok = _call(registry, "read_plan_file", path="findings/problem-scope.md")

        assert ok.success is True
        assert "does not exist yet" not in str(ok.result)

    def test_a_failed_read_of_something_that_IS_there_is_not_called_missing(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The `exists` guard: a read can fail for reasons other than absence.

        Reading ``findings`` (the directory, which exists) fails with EISDIR.
        Answering that with "does not exist yet" would have the tool contradict
        its own filesystem, and would tell the role to `write_plan_file` over a
        directory.
        """
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(registry, "read_plan_file", path=ArtifactNames.FINDINGS_DIR)

        assert failed.success is False
        assert "does not exist yet" not in failed.error
        assert (plan_dir / ArtifactNames.FINDINGS_DIR).is_dir()

    def test_a_bare_basename_is_pointed_at_the_file_that_really_exists(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """The measured second shape: the owning directory dropped from the path.

        One step-25 run called ``read_plan_file('problem-scope.md')`` eleven
        times while the file sat at ``findings/problem-scope.md``.  The routing
        hint alone answered that with "use `read_file`" -- true of the path as
        typed, and useless, because the file is not in the workspace either.
        """
        registry, _ = _plan_registry(plan_dir, workspace)
        (plan_dir / ArtifactNames.FINDINGS_DIR / "problem-scope.md").write_text("hi\n")

        failed = _call(registry, "read_plan_file", path="problem-scope.md")

        assert failed.success is False
        assert "Did you mean `findings/problem-scope.md`?" in failed.error
        assert "use `read_file`" not in failed.error

    def test_a_basename_that_matches_nothing_keeps_the_routing_hint(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """No suggestion is invented: with nothing on disk, D-027's hint stands."""
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(registry, "read_plan_file", path="uploader.py")

        assert failed.success is False
        assert "Did you mean" not in failed.error
        assert "`read_file`" in failed.error

    def test_a_failed_WRITE_is_never_told_to_write(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """Scoped to reads: an ownership refusal must not read as encouragement."""
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(
            registry, "write_plan_file", path=ArtifactNames.PLAN, content="mine now\n"
        )

        assert failed.success is False
        assert "does not exist yet" not in failed.error
        assert not (plan_dir / ArtifactNames.PLAN).exists()

    def test_a_workspace_read_is_untouched(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """No plan memory, no plan-artifact claim about a workspace path."""
        registry, _ = _plan_registry(plan_dir, workspace)

        failed = _call(registry, "read_file", path="uploader.py")

        assert failed.success is False
        assert "does not exist yet" not in failed.error

    def test_the_ownership_refusal_still_refuses(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """A role reading an artifact it does not own is a refusal, not a hint.

        The explorer may READ any artifact -- so this asserts the other half:
        the hint fires on ABSENCE, and an artifact that is present is read.
        """
        registry, _ = _plan_registry(plan_dir, workspace)
        (plan_dir / ArtifactNames.PLAN).write_text("# Plan\n")

        ok = _call(registry, "read_plan_file", path=ArtifactNames.PLAN)

        assert ok.success is True
        assert "does not exist yet" not in str(ok.result)


# ---------------------------------------------------------------------------
# Step 13 gap closure: the branches 430 green tests never entered
# ---------------------------------------------------------------------------


class TestRoleSpecLookupFailsLoudly:
    """An unknown state is a PROGRAMMER error and must not degrade to a default.

    Silently returning a spec for an unrecognised state would hand a role the
    wrong tool scope and the wrong writable keys -- the two things every gate
    downstream trusts.
    """

    def test_an_unknown_state_raises_state_not_found(self) -> None:
        with pytest.raises(StateNotFoundError) as excinfo:
            get_role_spec("EXPLORE_V2")

        assert excinfo.value.state_id == "EXPLORE_V2"
        assert "EXPLORE_V2" in str(excinfo.value)
        for state in HarnessStates.ALL:
            assert state in str(excinfo.value), "the message names the real six"

    def test_the_cause_is_dropped_so_the_message_is_the_message(self) -> None:
        """``from None``: a bare ``KeyError`` traceback helps nobody here."""
        with pytest.raises(StateNotFoundError) as excinfo:
            get_role_spec("")

        assert excinfo.value.__cause__ is None

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_every_real_state_still_resolves(self, state: str) -> None:
        """The control: the raise did not become unconditional."""
        assert get_role_spec(state).state == state


class TestTopLevelObjectCount:
    """The D-031 fail-open DETECTOR.  It observes; it must observe correctly.

    A count that under-reports would make the draft-then-correction path
    invisible in the live bench, which is the only place it is measured.
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ('{"a": 1}', 1),
            ('{"a": 1} {"b": 2}', 2),
            ('{"a": 1}\n\n{"b": 2}\n\n{"c": 3}', 3),
            ('{"a": {"b": {"c": 1}}}', 1),
            ("", 0),
            ("no braces at all", 0),
            ('{"a": 1', 0),
            ('{"a": "}"}', 1),
            ('prose {"a": 1} more prose {"b": 2} end', 2),
        ],
    )
    def test_the_count_matches_the_extractor_definition_of_top_level(
        self, text: str, expected: int
    ) -> None:
        assert count_top_level_json_objects(text) == expected

    def test_a_nested_object_is_not_counted_separately(self) -> None:
        """``text.count("{")`` -- the obvious shortcut -- returns 3 here."""
        assert count_top_level_json_objects('{"a": {"b": 1}, "c": {"d": 2}}') == 1

    def test_an_unbalanced_open_brace_between_two_objects_is_skipped(self) -> None:
        """The ``partners.get(start) is None`` branch: a truncated middle."""
        assert count_top_level_json_objects('{"a": 1} { {"b": 2}') == 2

    def test_a_brace_inside_a_string_does_not_open_an_object(self) -> None:
        """This is why the count borrows core's scanner instead of counting."""
        assert count_top_level_json_objects('{"note": "a { brace"}') == 1

    @pytest.mark.parametrize("value", [None, 3, 3.5, True, [], {}, object()])
    def test_a_non_string_counts_zero(self, value: Any) -> None:
        assert count_top_level_json_objects(value) == 0

    @pytest.mark.parametrize("text", ["{" * 500, "}" * 500, '{"a": "' + "{" * 200])
    def test_never_raises_on_hostile_input(self, text: str) -> None:
        """Surviving the call is the guard; the count itself is exact.

        The old ``>= 0`` form was vacuous -- true of every int the function
        could ever return (flagged by the W3 pre-audit, fixed under D-007).
        None of these inputs contains one balanced top-level object, so the
        exact answer is 0: unbalanced opens have no partner, and the third
        case buries its braces in an unterminated string value.
        """
        assert count_top_level_json_objects(text) == 0


class TestContextSnapshotIsBounded:
    """A 4B context window is the scarce resource; the snapshot must respect it."""

    def _prompt_for(self, context: dict[str, Any], plan_dir: Path) -> str:
        request = _role_request(
            HarnessStates.EXPLORE, plan_dir=plan_dir, context=context
        )
        return build_role_prompt(request, get_role_spec(HarnessStates.EXPLORE))

    def test_a_long_value_is_truncated_with_an_ellipsis(self, plan_dir: Path) -> None:
        prompt = self._prompt_for({"blob": "x" * 5000}, plan_dir)

        assert "x" * 120 + "..." in prompt
        assert "x" * 121 not in prompt

    def test_the_number_of_keys_is_capped(self, plan_dir: Path) -> None:
        context = {f"k{index:02d}": f"v{index}" for index in range(40)}

        prompt = self._prompt_for(context, plan_dir)

        rendered = [key for key in context if f"- {key}: " in prompt]
        assert len(rendered) == 12, "the cap is 12 keys, not 40"

    def test_internal_and_bulk_keys_never_reach_the_prompt(
        self, plan_dir: Path
    ) -> None:
        context = {
            "_secret": "internal",
            ContextKeys.DISPATCH_LEDGER: "ledger blob",
            ContextKeys.ROLE_RESULTS: "results blob",
            ContextKeys.CURRENT_ROLE_RESULT: "current blob",
            ContextKeys.GOAL: "the goal",
            "visible": "kept",
        }

        prompt = self._prompt_for(context, plan_dir)

        assert "- visible: kept" in prompt
        for hidden in ("internal", "ledger blob", "results blob", "current blob"):
            assert hidden not in prompt

    def test_a_none_value_is_omitted_rather_than_rendered_as_none(
        self, plan_dir: Path
    ) -> None:
        prompt = self._prompt_for({"unset": None, "set": "yes"}, plan_dir)

        assert "- unset:" not in prompt
        assert "- set: yes" in prompt


class TestReconcileStructuredFailsClosed:
    """A payload that cannot carry the DERIVED count is dropped, never passed on."""

    def test_an_unrebuildable_payload_is_dropped_entirely(
        self, tmp_path: Path, plan_dir: Path, captured_logs: Any
    ) -> None:
        """A schema that rejects the real disk count must not survive uncorrected.

        Returning the payload unchanged would leave the worker's CLAIM in
        ``structured_output`` for ``_apply_role_result`` to merge.
        """

        class _Strict(BaseModel):
            findings_count: int = Field(ge=3)
            needs_explore: bool = False

        spec = get_role_spec(HarnessStates.EXPLORE)
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry,
                (),
                '{"findings_count": 3, "needs_explore": false}',
                _Strict(findings_count=3),
            ),
        )

        result = factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert result.structured_output is None, "0 files on disk cannot pass ge=3"
        assert result.final_context.get(ContextKeys.FINDINGS_COUNT) == 0
        assert any("dropping the payload" in line for line in captured_logs)
        assert spec.state == HarnessStates.EXPLORE

    def test_a_payload_without_the_derived_key_is_returned_untouched(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """The control: reconciliation only rebuilds what it must."""

        class _Unrelated(BaseModel):
            message: str

        payload = _Unrelated(message="done")
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry, (), '{"message": "done"}', payload
            ),
        )

        result = factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert result.structured_output is payload


class TestWorkerFactoryFailurePaths:
    """What the factory does when the agent itself blows up mid-dispatch."""

    def test_an_agent_exception_propagates_after_being_observed(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """The driver must SEE the exception -- and the bench must see the record.

        Swallowing it here would also swallow ``HarnessReentrancyError``, the
        one exception the driver must never lose (invariant I5).
        """
        seen: list[dict[str, Any]] = []

        class _Exploding:
            def run(self, task: str) -> AgentResult:
                raise RuntimeError("the agent loop died")

        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            sleep=lambda _s: None,
            agent_builder=lambda spec_, registry, config: _Exploding(),
        )

        with pytest.raises(RuntimeError, match="the agent loop died"):
            factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert len(seen) == 1
        assert seen[0]["failure_reason"] == "exception:RuntimeError"
        assert seen[0]["success"] is False
        assert seen[0]["agent_success"] is False
        # Not `>= 0.0` (vacuous for a monotonic-clock delta -- W3 pre-audit,
        # D-007): what an observer row consumer relies on is the TYPE contract.
        assert isinstance(seen[0]["elapsed_s"], float)

    def test_the_failure_record_has_the_SAME_keys_as_a_success_record(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """An observer reading the verification fields must not have to branch."""
        seen: list[dict[str, Any]] = []
        payload = _explore_payload()
        spec = get_role_spec(HarnessStates.EXPLORE)

        class _Exploding:
            def run(self, task: str) -> AgentResult:
                raise TimeoutError("provider gone")

        ok_factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry, (), json.dumps(payload), spec.output_schema(**payload)
            ),
        )
        ok_factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        bad_factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            sleep=lambda _s: None,
            retry_attempts=1,
            agent_builder=lambda spec_, registry, config: _Exploding(),
        )
        with pytest.raises(TimeoutError):
            bad_factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert set(seen[0]) == set(seen[1])

    def test_a_transient_agent_fault_is_retried_before_it_is_reported(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """``retry_attempts`` is wired to the real backoff, not decorative."""
        seen: list[dict[str, Any]] = []
        delays: list[float] = []
        attempts: list[int] = []

        class _FlakyAgent:
            def run(self, task: str) -> AgentResult:
                attempts.append(1)
                raise ConnectionError("ollama not up yet")

        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            retry_attempts=3,
            sleep=delays.append,
            agent_builder=lambda spec_, registry, config: _FlakyAgent(),
        )

        with pytest.raises(ConnectionError):
            factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert len(attempts) == 3, "one dispatch, three provider attempts"
        assert len(delays) == 2
        assert len(seen) == 1, "one dispatch is one observation, not three"


class TestPayloadPreference:
    """Which channel the worker seam believes, and in what order."""

    def _dispatch_with(
        self, tmp_path: Path, plan_dir: Path, *, answer: str, structured: Any
    ) -> AgentResult:
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry, (), answer, structured
            ),
        )
        return factory(_role_request(HarnessStates.PLAN, plan_dir=plan_dir))

    def test_a_validated_structured_output_beats_the_free_text_answer(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        spec = get_role_spec(HarnessStates.PLAN)
        structured = spec.output_schema(
            **{
                ContextKeys.TOTAL_STEPS: 7,
                ContextKeys.NEEDS_EXPLORE: False,
                "message": "plan drafted",
                # PLAN's schema now requires the 11 SECTIONS-derived content
                # fields; they are schema-visible but not writable, so they do
                # not reach context (only TOTAL_STEPS/NEEDS_EXPLORE do).
                **{slug: f"{slug} body" for slug in PlanSchema.SECTION_SLUGS},
            }
        )

        result = self._dispatch_with(
            tmp_path,
            plan_dir,
            answer='{"total_steps": 99, "needs_explore": true}',
            structured=structured,
        )

        assert result.final_context[ContextKeys.TOTAL_STEPS] == 7

    def test_a_mapping_structured_output_is_accepted_without_a_model(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        result = self._dispatch_with(
            tmp_path,
            plan_dir,
            answer="prose only, no json here",
            structured={
                ContextKeys.TOTAL_STEPS: 5,
                ContextKeys.NEEDS_EXPLORE: False,
                "message": "plan drafted",
            },
        )

        assert result.final_context[ContextKeys.TOTAL_STEPS] == 5

    def test_the_answer_is_used_when_there_is_no_structured_output(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        result = self._dispatch_with(
            tmp_path,
            plan_dir,
            answer='```json\n{"total_steps": 4, "needs_explore": false, '
            '"message": "ok"}\n```',
            structured=None,
        )

        assert result.final_context[ContextKeys.TOTAL_STEPS] == 4

    def test_garbage_in_both_channels_writes_no_gate_key(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """The fail-closed end state: nothing accepted, so the gate stays shut."""
        result = self._dispatch_with(
            tmp_path, plan_dir, answer="I am unable to comply.", structured=None
        )

        assert result.success is False
        assert ContextKeys.TOTAL_STEPS not in result.final_context


class TestVerifiedWritesIgnoresUnusableCalls:
    """Write EVIDENCE is bytes on disk, reached through the confined root."""

    def test_a_write_call_with_no_path_parameter_is_not_evidence(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """A malformed tool call cannot count as a write."""

        class _BadCallAgent:
            def run(self, task: str) -> AgentResult:
                return AgentResult(
                    answer='{"findings_count": 3, "needs_explore": false, '
                    '"message": "done"}',
                    success=True,
                    trace=AgentTrace(
                        tool_calls=[
                            ToolCall(tool_name="write_plan_file", parameters={}),
                            ToolCall(
                                tool_name="write_plan_file", parameters={"path": "   "}
                            ),
                            ToolCall(
                                tool_name="write_plan_file", parameters={"path": 42}
                            ),
                            ToolCall(
                                tool_name="read_plan_file",
                                parameters={"path": "plan.md"},
                            ),
                        ],
                        total_iterations=4,
                    ),
                    final_context={},
                )

        seen: list[dict[str, Any]] = []
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _BadCallAgent(),
        )

        result = factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert seen[-1]["write_evidence"] == 0
        assert result.success is False
        assert seen[-1]["failure_reason"] == "unverified-write"

    def test_a_non_string_path_is_not_evidence_even_when_the_file_exists(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """Evidence must agree with what the TOOL layer would have accepted.

        ``Workspace.resolve`` refuses a non-``str`` path outright, so a trace
        entry carrying a ``Path`` object cannot have been the call that wrote
        those bytes.  Stringifying it here -- the obvious relaxation -- would
        credit a dispatch for a file some EARLIER dispatch wrote.
        """
        (plan_dir / "findings" / "scope.md").write_text("# written earlier\n")

        class _PathObjectAgent:
            def run(self, task: str) -> AgentResult:
                return AgentResult(
                    answer='{"findings_count": 3, "needs_explore": false, '
                    '"message": "done"}',
                    success=True,
                    trace=AgentTrace(
                        tool_calls=[
                            ToolCall(
                                tool_name="write_plan_file",
                                parameters={"path": Path("findings/scope.md")},
                            )
                        ],
                        total_iterations=1,
                    ),
                    final_context={},
                )

        seen: list[dict[str, Any]] = []
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _PathObjectAgent(),
        )

        result = factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert seen[-1]["write_evidence"] == 0, "a pre-existing file is not this write"
        assert result.success is False
        assert seen[-1]["failure_reason"] == "unverified-write"

    def test_the_same_path_as_a_real_string_IS_evidence(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """The control: the check above rejects the TYPE, not the file."""
        (plan_dir / "findings" / "scope.md").write_text("# written earlier\n")

        seen: list[dict[str, Any]] = []
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry,
                (
                    (
                        "write_plan_file",
                        {"path": "findings/scope.md", "content": "# rewritten\n"},
                    ),
                ),
                '{"findings_count": 3, "needs_explore": false, "message": "done"}',
                None,
            ),
        )

        factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        assert seen[-1]["write_evidence"] == 1

    def test_a_write_with_no_plan_directory_leaves_no_plan_evidence(
        self, tmp_path: Path
    ) -> None:
        """``plan_dir=None`` means no ``PlanMemory``, so no reader, so no evidence."""

        class _ClaimingAgent:
            def run(self, task: str) -> AgentResult:
                return AgentResult(
                    answer='{"findings_count": 3, "needs_explore": false, '
                    '"message": "wrote three findings"}',
                    success=True,
                    trace=AgentTrace(
                        tool_calls=[
                            ToolCall(
                                tool_name="write_plan_file",
                                parameters={"path": "findings/a.md"},
                            )
                        ],
                        total_iterations=1,
                    ),
                    final_context={},
                )

        seen: list[dict[str, Any]] = []
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _ClaimingAgent(),
        )

        result = factory(_role_request(HarnessStates.EXPLORE, plan_dir=None))

        assert seen[-1]["write_evidence"] == 0
        assert ContextKeys.FINDINGS_COUNT not in result.final_context


class TestWriteEvidenceRootSplit:
    """The observation's root-attributed write split (D-005).

    ``_verified_writes`` labels every verified write ``"<root>:<path>"``, but
    the observation record used to collapse the labels to a bare ``len()``
    (roles.py:1285 pre-D-005) -- the workspace-vs-plan attribution was
    computed and then thrown away, so no observer could tell an EXECUTE-state
    WORKSPACE edit from a plan-directory note.  The bare int stays: its
    existing consumer (test_live_ollama.py) must read the same number.
    """

    def _observe_execute(
        self,
        tmp_path: Path,
        plan_dir: Path | None,
        calls: tuple[tuple[str, dict[str, Any]], ...],
    ) -> dict[str, Any]:
        """One EXECUTE dispatch through the REAL factory; its observation."""
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        seen: list[dict[str, Any]] = []
        factory = build_default_worker_factory(
            Workspace(ws),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry,
                calls,
                '{"summary": "did the step", "message": "done"}',
                None,
            ),
        )
        factory(
            _role_request(HarnessStates.EXECUTE, plan_dir=plan_dir, workspace_root=ws)
        )
        return seen[-1]

    def test_a_workspace_only_trace_attributes_to_workspace(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        record = self._observe_execute(
            tmp_path,
            plan_dir,
            (("write_file", {"path": "uploader.py", "content": "x = 1\n"}),),
        )

        assert record["write_evidence"] == 1
        assert record["write_evidence_workspace"] == 1
        assert record["write_evidence_plan"] == 0
        # D-010: the raw labels ride along so an observer can attribute a
        # workspace diff to THIS dispatch's own writes, not just count them.
        assert record["write_evidence_paths"] == ("workspace:uploader.py",)

    def test_a_plan_only_trace_attributes_to_plan(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """An executor that only wrote protocol notes shows NO workspace write."""
        record = self._observe_execute(
            tmp_path,
            plan_dir,
            (("write_plan_file", {"path": "changelog.md", "content": "# log\n"}),),
        )

        assert record["write_evidence"] == 1
        assert record["write_evidence_workspace"] == 0
        assert record["write_evidence_plan"] == 1

    def test_a_mixed_trace_splits_by_root(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        record = self._observe_execute(
            tmp_path,
            plan_dir,
            (
                ("write_file", {"path": "uploader.py", "content": "x = 1\n"}),
                ("write_plan_file", {"path": "changelog.md", "content": "# log\n"}),
            ),
        )

        assert record["write_evidence"] == 2
        assert record["write_evidence_workspace"] == 1
        assert record["write_evidence_plan"] == 1
        assert (
            record["write_evidence_workspace"] + record["write_evidence_plan"]
            == record["write_evidence"]
        )
        assert record["write_evidence_paths"] == (
            "workspace:uploader.py",
            "plan:changelog.md",
        )

    def test_an_empty_trace_reports_zero_in_all_three(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        record = self._observe_execute(tmp_path, plan_dir, ())

        assert record["write_evidence"] == 0
        assert record["write_evidence_workspace"] == 0
        assert record["write_evidence_plan"] == 0

    def test_the_exception_record_carries_the_split_keys_too(
        self, tmp_path: Path, plan_dir: Path
    ) -> None:
        """Same keys on both paths: an observer must never branch on shape."""

        class _RaisingAgent:
            def run(self, task: str) -> AgentResult:
                raise RuntimeError("boom")

        seen: list[dict[str, Any]] = []
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _RaisingAgent(),
        )

        with pytest.raises(RuntimeError):
            factory(_role_request(HarnessStates.EXECUTE, plan_dir=plan_dir))

        assert seen[-1]["write_evidence"] == 0
        assert seen[-1]["write_evidence_workspace"] == 0
        assert seen[-1]["write_evidence_plan"] == 0
        assert seen[-1]["write_evidence_paths"] == ()


class TestDefaultAgentBuilder:
    """The one construction choice the factory makes when nobody overrides it."""

    def test_the_react_arm_is_still_reachable_with_the_output_schema(
        self, tmp_path: Path
    ) -> None:
        """D-049 flipped the DEFAULT; it did not delete the ReAct arm.

        The arm stays reachable because it is the control criterion 1 is
        measured against (0/5 tool calls live), and a default chosen against a
        0/5 must stay falsifiable.  The schema still reaches it, so this arm
        keeps the constrained decoding that was D-031's mitigation.
        """
        spec = get_role_spec(HarnessStates.EXPLORE)
        registry = build_workspace_tools(
            Workspace(tmp_path / "ws"), allowed=spec.tool_scope
        )
        config = AgentConfig(model="ollama_chat/x", output_schema=spec.output_schema)

        agent = _default_agent_builder(native_function_calling=False)(
            spec, registry, config
        )

        assert isinstance(agent, ReactAgent)
        assert agent.config.output_schema is spec.output_schema

    def test_native_function_calling_swaps_the_agent_not_the_schema(
        self, tmp_path: Path
    ) -> None:
        spec = get_role_spec(HarnessStates.EXPLORE)
        registry = build_workspace_tools(
            Workspace(tmp_path / "ws"), allowed=spec.tool_scope
        )
        config = AgentConfig(model="ollama_chat/x", output_schema=spec.output_schema)

        agent = _default_agent_builder(native_function_calling=True)(
            spec, registry, config
        )

        assert isinstance(agent, NativeFunctionCallingReactAgent)
        assert agent.config.output_schema is spec.output_schema

    def test_seed_reaches_the_native_agent_and_defaults_to_none(
        self, tmp_path: Path
    ) -> None:
        """D-008: the live probe measured Ollama honoring ``seed`` on
        `qwen3.5:4b`, so the factory chain must carry it to the native agent
        (native_fc bypasses ``apply_ollama_params`` -- its own call site is the
        only place the key can land). Unset must stay unset: ``None`` means
        the provider payload carries no seed key at all.
        """
        spec = get_role_spec(HarnessStates.EXPLORE)
        registry = build_workspace_tools(
            Workspace(tmp_path / "ws"), allowed=spec.tool_scope
        )
        config = AgentConfig(model="ollama_chat/x", output_schema=spec.output_schema)

        seeded = _default_agent_builder(native_function_calling=True, seed=1234)(
            spec, registry, config
        )
        unseeded = _default_agent_builder(native_function_calling=True)(
            spec, registry, config
        )

        assert seeded.seed == 1234
        assert unseeded.seed is None

    def test_a_factory_with_no_agent_builder_uses_the_native_arm(
        self, tmp_path: Path, plan_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D-049: the SHIPPED default is native function calling.

        ``agent_builder=None`` is the production shape -- it is what
        ``__main__.py`` builds -- so this pins what a user of this package
        actually gets, not what a live bench opts into.  Until D-049 the same
        shape reached ``create_agent`` with the role's pattern; it must now
        reach ``NativeFunctionCallingReactAgent`` instead, and ``create_agent``
        must not be called at all.  The output schema still travels with it:
        the flip changes the agent, never the schema (D-002's repair turn is
        what makes that safe).

        The bar this pins is a CONTRACT, and it was chosen against 0/5, not
        4/5 -- see the D-049 block in ``roles.py``.
        """
        native_built: list[Any] = []
        react_built: list[tuple[str, Any]] = []

        class _Stub:
            def run(self, task: str) -> AgentResult:
                return AgentResult(answer="{}", success=True, final_context={})

        def _fake_create_agent(*, pattern: str, tools: Any, config: Any) -> Any:
            react_built.append((pattern, config.output_schema))
            return _Stub()

        def _fake_native(*, tools: Any, config: Any, seed: Any = None) -> Any:
            native_built.append(config.output_schema)
            assert seed is None, "the default factory shape carries no seed"
            return _Stub()

        monkeypatch.setattr("fsm_llm_harness.roles.create_agent", _fake_create_agent)
        monkeypatch.setattr(
            "fsm_llm_harness.roles.NativeFunctionCallingReactAgent", _fake_native
        )
        factory = build_default_worker_factory(Workspace(tmp_path / "ws"))

        factory(_role_request(HarnessStates.EXPLORE, plan_dir=plan_dir))

        spec = get_role_spec(HarnessStates.EXPLORE)
        assert native_built == [spec.output_schema]
        assert react_built == [], "the ReAct loop is the opt-in arm, not the default"


class TestMultiObjectReplyIsWarnedAbout:
    """The D-031 fail-open is invisible unless the dispatch says so out loud."""

    def test_a_two_object_answer_is_counted_and_warned(
        self, tmp_path: Path, plan_dir: Path, captured_logs: Any
    ) -> None:
        seen: list[dict[str, Any]] = []
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            observer=seen.append,
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry,
                (),
                '{"total_steps": 2, "needs_explore": false, "message": "draft"}\n'
                '{"total_steps": 9, "needs_explore": false, "message": "final"}',
                None,
            ),
        )

        result = factory(_role_request(HarnessStates.PLAN, plan_dir=plan_dir))

        assert seen[-1]["top_level_objects"] == 2
        assert any("top-level JSON objects" in line for line in captured_logs)
        assert result.final_context[ContextKeys.TOTAL_STEPS] == 2, "first-wins"

    def test_a_single_object_answer_is_not_warned_about(
        self, tmp_path: Path, plan_dir: Path, captured_logs: Any
    ) -> None:
        """The control: the warning fires on 2+, not on every dispatch."""
        factory = build_default_worker_factory(
            Workspace(tmp_path / "ws"),
            agent_builder=lambda spec_, registry, config: _ScriptedAgent(
                registry,
                (),
                '{"total_steps": 2, "needs_explore": false, "message": "only"}',
                None,
            ),
        )

        factory(_role_request(HarnessStates.PLAN, plan_dir=plan_dir))

        assert not [line for line in captured_logs if "top-level JSON" in line]


# ---------------------------------------------------------------------------
# Step 13 gap closure: `tools.py`'s bounds, refusals and corrective hints
# ---------------------------------------------------------------------------


class TestWorkspaceOutputIsBounded:
    """A 4B context window is finite; every read path has to end.

    An unbounded listing or grep does not fail loudly -- it fills the window
    and the dispatch silently loses its instructions.
    """

    def test_a_long_listing_is_truncated_and_says_so(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws")
        for index in range(MAX_LIST_ENTRIES + 25):
            (ws.root / f"f{index:04d}.py").write_text("x")

        names = ws.list_dir(".")

        assert len(names) == MAX_LIST_ENTRIES + 1
        assert names[-1] == "... [truncated: 25 more entries]"

    def test_a_short_listing_is_not_marked(self, tmp_path: Path) -> None:
        """The control: the marker is evidence of a cut, not decoration."""
        ws = Workspace(tmp_path / "ws")
        (ws.root / "a.py").write_text("x")

        assert ws.list_dir(".") == ["a.py"]

    def test_grep_stops_at_max_hits(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws")
        (ws.root / "many.py").write_text("needle\n" * (MAX_GREP_HITS + 10))

        hits = ws.grep("needle")

        assert len(hits) == MAX_GREP_HITS + 1
        assert "search stopped at" in hits[-1]

    def test_a_matching_line_is_itself_truncated(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws")
        (ws.root / "wide.py").write_text("needle" + "x" * 5000 + "\n")

        (hit,) = ws.grep("needle")

        assert "[truncated:" in hit
        assert len(hit) < 400

    def test_an_oversized_file_is_skipped_by_grep(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws")
        (ws.root / "huge.py").write_text("needle\n" + "x" * MAX_GREP_FILE_BYTES)
        (ws.root / "small.py").write_text("needle\n")

        hits = ws.grep("needle")

        assert hits == ["small.py:1: needle"]

    def test_a_binary_file_is_skipped_by_grep(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws")
        (ws.root / "blob.bin").write_bytes(b"needle\xff\xfe\x00binary")
        (ws.root / "text.py").write_text("needle\n")

        hits = ws.grep("needle")

        assert hits == ["text.py:1: needle"]

    def test_a_symlink_met_during_the_walk_is_skipped(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws")
        (ws.root / "real.py").write_text("needle\n")
        (ws.root / "link.py").symlink_to(ws.root / "real.py")

        hits = ws.grep("needle")

        assert hits == ["real.py:1: needle"], "the same bytes must not count twice"

    def test_an_invalid_regex_is_a_refusal_not_a_crash(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws")

        with pytest.raises(HarnessError, match="Invalid grep pattern"):
            ws.grep("(unclosed")

    def test_grep_over_a_single_file_target_works(self, tmp_path: Path) -> None:
        """``_walk_files``'s file branch: a role may grep one named file."""
        ws = Workspace(tmp_path / "ws")
        (ws.root / "one.py").write_text("needle\n")

        assert ws.grep("needle", "one.py") == ["one.py:1: needle"]


class TestWorkspaceRefusals:
    """The confinement layer's refusals, driven directly rather than inferred."""

    def test_a_non_string_path_is_a_type_error(self, tmp_path: Path) -> None:
        """A model that emits ``{"path": 3}`` must not index into a Path."""
        ws = Workspace(tmp_path / "ws")

        with pytest.raises(TypeError, match="must be a str"):
            ws.resolve(3)  # type: ignore[arg-type]

    def test_deleting_a_directory_is_refused(self, tmp_path: Path) -> None:
        """``delete`` removes single files; a recursive delete is not on offer."""
        ws = Workspace(tmp_path / "ws")
        (ws.root / "pkg").mkdir()
        (ws.root / "pkg" / "keep.py").write_text("x")

        with pytest.raises(HarnessError, match="delete only removes single files"):
            ws.delete("pkg")
        assert (ws.root / "pkg" / "keep.py").exists()

    def test_the_allowed_command_list_is_readable(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws", allow_shell=True)

        assert ws.allowed_commands == COMMAND_ALLOWLIST

    def test_the_reprs_name_the_root_and_the_scope(self, tmp_path: Path) -> None:
        """Diagnostics: a refusal is unreadable if the object prints as an id."""
        ws = Workspace(tmp_path / "ws")

        assert "Workspace(root=" in repr(ws)
        assert "allow_shell=False" in repr(ws)

    def test_the_plan_memory_repr_names_the_role(self, plan_dir: Path) -> None:
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        assert "PlanMemory(plan_dir=" in repr(memory)
        assert Role.EXPLORER in repr(memory)


class TestRunCommandFailurePaths:
    """``run_command`` is the only tool that leaves the process; it must not throw."""

    def test_an_empty_argv_is_refused(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "ws", allow_shell=True)

        with pytest.raises(HarnessError, match="non-empty argument vector"):
            ws.run_command([])
        with pytest.raises(HarnessError, match="non-empty argument vector"):
            ws.run_command(["   "])

    def test_a_missing_executable_is_a_failed_result_not_an_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An allowlisted command absent from PATH is a normal tool failure."""
        ws = Workspace(tmp_path / "ws", allow_shell=True)
        monkeypatch.setattr("fsm_llm_harness.tools.shutil.which", lambda *a, **k: None)

        result = ws.run_command([COMMAND_ALLOWLIST[0]])

        assert result.success is False
        assert "not found on PATH" in result.error

    def test_a_timeout_is_a_failed_result_not_an_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = Workspace(tmp_path / "ws", allow_shell=True)

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="x", timeout=1.0)

        monkeypatch.setattr("fsm_llm_harness.tools.subprocess.run", _boom)

        result = ws.run_command([COMMAND_ALLOWLIST[0]])

        assert result.success is False
        assert "timed out after" in result.error

    def test_a_start_failure_is_a_failed_result_not_an_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = Workspace(tmp_path / "ws", allow_shell=True)

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise OSError("exec format error")

        monkeypatch.setattr("fsm_llm_harness.tools.subprocess.run", _boom)

        result = ws.run_command([COMMAND_ALLOWLIST[0]])

        assert result.success is False
        assert "failed to start" in result.error

    def test_a_non_zero_exit_is_reported_as_a_failure_with_its_output(
        self, tmp_path: Path
    ) -> None:
        """A failing verification command must READ as failing to the model."""
        ws = Workspace(tmp_path / "ws", allow_shell=True)

        result = ws.run_command(["cat", "no-such-file.txt"])

        assert result.success is False
        assert "no-such-file" in result.error


class TestCorrectiveHintEdges:
    """The wrong-root hints must stay silent when they have nothing to say."""

    def test_a_tool_with_no_counterpart_gets_no_routing_hint(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        assert WorkspaceTools.RUN_COMMAND not in _COUNTERPART_TOOL
        assert (
            _routing_hint("x.py", tool_name=WorkspaceTools.RUN_COMMAND, memory=None)
            == ""
        )

    @pytest.mark.parametrize("path", ["", ".", "./", "   "])
    def test_the_current_directory_is_never_a_wrong_root_guess(self, path: str) -> None:
        assert _routing_hint(path, tool_name=WorkspaceTools.LIST_DIR, memory=None) == ""

    @pytest.mark.parametrize("path", ["", ".", "./", "   "])
    def test_the_current_directory_is_not_a_wrong_root_guess_in_the_PLAN_tier_either(
        self, plan_dir: Path, path: str
    ) -> None:
        """The plan side is the half that actually needs the short circuit.

        ``artifact_for(".")`` legitimately answers ``None`` -- the plan
        directory itself is not an artifact -- so without the guard a role
        listing its OWN directory would be told to go use the workspace tool.
        """
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        assert (
            _routing_hint(path, tool_name=PlanTools.LIST_PLAN_DIR, memory=memory) == ""
        )

    def test_a_workspace_path_handed_to_a_workspace_tool_gets_no_hint(self) -> None:
        assert (
            _routing_hint(
                "src/uploader.py", tool_name=WorkspaceTools.READ_FILE, memory=None
            )
            == ""
        )

    def test_a_plan_path_handed_to_a_workspace_tool_names_the_plan_tool(self) -> None:
        hint = _routing_hint(
            "findings/scope.md", tool_name=WorkspaceTools.READ_FILE, memory=None
        )

        assert "belongs to the plan directory" in hint
        assert _COUNTERPART_TOOL[WorkspaceTools.READ_FILE] in hint

    def test_a_workspace_path_handed_to_a_plan_tool_names_the_workspace_tool(
        self, plan_dir: Path
    ) -> None:
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        hint = _routing_hint(
            "src/uploader.py", tool_name=PlanTools.READ_PLAN_FILE, memory=memory
        )

        assert "belongs to the workspace" in hint

    def test_an_unclassifiable_path_still_answers_rather_than_raising(
        self, plan_dir: Path
    ) -> None:
        """``artifact_for`` raising must not turn a hint into a crashed dispatch."""
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        hint = _routing_hint(
            "/etc/passwd", tool_name=PlanTools.READ_PLAN_FILE, memory=memory
        )

        assert "belongs to the workspace" in hint

    @pytest.mark.parametrize("path", ["", "   ", "/", "."])
    def test_addresses_plan_memory_is_false_for_a_rootless_path(
        self, path: str
    ) -> None:
        assert _addresses_plan_memory(path) is False

    def test_a_relocated_hint_is_read_off_the_filesystem(self, plan_dir: Path) -> None:
        """The suggestion is a FACT about disk, never a guess about names."""
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)
        (plan_dir / "findings" / "scope.md").write_text("# scope\n")

        assert (
            _relocated_hint("scope.md", memory) == "Did you mean `findings/scope.md`?"
        )
        assert _relocated_hint("never-written.md", memory) == ""

    def test_a_relocated_hint_on_a_pathless_name_is_silent(
        self, plan_dir: Path
    ) -> None:
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        assert _relocated_hint("   ", memory) == ""

    def test_a_missing_per_plan_directory_suggests_nothing_and_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        bare = tmp_path / "plans" / "plan-2026-07-21T000000-bareplan"
        bare.mkdir(parents=True)
        memory = PlanMemory(bare, role=Role.EXPLORER)

        assert _relocated_hint("scope.md", memory) == ""


class TestToolRegistryConstruction:
    """An unknown tool name is a wiring bug and must fail at BUILD time.

    Degrading to "register what I recognise" would hand a role a silently
    narrower scope than its ``RoleSpec`` declares -- the exact review-C2 shape.
    """

    def test_an_unknown_workspace_tool_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown workspace tool"):
            build_workspace_tools(
                Workspace(tmp_path / "ws"), allowed=["read_file", "sudo_rm"]
            )

    def test_an_unknown_plan_tool_is_refused(self, plan_dir: Path) -> None:
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        with pytest.raises(ValueError, match="Unknown plan tool"):
            build_plan_tools(memory, allowed=["read_plan_file", "rm_plan"])

    def test_every_workspace_tool_is_reachable_through_the_registry(
        self, tmp_path: Path
    ) -> None:
        """Each ``@tool`` wrapper is exercised once, through the agent's path."""
        ws = Workspace(tmp_path / "ws", allow_shell=True)
        registry = build_workspace_tools(ws)

        assert _call(registry, "write_file", path="a.py", content="one\n").success
        assert _call(registry, "append_file", path="a.py", content="two\n").success
        assert (ws.root / "a.py").read_text() == "one\ntwo\n"
        assert _call(registry, "read_file", path="a.py").success
        assert _call(registry, "path_exists", path="a.py").result == "yes"
        assert _call(registry, "path_exists", path="zz.py").result == "no"
        assert "a.py" in str(_call(registry, "list_dir", path=".").result)
        assert "a.py:1" in str(_call(registry, "grep_files", pattern="one").result)
        assert "no matches" in str(
            _call(registry, "grep_files", pattern="zzz").result
        ).replace("(", "")
        assert _call(registry, "run_command", command="cat", args="a.py").success
        assert _call(registry, "delete_file", path="a.py").success
        assert not (ws.root / "a.py").exists()

    def test_an_empty_workspace_directory_lists_as_empty_not_as_nothing(
        self, tmp_path: Path
    ) -> None:
        registry = build_workspace_tools(Workspace(tmp_path / "ws"))

        assert _call(registry, "list_dir", path=".").result == "(empty)"

    def test_plan_path_exists_answers_both_ways(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        registry, _ = _plan_registry(plan_dir, workspace)
        (plan_dir / ArtifactNames.PLAN).write_text("# Plan\n")

        assert _call(registry, "plan_path_exists", path=ArtifactNames.PLAN).result == (
            "yes"
        )
        assert _call(registry, "plan_path_exists", path="state.md").result == "no"

    def test_the_memory_root_itself_classifies_as_no_artifact(
        self, plan_dir: Path
    ) -> None:
        """``_classify``'s rootless branch: the root is not a protocol artifact."""
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        assert memory.artifact_for(f"{plan_dir.name}/..") is None

    def test_an_empty_path_locates_to_itself(self, plan_dir: Path) -> None:
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        assert memory.locate("   ") == ""


class TestNeverRaisesOnAnUnclassifiablePath:
    """The three ``except Exception: return ''`` guards, driven directly.

    Each of these sits between a REFUSED tool call and the string the model
    reads back.  If any of them propagated, a confinement refusal -- the layer
    working correctly -- would surface as a crashed dispatch instead.
    """

    def test_gate_state_reports_nothing_for_a_refused_path(
        self, plan_dir: Path
    ) -> None:
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        with pytest.raises(HarnessConfinementError):
            memory.artifact_for("/etc/passwd")
        assert _gate_state(memory, "/etc/passwd", existed=False) == ""

    def test_owned_empty_directory_is_false_for_a_refused_path(
        self, plan_dir: Path
    ) -> None:
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        assert _owned_empty_directory(memory, "/etc") is False

    def test_the_missing_target_hint_defers_to_routing_for_a_refused_path(
        self, plan_dir: Path
    ) -> None:
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)

        hint = _missing_target_hint(
            "/etc/passwd", tool_name=PlanTools.READ_PLAN_FILE, memory=memory
        )

        assert hint == ""

    def test_the_same_helpers_still_answer_for_a_real_artifact(
        self, plan_dir: Path
    ) -> None:
        """The control: these guards did not swallow the normal answer too."""
        memory = PlanMemory(plan_dir, role=Role.EXPLORER)
        (plan_dir / "findings" / "scope.md").write_text("# scope\n")

        assert _gate_state(memory, "findings/scope.md", existed=False) != ""
        assert _owned_empty_directory(memory, ArtifactNames.CHECKPOINTS_DIR) is True
        assert (
            _missing_target_hint(
                ArtifactNames.PLAN, tool_name=PlanTools.READ_PLAN_FILE, memory=memory
            )
            != ""
        )


class TestRulesLookupEdges:
    """``rules.py``'s two remaining branches: one topic, and an unknown state."""

    def test_an_unknown_state_raises_from_get_rules_too(self) -> None:
        with pytest.raises(StateNotFoundError) as excinfo:
            get_rules("EXPLORE_V2")

        assert excinfo.value.state_id == "EXPLORE_V2"

    @pytest.mark.parametrize("threshold", [0, 1, 2, 3, 4, 7])
    def test_the_topic_list_never_shrinks_below_the_protocol_axes(
        self, threshold: int
    ) -> None:
        """REPORTED DEAD BRANCH, pinned rather than tested into permanence.

        ``rules._topic_phrase`` carries a ``len(labels) == 1`` branch that is
        structurally UNREACHABLE: ``explore_topics`` floors its result at
        ``EXPLORE_TOPICS`` (3 axes) for every threshold, including 0 and 1, so
        the single-label rendering can never be selected.  This test pins the
        FACT that makes it dead -- if someone ever shrinks ``EXPLORE_TOPICS``
        to one axis, this fails and the branch stops being dead in the same
        change that needs it.
        """
        labels = explore_topics(threshold)

        assert len(labels) >= 3
        assert len(labels) >= max(threshold, 3)
        assert len({topic.slug for topic in labels}) == len(labels)

    def test_the_rendered_topic_phrase_is_an_english_list(self) -> None:
        fsm = build_harness_fsm("exercise the harness protocol")
        purpose = fsm["states"][HarnessStates.EXPLORE]["purpose"]

        for topic in explore_topics():
            assert topic.label in purpose
        assert ", and " in purpose


class TestExploreOnlyForcesTheFinalWriteTool:
    """EXPLORE and PLAN arm ``force_final_tool``; nothing else does (D-003/D-001).

    The forced finalization turn (native_fc D-003) is additive and default-off;
    ``build_default_worker_factory``'s ``worker()`` closure is the SOLE site
    that arms it.  Two roles arm it, each with the tool its deliverable needs:
    EXPLORER -> ``write_plan_file`` (it CREATES its findings file), and
    PLAN_WRITER -> ``append_plan_file`` (it FILLS the driver-seeded scaffold, so
    forcing ``write_plan_file`` would OVERWRITE the headers -- D-001).
    EXECUTE/REFLECT/PIVOT/CLOSE arm nothing.  These tests drive the REAL
    per-dispatch ``AgentConfig`` construction inside that closure -- the same
    path a live dispatch takes -- rather than asserting against a hand-built
    ``AgentConfig`` that could drift from the wiring.
    """

    class _StopBeforeRun(Exception):
        """Raised inside the agent builder to short-circuit before dispatch."""

    def _config_built_for(
        self, state: str, plan_dir: Path, workspace: Path
    ) -> AgentConfig:
        """Run the factory far enough to capture the config ``worker()`` built."""
        captured: dict[str, AgentConfig] = {}

        def builder(spec_: Any, registry: Any, config: AgentConfig) -> Any:
            captured["config"] = config
            raise self._StopBeforeRun

        factory = build_default_worker_factory(
            Workspace(workspace), agent_builder=builder
        )
        with pytest.raises(self._StopBeforeRun):
            factory(
                _role_request(
                    state, plan_dir=plan_dir, workspace_root=workspace
                )
            )
        return captured["config"]

    def test_explore_config_carries_write_plan_file(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        config = self._config_built_for(
            HarnessStates.EXPLORE, plan_dir, workspace
        )

        assert config.force_final_tool == PlanTools.WRITE_PLAN_FILE
        assert config.force_final_tool == "write_plan_file"

    def test_plan_config_carries_append_plan_file(
        self, plan_dir: Path, workspace: Path
    ) -> None:
        """PLAN forces APPEND, never write: the driver seeds the scaffold and a
        forced overwrite would destroy the 11 headers (D-001)."""
        config = self._config_built_for(HarnessStates.PLAN, plan_dir, workspace)

        assert config.force_final_tool == PlanTools.APPEND_PLAN_FILE
        assert config.force_final_tool == "append_plan_file"
        assert config.force_final_tool != PlanTools.WRITE_PLAN_FILE

    @pytest.mark.parametrize(
        "state",
        [
            s
            for s in HarnessStates.ALL
            if s not in (HarnessStates.EXPLORE, HarnessStates.PLAN)
        ],
    )
    def test_execute_reflect_pivot_close_do_not_force_a_final_tool(
        self, state: str, plan_dir: Path, workspace: Path
    ) -> None:
        config = self._config_built_for(state, plan_dir, workspace)

        assert config.force_final_tool is None
