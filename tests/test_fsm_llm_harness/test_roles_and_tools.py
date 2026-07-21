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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fsm_llm.llm import _GENERIC_FALLBACK_MESSAGE, LiteLLMInterface
from fsm_llm_agents.definitions import AgentResult, AgentTrace, ToolCall
from fsm_llm_harness.constants import ArtifactNames, ContextKeys, HarnessStates, Role
from fsm_llm_harness.exceptions import HarnessConfinementError, HarnessOwnershipError
from fsm_llm_harness.hardening import coerce_worker_output, parse_role_output
from fsm_llm_harness.harness import _WORKER_WRITABLE, RoleRequest
from fsm_llm_harness.roles import (
    ROLE_SPECS,
    build_default_worker_factory,
    build_role_prompt,
    get_role_spec,
    held_tools,
)
from fsm_llm_harness.rules import (
    OWNERSHIP,
    ROLE_BY_STATE,
    artifacts_writable_by,
    get_rules,
)
from fsm_llm_harness.tools import (
    _PER_PLAN_DIRS,
    COMMAND_ALLOWLIST,
    PLAN_READ_TOOLS,
    PLAN_WRITE_TOOLS,
    READ_ONLY_TOOLS,
    SHELL_TOOLS,
    VERIFICATION_COMMANDS,
    WRITE_TOOLS,
    PlanMemory,
    Workspace,
    build_plan_tools,
    build_workspace_tools,
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
        context={ContextKeys.GOAL: "exercise the harness protocol"},
        plan_dir=None if plan_dir is None else str(plan_dir),
        workspace_root=None if workspace_root is None else str(workspace_root),
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
            "/workspace",
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
            registry, calls, answer if answer is not None else json.dumps(payload),
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
                    {"path": f"{ArtifactNames.FINDINGS_DIR}/c.md", "content": "found\n"},
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

        assert ArtifactNames.FINDINGS_INDEX not in explore.extraction_instructions
        assert f"{ArtifactNames.FINDINGS_DIR}/" in explore.extraction_instructions
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
                ("write_plan_file", {"path": ArtifactNames.PLAN, "content": "# plan\n"}),
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

    @pytest.mark.parametrize(
        "path", ["/plan/x.md", "/plan/findings/x.md"]
    )
    def test_plan_memory_still_repairs_the_same_shape(
        self, tmp_path: Path, path: str
    ) -> None:
        """The other direction of the same pin: the sentinel moved, not vanished."""
        memory = PlanMemory(tmp_path / "plans" / "plan-x", role=Role.EXPLORER)

        assert memory.locate(path).startswith("plan-x/")


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
    def test_the_false_discard_claim_is_gone(
        self, state: str, plan_dir: Path
    ) -> None:
        prompt = build_role_prompt(_role_request(state, plan_dir=plan_dir), spec=get_role_spec(state))

        assert "is discarded before this protocol sees it" not in prompt
        assert "The sentence is required" not in prompt

    @pytest.mark.parametrize("state", HarnessStates.ALL)
    def test_the_prompt_asks_for_exactly_one_json_object(
        self, state: str, plan_dir: Path
    ) -> None:
        prompt = build_role_prompt(_role_request(state, plan_dir=plan_dir), spec=get_role_spec(state))

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
