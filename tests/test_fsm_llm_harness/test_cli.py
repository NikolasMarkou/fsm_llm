"""
Tests for ``fsm_llm_harness.__main__`` -- the five-subcommand CLI.

The load-bearing properties, and why each is here:

* **Three exit codes, pinned separately.**  ``2`` is the source protocol's
  RESERVED "a HARD gate refused this step" code and may never be produced by
  anything else -- including argparse, whose own usage-error code is 2 and is
  therefore overridden (D-042).
* **WARN is not a failure.**  A real plan directory carries a dozen advisory
  findings; ``validate`` exiting non-zero on those would make it useless.
* **``--model`` precedence** is flag > ``$LLM_MODEL`` > package default, with a
  blank value at either tier treated as absent.
* **Fail closed.**  An unreadable plan directory is an error exit, never a
  silent success, and nothing here may CREATE the directory it was asked to
  look at.
* **``close`` is a dry run** unless ``--apply`` is passed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fsm_llm.constants import DEFAULT_LLM_MODEL
from fsm_llm_agents.definitions import AgentResult
from fsm_llm_harness import __main__ as cli
from fsm_llm_harness import harness as harness_module
from fsm_llm_harness.artifacts import StateDoc
from fsm_llm_harness.constants import (
    ArtifactNames,
    ContextKeys,
    Defaults,
    GateSlug,
    HarnessStates,
    PlanSchema,
    Severity,
)
from fsm_llm_harness.harness import HarnessAgent, Presentation, RevertDirective
from fsm_llm_harness.plan_validator import Issue

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _write_state(
    directory: Path,
    *,
    state: str = HarnessStates.EXPLORE,
    iteration: int = 0,
    step: str = "1",
    attempts: tuple[str, ...] = (),
) -> None:
    """Seed a plan directory's ``state.md`` through the real serializer."""
    directory.mkdir(parents=True, exist_ok=True)
    doc = StateDoc(
        state=state,
        iteration=iteration,
        current_step=step,
        fix_attempts=list(attempts),
    )
    (directory / ArtifactNames.STATE).write_text(doc.to_markdown(), encoding="utf-8")


@pytest.fixture
def plans_root(tmp_path: Path) -> Path:
    """A cross-plan root (the ``plans/`` directory) with nothing in it yet."""
    root = tmp_path / "plans"
    root.mkdir()
    return root


def _write_plan(directory: Path, goal: str) -> None:
    """Write a plan.md the real ``PlanDoc`` schema ACCEPTS, carrying *goal*.

    All 11 sections, in order: ``PlanDoc.REQUIRE_ORDER`` is True and a missing
    section is an ERROR there, not advice, so a two-section stub would exercise
    the rejection path instead of the reading path.
    """
    body = f"# Plan v1\n\n## {PlanSchema.SECTIONS[0]}\n{goal}\n"
    body += "".join(
        f"\n## {section}\nplaceholder\n" for section in PlanSchema.SECTIONS[1:]
    )
    (directory / ArtifactNames.PLAN).write_text(body, encoding="utf-8")


@pytest.fixture
def seeded_plan(plans_root: Path) -> Path:
    """A plan directory carrying a parseable ``state.md`` and nothing else."""
    directory = plans_root / "plan-2026-07-22T101500-1a2b3c4d"
    _write_state(directory)
    return directory


class _RecordingAgent:
    """A stand-in for ``HarnessAgent`` that records how the CLI configured it.

    It deliberately borrows the REAL ``_default_config``, so the assertion that
    the CLI replaces exactly one field of the driver profile is a statement
    about the shipped profile rather than about a test double.
    """

    calls: ClassVar[list[_RecordingAgent]] = []
    result = AgentResult(answer="done", success=True, final_context={})
    presentations: tuple[Presentation, ...] = ()
    reverts: tuple[RevertDirective, ...] = ()
    audit_issues: tuple[Issue, ...] | None = None

    _default_config = staticmethod(HarnessAgent._default_config)

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.goal: str | None = None
        self.initial_context: dict[str, Any] = {}
        _RecordingAgent.calls.append(self)

    def run(
        self, goal: str, initial_context: dict[str, Any] | None = None
    ) -> AgentResult:
        self.goal = goal
        self.initial_context = dict(initial_context or {})
        return type(self).result


@pytest.fixture
def fake_agent(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingAgent]:
    """Replace ``HarnessAgent`` so a run costs no LLM call."""
    _RecordingAgent.calls = []
    _RecordingAgent.result = AgentResult(answer="done", success=True, final_context={})
    _RecordingAgent.presentations = ()
    _RecordingAgent.reverts = ()
    _RecordingAgent.audit_issues = None
    monkeypatch.setattr(harness_module, "HarnessAgent", _RecordingAgent)
    return _RecordingAgent


# ---------------------------------------------------------------------------
# --help / --version / no subcommand
# ---------------------------------------------------------------------------


class TestEntryPoints:
    def test_help_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main_cli(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for command in ("new", "resume", "status", "validate", "close"):
            assert command in out

    def test_help_needs_no_package_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--help`` must not reach the optional-dep guard at all.

        The parser is built from stdlib only, so a broken install still gets a
        usage message instead of a traceback.
        """

        def _boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("--help imported a package module")

        monkeypatch.setattr(cli, "import_module", _boom)
        with pytest.raises(SystemExit) as exc:
            cli.main_cli(["--help"])
        assert exc.value.code == 0

    def test_module_level_imports_are_stdlib_only(self) -> None:
        """No harness/agents/litellm import may run at ``__main__`` import time."""
        import ast

        source = Path(cli.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in tree.body:  # module level only -- function bodies are lazy
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add((node.module or "").split(".")[0])
        assert imported <= {
            "__future__",
            "argparse",
            "collections",
            "datetime",
            "importlib",
            "os",
            "pathlib",
            "sys",
            "types",
            "typing",
        }

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        from fsm_llm_harness.__version__ import __version__

        assert cli.main_cli(["--version"]) == cli.EXIT_PASS
        assert __version__ in capsys.readouterr().out

    def test_no_subcommand_fails_closed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main_cli([]) == cli.EXIT_ERROR
        assert "usage:" in capsys.readouterr().err

    def test_subprocess_help(self) -> None:
        """The real ``python -m fsm_llm_harness --help`` entry point."""
        completed = subprocess.run(
            [sys.executable, "-m", "fsm_llm_harness", "--help"],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        assert "fsm-llm-harness" in completed.stdout

    def test_main_cli_is_the_console_script_symbol(self) -> None:
        """The name ``pyproject.toml`` will point at must exist and be callable."""
        assert callable(cli.main_cli)
        assert cli.__all__ == ["main_cli"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    @pytest.mark.parametrize(
        "argv",
        [
            ["nonsense"],
            ["validate"],  # missing the required plan_dir
            ["validate", "somewhere", "--nope"],
            ["new"],  # missing the required goal
            ["close", "somewhere", "--apply=maybe"],
        ],
    )
    def test_nonsense_is_rejected(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main_cli(argv)
        assert exc.value.code == cli.EXIT_ERROR

    def test_usage_error_never_exits_two(self) -> None:
        """D-042: exit 2 is reserved, so argparse's own code 2 is overridden."""
        with pytest.raises(SystemExit) as exc:
            cli.main_cli(["--not-a-flag"])
        assert exc.value.code != cli.EXIT_GATE
        assert exc.value.code == cli.EXIT_ERROR

    def test_every_subcommand_binds_a_handler(self) -> None:
        parser = cli.build_parser()
        for argv in (
            ["new", "g"],
            ["resume", "d"],
            ["status", "d"],
            ["validate", "d"],
            ["close", "d"],
        ):
            assert callable(parser.parse_args(argv).func)


# ---------------------------------------------------------------------------
# --model precedence
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(Defaults.ENV_MODEL, raising=False)
        assert cli.resolve_model(None) == Defaults.MODEL
        assert Defaults.MODEL == DEFAULT_LLM_MODEL

    def test_env_beats_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(Defaults.ENV_MODEL, "ollama_chat/from-env")
        assert cli.resolve_model(None) == "ollama_chat/from-env"

    def test_flag_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(Defaults.ENV_MODEL, "ollama_chat/from-env")
        assert cli.resolve_model("openai/from-flag") == "openai/from-flag"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_env_falls_through_to_default(
        self, monkeypatch: pytest.MonkeyPatch, blank: str
    ) -> None:
        monkeypatch.setenv(Defaults.ENV_MODEL, blank)
        assert cli.resolve_model(None) == Defaults.MODEL

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_flag_falls_through_to_env(
        self, monkeypatch: pytest.MonkeyPatch, blank: str
    ) -> None:
        monkeypatch.setenv(Defaults.ENV_MODEL, "ollama_chat/from-env")
        assert cli.resolve_model(blank) == "ollama_chat/from-env"

    def test_values_are_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(Defaults.ENV_MODEL, raising=False)
        assert cli.resolve_model("  openai/gpt-4o-mini  ") == "openai/gpt-4o-mini"

    def test_flag_reaches_the_driver_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_agent: type[_RecordingAgent],
        plans_root: Path,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(Defaults.ENV_MODEL, "ollama_chat/from-env")
        code = cli.main_cli(
            [
                "new",
                "add a retry",
                "--plans-dir",
                str(plans_root),
                "--workspace",
                str(tmp_path / "ws"),
                "--model",
                "openai/gpt-4o-mini",
            ]
        )
        assert code == cli.EXIT_PASS
        config = fake_agent.calls[0].kwargs["config"]
        assert config.model == "openai/gpt-4o-mini"

    def test_env_reaches_the_driver_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_agent: type[_RecordingAgent],
        plans_root: Path,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv(Defaults.ENV_MODEL, "ollama_chat/from-env")
        cli.main_cli(
            [
                "new",
                "add a retry",
                "--plans-dir",
                str(plans_root),
                "--workspace",
                str(tmp_path / "ws"),
            ]
        )
        assert fake_agent.calls[0].kwargs["config"].model == "ollama_chat/from-env"

    def test_only_the_model_field_is_replaced(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_agent: type[_RecordingAgent],
        plans_root: Path,
        tmp_path: Path,
    ) -> None:
        """The other five fields of the driver profile are NOT restated by the CLI."""
        monkeypatch.delenv(Defaults.ENV_MODEL, raising=False)
        cli.main_cli(
            [
                "new",
                "g",
                "--plans-dir",
                str(plans_root),
                "--workspace",
                str(tmp_path / "ws"),
                "--model",
                "openai/gpt-4o-mini",
            ]
        )
        config = fake_agent.calls[0].kwargs["config"]
        profile = HarnessAgent._default_config()
        assert config.model_dump(exclude={"model"}) == profile.model_dump(
            exclude={"model"}
        )


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------


class TestNew:
    def test_create_only_mints_and_seeds(
        self, plans_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            cli.main_cli(
                ["new", "add a retry", "--plans-dir", str(plans_root), "--create-only"]
            )
            == cli.EXIT_PASS
        )
        minted = [child for child in plans_root.iterdir() if child.is_dir()]
        assert len(minted) == 1
        state = minted[0] / ArtifactNames.STATE
        assert state.is_file()
        doc = StateDoc.from_markdown(state.read_text(encoding="utf-8"))
        assert doc.state == HarnessStates.INITIAL
        assert doc.iteration == 0
        assert minted[0].name in capsys.readouterr().out

    def test_create_only_calls_no_agent(
        self, plans_root: Path, fake_agent: type[_RecordingAgent]
    ) -> None:
        cli.main_cli(["new", "g", "--plans-dir", str(plans_root), "--create-only"])
        assert fake_agent.calls == []

    def test_run_passes_both_roots_and_the_goal(
        self,
        plans_root: Path,
        tmp_path: Path,
        fake_agent: type[_RecordingAgent],
    ) -> None:
        workspace = tmp_path / "ws"
        assert (
            cli.main_cli(
                [
                    "new",
                    "add a retry to the uploader",
                    "--plans-dir",
                    str(plans_root),
                    "--workspace",
                    str(workspace),
                ]
            )
            == cli.EXIT_PASS
        )
        agent = fake_agent.calls[0]
        assert agent.goal == "add a retry to the uploader"
        minted = next(child for child in plans_root.iterdir() if child.is_dir())
        assert agent.initial_context[ContextKeys.PLAN_DIR] == str(minted.resolve())
        assert agent.initial_context[ContextKeys.WORKSPACE_ROOT] == str(
            workspace.resolve()
        )

    def test_two_news_mint_two_directories(self, plans_root: Path) -> None:
        for _ in range(2):
            cli.main_cli(["new", "g", "--plans-dir", str(plans_root), "--create-only"])
        assert len({child.name for child in plans_root.iterdir()}) == 2

    def test_unusable_plans_root_is_an_error_exit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A FILE where the plans root should be: reported, never a traceback.

        ``PlanMemory``'s own ``mkdir`` raises ``NotADirectoryError`` here, which
        is an ``OSError`` and NOT a ``HarnessError`` -- catching only the latter
        is the fail-open shape ``_failures()`` exists to close.
        """
        blocked = tmp_path / "plans-is-a-file"
        blocked.write_text("not a directory\n", encoding="utf-8")
        assert (
            cli.main_cli(["new", "g", "--plans-dir", str(blocked), "--create-only"])
            == cli.EXIT_ERROR
        )
        assert "could not create a plan directory" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


class TestResume:
    def test_happy_path(
        self, seeded_plan: Path, tmp_path: Path, fake_agent: type[_RecordingAgent]
    ) -> None:
        code = cli.main_cli(
            [
                "resume",
                str(seeded_plan),
                "--goal",
                "keep going",
                "--workspace",
                str(tmp_path / "ws"),
            ]
        )
        assert code == cli.EXIT_PASS
        assert fake_agent.calls[0].goal == "keep going"
        assert fake_agent.calls[0].initial_context[ContextKeys.PLAN_DIR] == str(
            seeded_plan.resolve()
        )

    def test_absent_directory_is_the_reserved_gate_code(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "plans" / "not-there"
        assert cli.main_cli(["resume", str(missing), "--goal", "g"]) == cli.EXIT_GATE
        assert GateSlug.NO_PLAN in capsys.readouterr().err
        assert not missing.exists(), "resume must not CREATE the directory it opens"

    def test_directory_without_state_md_is_no_plan(
        self, plans_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = plans_root / "plan-2026-07-22T101500-deadbeef"
        empty.mkdir()
        assert cli.main_cli(["resume", str(empty), "--goal", "g"]) == cli.EXIT_GATE
        assert GateSlug.NO_PLAN in capsys.readouterr().err

    def test_unparseable_state_md_is_no_plan(
        self, seeded_plan: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (seeded_plan / ArtifactNames.STATE).write_text("garbage\n", encoding="utf-8")
        assert (
            cli.main_cli(["resume", str(seeded_plan), "--goal", "g"]) == cli.EXIT_GATE
        )
        assert GateSlug.NO_PLAN in capsys.readouterr().err

    def test_goal_is_read_from_plan_md(
        self, seeded_plan: Path, tmp_path: Path, fake_agent: type[_RecordingAgent]
    ) -> None:
        _write_plan(seeded_plan, "Make the uploader retry.")
        assert (
            cli.main_cli(["resume", str(seeded_plan), "--workspace", str(tmp_path)])
            == cli.EXIT_PASS
        )
        assert fake_agent.calls[0].goal == "Make the uploader retry."

    def test_a_rejected_plan_md_records_no_goal(
        self,
        seeded_plan: Path,
        fake_agent: type[_RecordingAgent],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A plan.md the schema REJECTS must not have its Goal lifted out of it.

        Two sections is a plan.md ``PlanDoc`` refuses.  Reading a goal from it
        anyway would mean dispatching a whole run against text the validator
        fails, so the honest answer is "no goal on record".
        """
        (seeded_plan / ArtifactNames.PLAN).write_text(
            "# Plan v1\n\n## Goal\nLifted from a broken plan.\n\n"
            "## Problem Statement\nx\n",
            encoding="utf-8",
        )
        assert cli.main_cli(["resume", str(seeded_plan)]) == cli.EXIT_ERROR
        assert "--goal" in capsys.readouterr().err
        assert fake_agent.calls == []

    def test_explicit_goal_beats_plan_md(
        self, seeded_plan: Path, tmp_path: Path, fake_agent: type[_RecordingAgent]
    ) -> None:
        _write_plan(seeded_plan, "Recorded goal.")
        cli.main_cli(
            [
                "resume",
                str(seeded_plan),
                "--goal",
                "override",
                "--workspace",
                str(tmp_path),
            ]
        )
        assert fake_agent.calls[0].goal == "override"

    def test_no_goal_anywhere_is_an_error_exit(
        self, seeded_plan: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main_cli(["resume", str(seeded_plan)]) == cli.EXIT_ERROR
        assert "--goal" in capsys.readouterr().err

    def test_no_goal_does_not_dispatch(
        self, seeded_plan: Path, fake_agent: type[_RecordingAgent]
    ) -> None:
        cli.main_cli(["resume", str(seeded_plan)])
        assert fake_agent.calls == []


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_reports_the_recorded_position(
        self, seeded_plan: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_state(seeded_plan, state=HarnessStates.EXECUTE, iteration=2, step="7")
        assert cli.main_cli(["status", str(seeded_plan)]) == cli.EXIT_PASS
        out = capsys.readouterr().out
        assert HarnessStates.EXECUTE in out
        assert "iteration: 2" in out
        assert "step:      7" in out
        assert seeded_plan.name in out

    def test_a_plan_in_explore_still_passes(self, seeded_plan: Path) -> None:
        """D-043: `wrong-state` is the dispatcher's question, not the reporter's."""
        _write_state(seeded_plan, state=HarnessStates.EXPLORE)
        assert cli.main_cli(["status", str(seeded_plan)]) == cli.EXIT_PASS

    @pytest.mark.parametrize(
        "state",
        [
            HarnessStates.EXPLORE,
            HarnessStates.PLAN,
            HarnessStates.EXECUTE,
            HarnessStates.REFLECT,
            HarnessStates.PIVOT,
            HarnessStates.CLOSE,
        ],
    )
    def test_every_state_passes_when_nothing_is_capped(
        self, seeded_plan: Path, state: str
    ) -> None:
        _write_state(seeded_plan, state=state)
        assert cli.main_cli(["status", str(seeded_plan)]) == cli.EXIT_PASS

    def test_leash_cap_is_the_reserved_gate_code(
        self, seeded_plan: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        attempts = tuple(
            f"Step 7, attempt {n}: failed"
            for n in range(1, Defaults.MAX_FIX_ATTEMPTS + 1)
        )
        _write_state(seeded_plan, state=HarnessStates.EXECUTE, attempts=attempts)
        assert cli.main_cli(["status", str(seeded_plan)]) == cli.EXIT_GATE
        assert GateSlug.LEASH_CAP in capsys.readouterr().out

    def test_iteration_cap_is_the_reserved_gate_code(
        self, seeded_plan: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_state(
            seeded_plan,
            state=HarnessStates.EXECUTE,
            iteration=Defaults.ITERATION_HARD_CAP,
        )
        assert cli.main_cli(["status", str(seeded_plan)]) == cli.EXIT_GATE
        assert GateSlug.ITERATION_CAP in capsys.readouterr().out

    def test_absent_directory_is_no_plan_and_is_not_created(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "plans" / "not-there"
        assert cli.main_cli(["status", str(missing)]) == cli.EXIT_GATE
        assert GateSlug.NO_PLAN in capsys.readouterr().err
        assert not missing.exists()

    def test_status_writes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, seeded_plan: Path
    ) -> None:
        """`status` must not touch the write path at all.

        A byte comparison is NOT enough here and the omission is the interesting
        part: re-serializing the ``StateDoc`` it just parsed reproduces
        ``state.md`` verbatim, so a spurious ``save_run_state`` leaves identical
        bytes and passes a content assertion while having genuinely rewritten a
        file the command promised only to read (mutation M19 escaped exactly
        that test).  Spying on the single atomic-write chokepoint names the
        property instead of a proxy for it.
        """
        from fsm_llm_harness import storage

        writes: list[str] = []
        monkeypatch.setattr(
            storage,
            "_atomic_write_text",
            lambda target, content, *, artifact: writes.append(artifact),
        )
        before = {
            path.name: path.read_bytes()
            for path in seeded_plan.rglob("*")
            if path.is_file()
        }
        cli.main_cli(["status", str(seeded_plan)])
        after = {
            path.name: path.read_bytes()
            for path in seeded_plan.rglob("*")
            if path.is_file()
        }
        assert writes == []
        assert before == after


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_warnings_only_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, seeded_plan: Path
    ) -> None:
        from fsm_llm_harness import plan_validator

        warnings = [
            Issue(severity=Severity.WARNING, check="progress", message="advisory"),
            Issue(severity=Severity.INFO, check="plan", message="note"),
        ]
        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: warnings)
        assert cli.main_cli(["validate", str(seeded_plan)]) == cli.EXIT_PASS

    def test_one_error_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, seeded_plan: Path
    ) -> None:
        from fsm_llm_harness import plan_validator

        mixed = [
            Issue(severity=Severity.WARNING, check="progress", message="advisory"),
            Issue(severity=Severity.ERROR, check="plan", message="plan.md is missing"),
        ]
        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: mixed)
        assert cli.main_cli(["validate", str(seeded_plan)]) == cli.EXIT_ERROR

    def test_validate_never_returns_the_gate_code(
        self, monkeypatch: pytest.MonkeyPatch, seeded_plan: Path
    ) -> None:
        from fsm_llm_harness import plan_validator

        errors = [Issue(severity=Severity.ERROR, check="state", message="broken")]
        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: errors)
        assert cli.main_cli(["validate", str(seeded_plan)]) != cli.EXIT_GATE

    def test_absent_directory_is_an_error_exit(self, tmp_path: Path) -> None:
        missing = tmp_path / "plans" / "not-there"
        assert cli.main_cli(["validate", str(missing)]) == cli.EXIT_ERROR
        assert not missing.exists(), "validate must not CREATE the directory it audits"

    def test_real_audit_on_a_bare_plan_directory_reports_errors(
        self, seeded_plan: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No monkeypatch: the real `audit()` on a directory missing plan.md."""
        assert cli.main_cli(["validate", str(seeded_plan)]) == cli.EXIT_ERROR
        out = capsys.readouterr().out
        assert "errors," in out
        assert Severity.ERROR in out

    def test_workspace_flag_reaches_audit(
        self, monkeypatch: pytest.MonkeyPatch, seeded_plan: Path, tmp_path: Path
    ) -> None:
        from fsm_llm_harness import plan_validator

        seen: dict[str, Any] = {}

        def _spy(plan_dir: Any, *, workspace_root: Any = None) -> list[Issue]:
            seen["plan_dir"] = plan_dir
            seen["workspace_root"] = workspace_root
            return []

        monkeypatch.setattr(plan_validator, "audit", _spy)
        cli.main_cli(["validate", str(seeded_plan), "--workspace", str(tmp_path)])
        assert seen["workspace_root"] == str(tmp_path)

    def test_no_workspace_means_no_anchor_scan(
        self, monkeypatch: pytest.MonkeyPatch, seeded_plan: Path
    ) -> None:
        from fsm_llm_harness import plan_validator

        seen: dict[str, Any] = {}

        def _spy(plan_dir: Any, *, workspace_root: Any = None) -> list[Issue]:
            seen["workspace_root"] = workspace_root
            return []

        monkeypatch.setattr(plan_validator, "audit", _spy)
        cli.main_cli(["validate", str(seeded_plan)])
        assert seen["workspace_root"] is None


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def _seed_cross_plan(plans_root: Path, *, lessons_lines: int) -> None:
    """Write a LESSONS.md whose bullet count is controllable."""
    bullets = "\n".join(
        f"- [I:{1 + (n % 4)}] lesson number {n}" for n in range(lessons_lines)
    )
    (plans_root / ArtifactNames.LESSONS).write_text(
        f"# Lessons\n\n## Process\n{bullets}\n", encoding="utf-8"
    )


class TestClose:
    def test_refuses_while_the_audit_has_errors(
        self, seeded_plan: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main_cli(["close", str(seeded_plan)]) == cli.EXIT_ERROR
        assert "REFUSED" in capsys.readouterr().err

    def test_refusal_performs_no_housekeeping(
        self, monkeypatch: pytest.MonkeyPatch, seeded_plan: Path
    ) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            cli, "_housekeeping", lambda *a, **k: called.append("ran") or True
        )
        cli.main_cli(["close", str(seeded_plan)])
        assert called == []

    def test_clean_directory_reports_the_policies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seeded_plan: Path,
        plans_root: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from fsm_llm_harness import plan_validator

        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: [])
        _seed_cross_plan(plans_root, lessons_lines=10)
        assert cli.main_cli(["close", str(seeded_plan)]) == cli.EXIT_PASS
        out = capsys.readouterr().out
        assert ArtifactNames.LESSONS in out
        assert "dry run" in out

    def test_dry_run_writes_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seeded_plan: Path,
        plans_root: Path,
    ) -> None:
        from fsm_llm_harness import plan_validator

        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: [])
        _seed_cross_plan(plans_root, lessons_lines=Defaults.LESSONS_LINE_CAP + 50)
        lessons = plans_root / ArtifactNames.LESSONS
        before = lessons.read_bytes()
        assert cli.main_cli(["close", str(seeded_plan)]) == cli.EXIT_PASS
        assert lessons.read_bytes() == before
        assert not (plans_root / ArtifactNames.LESSONS_ARCHIVE).exists()

    def test_apply_performs_the_eviction(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seeded_plan: Path,
        plans_root: Path,
    ) -> None:
        from fsm_llm_harness import plan_validator

        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: [])
        _seed_cross_plan(plans_root, lessons_lines=Defaults.LESSONS_LINE_CAP + 50)
        lessons = plans_root / ArtifactNames.LESSONS
        before = len(lessons.read_text(encoding="utf-8").splitlines())
        assert cli.main_cli(["close", str(seeded_plan), "--apply"]) == cli.EXIT_PASS
        after = len(lessons.read_text(encoding="utf-8").splitlines())
        assert after < before
        assert (plans_root / ArtifactNames.LESSONS_ARCHIVE).is_file()

    def test_apply_uses_the_archivist_role(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seeded_plan: Path,
        plans_root: Path,
    ) -> None:
        """D-044: the cross-plan tier is ARCHIVIST-owned; ORCHESTRATOR is refused."""
        from fsm_llm_harness import plan_validator
        from fsm_llm_harness.constants import Role

        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: [])
        _seed_cross_plan(plans_root, lessons_lines=10)
        seen: list[str] = []
        original = cli._housekeeping

        def _spy(directory: Any, *, apply: bool) -> bool:
            seen.append(directory.role)
            return original(directory, apply=apply)

        monkeypatch.setattr(cli, "_housekeeping", _spy)
        cli.main_cli(["close", str(seeded_plan), "--apply"])
        assert seen == [Role.ARCHIVIST]

    def test_absent_cross_plan_files_are_reported_not_fatal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seeded_plan: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from fsm_llm_harness import plan_validator

        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: [])
        assert cli.main_cli(["close", str(seeded_plan)]) == cli.EXIT_PASS
        assert f"{ArtifactNames.LESSONS}: absent" in capsys.readouterr().out

    def test_unreadable_cross_plan_file_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seeded_plan: Path,
        plans_root: Path,
    ) -> None:
        from fsm_llm_harness import plan_validator

        monkeypatch.setattr(plan_validator, "audit", lambda *a, **k: [])
        (plans_root / ArtifactNames.LESSONS).write_text(
            "no h1 here, so this cannot parse\n", encoding="utf-8"
        )
        assert cli.main_cli(["close", str(seeded_plan)]) == cli.EXIT_ERROR


# ---------------------------------------------------------------------------
# Run reporting and the exit-code contract
# ---------------------------------------------------------------------------


class TestRunReporting:
    def _run(
        self,
        fake_agent: type[_RecordingAgent],
        plans_root: Path,
        tmp_path: Path,
    ) -> int:
        return cli.main_cli(
            [
                "new",
                "g",
                "--plans-dir",
                str(plans_root),
                "--workspace",
                str(tmp_path / "ws"),
            ]
        )

    def test_success_exits_zero(
        self, fake_agent: type[_RecordingAgent], plans_root: Path, tmp_path: Path
    ) -> None:
        assert self._run(fake_agent, plans_root, tmp_path) == cli.EXIT_PASS

    def test_failed_run_exits_one(
        self, fake_agent: type[_RecordingAgent], plans_root: Path, tmp_path: Path
    ) -> None:
        fake_agent.result = AgentResult(answer="stalled", success=False)
        assert self._run(fake_agent, plans_root, tmp_path) == cli.EXIT_ERROR

    @pytest.mark.parametrize("slug", list(GateSlug.ORDER))
    def test_every_hard_slug_exits_two(
        self,
        fake_agent: type[_RecordingAgent],
        plans_root: Path,
        tmp_path: Path,
        slug: str,
    ) -> None:
        fake_agent.result = AgentResult(
            answer="halted",
            success=False,
            final_context={ContextKeys.LAST_GATE_SLUG: slug},
        )
        assert self._run(fake_agent, plans_root, tmp_path) == cli.EXIT_GATE

    def test_a_non_gate_slug_does_not_exit_two(
        self, fake_agent: type[_RecordingAgent], plans_root: Path, tmp_path: Path
    ) -> None:
        """`explore-cap` is a driver halt, not a pre-step gate slug."""
        fake_agent.result = AgentResult(
            answer="halted",
            success=False,
            final_context={ContextKeys.LAST_GATE_SLUG: GateSlug.EXPLORE_CAP},
        )
        assert self._run(fake_agent, plans_root, tmp_path) == cli.EXIT_ERROR

    def test_audit_errors_exit_one(
        self, fake_agent: type[_RecordingAgent], plans_root: Path, tmp_path: Path
    ) -> None:
        fake_agent.audit_issues = (
            Issue(severity=Severity.ERROR, check="plan", message="plan.md is missing"),
        )
        assert self._run(fake_agent, plans_root, tmp_path) == cli.EXIT_ERROR

    def test_audit_warnings_do_not_fail_a_successful_run(
        self, fake_agent: type[_RecordingAgent], plans_root: Path, tmp_path: Path
    ) -> None:
        fake_agent.audit_issues = (
            Issue(severity=Severity.WARNING, check="progress", message="advisory"),
        )
        assert self._run(fake_agent, plans_root, tmp_path) == cli.EXIT_PASS

    def test_gate_slug_outranks_audit_errors(
        self, fake_agent: type[_RecordingAgent], plans_root: Path, tmp_path: Path
    ) -> None:
        fake_agent.audit_issues = (
            Issue(severity=Severity.ERROR, check="plan", message="plan.md is missing"),
        )
        fake_agent.result = AgentResult(
            answer="halted",
            success=False,
            final_context={ContextKeys.LAST_GATE_SLUG: GateSlug.LEASH_CAP},
        )
        assert self._run(fake_agent, plans_root, tmp_path) == cli.EXIT_GATE

    def test_presentations_and_reverts_are_printed(
        self,
        fake_agent: type[_RecordingAgent],
        plans_root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fake_agent.presentations = (
            Presentation(
                name="PC-EXECUTE-LEASH",
                fields={},
                missing_floor=(),
                block="### PC-EXECUTE-LEASH\nstep 7 failed twice",
            ),
        )
        fake_agent.reverts = (
            RevertDirective(
                root=str(tmp_path / "ws"),
                exclude=("plans",),
                commands=("git checkout -- .",),
            ),
        )
        self._run(fake_agent, plans_root, tmp_path)
        out = capsys.readouterr().out
        assert "PC-EXECUTE-LEASH" in out
        assert "revert (NOT executed)" in out


# ---------------------------------------------------------------------------
# The optional-dependency guard
# ---------------------------------------------------------------------------


class TestOptionalDependencyGuard:
    def test_import_error_becomes_exit_one_with_a_hint(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def _boom(name: str, package: str | None = None) -> Any:
            raise ImportError("No module named 'litellm'")

        monkeypatch.setattr(cli, "import_module", _boom)
        with pytest.raises(SystemExit) as exc:
            cli.main_cli(["status", "somewhere"])
        assert exc.value.code == cli.EXIT_ERROR
        err = capsys.readouterr().err
        assert "fsm-llm[harness]" in err

    def test_guard_never_produces_the_gate_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(name: str, package: str | None = None) -> Any:
            raise ImportError("nope")

        monkeypatch.setattr(cli, "import_module", _boom)
        for argv in (["new", "g"], ["resume", "d"], ["status", "d"], ["validate", "d"]):
            with pytest.raises(SystemExit) as exc:
                cli.main_cli(argv)
            assert exc.value.code != cli.EXIT_GATE


# ---------------------------------------------------------------------------
# The package's public surface (step 12's second half)
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_all_is_a_single_literal_list(self) -> None:
        """Repo rule: one literal ``__all__``, no dynamic extend/append."""
        import ast

        import fsm_llm_harness

        tree = ast.parse(Path(fsm_llm_harness.__file__).read_text(encoding="utf-8"))
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ]
        assert len(assignments) == 1
        assert isinstance(assignments[0].value, ast.List)
        assert "__all__" not in {
            getattr(getattr(node, "func", None), "attr", None)
            for node in ast.walk(tree)
        }

    def test_every_exported_name_resolves(self) -> None:
        import fsm_llm_harness

        missing = [
            name
            for name in fsm_llm_harness.__all__
            if not hasattr(fsm_llm_harness, name)
        ]
        assert missing == []

    def test_no_duplicates(self) -> None:
        import fsm_llm_harness

        assert len(fsm_llm_harness.__all__) == len(set(fsm_llm_harness.__all__))

    @pytest.mark.parametrize(
        "name",
        [
            # The tier this plan built, which is the point of the finalisation.
            "PlanDirectory",
            "RunState",
            "mint_plan_id",
            "audit",
            "pre_step_gate",
            "GateResult",
            "Issue",
            "StateDoc",
            "PlanDoc",
            "DecisionsDoc",
            "PRESENTATION_CONTRACTS",
            "Presentation",
            "RevertDirective",
            # The pre-existing surface, which must not have been dropped.
            "HarnessAgent",
            "OWNERSHIP",
            "PlanMemory",
            "Workspace",
            "HarnessConfinementError",
        ],
    )
    def test_expected_names_are_exported(self, name: str) -> None:
        import fsm_llm_harness

        assert name in fsm_llm_harness.__all__

    def test_main_cli_is_not_exported_from_the_package(self) -> None:
        """Importing ``__main__`` from ``__init__`` double-imports it under -m."""
        import fsm_llm_harness

        assert "main_cli" not in fsm_llm_harness.__all__
