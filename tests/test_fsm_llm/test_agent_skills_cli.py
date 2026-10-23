"""Tests for ``fsm-llm-skills`` (``fsm_llm.agent_skills_cli``): commands, exit
codes, output hygiene, wiring. Control, bidi and non-ASCII characters are built
with ``chr`` or YAML escapes: this file stays pure ASCII."""

from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import fsm_llm.agent_skills_cli as skills_cli
from fsm_llm.agent_skills import AgentSkill, SkillCatalog, SkillIssue
from fsm_llm.agent_skills_cli import main
from fsm_llm.constants import (
    SKILL_CATALOG_DESCRIPTION_CHARS,
    SKILL_INCOMPLETE_NOTE,
    SKILL_LIST_MAX_ITEMS,
    SKILL_READ_PAGE_CHARS,
)

_REPO = Path(__file__).resolve().parents[2]
_ESC, _BIDI, _ZWSP, _CJK = chr(0x1B), chr(0x202E), chr(0x200B), chr(0x7F51)
_HOSTILE = f"a{_ESC}[31mred{_ESC}]0;title{chr(7)}b{_BIDI}c{_ZWSP}d"
_TINY = "1e-9"  # a load budget that is spent before the first skill


def _yaml(text: str) -> str:
    """A double-quoted YAML scalar for *text* using escapes only (ASCII source)."""
    body = "".join(
        f"\\u{ord(c):04x}" if ord(c) > 126 or ord(c) < 32 else c for c in text
    )
    return '"' + body.replace('"', '\\"') + '"'


def _skill(
    root: Path,
    name: str,
    description: str = "Does a thing. Use when asked.",
    body: str = "Body line.\n",
    extra: str = "",
    files: dict[str, str] | None = None,
) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    front = f"name: {name}\ndescription: {description}\n{extra}"
    (folder / "SKILL.md").write_text(f"---\n{front}---\n{body}", encoding="utf-8")
    for rel, text in (files or {}).items():
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return folder


def _run(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def clean(tmp_path: Path) -> Path:
    """A skills directory with two clean skills, one with bundled files."""
    root = tmp_path / "skills"
    _skill(root, "alpha", files={"references/guide.md": "Guide text.\n", "x.txt": "x"})
    _skill(root, "beta", description="Second skill.")
    return root


def _stub_catalog(
    monkeypatch: pytest.MonkeyPatch, *, issues: tuple[SkillIssue, ...] = ()
) -> None:
    """Make the CLI load a catalog whose strings were built WITHOUT the models'
    own control-stripping, so only the CLI's printing can keep them off the
    terminal."""
    skill = AgentSkill.model_construct(
        name=_HOSTILE, description=_HOSTILE, resources=(f"r{_ESC}[31m.md",)
    )
    catalog = SkillCatalog([skill], issues=issues)
    monkeypatch.setattr(skills_cli, "load_skills", lambda *a, **k: catalog)


class TestValidate:
    def test_a_clean_tree_prints_nothing_and_summarizes_on_stderr(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "validate", clean)

        assert (code, out) == (0, "")
        assert err.strip() == "2 skills, 0 errors, 0 warnings, 0 info"

    def test_an_error_issue_prints_severity_code_path_message_and_exits_1(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (clean / "broken").mkdir()
        (clean / "broken" / "SKILL.md").write_text("---\nname: broken\n---\nx\n")

        code, out, err = _run(capsys, "validate", clean)

        assert code == 1
        severity, issue_code, path, message = out.splitlines()[0].split(" ", 3)
        assert (severity, issue_code, path) == (
            "error",
            "description-missing",
            "broken/SKILL.md",
        )
        assert message
        assert "1 errors" in err and "2 skills" in err

    @pytest.mark.parametrize(
        ("extra", "body", "code", "strict", "expected"),
        [
            ("", "x\n", "", False, 0),
            ("version: 1\n", "x\n", "key-unknown", False, 0),
            ("version: 1\n", "x\n", "key-unknown", True, 1),  # --strict boundary
            ("", "l\n" * 600, "body-long", False, 0),
            ("", "l\n" * 600, "body-long", True, 0),  # info never trips --strict
        ],
    )
    def test_the_strict_boundary_is_warnings_in_info_out(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        extra: str,
        body: str,
        code: str,
        strict: bool,
        expected: int,
    ) -> None:
        _skill(tmp_path, "demo", extra=extra, body=body)
        flags = ["--strict"] if strict else []

        got, out, _ = _run(capsys, "validate", tmp_path, *flags)

        assert got == expected
        assert code in out and bool(out) == bool(code)

    def test_a_directory_with_no_skill_warns_and_strict_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = _run(capsys, "validate", tmp_path)

        assert code == 0 and out.startswith("warning no-skills ")
        assert _run(capsys, "validate", tmp_path, "--strict")[0] == 1

    def test_two_roots_with_a_clashing_name_keep_the_first(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = tmp_path / "one"
        second = tmp_path / "two"
        _skill(first, "dup", description="From the first root.")
        _skill(second, "dup", description="From the second root.")

        code, out, _ = _run(capsys, "validate", first, second)
        assert code == 0 and "name-duplicate" in out

        _, listing, _ = _run(capsys, "list", first, second)
        assert "From the first root." in listing
        assert "From the second root." not in listing

    def test_a_spent_budget_is_an_error_with_a_hint_and_exits_1(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "validate", clean, "--max-seconds", _TINY)

        assert code == 1
        assert "error load-deadline" in out and "raise with --max-seconds" in out
        assert "loaded incompletely" not in err  # validate reports the issue itself


class TestListShow:
    def test_list_prints_name_capped_description_and_file_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _skill(tmp_path, "alpha", files={"a.md": "1", "b/c.md": "2"})
        _skill(tmp_path, "wordy", description="w" * 1000)

        code, out, err = _run(capsys, "list", tmp_path)

        rows = [line.split("\t") for line in out.splitlines()]
        assert (code, err) == (0, "")
        assert rows[0] == ["alpha", "Does a thing. Use when asked.", "2 files"]
        assert rows[1][0] == "wordy" and rows[1][2] == "0 files"
        assert len(rows[1][1]) == SKILL_CATALOG_DESCRIPTION_CHARS

    def test_list_with_no_skills_says_so_on_stderr_and_exits_0(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "list", tmp_path)

        assert (code, out) == (0, "")
        assert err == "no skills found\n"
        _skill(tmp_path, "bad", extra="")  # a skill that is skipped (name differs)
        (tmp_path / "bad" / "SKILL.md").write_text("---\nname: BAD\n---\n")
        assert _run(capsys, "list", tmp_path) == (0, "", "no skills found\n")

    def test_show_prints_a_header_the_file_list_and_the_body(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "show", "alpha", clean)

        assert (code, err) == (0, "")
        lines = out.splitlines()
        assert lines[0] == "name: alpha"
        assert lines[1] == "description: Does a thing. Use when asked."
        assert lines[2] == "files (2): references/guide.md, x.txt"
        assert out.rstrip().endswith("Body line.")

    @pytest.mark.parametrize("file", [None, "notes.txt"])
    def test_a_long_body_or_file_is_paged_and_the_pages_add_up(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], file: str | None
    ) -> None:
        text = "".join(f"line {i:04d}\n" for i in range(600))
        _skill(tmp_path, "long", body=text, files={"notes.txt": text})
        flags = [] if file is None else ["--file", file]
        pages, offset = [], 0
        for _ in range(20):  # bounded: a broken --offset must fail, not hang
            args = ["show", "long", tmp_path, *flags, "--offset", str(offset)]
            _, out, _ = _run(capsys, *args)
            head, _, rest = out.partition("\n\n")
            assert head.startswith("name: long")
            page, sep, tail = rest.rpartition("next offset: ")
            pages.append((page if sep else rest).rstrip("\n"))
            if not sep:
                break
            offset = int(tail)
        else:
            pytest.fail("the last page was never reached")
        assert len(pages) > 1 and len(pages[0]) <= SKILL_READ_PAGE_CHARS
        assert "".join(pages).replace("\n", "") == text.replace("\n", "").strip()

    def test_show_reads_a_bundled_file_page(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = _run(
            capsys, "show", "alpha", clean, "--file", "references/guide.md"
        )

        assert code == 0
        assert "file: references/guide.md" in out and "Guide text." in out
        assert "Body line." not in out

    @pytest.mark.parametrize("target", ["references", "references/"])
    def test_a_directory_is_listed_like_the_skill_read_tool(
        self, clean: Path, capsys: pytest.CaptureFixture[str], target: str
    ) -> None:
        code, out, err = _run(capsys, "show", "alpha", clean, "--file", target)

        assert (code, err) == (0, "")
        head, _, listing = out.partition("\n\n")
        assert f"files under {target!r} (1)" in head and "file:" not in head
        assert listing.strip() == "references/guide.md"
        assert "Guide text." not in out and "next offset" not in out

    def test_the_empty_path_lists_every_file_of_a_skill(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = _run(capsys, "show", "alpha", clean, "--file", "")

        assert code == 0 and "files under '' (2)" in out
        assert out.rstrip().endswith("references/guide.md\nx.txt")

    def test_an_empty_path_on_a_skill_without_files_stays_refused(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "show", "beta", clean, "--file", "")

        assert (code, out) == (1, "")
        assert err.strip().startswith("fsm-llm-skills: error: ")
        assert len(err.splitlines()) == 1

    def test_a_long_listing_pages_with_offset(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        names = [f"refs/file-{n:04d}-{'x' * 30}.md" for n in range(120)]
        _skill(tmp_path, "many", files={name: "." for name in names})
        seen, offset = [], 0
        for _ in range(20):  # bounded: a broken --offset must fail, not hang
            args = ["show", "many", tmp_path, "--file", "refs", "--offset", str(offset)]
            _, out, _ = _run(capsys, *args)
            head, _, rest = out.partition("\n\n")
            assert "files under 'refs' (120)" in head
            body, sep, tail = rest.rpartition("next offset: ")
            seen.append(body.rstrip("\n") if sep else rest.rstrip("\n"))
            if not sep:
                break
            offset = int(tail)
        else:
            pytest.fail("the last page was never reached")
        assert len(seen) > 1 and len(seen[0]) <= SKILL_READ_PAGE_CHARS
        assert "".join(seen).replace("\n", "") == "".join(names)

    @pytest.mark.parametrize(
        "target",
        [
            "../beta/SKILL.md",
            "/etc/passwd",
            ".env",
            "nope/",
            "a\\b",
            "link.txt",
        ],
    )
    def test_a_refused_file_is_one_clean_line_and_exit_1(
        self,
        clean: Path,
        capsys: pytest.CaptureFixture[str],
        target: str,
    ) -> None:
        (clean / "alpha" / ".env").write_text("SECRET=1")
        outside = clean.parent / "outside.txt"
        outside.write_text("SECRET=outside")
        (clean / "alpha" / "link.txt").symlink_to(outside)

        code, out, err = _run(capsys, "show", "alpha", clean, "--file", target)

        assert (code, out) == (1, "")
        assert err.strip().startswith("fsm-llm-skills: error: ")
        assert len(err.splitlines()) == 1 and "SECRET" not in err

    def test_an_unknown_skill_is_a_line_with_a_close_name_and_exit_1(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "show", "alpah", clean)

        assert (code, out) == (1, "")
        assert (
            err.strip() == "fsm-llm-skills: no skill named 'alpah'; did you mean: alpha"
        )

    def test_list_and_show_flag_a_stopped_load_on_stderr_and_exit_0(
        self, clean: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        note = SKILL_INCOMPLETE_NOTE.format(codes="load-deadline")

        for argv in (["list", clean], ["show", "alpha", clean]):
            _, _, err = _run(capsys, *argv, "--max-seconds", _TINY)
            assert err.splitlines()[0] == note
        assert _run(capsys, "list", clean, "--max-seconds", _TINY)[0] == 0

    def test_many_files_are_listed_up_to_the_cap_with_the_total(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        count = SKILL_LIST_MAX_ITEMS + 10
        files = {f"f{i:03d}.txt": "x" for i in range(count)}
        _skill(tmp_path, "many", files=files)

        _, out, _ = _run(capsys, "show", "many", tmp_path)

        listed = out.splitlines()[2]
        assert listed.startswith(f"files ({count}): ")
        assert len(listed.split(", ")) == SKILL_LIST_MAX_ITEMS

    def test_hostile_skill_text_never_reaches_the_terminal_raw(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _skill(
            tmp_path,
            "evil",
            description=_yaml(_HOSTILE),
            body=f"body {_HOSTILE}\n",
            files={"notes.txt": f"file {_HOSTILE}\n"},
        )
        bad = {_ESC, _BIDI, _ZWSP, chr(7)}

        for argv in (
            ["list", tmp_path],
            ["show", "evil", tmp_path],
            ["show", "evil", tmp_path, "--file", "notes.txt"],
        ):
            code, out, err = _run(capsys, *argv)
            assert code == 0, err
            assert not bad & set(out + err)
            assert "red" in out

    def test_what_the_catalog_returns_is_stripped_when_printed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        issue = SkillIssue.model_construct(
            severity="warning", code="x", message=_HOSTILE, path=_HOSTILE
        )
        _stub_catalog(monkeypatch, issues=(issue,))
        monkeypatch.setattr(
            SkillCatalog, "read_body", lambda *a, **k: (_HOSTILE, 7, 99), raising=True
        )
        bad = {_ESC, _BIDI, _ZWSP, chr(7)}

        for argv in (["validate", "x"], ["list", "x"], ["show", _HOSTILE, "x"]):
            code, out, err = _run(capsys, *argv)
            assert code == 0 and "red" in out
            assert not bad & set(out + err)
        assert "next offset: 7" in out

    def test_unencodable_text_is_written_as_escapes_on_every_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _skill(tmp_path, "cjk", description=_yaml(_CJK + "x"))
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(
            sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        )

        assert main(["list", str(tmp_path)]) == 0
        assert main(["list", str(tmp_path)]) == 0  # idempotent
        stream.flush()

        assert raw.getvalue().count(b"\\u7f51x") == 2


class TestExitCodes:
    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["bogus"],
            ["validate"],
            ["show", "only-a-name"],
            ["list", ".", "--nope"],
            ["show", "n", ".", "--offset", "x"],
            *(["list", ".", "--max-seconds", s] for s in "0 -1 nan inf abc".split()),
        ],
    )
    def test_usage_errors_exit_2(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as info:
            main(argv)

        assert info.value.code == 2

    def test_help_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as info:
            main(["--help"])

        assert info.value.code == 0 and "fsm-llm-skills" in capsys.readouterr().out

    @pytest.mark.parametrize("command", ["validate", "list", "show"])
    def test_a_missing_path_is_one_stderr_line_and_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
    ) -> None:
        argv = (
            [command, "n", str(tmp_path / "nope")]
            if command == "show"
            else [
                command,
                str(tmp_path / "nope"),
            ]
        )

        code, out, err = _run(capsys, *argv)

        assert (code, out) == (1, "")
        assert err.startswith("fsm-llm-skills: error: ") and len(err.splitlines()) == 1
        assert "Traceback" not in err

    def test_a_very_long_error_line_is_capped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        far = tmp_path.joinpath(*["d" * 200] * 12, "x")  # about 2,400 characters

        code, _, err = _run(capsys, "list", far)

        assert code == 1 and 1000 < len(err) < 1100

    @pytest.mark.parametrize("command", ["_cmd_validate", "_cmd_list", "_cmd_show"])
    def test_ctrl_c_exits_130_for_every_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        command: str,
    ) -> None:
        def boom(args: Any) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(skills_cli, command, boom)
        extra = ["name", "."] if command == "_cmd_show" else ["."]

        try:
            code, out, err = _run(capsys, command[5:], *extra)
        except KeyboardInterrupt:  # a leaked interrupt must fail this test only
            pytest.fail("main() let a KeyboardInterrupt escape")

        assert (code, out) == (130, "")
        assert "interrupted" in err

    def test_a_broken_pipe_inside_a_command_is_left_to_main_cli(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(args: Any) -> int:
            raise BrokenPipeError

        monkeypatch.setattr(skills_cli, "_cmd_list", boom)

        with pytest.raises(BrokenPipeError):
            main(["list", "."])

    def test_main_cli_exits_with_the_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", [skills_cli._PROG, "list", "/no/such/dir"])

        with pytest.raises(SystemExit) as info:
            skills_cli.main_cli()

        assert info.value.code == 1

    def test_a_broken_stdout_pipe_parks_stdout_and_exits_1(
        self, clean: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        read_fd, write_fd = os.pipe()
        os.close(read_fd)  # the reader is gone: the first flush gets EPIPE
        stream = open(write_fd, "w", closefd=True)
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "argv", [skills_cli._PROG, "list", str(clean)])

        try:
            with pytest.raises(SystemExit) as info:
                skills_cli.main_cli()
            assert info.value.code == 1
            stream.flush()  # parked on /dev/null now: nothing left to fail
        finally:
            monkeypatch.undo()
            stream.close()

    def test_python_dash_m_runs_the_tool(self, clean: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "fsm_llm.agent_skills_cli",
                "list",
                str(clean),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_REPO,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("alpha\t")


class TestWiring:
    def test_the_console_script_is_declared_and_its_target_imports(self) -> None:
        with open(_REPO / "pyproject.toml", "rb") as handle:
            scripts = tomllib.load(handle)["project"]["scripts"]

        assert scripts["fsm-llm-skills"] == "fsm_llm.agent_skills_cli:main_cli"
        module, attr = scripts["fsm-llm-skills"].split(":")
        assert callable(getattr(importlib.import_module(module), attr))

    def test_importing_the_cli_loads_no_subpackage(self) -> None:
        probe = (
            "import sys\n"
            "import fsm_llm.agent_skills_cli\n"
            "subs = ('agents', 'harness', 'reasoning', 'workflows', 'monitor', 'eval')\n"
            "bad = [m for m in sys.modules if m.split('.')[:2] in "
            "[['fsm_llm', s] for s in subs]]\n"
            "assert not bad, bad\n"
        )

        result = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_REPO,
        )

        assert result.returncode == 0, result.stderr

    def test_the_program_name_is_its_own_and_the_source_is_ascii(self) -> None:
        source = (_REPO / "src" / "fsm_llm" / "agent_skills_cli.py").read_text("utf-8")

        assert skills_cli._PROG == "fsm-llm-skills"
        assert source.isascii() and source.index('"""') < source.index(
            "from __future__"
        )
        assert skills_cli.__all__ == ["main", "main_cli"]
