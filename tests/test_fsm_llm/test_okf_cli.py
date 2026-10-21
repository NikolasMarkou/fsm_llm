"""Tests for ``fsm-llm-okf`` (``fsm_llm.okf_cli``): commands, exit codes, wiring."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import fsm_llm.okf_cli as okf_cli
from fsm_llm import okf as okf_module
from fsm_llm.constants import (
    OKF_LIST_MAX_ITEMS,
    OKF_MAX_LOAD_SECONDS,
    OKF_READ_PAGE_CHARS,
)
from fsm_llm.okf import (
    bundle_from_memory_entries,
    load_bundle,
    strip_unsafe_text,
    write_bundle,
)
from fsm_llm.okf_cli import main
from tests.test_fsm_llm.test_okf import _FakeClock

_REPO = Path(__file__).resolve().parents[2]
_ESC = "\x1b"


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _doc(type_: str = "Note", body: str = "text\n", **keys: str) -> str:
    front = "".join(f"{k}: {v}\n" for k, v in keys.items())
    return f"---\ntype: {type_}\n{front}---\n{body}"


def _run(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def clean_bundle(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "kb",
        {
            "orders.md": _doc("Dataset", "Orders live here.\n", title="Orders"),
            "guides/setup.md": _doc("Guide", "Install it.\n", title="Setup"),
        },
    )


# --------------------------------------------------------------
# validate
# --------------------------------------------------------------


class TestValidate:
    def test_a_clean_bundle_exits_0_and_prints_no_issue_lines(
        self, clean_bundle: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "validate", clean_bundle)

        assert code == 0
        assert out == ""
        assert "2 concepts, 0 errors" in err

    def test_an_error_issue_exits_1_and_is_printed_as_four_fields(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(tmp_path / "kb", {"bad.md": "no frontmatter at all\n"})

        code, out, _ = _run(capsys, "validate", root)

        assert code == 1
        line = out.splitlines()[0]
        severity, issue_code, where, message = line.split(" ", 3)
        assert severity == "error"
        assert issue_code and where and message

    def test_strict_is_the_only_difference_for_warnings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A date-only stale_after is accepted with a warning (D-003).
        root = _write(
            tmp_path / "kb", {"a.md": _doc(stale_after="2999-01-01", title="A")}
        )

        relaxed, out, _ = _run(capsys, "validate", root)
        strict, out_strict, _ = _run(capsys, "validate", root, "--strict")

        assert out.startswith("warning ")
        assert out == out_strict
        assert (relaxed, strict) == (0, 1)

    def test_strict_does_not_fail_on_info_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(
            tmp_path / "kb", {"a.md": _doc(body="[gone](/missing.md)\n", title="A")}
        )

        code, out, _ = _run(capsys, "validate", root, "--strict")

        assert out.startswith("info ")
        assert code == 0

    def test_hostile_text_in_issues_is_stripped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(
            tmp_path / "kb",
            {f"x{_ESC}[31m.md": "x", "ok.md": _doc(title="fine")},
        )

        _, out, err = _run(capsys, "validate", root)

        assert _ESC not in out + err

    def test_a_missing_root_is_exit_1_with_one_stderr_line_no_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "validate", tmp_path / "nope")

        assert code == 1
        assert out == ""
        assert err.startswith("fsm-llm-okf: error: ")
        assert len(err.strip().splitlines()) == 1
        assert "Traceback" not in err

    def test_a_file_as_root_is_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a_file = tmp_path / "f.md"
        a_file.write_text("x", encoding="utf-8")

        code, _, err = _run(capsys, "validate", a_file)

        assert code == 1
        assert "not a directory" in err


# --------------------------------------------------------------
# list and search
# --------------------------------------------------------------


class TestList:
    def test_lists_id_type_title_sorted_by_id(
        self, clean_bundle: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = _run(capsys, "list", clean_bundle)

        assert code == 0
        assert out.splitlines() == [
            "guides/setup\tGuide\tSetup",
            "orders\tDataset\tOrders",
        ]

    def test_type_filter_is_case_insensitive(
        self, clean_bundle: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, out, _ = _run(capsys, "list", clean_bundle, "--type", "guide")

        assert [ln.split("\t")[0] for ln in out.splitlines()] == ["guides/setup"]

    def test_deprecated_is_hidden_unless_asked_and_flags_show(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(
            tmp_path / "kb",
            {
                "live.md": _doc(title="Live", status="draft"),
                "old.md": _doc(title="Old", status="deprecated"),
                "late.md": _doc(title="Late", stale_after="'2000-01-01T00:00:00Z'"),
            },
        )

        _, default_out, _ = _run(capsys, "list", root)
        _, all_out, _ = _run(capsys, "list", root, "--include-deprecated")

        assert "old" not in default_out
        assert "live\tNote\tLive\tdraft" in default_out
        assert "late\tNote\tLate\tstale" in default_out
        assert "old\tNote\tOld\tdeprecated" in all_out

    def test_output_is_capped_with_a_more_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        total = OKF_LIST_MAX_ITEMS + 4
        root = _write(
            tmp_path / "kb", {f"c{i:03d}.md": _doc(title=f"T{i}") for i in range(total)}
        )

        _, out, _ = _run(capsys, "list", root)

        lines = out.splitlines()
        assert len(lines) == OKF_LIST_MAX_ITEMS + 1
        assert lines[-1].startswith("... and 4 more")

    def test_a_missing_root_is_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = _run(capsys, "list", tmp_path / "nope")

        assert code == 1
        assert "Traceback" not in err

    def test_control_characters_in_titles_never_reach_the_terminal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        title = json.dumps(f"a{_ESC}[31mred{_ESC}[0m‮b\x07")
        root = _write(tmp_path / "kb", {"a.md": _doc(title=title)})

        _, out, _ = _run(capsys, "list", root)

        assert out == "a\tNote\taredb\n"


class TestSearch:
    def test_hits_are_scored_best_first(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(
            tmp_path / "kb",
            {
                "weak.md": _doc(title="Notes", body="mentions refunds once\n"),
                "strong.md": _doc(title="Refunds policy", body="refunds\n"),
                "none.md": _doc(title="Unrelated", body="nothing\n"),
            },
        )

        code, out, _ = _run(capsys, "search", root, "refunds")

        assert code == 0
        rows = [ln.split("\t") for ln in out.splitlines()]
        assert [r[1] for r in rows] == ["strong", "weak"]
        assert float(rows[0][0]) > float(rows[1][0])

    def test_limit_is_clamped_to_one_and_to_the_list_cap(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        total = OKF_LIST_MAX_ITEMS + 5
        root = _write(
            tmp_path / "kb",
            {f"c{i:03d}.md": _doc(title="shared word") for i in range(total)},
        )

        _, one, _ = _run(capsys, "search", root, "shared", "--limit", "0")
        _, many, _ = _run(capsys, "search", root, "shared", "--limit", "9999")

        assert len(one.splitlines()) == 1
        assert len(many.splitlines()) == OKF_LIST_MAX_ITEMS

    def test_deprecated_is_excluded(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(
            tmp_path / "kb", {"old.md": _doc(title="gizmo", status="deprecated")}
        )

        code, out, _ = _run(capsys, "search", root, "gizmo")

        assert (code, out) == (0, "")

    def test_no_hit_is_exit_0(
        self, clean_bundle: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = _run(capsys, "search", clean_bundle, "zzzzzz")

        assert (code, out) == (0, "")

    def test_a_missing_root_is_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(capsys, "search", tmp_path / "nope", "x")[0] == 1


# --------------------------------------------------------------
# show
# --------------------------------------------------------------


class TestShow:
    def test_header_then_body(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(
            tmp_path / "kb",
            {
                "a.md": "---\ntype: Note\ntitle: Alpha\nstatus: draft\n"
                "verified:\n- by: human:ann\n---\nhello body\n"
            },
        )

        code, out, _ = _run(capsys, "show", root, "a")

        assert code == 0
        assert out.splitlines()[:5] == [
            "id: a",
            "type: Note",
            "title: Alpha",
            "trust (claimed by bundle): human-reviewed",
            "flags: draft",
        ]
        assert "hello body" in out
        assert "next offset" not in out

    def test_a_long_body_pages_and_the_pages_add_up(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        body = "".join(f"line {i:05d}\n" for i in range(600))
        root = _write(tmp_path / "kb", {"big.md": _doc(body=body)})
        expected = strip_unsafe_text(load_bundle(root).get("big").body, multiline=True)  # type: ignore[union-attr]
        assert len(expected) > 2 * OKF_READ_PAGE_CHARS

        collected, offset, pages = "", 0, 0
        while True:
            code, out, _ = _run(capsys, "show", root, "big", "--offset", str(offset))
            assert code == 0
            page = out.split("\n\n", 1)[1]
            nxt = re.search(r"\nnext offset: (\d+)\n$", page)
            pages += 1
            if nxt is None:
                collected += page[:-1]  # print()'s trailing newline
                break
            collected += page[: nxt.start()]
            assert int(nxt.group(1)) == offset + OKF_READ_PAGE_CHARS
            offset = int(nxt.group(1))

        assert pages >= 3
        assert collected == expected

    def test_offset_is_clamped_both_ways(
        self, clean_bundle: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, first, _ = _run(capsys, "show", clean_bundle, "orders", "--offset", "-50")
        code, past, _ = _run(
            capsys, "show", clean_bundle, "orders", "--offset", "99999"
        )

        assert "Orders live here." in first
        assert code == 0
        assert "Orders live here." not in past
        assert "next offset" not in past

    def test_unknown_id_is_exit_1_with_at_most_five_close_matches(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(
            tmp_path / "kb", {f"orders{i}.md": _doc(title=f"O{i}") for i in range(8)}
        )

        code, out, err = _run(capsys, "show", root, "orders")

        assert code == 1
        assert out == ""
        lines = err.splitlines()
        assert "no concept with id 'orders'" in lines[0]
        suggestions = lines[1].removeprefix("did you mean: ").split(", ")
        assert lines[1].startswith("did you mean: ")
        assert 1 <= len(suggestions) <= 5

    def test_unknown_id_without_close_match_prints_one_line(
        self, clean_bundle: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = _run(capsys, "show", clean_bundle, "zzzzqqqq")

        assert code == 1
        assert len(err.strip().splitlines()) == 1

    def test_ansi_and_control_characters_are_stripped_everywhere(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        title = json.dumps(f"T{_ESC}[31mitle{_ESC}]0;pwn\x07‮")
        body = f"before {_ESC}[2J{_ESC}[31mred\x07 \x00 zero​width\nnext line\n"
        root = _write(tmp_path / "kb", {"a.md": _doc(title=title, body=body)})

        code, out, err = _run(capsys, "show", root, "a")

        assert code == 0
        assert _ESC not in out + err
        assert not any(ch in out for ch in "\x07\x00‮​")
        assert "title: Title" in out
        assert "before red" in out
        assert "next line" in out  # newlines are kept in a body page

    def test_a_deprecated_concept_is_readable_by_explicit_id(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _write(
            tmp_path / "kb", {"old.md": _doc(title="Old", status="deprecated")}
        )

        code, out, _ = _run(capsys, "show", root, "old")

        assert code == 0
        assert "flags: deprecated" in out

    def test_a_hostile_id_argument_is_not_echoed_raw(
        self, clean_bundle: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = _run(capsys, "show", clean_bundle, f"a{_ESC}[31m" + "x" * 5000)

        assert code == 1
        assert _ESC not in out + err
        assert len(err) < 3000


# --------------------------------------------------------------
# export
# --------------------------------------------------------------


def _store(path: Path, entries: list[Any]) -> Path:
    path.write_text(
        json.dumps(
            {
                "embedding_model": "m",
                "counter": len(entries),
                "max_entries": 100,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


_ENTRIES: list[Any] = [
    {
        "id": "mem_1",
        "text": "The sky is blue.\nSecond line.",
        "metadata": {"tags": ["sky"], "api_key": "sk-secret-value-123456"},
        "embedding": [0.1, 0.2],
    },
    {"id": "mem_2", "text": "Water is wet.", "metadata": {}, "embedding": None},
]


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class TestExport:
    def test_writes_a_loadable_bundle_and_reports_count_and_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _store(tmp_path / "store.json", _ENTRIES)
        out_dir = tmp_path / "out"

        code, out, _ = _run(capsys, "export", store, out_dir)

        assert code == 0
        assert out.strip() == f"wrote 3 files under {out_dir.resolve()}"
        bundle = load_bundle(out_dir)
        assert sorted(c.id for c in bundle) == ["memory/mem_1", "memory/mem_2"]
        assert not [i for i in bundle.issues if i.severity == "error"]
        on_disk = "".join(p.read_text() for p in out_dir.rglob("*.md"))
        assert "sk-secret-value-123456" not in on_disk
        assert "embedding" not in on_disk

    def test_same_files_as_the_library_route(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI half of the D-008 parity test (agents half arrives in step 7)."""
        store = _store(tmp_path / "store.json", _ENTRIES)
        via_cli = tmp_path / "cli"
        via_lib = tmp_path / "lib"
        stamp = "2026-01-01T00:00:00+00:00"
        real = bundle_from_memory_entries

        # The converter stamps `generated.at` from the clock; pin it on the CLI
        # route so the two outputs can be compared byte for byte.
        monkeypatch.setattr(
            okf_cli, "bundle_from_memory_entries", lambda e: real(e, now=stamp)
        )
        assert main(["export", str(store), str(via_cli)]) == 0
        write_bundle(
            real(json.loads(store.read_text(encoding="utf-8"))["entries"], now=stamp),
            via_lib,
        )

        assert _files(via_cli) == _files(via_lib)
        assert len(_files(via_cli)) == 3

    def test_existing_target_needs_overwrite(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _store(tmp_path / "store.json", _ENTRIES)
        out_dir = tmp_path / "out"
        assert _run(capsys, "export", store, out_dir)[0] == 0
        before = _files(out_dir)

        code, out, err = _run(capsys, "export", store, out_dir)
        after_refusal = _files(out_dir)
        code_ok, _, _ = _run(capsys, "export", store, out_dir, "--overwrite")

        assert code == 1
        assert out == ""
        assert "Traceback" not in err
        assert after_refusal == before
        assert code_ok == 0

    @pytest.mark.parametrize(
        ("content", "fragment"),
        [
            (b"{not json", "not valid JSON"),
            (b"\xff\xfe\x00", "not valid JSON"),
            (b"[]", "'entries' list"),
            (b'{"entries": "nope"}', "'entries' list"),
            (b'{"entries": [1, 2]}', "'entries' list"),
            (b'{"entries": [{"id": "a", "text": "t"}, "x"]}', "'entries' list"),
            (b'{"other": []}', "'entries' list"),
            (b"[" * 100_000, "not valid JSON"),
        ],
    )
    def test_corrupt_or_wrong_shape_is_exit_1_never_a_traceback(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        content: bytes,
        fragment: str,
    ) -> None:
        store = tmp_path / "store.json"
        store.write_bytes(content)

        code, out, err = _run(capsys, "export", store, tmp_path / "out")

        assert code == 1
        assert out == ""
        assert fragment in err
        assert "Traceback" not in err
        assert not (tmp_path / "out").exists()

    def test_missing_file_is_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = _run(capsys, "export", tmp_path / "nope.json", tmp_path / "out")

        assert code == 1
        assert "cannot read memory file" in err

    def test_an_oversize_file_is_refused_without_reading_it_all(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(okf_cli, "_MAX_STORE_BYTES", 50)
        store = _store(tmp_path / "store.json", _ENTRIES)

        code, _, err = _run(capsys, "export", store, tmp_path / "out")

        assert code == 1
        assert "larger than 50 bytes" in err

    def test_no_usable_entry_writes_nothing_and_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _store(tmp_path / "store.json", [{"id": "a", "text": "  "}])

        code, _, err = _run(capsys, "export", store, tmp_path / "out")

        assert code == 1
        assert "warning entry-skipped" in err
        assert "no exportable entries" in err
        assert not (tmp_path / "out").exists()

    def test_skipped_entries_warn_on_stderr_but_the_rest_is_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _store(tmp_path / "store.json", [{"id": "a", "text": ""}, *_ENTRIES])

        code, out, err = _run(capsys, "export", store, tmp_path / "out")

        assert code == 0
        assert "entry-skipped" in err
        assert out.startswith("wrote 3 files")

    def test_a_target_that_is_a_file_is_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = _store(tmp_path / "store.json", _ENTRIES)
        target = tmp_path / "taken"
        target.write_text("x", encoding="utf-8")

        code, _, err = _run(capsys, "export", store, target)

        assert code == 1
        assert "Traceback" not in err


# --------------------------------------------------------------
# Usage errors, Ctrl-C, entry points
# --------------------------------------------------------------


class TestUsageAndExit:
    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["bogus"],
            ["validate"],
            ["show", "only-one-arg"],
            ["list", "x", "--nope"],
            ["search", "x", "q", "--limit", "many"],
            ["show", "x", "a", "--offset", "1.5"],
        ],
    )
    def test_usage_errors_exit_2(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as info:
            main(argv)

        assert info.value.code == 2

    def test_help_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as info:
            main(["--help"])

        assert info.value.code == 0
        assert "validate" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("command", "extra"),
        [
            ("_cmd_validate", ["validate", "p"]),
            ("_cmd_list", ["list", "p"]),
            ("_cmd_show", ["show", "p", "i"]),
            ("_cmd_search", ["search", "p", "q"]),
            ("_cmd_export", ["export", "m", "o"]),
        ],
    )
    def test_ctrl_c_exits_130_for_every_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        command: str,
        extra: list[str],
    ) -> None:
        def boom(args: Any) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(okf_cli, command, boom)

        code, out, err = _run(capsys, *extra)

        assert code == 130
        assert out == ""
        assert "interrupted" in err

    def test_main_cli_exits_with_the_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", [okf_cli._PROG, "validate", "/no/such/dir"])

        with pytest.raises(SystemExit) as info:
            okf_cli.main_cli()

        assert info.value.code == 1

    def test_python_dash_m_runs_the_tool(self, clean_bundle: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-I", "-m", "fsm_llm.okf_cli", "list", str(clean_bundle)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_REPO,
        )

        assert result.returncode == 0, result.stderr
        assert "orders\tDataset\tOrders" in result.stdout


# --------------------------------------------------------------
# Packaging wiring and layering
# --------------------------------------------------------------


def _pyproject() -> dict[str, Any]:
    with open(_REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


class TestWiring:
    def test_pyyaml_is_a_core_dependency_with_a_floor_of_6(self) -> None:
        from packaging.requirements import Requirement
        from packaging.version import Version

        deps = _pyproject()["project"]["dependencies"]
        found = [Requirement(d) for d in deps if Requirement(d).name == "pyyaml"]

        assert len(found) == 1
        spec = found[0].specifier
        assert spec.contains(Version("6.0"))
        assert not spec.contains(Version("5.4.1"))
        assert not found[0].marker

    def test_the_console_script_is_declared_and_its_target_imports(self) -> None:
        scripts = _pyproject()["project"]["scripts"]

        assert scripts["fsm-llm-okf"] == "fsm_llm.okf_cli:main_cli"
        module, attr = scripts["fsm-llm-okf"].split(":")
        assert callable(getattr(importlib.import_module(module), attr))

    def test_importing_the_cli_loads_no_subpackage_and_no_litellm(self) -> None:
        probe = (
            "import sys\n"
            "import fsm_llm.okf_cli\n"
            "bad = [m for m in sys.modules if m.split('.')[:2] in "
            "(['fsm_llm', 'agents'], ['fsm_llm', 'harness'], ['fsm_llm', 'reasoning'], "
            "['fsm_llm', 'workflows'], ['fsm_llm', 'monitor'], ['fsm_llm', 'eval'])]\n"
            "assert not bad, bad\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_REPO,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_the_cli_source_imports_no_subpackage(self) -> None:
        source = (_REPO / "src" / "fsm_llm" / "okf_cli.py").read_text(encoding="utf-8")

        assert not re.search(
            r"(?m)^\s*(from|import)\s+(fsm_llm\.|\.)(agents|harness|reasoning|"
            r"workflows|monitor|eval)\b",
            source,
        )
        assert not re.search(r"(?m)^\s*(from|import)\s+litellm", source)


class TestSharedHelpersAndIssueCap:
    """Step 7.1: the CLI uses the core helpers and no longer re-caps issues."""

    def test_the_cli_has_no_private_copy_of_the_shared_helpers(self) -> None:
        for name in ("_cell", "_close_ids"):
            assert not hasattr(okf_cli, name)

    def test_the_issues_truncated_line_is_printed_not_hidden(
        self,
        clean_bundle: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fsm_llm.constants import OKF_MAX_ISSUES
        from fsm_llm.okf import OKFIssue

        issues = [
            OKFIssue(severity="warning", code="w", message=f"m{i}")
            for i in range(OKF_MAX_ISSUES)
        ] + [OKFIssue(severity="info", code="issues-truncated", message="1 more")]
        monkeypatch.setattr(okf_cli, "check_bundle", lambda bundle: issues)

        code, out, _ = _run(capsys, "validate", clean_bundle)

        assert code == 0
        assert "issues-truncated" in out and "... and" not in out
        assert len(out.splitlines()) == OKF_MAX_ISSUES + 1

    def test_show_close_matches_use_the_shared_helper(
        self, clean_bundle: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = _run(capsys, "show", clean_bundle, "orderz")
        assert code == 1 and "did you mean:" in err


# --------------------------------------------------------------
# Step 4.2: export of hostile metadata exits cleanly (D-017 item 1)
# --------------------------------------------------------------


def _deep_store(path: Path, depth: int) -> Path:
    # built by string repetition: json.dumps on a deep structure recurses itself
    path.write_text(
        '{"entries":[{"id":"a","text":"hi","metadata":{"m":'
        + "[" * depth
        + "]" * depth
        + '}},{"id":"b","text":"second"}]}',
        encoding="utf-8",
    )
    return path


class TestExportOfDeepMetadata:
    @pytest.mark.parametrize("depth", [500, 800])
    def test_metadata_json_can_parse_is_cut_and_both_entries_are_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], depth: int
    ) -> None:
        store = _deep_store(tmp_path / "store.json", depth)

        code, out, err = _run(capsys, "export", store, tmp_path / "kb")

        assert code == 0, err
        assert "wrote 3 files" in out
        assert sorted(c.id for c in load_bundle(tmp_path / "kb")) == [
            "memory/a",
            "memory/b",
        ]

    def test_metadata_json_itself_cannot_parse_exits_one_without_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # deep enough that json.loads raises RecursionError on 3.10, 3.11 and
        # 3.12; the C scanner's limit differs per interpreter (3.12 parses 5,000)
        store = _deep_store(tmp_path / "store.json", 100_000)

        code, out, err = _run(capsys, "export", store, tmp_path / "kb")

        assert code == 1 and out == ""
        assert err.startswith("fsm-llm-okf:") and "Traceback" not in err


# --------------------------------------------------------------
# Step 7.2 (D-017): --max-seconds on the loading subcommands
# --------------------------------------------------------------

_LOADING = {
    "validate": ["validate", "KB"],
    "list": ["list", "KB"],
    "show": ["show", "KB", "orders"],
    "search": ["search", "KB", "orders"],
}


class TestMaxSeconds:
    @pytest.mark.parametrize("command", sorted(_LOADING))
    def test_the_flag_reaches_load_bundle_as_okf_limits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        clean_bundle: Path,
        command: str,
    ) -> None:
        seen: list[Any] = []
        real = okf_cli.load_bundle

        def spy(path: Any, *, limits: Any = None) -> Any:
            seen.append(limits)
            return real(path, limits=limits)

        monkeypatch.setattr(okf_cli, "load_bundle", spy)
        argv = [str(clean_bundle) if a == "KB" else a for a in _LOADING[command]]

        _run(capsys, *argv)
        _run(capsys, *argv, "--max-seconds", "7.5")

        assert [limit.max_seconds for limit in seen] == [OKF_MAX_LOAD_SECONDS, 7.5]

    @pytest.mark.parametrize("command", sorted(_LOADING))
    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "1e999", "abc"])
    def test_a_bad_value_is_a_usage_error(self, command: str, value: str) -> None:
        with pytest.raises(SystemExit) as info:
            main([*_LOADING[command], "--max-seconds", value])

        assert info.value.code == 2

    def test_export_has_no_such_flag(self) -> None:
        with pytest.raises(SystemExit) as info:
            main(["export", "m.json", "out", "--max-seconds", "5"])

        assert info.value.code == 2

    def test_a_spent_budget_gives_a_load_deadline_error_and_exit_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        root = _write(tmp_path / "kb", {f"{n}.md": _doc() for n in "abcde"})
        clock = _FakeClock()
        real = okf_module.parse_concept_text

        def slow(cid: str, text: str) -> Any:
            clock.now += 1.0  # every parsed concept costs one fake second
            return real(cid, text)

        monkeypatch.setattr(okf_module, "time", clock)
        monkeypatch.setattr(okf_module, "parse_concept_text", slow)

        code, out, err = _run(capsys, "validate", root, "--max-seconds", "2.5")

        assert code == 1
        line = next(x for x in out.splitlines() if " load-deadline " in x)
        assert line.startswith("error load-deadline d.md ")
        assert "--max-seconds" in line  # the validate output names the flag
        assert "3 concepts, 1 errors" in err

        code, out, _ = _run(capsys, "validate", root, "--max-seconds", "100")

        assert code == 0 and "load-deadline" not in out
