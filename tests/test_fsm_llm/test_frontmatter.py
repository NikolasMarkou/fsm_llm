"""Tests for ``fsm_llm.frontmatter``, the one YAML frontmatter seam.

The seam was extracted from ``okf.py`` (plan 4c2042f3, D-003) and is shared by
OKF and Agent Skills, so it is pinned here on its own: split, bounded load
(limits are parameters), the no-follow read, the surrogate walk, layering, and
the wiring ``okf.py`` keeps to it.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

import fsm_llm.frontmatter as frontmatter
import fsm_llm.okf as okf_module
from fsm_llm.constants import OKF_MAX_FRONTMATTER_BYTES, OKF_MAX_YAML_DEPTH
from fsm_llm.definitions import OKFError
from fsm_llm.frontmatter import (
    SURROGATE_RE,
    load_yaml,
    lone_surrogate,
    read_bytes,
    split_frontmatter,
)

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "fsm_llm"
_BIG = 1_000_000  # a limit that never trips


def _load(raw: str, **limits: int) -> tuple[dict[Any, Any], list[tuple[str, str]]]:
    """Run ``load_yaml`` and collect ``(severity, code)`` of every note."""
    notes: list[tuple[str, str]] = []

    def note(severity: str, code: str, message: str) -> None:
        assert isinstance(message, str) and message
        notes.append((severity, code))

    params = {"max_bytes": _BIG, "max_depth": 64, **limits}
    return load_yaml(raw, note, **params), notes  # type: ignore[arg-type]


class TestSplit:
    def test_a_block_is_cut_from_its_body(self) -> None:
        assert split_frontmatter("---\na: 1\n---\nbody\n") == ("a: 1\n", "body\n", None)

    def test_a_bom_is_removed(self) -> None:
        assert split_frontmatter("\ufeff---\na: 1\n---\nb") == ("a: 1\n", "b", None)

    def test_crlf_and_cr_line_ends_are_normalised(self) -> None:
        assert split_frontmatter("---\r\na: 1\r\n---\r\nx\r\ny") == (
            "a: 1\n",
            "x\ny",
            None,
        )
        assert split_frontmatter("---\ra: 1\r---\rx") == ("a: 1\n", "x", None)

    def test_text_without_a_fence_is_all_body(self) -> None:
        text = "# Title\n---\nnot yaml\n---\n"
        assert split_frontmatter(text) == (None, text, "frontmatter-missing")
        assert split_frontmatter("") == (None, "", "frontmatter-missing")

    def test_a_fence_that_never_closes_is_unterminated(self) -> None:
        text = "---\na: 1\nb: 2\n"
        assert split_frontmatter(text) == (None, text, "frontmatter-unterminated")
        assert split_frontmatter("---") == (None, "---", "frontmatter-unterminated")

    def test_fences_may_carry_trailing_spaces_and_tabs(self) -> None:
        assert split_frontmatter("---  \na: 1\n---\t \nb") == ("a: 1\n", "b", None)

    def test_a_longer_dash_line_is_not_a_fence(self) -> None:
        text = "----\na: 1\n---\nb"
        assert split_frontmatter(text) == (None, text, "frontmatter-missing")
        text = "---\na: 1\n----\nb\n"
        assert split_frontmatter(text) == (None, text, "frontmatter-unterminated")

    def test_a_rule_in_the_body_stays_in_the_body(self) -> None:
        yaml_text, body, problem = split_frontmatter("---\na: 1\n---\nx\n---\ny\n")
        assert (yaml_text, body, problem) == ("a: 1\n", "x\n---\ny\n", None)

    def test_an_empty_block_gives_empty_yaml(self) -> None:
        assert split_frontmatter("---\n---\nbody") == ("", "body", None)

    def test_the_first_closing_fence_wins(self) -> None:
        assert split_frontmatter("---\na: 1\n---\n---\nb: 2\n---\n")[0] == "a: 1\n"


class TestLoadYaml:
    def test_a_mapping_loads_with_no_note(self) -> None:
        data, notes = _load("name: x\ndescription: y\nn: 3\nlst: [a, b]\n")
        assert data == {"name": "x", "description": "y", "n": 3, "lst": ["a", "b"]}
        assert notes == []

    def test_max_bytes_is_a_parameter(self) -> None:
        raw = "a: 1\nb: 2\n"
        assert _load(raw, max_bytes=len(raw))[1] == []
        data, notes = _load(raw, max_bytes=len(raw) - 1)
        assert data == {}
        assert notes == [("error", "frontmatter-too-large")]

    def test_the_size_is_counted_in_utf8_bytes_before_parsing(self) -> None:
        raw = "a: " + "\u00e9" * 10  # 10 characters, 20 bytes
        assert _load(raw, max_bytes=23)[1] == []
        assert _load(raw, max_bytes=22)[1] == [("error", "frontmatter-too-large")]
        # not even parseable YAML reaches the parser when it is too large
        assert _load("[" * 50, max_bytes=10)[1] == [("error", "frontmatter-too-large")]

    def test_max_depth_is_a_parameter_for_flow_nesting(self) -> None:
        raw = "a: [[[[1]]]]"
        assert _load(raw, max_depth=5)[1] == []
        for depth in (4, 3):  # 3: the bracket check; 4: the structure walk
            data, notes = _load(raw, max_depth=depth)
            assert data == {}
            assert notes == [("error", "frontmatter-too-deep")]

    def test_max_depth_is_a_parameter_for_block_nesting(self) -> None:
        raw = "a:\n  b:\n    c:\n      d: 1\n"
        assert _load(raw, max_depth=64)[1] == []
        data, notes = _load(raw, max_depth=3)
        assert data == {}
        assert notes == [("error", "frontmatter-too-deep")]

    def test_a_very_deep_flow_is_refused_without_parsing_it(self) -> None:
        data, notes = _load("a: " + "[" * 5000 + "]" * 5000, max_depth=64)
        assert data == {}
        assert notes == [("error", "frontmatter-too-deep")]

    def test_flow_nesting_is_refused_before_the_parser_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # PyYAML's scanner is quadratic on nested brackets: the cheap character
        # count must run first, so the parser is never reached.
        def must_not_run(self: Any) -> Any:
            raise AssertionError("the parser ran on a too-deep document")

        monkeypatch.setattr(
            frontmatter._FrontmatterLoader, "get_single_data", must_not_run
        )

        data, notes = _load("a: " + "{" * 100 + "}" * 100, max_depth=10)

        assert data == {}
        assert notes == [("error", "frontmatter-too-deep")]

    def test_an_alias_is_refused(self) -> None:
        data, notes = _load("a: &x [1, 2]\nb: *x\n")
        assert data == {}
        assert notes == [("error", "frontmatter-invalid")]

    def test_an_anchor_alone_is_refused(self) -> None:
        data, notes = _load("a: &x 1\n")
        assert data == {}
        assert notes == [("error", "frontmatter-invalid")]

    def test_a_billion_laughs_document_is_refused_before_expansion(self) -> None:
        raw = "a: &a [x, x]\nb: &b [*a, *a]\nc: &c [*b, *b]\nd: [*c, *c]\n"
        data, notes = _load(raw)
        assert data == {}
        assert notes == [("error", "frontmatter-invalid")]

    def test_a_date_like_scalar_stays_a_string(self) -> None:
        data, notes = _load(
            "d: 2026-01-01\nt: 2026-01-01T10:00:00Z\nu: 2026-01-01 10:00:00\n"
        )
        assert data == {
            "d": "2026-01-01",
            "t": "2026-01-01T10:00:00Z",
            "u": "2026-01-01 10:00:00",
        }
        assert notes == []

    def test_the_global_safe_loader_still_builds_dates(self) -> None:
        # R7 pin: the loader's resolver table is a COPY. If it ever edits the
        # shared table, every other yaml.safe_load in the process changes.
        import datetime

        _load("d: 2026-01-01\n")  # the loader class was used at least once
        assert yaml.safe_load("2026-01-01") == datetime.date(2026, 1, 1)
        assert "tag:yaml.org,2002:timestamp" in {
            tag
            for rules in yaml.SafeLoader.yaml_implicit_resolvers.values()
            for tag, _ in rules
        }

    def test_a_repeated_key_warns_and_the_last_value_wins(self) -> None:
        data, notes = _load("a: 1\nb: 2\na: 3\n")
        assert data == {"a": 3, "b": 2}
        assert notes == [("warning", "key-duplicate")]

    def test_every_repeated_key_is_named_once_per_repeat(self) -> None:
        _, notes = _load("a: 1\na: 2\na: 3\nb: 1\nb: 2\n")
        assert [code for _, code in notes] == ["key-duplicate"] * 3

    @pytest.mark.parametrize("raw", ["", "   \n\n", "~", "null", "# only a comment\n"])
    def test_an_empty_block_warns_and_gives_an_empty_mapping(self, raw: str) -> None:
        assert _load(raw) == ({}, [("warning", "frontmatter-empty")])

    @pytest.mark.parametrize("raw", ["- a\n- b\n", "just text", "42", "[1, 2]", "true"])
    def test_a_non_mapping_is_an_error(self, raw: str) -> None:
        assert _load(raw) == ({}, [("error", "frontmatter-not-mapping")])

    def test_broken_yaml_is_invalid_with_a_capped_message(self) -> None:
        messages: list[str] = []
        data = load_yaml(
            "a: [unclosed " + "x" * 400,
            lambda s, c, m: messages.append(f"{s} {c} {m}"),
            max_bytes=_BIG,
            max_depth=64,
        )
        assert data == {}
        assert len(messages) == 1
        assert messages[0].startswith("error frontmatter-invalid ")
        assert len(messages[0].split(" ", 2)[2]) <= 160

    def test_a_constructor_error_is_invalid_never_too_deep(self) -> None:
        # ``!!timestamp x`` makes a PyYAML constructor raise ValueError.
        assert _load("k: !!timestamp x\n") == ({}, [("error", "frontmatter-invalid")])
        assert _load("k: !!int zz\n") == ({}, [("error", "frontmatter-invalid")])
        assert _load("k: !!float zz\n") == ({}, [("error", "frontmatter-invalid")])

    def test_an_unknown_or_python_tag_is_invalid_not_executed(self) -> None:
        raw = "k: !!python/object/apply:os.system ['echo hi']\n"
        assert _load(raw) == ({}, [("error", "frontmatter-invalid")])
        assert _load("k: !nope 1\n") == ({}, [("error", "frontmatter-invalid")])

    def test_a_lone_surrogate_value_still_loads(self) -> None:
        data, notes = _load('k: "\\ud800"\n')
        assert data == {"k": "\ud800"}
        assert notes == []

    def test_a_non_string_stream_error_does_not_escape(self) -> None:
        # control characters make the PyYAML reader raise a ReaderError
        assert _load("a: \x00\x01")[1] == [("error", "frontmatter-invalid")]

    def test_a_note_is_called_once_per_problem(self) -> None:
        calls: list[tuple[str, str, str]] = []
        load_yaml(
            "x: !!int zz",
            lambda s, c, m: calls.append((s, c, m)),
            max_bytes=_BIG,
            max_depth=64,
        )
        assert len(calls) == 1 and calls[0][0] == "error"


_FUZZ_TAGS = sorted(t for t in yaml.SafeLoader.yaml_constructors if t)
_FUZZ_SHAPES = {
    "empty": "",
    "garbage": " zz!@#",
    "sequence": " [a, 1, [b]]",
    "mapping": " {a: 1, b: [c]}",
}
_ALLOWED_CODES = {
    "frontmatter-too-large",
    "frontmatter-too-deep",
    "frontmatter-invalid",
    "frontmatter-empty",
    "frontmatter-not-mapping",
    "key-duplicate",
}


class TestEveryYamlTagIsTotal:
    """No tag in the SafeLoader constructor table can make ``load_yaml`` raise."""

    def test_the_fuzz_covers_the_whole_constructor_table(self) -> None:
        # Built from the live table, so a PyYAML upgrade that adds a tag is covered.
        assert len(_FUZZ_TAGS) >= 10
        assert "tag:yaml.org,2002:timestamp" in _FUZZ_TAGS
        assert set(_FUZZ_TAGS) == {t for t in yaml.SafeLoader.yaml_constructors if t}

    @pytest.mark.parametrize("shape", sorted(_FUZZ_SHAPES))
    @pytest.mark.parametrize("tag", _FUZZ_TAGS)
    def test_no_exception_escapes(self, tag: str, shape: str) -> None:
        data, notes = _load(f"k: !<{tag}>{_FUZZ_SHAPES[shape]}\n")

        assert isinstance(data, dict)
        assert {code for _, code in notes} <= _ALLOWED_CODES
        assert all(sev in {"error", "warning", "info"} for sev, _ in notes)
        if any(sev == "error" for sev, _ in notes):
            assert data == {}

    @pytest.mark.parametrize("tag", _FUZZ_TAGS)
    def test_a_bad_tag_as_a_key_or_a_top_level_value_is_total_too(
        self, tag: str
    ) -> None:
        for raw in (f"!<{tag}> zz: 1\n", f"!<{tag}> zz\n", f"- !<{tag}>\n"):
            data, notes = _load(raw)
            assert isinstance(data, dict)
            assert {code for _, code in notes} <= _ALLOWED_CODES


def _on_a_thread(fn: Any, timeout: float = 10.0) -> Any:
    """Run *fn* on a daemon thread; fail (do not hang) when it does not finish."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "the call blocked"
    if "error" in box:
        raise box["error"]
    return box["value"]


class TestReadBytes:
    def test_it_reads_at_most_limit_plus_one_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "big.bin"
        path.write_bytes(b"x" * 100)
        assert read_bytes(path, 10) == b"x" * 11
        assert read_bytes(path, 99) == b"x" * 100
        assert read_bytes(path, 100) == b"x" * 100

    def test_a_small_file_is_returned_whole(self, tmp_path: Path) -> None:
        path = tmp_path / "s.md"
        path.write_bytes(b"hello")
        assert read_bytes(path, 1000) == b"hello"
        assert read_bytes(path, 0) == b"h"

    def test_a_missing_file_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_bytes(tmp_path / "nope", 10)

    def test_a_directory_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            read_bytes(tmp_path, 10)

    @pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="no O_NOFOLLOW")
    def test_a_final_symlink_is_not_followed(self, tmp_path: Path) -> None:
        target = tmp_path / "secret.txt"
        target.write_bytes(b"top secret")
        link = tmp_path / "link.md"
        link.symlink_to(target)

        with pytest.raises(OSError):
            read_bytes(link, 100)

        assert read_bytes(target, 100) == b"top secret"

    @pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="no O_NOFOLLOW")
    def test_a_dangling_symlink_raises_oserror(self, tmp_path: Path) -> None:
        link = tmp_path / "dangling.md"
        link.symlink_to(tmp_path / "gone")
        with pytest.raises(OSError):
            read_bytes(link, 100)

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no mkfifo")
    def test_a_fifo_does_not_hang(self, tmp_path: Path) -> None:
        fifo = tmp_path / "pipe.md"
        os.mkfifo(fifo)

        data = _on_a_thread(lambda: read_bytes(fifo, 100))

        assert data == b""  # nobody writes: it opens without blocking and is empty


class TestLoneSurrogate:
    def _walk(self, value: Any, depth: int = 64, nodes: int = _BIG) -> bool:
        return lone_surrogate(value, max_depth=depth, max_nodes=nodes)

    def test_a_clean_value_is_not_flagged(self) -> None:
        assert self._walk({"a": ["b", ("c", 1)], "d": {1, 2.5, None}}) is False
        assert self._walk("plain") is False
        assert self._walk(5) is False

    @pytest.mark.parametrize(
        "value",
        [
            "\ud800",
            ["ok", "x\udc00y"],
            {"k": "\udfff"},
            {"\ud800": "key"},  # in a key
            {"a": {"b": [("deep", frozenset({"\ud801"}))]}},
        ],
    )
    def test_a_lone_surrogate_anywhere_is_found(self, value: Any) -> None:
        assert self._walk(value) is True

    def test_the_regex_matches_the_whole_surrogate_range(self) -> None:
        assert SURROGATE_RE.search("\ud800") and SURROGATE_RE.search("\udfff")
        assert SURROGATE_RE.search("\ud7ff") is None
        assert SURROGATE_RE.search("\ue000") is None
        assert SURROGATE_RE.search("\U0001f600") is None

    def test_depth_is_a_parameter_and_the_signal_is_recursionerror(self) -> None:
        assert self._walk([[[1]]], depth=3) is False
        with pytest.raises(RecursionError, match="too deep"):
            self._walk([[[[1]]]], depth=3)

    def test_a_too_deep_value_raises_even_without_a_surrogate(self) -> None:
        deep: Any = "end"
        for _ in range(100):
            deep = [deep]
        with pytest.raises(RecursionError):
            self._walk(deep, depth=64)

    def test_it_is_not_cut_short_by_a_surrogate_it_already_found(self) -> None:
        deep: Any = "end"
        for _ in range(100):
            deep = [deep]
        with pytest.raises(RecursionError):
            self._walk(["\ud800", deep], depth=64)

    def test_node_count_is_a_parameter(self) -> None:
        assert self._walk(list(range(10)), nodes=10) is False
        with pytest.raises(RecursionError):
            self._walk(list(range(11)), nodes=10)

    def test_a_cycle_ends_in_recursionerror(self) -> None:
        loop: list[Any] = []
        loop.append(loop)
        with pytest.raises(RecursionError):
            self._walk(loop)
        mapping: dict[str, Any] = {}
        mapping["self"] = mapping
        with pytest.raises(RecursionError):
            self._walk(mapping)

    def test_a_deep_value_is_walked_without_python_recursion(self) -> None:
        deep: Any = "end"
        for _ in range(5000):
            deep = [deep]
        # iterative: the limit, not the interpreter stack, ends the walk
        with pytest.raises(RecursionError, match="too deep"):
            self._walk(deep, depth=4000, nodes=_BIG)
        assert self._walk(deep, depth=6000, nodes=_BIG) is False


class TestLayering:
    def test_importing_frontmatter_pulls_in_neither_okf_nor_a_subpackage(self) -> None:
        probe = (
            "import sys\n"
            "import fsm_llm.frontmatter\n"
            "bad = [m for m in sys.modules if m.split('.')[:2] in "
            "(['fsm_llm', 'okf'], ['fsm_llm', 'okf_cli'], ['fsm_llm', 'okf_bridge'], "
            "['fsm_llm', 'cli_support'], "
            "['fsm_llm', 'agents'], "
            "['fsm_llm', 'reasoning'], "
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

    def test_core_does_not_import_frontmatter_eagerly(self) -> None:
        probe = (
            "import sys\n"
            "import fsm_llm\n"
            "assert 'fsm_llm.frontmatter' not in sys.modules\n"
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

    def test_the_module_imports_only_stdlib_and_yaml(self) -> None:
        tree = ast.parse((_SRC / "frontmatter.py").read_text(encoding="utf-8"))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import in frontmatter.py"
                roots.add((node.module or "").split(".")[0])
        assert roots <= set(sys.stdlib_module_names) | {"yaml"}, roots
        assert "yaml" in roots

    def test_the_module_has_one_static_all_naming_real_names(self) -> None:
        tree = ast.parse((_SRC / "frontmatter.py").read_text(encoding="utf-8"))
        assigns = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
        ]
        assert len(assigns) == 1
        names = ast.literal_eval(assigns[0].value)
        assert all(hasattr(frontmatter, name) for name in names)
        assert set(names) == {
            "SURROGATE_RE",
            "Note",
            "load_yaml",
            "lone_surrogate",
            "read_bytes",
            "split_frontmatter",
        }

    def test_the_source_is_pure_ascii(self) -> None:
        # a BOM or surrogate must be spelled as an escape, never as the character
        text = (_SRC / "frontmatter.py").read_text(encoding="utf-8")
        assert text.isascii()
        assert '"\\ufeff"' in text and '"[\\ud800-\\udfff]"' in text

    def test_the_docstring_comes_before_the_future_import(self) -> None:
        tree = ast.parse((_SRC / "frontmatter.py").read_text(encoding="utf-8"))
        assert ast.get_docstring(tree)
        first = tree.body[1]
        assert isinstance(first, ast.ImportFrom) and first.module == "__future__"


class TestOneYamlLoader:
    def test_only_frontmatter_defines_a_safeloader_subclass(self) -> None:
        owners: list[str] = []
        for path in sorted(_SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and any(
                    ast.unparse(base) in {"yaml.SafeLoader", "yaml.Loader"}
                    for base in node.bases
                ):
                    owners.append(f"{path.relative_to(_SRC)}:{node.name}")
        assert owners == ["frontmatter.py:_FrontmatterLoader"]


class TestOkfKeepsItsWiring:
    """``okf.py`` imports the seam under its old private names (monkeypatch
    targets of the OKF tests) and hands it the OKF limits."""

    def test_the_private_names_are_the_shared_functions(self) -> None:
        assert okf_module._split_frontmatter is split_frontmatter
        assert okf_module._read_bytes is read_bytes
        assert okf_module._SURROGATE_RE is SURROGATE_RE

    def test_okf_passes_its_own_limits(self) -> None:
        load = okf_module._load_yaml
        assert load.func is load_yaml  # type: ignore[attr-defined]
        assert load.keywords == {  # type: ignore[attr-defined]
            "max_bytes": OKF_MAX_FRONTMATTER_BYTES,
            "max_depth": OKF_MAX_YAML_DEPTH,
        }

    def test_the_okf_adapter_restores_the_okferror_type(self) -> None:
        assert okf_module.has_lone_surrogate({"a": ["\ud800"]}) is True
        assert okf_module.has_lone_surrogate({"a": ["ok"]}) is False
        loop: list[Any] = []
        loop.append(loop)
        with pytest.raises(OKFError, match="too deep"):
            okf_module.has_lone_surrogate(loop)

    def test_okf_has_no_loader_class_of_its_own(self) -> None:
        assert not hasattr(okf_module, "_OKFLoader")
        assert not hasattr(okf_module, "_flow_depth_exceeded")
