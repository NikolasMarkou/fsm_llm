"""The current docs name only modules that exist in the Section 5 layout.

Plan d454fd62, step 96. The 0.4 layout split moved or split every large module (`pipeline.py`
became the package `engine/turn/`, `llm.py` the package `llm/`, and so on), so the guides,
`AGENTS.md` and `README.md` files that still say `pipeline.py` or `fsm_llm.validator` are wrong.
This module reads every non-historical Markdown file and fails on stale paths of three kinds (and exempts one):

* a dotted path `fsm_llm.x.y` that does not resolve: the longest prefix that is a module is
  found with `importlib.util.find_spec`, and each remaining part must be an attribute of it
  (a class member, a pydantic field or a module global), so a removed module and a removed name
  both fail;
* a `from fsm_llm.x import name` line whose module resolves but lacks the name;
* a file mention `name.py` whose base name is one of the OLD module names (the keys of
  `tests/fixtures/module_path_map.json` `moved`, minus the ones still on disk as a plain file),
  or a rooted path `src/fsm_llm/...py`, `tests/...py`, `scripts/...py` that is not a file. A
  mention with a directory (`agents/tools.py`) must match the tail of a real path. A bare mention
  (`pipeline.py`) is judged in the scope of the doc: a doc inside `src/fsm_llm/<pkg>/` may name a
  file of that package (`reasoning/AGENTS.md` says `definitions.py`), a doc anywhere else may name
  a top-level `src/fsm_llm/*.py` file or a file in a directory the same line names (a row that
  says `context/layout/` may list its `base.py`), so a root or `docs/` text has to say where;
* exempt: a path named as the past of a move (the clause before it says "the old", "formerly", "was",
  "replaces", "retired", "moved from") is provenance and is not judged: the docs keep saying where a
  module came from, and only a statement of the CURRENT layout has to be true.

Historical files are listed in `HISTORICAL` and are never read: release notes, run logs, design
records, the roadmap and the bench data.
`ALLOWED` lists the few statements that stay on purpose, each with its reason, so a new stale
path is a failure and an old one that stays is a visible, argued line. Nothing here imports a
model or touches the network; the resolver imports only `fsm_llm` modules the suite imports anyway.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tests.test_fsm_llm_regression.support import PACKAGE, REPO, package_dir

PKG_DIR = package_dir()
PATH_MAP = REPO / "tests" / "fixtures" / "module_path_map.json"

#: Files that record the past: never read. Directories end with a slash.
HISTORICAL = (
    "CHANGELOG.md",  # release history; its entries name the paths of their day
    "EVALUATE.md",  # eval run log
    "docs/2026_roadmap.md",  # the roadmap: plans, written before the layout
    "docs/strands_features",  # dated design records (prefix)
    "scripts/bench_data/",  # frozen bench record
    "plans/",  # planning artifacts
)

#: Stale-looking statements that stay on purpose: (doc path, token) -> reason.
ALLOWED: dict[tuple[str, str], str] = {
    ("tests/test_fsm_llm_regression/AGENTS.md", "fsm_llm.workflows.cli"): (
        "the test asserts that this module does not exist"
    ),
    ("tests/test_fsm_llm_regression/README.md", "fsm_llm.workflows.cli"): (
        "the test asserts that this module does not exist"
    ),
}

_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "plans",
}

# `fsm_llm.a.b.c`, not inside a longer word or file name (`fsm_llm.egg-info`, `llm_fsm`).
DOTTED = re.compile(r"(?<![\w/.\-])fsm_llm((?:\.[A-Za-z_]\w*)+)(?![\w\-])")
# `a/b/c.py` or `c.py` (a URL never starts a match: the path part must not follow `:`, `/`, `.`).
PY_TOKEN = re.compile(r"(?<![\w./:\-])((?:[\w.\-]+/)*\w+\.py)(?![\w])")


def _walk_files(root: Path) -> Iterator[Path]:
    for base, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for name in files:
            yield Path(base) / name


def _is_historical(rel: str) -> bool:
    return any(rel == h or rel.startswith(h) for h in HISTORICAL)


def doc_files(repo: Path = REPO) -> list[Path]:
    """The Markdown files this test reads, sorted: the root docs, every `AGENTS.md` and
    `README.md` below `src/` and `tests/`, and `docs/**/*.md`, minus `HISTORICAL`."""
    found: list[Path] = []
    for name in ("README.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md"):
        if (repo / name).is_file():
            found.append(repo / name)
    for top in ("src", "tests"):
        for path in _walk_files(repo / top):
            if path.name in ("AGENTS.md", "README.md"):
                found.append(path)
    for path in _walk_files(repo / "docs"):
        if path.suffix == ".md":
            found.append(path)
    return sorted(
        p for p in set(found) if not _is_historical(p.relative_to(repo).as_posix())
    )


def python_files(repo: Path = REPO) -> list[str]:
    """Repo-relative posix path of every `.py` file the docs may name."""
    out: list[str] = []
    for top in ("src", "tests", "scripts", "examples", "docs"):
        out += [
            p.relative_to(repo).as_posix()
            for p in _walk_files(repo / top)
            if p.suffix == ".py"
        ]
    out += [p.name for p in repo.glob("*.py")]
    return sorted(out)


def old_basenames(map_path: Path = PATH_MAP) -> set[str]:
    """Base names of the modules the layout split moved (`pipeline`, `llm`, `tools` ...).

    Interface contract: reads the `moved` keys of the path map; a key `fsm_llm.a.b` gives `b`
    (the key `fsm_llm.x` gives `x`). Returns `{}` for a missing file. A name still on disk as a
    plain file is handled by the caller (a bare mention is judged in the doc's scope).
    """
    if not map_path.is_file():
        return set()
    moved = json.loads(map_path.read_text(encoding="utf-8")).get("moved", {})
    return {key.rsplit(".", 1)[-1] for key in moved}


# A path named as the PAST of a move ("the old `api.py`", "formerly `fsm_llm.validator`", "`x` was
# split") is history, not a claim about the layout: the clause before it says so. The window is the
# text of the same clause (no sentence end, `;`, `|`, `)` or newline) up to 70 characters back.
_PAST = re.compile(
    r"\b(old|formerly|was|were|previously|replaces|replaced|retired|historical|original"
    r"|(?:moved|copied|extracted|split|derived)(?: verbatim)? from)\b",
    re.IGNORECASE,
)


def is_past(text: str, start: int) -> bool:
    """True when the clause before position *start* of *text* marks the path as history.

    Interface contract: *text* is the whole document, *start* the offset of the matched path.
    Pure; a clause ends at `. `, `;`, `|`, `)` or a newline, so the marker has to be in the same
    clause as the path.
    """
    window = text[max(0, start - 70) : start]
    clause = re.split(r"\.\s|;|\||\)|\n", window)[-1]
    return _PAST.search(clause) is not None


# --- dotted paths -------------------------------------------------------------------------


def resolve_dotted(dotted: str) -> str | None:
    """None when `fsm_llm<dotted>` resolves, else a short reason.

    The longest module prefix is found without importing it (`find_spec`); the rest must be
    attributes of that module (`getattr`, a pydantic `model_fields` entry, or a class
    annotation), importing the module once. A module that cannot be imported because an
    optional dependency is missing resolves by its spec alone.
    """
    parts = (PACKAGE + dotted).split(".")
    module_end = 0
    for k in range(len(parts), 0, -1):
        try:
            spec = importlib.util.find_spec(".".join(parts[:k]))
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            module_end = k
            break
    if module_end == 0:
        return "no such module"
    if module_end == len(parts):
        return None
    module_name = ".".join(parts[:module_end])
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError:
        return (
            None  # an optional dependency is absent: the spec is all that can be judged
        )
    for depth, name in enumerate(parts[module_end:], start=module_end):
        if hasattr(obj, name):
            obj = getattr(obj, name)
            continue
        fields = getattr(obj, "model_fields", None) or {}
        if name in fields or name in getattr(obj, "__annotations__", {}):
            return None if depth == len(parts) - 1 else "attribute of a field"
        return f"{'.'.join(parts[:depth])} has no attribute {name!r}"
    return None


def stale_dotted(text: str) -> list[tuple[str, str]]:
    """`(dotted path, reason)` for every unresolvable `fsm_llm.x.y` in *text* (each once)."""
    bad: dict[str, str] = {}
    for match in DOTTED.finditer(text):
        dotted = match.group(0)
        if dotted in bad or is_past(text, match.start()):
            continue
        reason = resolve_dotted(match.group(1))
        if reason is not None:
            bad[dotted] = reason
    return sorted(bad.items())


# `from fsm_llm.x import a, b as c` on one line, or `from fsm_llm.x import (` with the names on the
# following lines up to `)`. The module part is judged by `stale_dotted`; here each NAME must exist.
FROM_IMPORT = re.compile(
    r"^[ \t]*from[ \t]+(fsm_llm(?:\.[A-Za-z_]\w*)*)[ \t]+import[ \t]+"
    r"(\([^)]*\)|[^\n(]+)",
    re.MULTILINE,
)


def stale_import_names(text: str) -> list[tuple[str, str]]:
    """`("from <module> import <name>", reason)` for each imported name that the module lacks.

    Interface contract: only modules that resolve are judged (an unresolvable module is
    `stale_dotted`'s finding); a name resolves when it is an attribute of the imported module
    or a submodule of that package; `*` and a module that cannot be imported (an optional
    dependency is absent) are skipped. Pure apart from importing `fsm_llm` modules.
    """
    bad: dict[str, str] = {}
    for match in FROM_IMPORT.finditer(text):
        module_name = match.group(1)
        if resolve_dotted(module_name[len(PACKAGE) :]) is not None:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        body = re.sub(r"#[^\n]*", "", match.group(2)).strip("() \t\n")
        for part in body.replace("\n", " ").split(","):
            name = part.split(" as ")[0].strip()
            if not name or name == "*" or not name.isidentifier():
                continue
            if hasattr(module, name):
                continue
            try:
                if importlib.util.find_spec(f"{module_name}.{name}") is not None:
                    continue
            except (ImportError, ValueError):
                pass
            bad[f"from {module_name} import {name}"] = "the module has no such name"
    return sorted(bad.items())


# --- file mentions ------------------------------------------------------------------------


def _doc_scope(doc_rel: str, files: list[str]) -> list[str]:
    """The `.py` files a bare `name.py` in this doc may mean (see the module docstring)."""
    prefix = "src/fsm_llm/"
    if doc_rel.startswith(prefix):
        top = doc_rel[len(prefix) :].split("/")[0]
        if (PKG_DIR / top).is_dir():
            return [f for f in files if f.startswith(prefix + top + "/")] + [
                f for f in files if f.count("/") == 2 and f.startswith(prefix)
            ]
    if doc_rel.startswith("tests/"):
        return files
    return [
        f
        for f in files
        if (
            f.count("/") == 2 and f.startswith(prefix)
        )  # a top-level `src/fsm_llm/*.py`
        or f.startswith("scripts/")
        or "/" not in f
    ]


def _line_at(text: str, start: int) -> str:
    begin = text.rfind("\n", 0, start) + 1
    end = text.find("\n", start)
    return text[begin : end if end != -1 else len(text)]


def _named_by_its_directory(token: str, line: str, files: list[str]) -> bool:
    """True when *line* names a directory that holds a file called *token*.

    A row such as "`context/layout/` (`base.py`, `response.py`)" lists the modules of a package
    by their bare names; the bare name is right when the line also says which directory it is
    in. Any tail of the directory path counts (`layout/`, `context/layout/`).
    """
    for f in files:
        parts = f.split("/")
        if parts[-1] != token:
            continue
        dirs = parts[:-1]
        if any("/".join(dirs[i:]) + "/" in line for i in range(len(dirs))):
            return True
    return False


def stale_file_mentions(
    text: str, doc_rel: str, files: list[str], old: set[str]
) -> list[tuple[str, str]]:
    """`(token, reason)` for every stale `.py` mention of *text*, each once.

    Judged: a rooted path (`src/`, `fsm_llm/`, `tests/`, `scripts/`, `examples/`) must be a file;
    a token whose base name is an OLD module name must match the tail of a real path (when it
    has a directory) or, when it is bare, be a file of the doc's scope or of a directory the
    same line names. A mention the clause marks as past (`is_past`) is not judged.
    """
    bad: dict[str, str] = {}
    scope: list[str] | None = None
    for match in PY_TOKEN.finditer(text):
        token = match.group(1)
        if token in bad or "<" in token or is_past(text, match.start(1)):
            continue
        base = token.rsplit("/", 1)[-1][:-3]
        if token.startswith(("src/", "tests/", "scripts/", "examples/", "docs/")):
            if token not in files:
                bad[token] = "no such file"
            continue
        if token.startswith("fsm_llm/"):
            if "src/" + token not in files:
                bad[token] = "no such file"
            continue
        if base not in old:
            continue
        if "/" in token:
            if not any(f == token or f.endswith("/" + token) for f in files):
                bad[token] = "no module at that path"
            continue
        if scope is None:
            scope = _doc_scope(doc_rel, files)
        if any(f.rsplit("/", 1)[-1] == token for f in scope):
            continue
        if _named_by_its_directory(token, _line_at(text, match.start(1)), files):
            continue
        bad[token] = "a module of the old layout (name the new path)"
    return sorted(bad.items())


# --- the tests ----------------------------------------------------------------------------

_FILES = python_files()
_OLD = old_basenames()
_DOCS = doc_files()


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _problems(path: Path) -> list[str]:
    rel = _rel(path)
    text = path.read_text(encoding="utf-8")
    found = [f"{tok} ({why})" for tok, why in stale_dotted(text)]
    found += [f"{tok} ({why})" for tok, why in stale_import_names(text)]
    found += [
        f"{tok} ({why})" for tok, why in stale_file_mentions(text, rel, _FILES, _OLD)
    ]
    return [p for p in found if (rel, p.split(" (", 1)[0]) not in ALLOWED]


def test_every_current_doc_names_only_modules_of_the_layout() -> None:
    """One test over every doc, so the test count does not move when a guide is added."""
    report = {_rel(doc): _problems(doc) for doc in _DOCS}
    report = {rel: found for rel, found in report.items() if found}

    assert not report, (
        "these docs name a path the Section 5 layout no longer has:\n"
        + "\n".join(f"  {rel}: " + "; ".join(found) for rel, found in report.items())
        + "\nName the new dotted path or package (see tests/fixtures/module_path_map.json), "
        "write the old name as the past of a move (`the old ...`), or list the statement in "
        "ALLOWED with its reason."
    )


class TestTheScannerItself:
    """The scanners are pure over text, so they are checked on synthetic input."""

    def test_the_doc_set_has_the_expected_roots_and_no_historical_file(self) -> None:
        rels = {_rel(p) for p in _DOCS}

        assert {
            "README.md",
            "AGENTS.md",
            "src/fsm_llm/AGENTS.md",
            "docs/architecture.md",
        } <= rels
        assert "CHANGELOG.md" not in rels and "docs/2026_roadmap.md" not in rels
        assert not [r for r in rels if _is_historical(r)]

    def test_a_removed_module_and_a_removed_name_both_fail(self) -> None:
        bad = dict(
            stale_dotted(
                "see fsm_llm.validator and fsm_llm.engine.api.API.no_such_method "
                "and fsm_llm.pipeline.MessagePipeline"
            )
        )

        assert set(bad) == {
            "fsm_llm.validator",
            "fsm_llm.engine.api.API.no_such_method",
            "fsm_llm.pipeline.MessagePipeline",
        }

    def test_a_real_module_class_member_and_field_resolve(self) -> None:
        text = (
            "fsm_llm.API.converse fsm_llm.core.verify.validator.FSMValidator "
            "fsm_llm.engine.turn fsm_llm.core.definitions.State.kind "
            "fsm_llm.core.definitions.FSMDefinition.initial_state"
        )

        assert stale_dotted(text) == []

    def test_an_imported_name_the_module_lacks_fails(self) -> None:
        text = (
            "from fsm_llm import API, NoSuchName\n"
            "from fsm_llm.agents import (\n    ReactAgent,  # kept\n    GoneAgent,\n)\n"
            "from fsm_llm.engine import fsm, nothing_here\n"
            "from fsm_llm.core.definitions import *\n"
        )

        bad = [tok for tok, _ in stale_import_names(text)]

        assert bad == [
            "from fsm_llm import NoSuchName",
            "from fsm_llm.agents import GoneAgent",
            "from fsm_llm.engine import nothing_here",
        ]

    def test_a_package_name_in_a_file_name_is_not_a_dotted_path(self) -> None:
        assert stale_dotted("src/fsm_llm.egg-info and fsm_llm-okf and llm_fsm.x") == []

    def test_an_old_bare_module_mention_fails_at_the_root_and_passes_in_its_package(
        self,
    ) -> None:
        old = {"pipeline", "definitions", "api", "base"}
        files = [
            "src/fsm_llm/engine/turn/driver.py",
            "src/fsm_llm/reasoning/definitions.py",
            "src/fsm_llm/__init__.py",
        ]

        root = stale_file_mentions(
            "`pipeline.py` and `definitions.py`", "AGENTS.md", files, old
        )
        pkg = stale_file_mentions(
            "`pipeline.py` and `definitions.py`",
            "src/fsm_llm/reasoning/AGENTS.md",
            files,
            old,
        )

        assert [t for t, _ in root] == ["definitions.py", "pipeline.py"]
        assert [t for t, _ in pkg] == ["pipeline.py"]

    def test_a_path_with_a_directory_must_match_a_real_tail(self) -> None:
        old = {"engine", "tools"}
        files = [
            "src/fsm_llm/agents/tools.py",
            "src/fsm_llm/workflows/engine/engine.py",
        ]

        bad = stale_file_mentions(
            "`agents/tools.py` `workflows/engine.py` `src/fsm_llm/nothing.py` "
            "`fsm_llm/agents/tools.py` `workflows/engine/engine.py`",
            "docs/architecture.md",
            files,
            old,
        )

        assert dict(bad).keys() == {"workflows/engine.py", "src/fsm_llm/nothing.py"}

    def test_a_url_and_a_placeholder_are_not_judged(self) -> None:
        old = {"pipeline"}

        assert (
            stale_file_mentions(
                "https://example.org/a/pipeline.py and examples/<name>/pipeline.py",
                "x.md",
                [],
                old,
            )
            == []
        )

    def test_the_old_names_come_from_the_path_map(self) -> None:
        names = old_basenames()

        assert {"pipeline", "llm", "validator", "okf_bridge", "agent_skills"} <= names
        assert old_basenames(REPO / "no-such-map.json") == set()

    def test_every_allowed_statement_names_an_existing_doc_and_a_reason(self) -> None:
        for (rel, token), reason in ALLOWED.items():
            assert (REPO / rel).is_file(), rel
            assert token and reason.strip()
