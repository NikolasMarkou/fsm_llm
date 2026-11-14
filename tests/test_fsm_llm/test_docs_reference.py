"""The generated reference, CLI and package pages of the docs site match the code.

Plan d454fd62, step 103 (roadmap row 0.7). Four kinds of page are derived from something else
in the repository, and each can drift without any test noticing until a strict build runs in CI:

* `docs/reference/<dotted.module>.md`: one mkdocstrings stub per entry of `MODULES` in
  `scripts/api_snapshot.py` (the modules whose public API `docs/api.txt` records). A module
  added to `MODULES` needs a page, a page for a module that left `MODULES` is stale, and the
  `:::` identifier must be the file name and must resolve to a module on disk.
* `docs/reference/api-snapshot.md`: `docs/api.txt` inside a fenced block. The file must not
  hold a fence of its own, or the page would close the block early.
* `docs/reference/cli.md`: every console script of `pyproject.toml` `[project.scripts]` has one
  section with its entry point, and no section names a script that is not declared.
* `docs/packages/<pkg>.md`: one page per `src/fsm_llm/<pkg>/README.md`, an include of that file.

Every one of those pages must be in the `nav` of `mkdocs.yml`. `mkdocs` is NOT imported here: the
checks are functions of a project root, so the mutation tests run them on a small scratch tree
with one defect each and the real tree is judged by `TestTheRealReference`. The strict build
(`make docs`) is the second line of defence: it fails on a `:::` name mkdocstrings cannot resolve.
"""

from __future__ import annotations

import ast
import importlib
import re
import tomllib
from pathlib import Path

import pytest

from tests.test_fsm_llm.test_docs_site import (
    _config,
    _docs_dir,
    _include_targets,
    _nav_files,
    _site_problems,
)

_ROOT = Path(__file__).resolve().parents[2]

#: The two pages of `docs/reference/` that are not mkdocstrings stubs.
_NOT_STUBS = {"api-snapshot", "cli"}

_AUTODOC = re.compile(r"^:::[ \t]+(\S+)[ \t]*$", re.M)
_SCRIPT_NAME = re.compile(r"(?<![\w-])fsm-llm(?:-[a-z]+)*(?![\w-])")
_SCRIPT_HEADING = re.compile(r"^##[ \t]+(fsm-llm(?:-[a-z]+)*)[ \t]*$", re.M)
_FENCE = "```"


# --------------------------------------------------------------------------------------
# What the pages are derived from
# --------------------------------------------------------------------------------------


def _modules(root: Path) -> tuple[str, ...]:
    """`MODULES` of `scripts/api_snapshot.py`, read with `ast` (the script is not imported)."""
    tree = ast.parse((root / "scripts" / "api_snapshot.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODULES" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple) and all(isinstance(m, str) for m in value)
            return value
    raise AssertionError("scripts/api_snapshot.py has no MODULES assignment")


def _scripts(root: Path) -> dict[str, str]:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return dict(data["project"]["scripts"])


def _module_file(root: Path, dotted: str) -> Path | None:
    """The source file of a dotted module under `src/`, without importing anything."""
    base = root / "src" / Path(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _stub_pages(root: Path) -> dict[str, Path]:
    """Reference pages that are stubs, keyed by file stem (the dotted module name)."""
    folder = root / "docs" / "reference"
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in sorted(folder.glob("*.md")) if p.stem not in _NOT_STUBS}


def _package_readmes(root: Path) -> dict[str, Path]:
    return {
        p.parent.name: p
        for p in sorted((root / "src" / "fsm_llm").glob("*/README.md"))
        if p.parent.is_dir()
    }


def _fenced_includes(text: str) -> list[tuple[str, bool]]:
    """Each include-markdown target of a page and whether it sits inside a fenced block."""
    inside = False
    found: list[tuple[str, bool]] = []
    for line in text.splitlines():
        if line.strip().startswith(_FENCE):
            inside = not inside
            continue
        for match in re.finditer(r"\{%\s*include(?:-markdown)?\s+(['\"])(.+?)\1", line):
            found.append((match.group(2), inside))
    return found


# --------------------------------------------------------------------------------------
# The checks: a function of a project root, one string per problem
# --------------------------------------------------------------------------------------


def _reference_problems(root: Path) -> list[str]:
    problems: list[str] = []
    config = _config(root)
    docs = _docs_dir(root, config)
    nav = set(_nav_files(config.get("nav")))

    def need_in_nav(page: Path) -> None:
        rel = page.relative_to(docs).as_posix()
        if rel not in nav:
            problems.append(f"{rel} is not in the nav of mkdocs.yml")

    # 1. one stub per MODULES entry, identifier equal to the file name, resolvable.
    modules = _modules(root)
    stubs = _stub_pages(root)
    for name in sorted(set(modules) - set(stubs)):
        problems.append(
            f"MODULES names {name}, but docs/reference/{name}.md is missing"
        )
    for name in sorted(set(stubs) - set(modules)):
        problems.append(
            f"docs/reference/{name}.md is a stale page: {name} is not in MODULES"
        )
    if len(set(modules)) != len(modules):
        problems.append("MODULES names a module twice")
    for name, page in stubs.items():
        ids = _AUTODOC.findall(page.read_text(encoding="utf-8"))
        if ids != [name]:
            problems.append(
                f"docs/reference/{name}.md must hold exactly one `::: {name}` block, "
                f"it holds {ids!r}"
            )
        if _module_file(root, name) is None:
            problems.append(f"docs/reference/{name}.md: {name} is not a module of src/")
        need_in_nav(page)

    # 2. the API snapshot page: api.txt inside a fence it cannot close.
    snapshot = docs / "reference" / "api-snapshot.md"
    api_txt = docs / "api.txt"
    if not snapshot.is_file():
        problems.append("docs/reference/api-snapshot.md is missing")
    else:
        need_in_nav(snapshot)
        includes = _fenced_includes(snapshot.read_text(encoding="utf-8"))
        if includes != [("../api.txt", True)]:
            problems.append(
                "docs/reference/api-snapshot.md must include '../api.txt' once, "
                f"inside a fenced block (found {includes!r})"
            )
    if api_txt.is_file() and _FENCE in api_txt.read_text(encoding="utf-8"):
        problems.append(
            "docs/api.txt holds a triple backtick: it would close the fence"
        )

    # 3. the CLI page: the declared console scripts, no more and no fewer.
    cli = docs / "reference" / "cli.md"
    scripts = _scripts(root)
    if not cli.is_file():
        problems.append("docs/reference/cli.md is missing")
    else:
        need_in_nav(cli)
        text = cli.read_text(encoding="utf-8")
        mentioned = set(_SCRIPT_NAME.findall(text))
        sections = set(_SCRIPT_HEADING.findall(text))
        for name in sorted(set(scripts) - mentioned):
            problems.append(f"cli.md does not name the console script {name}")
        for name in sorted(mentioned - set(scripts)):
            problems.append(f"cli.md names {name}, which is not in [project.scripts]")
        for name in sorted(set(scripts) - sections):
            problems.append(f"cli.md has no `## {name}` section")
        for name in sorted(sections - set(scripts)):
            problems.append(
                f"cli.md has a `## {name}` section for an undeclared script"
            )
        for name, target in sorted(scripts.items()):
            if target not in text:
                problems.append(
                    f"cli.md does not give the entry point of {name}: {target}"
                )
        for needle in ("python -m fsm_llm.reasoning", "--env-file", "--model", "130"):
            if needle not in text:
                problems.append(f"cli.md does not mention {needle!r}")

    # 4. one package page per subpackage README, an include of exactly that file.
    readmes = _package_readmes(root)
    folder = docs / "packages"
    pages = {p.stem: p for p in sorted(folder.glob("*.md"))} if folder.is_dir() else {}
    for pkg in sorted(set(readmes) - set(pages)):
        problems.append(f"src/fsm_llm/{pkg}/README.md has no docs/packages/{pkg}.md")
    for pkg in sorted(set(pages) - set(readmes)):
        problems.append(f"docs/packages/{pkg}.md has no src/fsm_llm/{pkg}/README.md")
    for pkg, page in pages.items():
        need_in_nav(page)
        want = f"../../src/fsm_llm/{pkg}/README.md"
        targets = _include_targets(page)
        if targets != [want]:
            problems.append(
                f"docs/packages/{pkg}.md must include {want!r} once, found {targets!r}"
            )
        elif not (page.parent / want).is_file():
            problems.append(
                f"docs/packages/{pkg}.md includes {want!r}, which is missing"
            )
    return problems


# --------------------------------------------------------------------------------------
# The real tree
# --------------------------------------------------------------------------------------


class TestTheRealReference:
    def test_no_reference_problem(self):
        assert _reference_problems(_ROOT) == []

    def test_the_whole_site_is_still_structurally_sound(self):
        assert _site_problems(_ROOT) == []

    def test_modules_hold_the_known_surface(self):
        modules = _modules(_ROOT)
        assert "fsm_llm" in modules
        assert "fsm_llm.testing" in modules
        assert "fsm_llm.core.verify.flow" in modules
        assert len(modules) == len(set(modules))

    def test_the_stub_pages_are_exactly_modules(self):
        assert set(_stub_pages(_ROOT)) == set(_modules(_ROOT))

    def test_every_stub_is_title_and_one_block(self):
        for name, page in _stub_pages(_ROOT).items():
            lines = [
                ln for ln in page.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            assert lines[0].startswith("# "), name
            assert lines[1] == f"::: {name}", name
            assert len(lines) == 2, f"{name}: a stub is a title and one block"

    def test_every_console_script_has_a_section(self):
        text = (_ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")
        assert set(_SCRIPT_HEADING.findall(text)) == set(_scripts(_ROOT))

    def test_every_subpackage_readme_has_a_page(self):
        readmes = _package_readmes(_ROOT)
        assert {
            "agents",
            "eval",
            "monitor",
            "reasoning",
            "synthesis",
            "testing",
            "workflows",
        } <= set(readmes)
        for pkg in readmes:
            assert (_ROOT / "docs" / "packages" / f"{pkg}.md").is_file(), pkg

    def test_the_snapshot_page_puts_api_txt_in_a_text_fence(self):
        text = (_ROOT / "docs" / "reference" / "api-snapshot.md").read_text(
            encoding="utf-8"
        )
        assert f"{_FENCE}text" in text
        assert _fenced_includes(text) == [("../api.txt", True)]
        assert _FENCE not in (_ROOT / "docs" / "api.txt").read_text(encoding="utf-8")

    def test_no_included_readme_carries_a_relative_link(self):
        """A relative link in an included README resolves against the README, not the page."""
        link = re.compile(r"\]\((?!https?://|#|mailto:)[^)]+\)|<img\b", re.I)
        offenders = {
            pkg: link.findall(path.read_text(encoding="utf-8"))
            for pkg, path in _package_readmes(_ROOT).items()
        }
        assert {k: v for k, v in offenders.items() if v} == {}


class TestEveryStubResolves:
    """A stale stub fails here, in the non-slow suite, even where mkdocs is not installed."""

    @pytest.mark.parametrize("dotted", _modules(_ROOT))
    def test_the_module_is_a_file_of_src(self, dotted):
        assert _module_file(_ROOT, dotted) is not None

    @pytest.mark.parametrize("dotted", _modules(_ROOT))
    def test_the_module_imports_and_declares_its_public_names(self, dotted):
        if dotted == "fsm_llm.monitor":
            pytest.importorskip(
                "fastapi"
            )  # the monitor package needs the dashboard stack
        module = importlib.import_module(dotted)
        public = getattr(module, "__all__", None)
        if public is not None:
            missing = [n for n in public if not hasattr(module, n)]
            assert missing == [], f"{dotted}.__all__ names {missing}"


# --------------------------------------------------------------------------------------
# Mutation tests: one defect each on a small good tree; the checks must name it
# --------------------------------------------------------------------------------------

_MODULES = ("fsm_llm", "fsm_llm.core.codes", "fsm_llm.agents")

_GOOD_NAV = """\
nav:
  - Home: index.md
  - Reference:
      - API snapshot: reference/api-snapshot.md
      - CLI: reference/cli.md
      - Modules:
          - fsm_llm: reference/fsm_llm.md
          - fsm_llm.core.codes: reference/fsm_llm.core.codes.md
          - fsm_llm.agents: reference/fsm_llm.agents.md
  - Packages:
      - agents: packages/agents.md
"""

_GOOD_CLI = """\
# Command line

## fsm-llm

Entry point `fsm_llm.__main__:main_cli`. Takes `--model` and `--env-file`. Exit codes 0, 1, 130.

## fsm-llm-validate

Entry point `fsm_llm.core.verify.cli:main_cli`.

## python -m fsm_llm.reasoning

The reasoning CLI has no console script.
"""


def _put(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _good_tree(tmp_path: Path) -> Path:
    root = tmp_path
    _put(root, "mkdocs.yml", "site_name: S\ndocs_dir: docs\n" + _GOOD_NAV)
    _put(
        root,
        "scripts/api_snapshot.py",
        "MODULES = (\n" + "".join(f'    "{m}",\n' for m in _MODULES) + ")\n",
    )
    _put(
        root,
        "pyproject.toml",
        "[project.scripts]\n"
        'fsm-llm = "fsm_llm.__main__:main_cli"\n'
        'fsm-llm-validate = "fsm_llm.core.verify.cli:main_cli"\n',
    )
    for name in _MODULES:
        _put(root, f"docs/reference/{name}.md", f"# {name}\n\n::: {name}\n")
    _put(root, "src/fsm_llm/__init__.py", "")
    _put(root, "src/fsm_llm/core/codes.py", "")
    _put(root, "src/fsm_llm/agents/__init__.py", "")
    _put(root, "docs/index.md", "# Home\n")
    _put(root, "docs/api.txt", "fsm_llm.API\n")
    _put(
        root,
        "docs/reference/api-snapshot.md",
        '# API snapshot\n\n```text\n{% include-markdown "../api.txt" %}\n```\n',
    )
    _put(root, "docs/reference/cli.md", _GOOD_CLI)
    _put(root, "src/fsm_llm/agents/README.md", "# fsm_llm.agents\n")
    _put(
        root,
        "docs/packages/agents.md",
        '{% include-markdown "../../src/fsm_llm/agents/README.md" %}\n',
    )
    return root


def _replace(root: Path, rel: str, old: str, new: str) -> None:
    text = (root / rel).read_text(encoding="utf-8")
    assert old in text, f"the mutant does not apply: {old!r} is not in {rel}"
    (root / rel).write_text(text.replace(old, new, 1), encoding="utf-8")


class TestDefectsAreCaught:
    def test_the_good_tree_has_no_problem(self, tmp_path):
        assert _reference_problems(_good_tree(tmp_path)) == []

    def test_a_missing_stub_page(self, tmp_path):
        root = _good_tree(tmp_path)
        (root / "docs/reference/fsm_llm.core.codes.md").unlink()
        got = _reference_problems(root)
        assert any("docs/reference/fsm_llm.core.codes.md is missing" in p for p in got)

    def test_a_stale_stub_page_for_a_module_out_of_modules(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(
            root,
            "docs/reference/fsm_llm.flow_analysis.md",
            "# x\n\n::: fsm_llm.flow_analysis\n",
        )
        got = _reference_problems(root)
        assert any("stale page" in p and "fsm_llm.flow_analysis" in p for p in got)

    def test_a_module_added_to_modules_without_a_page(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(root, "scripts/api_snapshot.py", ")\n", '    "fsm_llm.okf",\n)\n')
        got = _reference_problems(root)
        assert any("MODULES names fsm_llm.okf, but" in p for p in got)

    def test_a_stub_with_a_wrong_identifier(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(
            root,
            "docs/reference/fsm_llm.agents.md",
            "::: fsm_llm.agents",
            "::: fsm_llm.agent",
        )
        got = _reference_problems(root)
        assert any("must hold exactly one `::: fsm_llm.agents` block" in p for p in got)

    def test_a_stub_with_two_blocks(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(
            root,
            "docs/reference/fsm_llm.md",
            "# fsm_llm\n\n::: fsm_llm\n\n::: fsm_llm.agents\n",
        )
        got = _reference_problems(root)
        assert any("must hold exactly one" in p and "fsm_llm.md" in p for p in got)

    def test_a_stub_with_no_block(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(root, "docs/reference/fsm_llm.md", "# fsm_llm\n")
        got = _reference_problems(root)
        assert any("must hold exactly one" in p for p in got)

    def test_a_stub_for_a_module_that_is_not_in_src(self, tmp_path):
        root = _good_tree(tmp_path)
        (root / "src/fsm_llm/core/codes.py").unlink()
        got = _reference_problems(root)
        assert any("fsm_llm.core.codes is not a module of src/" in p for p in got)

    def test_a_stub_not_in_the_nav(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(
            root,
            "mkdocs.yml",
            "          - fsm_llm.agents: reference/fsm_llm.agents.md\n",
            "",
        )
        got = _reference_problems(root)
        assert "reference/fsm_llm.agents.md is not in the nav of mkdocs.yml" in got

    def test_the_cli_page_not_in_the_nav(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(root, "mkdocs.yml", "      - CLI: reference/cli.md\n", "")
        assert (
            "reference/cli.md is not in the nav of mkdocs.yml"
            in _reference_problems(root)
        )

    def test_a_package_page_not_in_the_nav(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(root, "mkdocs.yml", "      - agents: packages/agents.md\n", "")
        assert (
            "packages/agents.md is not in the nav of mkdocs.yml"
            in _reference_problems(root)
        )

    def test_a_script_missing_from_the_cli_page(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(
            root,
            "pyproject.toml",
            "[project.scripts]\n"
            'fsm-llm = "fsm_llm.__main__:main_cli"\n'
            'fsm-llm-validate = "fsm_llm.core.verify.cli:main_cli"\n'
            'fsm-llm-skills = "fsm_llm.skills.cli:main_cli"\n',
        )
        got = _reference_problems(root)
        assert "cli.md does not name the console script fsm-llm-skills" in got
        assert "cli.md has no `## fsm-llm-skills` section" in got

    def test_an_extra_script_on_the_cli_page(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(
            root,
            "docs/reference/cli.md",
            "## python -m",
            "## fsm-llm-ghost\n\nGone.\n\n## python -m",
        )
        got = _reference_problems(root)
        assert "cli.md names fsm-llm-ghost, which is not in [project.scripts]" in got
        assert "cli.md has a `## fsm-llm-ghost` section for an undeclared script" in got

    def test_an_extra_script_only_in_prose(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(
            root,
            "docs/reference/cli.md",
            "no console script.",
            "no console script, unlike fsm-llm-old.",
        )
        got = _reference_problems(root)
        assert "cli.md names fsm-llm-old, which is not in [project.scripts]" in got

    def test_a_stale_entry_point(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(
            root,
            "docs/reference/cli.md",
            "fsm_llm.core.verify.cli:main_cli",
            "fsm_llm.validator:main_cli",
        )
        got = _reference_problems(root)
        assert any(
            "does not give the entry point of fsm-llm-validate" in p for p in got
        )

    def test_a_cli_page_without_the_exit_code_note(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(root, "docs/reference/cli.md", "130", "13")
        assert "cli.md does not mention '130'" in _reference_problems(root)

    def test_a_package_include_target_that_does_not_exist(self, tmp_path):
        root = _good_tree(tmp_path)
        (root / "src/fsm_llm/agents/README.md").unlink()
        got = _reference_problems(root)
        assert any(
            "src/fsm_llm/agents/README.md has no" in p or "is missing" in p for p in got
        )

    def test_a_package_page_that_includes_the_wrong_file(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(root, "src/fsm_llm/eval/README.md", "# eval\n")
        _put(
            root,
            "docs/packages/eval.md",
            '{% include-markdown "../../src/fsm_llm/agents/README.md" %}\n',
        )
        _replace(
            root,
            "mkdocs.yml",
            "      - agents: packages/agents.md\n",
            "      - agents: packages/agents.md\n      - eval: packages/eval.md\n",
        )
        got = _reference_problems(root)
        assert any("docs/packages/eval.md must include" in p for p in got)

    def test_a_readme_without_a_package_page(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(root, "src/fsm_llm/eval/README.md", "# eval\n")
        got = _reference_problems(root)
        assert "src/fsm_llm/eval/README.md has no docs/packages/eval.md" in got

    def test_a_package_page_without_a_readme(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(
            root,
            "docs/packages/ghost.md",
            '{% include-markdown "../../src/fsm_llm/ghost/README.md" %}\n',
        )
        got = _reference_problems(root)
        assert "docs/packages/ghost.md has no src/fsm_llm/ghost/README.md" in got

    def test_the_snapshot_include_outside_a_fence(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(
            root,
            "docs/reference/api-snapshot.md",
            '# s\n\n{% include-markdown "../api.txt" %}\n',
        )
        got = _reference_problems(root)
        assert any("inside a fenced block" in p for p in got)

    def test_the_snapshot_include_aimed_elsewhere(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(root, "docs/reference/api-snapshot.md", "../api.txt", "../index.md")
        got = _reference_problems(root)
        assert any("must include '../api.txt'" in p for p in got)

    def test_api_txt_with_a_fence_of_its_own(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(root, "docs/api.txt", "fsm_llm.API\n```\nx\n")
        got = _reference_problems(root)
        assert "docs/api.txt holds a triple backtick: it would close the fence" in got

    def test_the_snapshot_page_missing(self, tmp_path):
        root = _good_tree(tmp_path)
        (root / "docs/reference/api-snapshot.md").unlink()
        assert "docs/reference/api-snapshot.md is missing" in _reference_problems(root)

    def test_modules_naming_a_module_twice(self, tmp_path):
        root = _good_tree(tmp_path)
        _replace(root, "scripts/api_snapshot.py", ")\n", '    "fsm_llm",\n)\n')
        assert "MODULES names a module twice" in _reference_problems(root)

    def test_modules_without_an_assignment_is_refused(self, tmp_path):
        root = _good_tree(tmp_path)
        _put(root, "scripts/api_snapshot.py", "OTHER = ()\n")
        with pytest.raises(AssertionError, match="no MODULES"):
            _reference_problems(root)
