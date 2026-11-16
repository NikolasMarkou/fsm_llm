"""One version, everywhere it is written (step 119 of plan d454fd62, roadmap row 0.1).

A release is cut by a version bump that touches a dozen files. The literal pins of
`test_app.py` and `TestVersionAlignment` are a deliberate release gate (they fail until the
bump is made on purpose); this file adds the check that the places which merely QUOTE the
version agree with the one place that defines it, so a bump that forgets the README badge, a
`> Covers` header, the changelog heading or a compare link fails here and names the file.

The source of truth is `pyproject.toml` (`[project] version`). Every other place is read as
text from the working tree (no import of the package for the file-level checks, so a stale
editable install cannot hide a stale file) and compared with it. The release section of the
CHANGELOG also opens with a short "Highlights" preamble that survives the cut of the GitHub
release body (`release.yml` cuts a body at 120,000 characters; the section is much longer);
`TestTheHighlights` pins its shape and its size, and the replay of the workflow's own extract
and cut steps is in `test_release_workflow.py::TestTheRealReleaseNotes`.
"""

from __future__ import annotations

import ast
import importlib
import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "fsm_llm"

_REPO_URL = "https://github.com/NikolasMarkou/fsm_llm"
_RELEASE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})(?: \[YANKED\])?$")

#: The Highlights block must stay short: it is the part of the release body every reader sees.
HIGHLIGHTS_MAX_CHARS = 8_000
HIGHLIGHTS_MIN_BULLETS = 5


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def _version() -> str:
    return tomllib.loads(_read("pyproject.toml"))["project"]["version"]


def _declared_in_version_file() -> str:
    """The string `src/fsm_llm/__version__.py` assigns to `__version__`, read without importing."""
    tree = ast.parse(_read("src/fsm_llm/__version__.py"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError(
        "no __version__ string assignment in src/fsm_llm/__version__.py"
    )


def _sections() -> list[tuple[str, int]]:
    """`(heading, 0-based line)` of every `## ` line of the CHANGELOG outside a fence."""
    out: list[tuple[str, int]] = []
    fenced = False
    for number, line in enumerate(_read("CHANGELOG.md").splitlines()):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            out.append((line.rstrip(), number))
    return out


def _release_headings() -> list[tuple[str, str]]:
    """`(version, date)` of the dated sections, newest first."""
    found = []
    for heading, _ in _sections():
        match = _RELEASE.match(heading)
        if match:
            found.append((match.group(1), match.group(2)))
    return found


def _release_section(version: str) -> list[str]:
    """The lines under `## [version] - date` up to the next `## ` heading."""
    lines = _read("CHANGELOG.md").splitlines()
    sections = _sections()
    for index, (heading, number) in enumerate(sections):
        match = _RELEASE.match(heading)
        if match and match.group(1) == version:
            end = sections[index + 1][1] if index + 1 < len(sections) else len(lines)
            return lines[number + 1 : end]
    raise AssertionError(f"no `## [{version}] - date` section in CHANGELOG.md")


def _preamble(version: str) -> list[str]:
    """The lines of a release section before its first `###` heading (blank ends trimmed)."""
    body = _release_section(version)
    head: list[str] = []
    for line in body:
        if line.startswith("### "):
            break
        head.append(line)
    while head and not head[-1].strip():
        head.pop()
    while head and not head[0].strip():
        head.pop(0)
    return head


class TestTheVersionIsDefinedOnce:
    def test_the_pin_of_the_release(self):
        # The deliberate gate: a literal, like the pins of `test_app.py`, so a bump is a decision.
        assert _version() == "0.13.0"

    def test_pyproject_and_the_version_file_agree(self):
        assert _declared_in_version_file() == _version()

    def test_the_version_is_a_plain_final_release(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", _version())

    def test_every_subpackage_re_exports_the_one_version(self):
        files = sorted(_SRC.glob("*/__version__.py"))
        assert len(files) >= 7, [f.parent.name for f in files]  # not vacuous
        for file in files:
            tree = ast.parse(file.read_text(encoding="utf-8"))
            assigned = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets
                )
            ]
            assert not assigned, f"{file.relative_to(_ROOT)} defines its own version"
            module = f"fsm_llm.{file.parent.name}.__version__"
            assert (
                importlib.import_module(module).__version__
                == _declared_in_version_file()
            )

    def test_the_package_reports_the_version_of_the_file(self):
        import fsm_llm

        assert fsm_llm.__version__ == _version()


class TestEveryQuotedVersionFollows:
    def test_the_readme_badge(self):
        assert f"img.shields.io/badge/version-{_version()}-" in _read("README.md")

    def test_the_covers_header_of_every_current_guide(self):
        guides = [
            p
            for p in sorted((_ROOT / "docs").glob("*.md"))
            if "> Covers FSM-LLM v" in "\n".join(p.read_text("utf-8").splitlines()[:5])
        ]
        assert len(guides) >= 11, [
            p.name for p in guides
        ]  # ten current guides and the landing page (step 123)
        for page in guides:
            header = re.search(
                r"^> Covers FSM-LLM v(\S+)$", page.read_text("utf-8"), re.M
            )
            assert header and header.group(1) == _version(), page.name

    def test_the_docs_agents_file_names_the_header(self):
        assert f"`> Covers FSM-LLM v{_version()}`" in _read("docs/AGENTS.md")
        assert f"for FSM-LLM v{_version()} " in _read("docs/AGENTS.md")

    def test_the_docs_readme_names_the_version_of_the_current_guides(self):
        text = _read("docs/README.md")
        assert f"current guides for version {_version()}" in text
        assert f"`> Covers FSM-LLM v{_version()}`" in text

    def test_the_status_line_of_the_concepts_page(self):
        assert f"* Version {_version()}. " in _read("docs/concepts.md")

    def test_the_agents_files_that_name_the_version(self):
        assert f"Python framework (v{_version()}," in _read("AGENTS.md")
        assert f"- Version {_version()} in `__version__.py`;" in _read(
            "src/fsm_llm/AGENTS.md"
        )
        assert f"version ({_version()}, shared by all subpackages)" in _read(
            "src/fsm_llm/README.md"
        )

    def test_the_contributing_examples_name_real_sections(self):
        text = _read("CONTRIBUTING.md")
        named = set(
            re.findall(
                r"changelog_check\.py (?:check --from|extract) v?(\d+\.\d+\.\d+)", text
            )
        )
        assert named, "CONTRIBUTING.md shows no versioned changelog_check example"
        existing = {version for version, _ in _release_headings()}
        assert named <= existing, sorted(named - existing)


class TestTheChangelogCarriesTheRelease:
    def test_the_newest_dated_section_is_the_version(self):
        headings = _release_headings()
        assert headings, "no dated section"
        assert headings[0][0] == _version(), headings[:2]

    def test_the_sections_are_newest_first_and_the_release_is_not_older_than_the_last(
        self,
    ):
        headings = _release_headings()
        versions = [tuple(int(x) for x in v.split(".")) for v, _ in headings]
        assert versions == sorted(versions, reverse=True)
        # History is frozen (0.8.0 carries an earlier date than 0.7.0): judge the newest pair only.
        assert headings[0][1] >= headings[1][1], headings[:2]

    def test_unreleased_comes_first(self):
        assert _sections()[0][0] == "## [Unreleased]"

    def test_the_compare_references_follow_the_sections(self):
        text = _read("CHANGELOG.md")
        headings = _release_headings()
        newest, previous = headings[0][0], headings[1][0]
        assert f"\n[Unreleased]: {_REPO_URL}/compare/v{newest}...HEAD\n" in text
        assert f"\n[{newest}]: {_REPO_URL}/compare/v{previous}...v{newest}\n" in text

    def test_the_release_date_is_an_iso_date(self):
        # The date is the day of the release: the owner edits it when the tag is cut on another day.
        version, date = _release_headings()[0]
        assert version == _version()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)


class TestTheHighlights:
    """The release section opens with a short preamble that every cut of the body keeps."""

    def test_the_section_opens_with_a_preamble_before_the_first_heading(self):
        head = _preamble(_version())
        assert head, "the release section has no Highlights preamble"
        assert head[0] == "**Highlights**", head[0]

    def test_it_is_short(self):
        text = "\n".join(_preamble(_version()))
        assert len(text) < HIGHLIGHTS_MAX_CHARS, len(text)
        assert len(text.splitlines()) <= 60

    def test_it_has_bullets_and_points_at_the_breaking_entries_and_the_full_list(self):
        head = _preamble(_version())
        text = "\n".join(head)
        bullets = [line for line in head if line.startswith("- ")]
        assert len(bullets) >= HIGHLIGHTS_MIN_BULLETS, len(bullets)
        assert "docs/upgrade-guide" not in text  # step 123: the guides are gone
        assert "entries marked BREAKING" in text
        assert "list of every change follows" in text

    def test_it_uses_only_absolute_links(self):
        """The release body does not resolve a repository-relative link against the tag."""
        text = "\n".join(_preamble(_version()))
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            assert target.startswith("https://"), target

    def test_it_holds_no_word_the_changelog_checker_reads_as_a_breaking_marker(self):
        # CL005 reads the whole section, preamble included.
        text = "\n".join(_preamble(_version()))
        assert not re.search(r"breaking[ \t\n]+(?:changes?|for)\b", text, re.I)
        assert "**BREAKING" not in text.upper().replace("**HIGHLIGHTS**", "")

    @pytest.mark.parametrize(
        "claim", ["`llm-fsm`", "Python 3.11", "3.15", "fsm_llm.testing"]
    )
    def test_it_names_what_the_release_is(self, claim: str):
        assert claim in "\n".join(_preamble(_version()))
