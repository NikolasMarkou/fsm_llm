"""The README is the PyPI long description: it must stand alone, with no relative URL.

`pyproject.toml` sets `readme = "README.md"`, so PyPI renders this one file with no repository
around it: a relative image or link breaks there (plan d454fd62 D-020), and a mkdocs-only
construct (`{% include %}`, an admonition, a tab) shows as raw text. Step 105 trimmed the
README to the intro, the installation, two Quick Starts and a links block; the rest lives on
the docs site and in CONTRIBUTING.md.

`_readme_problems(text, root)` is a pure function of the README text and a project root (the
root is only read to check that a GitHub URL names a path that exists). The real README is one
test; the permanent mutants below run the same function over edited copies of it, so each rule
is proved to fire on a defect it names.

Contract: `_readme_problems(text, root=ROOT) -> list[str]`, one message per problem, `[]` when the
text is clean. It never raises on a text that merely breaks a rule.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from fsm_llm import FSMDefinition

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
SITE_URL = "https://nikolasmarkou.github.io/fsm_llm/"
REPO_URL = "https://github.com/NikolasMarkou/fsm_llm"
MIN_LINES, MAX_LINES = 100, 200

#: A fenced block, indented or not (the Installation steps nest theirs in a list).
_FENCE = re.compile(
    r"^[ \t]*```([^\n]*)\n(.*?)^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE
)
_INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*<?([^)\s>]*)")
_REF_DEFINITION = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*<?(\S+)", re.MULTILINE)
_HTML_URL = re.compile(r"""\b(?:src|href|srcset|poster)\s*=\s*["']([^"']*)["']""")
_REPO_PATH = re.compile(re.escape(REPO_URL) + r"/(?:blob|tree)/main/([^\s)#>\"']*)")
_RAW_IMAGE = re.compile(
    r"https://raw\.githubusercontent\.com/NikolasMarkou/fsm_llm/main/images/([^\s)\"']+)"
)
_MKDOCS_ONLY = (
    (re.compile(r"\{%|%\}"), "a Jinja directive"),
    (re.compile(r"^!!! ?\w", re.MULTILINE), "an admonition"),
    (re.compile(r'^=== "', re.MULTILINE), "a content tab"),
    (re.compile(r"^:::", re.MULTILINE), "a mkdocstrings block"),
    (re.compile(r"--8<--"), "a snippet include"),
    (re.compile(r"\{\s*[#.][\w-]+[^}]*\}\s*$", re.MULTILINE), "an attribute list"),
)
_PLAIN_LANG = re.compile(r"[A-Za-z0-9_+-]*")


def _outside_fences(text: str) -> str:
    """The text with every fenced block blanked (a `](` in code is not a link)."""
    return _FENCE.sub("", text)


def _is_absolute_or_anchor(target: str) -> bool:
    return target.startswith(("https://", "#"))


def _readme_problems(text: str, root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    prose = _outside_fences(text)

    targets = [m.group(1) for m in _INLINE_LINK.finditer(prose)]
    targets += [m.group(1) for m in _REF_DEFINITION.finditer(prose)]
    targets += [m.group(1) for m in _HTML_URL.finditer(prose)]
    for target in targets:
        if not _is_absolute_or_anchor(target):
            problems.append(f"relative or non-https URL {target!r}")

    for pattern, what in _MKDOCS_ONLY:
        if pattern.search(prose):
            problems.append(f"mkdocs-only syntax: {what}")

    blocks: list[tuple[str, str]] = []
    for m in _FENCE.finditer(text):
        info, body = m.group(1).strip(), m.group(2)
        if not _PLAIN_LANG.fullmatch(info):
            problems.append(f"fence info string {info!r} is not a plain language")
        blocks.append((info, body))

    full_fsms = [b for info, b in blocks if info == "json" and '"initial_state"' in b]
    if len(full_fsms) != 1:
        problems.append(f"{len(full_fsms)} full FSM blocks, expected exactly 1")
    for body in full_fsms:
        try:
            FSMDefinition(**json.loads(body))
        except Exception as err:
            problems.append(f"the full FSM block does not load: {err!r:.120}")

    if SITE_URL not in text:
        problems.append(f"no link to the docs site {SITE_URL}")

    for rel in _REPO_PATH.findall(text):
        if rel and not (root / rel).exists():
            problems.append(f"links to a repository path that does not exist: {rel}")
    for name in _RAW_IMAGE.findall(text):
        if not (root / "images" / name).is_file():
            problems.append(f"image {name!r} is not in images/")
    if not _RAW_IMAGE.search(text):
        problems.append("the logo is not an absolute raw.githubusercontent.com image")

    n = len(text.splitlines())
    if not MIN_LINES <= n <= MAX_LINES:
        problems.append(f"{n} lines, expected {MIN_LINES} to {MAX_LINES}")
    return problems


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


class TestTheRealReadme:
    def test_the_readme_stands_alone_on_pypi(self, readme):
        assert _readme_problems(readme) == []

    def test_pyproject_still_publishes_the_readme(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
        assert project["readme"] == "README.md"

    def test_the_readme_links_the_docs_site_and_the_repository(self, readme):
        assert SITE_URL in readme
        assert REPO_URL in readme

    def test_the_trimmed_sections_are_gone(self, readme):
        # They moved to docs/concepts.md, docs/quickstart.md and CONTRIBUTING.md.
        headings = re.findall(r"^#{1,3} (.+)$", readme, re.MULTILINE)
        for gone in (
            "Key Features",
            "How It Works",
            "Project Structure",
            "Contributing",
        ):
            assert not any(gone in h for h in headings), gone

    def test_the_readme_has_no_test_count_line(self, readme):
        # The count pin moved to CONTRIBUTING.md; a count here would drift with no pin.
        assert not re.search(r"\d[\d,]* tests\b", readme)


class TestDefectsAreCaught:
    """Each rule fires on a copy of the real README that breaks it (and only on that)."""

    @staticmethod
    def _mutate(readme: str, old: str, new: str) -> str:
        assert readme.count(old) >= 1, f"the mutant anchor {old!r} is gone"
        return readme.replace(old, new)

    @pytest.mark.parametrize(
        ("old", "new", "expect"),
        [
            (
                "https://raw.githubusercontent.com/NikolasMarkou/fsm_llm/main/images/fsm-llm-logo-1.png",
                "./images/fsm-llm-logo-1.png",
                "relative or non-https URL",
            ),
            (
                f"{REPO_URL}/blob/main/docs/quickstart.md",
                "docs/quickstart.md",
                "relative or non-https URL",
            ),
            (f"{REPO_URL}/blob/main/LICENSE", "LICENSE", "relative or non-https URL"),
            (
                f"{REPO_URL}/blob/main/LICENSE",
                "http://example.org/LICENSE",
                "non-https",
            ),
            (
                '<img alt="Tests" src="https://github.com',
                '<img alt="Tests" src="images/x.png" data-x="https://github.com',
                "relative or non-https URL",
            ),
            (
                f"{REPO_URL}/blob/main/docs/quickstart.md",
                f"{REPO_URL}/blob/main/docs/gone.md",
                "does not exist",
            ),
            ("```text\nbot:", "```text title\nbot:", "not a plain language"),
            (
                "```bash\n    git clone",
                "```bash title\n    git clone",
                "not a plain language",
            ),
            (
                "```python\nfrom fsm_llm import API\n",
                "```python {.x}\nfrom fsm_llm import API\n",
                "not a plain language",
            ),
            ("## License", "!!! note\n    x\n\n## License", "admonition"),
            ("## License", '{% include "x.md" %}\n\n## License', "Jinja"),
            (SITE_URL, "https://example.org/", "no link to the docs site"),
        ],
    )
    def test_a_defect_is_reported(self, readme, old, new, expect):
        problems = _readme_problems(self._mutate(readme, old, new))
        assert any(expect in p for p in problems), problems

    def test_the_full_fsm_block_removed(self, readme):
        mutated = readme.replace('"initial_state"', '"start_state"')
        assert any("0 full FSM blocks" in p for p in _readme_problems(mutated))

    def test_a_second_full_fsm_block(self, readme):
        bodies = [
            m.group(2)
            for m in _FENCE.finditer(readme)
            if '"initial_state"' in m.group(2)
        ]
        assert len(bodies) == 1
        mutated = readme + "\n```json\n" + bodies[0] + "```\n"
        assert any("2 full FSM blocks" in p for p in _readme_problems(mutated))

    def test_a_broken_full_fsm_block(self, readme):
        mutated = readme.replace(
            '"initial_state": "ask_name"', '"initial_state": "nowhere"'
        )
        assert any("does not load" in p for p in _readme_problems(mutated))

    def test_over_length(self, readme):
        mutated = readme + "\nline\n" * (MAX_LINES - len(readme.splitlines()) + 1)
        assert any("lines, expected" in p for p in _readme_problems(mutated))

    def test_under_length(self):
        assert any("lines, expected" in p for p in _readme_problems("# x\n"))

    def test_a_link_inside_a_fence_is_not_a_link(self, readme):
        mutated = readme + "\n```text\n[a](relative.md)\n```\n"
        assert not any("relative" in p for p in _readme_problems(mutated))

    def test_a_reference_definition_is_judged(self, readme):
        mutated = readme + "\n[ref]: docs/quickstart.md\n"
        assert any("relative" in p for p in _readme_problems(mutated))

    def test_a_missing_image_file(self, readme):
        mutated = readme.replace("fsm-llm-logo-1.png", "no-such-logo.png")
        assert any("not in images/" in p for p in _readme_problems(mutated))

    def test_an_anchor_link_is_allowed(self, readme):
        assert _readme_problems(readme + "\nSee [here](#installation).\n") == []


def _links(text: str) -> list[str]:
    return [m.group(1) for m in _INLINE_LINK.finditer(_outside_fences(text))]


class TestWhereTheTrimmedTextWent:
    """What the README lost is somewhere a reader can find it (step 105 of plan d454fd62)."""

    def test_quickstart_holds_the_four_moved_sections(self):
        text = (ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
        for heading in (
            "Assemble It Step by Step",
            "Create an Agent in One Line",
            "Run a Workflow (No LLM Needed)",
            "Solve a Problem with the Reasoning Engine",
        ):
            assert f"### {heading}\n" in text, heading

    def test_every_python_block_of_quickstart_parses(self):
        import ast

        text = (ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
        blocks = [m.group(2) for m in _FENCE.finditer(text) if m.group(1) == "python"]
        assert len(blocks) >= 9
        for body in blocks:
            ast.parse(body)

    def test_quickstart_fences_are_plain(self):
        text = (ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
        for m in _FENCE.finditer(text):
            assert _PLAIN_LANG.fullmatch(m.group(1).strip()), m.group(1)

    def test_contributing_holds_the_moved_sections(self):
        text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        for heading in (
            "Project structure",
            "Good places to help",
            "Tests, lint and types",
        ):
            assert f"## {heading}\n" in text, heading
        assert "make test           # Run full test suite (" in text
        # The clone-upgrade note moved to CONTRIBUTING.md in an earlier step and stays there.
        assert "Upgrading a clone from before the 2026-09-29 restructure" in text

    def test_contributing_has_no_relative_link(self):
        # The site includes this file from docs/contributing.md: a root-relative link breaks there.
        for target in _links((ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")):
            assert target.startswith(("https://", "#")), target

    def test_the_docs_indexes_name_no_deleted_file(self):
        for name in ("README.md", "AGENTS.md"):
            text = (ROOT / "docs" / name).read_text(encoding="utf-8")
            assert "strands_features" not in text, name
            assert "9 Markdown files" not in text, name
            assert "2026_roadmap.md" in text, name
