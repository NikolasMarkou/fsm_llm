"""The content pages and the navigation of the docs site are complete, without importing mkdocs.

Plan d454fd62, step 104 (roadmap row 0.7). `test_docs_site.py` judges the structure of the site
(every page in the nav, every include target exists) and `test_docs_links.py` judges its links.
This file pins what is specific to the content step:

* the nav is grouped Home, Getting started, Guides, Reference, Packages, Architecture,
  Decisions, Project (later steps add groups, for example the upgrade guides, between them: the
  check is on the order of the named groups, not on a closed list);
* the project pages `contributing`, `security`, `evaluate` and `changelog` are thin wrappers:
  one `include-markdown` of the root file and nothing else (the changelog wrapper also keeps its
  473 KB out of the search index with the Material front matter `search: exclude: true`);
* both roadmaps are published at their paths and unedited: no front matter, no directive (the
  files are records; a byte-level `git diff` over them is the step's own proof);
* `examples.md` links the `examples/` folder by absolute GitHub URL (the examples are not part
  of the site) and each linked path exists;
* the landing page links each new page.

`_content_problems(root)` is a function of a project root, so `TestDefectsAreCaught` runs it on
a copy of the real site with one defect each.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.test_fsm_llm.test_docs_links import _links
from tests.test_fsm_llm.test_docs_site import (
    _config,
    _docs_dir,
    _include_targets,
    _nav_files,
)

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = (
    _ROOT / "examples"
)  # read only: the copies of the site do not carry the examples

_GROUPS = [
    "Home",
    "Getting started",
    "Guides",
    "Reference",
    "Packages",
    "Architecture",
    "Decisions",
    "Project",
]
_GETTING_STARTED = ["quickstart.md", "concepts.md", "examples.md"]
_PROJECT = [
    "contributing.md",
    "security.md",
    "evaluate.md",
    "changelog.md",
    "2026_roadmap.md",
    "agents_roadmap.md",
]
#: wrapper page -> the root file it includes
_WRAPPERS = {
    "contributing.md": "../CONTRIBUTING.md",
    "security.md": "../SECURITY.md",
    "evaluate.md": "../EVALUATE.md",
    "changelog.md": "../CHANGELOG.md",
}
_ROADMAPS = ("2026_roadmap.md", "agents_roadmap.md")
_GITHUB_EXAMPLES = "https://github.com/NikolasMarkou/fsm_llm/tree/main/examples"
_LANDING_LINKS = [
    "concepts.md",
    "examples.md",
    "contributing.md",
    "security.md",
    "evaluate.md",
    "changelog.md",
]


def _group(nav: list[Any], title: str) -> Any:
    for item in nav:
        if isinstance(item, dict) and title in item:
            return item[title]
    return None


def _front_matter(text: str) -> str | None:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    return m.group(1) if m else None


def _content_problems(root: Path, examples: Path | None = None) -> list[str]:
    """One string per problem; `examples` is the examples folder (default `root/examples`)."""
    examples = examples or root / "examples"
    config = _config(root)
    docs = _docs_dir(root, config)
    nav = config.get("nav") or []
    problems: list[str] = []

    titles = [next(iter(i)) if isinstance(i, dict) else i for i in nav]
    positions = [titles.index(g) if g in titles else -1 for g in _GROUPS]
    for group, pos in zip(_GROUPS, positions, strict=True):
        if pos < 0:
            problems.append(f"nav has no group {group!r}")
    found = [p for p in positions if p >= 0]
    if found != sorted(found):
        problems.append(f"nav groups are out of order: {titles}")

    for title, wanted in (("Getting started", _GETTING_STARTED), ("Project", _PROJECT)):
        have = _nav_files(_group(nav, title))
        for name in wanted:
            if name not in have:
                problems.append(f"nav group {title!r} does not list {name}")

    for page, target in _WRAPPERS.items():
        path = docs / page
        if not path.is_file():
            problems.append(f"{page} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        if _include_targets(path) != [target]:
            problems.append(
                f"{page} must include exactly {target!r}, found {_include_targets(path)}"
            )
        body = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
        left = re.sub(r"\{%\s*include-markdown\s+.+?%\}", "", body, flags=re.S).strip()
        if left:
            problems.append(f"{page} is not a thin wrapper: text beside the include")
        if not (root / target.removeprefix("../")).is_file():
            problems.append(f"{page} includes {target!r}, which does not exist")
    changelog = docs / "changelog.md"
    if changelog.is_file():
        front = _front_matter(changelog.read_text(encoding="utf-8")) or ""
        if not re.search(r"^search:\s*\n\s+exclude:\s*true\s*$", front, re.M):
            problems.append(
                "changelog.md does not exclude itself from the search index"
            )

    for name in _ROADMAPS:
        path = docs / name
        if not path.is_file():
            problems.append(f"{name} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        if _front_matter(text) is not None or text.startswith("---"):
            problems.append(f"{name} has front matter: it is published as written")
        if _include_targets(path):
            problems.append(
                f"{name} holds an include directive: it is published as written"
            )
        if not text.startswith("# "):
            problems.append(f"{name} does not start with its own title")

    page = docs / "examples.md"
    if not page.is_file():
        problems.append("examples.md is missing")
    else:
        text = page.read_text(encoding="utf-8")
        linked = [d for _, d in _links(text) if "examples" in d]
        if not linked:
            problems.append("examples.md links no example folder")
        for dest in linked:
            if not dest.startswith(_GITHUB_EXAMPLES):
                problems.append(
                    f"examples.md links {dest!r}: use an absolute GitHub URL"
                )
                continue
            rel = dest[len(_GITHUB_EXAMPLES) :].strip("/")
            if not (examples / rel).exists():
                problems.append(
                    f"examples.md links {dest!r}, which is not in examples/"
                )

    landing = docs / "index.md"
    if landing.is_file():
        dests = {
            d.split("#")[0] for _, d in _links(landing.read_text(encoding="utf-8"))
        }
        for name in _LANDING_LINKS:
            if name not in dests:
                problems.append(f"index.md does not link {name}")
    return problems


class TestTheRealSite:
    def test_the_content_pages_and_the_nav_are_complete(self):
        assert _content_problems(_ROOT) == []

    def test_every_wrapper_page_is_in_the_nav(self):
        names = _nav_files(_config(_ROOT)["nav"])
        assert all(w in names for w in _WRAPPERS)

    def test_the_project_pages_name_the_root_files_they_show(self):
        # the page text a reader sees is the root file: a rename of a root file must fail here
        for target in _WRAPPERS.values():
            assert (_ROOT / target.removeprefix("../")).is_file()


# --------------------------------------------------------------------------------------
# A copy of the real site with one defect each
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _site_copy(tmp_path_factory):
    base = tmp_path_factory.mktemp("site")
    shutil.copytree(_ROOT / "docs", base / "docs")
    for name in (
        "mkdocs.yml",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "EVALUATE.md",
    ):
        shutil.copy(_ROOT / name, base / name)
    return base


def _fresh(site_copy: Path, tmp_path: Path) -> Path:
    root = tmp_path / "t"
    root.mkdir()
    shutil.copytree(site_copy / "docs", root / "docs")
    for name in (
        "mkdocs.yml",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "EVALUATE.md",
    ):
        shutil.copy(site_copy / name, root / name)
    return root


def _replace(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, (rel, old)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _swap_groups(root: Path) -> None:
    _replace(root, "mkdocs.yml", "  - Decisions: decisions/index.md\n", "")
    _replace(
        root,
        "mkdocs.yml",
        "  - Home: index.md\n",
        "  - Home: index.md\n  - Decisions: decisions/index.md\n",
    )


_MUTANTS = [
    pytest.param(
        lambda r: _replace(r, "mkdocs.yml", "  - Project:", "  - Projects:"),
        "nav has no group 'Project'",
        id="project-group-missing",
    ),
    pytest.param(_swap_groups, "out of order", id="groups-out-of-order"),
    pytest.param(
        lambda r: _replace(r, "mkdocs.yml", "      - Concepts: concepts.md\n", ""),
        "does not list concepts.md",
        id="concepts-not-in-getting-started",
    ),
    pytest.param(
        lambda r: _replace(r, "mkdocs.yml", "      - Security: security.md\n", ""),
        "does not list security.md",
        id="security-not-in-project",
    ),
    pytest.param(
        lambda r: (r / "docs" / "changelog.md").write_text(
            "# Changelog\n", encoding="utf-8"
        ),
        "must include exactly '../CHANGELOG.md'",
        id="changelog-include-removed",
    ),
    pytest.param(
        lambda r: _replace(
            r, "docs/changelog.md", "---\nsearch:\n  exclude: true\n---\n", ""
        ),
        "does not exclude itself from the search index",
        id="changelog-in-the-search-index",
    ),
    pytest.param(
        lambda r: (r / "docs" / "contributing.md").write_text(
            '# Extra\n\n{% include-markdown "../CONTRIBUTING.md" %}\n', encoding="utf-8"
        ),
        "not a thin wrapper",
        id="wrapper-with-extra-text",
    ),
    pytest.param(
        lambda r: _replace(
            r, "docs/evaluate.md", "../EVALUATE.md", "../CONTRIBUTING.md"
        ),
        "must include exactly '../EVALUATE.md'",
        id="wrapper-includes-the-wrong-file",
    ),
    pytest.param(
        lambda r: (r / "docs" / "2026_roadmap.md").write_text(
            "---\nhide:\n  - toc\n---\n"
            + (r / "docs" / "2026_roadmap.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        ),
        "has front matter",
        id="roadmap-gets-front-matter",
    ),
    pytest.param(
        lambda r: (r / "docs" / "agents_roadmap.md").write_text(
            '{% include-markdown "../CHANGELOG.md" %}\n', encoding="utf-8"
        ),
        "holds an include directive",
        id="roadmap-turned-into-an-include",
    ),
    pytest.param(
        lambda r: _replace(r, "docs/examples.md", _GITHUB_EXAMPLES, "../examples"),
        "use an absolute GitHub URL",
        id="examples-link-relative",
    ),
    pytest.param(
        lambda r: (r / "docs" / "examples.md").write_text(
            (r / "docs" / "examples.md").read_text(encoding="utf-8")
            + f"\n[x]({_GITHUB_EXAMPLES}/basic/no_such_example)\n",
            encoding="utf-8",
        ),
        "which is not in examples/",
        id="examples-link-to-a-missing-folder",
    ),
    pytest.param(
        lambda r: _replace(r, "docs/index.md", "(concepts.md)", "(quickstart.md)"),
        "index.md does not link concepts.md",
        id="landing-page-forgets-a-page",
    ),
]


class TestDefectsAreCaught:
    def test_the_copy_is_clean(self, _site_copy, tmp_path):
        assert _content_problems(_fresh(_site_copy, tmp_path), _EXAMPLES) == []

    @pytest.mark.parametrize(("mutate", "expected"), _MUTANTS)
    def test_each_defect_is_named(self, _site_copy, tmp_path, mutate, expected):
        root = _fresh(_site_copy, tmp_path)
        mutate(root)
        problems = _content_problems(root, _EXAMPLES)
        assert any(expected in p for p in problems), (expected, problems)
