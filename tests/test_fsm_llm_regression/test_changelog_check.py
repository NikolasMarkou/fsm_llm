"""`scripts/changelog_check.py`: the shape of CHANGELOG.md and the release-notes extractor.

Plan d454fd62, step 111 (roadmap row 0.5, CHANGELOG form). The script reads a CHANGELOG in the
Keep a Changelog form and judges only the sections being prepared (`[Unreleased]`, plus the
newest release when `[Unreleased]` is empty): the `###` headings and their order, the
`[Unreleased]` compare reference, the order of the sections, and every `**BREAKING:**` bullet
(it must link `docs/upgrade-guide/vX.Y.Z/<name>.md#<slug>` to an existing file and heading).
`extract VERSION` prints one section for the release notes.

Fixtures only: every test builds its CHANGELOG and its `docs/` tree under `tmp_path`. The real
`CHANGELOG.md` is deliberately NOT read here: `tests/test_packaging.py::TestTheRealChangelog`
enforces it (step 118), so one place owns that rule.

`extract --link-base URL` (step 118) makes the `docs/...` links of the notes absolute under a tag
URL for the GitHub release body; `LINK_CASES` and the CLI scenarios below pin it.

* `CASES` is a table of CHANGELOG texts and the finding codes each must give; `EXTRACTS` the same
  for `extract`; `LINK_CASES` the same for the link rewrite; `EXTRA` holds the scenarios that
  need the CLI. `scenario_problems(module, tmp)`
  returns the ids of the scenarios that fail on a loaded copy of the script: the real script
  returns the empty set, and each of the mutants (`MUTANTS`) is the real source with one edit
  and must fail the scenario it names. A check that cannot fail is not a check.
* The slug of a heading is a copy of the rule of `tests/test_fsm_llm/test_docs_links.py` (the
  script imports nothing of the repository); `TestSlugAgreement` holds the two equal on a table.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.test_fsm_llm.test_docs_links import _heading_ids as docs_heading_ids

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "changelog_check.py"

B = "**BREAKING:**"
REF_OK = "https://github.com/o/r/compare/v0.12.0...HEAD"


def _load_module(source: str, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(_SCRIPT)
    sys.modules[name] = module
    try:
        exec(compile(source, str(_SCRIPT), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def cc() -> types.ModuleType:
    assert _SCRIPT.is_file(), "scripts/changelog_check.py is missing"
    module = _load_module(_SCRIPT.read_text(encoding="utf-8"), "_changelog_check_real")
    yield module
    sys.modules.pop("_changelog_check_real", None)


# --------------------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------------------

GUIDES = {
    "upgrade-guide/v0.13.0/core.md": (
        "# Core\n\n## Renamed Thing\n\ntext\n\n## Second `thing`\n\ntext\n\n"
        "## Same\n\n## Same\n\n```\n## Fenced\n```\n"
    ),
    "upgrade-guide/v0.12.0/old.md": "# Old\n\n## Old rename\n\ntext\n",
}

HEAD = "# Changelog\n\nAll notable changes.\n\n"
REFS = (
    f"[Unreleased]: {REF_OK}\n"
    "[0.12.0]: https://github.com/o/r/compare/v0.11.0...v0.12.0\n"
    "[0.11.0]: https://github.com/o/r/releases/tag/v0.11.0\n"
)
RELEASES = (
    "## [0.12.0] - 2026-10-02\n\n### Added\n\n- Old thing.\n\n"
    "## [0.11.0] - 2026-09-29\n\n### Added\n\n- Older thing.\n\n"
)
ADDED = "### Added\n\n- A new thing.\n\n"


def doc(
    unreleased: str = ADDED,
    *,
    middle: str = RELEASES,
    refs: str = REFS,
    head: str = HEAD,
) -> str:
    return f"{head}## [Unreleased]\n\n{unreleased}{middle}{refs}"


def changed(*bullets: str) -> str:
    """An `[Unreleased]` body: one `### Changed` section holding the given bullet lines."""
    return "### Changed\n\n" + "\n".join(bullets) + "\n\n"


def link(target: str, text: str = "guide") -> str:
    return f"[{text}]({target})"


G = "docs/upgrade-guide/v0.13.0/core.md"
OK_LINK = link(f"{G}#renamed-thing")


@pytest.fixture
def docs_root(tmp_path: Path) -> Path:
    return _write_docs(tmp_path)


def _write_docs(tmp: Path) -> Path:
    root = tmp / "docs"
    for rel, text in GUIDES.items():
        page = root / rel
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(text, encoding="utf-8")
    return root


# --------------------------------------------------------------------------------------
# The table of CHANGELOG texts and the codes each must give
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    id: str
    text: str
    expected: tuple[str, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


_HISTORY_MESS = (
    "## [0.12.0] - 2026-10-02\n\n### Fixed\n\n- x\n\n### Added\n\n- y\n\n"
    "### Known open\n\n"
    f"- **Breaking:** old style.\n- {B} no guide at all.\n\n"
    "## [0.11.0] - 2026-09-29\n\n### Added\n\n- Older thing.\n\n"
)
_EMPTY_UNRELEASED_DOC = (
    HEAD
    + "## [Unreleased]\n\n"
    + "## [0.13.0] - 2026-11-20\n\n"
    + "### Changed\n\n"
    + "{bullet}\n\n"
    + RELEASES
    + "[Unreleased]: https://github.com/o/r/compare/v0.13.0...HEAD\n"
    + "[0.13.0]: https://github.com/o/r/compare/v0.12.0...v0.13.0\n"
)

CASES: list[Case] = [
    # ---- clean files -------------------------------------------------------------
    Case("clean-minimal", doc()),
    Case(
        "clean-all-six-headings-in-order",
        doc(
            "### Added\n\n- a\n\n### Changed\n\n- b\n\n### Deprecated\n\n- c\n\n"
            "### Removed\n\n- d\n\n### Fixed\n\n- e\n\n### Security\n\n- f\n\n"
        ),
    ),
    Case("clean-ref-main", doc(refs=REFS.replace("...HEAD", "...main"))),
    Case(
        "clean-no-breaking",
        doc(changed("- Not BREAKING: a note.", "- there is no BREAKING entry for it.")),
    ),
    # ---- CL001 -------------------------------------------------------------------
    Case(
        "bad-first-heading",
        doc().replace("## [Unreleased]", "## Unreleased"),
        ("CL001",),
    ),
    Case(
        "bad-date",
        doc(middle="").replace("## [Unreleased]\n", "## [0.13.0] - 2026-13-45\n"),
        ("CL001",),
    ),
    Case(
        "bad-unreleased-suffix",
        doc().replace("## [Unreleased]", "## [Unreleased] - soon"),
        ("CL001",),
    ),
    Case(
        "bad-middle-heading-in-scope",
        doc(
            middle=RELEASES.replace("## [0.12.0] - 2026-10-02", "## 0.12.0"),
            refs=REFS.replace("compare/v0.12.0...HEAD", "compare/v0.11.0...HEAD"),
        ),
        ("CL001",),
        {"from_version": "0.11.0"},
    ),
    Case("no-sections", "# Changelog\n\nNothing here.\n", ("CL001",)),
    Case("empty-file", "", ("CL001",)),
    Case(
        "fenced-fake-heading", doc(ADDED + "```\n## Not a section\n### Nope\n```\n\n")
    ),
    # ---- CL002, CL003, CL010 -----------------------------------------------------
    Case("order", doc("### Fixed\n\n- a\n\n### Added\n\n- b\n\n"), ("CL002",)),
    Case(
        "order-security-first",
        doc("### Security\n\n- a\n\n### Changed\n\n- b\n\n"),
        ("CL002",),
    ),
    Case("duplicate", doc(ADDED + ADDED), ("CL003",)),
    Case("unknown-heading", doc(ADDED + "### Notes\n\n- n\n\n"), ("CL010",)),
    Case(
        "unknown-heading-known-open",
        doc(ADDED + "### Known open\n\n- n\n\n"),
        ("CL010",),
    ),
    Case("lower-case-heading-is-unknown", doc("### added\n\n- a\n\n"), ("CL010",)),
    # ---- CL004 -------------------------------------------------------------------
    Case("ref-missing", doc(refs=REFS.split("\n", 1)[1]), ("CL004",)),
    Case(
        "ref-no-compare",
        doc(refs=REFS.replace("compare/v0.12.0...HEAD", "releases/tag/v0.12.0")),
        ("CL004",),
    ),
    Case(
        "ref-wrong-tag",
        doc(refs=REFS.replace("compare/v0.12.0...HEAD", "compare/v0.11.0...HEAD")),
        ("CL004",),
    ),
    Case(
        "ref-no-v",
        doc(refs=REFS.replace("compare/v0.12.0...HEAD", "compare/0.12.0...HEAD")),
        ("CL004",),
    ),
    Case("ref-wrong-end", doc(refs=REFS.replace("...HEAD", "...v0.13.0")), ("CL004",)),
    Case(
        "ref-in-fence-does-not-count",
        doc(refs="```\n" + REFS.split("\n")[0] + "\n```\n"),
        ("CL004",),
    ),
    # ---- CL005 -------------------------------------------------------------------
    Case("spelling-Breaking", doc(changed("- **Breaking:** x.")), ("CL005",)),
    Case("spelling-lower", doc(changed("- **breaking:** x.")), ("CL005",)),
    Case("spelling-no-colon", doc(changed("- **BREAKING** x.")), ("CL005",)),
    Case(
        "spelling-bold-prose",
        doc(changed("- x (**breaking for a caller that relied on y**).")),
        ("CL005",),
    ),
    Case(
        "prose-breaking-change",
        doc(changed("- This is a breaking change for callers.")),
        ("CL005",),
    ),
    Case(
        "prose-breaking-for",
        doc(changed("- It is breaking for callers of x.")),
        ("CL005",),
    ),
    Case(
        "prose-on-continuation-line",
        doc(changed("- x is", "  breaking for callers.")),
        ("CL005",),
    ),
    Case(
        "spelling-in-code-span-is-fine",
        doc(changed("- The text `**Breaking:**` is quoted.")),
        (),
    ),
    # ---- CL006 -------------------------------------------------------------------
    Case("no-guide", doc(changed(f"- {B} x.")), ("CL006",)),
    Case(
        "link-no-version-dir",
        doc(changed(f"- {B} x. " + link("docs/upgrade-guide/core.md#core"))),
        ("CL006",),
    ),
    Case("link-bare-text", doc(changed(f"- {B} x. See {G}#core.")), ("CL006",)),
    Case(
        "link-in-code-span", doc(changed(f"- {B} x. See `[g]({G}#core)`.")), ("CL006",)
    ),
    Case(
        "link-other-directory",
        doc(changed(f"- {B} x. " + link("docs/guides/core.md#core"))),
        ("CL006",),
    ),
    Case(
        "link-not-markdown-file",
        doc(changed(f"- {B} x. " + link("docs/upgrade-guide/v0.13.0/core.txt#core"))),
        ("CL006",),
    ),
    Case(
        "second-bullet-lacks-guide",
        doc(changed(f"- {B} a. {OK_LINK}", f"- {B} b.")),
        ("CL006",),
    ),
    Case(
        "adjacent-bullets-first-linked-second-not",
        doc(changed(f"- {B} a. {OK_LINK}", f"- {B} b. no link")),
        ("CL006",),
    ),
    # ---- CL007, CL008, CL011 -----------------------------------------------------
    Case(
        "link-missing-file",
        doc(changed(f"- {B} x. " + link("docs/upgrade-guide/v0.13.0/nope.md#a"))),
        ("CL007",),
    ),
    Case(
        "link-missing-heading",
        doc(changed(f"- {B} x. " + link(f"{G}#nope"))),
        ("CL008",),
    ),
    Case("link-no-slug", doc(changed(f"- {B} x. " + link(G))), ("CL008",)),
    Case("link-empty-slug", doc(changed(f"- {B} x. " + link(f"{G}#"))), ("CL008",)),
    Case(
        "link-slug-is-case-sensitive",
        doc(changed(f"- {B} x. " + link(f"{G}#Renamed-Thing"))),
        ("CL008",),
    ),
    Case(
        "link-heading-inside-fence",
        doc(changed(f"- {B} x. " + link(f"{G}#fenced"))),
        ("CL008",),
    ),
    Case(
        "link-other-version",
        doc(
            changed(f"- {B} x. " + link("docs/upgrade-guide/v0.12.0/old.md#old-rename"))
        ),
        ("CL011",),
    ),
    Case(
        "link-next-version-option",
        doc(changed(f"- {B} x. {OK_LINK}")),
        ("CL011",),
        {"next_version": "0.14.0"},
    ),
    Case(
        "link-next-version-option-ok",
        doc(
            changed(f"- {B} x. " + link("docs/upgrade-guide/v0.12.0/old.md#old-rename"))
        ),
        (),
        {"next_version": "0.12.0"},
    ),
    # ---- valid guide links -------------------------------------------------------
    Case("valid-link", doc(changed(f"- {B} x. {OK_LINK}"))),
    Case(
        "valid-link-capitalised-heading",
        doc(changed(f"- {B} x. " + link(f"{G}#renamed-thing"))),
    ),
    Case("valid-link-dot-slash", doc(changed(f"- {B} x. " + link(f"./{G}#core")))),
    Case(
        "valid-link-absolute-url",
        doc(changed(f"- {B} x. " + link(f"https://github.com/o/r/blob/main/{G}#core"))),
    ),
    Case("valid-link-angle-brackets", doc(changed(f"- {B} x. " + link(f"<{G}#core>")))),
    Case("valid-link-with-title", doc(changed(f"- {B} x. " + f'[g]({G}#core "Core")'))),
    Case(
        "valid-link-code-heading",
        doc(changed(f"- {B} x. " + link(f"{G}#second-thing"))),
    ),
    Case(
        "valid-link-duplicate-heading-ids",
        doc(
            changed(
                f"- {B} a. " + link(f"{G}#same"), f"- {B} b. " + link(f"{G}#same_1")
            )
        ),
    ),
    Case(
        "valid-link-percent-encoded",
        doc(changed(f"- {B} x. " + link(f"{G}#renamed%2Dthing"))),
    ),
    Case(
        "valid-link-continuation-line",
        doc(changed(f"- {B} x.", f"  See {OK_LINK} for the steps.")),
    ),
    Case(
        "valid-link-after-blank-line",
        doc(f"### Changed\n\n- {B} x.\n\n  See {OK_LINK}.\n\n"),
    ),
    Case(
        "valid-link-two-links-one-bullet",
        doc(changed(f"- {B} x. {OK_LINK} and " + link(f"{G}#core"))),
    ),
    Case(
        "two-links-one-bad",
        doc(changed(f"- {B} x. {OK_LINK} and " + link(f"{G}#nope"))),
        ("CL008",),
    ),
    Case(
        "two-bullets-one-line-each",
        doc(changed(f"- {B} a. {OK_LINK}", f"- {B} b. " + link(f"{G}#core"))),
    ),
    Case(
        "two-markers-in-one-bullet-one-link",
        doc(changed(f"- {B} a. and {B} b. {OK_LINK}")),
    ),
    Case(
        "nested-bullets-each-checked",
        doc(changed("- Parent.", f"  - {B} a. {OK_LINK}", f"  - {B} b.")),
        ("CL006",),
    ),
    Case(
        "star-bullets-and-numbers",
        doc(changed(f"* {B} a.", f"1. {B} b. {OK_LINK}")),
        ("CL006",),
    ),
    # ---- the marker is not a marker inside code ----------------------------------
    Case(
        "marker-in-code-span",
        doc(changed(f"- The text `{B}` is quoted, no guide needed.")),
    ),
    Case("marker-in-double-backticks", doc(changed(f"- The text ``{B}`` is quoted."))),
    Case(
        "marker-in-fence",
        doc(f"### Changed\n\n- Example:\n\n  ```\n  - {B} x\n  ```\n\n"),
    ),
    Case(
        "marker-in-multiline-code-span",
        doc(changed("- A `code", f"  {B} span` ends here.")),
    ),
    # ---- CL009 -------------------------------------------------------------------
    Case(
        "unreleased-last",
        HEAD + RELEASES + "## [Unreleased]\n\n" + ADDED + REFS,
        ("CL009",),
    ),
    Case("no-unreleased", HEAD + RELEASES + REFS.split("\n", 1)[1], ("CL009",)),
    Case(
        "versions-not-descending",
        doc(
            middle=(
                "## [0.11.0] - 2026-09-29\n\n### Added\n\n- b\n\n## [0.12.0] - 2026-10-02\n\n### Added\n\n- a\n\n"
            ),
            refs=REFS.replace("compare/v0.12.0...HEAD", "compare/v0.11.0...HEAD"),
        ),
        ("CL009",),
        {"from_version": "0.12.0"},
    ),
    Case(
        "versions-duplicated",
        doc(middle=RELEASES.replace("[0.11.0] - 2026-09-29", "[0.12.0] - 2026-09-29")),
        ("CL009",),
        {"from_version": "0.12.0"},
    ),
    Case("versions-descending-wide-scope-ok", doc(), (), {"from_version": "0.11.0"}),
    # ---- scope -------------------------------------------------------------------
    Case("history-is-frozen", doc(middle=_HISTORY_MESS)),
    Case(
        "from-version-widens",
        doc(middle=_HISTORY_MESS),
        ("CL002", "CL005", "CL006", "CL010"),
        {"from_version": "0.12.0"},
    ),
    Case(
        "from-version-v-prefix",
        doc(middle=_HISTORY_MESS),
        ("CL002", "CL005", "CL006", "CL010"),
        {"from_version": "v0.12.0"},
    ),
    Case(
        "from-unreleased-is-narrowest",
        doc(middle=_HISTORY_MESS),
        (),
        {"from_version": "Unreleased"},
    ),
    Case(
        "from-version-includes-unreleased",
        doc(changed(f"- {B} x."), middle=_HISTORY_MESS),
        ("CL002", "CL005", "CL006", "CL006", "CL010"),
        {"from_version": "0.12.0"},
    ),
    Case(
        "nonempty-unreleased-skips-newest-release",
        doc(changed(f"- {B} x. {OK_LINK}"), middle=_HISTORY_MESS),
    ),
    Case(
        "empty-unreleased-includes-newest-release",
        _EMPTY_UNRELEASED_DOC.format(bullet=f"- {B} x. " + link(f"{G}#core")),
    ),
    Case(
        "empty-unreleased-checks-newest-release",
        _EMPTY_UNRELEASED_DOC.format(bullet=f"- {B} x."),
        ("CL006",),
    ),
    Case(
        "empty-unreleased-release-uses-its-own-version-dir",
        _EMPTY_UNRELEASED_DOC.format(
            bullet=f"- {B} x. " + link("docs/upgrade-guide/v0.12.0/old.md#old-rename")
        ),
        ("CL011",),
    ),
    Case(
        "empty-unreleased-newest-release-ref-tag",
        _EMPTY_UNRELEASED_DOC.replace(
            "compare/v0.13.0...HEAD", "compare/v0.12.0...HEAD"
        ).format(bullet=f"- {B} x. " + link(f"{G}#core")),
        ("CL004",),
    ),
    # ---- line ends ---------------------------------------------------------------
    Case("crlf", doc(changed(f"- {B} x. {OK_LINK}")).replace("\n", "\r\n")),
    Case("cr-only", doc(changed(f"- {B} x. {OK_LINK}")).replace("\n", "\r")),
    Case("crlf-finding", doc(changed(f"- {B} x.")).replace("\n", "\r\n"), ("CL006",)),
    Case("no-final-newline", doc().rstrip("\n")),
]


def _codes(findings: list[Any]) -> list[str]:
    return [f.code for f in findings]


def run_case(module: types.ModuleType, case: Case, docs: Path) -> bool:
    got = _codes(module.check_text(case.text, docs_root=docs, **case.kwargs))
    return sorted(got) == sorted(case.expected)


class TestFindings:
    def test_there_are_cases_for_every_code(self):
        codes = {c for case in CASES for c in case.expected}
        assert codes == {f"CL{n:03d}" for n in range(1, 12)}

    def test_case_ids_are_unique(self):
        ids = [c.id for c in CASES]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
    def test_case(self, cc: types.ModuleType, docs_root: Path, case: Case):
        got = _codes(cc.check_text(case.text, docs_root=docs_root, **case.kwargs))
        assert sorted(got) == sorted(case.expected), cc.check_text(
            case.text, docs_root=docs_root, **case.kwargs
        )

    def test_a_missing_from_version_raises_keyerror(self, cc, docs_root):
        with pytest.raises(KeyError):
            cc.check_text(doc(), docs_root=docs_root, from_version="9.9.9")

    def test_the_docs_root_may_not_exist(self, cc, tmp_path):
        found = cc.check_text(
            doc(changed(f"- {B} x. {OK_LINK}")), docs_root=tmp_path / "nowhere"
        )
        assert _codes(found) == ["CL007"]

    def test_a_guide_that_is_a_directory_is_missing(self, cc, tmp_path):
        root = tmp_path / "docs"
        (root / "upgrade-guide" / "v0.13.0" / "core.md").mkdir(parents=True)
        found = cc.check_text(doc(changed(f"- {B} x. {OK_LINK}")), docs_root=root)
        assert _codes(found) == ["CL007"]

    def test_a_guide_that_is_not_utf8_is_missing(self, cc, tmp_path):
        root = tmp_path / "docs" / "upgrade-guide" / "v0.13.0"
        root.mkdir(parents=True)
        (root / "core.md").write_bytes(b"# Core\n\xff\xfe\n")
        found = cc.check_text(
            doc(changed(f"- {B} x. {OK_LINK}")), docs_root=tmp_path / "docs"
        )
        assert _codes(found) == ["CL007"]

    def test_an_explicit_id_and_an_html_anchor_count_as_headings(self, cc, tmp_path):
        root = tmp_path / "docs" / "upgrade-guide" / "v0.13.0"
        root.mkdir(parents=True)
        (root / "core.md").write_text(
            '# Core\n\n## Thing {#explicit-id}\n\n<a id="html-anchor"></a>\n',
            encoding="utf-8",
        )
        text = doc(
            changed(
                f"- {B} a. " + link(f"{G}#explicit-id"),
                f"- {B} b. " + link(f"{G}#html-anchor"),
            )
        )
        assert cc.check_text(text, docs_root=tmp_path / "docs") == []

    def test_findings_carry_line_numbers_and_stay_sorted(self, cc, docs_root):
        # line 9 holds the bullet, line 11 the duplicate heading: the bullet is reported first
        # although the heading checks run first.
        text = doc(f"### Added\n\n- {B} x.\n\n### Added\n\n- y\n\n")
        found = cc.check_text(text, docs_root=docs_root)
        assert [(f.line, f.code) for f in found] == [(9, "CL006"), (11, "CL003")]

    def test_the_reference_finding_points_at_the_unreleased_heading(
        self, cc, docs_root
    ):
        found = cc.check_text(doc(refs=""), docs_root=docs_root)
        assert [(f.line, f.code) for f in found] == [(5, "CL004")]

    def test_the_reference_finding_points_at_the_bad_reference_line(
        self, cc, docs_root
    ):
        text = doc(refs=REFS.replace("compare/v0.12.0...HEAD", "x"))
        found = cc.check_text(text, docs_root=docs_root)
        assert [f.code for f in found] == ["CL004"]
        assert text.split("\n")[found[0].line - 1].startswith("[Unreleased]:")

    def test_a_bullet_finding_points_at_the_first_line_of_the_bullet(
        self, cc, docs_root
    ):
        text = doc(changed(f"- {B} x", "  continues", "  and ends"))
        found = cc.check_text(text, docs_root=docs_root)
        assert [f.line for f in found] == [9]

    def test_a_nested_bullet_finding_points_at_the_nested_line(self, cc, docs_root):
        text = doc(changed("- Parent", f"  - {B} child"))
        found = cc.check_text(text, docs_root=docs_root)
        assert [f.line for f in found] == [10]

    def test_one_finding_per_defect_no_duplicates(self, cc, docs_root):
        text = doc(changed(f"- {B} a. {B} b."))
        assert _codes(cc.check_text(text, docs_root=docs_root)) == ["CL006"]


# --------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------

_UNRELEASED_BODY = "### Added\n\n- A new thing.\n"
_BODY_0_12 = "### Added\n\n- Old thing.\n"
_BODY_0_11 = "### Added\n\n- Older thing.\n"


@dataclass(frozen=True)
class Extract:
    id: str
    version: str
    expected: str | None
    with_heading: bool = False
    text: str = field(default_factory=doc)


EXTRACTS: list[Extract] = [
    Extract("first", "Unreleased", _UNRELEASED_BODY),
    Extract("first-lowercase", "unreleased", _UNRELEASED_BODY),
    Extract("middle", "0.12.0", _BODY_0_12),
    Extract("last-stops-before-the-reference-block", "0.11.0", _BODY_0_11),
    Extract("v-prefix", "v0.12.0", _BODY_0_12),
    Extract("missing", "9.9.9", None),
    Extract("missing-unreleased", "Unreleased", None, text=RELEASES + REFS),
    Extract(
        "with-heading", "0.12.0", "## [0.12.0] - 2026-10-02\n\n" + _BODY_0_12, True
    ),
    Extract(
        "unreleased-with-heading",
        "Unreleased",
        "## [Unreleased]\n\n" + _UNRELEASED_BODY,
        True,
    ),
    Extract("does-not-include-the-next-heading", "0.12.0", _BODY_0_12),
    Extract(
        "trailing-blank-lines-are-dropped",
        "0.12.0",
        _BODY_0_12,
        text=doc(middle=RELEASES.replace("- Old thing.\n\n", "- Old thing.\n\n\n\n\n")),
    ),
    Extract(
        "leading-blank-lines-are-dropped",
        "0.12.0",
        _BODY_0_12,
        text=doc(
            middle=RELEASES.replace(
                "[0.12.0] - 2026-10-02\n\n", "[0.12.0] - 2026-10-02\n\n\n\n"
            )
        ),
    ),
    Extract("empty-body", "Unreleased", "", text=doc("")),
    Extract("crlf", "0.12.0", _BODY_0_12, text=doc().replace("\n", "\r\n")),
    Extract(
        "fenced-heading-stays-in-the-body",
        "Unreleased",
        "### Added\n\n```\n## Not a section\n```\n\n- x\n",
        text=doc("### Added\n\n```\n## Not a section\n```\n\n- x\n\n"),
    ),
    Extract(
        "last-section-without-reference-block", "0.11.0", _BODY_0_11, text=doc(refs="")
    ),
    Extract(
        "last-section-with-blank-lines-before-references",
        "0.11.0",
        _BODY_0_11,
        text=doc(refs="\n\n" + REFS),
    ),
    Extract(
        "an-inner-link-reference-line-stays",
        "0.12.0",
        "### Added\n\n- Old thing.\n\n[x]: https://example.com\n\n- more\n",
        text=doc(
            middle=RELEASES.replace(
                "- Old thing.\n", "- Old thing.\n\n[x]: https://example.com\n\n- more\n"
            )
        ),
    ),
    Extract("version-is-matched-whole", "0.1.0", None),
    Extract(
        "a-bad-heading-is-not-a-version",
        "0.13.0",
        None,
        text=doc(middle="## [0.13.0] - nodate\n\nx\n\n"),
    ),
]


BASE = "https://github.com/o/r/blob/v0.13.0/"


@dataclass(frozen=True)
class LinkCase:
    """`rewrite_doc_links(text, base)` must return `expected`."""

    id: str
    text: str
    expected: str
    base: str = BASE


_GUIDE = "docs/upgrade-guide/v0.13.0/core.md#renamed-thing"

LINK_CASES: list[LinkCase] = [
    LinkCase(
        "plain",
        f"- [Upgrade guide]({_GUIDE})\n",
        f"- [Upgrade guide]({BASE}{_GUIDE})\n",
    ),
    LinkCase("dot-slash", "[a](./docs/x.md)", f"[a]({BASE}docs/x.md)"),
    LinkCase("angle", "[a](<docs/up guide.md>)", f"[a](<{BASE}docs/up guide.md>)"),
    LinkCase(
        "two-per-line",
        "[a](docs/a.md) and [b](docs/b.md)",
        f"[a]({BASE}docs/a.md) and [b]({BASE}docs/b.md)",
    ),
    LinkCase(
        "code-span",
        "`](docs/x.md)` and [a](docs/y.md)",
        f"`](docs/x.md)` and [a]({BASE}docs/y.md)",
    ),
    LinkCase(
        "code-span-over-a-line-break",
        "the `form\n](docs/x.md)` stays, [a](docs/y.md) goes",
        f"the `form\n](docs/x.md)` stays, [a]({BASE}docs/y.md) goes",
    ),
    LinkCase(
        "double-backtick-span",
        "``a ` ](docs/x.md)`` and [a](docs/y.md)",
        f"``a ` ](docs/x.md)`` and [a]({BASE}docs/y.md)",
    ),
    LinkCase(
        "fence",
        "[a](docs/a.md)\n```text\n[k](docs/k.md)\n```\n[b](docs/b.md)\n",
        f"[a]({BASE}docs/a.md)\n```text\n[k](docs/k.md)\n```\n[b]({BASE}docs/b.md)\n",
    ),
    LinkCase(
        "tilde-fence",
        "~~~\n[k](docs/k.md)\n~~~\n[b](docs/b.md)",
        f"~~~\n[k](docs/k.md)\n~~~\n[b]({BASE}docs/b.md)",
    ),
    LinkCase(
        "an-unclosed-fence-holds-the-rest",
        "[a](docs/a.md)\n```\n[k](docs/k.md)\n[l](docs/l.md)\n",
        f"[a]({BASE}docs/a.md)\n```\n[k](docs/k.md)\n[l](docs/l.md)\n",
    ),
    LinkCase(
        "other-targets-are-left-alone",
        "[a](CONTRIBUTING.md) [b](src/docs/x.md) [c](https://x.org/docs/y.md) [d](#docs/e) docs/f.md",
        "[a](CONTRIBUTING.md) [b](src/docs/x.md) [c](https://x.org/docs/y.md) [d](#docs/e) docs/f.md",
    ),
    LinkCase(
        "no-slash-in-the-base",
        "[a](docs/x.md)",
        "[a](https://h/blob/t/docs/x.md)",
        "https://h/blob/t",
    ),
    LinkCase(
        "a-text-with-no-link-is-byte-identical", "a \r\n  b  \n\n", "a \r\n  b  \n\n"
    ),
    LinkCase(
        "an-absolute-target-is-not-rewritten-again",
        f"[a]({BASE}docs/x.md)",
        f"[a]({BASE}docs/x.md)",
    ),
]


def run_link_case(module: types.ModuleType, case: LinkCase) -> bool:
    once = module.rewrite_doc_links(case.text, case.base)
    return once == case.expected and module.rewrite_doc_links(once, case.base) == once


def run_extract(module: types.ModuleType, case: Extract) -> bool:
    return (
        module.extract_text(case.text, case.version, with_heading=case.with_heading)
        == case.expected
    )


class TestExtract:
    def test_case_ids_are_unique(self):
        ids = [c.id for c in EXTRACTS]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("case", EXTRACTS, ids=[c.id for c in EXTRACTS])
    def test_case(self, cc: types.ModuleType, case: Extract):
        got = cc.extract_text(case.text, case.version, with_heading=case.with_heading)
        assert got == case.expected

    def test_the_output_ends_in_exactly_one_newline(self, cc):
        out = cc.extract_text(doc(), "0.12.0")
        assert out.endswith("\n") and not out.endswith("\n\n")

    def test_extract_equals_the_check_scope_headings(self, cc):
        # the body of a section holds its `###` headings, so a section can be re-checked alone
        assert cc.extract_text(doc(), "Unreleased").startswith("### Added")


class TestLinkBase:
    def test_case_ids_are_unique(self):
        ids = [c.id for c in LINK_CASES]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("case", LINK_CASES, ids=[c.id for c in LINK_CASES])
    def test_case(self, cc: types.ModuleType, case: LinkCase):
        once = cc.rewrite_doc_links(case.text, case.base)
        assert once == case.expected
        assert cc.rewrite_doc_links(once, case.base) == once  # idempotent

    def test_the_rewrite_runs_on_the_extracted_notes_before_anything_else(
        self, cc, tmp_path
    ):
        text = doc(changed(f"- {B} x. {OK_LINK}"))
        path = _write(tmp_path, text)
        code, plain = run_main(cc, ["extract", "Unreleased", "--file", str(path)])
        code2, linked = run_main(
            cc, ["extract", "Unreleased", "--file", str(path), "--link-base", BASE]
        )
        assert (code, code2) == (0, 0)
        assert linked == cc.rewrite_doc_links(plain, BASE) != plain
        assert f"]({BASE}{G}#renamed-thing)" in linked and "](docs/" not in linked

    def test_the_link_base_applies_to_the_heading_line_too(self, cc, tmp_path):
        path = _write(tmp_path, doc(changed(f"- {B} x. {OK_LINK}")))
        code, out = run_main(
            cc,
            [
                "extract",
                "Unreleased",
                "--file",
                str(path),
                "--with-heading",
                "--link-base",
                BASE,
            ],
        )
        assert code == 0 and out.startswith("## [Unreleased]\n\n")

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "ftp://h/",
            "github.com/o/r/",
            "https://",
            "file:///tmp/x/",
            "https:// x/",
        ],
    )
    def test_a_base_that_is_not_an_http_address_is_a_usage_error(
        self, cc, tmp_path, bad, capsys
    ):
        path = _write(tmp_path, doc())
        argv = ["extract", "Unreleased", "--file", str(path), "--link-base", bad]
        assert run_main(cc, argv) == (2, "")
        assert "--link-base" in capsys.readouterr().err

    def test_check_does_not_take_the_option(self, cc, tmp_path):
        path = _write(tmp_path, doc())
        assert run_main(cc, ["check", "--file", str(path), "--link-base", BASE])[0] == 2

    def test_a_trailing_slash_is_added_once(self, cc, tmp_path):
        path = _write(tmp_path, doc(changed(f"- {B} x. {OK_LINK}")))
        for base in ("https://h/blob/t", "https://h/blob/t/"):
            _, out = run_main(
                cc, ["extract", "Unreleased", "--file", str(path), "--link-base", base]
            )
            assert "](https://h/blob/t/docs/" in out and "t//docs" not in out

    def test_the_rewrite_is_pure_and_changes_nothing_else(self, cc):
        text = "# t\n\n- a [b](docs/x.md) `c` **d**\n\n```\ncode\n```\n"
        out = cc.rewrite_doc_links(text, BASE)
        assert out.replace(BASE, "") == text

    def test_the_real_changelog_notes_hold_no_relative_docs_link_after_it(self, cc):
        """Every link of the real [Unreleased] section: absolute after the rewrite."""
        import re

        text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        notes = cc.extract_text(text, "Unreleased")
        if notes is None or not notes:
            pytest.skip("[Unreleased] is empty (the release was cut)")
        out = cc.rewrite_doc_links(notes, BASE)
        no_fence = re.sub(r"^(```+|~~~+).*?^\1[ \t]*$", "", out, flags=re.S | re.M)
        no_code = re.sub(r"(`+)(.+?)\1", "", no_fence, flags=re.S)
        assert re.findall(r"\]\(<?(?:\./)?docs/", no_code) == []
        assert out.count(f"]({BASE}docs/") >= 1


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------


def run_main(module: types.ModuleType, argv: list[str]) -> tuple[int, str]:
    out: list[str] = []
    code = module.main(argv, stdout=out.append)
    return code, "".join(out)


def _write(tmp: Path, text: str, name: str = "CHANGELOG.md") -> Path:
    path = tmp / name
    path.write_bytes(text.encode("utf-8"))
    return path


class TestCli:
    def test_a_clean_file_exits_0_and_prints_nothing(self, cc, tmp_path):
        _write_docs(tmp_path)
        path = _write(tmp_path, doc(changed(f"- {B} x. {OK_LINK}")))
        assert run_main(cc, ["check", "--file", str(path)]) == (0, "")

    def test_findings_exit_1_one_per_line_in_the_documented_form(self, cc, tmp_path):
        _write_docs(tmp_path)
        path = _write(tmp_path, doc(changed(f"- {B} x.", "- **Breaking:** y.")))
        code, out = run_main(cc, ["check", "--file", str(path)])
        assert code == 1
        lines = out.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith(f"{path}:9: CL006: ")
        assert lines[1].startswith(f"{path}:10: CL005: ")

    def test_the_docs_root_defaults_to_docs_beside_the_changelog(self, cc, tmp_path):
        _write_docs(tmp_path)
        path = _write(tmp_path, doc(changed(f"- {B} x. {OK_LINK}")))
        assert run_main(cc, ["check", "--file", str(path)])[0] == 0
        assert (
            run_main(
                cc, ["check", "--file", str(path), "--docs-root", str(tmp_path / "x")]
            )[0]
            == 1
        )

    def test_an_explicit_docs_root_is_used(self, cc, tmp_path):
        other = tmp_path / "elsewhere"
        other.mkdir()
        _write_docs(other)
        path = _write(tmp_path, doc(changed(f"- {B} x. {OK_LINK}")))
        argv = ["check", "--file", str(path), "--docs-root", str(other / "docs")]
        assert run_main(cc, argv)[0] == 0

    def test_next_version_and_from_are_passed_on(self, cc, tmp_path):
        _write_docs(tmp_path)
        path = _write(
            tmp_path, doc(changed(f"- {B} x. {OK_LINK}"), middle=_HISTORY_MESS)
        )
        base = ["check", "--file", str(path)]
        assert run_main(cc, base)[0] == 0
        assert run_main(cc, [*base, "--next-version", "0.14.0"])[0] == 1
        assert run_main(cc, [*base, "--from", "0.12.0"])[0] == 1
        assert run_main(cc, [*base, "--from", "Unreleased"])[0] == 0

    def test_a_from_version_that_does_not_exist_is_a_usage_error(
        self, cc, tmp_path, capsys
    ):
        path = _write(tmp_path, doc())
        assert run_main(cc, ["check", "--file", str(path), "--from", "9.9.9"])[0] == 2
        assert "9.9.9" in capsys.readouterr().err

    def test_an_unreadable_file_exits_2(self, cc, tmp_path, capsys):
        assert run_main(cc, ["check", "--file", str(tmp_path / "none.md")])[0] == 2
        assert "cannot read" in capsys.readouterr().err
        assert (
            run_main(cc, ["extract", "0.1.0", "--file", str(tmp_path / "none.md")])[0]
            == 2
        )

    def test_a_file_that_is_not_utf8_exits_2(self, cc, tmp_path):
        path = tmp_path / "CHANGELOG.md"
        path.write_bytes(b"\xff\xfe## [Unreleased]\n")
        assert run_main(cc, ["check", "--file", str(path)])[0] == 2

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["frobnicate"],
            ["check", "--bogus"],
            ["check", "--next-version", "1.0"],
            ["check", "--next-version", "v1.0.0"],
            ["extract"],
            ["extract", "0.1.0", "--bogus"],
            ["check", "--file"],
        ],
    )
    def test_usage_errors_exit_2(self, cc, argv, capsys):
        assert run_main(cc, argv)[0] == 2
        assert "error" in capsys.readouterr().err

    def test_help_exits_0(self, cc, capsys):
        assert run_main(cc, ["--help"])[0] == 0
        assert "extract" in capsys.readouterr().out

    def test_extract_prints_the_body(self, cc, tmp_path):
        path = _write(tmp_path, doc())
        code, out = run_main(cc, ["extract", "0.12.0", "--file", str(path)])
        assert (code, out) == (0, _BODY_0_12)

    def test_extract_with_heading(self, cc, tmp_path):
        path = _write(tmp_path, doc())
        code, out = run_main(
            cc, ["extract", "v0.12.0", "--file", str(path), "--with-heading"]
        )
        assert (code, out) == (0, "## [0.12.0] - 2026-10-02\n\n" + _BODY_0_12)

    def test_extract_of_a_missing_version_exits_1(self, cc, tmp_path, capsys):
        path = _write(tmp_path, doc())
        assert run_main(cc, ["extract", "9.9.9", "--file", str(path)]) == (1, "")
        assert "9.9.9" in capsys.readouterr().err

    def test_extract_of_an_empty_section_exits_1(self, cc, tmp_path, capsys):
        path = _write(tmp_path, doc(""))
        assert run_main(cc, ["extract", "Unreleased", "--file", str(path)]) == (1, "")
        assert "no body" in capsys.readouterr().err

    def test_crlf_and_lf_files_give_the_same_output(self, cc, tmp_path):
        _write_docs(tmp_path)
        text = doc(changed(f"- {B} x.", "- **Breaking:** y."))
        lf = _write(tmp_path, text, "lf.md")
        crlf = _write(tmp_path, text.replace("\n", "\r\n"), "crlf.md")
        out_lf = run_main(cc, ["check", "--file", str(lf)])[1].replace("lf.md", "X")
        out_crlf = run_main(cc, ["check", "--file", str(crlf)])[1].replace(
            "crlf.md", "X"
        )
        assert out_lf == out_crlf != ""
        a = run_main(cc, ["extract", "0.12.0", "--file", str(lf)])
        b = run_main(cc, ["extract", "0.12.0", "--file", str(crlf)])
        assert a == b

    def test_the_default_file_is_the_changelog_beside_the_scripts_folder(self, cc):
        assert cc.DEFAULT_FILE == _ROOT / "CHANGELOG.md"

    def test_default_run_uses_the_real_stdout(self, cc, tmp_path, capsys):
        path = _write(tmp_path, doc())
        assert cc.main(["extract", "0.12.0", "--file", str(path)]) == 0
        assert capsys.readouterr().out == _BODY_0_12

    def test_running_it_as_a_script(self, tmp_path):
        _write_docs(tmp_path)
        path = _write(tmp_path, doc(changed(f"- {B} x.")))
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        done = subprocess.run(
            [sys.executable, str(_SCRIPT), "check", "--file", str(path)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            timeout=60,
        )
        assert done.returncode == 1
        assert f"{path}:9: CL006: " in done.stdout
        done = subprocess.run(
            [sys.executable, str(_SCRIPT), "extract", "0.12.0", "--file", str(path)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            timeout=60,
        )
        assert (done.returncode, done.stdout) == (0, _BODY_0_12)
        done = subprocess.run(
            [sys.executable, str(_SCRIPT), "bogus"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            timeout=60,
        )
        assert done.returncode == 2


# --------------------------------------------------------------------------------------
# Robustness: determinism, read-only, standard library only, slug agreement
# --------------------------------------------------------------------------------------


def _tree_state(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class TestRobustness:
    def test_the_output_is_the_same_twice(self, cc, tmp_path):
        _write_docs(tmp_path)
        text = doc(
            changed(f"- {B} a.", "- **Breaking:** b.", f"- {B} c. " + link(f"{G}#no")),
        )
        path = _write(tmp_path, text)
        first = run_main(cc, ["check", "--file", str(path)])
        second = run_main(cc, ["check", "--file", str(path)])
        assert first == second and first[1]

    def test_it_never_writes_a_file(self, cc, tmp_path):
        _write_docs(tmp_path)
        path = _write(tmp_path, doc(changed(f"- {B} a.")))
        before = _tree_state(tmp_path)
        run_main(cc, ["check", "--file", str(path)])
        run_main(cc, ["extract", "0.12.0", "--file", str(path)])
        run_main(cc, ["check", "--file", str(tmp_path / "missing.md")])
        assert _tree_state(tmp_path) == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["CHANGELOG.md", "docs"]

    def test_the_script_has_no_write_call(self):
        tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
        banned = {
            "write_text",
            "write_bytes",
            "unlink",
            "mkdir",
            "touch",
            "rename",
            "rmdir",
            "chmod",
            "open",
            "remove",
            "makedirs",
            "symlink_to",
        }
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (
                    fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                )
                if name in banned:
                    found.add(name)
        assert not found, found

    def test_it_imports_only_the_standard_library(self):
        tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "no relative import"
                names.add((node.module or "").split(".")[0])
        assert names, "no imports found"
        assert names <= set(sys.stdlib_module_names) | {"__future__"}, sorted(
            names - set(sys.stdlib_module_names)
        )

    def test_it_reaches_no_network_and_starts_no_process(self):
        text = _SCRIPT.read_text(encoding="utf-8")
        for word in (
            "urllib.request",
            "socket",
            "subprocess",
            "http.client",
            "requests",
            "os.system",
        ):
            assert word not in text, word

    def test_it_is_a_main_guarded_script(self):
        text = _SCRIPT.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env python")
        assert 'if __name__ == "__main__":' in text

    def test_the_docstring_documents_every_code(self, cc):
        documented = set()
        for line in cc.__doc__.splitlines():
            stripped = line.strip()
            if stripped.startswith("CL") and stripped[2:5].isdigit():
                documented.add(stripped[:5])
        assert documented == {f"CL{n:03d}" for n in range(1, 12)}

    def test_the_script_codes_are_the_documented_ones(self):
        import re

        used = set(re.findall(r'"(CL\d{3})"', _SCRIPT.read_text(encoding="utf-8")))
        assert used == {f"CL{n:03d}" for n in range(1, 12)}

    def test_the_six_headings_are_the_keep_a_changelog_ones(self, cc):
        assert cc.KAC_HEADINGS == (
            "Added",
            "Changed",
            "Deprecated",
            "Removed",
            "Fixed",
            "Security",
        )

    def test_the_marker_is_exactly_the_documented_one(self, cc):
        assert cc.BREAKING_MARKER == "**BREAKING:**"


HEADING_TABLES = [
    "# Title",
    "## Renamed Thing",
    "## `code` heading",
    "## Dup\n\n## Dup\n\n## Dup",
    "## Unicode café and ñ",
    "## A  --  b",
    "## Heading {#custom-id}\n\n## Other",
    "## **Bold** and _em_ and __strong__",
    "## [link](x.md) text and ![img](a.png)",
    "```\n## fenced\n```\n## after",
    "~~~\n## tilde fenced\n~~~\n## after tilde",
    "## 3. Numbers & symbols!",
    "##   spaced   ",
    "### Section_with_underscores",
    "## trailing ###",
    "## <b>html</b> heading",
    "## Same\n\n## same\n\n## Same_1",
    "## --leading dashes--",
    "## éè only accents",
    "#NoSpace\n\n## ok",
    "    ## indented code",
    "   ## three spaces",
]


class TestSlugAgreement:
    """The script's slug rule equals the docs link test's (the one the site build follows)."""

    @pytest.mark.parametrize("text", HEADING_TABLES)
    def test_heading_ids_agree(self, cc, text):
        assert cc.heading_ids(text) == docs_heading_ids(text)

    def test_the_slug_table(self, cc):
        table = {
            "Renamed Thing": "renamed-thing",
            "Second thing": "second-thing",
            "A  --  b": "a-b",
            "3. Numbers & symbols!": "3-numbers-symbols",
            "Café": "cafe",
            "UPPER": "upper",
            "  padded  ": "padded",
            "": "",
        }
        for heading, slug in table.items():
            assert cc.slugify(heading) == slug, heading


# --------------------------------------------------------------------------------------
# Mutants: each edit of the real source must fail the scenario it names
# --------------------------------------------------------------------------------------


def _scn_exit_codes(module: types.ModuleType, tmp: Path) -> bool:
    _write_docs(tmp)
    clean = _write(tmp, doc(), "clean.md")
    dirty = _write(tmp, doc(changed(f"- {B} x.")), "dirty.md")
    return (
        run_main(module, ["check", "--file", str(clean)])[0] == 0
        and run_main(module, ["check", "--file", str(dirty)])[0] == 1
    )


def _scn_exit_unreadable(module: types.ModuleType, tmp: Path) -> bool:
    return run_main(module, ["check", "--file", str(tmp / "none.md")])[0] == 2


def _scn_exit_extract_missing(module: types.ModuleType, tmp: Path) -> bool:
    path = _write(tmp, doc(), "c.md")
    return run_main(module, ["extract", "9.9.9", "--file", str(path)])[0] == 1


def _scn_exit_extract_empty(module: types.ModuleType, tmp: Path) -> bool:
    path = _write(tmp, doc(""), "c.md")
    return run_main(module, ["extract", "Unreleased", "--file", str(path)])[0] == 1


def _scn_cli_output(module: types.ModuleType, tmp: Path) -> bool:
    _write_docs(tmp)
    path = _write(tmp, doc(changed(f"- {B} x.")), "c.md")
    code, out = run_main(module, ["check", "--file", str(path)])
    return code == 1 and out.startswith(f"{path}:9: CL006: ") and out.endswith("\n")


def _scn_sorted(module: types.ModuleType, tmp: Path) -> bool:
    docs = _write_docs(tmp)
    text = doc(f"### Added\n\n- {B} x.\n\n### Added\n\n- y\n\n")
    return [(f.line, f.code) for f in module.check_text(text, docs_root=docs)] == [
        (9, "CL006"),
        (11, "CL003"),
    ]


def _scn_slug_table(module: types.ModuleType, tmp: Path) -> bool:
    return module.heading_ids("## Renamed Thing\n\n## Same\n\n## Same\n") == [
        "renamed-thing",
        "same",
        "same_1",
    ]


def _scn_cli_link_base(module: types.ModuleType, tmp: Path) -> bool:
    path = _write(tmp, doc(changed(f"- {B} x. {OK_LINK}")), "c.md")
    argv = ["extract", "Unreleased", "--file", str(path)]
    plain = run_main(module, argv)
    linked = run_main(module, [*argv, "--link-base", BASE])
    return (
        plain[0] == 0
        and f"]({G}#renamed-thing)" in plain[1]
        and linked[0] == 0
        and f"]({BASE}{G}#renamed-thing)" in linked[1]
        and "](docs/" not in linked[1]
    )


def _scn_cli_link_base_usage(module: types.ModuleType, tmp: Path) -> bool:
    path = _write(tmp, doc(), "c.md")
    argv = ["extract", "Unreleased", "--file", str(path), "--link-base"]
    return run_main(module, [*argv, "ftp://h/"])[0] == 2 and (
        run_main(module, [*argv, "https://h/t/"])[0] == 0
    )


EXTRA: dict[str, Callable[[types.ModuleType, Path], bool]] = {
    "cli-link-base": _scn_cli_link_base,
    "cli-link-base-usage": _scn_cli_link_base_usage,
    "exit-codes": _scn_exit_codes,
    "exit-unreadable": _scn_exit_unreadable,
    "exit-extract-missing": _scn_exit_extract_missing,
    "exit-extract-empty": _scn_exit_extract_empty,
    "cli-output": _scn_cli_output,
    "findings-sorted": _scn_sorted,
    "slug-table": _scn_slug_table,
}


def scenario_problems(module: types.ModuleType, tmp: Path) -> set[str]:
    """Ids of the scenarios the (possibly mutated) module fails; empty for the real script."""
    docs = _write_docs(tmp)
    bad: set[str] = set()
    for case in CASES:
        try:
            ok = run_case(module, case, docs)
        except Exception:
            ok = False
        if not ok:
            bad.add(case.id)
    for ex in EXTRACTS:
        try:
            ok = run_extract(module, ex)
        except Exception:
            ok = False
        if not ok:
            bad.add("extract-" + ex.id)
    for link_case in LINK_CASES:
        try:
            ok = run_link_case(module, link_case)
        except Exception:
            ok = False
        if not ok:
            bad.add("links-" + link_case.id)
    for name, fn in EXTRA.items():
        sub = tmp / name
        sub.mkdir()
        try:
            ok = fn(module, sub)
        except Exception:
            ok = False
        if not ok:
            bad.add(name)
    return bad


# (label, scenario id that must fail, old text, new text); `old` occurs exactly once
MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "BREAKING is matched case-insensitively",
        "spelling-Breaking",
        "if m.group(0) == BREAKING_MARKER:",
        "if m.group(0).upper() == BREAKING_MARKER:",
    ),
    (
        "heading order is not checked",
        "order",
        "if rank < highest:",
        "if False:",
    ),
    (
        "a duplicate heading is not reported",
        "duplicate",
        "if name in seen:",
        "if False:",
    ),
    (
        "an unknown heading is not reported",
        "unknown-heading",
        "if name not in KAC_HEADINGS:",
        "if False:",
    ),
    (
        "the compare reference is not required",
        "ref-missing",
        "    if unreleased is None:\n        return\n    newest",
        "    if True:\n        return\n    newest",
    ),
    (
        "the compare reference tag is not compared with the newest release",
        "ref-wrong-tag",
        "elif newest is not None and compare.group(1) != newest.version_text:",
        "elif False:",
    ),
    (
        "the compare reference may omit the v of the tag",
        "ref-no-v",
        'r"/compare/v(\\d+',
        'r"/compare/v?(\\d+',
    ),
    (
        "the guide link is not resolved against the disk",
        "link-missing-file",
        "            except (OSError, UnicodeError):\n                cache[page] = None",
        '            except (OSError, UnicodeError):\n                cache[page] = set(anchors_of("# x"))',
    ),
    (
        "the slug keeps its capitals",
        "valid-link-capitalised-heading",
        'value = re.sub(r"[^\\w\\s-]", "", value).strip().lower()',
        'value = re.sub(r"[^\\w\\s-]", "", value).strip()',
    ),
    (
        "the slug keeps the spaces",
        "slug-table",
        'return re.sub(r"[-\\s]+", "-", value)',
        "return value",
    ),
    (
        "the heading of the guide is not checked",
        "link-missing-heading",
        "elif slug not in anchors:",
        "elif False:",
    ),
    (
        "a percent-encoded slug is not decoded",
        "valid-link-percent-encoded",
        'slug = unquote(fragment or "")',
        'slug = fragment or ""',
    ),
    (
        "the version directory is not checked",
        "link-other-version",
        "if version != version_dir:",
        "if False:",
    ),
    (
        "the next version is not bumped",
        "valid-link",
        'return f"{major}.{minor + 1}.0"',
        'return f"{major}.{minor}.0"',
    ),
    (
        "continuation lines are ignored",
        "valid-link-continuation-line",
        "            body.append(line)\n            j += 1",
        "            j += 1",
    ),
    (
        "lines after a blank line are never part of the bullet",
        "valid-link-after-blank-line",
        "and _indent(masked[k]) >= marker_indent + 2",
        "and False",
    ),
    (
        "adjacent bullets are one bullet",
        "adjacent-bullets-first-linked-second-not",
        "if _ITEM.match(line) or _ANY_HEADING.match(line):",
        "if _ANY_HEADING.match(line):",
    ),
    (
        "inline code spans are not masked",
        "marker-in-code-span",
        'text = _CODE_SPAN.sub(_MASK, "\\n".join(body))',
        'text = "\\n".join(body)',
    ),
    (
        "code fences are not masked",
        "marker-in-fence",
        "    masked = mask_fences(lines)\n    ref_start",
        "    masked = lines\n    ref_start",
    ),
    (
        "the prose rule is gone",
        "prose-breaking-change",
        r'_PROSE_BREAKING = re.compile(r"\bbreaking[ \t\n]+(?:changes?|for)\b", re.I)',
        '_PROSE_BREAKING = re.compile(r"(?!x)x")',
    ),
    (
        "the whole file is the scope",
        "history-is-frozen",
        "    scope = sections[:1]\n",
        "    scope = sections\n",
    ),
    (
        "an empty [Unreleased] does not bring the newest release into scope",
        "empty-unreleased-checks-newest-release",
        "        and not _has_content(parsed, sections[0])\n",
        "        and False\n",
    ),
    (
        "--from keeps only the named section",
        "from-version-includes-unreleased",
        "return sections[: index + 1]",
        "return sections[index : index + 1]",
    ),
    (
        "sections not newest first are accepted",
        "versions-not-descending",
        "here.version >= above.version",
        "here.version == above.version",
    ),
    (
        "[Unreleased] may sit anywhere",
        "unreleased-last",
        '        if here.kind == "unreleased":\n            yield Finding(',
        "        if False:\n            yield Finding(",
    ),
    (
        "the date of a release heading is not validated",
        "bad-date",
        "                    date.fromisoformat(m.group(4))",
        "                    pass",
    ),
    (
        "CRLF is not normalised",
        "crlf",
        'return text.replace("\\r\\n", "\\n").replace("\\r", "\\n").split("\\n")',
        'return text.split("\\n")',
    ),
    (
        "the findings come in generation order",
        "findings-sorted",
        "return sorted(set(findings), key=lambda f: (f.line, f.code, f.message))",
        "return list(findings)",
    ),
    (
        "findings are not reported with the right exit code",
        "exit-codes",
        "return 1 if findings else 0",
        "return 0",
    ),
    (
        "an unreadable file exits 1",
        "exit-unreadable",
        'print(f"changelog_check: cannot read {path}: {exc}", file=sys.stderr)\n        return 2',
        'print(f"changelog_check: cannot read {path}: {exc}", file=sys.stderr)\n        return 1',
    ),
    (
        "extract includes the next heading",
        "extract-does-not-include-the-next-heading",
        "body = parsed.lines[section.start : section.end]",
        "body = parsed.lines[section.start : section.end + 1]",
    ),
    (
        "extract keeps the trailing blank lines",
        "extract-trailing-blank-lines-are-dropped",
        "        while body and not body[-1].strip():\n            body.pop()\n",
        "        while False:\n            body.pop()\n",
    ),
    (
        "extract keeps the leading blank lines",
        "extract-leading-blank-lines-are-dropped",
        "        while body and not body[0].strip():\n            body.pop(0)\n",
        "        while False:\n            body.pop(0)\n",
    ),
    (
        "extract of the last section runs into the reference block",
        "extract-last-stops-before-the-reference-block",
        "section.end = max(section.start, ref_start)",
        "section.end = len(parsed.lines)",
    ),
    (
        "extract does not accept a v prefix",
        "extract-v-prefix",
        'return value[1:] if value[:1] in "vV" else value',
        "return value",
    ),
    (
        "extract of a missing version exits 0",
        "exit-extract-missing",
        'changelog_check: no section [{args.version}] in {label}",\n'
        "                file=sys.stderr,\n            )\n            return 1",
        'changelog_check: no section [{args.version}] in {label}",\n'
        "                file=sys.stderr,\n            )\n            return 0",
    ),
    (
        "extract of an empty body exits 0",
        "exit-extract-empty",
        'changelog_check: section [{args.version}] has no body",\n'
        "                file=sys.stderr,\n            )\n            return 1",
        'changelog_check: section [{args.version}] has no body",\n'
        "                file=sys.stderr,\n            )\n            return 0",
    ),
    (
        "extract with the heading prints no heading",
        "extract-with-heading",
        'out = ([parsed.lines[section.line - 1], ""] if with_heading else []) + body',
        "out = body",
    ),
    (
        "a duplicate id is not made unique",
        "slug-table",
        "    while ident in taken or not ident:",
        "    while not ident:",
    ),
    (
        "the link base is not applied",
        "cli-link-base",
        "        if args.link_base:",
        "        if False:",
    ),
    (
        "a link in a code span is rewritten",
        "links-code-span",
        "        out.append(span.group(0))",
        "        out.append(_DOC_LINK.sub(replace, span.group(0)))",
    ),
    (
        "a link in a tilde code fence is rewritten",
        "links-tilde-fence",
        "        if is_fence:",
        "        if False:",
    ),
    (
        "the ./docs form is not rewritten",
        "links-dot-slash",
        '_DOC_LINK = re.compile(r"\\]\\((<?)(?:\\./)?docs/")',
        '_DOC_LINK = re.compile(r"\\]\\((<?)docs/")',
    ),
    (
        "the angle-bracket form loses its bracket",
        "links-angle",
        '    return f"]({match.group(1)}{base}docs/"'.replace(
            "    return", "        return"
        ),
        '        return f"]({base}docs/"',
    ),
    (
        "only the first link of a line is rewritten",
        "links-two-per-line",
        "    out.append(_DOC_LINK.sub(replace, chunk[position:]))",
        "    out.append(_DOC_LINK.sub(replace, chunk[position:], count=1))",
    ),
    (
        "a base with no trailing slash is joined as written",
        "links-no-slash-in-the-base",
        '    base = link_base if link_base.endswith("/") else link_base + "/"',
        "    base = link_base",
    ),
    (
        "any base is accepted by the command line",
        "cli-link-base-usage",
        '    if not re.fullmatch(r"https?://\\S+", value):',
        "    if False:",
    ),
    (
        "a code span over a line break is not protected",
        "links-code-span-over-a-line-break",
        "        for span in _CODE_SPAN.finditer(chunk):",
        '        for span in re.finditer(r"(`+)(.+?)\\1", chunk):',
    ),
]


def _mutated(old: str, new: str) -> str:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"the mutation anchor is not unique: {old[:70]!r}"
    return source.replace(old, new)


class TestScenarios:
    def test_the_real_script_passes_every_scenario(self, cc, tmp_path):
        assert scenario_problems(cc, tmp_path) == set()


class TestMutantsAreKilled:
    def test_there_are_at_least_twelve_mutants(self):
        assert len(MUTANTS) >= 12

    def test_every_named_scenario_exists(self):
        ids = (
            {c.id for c in CASES}
            | {"extract-" + e.id for e in EXTRACTS}
            | {"links-" + c.id for c in LINK_CASES}
            | set(EXTRA)
        )
        assert {m[1] for m in MUTANTS} <= ids

    @pytest.mark.parametrize(
        ("label", "scenario", "old", "new"), MUTANTS, ids=[m[0] for m in MUTANTS]
    )
    def test_mutant(self, tmp_path, label, scenario, old, new):
        # the mutated source still loads, and fails the scenario that guards the edit
        try:
            module = _load_module(_mutated(old, new), "_changelog_check_mutant")
        except SyntaxError:
            pytest.fail(f"the mutant does not load: {label}")
        try:
            assert scenario in scenario_problems(module, tmp_path), label
        finally:
            sys.modules.pop("_changelog_check_mutant", None)
