"""The code of the upgrade guides (`docs/upgrade-guide/`) is real code.

Plan d454fd62, steps 117 and 118 (roadmap row 0.5). Every `**BREAKING:**` bullet of the CHANGELOG
links a guide section that shows the old code and the new code (the rule itself is enforced by
`tests/test_packaging.py::TestTheRealChangelog`, which runs `scripts/changelog_check.py` on the
real file). A migration guide whose new code does not run is worse than none, so this file judges
the code blocks of every page under `docs/upgrade-guide/v*/` without a model and without a
network:

* a page opens with `# Title` and an intro that starts `Covers FSM-LLM <old> to <new>.` (the new
  version is the name of its folder, `v0.13.0`);
* every `##` section holds at least one fenced block;
* a fence has one of the bare info strings `python`, `bash`, `text`, `json` or none (the site's
  fence parser opens no other shape, see `test_docs_links.py`), and a `python` fence is introduced by one of the bold labels `**Before**`,
  `**After**` or `**Sketch**` on the line above it;
* every `python` block parses (`ast.parse`);
* `After` and `Sketch` blocks: every `from fsm_llm... import name` and `import fsm_llm...`
  resolves to a real module and a real name (a `Before` block may name a removed path: that is
  its point);
* `After` blocks also RUN, in a fresh namespace in an empty directory, with no model named and
  the model server address pointed at a dead port; a `Sketch` block needs a model or a file the
  page cannot create, so it is only parsed and import-checked.

`_guide_problems(root)` is a function of a project root so `TestDefectsAreCaught` runs it on a
scratch tree with one defect each; `TestTheRealGuides` judges the real pages.

Step 118 adds three more classes. `TestTheIndexPage` pins `docs/upgrade-guide/index.md` (it lists
every guide once and heads the nav group). `TestEveryGuideIsLinked` pins the other direction of
the CHANGELOG rule: no guide page and no section of one is an orphan. `TestModulePathsGuide` ties
`module-paths.md` to the path map: a section for every old module of 0.12.0 that the map lists,
an explicit reason for each old key that was never released, a section or a named mention for
every released module that stopped importing, and a codemod that is run on the Before blocks.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.test_fsm_llm.test_docs_site import _config, _nav_files

_ROOT = Path(__file__).resolve().parents[2]
_LABELS = {"**Before**": "before", "**After**": "after", "**Sketch**": "sketch"}
_INFOS = {"", "python", "bash", "text", "json"}
_OPENING = re.compile(r"^(```+|~~~+)\s*(.*?)\s*$")
_COVERS = re.compile(r"^Covers FSM-LLM (\d+\.\d+\.\d+) to (\d+\.\d+\.\d+)\.")


@dataclass(frozen=True)
class Fence:
    """One fenced block: its info string, body, 1-based opening line and the label above it."""

    info: str
    code: str
    line: int
    label: str


def _fences(text: str) -> list[Fence]:
    """Every fenced block of a page, with the bold label on the nearest non-blank line above."""
    lines = text.splitlines()
    out: list[Fence] = []
    i = 0
    while i < len(lines):
        m = _OPENING.match(lines[i])
        if not m:
            i += 1
            continue
        marker, info = m.group(1), m.group(2)
        j = i + 1
        while j < len(lines) and not (
            lines[j].strip().startswith(marker[0] * len(marker))
            and set(lines[j].strip()) == {marker[0]}
        ):
            j += 1
        above = next((x.strip() for x in reversed(lines[:i]) if x.strip()), "")
        out.append(
            Fence(
                info, "\n".join(lines[i + 1 : j]) + "\n", i + 1, _LABELS.get(above, "")
            )
        )
        i = j + 1
    return out


def _sections(text: str) -> Iterator[tuple[str, str]]:
    """`(heading, body)` of every `##` section outside fences."""
    heading, body, marker = "", [], ""
    for line in text.splitlines():
        fence = _OPENING.match(line)
        if marker:
            if fence and fence.group(1)[0] == marker[0]:
                marker = ""
            body.append(line)
            continue
        if fence:
            marker = fence.group(1)
            body.append(line)
            continue
        if line.startswith("## "):
            if heading:
                yield heading, "\n".join(body)
            heading, body = line[3:].strip(), []
        elif heading:
            body.append(line)
    if heading:
        yield heading, "\n".join(body)


def _unresolved_imports(code: str) -> list[str]:
    """Every `fsm_llm` import of a parsed block that names no real module or name."""
    bad: list[str] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "fsm_llm":
                    continue
                try:
                    importlib.import_module(alias.name)
                except ImportError:
                    bad.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module.split(".")[0] != "fsm_llm":
                continue
            try:
                imported = importlib.import_module(module)
            except ImportError:
                bad.append(f"from {module} import ...")
                continue
            for alias in node.names:
                if alias.name == "*" or hasattr(imported, alias.name):
                    continue
                try:
                    importlib.import_module(f"{module}.{alias.name}")
                except ImportError:
                    bad.append(f"from {module} import {alias.name}")
    return bad


def _run(code: str, where: str, workdir: Path) -> str | None:
    """Run one `After` block; the message of its failure, or `None`."""
    saved = {k: os.environ.get(k) for k in ("LLM_MODEL", "OLLAMA_API_BASE")}
    cwd = Path.cwd()
    os.environ.pop("LLM_MODEL", None)
    os.environ["OLLAMA_API_BASE"] = "http://127.0.0.1:9"
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    try:
        exec(compile(code, where, "exec"), {"__name__": "__guide__"})
    except BaseException as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(cwd)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return None


def _page_problems(page: Path, root: Path, workdir: Path) -> list[str]:
    rel = page.relative_to(root).as_posix()
    text = page.read_text(encoding="utf-8")
    problems: list[str] = []
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        problems.append(f"{rel}: the first line is not a `# Title`")
    intro = next((x for x in lines[1:] if x.strip()), "")
    covers = _COVERS.match(intro)
    folder = page.parent.name.removeprefix("v")
    if not covers:
        problems.append(f"{rel}: the intro does not start `Covers FSM-LLM X to Y.`")
    elif covers.group(2) != folder:
        problems.append(
            f"{rel}: the intro covers {covers.group(2)}, the folder is v{folder}"
        )
    for heading, body in _sections(text):
        if not _fences(body):
            problems.append(f"{rel}: section {heading!r} holds no fenced block")
    for number, fence in enumerate(_fences(text), 1):
        where = f"{rel}:{fence.line}"
        if fence.info not in _INFOS:
            problems.append(
                f"{where}: the info string {fence.info!r} is not one of {sorted(_INFOS)}"
            )
            continue
        if fence.info != "python":
            continue
        if not fence.label:
            problems.append(
                f"{where}: a python block needs **Before**, **After** or **Sketch** above it"
            )
            continue
        try:
            ast.parse(fence.code)
        except SyntaxError as exc:
            problems.append(f"{where}: does not parse ({exc.msg}, line {exc.lineno})")
            continue
        if fence.label == "before":
            continue
        for name in _unresolved_imports(fence.code):
            problems.append(f"{where}: {name} does not resolve")
        if fence.label == "after" and not _unresolved_imports(fence.code):
            failure = _run(fence.code, where, workdir / f"block{number}")
            if failure:
                problems.append(f"{where}: the After block fails: {failure}")
    return problems


def _pages(root: Path) -> list[Path]:
    """The guide pages: `docs/upgrade-guide/v<version>/<theme>.md` (not the overview `index.md`)."""
    return sorted(
        page
        for page in (root / "docs" / "upgrade-guide").rglob("*.md")
        if page.parent.name.startswith("v")
    )


def _guide_problems(root: Path, workdir: Path) -> list[str]:
    return [p for page in _pages(root) for p in _page_problems(page, root, workdir)]


_GOOD = """# Guide

Covers FSM-LLM 0.12.0 to 0.13.0. Text.

## A rename

**Before**

```python
from fsm_llm.gone import thing
```

**After**

```python
from fsm_llm import API

assert API is not None
```
"""


class TestTheRealGuides:
    def test_there_are_guides(self):
        pages = _pages(_ROOT)
        assert pages, "docs/upgrade-guide/ holds no page"
        assert all(p.parent.name.startswith("v") for p in pages)

    @pytest.mark.parametrize(
        "page", _pages(_ROOT), ids=lambda p: p.relative_to(_ROOT / "docs").as_posix()
    )
    def test_the_page_is_sound(self, page, tmp_path):
        assert _page_problems(page, _ROOT, tmp_path) == []

    def test_the_blocks_are_really_read(self):
        """Guard against a scanner that finds nothing: the guides carry labelled code."""
        labels = [
            f.label
            for page in _pages(_ROOT)
            for f in _fences(page.read_text(encoding="utf-8"))
            if f.info == "python"
        ]
        assert labels.count("before") >= 10
        assert labels.count("after") >= 10


def _checker():
    """`scripts/changelog_check.py` loaded by path (the slug rule and the check)."""
    script = _ROOT / "scripts" / "changelog_check.py"
    spec = importlib.util.spec_from_file_location("_changelog_check_guides", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses read the module of the class
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


_CHANGELOG_TEXT = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
_GUIDE_LINK = re.compile(r"docs/upgrade-guide/(v[\d.]+/[\w.-]+\.md)#([^)\s>]+)")


class TestEveryGuideIsLinked:
    """No guide page and no guide section is an orphan of the CHANGELOG (step 118).

    `tests/test_packaging.py::TestTheRealChangelog` holds the other direction (every BREAKING
    bullet links an existing section). A section nobody links is a guide for a change that the
    changelog no longer admits, or a bullet that lost its link.
    """

    def test_every_page_is_linked_from_the_changelog(self):
        for page in _pages(_ROOT):
            assert f"docs/upgrade-guide/v0.13.0/{page.name}#" in _CHANGELOG_TEXT, (
                page.name
            )

    def test_every_section_of_a_theme_page_is_linked_by_a_bullet(self):
        slugify = _checker().slugify
        linked = {(page, slug) for page, slug in _GUIDE_LINK.findall(_CHANGELOG_TEXT)}
        orphans = []
        for page in _pages(_ROOT):
            if page.name == "module-paths.md":
                continue  # a section for every old module, linked only where a bullet exists
            rel = f"v0.13.0/{page.name}"
            for heading, _ in _sections(page.read_text(encoding="utf-8")):
                if (rel, slugify(heading)) not in linked:
                    orphans.append(f"{rel}: {heading}")
        assert orphans == []

    def test_the_module_paths_page_is_linked_by_the_module_bullets(self):
        slugs = [
            s
            for page, s in _GUIDE_LINK.findall(_CHANGELOG_TEXT)
            if page.endswith("module-paths.md")
        ]
        assert len(set(slugs)) >= 16, sorted(set(slugs))


_INDEX = _ROOT / "docs" / "upgrade-guide" / "index.md"


class TestTheIndexPage:
    """`docs/upgrade-guide/index.md`: the overview that lists every guide (step 118)."""

    def test_it_exists_and_has_a_title(self):
        assert _INDEX.is_file()
        assert _INDEX.read_text(encoding="utf-8").startswith("# Upgrade guides\n")

    def test_it_links_every_guide_page_exactly_once(self):
        text = _INDEX.read_text(encoding="utf-8")
        listed = re.findall(r"^\d+\. \[[^\]]+\]\((v[\d.]+/[\w.-]+\.md)\)", text, re.M)
        pages = [f"v0.13.0/{p.name}" for p in _pages(_ROOT)]
        assert sorted(listed) == sorted(pages)
        assert len(listed) == len(set(listed))

    def test_every_link_of_the_index_resolves(self):
        text = _INDEX.read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)#\s]+\.md)", text)
        assert targets
        for target in targets:
            assert (_INDEX.parent / target).is_file(), target

    def test_the_nav_group_lists_the_index_first_then_every_page(self):
        nav = _config(_ROOT)["nav"]
        group = next(
            next(iter(item.values()))
            for item in nav
            if isinstance(item, dict) and "Upgrade guides" in item
        )
        files = _nav_files(group)
        pages = [f"upgrade-guide/v0.13.0/{p.name}" for p in _pages(_ROOT)]
        assert files[0] == "upgrade-guide/index.md"
        assert sorted(files[1:]) == sorted(pages)

    def test_the_landing_page_points_at_it(self):
        text = (_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
        assert "](upgrade-guide/index.md)" in text


# --------------------------------------------------------------------------------------
# module-paths.md and the path map
# --------------------------------------------------------------------------------------

_MODULE_PATHS = _ROOT / "docs" / "upgrade-guide" / "v0.13.0" / "module-paths.md"
_PATH_MAP = _ROOT / "tests" / "fixtures" / "module_path_map.json"
_RELEASED = _ROOT / "tests" / "fixtures" / "released_modules_0_12_0.json"
_CODEMOD_TITLE = "Rewrite the imports with a script"
_OTHER_TITLE = "Removed and renamed modules that are not in the map"
_UNRELEASED_TITLE = "Modules added after 0.12.0"

#: The old keys of the path map that were NEVER released: each is absent from the v0.12.0 tree
#: (`released_modules_0_12_0.json`), so no released import of it stopped working and the guide
#: lists it in "Modules added after 0.12.0" instead of giving it a section. The reason is the
#: evidence; a key leaves this table only by gaining a released path (it then needs a section).
_UNRELEASED_KEYS = {
    "fsm_llm.agent_skills": "Agent Skills were added after 0.12.0 (agent_skills.py is not in the v0.12.0 tree)",
    "fsm_llm.agent_skills_cli": "the Agent Skills CLI was added after 0.12.0",
    "fsm_llm.codes": "the status and reason codes were added after 0.12.0",
    "fsm_llm.executors": "the state-kind executors were added after 0.12.0 (roadmap row 0.14)",
    "fsm_llm.flow_analysis": "the data-flow and liveness analyses were added after 0.12.0",
    "fsm_llm.okf": "the Open Knowledge Format loader was added after 0.12.0",
    "fsm_llm.okf_bridge": "the OKF bridge was added after 0.12.0",
    "fsm_llm.okf_cli": "the OKF command line was added after 0.12.0",
    "fsm_llm.schema_registry": "the schema registry was added after 0.12.0",
    "fsm_llm.state_kinds": "the state-kind registry was added after 0.12.0 (roadmap row 0.14)",
}


def _map() -> dict:
    return json.loads(_PATH_MAP.read_text(encoding="utf-8"))["moved"]


def _released() -> set[str]:
    return set(json.loads(_RELEASED.read_text(encoding="utf-8"))["modules"])


def _guide_text() -> str:
    return "\n".join(page.read_text(encoding="utf-8") for page in _pages(_ROOT))


def _module_path_sections() -> dict[str, str]:
    return dict(_sections(_MODULE_PATHS.read_text(encoding="utf-8")))


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _codemod_script() -> str:
    section = _module_path_sections()[_CODEMOD_TITLE]
    (fence,) = [f for f in _fences(section) if f.info == "python"]
    assert fence.label == "sketch"
    return fence.code


def _renames() -> dict[str, str]:
    tree = ast.parse(_codemod_script())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "RENAMES":  # type: ignore[attr-defined]
            return ast.literal_eval(node.value)
    raise AssertionError("the codemod has no RENAMES table")


def _run_codemod(tmp_path: Path, files: dict[str, str]) -> dict[str, str]:
    """Write the codemod and `files` (name to text) under `tmp_path`, run it, read them back."""
    tree = tmp_path / "tree"
    tree.mkdir(parents=True)
    for name, text in files.items():
        (tree / name).write_text(text, encoding="utf-8")
    script = tmp_path / "retarget.py"
    script.write_text(_codemod_script(), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(script), str(tree)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return {name: (tree / name).read_text(encoding="utf-8") for name in files}


class TestModulePathsGuide:
    def test_the_released_fixture_is_the_v0_12_0_tree(self):
        data = json.loads(_RELEASED.read_text(encoding="utf-8"))
        modules = data["modules"]
        assert data["tag"] == "v0.12.0"
        assert modules == sorted(set(modules)) and len(modules) > 100
        assert {
            "fsm_llm",
            "fsm_llm.api",
            "fsm_llm.harness",
            "fsm_llm.agents.hitl",
        } <= set(modules)
        assert not {"fsm_llm.okf", "fsm_llm.core", "fsm_llm.engine"} & set(modules)

    def test_the_released_fixture_equals_the_tag_when_the_tag_exists(self):
        git = shutil.which("git")
        if git is None:
            pytest.skip("no git")
        proc = subprocess.run(
            [git, "ls-tree", "-r", "--name-only", "v0.12.0", "--", "src"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            pytest.skip("the v0.12.0 tag is not in this clone")
        modules = set()
        for name in proc.stdout.split():
            if name.startswith("src/fsm_llm/") and name.endswith(".py"):
                dotted = name[len("src/") : -len(".py")].replace("/", ".")
                modules.add(dotted.removesuffix(".__init__"))
        assert modules == _released()

    def test_every_old_key_is_released_or_listed_as_unreleased_with_a_reason(self):
        moved, released = set(_map()), _released()
        assert set(_UNRELEASED_KEYS) <= moved
        assert not set(_UNRELEASED_KEYS) & released, (
            "a released key is listed as unreleased"
        )
        assert moved - released == set(_UNRELEASED_KEYS), sorted(
            (moved - released) ^ set(_UNRELEASED_KEYS)
        )
        assert all(reason.strip() for reason in _UNRELEASED_KEYS.values())

    def test_every_released_old_key_has_a_section(self):
        headings = set(_module_path_sections())
        missing = sorted((set(_map()) & _released()) - headings)
        assert missing == []

    def test_no_section_names_a_module_that_is_not_an_old_key(self):
        headings = set(_module_path_sections())
        extras = {_CODEMOD_TITLE, _OTHER_TITLE, _UNRELEASED_TITLE}
        assert headings - extras == set(_map()) & _released()
        assert extras <= headings

    def test_the_unreleased_keys_are_accounted_for_in_their_own_section(self):
        section = _module_path_sections()[_UNRELEASED_TITLE]
        (fence,) = [f for f in _fences(section) if f.info == "text"]
        listed = {line.split()[0] for line in fence.code.splitlines() if line.strip()}
        assert listed == set(_UNRELEASED_KEYS)

    def test_every_released_module_that_stopped_importing_is_named_in_a_guide(self):
        """The module, or a package above it that is gone too, appears by name in a guide.

        A package that still imports (`fsm_llm.agents`) is not evidence: it is named everywhere.
        """
        text = _guide_text()
        missing = []
        for module in sorted(_released()):
            if _importable(module):
                continue
            parts = module.split(".")
            names = [module] + [
                ".".join(parts[:i])
                for i in range(2, len(parts))
                if not _importable(".".join(parts[:i]))
            ]
            if not any(
                re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", text)
                for name in names
            ):
                missing.append(module)
        assert missing == []

    def test_a_module_the_map_does_not_list_is_named_where_it_is_covered(self):
        sections = _module_path_sections()
        other = sections[_OTHER_TITLE]
        assert "fsm_llm.harness" in other and "fsm_llm.agents.skills" in other

    def test_the_codemod_table_matches_the_map(self):
        renames, moved = _renames(), _map()
        assert renames, "the codemod has an empty table"
        for old, new in renames.items():
            assert old in moved and old in _released(), old
            assert _importable(new), f"{new} does not import"
            assert not _importable(old), f"{old} still imports: no rename needed"
            if isinstance(moved[old], str):
                assert new == moved[old], (old, new)

    def test_every_gone_whole_module_is_in_the_codemod(self):
        renames = _renames()
        gone_whole = {
            old
            for old, new in _map().items()
            if isinstance(new, str) and old in _released() and not _importable(old)
        }
        assert gone_whole <= set(renames), sorted(gone_whole - set(renames))

    def test_the_codemod_turns_each_before_block_into_its_after_block(self, tmp_path):
        sections = _module_path_sections()
        files, expected = {}, {}
        for old in _renames():
            blocks = [f for f in _fences(sections[old]) if f.info == "python"]
            before = next(f for f in blocks if f.label == "before")
            after = next(f for f in blocks if f.label == "after")
            files[old + ".py"] = before.code
            expected[old + ".py"] = after.code
        assert len(files) >= 10
        assert _run_codemod(tmp_path, files) == expected

    def test_the_codemod_resolves_every_name_it_writes(self, tmp_path):
        sections = _module_path_sections()
        for old in _renames():
            before = next(
                f
                for f in _fences(sections[old])
                if f.info == "python" and f.label == "before"
            )
            out = _run_codemod(tmp_path / old, {"m.py": before.code})["m.py"]
            assert _unresolved_imports(out) == [], (old, out)

    def test_the_codemod_touches_only_the_dotted_names_of_its_table(self, tmp_path):
        text = (
            "import fsm_llm.api as a\n"
            "from fsm_llm.api_reference import x\n"
            "from myfsm_llm.api import y\n"
            "z = other.fsm_llm.api\n"
            "from fsm_llm.agents.handlers import h\n"
            "from fsm_llm import API\n"
            "# fsm_llm.session in a comment is a name too\n"
        )
        out = _run_codemod(tmp_path, {"m.py": text})["m.py"]
        assert out == (
            "import fsm_llm.engine.api as a\n"
            "from fsm_llm.api_reference import x\n"
            "from myfsm_llm.api import y\n"
            "z = other.fsm_llm.api\n"
            "from fsm_llm.agents.handlers import h\n"
            "from fsm_llm import API\n"
            "# fsm_llm.engine.session in a comment is a name too\n"
        )

    def test_the_codemod_is_idempotent_and_reads_only_python_files(self, tmp_path):
        files = {
            "m.py": "from fsm_llm.fsm import FSMManager\n",
            "notes.txt": "fsm_llm.fsm\n",
        }
        once = _run_codemod(tmp_path / "a", files)
        assert once == {
            "m.py": "from fsm_llm.engine.fsm import FSMManager\n",
            "notes.txt": "fsm_llm.fsm\n",
        }
        assert _run_codemod(tmp_path / "b", once) == once

    def test_a_path_kept_section_names_the_defining_module_of_its_private_name(self):
        moved = _map()
        sections = _module_path_sections()
        for old in (
            "fsm_llm.security",
            "fsm_llm.agents.base",
            "fsm_llm.agents.fsm_definitions",
        ):
            after = next(
                f
                for f in _fences(sections[old])
                if f.info == "python" and f.label == "after"
            )
            names = {
                alias.name: node.module
                for node in ast.walk(ast.parse(after.code))
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            }
            assert any(moved[old].get(name) == mod for name, mod in names.items()), old


class TestScanner:
    def test_fences_carry_the_label_above(self):
        fences = _fences(_GOOD)
        assert [(f.info, f.label, f.line) for f in fences] == [
            ("python", "before", 9),
            ("python", "after", 15),
        ]

    def test_a_longer_marker_holds_a_shorter_one(self):
        text = "x\n\n**After**\n\n````python\n```\ninner\n```\n````\n"
        (fence,) = _fences(text)
        assert fence.code == "```\ninner\n```\n"
        assert fence.label == "after"

    def test_sections_are_split_outside_fences(self):
        text = "# T\n\n## One\n\n```text\n## not a heading\n```\n\n## Two\n\nbody\n"
        assert [h for h, _ in _sections(text)] == ["One", "Two"]

    def test_a_missing_import_is_named(self):
        assert _unresolved_imports("from fsm_llm import API, NoSuchName\n") == [
            "from fsm_llm import NoSuchName"
        ]
        assert _unresolved_imports("from fsm_llm.nowhere import x\n") == [
            "from fsm_llm.nowhere import ..."
        ]
        assert _unresolved_imports("import fsm_llm.nowhere\n") == [
            "import fsm_llm.nowhere"
        ]
        assert _unresolved_imports("import os\nfrom pathlib import Path\n") == []

    def test_a_submodule_import_resolves(self):
        assert _unresolved_imports("from fsm_llm.core import loaders\n") == []


def _scratch(tmp_path: Path, page: str, name: str = "core.md") -> Path:
    folder = tmp_path / "docs" / "upgrade-guide" / "v0.13.0"
    folder.mkdir(parents=True)
    (folder / name).write_text(page, encoding="utf-8")
    return tmp_path


class TestDefectsAreCaught:
    def test_the_scratch_page_is_clean(self, tmp_path):
        root = _scratch(tmp_path / "r", _GOOD)
        assert _guide_problems(root, tmp_path / "w") == []

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            pytest.param(
                "assert API is not None",
                "assert API is not None(",
                "does not parse",
                id="unparsable-python-fence",
            ),
            pytest.param(
                "from fsm_llm import API",
                "from fsm_llm.gone import API",
                "does not resolve",
                id="after-imports-a-removed-path",
            ),
            pytest.param(
                "from fsm_llm import API",
                "from fsm_llm import NoSuchName as API",
                "from fsm_llm import NoSuchName does not resolve",
                id="after-imports-a-removed-name",
            ),
            pytest.param(
                "**Before**\n\n```python\nfrom fsm_llm.gone import thing",
                "**After**\n\n```python\nfrom fsm_llm.gone import thing",
                "from fsm_llm.gone import ... does not resolve",
                id="before-mislabelled-as-after",
            ),
            pytest.param(
                "assert API is not None",
                "assert API is None",
                "the After block fails: AssertionError",
                id="after-block-fails-at-runtime",
            ),
            pytest.param(
                "**After**\n\n```python",
                "```python",
                "needs **Before**, **After** or **Sketch**",
                id="python-block-without-label",
            ),
            pytest.param(
                "```python\nfrom fsm_llm import API",
                "```python title=x\nfrom fsm_llm import API",
                "the info string 'python title=x'",
                id="python-fence-with-an-info-string",
            ),
            pytest.param(
                "Covers FSM-LLM 0.12.0 to 0.13.0.",
                "Some intro.",
                "the intro does not start",
                id="missing-covers-sentence",
            ),
            pytest.param(
                "Covers FSM-LLM 0.12.0 to 0.13.0.",
                "Covers FSM-LLM 0.12.0 to 0.14.0.",
                "the folder is v0.13.0",
                id="covers-the-wrong-version",
            ),
            pytest.param("# Guide", "Guide", "is not a `# Title`", id="no-title"),
            pytest.param(
                "## A rename\n",
                "## A rename\n\n## Empty one\n\nnothing\n",
                "holds no fenced block",
                id="section-without-a-block",
            ),
        ],
    )
    def test_each_defect_is_named(self, tmp_path, old, new, expected):
        assert old in _GOOD
        page = _GOOD.replace(old, new, 1)
        root = _scratch(tmp_path / "r", page)
        problems = _guide_problems(root, tmp_path / "w")
        assert any(expected in p for p in problems), problems

    def test_a_sketch_block_is_parsed_and_import_checked_but_not_run(self, tmp_path):
        sketch = _GOOD.replace("**After**", "**Sketch**").replace(
            "assert API is not None", "raise SystemExit(1)"
        )
        root = _scratch(tmp_path / "r", sketch)
        assert _guide_problems(root, tmp_path / "w") == []

    def test_a_sketch_block_still_needs_real_imports(self, tmp_path):
        sketch = _GOOD.replace("**After**", "**Sketch**").replace(
            "from fsm_llm import API", "from fsm_llm import Gone"
        )
        root = _scratch(tmp_path / "r", sketch)
        assert any(
            "does not resolve" in p for p in _guide_problems(root, tmp_path / "w")
        )

    def test_a_before_block_may_name_a_removed_path(self, tmp_path):
        root = _scratch(tmp_path / "r", _GOOD)
        assert _guide_problems(root, tmp_path / "w") == []

    def test_a_run_restores_the_environment_and_the_directory(self, tmp_path):
        cwd = Path.cwd()
        before = (os.environ.get("LLM_MODEL"), os.environ.get("OLLAMA_API_BASE"))
        failing = _GOOD.replace("assert API is not None", "raise RuntimeError('x')")
        root = _scratch(tmp_path / "r", failing)
        assert any("RuntimeError" in p for p in _guide_problems(root, tmp_path / "w"))
        assert Path.cwd() == cwd
        assert (
            os.environ.get("LLM_MODEL"),
            os.environ.get("OLLAMA_API_BASE"),
        ) == before

    def test_an_After_block_runs_with_no_model_named(self, tmp_path):
        probe = _GOOD.replace(
            "assert API is not None",
            "import os\nassert 'LLM_MODEL' not in os.environ\nassert os.environ['OLLAMA_API_BASE'].endswith(':9')",
        )
        root = _scratch(tmp_path / "r", probe)
        assert _guide_problems(root, tmp_path / "w") == []
