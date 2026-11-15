"""The code of the upgrade guides (`docs/upgrade-guide/`) is real code.

Plan d454fd62, steps 117 and 118 (roadmap row 0.5). Every `**BREAKING:**` bullet of the CHANGELOG
links a guide section that shows the old code and the new code. A migration guide whose new code
does not run is worse than none, so this file judges the code blocks of every page under
`docs/upgrade-guide/` without a model and without a network:

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
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

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
    return sorted((root / "docs" / "upgrade-guide").rglob("*.md"))


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


def _changelog_findings():
    """The findings of `scripts/changelog_check.py` over the real CHANGELOG.md."""
    script = _ROOT / "scripts" / "changelog_check.py"
    spec = importlib.util.spec_from_file_location("_changelog_check_117", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses read the module of the class
    try:
        spec.loader.exec_module(module)
        text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        return module.check_text(text, docs_root=_ROOT / "docs")
    finally:
        sys.modules.pop(spec.name, None)


class TestTheRealChangelog:
    """The links the CHANGELOG already carries are sound (step 117).

    Until step 118 links every `**BREAKING:**` bullet, the only finding allowed on the real file
    is CL006 (a bullet with no guide link). Any other code is a defect of the file: the compare
    reference (CL004), a spelling variant of the marker (CL005), a link to a guide or a heading
    that does not exist (CL007, CL008), a link into another version's folder (CL011), a heading
    out of order or unknown (CL001 to CL003, CL009, CL010). Step 118 turns on the full check in
    `tests/test_packaging.py`, which makes this class redundant.
    """

    def test_the_only_open_finding_is_an_unlinked_bullet(self):
        findings = _changelog_findings()
        assert [f.format("CHANGELOG.md") for f in findings if f.code != "CL006"] == []

    def test_the_batch_one_guides_are_linked(self):
        """Not vacuous: the file does carry links into every page of the guides."""
        text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        for page in _pages(_ROOT):
            assert f"docs/upgrade-guide/v0.13.0/{page.name}#" in text, page.name


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
