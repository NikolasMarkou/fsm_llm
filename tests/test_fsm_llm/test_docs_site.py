"""The docs site configuration is consistent, without importing mkdocs.

Plan d454fd62, step 102 (roadmap row 0.7). `mkdocs.yml` is the one file that decides what the
published site holds, and a mistake in it is silent until a strict build runs in CI: a page
left out of the navigation, a nav entry for a file that is gone, an `AGENTS.md` (the notes for
coding agents, never published) pulled in, an `include-markdown` directive aimed at a path that
does not exist. These tests read the configuration and the `docs/` tree and fail on each of
those, so the strict build is a second line of defence and not the first.

* `mkdocs` is NOT imported (it is a docs-only tool: `[dependency-groups] docs`, never installed
  by the dev extra). The config is read with a `yaml.SafeLoader` subclass that tolerates the
  `!!python/name:` tags `pymdownx.superfences` needs (a plain `safe_load` raises on them).
* Every check is a function of a project root, so the mutation tests below run it on a small
  scratch tree with one defect each (`TestDefectsAreCaught`): a check that cannot fail is not a
  check. The real tree is judged by `TestTheRealSite`.
* The set of pages is computed from the files on disk, so a page added by a later step is judged
  without editing this file: it must be in the nav or match `exclude_docs`.
"""

from __future__ import annotations

import glob
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# Reading mkdocs.yml
# --------------------------------------------------------------------------------------


class _TagTolerantLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` that turns every unknown tag into plain data.

    `!!python/name:pymdownx.superfences.fence_code_format` (and `!ENV`, `!relative`) has no
    constructor in the safe loader; here a tagged scalar becomes the string `<tag> <value>` and a
    tagged collection becomes the plain collection, so the config can be read without running or
    importing anything the tag names.
    """


def _any_tag(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return f"!{tag_suffix} {loader.construct_scalar(node)}".strip()
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_mapping(node, deep=True)  # type: ignore[arg-type]


_TagTolerantLoader.add_multi_constructor("", _any_tag)


def _load_config(text: str) -> dict[str, Any]:
    data = yaml.load(text, Loader=_TagTolerantLoader)
    return data if isinstance(data, dict) else {}


def _config(root: Path) -> dict[str, Any]:
    return _load_config((root / "mkdocs.yml").read_text(encoding="utf-8"))


def _docs_dir(root: Path, config: dict[str, Any]) -> Path:
    return (root / str(config.get("docs_dir", "docs"))).resolve()


# --------------------------------------------------------------------------------------
# Navigation and exclusion
# --------------------------------------------------------------------------------------


def _nav_files(nav: Any) -> list[str]:
    """Every file name a `nav` value names (nested lists and `Title: file` mappings)."""
    if isinstance(nav, str):
        return [] if re.match(r"^[a-z][a-z0-9+.-]*://", nav) else [nav]
    if isinstance(nav, list):
        return [name for item in nav for name in _nav_files(item)]
    if isinstance(nav, dict):
        return [name for value in nav.values() for name in _nav_files(value)]
    return []


def _glob_regex(pattern: str) -> str:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return out


def _exclude_patterns(config: dict[str, Any]) -> list[str]:
    raw = config.get("exclude_docs") or ""
    return [
        line.strip()
        for line in str(raw).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _is_excluded(rel: str, patterns: list[str]) -> bool:
    """`exclude_docs` is gitignore syntax; this reads the subset the config uses.

    A name without a slash matches at any depth, a leading or inner slash anchors it to the docs
    root, a trailing slash matches directories only (and so everything under them), `*`, `?` and
    `**` are the usual wildcards and `!` re-includes. The last matching line wins. mkdocs reads
    the same patterns with `pathspec`; this is the part of it a docs tree needs.
    """
    parts = rel.split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    excluded = False
    for raw in patterns:
        negate = raw.startswith("!")
        pat = raw[1:] if negate else raw
        dir_only = pat.endswith("/")
        pat = pat.rstrip("/")
        anchored = pat.startswith("/") or "/" in pat
        rx = re.compile(_glob_regex(pat.lstrip("/")))
        hit = False
        for i, prefix in enumerate(prefixes):
            if dir_only and i == len(parts) - 1:
                continue
            if rx.fullmatch(prefix if anchored else parts[i]):
                hit = True
                break
        if hit:
            excluded = not negate
    return excluded


# --------------------------------------------------------------------------------------
# include-markdown
# --------------------------------------------------------------------------------------

#: `{% include-markdown "path" ... %}` and `{% include "path" %}` of mkdocs-include-markdown-plugin.
_INCLUDE = re.compile(r"\{%\s*include(?:-markdown)?\s+(['\"])(.+?)\1", re.DOTALL)


def _include_targets(page: Path) -> list[str]:
    return [m.group(2) for m in _INCLUDE.finditer(page.read_text(encoding="utf-8"))]


# --------------------------------------------------------------------------------------
# The checks: a function of a project root, returning one string per problem
# --------------------------------------------------------------------------------------

_VALIDATION = {
    ("nav", "omitted_files"),
    ("nav", "not_found"),
    ("nav", "absolute_links"),
    ("links", "not_found"),
    ("links", "absolute_links"),
    ("links", "unrecognized_links"),
    ("links", "anchors"),
}


def _site_problems(root: Path) -> list[str]:
    if not (root / "mkdocs.yml").is_file():
        return ["mkdocs.yml is missing"]
    config = _config(root)
    if not config:
        return ["mkdocs.yml is not a mapping"]
    problems: list[str] = []

    if config.get("strict") is not True:
        problems.append("strict is not true: warnings would not fail the build")

    validation = config.get("validation")
    if not isinstance(validation, dict):
        problems.append("the validation block is missing")
        validation = {}
    for section, key in sorted(_VALIDATION):
        if (validation.get(section) or {}).get(key) != "warn":
            problems.append(
                f"validation.{section}.{key} is not 'warn': strict would not fail on it"
            )

    docs = _docs_dir(root, config)
    if not docs.is_dir():
        problems.append(
            f"docs_dir {config.get('docs_dir', 'docs')!r} is not a directory"
        )
        return problems

    patterns = _exclude_patterns(config)
    if not all(_is_excluded(f"{prefix}AGENTS.md", patterns) for prefix in ("", "x/")):
        problems.append("exclude_docs does not exclude AGENTS.md at every depth")

    pages = sorted(p.relative_to(docs).as_posix() for p in docs.rglob("*.md"))
    nav = _nav_files(config.get("nav"))
    nav_set = set(nav)
    for name in nav:
        if not (docs / name).is_file():
            problems.append(f"nav names {name!r}, which is not a file under docs_dir")
        if Path(name).name == "AGENTS.md":
            problems.append(f"nav names {name!r}: AGENTS.md is never published")
        elif _is_excluded(name, patterns):
            problems.append(f"nav names {name!r}, which exclude_docs removes")
    for rel in pages:
        if rel not in nav_set and not _is_excluded(rel, patterns):
            problems.append(f"{rel} is neither in the nav nor in exclude_docs")

    for rel in pages:
        for target in _include_targets(docs / rel):
            if re.match(r"^[a-z][a-z0-9+.-]*://", target):
                continue
            if target.startswith("/"):
                problems.append(
                    f"{rel} includes {target!r}: use a path relative to the page"
                )
                continue
            found = glob.glob(str((docs / rel).parent / target))
            if not found:
                problems.append(f"{rel} includes {target!r}, which does not exist")
            for hit in found:
                if Path(hit).name == "AGENTS.md":
                    problems.append(
                        f"{rel} includes {target!r}: AGENTS.md is never published"
                    )
    return problems


# --------------------------------------------------------------------------------------
# The real tree
# --------------------------------------------------------------------------------------


class TestTheRealSite:
    """What `mkdocs.yml` and `docs/` of this repository must hold."""

    def test_the_configuration_has_no_structural_problem(self):
        assert _site_problems(_ROOT) == []

    def test_the_configuration_parses_to_a_mapping(self):
        config = _config(_ROOT)
        assert config["site_name"]
        assert config["site_url"] == "https://nikolasmarkou.github.io/fsm_llm/"
        assert config["repo_url"] == "https://github.com/NikolasMarkou/fsm_llm"

    def test_docs_dir_is_the_docs_folder(self):
        assert _docs_dir(_ROOT, _config(_ROOT)) == (_ROOT / "docs").resolve()

    def test_the_landing_page_exists_and_is_in_the_nav(self):
        config = _config(_ROOT)
        assert (_ROOT / "docs" / "index.md").is_file()
        assert "index.md" in _nav_files(config["nav"])

    def test_the_decision_index_is_a_page_and_api_txt_is_not(self):
        config = _config(_ROOT)
        assert "decisions/index.md" in _nav_files(config["nav"])
        assert _is_excluded("api.txt", _exclude_patterns(config))

    def test_no_agents_file_is_reachable(self):
        config = _config(_ROOT)
        docs = _docs_dir(_ROOT, config)
        patterns = _exclude_patterns(config)
        assert [
            n for n in _nav_files(config["nav"]) if Path(n).name == "AGENTS.md"
        ] == []
        loose = [
            p.relative_to(docs).as_posix()
            for p in docs.rglob("AGENTS.md")
            if not _is_excluded(p.relative_to(docs).as_posix(), patterns)
        ]
        assert loose == []

    def test_every_include_target_of_the_real_pages_exists(self):
        docs = _docs_dir(_ROOT, _config(_ROOT))
        missing = []
        for page in sorted(docs.rglob("*.md")):
            for target in _include_targets(page):
                if not glob.glob(str(page.parent / target)):
                    missing.append((page.relative_to(_ROOT).as_posix(), target))
        assert missing == []

    def test_the_theme_is_material_and_loads_no_web_font(self):
        theme = _config(_ROOT)["theme"]
        assert theme["name"] == "material"
        assert theme["font"] is False, "a font request leaves the reader's machine"

    def test_the_plugins(self):
        plugins = _config(_ROOT)["plugins"]
        names = [next(iter(p)) if isinstance(p, dict) else p for p in plugins]
        assert {"search", "mkdocstrings", "include-markdown"} <= set(names)
        python = next(
            p for p in plugins if isinstance(p, dict) and "mkdocstrings" in p
        )["mkdocstrings"]["handlers"]["python"]
        assert python["paths"] == ["src"]
        # `google` would give 10 strict warnings in files bench manifests pin; `null` gives none.
        assert "docstring_style" in python["options"]
        assert python["options"]["docstring_style"] is None

    def test_the_version_selector_is_mike(self):
        assert _config(_ROOT)["extra"]["version"]["provider"] == "mike"

    def test_markdown_extensions(self):
        extensions = _config(_ROOT)["markdown_extensions"]
        names = {next(iter(e)) if isinstance(e, dict) else e for e in extensions}
        assert {"admonition", "tables", "toc", "pymdownx.superfences"} <= names
        toc = next(e for e in extensions if isinstance(e, dict) and "toc" in e)["toc"]
        assert toc["permalink"]
        fences = next(
            e for e in extensions if isinstance(e, dict) and "pymdownx.superfences" in e
        )["pymdownx.superfences"]["custom_fences"]
        mermaid = next(f for f in fences if f["name"] == "mermaid")
        assert "pymdownx.superfences.fence_code_format" in mermaid["format"]


class TestDocsToolsAreAGroupNotAnExtra:
    """The docs tools need Python 3.11 and are for maintainers: a PEP 735 group, pinned."""

    @staticmethod
    def _pyproject() -> dict[str, Any]:
        return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_the_group_names_exactly_the_docs_tools(self):
        names = {
            Requirement(r).name for r in self._pyproject()["dependency-groups"]["docs"]
        }
        assert names == {
            "mkdocs",
            "mkdocs-material",
            "mkdocstrings",
            "mkdocstrings-python",
            "mike",
            "mkdocs-include-markdown-plugin",
            "pymdown-extensions",
        }

    def test_mkdocs_2_is_excluded(self):
        mkdocs = next(
            Requirement(r)
            for r in self._pyproject()["dependency-groups"]["docs"]
            if Requirement(r).name == "mkdocs"
        )
        assert not mkdocs.specifier.contains("2.0.0", prereleases=True)

    def test_no_extra_carries_a_docs_tool(self):
        extras = self._pyproject()["project"]["optional-dependencies"]
        assert "docs" not in extras
        tools = re.compile(r"mkdocs|mike\b|pymdown")
        carried = [
            (extra, req)
            for extra, reqs in extras.items()
            for req in reqs
            if tools.search(req)
        ]
        assert carried == []

    def test_the_makefile_has_the_docs_targets(self):
        makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
        phony = next(ln for ln in makefile.splitlines() if ln.startswith(".PHONY:"))
        for target in ("docs", "docs-serve", "docs-clean"):
            assert re.search(rf"^{target}:", makefile, re.M), target
            assert target in phony.split(), target
        build = re.search(r"^docs:.*\n\t(.+)$", makefile, re.M)
        assert build and "mkdocs build --strict" in build.group(1)
        clean = makefile.split("\nclean:", 1)[1].split("\n\n", 1)[0]
        assert "site/" in clean

    def test_the_build_output_is_ignored(self):
        lines = (_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert any(ln.strip().rstrip("/").lstrip("/") == "site" for ln in lines)

    def test_check_yaml_skips_mkdocs_yml(self):
        hooks = yaml.safe_load(
            (_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        )["repos"]
        check = next(h for r in hooks for h in r["hooks"] if h["id"] == "check-yaml")
        assert re.search(check["exclude"], "mkdocs.yml"), (
            "check-yaml cannot read the python tags of mkdocs.yml"
        )
        assert not re.search(check["exclude"], ".pre-commit-config.yaml")

    def test_the_sdist_does_not_carry_the_site_sources(self):
        # MANIFEST.in ships LICENSE, README, CHANGELOG, example JSON and src; docs/ is not in
        # the sdist today and mkdocs.yml follows that policy (the site is built from a checkout).
        manifest = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        assert "mkdocs" not in manifest
        assert "docs" not in manifest


# --------------------------------------------------------------------------------------
# The tolerant loader and the exclusion matcher
# --------------------------------------------------------------------------------------


class TestLoader:
    def test_python_name_tag_does_not_crash(self):
        text = (
            "markdown_extensions:\n"
            "  - pymdownx.superfences:\n"
            "      custom_fences:\n"
            "        - name: mermaid\n"
            "          class: mermaid\n"
            "          format: !!python/name:pymdownx.superfences.fence_code_format\n"
        )
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(text)
        fence = _load_config(text)["markdown_extensions"][0]["pymdownx.superfences"][
            "custom_fences"
        ][0]
        assert "pymdownx.superfences.fence_code_format" in fence["format"]

    def test_env_tag_and_tagged_collections_are_plain_data(self):
        data = _load_config("a: !ENV [HOME, x]\nb: !thing {k: 1}\nc: !!int '7'\n")
        assert data["a"] == ["HOME", "x"]
        assert data["b"] == {"k": 1}
        assert data["c"] == 7


class TestExclusion:
    @pytest.mark.parametrize(
        ("patterns", "rel", "want"),
        [
            (["AGENTS.md"], "AGENTS.md", True),
            (["AGENTS.md"], "a/b/AGENTS.md", True),
            (["AGENTS.md"], "AGENTS.mdx", False),
            (["/README.md"], "README.md", True),
            (["/README.md"], "sub/README.md", False),
            (["api.txt"], "api.txt", True),
            (["sub/"], "sub/a.md", True),
            (["sub/"], "sub", False),
            (["sub/*.md"], "sub/a.md", True),
            (["sub/*.md"], "sub/x/a.md", False),
            (["**/draft.md"], "a/b/draft.md", True),
            (["*.md", "!keep.md"], "keep.md", False),
            (["*.md", "!keep.md"], "other.md", True),
            (
                ["# comment"],
                "# comment",
                True,
            ),  # a comment is dropped by _exclude_patterns
            ([], "a.md", False),
        ],
    )
    def test_matcher(self, patterns, rel, want):
        if patterns == ["# comment"]:
            assert _exclude_patterns({"exclude_docs": "# comment\n\n"}) == []
            return
        assert _is_excluded(rel, patterns) is want

    def test_patterns_are_read_from_a_block_scalar(self):
        config = _load_config("exclude_docs: |\n  AGENTS.md\n  # why\n  api.txt\n")
        assert _exclude_patterns(config) == ["AGENTS.md", "api.txt"]

    def test_nav_files_flatten_nested_and_skip_urls(self):
        nav = [
            "index.md",
            {"Guides": ["a.md", {"B": "b/b.md"}]},
            {"Site": "https://example.org/x"},
        ]
        assert _nav_files(nav) == ["index.md", "a.md", "b/b.md"]


# --------------------------------------------------------------------------------------
# Mutation tests: one defect each on a small good tree; the checks must name it
# --------------------------------------------------------------------------------------

_GOOD_CONFIG = """\
site_name: Scratch
strict: true
docs_dir: docs
exclude_docs: |
  AGENTS.md
  api.txt
validation:
  nav:
    omitted_files: warn
    not_found: warn
    absolute_links: warn
  links:
    not_found: warn
    absolute_links: warn
    unrecognized_links: warn
    anchors: warn
nav:
  - Home: index.md
  - Guides:
      - guide.md
markdown_extensions:
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
"""


def _scratch(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "mkdocs.yml").write_text(_GOOD_CONFIG, encoding="utf-8")
    (tmp_path / "docs" / "index.md").write_text("# Home\n", encoding="utf-8")
    (tmp_path / "docs" / "guide.md").write_text(
        '# Guide\n\n{% include-markdown "../CHANGELOG.md" %}\n', encoding="utf-8"
    )
    (tmp_path / "docs" / "AGENTS.md").write_text("# notes\n", encoding="utf-8")
    (tmp_path / "docs" / "api.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Changes\n", encoding="utf-8")
    return tmp_path


def _edit(root: Path, old: str, new: str) -> None:
    text = (root / "mkdocs.yml").read_text(encoding="utf-8")
    assert old in text, old
    (root / "mkdocs.yml").write_text(text.replace(old, new), encoding="utf-8")


def _unlisted_page(root: Path) -> None:
    (root / "docs" / "orphan.md").write_text("# Orphan\n", encoding="utf-8")


def _agents_in_nav(root: Path) -> None:
    _edit(root, "      - guide.md\n", "      - guide.md\n      - AGENTS.md\n")


def _agents_not_excluded(root: Path) -> None:
    _edit(root, "  AGENTS.md\n", "")


def _missing_include(root: Path) -> None:
    (root / "CHANGELOG.md").unlink()


def _agents_included(root: Path) -> None:
    (root / "docs" / "index.md").write_text(
        '# Home\n\n{% include-markdown "AGENTS.md" %}\n', encoding="utf-8"
    )


def _nested_agents_included(root: Path) -> None:
    (root / "docs" / "sub").mkdir()
    (root / "docs" / "sub" / "AGENTS.md").write_text("# n\n", encoding="utf-8")
    (root / "docs" / "index.md").write_text(
        '# Home\n\n{% include "sub/AGENTS.md" %}\n', encoding="utf-8"
    )


def _drop_validation(root: Path) -> None:
    text = (root / "mkdocs.yml").read_text(encoding="utf-8")
    head, rest = text.split("validation:\n", 1)
    tail = rest.split("\nnav:\n", 1)[1]
    (root / "mkdocs.yml").write_text(head + "nav:\n" + tail, encoding="utf-8")


def _drop_one_validation_key(root: Path) -> None:
    _edit(root, "    anchors: warn\n", "    anchors: info\n")


def _validation_unrecognized_ignored(root: Path) -> None:
    _edit(root, "    unrecognized_links: warn\n", "    unrecognized_links: ignore\n")


def _all_pages_excluded_by_directory(root: Path) -> None:
    (root / "docs" / "extra").mkdir()
    (root / "docs" / "extra" / "page.md").write_text("# p\n", encoding="utf-8")


def _nav_page_is_excluded(root: Path) -> None:
    _edit(root, "  api.txt\n", "  api.txt\n  guide.md\n")


_MUTANTS = [
    (
        "strict off",
        lambda r: _edit(r, "strict: true", "strict: false"),
        "strict is not true",
    ),
    ("strict missing", lambda r: _edit(r, "strict: true\n", ""), "strict is not true"),
    (
        "strict a string",
        lambda r: _edit(r, "strict: true", "strict: 'true'"),
        "strict is not true",
    ),
    (
        "nav names a missing file",
        lambda r: _edit(r, "      - guide.md\n", "      - gone.md\n"),
        "'gone.md', which is not a file",
    ),
    (
        "page in no nav",
        _unlisted_page,
        "orphan.md is neither in the nav nor in exclude_docs",
    ),
    (
        "nested page in no nav",
        _all_pages_excluded_by_directory,
        "extra/page.md is neither",
    ),
    ("AGENTS.md in the nav", _agents_in_nav, "AGENTS.md is never published"),
    (
        "AGENTS.md not excluded",
        _agents_not_excluded,
        "AGENTS.md is neither in the nav nor in exclude_docs",
    ),
    (
        "exclude_docs missing",
        lambda r: _edit(r, "exclude_docs: |\n  AGENTS.md\n  api.txt\n", ""),
        "exclude_docs does not exclude AGENTS.md",
    ),
    (
        "exclude_docs covers only the root AGENTS.md",
        lambda r: _edit(r, "  AGENTS.md\n", "  /AGENTS.md\n"),
        "exclude_docs does not exclude AGENTS.md at every depth",
    ),
    (
        "include target missing",
        _missing_include,
        "includes '../CHANGELOG.md', which does not exist",
    ),
    (
        "AGENTS.md included",
        _agents_included,
        "includes 'AGENTS.md': AGENTS.md is never published",
    ),
    (
        "nested AGENTS.md included",
        _nested_agents_included,
        "AGENTS.md is never published",
    ),
    ("no validation block", _drop_validation, "the validation block is missing"),
    (
        "validation anchors left at info",
        _drop_one_validation_key,
        "validation.links.anchors is not 'warn'",
    ),
    (
        "validation unrecognized_links ignored",
        _validation_unrecognized_ignored,
        "validation.links.unrecognized_links",
    ),
    (
        "docs_dir wrong",
        lambda r: _edit(r, "docs_dir: docs", "docs_dir: nodocs"),
        "docs_dir 'nodocs' is not a directory",
    ),
    (
        "nav page removed by exclude_docs",
        _nav_page_is_excluded,
        "which exclude_docs removes",
    ),
]


class TestDefectsAreCaught:
    def test_the_scratch_tree_is_clean(self, tmp_path):
        assert _site_problems(_scratch(tmp_path)) == []

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [(m, e) for _, m, e in _MUTANTS],
        ids=[name for name, _, _ in _MUTANTS],
    )
    def test_each_defect_is_named(self, tmp_path, mutate, expected):
        root = _scratch(tmp_path)
        mutate(root)
        problems = _site_problems(root)
        assert any(expected in p for p in problems), (expected, problems)

    def test_a_missing_config_is_a_problem(self, tmp_path):
        assert _site_problems(tmp_path) == ["mkdocs.yml is missing"]

    def test_a_page_added_later_is_judged_from_the_disk(self, tmp_path):
        root = _scratch(tmp_path)
        (root / "docs" / "later.md").write_text("# Later\n", encoding="utf-8")
        assert _site_problems(root) == [
            "later.md is neither in the nav nor in exclude_docs"
        ]
        _edit(root, "      - guide.md\n", "      - guide.md\n      - later.md\n")
        assert _site_problems(root) == []
