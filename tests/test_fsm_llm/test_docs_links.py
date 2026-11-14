"""Every relative link and anchor of the docs pages resolves, without importing mkdocs.

Plan d454fd62, step 104 (roadmap row 0.7). The strict build (`make docs`) already fails on a
broken link, but it needs the docs tools; this test runs in the plain dev environment and names
the page, the link and the reason. It mirrors the `validation` block of `mkdocs.yml`
(`links.not_found`, `anchors`, `unrecognized_links`, `absolute_links`) on the page text:

* For every Markdown page under `docs/` (the nav pages and the pages `exclude_docs` removes, such
  as the GitHub-only `docs/README.md`): each relative inline link `[x](path#anchor)`, reference
  definition `[x]: path` and image `![x](path)` outside code fences and code spans must name an
  existing FILE (resolved against the page's directory, as MkDocs does), and when the target is a
  Markdown file the `#anchor` must be a heading id (the `toc` rules of Python-Markdown, see
  `_slug`), an explicit `{#id}` or an HTML `id`/`name`.
* A page the site publishes may not link to a file outside `docs/` (`../README.md`: MkDocs
  reports an unrecognised link; use an absolute GitHub URL or a site page), to a file that
  `exclude_docs` removes, to a directory, or by a root-absolute path (`/x.md`).
* `https://` and other URLs are never fetched (no network in tests).

What it does NOT check, by design: a page that includes a root file (`CHANGELOG.md`,
`CONTRIBUTING.md`, ...) with `include-markdown` is checked only on its own text, not on the
included region. `rewrite-relative-urls` (the default) rewrites an included link against the
INCLUDED file, so the page text cannot judge it; the strict build does. The headings of an
included file DO count as anchors of the including page (a link `changelog.md#unreleased`
resolves). A page with a `:::` block (mkdocstrings) generates anchors this test cannot see, so
anchors into it are not judged (its files are).

Every check is a function of a project root so a scratch tree with one defect each proves the
check can fail (`TestDefectsAreCaught`); the real tree is judged by `TestTheRealSite`.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import pytest

from tests.test_fsm_llm.test_docs_site import (
    _config,
    _docs_dir,
    _exclude_patterns,
    _is_excluded,
    _nav_files,
)

_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------------------
# Anchors: the `toc` extension of Python-Markdown (the one MkDocs and Material use)
# --------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    """`markdown.extensions.toc.slugify` with separator `-` (ASCII fold, strip, lowercase)."""
    value = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _unique(ident: str, taken: set[str]) -> str:
    """`markdown.extensions.toc.unique`: a repeated id becomes `id_1`, `id_2`, ..."""
    while ident in taken or not ident:
        m = re.match(r"^(.*)_([0-9]+)$", ident)
        ident = f"{m.group(1)}_{int(m.group(2)) + 1}" if m else f"{ident}_1"
    taken.add(ident)
    return ident


_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
_ATTR_ID = re.compile(r"\{[^}]*?#([\w.:-]+)[^}]*\}\s*$")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*$")
_FENCE = re.compile(r"^\s*(```+|~~~+)")


def _heading_text(raw: str) -> str:
    """The text content of a heading line after inline Markdown (approximate, see the cross-check)."""
    text = re.sub(r"[ \t]+#+[ \t]*$", "", raw)
    text = _CODE_SPAN.sub(lambda m: "\x00" + m.group(2).strip() + "\x00", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("*", "")
    text = re.sub(r"(?<!\w)(_{1,2})(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", text)
    return text.replace("\x00", "")


def _toc_ids(tokens: list[dict]) -> list[str]:
    """The ids of a `toc_tokens` tree of Python-Markdown, depth first (document order)."""
    out: list[str] = []
    for tok in tokens:
        out.append(tok["id"])
        out.extend(_toc_ids(tok["children"]))
    return out


def _lines_outside_fences(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    marker = ""
    for number, line in enumerate(text.splitlines(), 1):
        fence = _FENCE.match(line)
        if marker:
            if (
                fence
                and fence.group(1)[0] == marker[0]
                and len(fence.group(1)) >= len(marker)
            ):
                marker = ""
            continue
        if fence:
            marker = fence.group(1)
            continue
        out.append((number, line))
    return out


def _heading_ids(text: str) -> list[str]:
    """The id of every heading of a page, in order, as the `toc` extension assigns it."""
    taken: set[str] = set()
    explicit: list[str] = []
    rows: list[tuple[str | None, str]] = []
    for _, line in _lines_outside_fences(text):
        m = _HEADING.match(line)
        if not m:
            continue
        body = m.group(2)
        attr = _ATTR_ID.search(body)
        if attr:
            explicit.append(attr.group(1))
            body = body[: attr.start()].rstrip()
        rows.append((attr.group(1) if attr else None, body))
    taken.update(explicit)
    ids = []
    for ident, body in rows:
        ids.append(ident if ident else _unique(_slug(_heading_text(body)), taken))
    return ids


_HTML_ID = re.compile(r"""<[a-zA-Z][^>]*?\s(?:id|name)\s*=\s*["']([^"']+)["']""")
_BLOCK_ID = re.compile(r"\{:?\s*#([\w.:-]+)[^}]*\}")


def _own_anchors(text: str) -> set[str]:
    anchors = set(_heading_ids(text))
    anchors.update(m.group(1) for m in _HTML_ID.finditer(text))
    anchors.update(m.group(1) for m in _BLOCK_ID.finditer(text))
    return anchors


def _anchors_of(page: Path) -> set[str] | None:
    """Anchors of a page, headings of its `include-markdown` files included.

    `None` when the page holds a mkdocstrings block (`::: x`): its anchors are generated.
    """
    text = page.read_text(encoding="utf-8")
    if re.search(r"^:::\s", text, re.M):
        return None
    anchors = _own_anchors(text)
    in_fence_free = "\n".join(line for _, line in _lines_outside_fences(text))
    for target in _include_targets_outside_fences(in_fence_free):
        for hit in (
            sorted(page.parent.glob(target))
            if "*" in target
            else [page.parent / target]
        ):
            if hit.is_file() and hit.suffix == ".md":
                anchors |= _own_anchors(hit.read_text(encoding="utf-8"))
    return anchors


_INCLUDE = re.compile(r"\{%\s*include(?:-markdown)?\s+(['\"])(.+?)\1", re.DOTALL)


def _include_targets_outside_fences(text: str) -> list[str]:
    return [m.group(2) for m in _INCLUDE.finditer(text)]


# --------------------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------------------

_INLINE = re.compile(
    r"(!?)\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*(<[^>]*>|[^)\s]*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
_REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s+(<[^>]*>|\S+)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _links(text: str) -> list[tuple[int, str]]:
    """Every link destination outside fences and code spans, with its line number."""
    found: list[tuple[int, str]] = []
    for number, line in _lines_outside_fences(text):
        plain = _CODE_SPAN.sub("", line)
        for m in _INLINE.finditer(plain):
            found.append((number, m.group(2).strip("<>")))
        ref = _REFERENCE.match(plain)
        if ref:
            found.append((number, ref.group(1).strip("<>")))
    return found


def _link_problems(root: Path) -> list[str]:
    """One string per broken or unrecognised relative link of the docs pages of `root`."""
    if not (root / "mkdocs.yml").is_file():
        return ["mkdocs.yml is missing"]
    config = _config(root)
    docs = _docs_dir(root, config)
    patterns = _exclude_patterns(config)
    problems: list[str] = []
    cache: dict[Path, set[str] | None] = {}

    for page in sorted(docs.rglob("*.md")):
        rel = page.relative_to(docs).as_posix()
        published = not _is_excluded(rel, patterns)
        for number, dest in _links(page.read_text(encoding="utf-8")):
            where = f"{rel}:{number}: {dest!r}"
            if not dest or _SCHEME.match(dest) or dest.startswith("//"):
                continue  # an external URL, mailto: ...: never fetched
            if dest.startswith("/"):
                if published:
                    problems.append(
                        f"{where} is a root-absolute link: use a relative one"
                    )
                continue
            path_part, _, fragment = dest.partition("#")
            path_part = unquote(path_part.split("?", 1)[0])
            fragment = unquote(fragment)
            target = page if not path_part else (page.parent / path_part).resolve()
            inside = docs in target.parents or target == docs
            if not target.exists():
                problems.append(f"{where} names a file that does not exist")
                continue
            if target.is_dir():
                problems.append(f"{where} names a directory, not a page")
                continue
            if published and not inside:
                problems.append(
                    f"{where} leaves docs_dir: MkDocs reports it as unrecognised "
                    "(use an absolute GitHub URL or a site page)"
                )
                continue
            if (
                inside
                and published
                and _is_excluded(target.relative_to(docs).as_posix(), patterns)
            ):
                problems.append(
                    f"{where} names a file that exclude_docs removes from the site"
                )
                continue
            if fragment and target.suffix == ".md":
                if target not in cache:
                    cache[target] = _anchors_of(target)
                anchors = cache[target]
                if anchors is not None and fragment not in anchors:
                    problems.append(
                        f"{where}: no heading or id {fragment!r} in {target.name}"
                    )
    return problems


# --------------------------------------------------------------------------------------
# The real tree
# --------------------------------------------------------------------------------------


class TestTheRealSite:
    def test_no_relative_link_or_anchor_is_broken(self):
        assert _link_problems(_ROOT) == []

    def test_every_nav_page_is_checked(self):
        docs = _docs_dir(_ROOT, _config(_ROOT))
        names = _nav_files(_config(_ROOT)["nav"])
        assert names and all((docs / n).is_file() for n in names)

    def test_some_links_and_anchors_are_really_read(self):
        # a link test that finds nothing passes vacuously: the real pages hold both kinds
        docs = _docs_dir(_ROOT, _config(_ROOT))
        links = [
            d
            for p in docs.rglob("*.md")
            for _, d in _links(p.read_text(encoding="utf-8"))
        ]
        relative = [d for d in links if not _SCHEME.match(d)]
        assert len(relative) >= 10
        assert any("#" in d for d in relative)
        assert any(d.endswith(".md") for d in relative)

    def test_the_heading_ids_agree_with_python_markdown(self):
        """Cross-check `_heading_ids` against the real `toc` extension (docs tools only)."""
        markdown = pytest.importorskip("markdown")
        pytest.importorskip(
            "pymdownx"
        )  # the fence parser MkDocs uses ("```python notest" is one)
        docs = _docs_dir(_ROOT, _config(_ROOT))
        wrong = []
        for page in sorted(docs.rglob("*.md")):
            text = page.read_text(encoding="utf-8")
            if re.search(r"^:::\s", text, re.M) or _INCLUDE.search(text):
                continue
            md = markdown.Markdown(
                extensions=["toc", "attr_list", "pymdownx.superfences", "tables"]
            )
            md.convert(text)
            real = _toc_ids(md.toc_tokens)
            if real != _heading_ids(text):
                wrong.append(page.relative_to(_ROOT).as_posix())
        assert wrong == []


# --------------------------------------------------------------------------------------
# Fence info strings (a defect found while writing this file: see `_fence_openings`)
# --------------------------------------------------------------------------------------

#: What the site's fence parser (`pymdownx.superfences`) reads after the opening backticks: a
#: language, then optionally attributes `{.class #id}` or `key="value"`. A bare word after the
#: language (```` ```python notest ````) is NOT an attribute: the parser then fails to open the
#: block, the text that follows is read as Markdown and the next closing fence opens a new block.
_FENCE_INFO = re.compile(
    r"""^(?:[\w#.+-]*|\{[^}]*\})(?:\s+(?:\{[^}]*\}|[A-Za-z_][\w-]*="[^"]*"))*$"""
)


def _fence_openings(text: str) -> list[tuple[int, str]]:
    """The info string of every opening fence of a page, with its line number."""
    out: list[tuple[int, str]] = []
    marker = ""
    for number, line in enumerate(text.splitlines(), 1):
        fence = _FENCE.match(line)
        if not fence:
            continue
        if marker:
            if fence.group(1)[0] == marker[0] and len(fence.group(1)) >= len(marker):
                marker = ""
            continue
        marker = fence.group(1)
        out.append((number, line.strip()[len(marker) :].strip()))
    return out


class TestFenceInfoStrings:
    @pytest.mark.parametrize(
        ("info", "ok"),
        [
            ("", True),
            ("python", True),
            ("python {.notest}", True),
            ('python title="a b" hl_lines="1"', True),
            ("{.python .notest}", True),
            ("python notest", False),
            ("diff notest", False),
            ("json whatever", False),
        ],
    )
    def test_the_shape_rule(self, info, ok):
        assert bool(_FENCE_INFO.match(info)) is ok

    def test_no_docs_page_has_a_fence_the_site_cannot_open(self):
        docs = _docs_dir(_ROOT, _config(_ROOT))
        bad = [
            f"{page.relative_to(_ROOT).as_posix()}:{number}: {info!r}"
            for page in sorted(docs.rglob("*.md"))
            for number, info in _fence_openings(page.read_text(encoding="utf-8"))
            if not _FENCE_INFO.match(info)
        ]
        assert bad == []

    def test_the_scanner_pairs_openings_and_closings(self):
        text = "```python notest\nx\n```\n\n````md\n```\n````\n"
        assert _fence_openings(text) == [(1, "python notest"), (5, "md")]


# --------------------------------------------------------------------------------------
# The slug rules
# --------------------------------------------------------------------------------------


class TestSlug:
    @pytest.mark.parametrize(
        ("heading", "want"),
        [
            ("Key Features", "key-features"),
            ("Why FSM-LLM?", "why-fsm-llm"),
            ("`fsm-llm-eval`", "fsm-llm-eval"),
            ("Quick Start: 1. A model", "quick-start-1-a-model"),
            ("Cut-off replies (and a_b)", "cut-off-replies-and-a_b"),
            ("Résumé of the café", "resume-of-the-cafe"),
            ("A  --  B", "a-b"),
            ("[Linked](x.md) *text*", "linked-text"),
            ("**Bold** heading", "bold-heading"),
        ],
    )
    def test_heading_to_id(self, heading, want):
        assert _slug(_heading_text(heading)) == want

    def test_repeated_headings_get_an_underscore_counter(self):
        text = "# Same\n\n## Same\n\n### Same\n"
        assert _heading_ids(text) == ["same", "same_1", "same_2"]

    def test_an_explicit_id_wins_and_the_slug_is_not_generated(self):
        assert _heading_ids("## Title {#custom-id}\n") == ["custom-id"]

    def test_headings_inside_a_fence_are_not_headings(self):
        text = "# Real\n\n```bash\n# a comment\n```\n\n~~~\n# another\n~~~\n"
        assert _heading_ids(text) == ["real"]

    def test_closing_hashes_are_dropped(self):
        assert _heading_ids("## Closed ##\n") == ["closed"]


class TestLinkExtraction:
    def test_inline_reference_image_and_title_forms(self):
        text = (
            '[a](one.md) and ![i](pic.png) and [b](two.md#x "title")\n'
            "[ref]: three.md#y\n"
            "[ext](https://example.com/p) and [m](mailto:x@y.z)\n"
        )
        dests = [d for _, d in _links(text)]
        assert dests == [
            "one.md",
            "pic.png",
            "two.md#x",
            "three.md#y",
            "https://example.com/p",
            "mailto:x@y.z",
        ]

    def test_code_is_not_read(self):
        text = "`[a](gone.md)`\n\n```md\n[b](gone.md)\n```\n\n[c](real.md)\n"
        assert [d for _, d in _links(text)] == ["real.md"]


# --------------------------------------------------------------------------------------
# A scratch tree with one defect each
# --------------------------------------------------------------------------------------

_CONFIG = """\
site_name: t
docs_dir: docs
strict: true
exclude_docs: |
  AGENTS.md
  /README.md
nav:
  - Home: index.md
  - Guide: guide.md
"""

_INDEX = """\
# Home

See the [guide](guide.md), its [section](guide.md#a-section), [this page](#home),
the [repeat](guide.md#again_1) and the [site](https://example.com/x#y).
"""

_GUIDE = """\
# Guide

## A section

## Again

## Again

Back to [home](index.md#home). An image: ![i](pic.png).

```bash
# [not](a-link.md)
```
"""


def _scratch(tmp_path: Path, **overrides: str | None) -> Path:
    files = {
        "mkdocs.yml": _CONFIG,
        "docs/index.md": _INDEX,
        "docs/guide.md": _GUIDE,
        "docs/pic.png": "x",
        "README.md": "# root\n",
    }
    files.update(overrides)
    for name, content in files.items():
        if content is None:
            continue
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


_DEFECTS = [
    pytest.param(
        {"docs/index.md": _INDEX + "[x](gone.md)\n"},
        "names a file that does not exist",
        id="broken-relative-link",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x](guide.md#no-such-heading)\n"},
        "no heading or id 'no-such-heading'",
        id="broken-anchor",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x](#nope)\n"},
        "no heading or id 'nope'",
        id="broken-same-page-anchor",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x](../README.md)\n"},
        "leaves docs_dir",
        id="link-out-of-docs-dir",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x](/guide.md)\n"},
        "root-absolute link",
        id="root-absolute-link",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x](sub)\n", "docs/sub/x.md": "# x\n"},
        "names a directory",
        id="directory-link",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x](AGENTS.md)\n", "docs/AGENTS.md": "# a\n"},
        "exclude_docs removes",
        id="link-to-an-excluded-file",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "![i](missing.png)\n"},
        "names a file that does not exist",
        id="broken-image",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x]: gone.md\n"},
        "names a file that does not exist",
        id="broken-reference-definition",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x](guide.md#again_2)\n"},
        "no heading or id 'again_2'",
        id="anchor-counter-beyond-the-repeats",
    ),
    pytest.param(
        {"docs/index.md": _INDEX + "[x](guide.md#again-1)\n"},
        "no heading or id 'again-1'",
        id="dash-counter-is-not-the-toc-form",
    ),
    pytest.param(
        {"docs/guide.md": _GUIDE + "\n[x](index.md#missing)\n"},
        "no heading or id 'missing'",
        id="anchor-in-the-other-direction",
    ),
]


class TestDefectsAreCaught:
    def test_the_scratch_tree_is_clean(self, tmp_path):
        assert _link_problems(_scratch(tmp_path)) == []

    @pytest.mark.parametrize(("overrides", "expected"), _DEFECTS)
    def test_each_defect_is_named(self, tmp_path, overrides, expected):
        problems = _link_problems(_scratch(tmp_path, **overrides))
        assert any(expected in p for p in problems), problems

    def test_an_excluded_page_may_link_out_of_docs_dir(self, tmp_path):
        root = _scratch(
            tmp_path, **{"docs/README.md": "[r](../README.md) [i](index.md)\n"}
        )
        assert _link_problems(root) == []

    def test_an_excluded_page_is_still_checked_for_missing_files(self, tmp_path):
        root = _scratch(tmp_path, **{"docs/README.md": "[r](gone.md)\n"})
        assert any("README.md:1" in p for p in _link_problems(root))

    def test_an_explicit_id_and_an_html_id_are_anchors(self, tmp_path):
        root = _scratch(
            tmp_path,
            **{
                "docs/guide.md": _GUIDE
                + '\n## Odd title {#custom}\n\n<a id="hand"></a>\n',
                "docs/index.md": _INDEX + "[x](guide.md#custom) [y](guide.md#hand)\n",
            },
        )
        assert _link_problems(root) == []

    def test_a_percent_encoded_link_is_decoded(self, tmp_path):
        root = _scratch(
            tmp_path,
            **{
                "docs/my page.md": "# P\n",
                "docs/index.md": _INDEX + "[x](my%20page.md)\n",
            },
        )
        assert _link_problems(root) == []

    def test_the_headings_of_an_included_file_are_anchors(self, tmp_path):
        root = _scratch(
            tmp_path,
            **{
                "CHANGELOG.md": "# Changelog\n\n## Unreleased\n",
                "docs/guide.md": _GUIDE
                + '\n{% include-markdown "../CHANGELOG.md" %}\n',
                "docs/index.md": _INDEX + "[x](guide.md#unreleased)\n",
            },
        )
        assert _link_problems(root) == []

    def test_a_link_inside_an_included_file_is_not_judged_here(self, tmp_path):
        # the strict build judges it (the plugin rewrites it against the included file)
        root = _scratch(
            tmp_path,
            **{
                "CHANGELOG.md": "# C\n\n[x](docs/missing.md)\n",
                "docs/guide.md": _GUIDE
                + '\n{% include-markdown "../CHANGELOG.md" %}\n',
            },
        )
        assert _link_problems(root) == []

    def test_an_anchor_into_a_mkdocstrings_page_is_not_judged(self, tmp_path):
        root = _scratch(
            tmp_path,
            **{
                "docs/ref.md": "# r\n\n::: some.module\n",
                "docs/index.md": _INDEX + "[x](ref.md#some.module.Thing)\n",
            },
        )
        assert _link_problems(root) == []

    def test_a_missing_config_is_a_problem(self, tmp_path):
        assert _link_problems(tmp_path) == ["mkdocs.yml is missing"]
