"""Tests for ``fsm_llm_harness.storage``.

Every fixture below is lifted from this repository's own ``plans/`` tree and
keeps the details a synthetic stub would smooth away.  ``plans/`` is gitignored
(``.gitignore:183``), so the content is EMBEDDED here rather than read from
disk -- but it is real content, not invented shapes:

* ``LESSONS_MD`` keeps real ``[I:N]`` bullets with real multi-line continuation
  lines and real bold-inside-bullet markup, and it keeps the protocol's exact
  four section headings including ``Failed Approaches (+ why)`` with its
  parenthesised suffix.  It carries an ``[I:5]`` in every section, which is what
  makes "never evict a 5" falsifiable rather than vacuous.
* ``SYSTEM_MD`` keeps ``## Identity`` as PROSE.  That single detail is why
  SYSTEM is capped-and-refused rather than bullet-evicted (D-017): a bullet
  rewrite of that section would delete it silently.
* ``DECISIONS_MD`` keeps the real ``<!-- COMPRESSED-SUMMARY -->`` block from
  ``plans/DECISIONS.md:3-68`` -- open marker as the last preamble line, an
  ``## Summary (compressed)`` H2 INSIDE the block, ``### `` sub-headings, and
  the close marker as the block's last line.  A window that treats the block as
  an ordinary section, or that wraps it instead of writing into it, fails here.
* ``FINDINGS_MD`` keeps the real ``plans/FINDINGS.md`` shape, which has NO
  compressed block, so block CREATION is exercised against a real file too.
* Plan ids are the real directory names from ``plans/``.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fsm_llm_harness.artifacts import (
    ConsolidatedDoc,
    LessonsDoc,
    Section,
    StateDoc,
    SystemAtlasDoc,
    compression_marker_issues,
    lesson_importance,
)
from fsm_llm_harness.constants import ArtifactNames, Defaults, Role
from fsm_llm_harness.exceptions import (
    HarnessArtifactError,
    HarnessConfinementError,
    HarnessOwnershipError,
)
from fsm_llm_harness.storage import (
    COMPRESSED_SUMMARY_CLOSE,
    COMPRESSED_SUMMARY_OPEN,
    COMPRESSED_SUMMARY_SECTION,
    PLAN_ID_RE,
    CapReport,
    PlanDirectory,
    RunState,
    _atomic_write_text,
    apply_sliding_window,
    check_system_cap,
    evict_lessons,
    mint_plan_id,
)

# ---------------------------------------------------------------------------
# Real fixtures
# ---------------------------------------------------------------------------

#: Real plan-directory names from this repository's ``plans/`` tree.
PLAN_A = "plan-2026-07-21T191807-bf7ffe24"
PLAN_B = "plan-2026-07-21T125237-191b2eb2"
PLAN_C = "plan-2026-07-21T110044-ed1ae68b"
PLAN_D = "plan-2026-07-21T101549-b662934a"
PLAN_E = "plan-2026-07-21T082818-4c63deac"
PLAN_F = "plan-2026-07-21T072826-e3131cc2"

LESSONS_MD = """# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines — rewrite, don't append forever.*
*Read before any PLAN state. This is institutional memory.*

## Recurring Patterns
- A remediation that fixes one code path of a bug cluster must sweep ALL
  sibling call sites, not just the one named in the finding. [I:5]
- Pre-staging the next iteration's plan draft while the current one executes
  keeps a multi-iteration plan moving without idle handoff gaps. [I:3]
- When an adversarial reviewer flags a behavioral asymmetry, check for
  existing precedent before treating it as a new bug. [I:2]

## Failed Approaches (+ why)
- Trusting a test fixture that stubs a narrow attribute shape instead of a
  real third-party object: it goes green on broken code. [I:5]
- **Ask of every fixture: does this make the interesting case degenerate?**
  Confirmed across two plans, 4+ instances. [I:4]

## Successful Strategies
- When a fix only needs to intercept a value at one seam that already drains
  through a single shared helper, fix the helper, not each call site. [I:5]
- Reuse existing lifecycle/rollback machinery instead of inventing new
  abstractions. Earn any new abstraction with 3+ concrete call sites. [I:4]
- In-code `# DECISION <plan-id>/D-NNN` anchors that state what NOT to do are
  worth the cost. Use the FULL plan-id from the start. [I:3]

## Codebase Gotchas
- litellm renames/deletes fields between provider response shapes across its
  own pinned version range; `hasattr()` guards are fragile. [I:5]
- `fsm.py`'s `_lock -> conv_lock` acquisition order must never invert. [I:4]
- An untagged bullet is treated as importance 3 by the protocol's own rule.
"""

SYSTEM_MD = """# System Atlas
*Last refreshed: plan-2026-07-21T125237-191b2eb2 | 2026-07-21*
*Domain-neutral system map. Rewritten at CLOSE — max 300 lines.*

## Identity
FSM-LLM (v0.5.0): a Python framework for building stateful conversational AI by
combining LLMs with Finite State Machines. Domain: codebase. Core idea: a 2-pass
turn cycle drives an explicit FSM per conversation.

## Components
- `fsm_llm` (core, 23 files) — FSM orchestration, 2-pass message pipeline.
- `fsm_llm_agents` (49 files) — 12 agentic patterns + swarm/graph/MCP/A2A.
- `fsm_llm_harness` (11 files, PARTIAL) — protocol emulation.

## Boundaries
- `examples/` (100 dirs) are stable evaluation baselines — never modify.

## Invariants
- Exactly one writing role per artifact (the File Ownership Model).

## Flows
- User input → Pass 1 (extract + evaluate) → Pass 2 (respond).

## Known Patterns
- `build_*_fsm()` dict factories returning `dict[str, Any]`.
"""

_SUMMARY_BLOCK = f"""{COMPRESSED_SUMMARY_OPEN}
## Summary (compressed)
*Auto-compressed from 2600 lines (4 plan sections). Read full content below if needed.*

### Key Outcomes
- **{PLAN_E}** (fsm_llm_harness, 65 decisions) — PARTIALLY COMPLETE, closed at
  user direction.

### Anchored Decisions (high-value, non-obvious)
- {PLAN_E} D-001/D-002 → `src/fsm_llm/utilities.py:99`, `fsm.py:452`.
{COMPRESSED_SUMMARY_CLOSE}"""


def _plan_section(plan_id: str, marker: str) -> str:
    return f"""## {plan_id}
### D-001 | EXPLORE → PLAN | 2026-07-21
**Context**: {marker} background.
**Decision**: {marker} approach.
**Trade-off**: speed **at the cost of** generality."""


DECISIONS_MD = (
    "\n\n".join(
        [
            "# Consolidated Decisions",
            "*Cross-plan decision archive. Newest first.*\n" + _SUMMARY_BLOCK,
            _plan_section(PLAN_A, "newest"),
            _plan_section(PLAN_B, "second"),
            _plan_section(PLAN_C, "third"),
            _plan_section(PLAN_D, "fourth"),
            _plan_section(PLAN_E, "fifth"),
            _plan_section(PLAN_F, "oldest"),
        ]
    )
    + "\n"
)

FINDINGS_MD = (
    "\n\n".join(
        [
            "# Consolidated Findings",
            "*Cross-plan findings archive. Merged from per-plan findings.md on close. "
            "Newest first.*",
            f"## {PLAN_A}\n### Index\n\n1. harness remaining tier — "
            "`findings/harness-remaining-tier.md`",
            f"## {PLAN_B}\n### Index\n\n1. iterative-planner source spec — "
            "`findings/iterative-planner-source.md`",
            f"## {PLAN_C}\n### Index\n\n1. monitor surface — `findings/monitor.md`",
            f"## {PLAN_D}\n### Index\n\n1. meta builder — `findings/meta.md`",
            f"## {PLAN_E}\n### Index\n\n1. utilities — `findings/utilities.md`",
        ]
    )
    + "\n"
)

STATE_MD = """# Current State: EXECUTE
*Skill: iterative-planner v2.56.0*
## Iteration: 1
## Current Plan Step: 8 (storage.py)
## Pre-Step Checklist (reset before each EXECUTE step)
- [x] Re-read state.md (this file)
- [ ] Re-read plan.md
## Fix Attempts (resets per plan step)
- Step 4b, attempt 1
- Step 4b, attempt 2
## Change Manifest (current iteration)
- step 7 (`65fc547`): `src/fsm_llm_harness/artifacts.py` NEW (1,627 lines).
## Last Transition: REFLECT → EXECUTE (2026-07-22T00:45:00Z)
## Transition History:
- INIT → EXPLORE (task started)
- EXPLORE → PLAN (gathered enough context)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_root(tmp_path: Path) -> Path:
    """A ``plans/``-shaped memory root."""
    root = tmp_path / "plans"
    root.mkdir()
    return root


@pytest.fixture
def archivist(memory_root: Path) -> PlanDirectory:
    return PlanDirectory(memory_root / PLAN_A, role=Role.ARCHIVIST)


@pytest.fixture
def orchestrator(memory_root: Path) -> PlanDirectory:
    return PlanDirectory(memory_root / PLAN_A, role=Role.ORCHESTRATOR)


def _seed(directory: PlanDirectory, path: str, text: str) -> None:
    """Place a file directly on disk, bypassing ownership (test setup only)."""
    target = directory.root / directory.memory.locate(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _tmp_files(directory: Path) -> list[str]:
    return [entry.name for entry in directory.iterdir() if entry.name.endswith(".tmp")]


# ---------------------------------------------------------------------------
# Plan-id minting
# ---------------------------------------------------------------------------


class TestMintPlanId:
    def test_shape_matches_the_published_grammar(self):
        assert PLAN_ID_RE.match(mint_plan_id()) is not None

    def test_stamps_the_supplied_clock(self):
        moment = datetime(2026, 7, 21, 19, 18, 7, tzinfo=timezone.utc)
        assert mint_plan_id(now=moment).startswith("plan-2026-07-21T191807-")

    def test_ids_minted_in_the_same_second_are_distinct(self):
        moment = datetime(2026, 7, 21, 19, 18, 7, tzinfo=timezone.utc)
        minted = {mint_plan_id(now=moment) for _ in range(200)}
        assert len(minted) == 200

    def test_hex_suffix_is_eight_lowercase_hex_digits(self):
        suffix = mint_plan_id().rsplit("-", 1)[1]
        assert re.fullmatch(r"[0-9a-f]{8}", suffix) is not None

    def test_minted_id_is_recognised_as_a_plan_section_heading(self):
        """The minter and the artifact layer's recogniser must agree.

        If they drift, a freshly minted plan's section is invisible to the
        sliding window and never ages out.
        """
        plan_id = mint_plan_id()
        doc = ConsolidatedDoc(
            title="Consolidated Findings", sections=[Section(name=plan_id)]
        )
        assert doc.plan_ids() == [plan_id]

    def test_every_real_plan_id_fixture_matches_the_grammar(self):
        for plan_id in (PLAN_A, PLAN_B, PLAN_C, PLAN_D, PLAN_E, PLAN_F):
            assert PLAN_ID_RE.match(plan_id) is not None


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWriteText:
    def test_writes_the_exact_content(self, tmp_path: Path):
        target = tmp_path / "state.md"
        _atomic_write_text(target, STATE_MD, artifact="state.md")
        assert target.read_text(encoding="utf-8") == STATE_MD

    def test_creates_missing_parent_directories(self, tmp_path: Path):
        target = tmp_path / "findings" / "topic.md"
        _atomic_write_text(target, "# Topic\n", artifact="findings")
        assert target.read_text(encoding="utf-8") == "# Topic\n"

    def test_replaces_existing_content_wholesale(self, tmp_path: Path):
        target = tmp_path / "state.md"
        target.write_text("a much longer previous body\n" * 20, encoding="utf-8")
        _atomic_write_text(target, "short\n", artifact="state.md")
        assert target.read_text(encoding="utf-8") == "short\n"

    def test_leaves_no_temp_file_on_success(self, tmp_path: Path):
        _atomic_write_text(tmp_path / "state.md", STATE_MD, artifact="state.md")
        assert _tmp_files(tmp_path) == []

    def test_crash_between_write_and_replace_leaves_the_old_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The property the whole module exists for: old-or-new, never a blend."""
        target = tmp_path / "LESSONS.md"
        target.write_text(LESSONS_MD, encoding="utf-8")

        def crash(src, dst):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(os, "replace", crash)
        with pytest.raises(HarnessArtifactError):
            _atomic_write_text(target, "TRUNCATED", artifact="LESSONS.md")
        assert target.read_text(encoding="utf-8") == LESSONS_MD

    def test_crash_between_write_and_replace_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        target = tmp_path / "LESSONS.md"
        target.write_text(LESSONS_MD, encoding="utf-8")
        monkeypatch.setattr(
            os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom"))
        )
        with pytest.raises(HarnessArtifactError):
            _atomic_write_text(target, "TRUNCATED", artifact="LESSONS.md")
        assert _tmp_files(tmp_path) == []

    def test_crash_leaves_no_file_at_all_when_the_target_did_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        target = tmp_path / "LESSONS.md"
        monkeypatch.setattr(
            os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom"))
        )
        with pytest.raises(HarnessArtifactError):
            _atomic_write_text(target, "half", artifact="LESSONS.md")
        assert not target.exists()
        assert _tmp_files(tmp_path) == []

    def test_temp_file_is_created_beside_the_target_not_in_slash_tmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """``os.replace`` is only atomic within one filesystem (D-019)."""
        seen: list[str | None] = []
        real = tempfile.mkstemp

        def spy(*args, **kwargs):
            seen.append(kwargs.get("dir"))
            return real(*args, **kwargs)

        monkeypatch.setattr(tempfile, "mkstemp", spy)
        target = tmp_path / "findings" / "topic.md"
        _atomic_write_text(target, "# Topic\n", artifact="findings")
        assert seen == [str(target.parent)]
        assert seen[0] != tempfile.gettempdir()

    def test_os_error_is_reported_as_a_tagged_artifact_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom"))
        )
        with pytest.raises(HarnessArtifactError) as excinfo:
            _atomic_write_text(tmp_path / "x.md", "body", artifact="decisions.md")
        assert excinfo.value.artifact == "decisions.md"
        assert isinstance(excinfo.value.cause, OSError)

    def test_unopenable_directory_is_reported_not_crashed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            tempfile,
            "mkstemp",
            lambda **kwargs: (_ for _ in ()).throw(OSError("read-only")),
        )
        with pytest.raises(HarnessArtifactError) as excinfo:
            _atomic_write_text(tmp_path / "x.md", "body", artifact="plan.md")
        assert "temp file" in str(excinfo.value)


# ---------------------------------------------------------------------------
# LESSONS.md -- 200-line cap, [I:N] eviction
# ---------------------------------------------------------------------------


def _lessons() -> LessonsDoc:
    doc = LessonsDoc.from_markdown(LESSONS_MD)
    assert isinstance(doc, LessonsDoc)
    return doc


class TestEvictLessons:
    def test_the_real_fixture_parses_with_the_expected_importance_mix(self):
        levels = sorted(importance for importance, _ in _lessons().lessons())
        assert levels == [2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5]

    def test_under_cap_returns_the_document_untouched(self):
        doc = _lessons()
        result, report = evict_lessons(doc, cap=Defaults.LESSONS_LINE_CAP)
        assert result is doc
        assert report.changed is False
        assert report.evicted == ()
        assert report.over_cap is False

    def test_the_real_default_cap_is_two_hundred_lines(self):
        _, report = evict_lessons(_lessons())
        assert report.cap == 200

    def test_lowest_importance_is_evicted_first(self):
        doc = _lessons()
        _, report = evict_lessons(doc, cap=doc.line_count - 1)
        assert len(report.evicted) == 1
        assert "behavioral asymmetry" in report.evicted[0]

    def test_eviction_walks_up_the_importance_tiers(self):
        doc = _lessons()
        # The one [I:2] and the three [I:3]-equivalents (one of them untagged)
        # occupy 7 lines between them; freeing exactly that much must take all
        # four and leave every [I:4] alone.
        _, report = evict_lessons(doc, cap=doc.line_count - 7)
        levels = sorted(lesson_importance(item) for item in report.evicted)
        assert levels == [2, 3, 3, 3]

    def test_oldest_first_within_a_tier(self):
        """Both ``[I:3]``-equivalent bullets are tied; document order decides."""
        doc = LessonsDoc(
            title="Lessons Learned",
            sections=[
                Section(name="Recurring Patterns", body="- older tie [I:3]"),
                Section(name="Codebase Gotchas", body="- newer tie [I:3]"),
            ],
        )
        _, report = evict_lessons(doc, cap=doc.line_count - 1)
        assert report.evicted == ("older tie [I:3]",)

    def test_never_evicts_a_protected_five(self):
        doc = _lessons()
        result, report = evict_lessons(doc, cap=1)
        surviving = result.lessons()
        assert surviving, "eviction must not empty the file"
        assert {importance for importance, _ in surviving} == {5}
        assert len(surviving) == 4
        assert all("[I:5]" not in item for item in report.evicted)

    def test_reports_still_over_cap_when_only_protected_lessons_remain(self):
        _, report = evict_lessons(_lessons(), cap=1)
        assert report.over_cap is True
        assert report.changed is True

    def test_untagged_bullets_are_treated_as_importance_three(self):
        doc = _lessons()
        _, report = evict_lessons(doc, cap=doc.line_count - 7)
        untagged = [item for item in report.evicted if "untagged bullet" in item]
        assert untagged and lesson_importance(untagged[0]) == 3

    def test_evicted_text_is_removed_from_the_returned_document(self):
        doc = _lessons()
        result, report = evict_lessons(doc, cap=doc.line_count - 1)
        rendered = result.to_markdown()
        assert report.evicted[0] not in rendered

    def test_multi_line_bullets_survive_eviction_intact(self):
        doc = _lessons()
        result, _ = evict_lessons(doc, cap=doc.line_count - 1)
        assert "sibling call sites, not just the one named in the finding" in (
            result.to_markdown()
        )

    def test_result_still_parses_as_a_lessons_document(self):
        result, _ = evict_lessons(_lessons(), cap=12)
        assert LessonsDoc.from_markdown(result.to_markdown()) == result

    def test_a_section_holding_prose_is_refused_not_silently_rewritten(self):
        """D-017: ``_parse_bullets`` discards prose, so a rewrite would eat it."""
        doc = LessonsDoc(
            title="Lessons Learned",
            sections=[
                Section(
                    name="Recurring Patterns",
                    body="A paragraph the archivist wrote.\n- a bullet [I:1]",
                )
            ],
        )
        with pytest.raises(HarnessArtifactError) as excinfo:
            evict_lessons(doc, cap=1)
        assert "not a bullet" in str(excinfo.value)

    def test_prose_is_refused_even_when_the_document_is_under_cap(self):
        doc = LessonsDoc(
            title="Lessons Learned",
            sections=[Section(name="Recurring Patterns", body="Just prose.")],
        )
        with pytest.raises(HarnessArtifactError):
            evict_lessons(doc, cap=Defaults.LESSONS_LINE_CAP)


# ---------------------------------------------------------------------------
# SYSTEM.md -- 300-line cap
# ---------------------------------------------------------------------------


def _system(extra_components: int = 0) -> SystemAtlasDoc:
    doc = SystemAtlasDoc.from_markdown(SYSTEM_MD)
    if not extra_components:
        return doc
    sections = [
        Section(
            name=section.name,
            body=section.body
            + "".join(f"\n- padding component {n}." for n in range(extra_components)),
        )
        if section.name == "Components"
        else section
        for section in doc.sections
    ]
    return SystemAtlasDoc(title=doc.title, preamble=doc.preamble, sections=sections)


class TestSystemCap:
    def test_the_real_fixture_is_under_cap(self):
        report = check_system_cap(_system())
        assert report.over_cap is False
        assert report.cap == 300

    def test_the_cap_is_the_protocol_constant(self):
        assert check_system_cap(_system()).cap == Defaults.SYSTEM_LINE_CAP

    def test_over_cap_is_reported(self):
        report = check_system_cap(_system(extra_components=400))
        assert report.over_cap is True
        assert report.lines_before > 300

    def test_check_never_evicts_anything(self):
        report = check_system_cap(_system(extra_components=400))
        assert report.evicted == ()
        assert report.changed is False
        assert report.lines_after == report.lines_before

    def test_identity_section_is_prose_which_is_why_it_is_not_bullet_evicted(self):
        """The measured fact behind D-017's asymmetry with LESSONS.md."""
        identity = _system().section("Identity")
        assert identity is not None
        assert not identity.body.startswith("- ")

    def test_a_custom_cap_is_honoured(self):
        assert check_system_cap(_system(), cap=5).over_cap is True


# ---------------------------------------------------------------------------
# The 4-plan sliding window
# ---------------------------------------------------------------------------


def _decisions() -> ConsolidatedDoc:
    return ConsolidatedDoc.from_markdown(DECISIONS_MD)


def _findings() -> ConsolidatedDoc:
    return ConsolidatedDoc.from_markdown(FINDINGS_MD)


class TestSlidingWindow:
    def test_the_real_fixture_carries_a_well_formed_compressed_block(self):
        assert compression_marker_issues(DECISIONS_MD) == []
        assert _decisions().section(COMPRESSED_SUMMARY_SECTION) is not None

    def test_keeps_exactly_the_four_most_recent_plan_sections(self):
        result, report = apply_sliding_window(_decisions())
        assert result.plan_ids() == [PLAN_A, PLAN_B, PLAN_C, PLAN_D]
        assert report.kept_plans == (PLAN_A, PLAN_B, PLAN_C, PLAN_D)
        assert report.trimmed_plans == (PLAN_E, PLAN_F)

    def test_the_default_window_is_the_protocol_constant(self):
        _, report = apply_sliding_window(_decisions())
        assert report.keep == Defaults.SLIDING_WINDOW_PLANS == 4

    def test_trimmed_content_is_gone_from_the_document(self):
        result, _ = apply_sliding_window(_decisions())
        assert "oldest background" not in result.to_markdown()
        assert "newest background" in result.to_markdown()

    def test_trimmed_plans_are_recorded_in_the_compressed_block(self):
        result, _ = apply_sliding_window(_decisions())
        block = result.section(COMPRESSED_SUMMARY_SECTION)
        assert block is not None
        assert PLAN_E in block.body
        assert PLAN_F in block.body

    def test_new_bullets_land_above_the_closing_marker(self):
        result, _ = apply_sliding_window(_decisions())
        block = result.section(COMPRESSED_SUMMARY_SECTION)
        assert block is not None
        lines = block.body.split("\n")
        assert COMPRESSED_SUMMARY_CLOSE in lines[-1]
        assert any(PLAN_F in line for line in lines[:-1])

    def test_the_pre_existing_summary_content_is_preserved(self):
        result, _ = apply_sliding_window(_decisions())
        block = result.section(COMPRESSED_SUMMARY_SECTION)
        assert block is not None
        assert "Key Outcomes" in block.body
        assert "Anchored Decisions" in block.body

    def test_the_block_is_never_nested_in_itself(self):
        result, _ = apply_sliding_window(_decisions())
        assert compression_marker_issues(result.to_markdown()) == []
        assert result.to_markdown().count(COMPRESSED_SUMMARY_OPEN) == 1

    def test_a_second_window_pass_writes_into_the_same_block(self):
        once, _ = apply_sliding_window(_decisions())
        grown = ConsolidatedDoc(
            title=once.title,
            preamble=once.preamble,
            sections=[
                once.sections[0],
                Section(
                    name=mint_plan_id(), body="### D-001 | EXPLORE → PLAN | 2026-07-22"
                ),
                *once.sections[1:],
            ],
        )
        twice, report = apply_sliding_window(grown)
        assert report.trimmed_plans == (PLAN_D,)
        assert twice.to_markdown().count(COMPRESSED_SUMMARY_OPEN) == 1
        assert compression_marker_issues(twice.to_markdown()) == []

    def test_the_summary_section_is_never_itself_trimmed(self):
        result, report = apply_sliding_window(_decisions(), keep=1)
        assert result.section(COMPRESSED_SUMMARY_SECTION) is not None
        assert COMPRESSED_SUMMARY_SECTION not in report.trimmed_plans

    def test_creates_the_block_when_the_file_has_none(self):
        result, report = apply_sliding_window(_findings())
        assert report.trimmed_plans == (PLAN_E,)
        assert COMPRESSED_SUMMARY_OPEN in result.preamble
        block = result.section(COMPRESSED_SUMMARY_SECTION)
        assert block is not None
        assert block.body.split("\n")[-1] == COMPRESSED_SUMMARY_CLOSE
        assert compression_marker_issues(result.to_markdown()) == []

    def test_the_created_block_is_the_first_section(self):
        result, _ = apply_sliding_window(_findings())
        assert result.sections[0].name == COMPRESSED_SUMMARY_SECTION

    def test_within_the_window_nothing_changes(self):
        doc = _findings()
        result, report = apply_sliding_window(doc, keep=10)
        assert result is doc
        assert report.changed is False
        assert report.trimmed_plans == ()
        assert report.kept_plans == tuple(doc.plan_ids())

    def test_result_round_trips_through_the_artifact_parser(self):
        result, _ = apply_sliding_window(_decisions())
        assert ConsolidatedDoc.from_markdown(result.to_markdown()) == result

    def test_a_custom_summariser_is_used(self):
        result, _ = apply_sliding_window(
            _decisions(), summarise=lambda plan_id, section: f"GONE {plan_id}"
        )
        block = result.section(COMPRESSED_SUMMARY_SECTION)
        assert block is not None
        assert f"GONE {PLAN_F}" in block.body

    def test_a_window_below_one_is_refused(self):
        with pytest.raises(HarnessArtifactError):
            apply_sliding_window(_decisions(), keep=0)


# ---------------------------------------------------------------------------
# PlanDirectory -- layout and minting
# ---------------------------------------------------------------------------


class TestPlanDirectoryLayout:
    def test_create_mints_and_opens_a_fresh_directory(self, memory_root: Path):
        directory = PlanDirectory.create(memory_root)
        assert PLAN_ID_RE.match(directory.plan_id) is not None
        assert directory.path.is_dir()
        assert directory.root == memory_root.resolve()

    def test_create_never_reuses_an_existing_directory(self, memory_root: Path):
        moment = datetime(2026, 7, 21, 19, 18, 7, tzinfo=timezone.utc)
        first = PlanDirectory.create(memory_root, now=moment)
        second = PlanDirectory.create(memory_root, now=moment)
        assert first.plan_id != second.plan_id

    def test_create_gives_up_rather_than_colliding(
        self, memory_root: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from fsm_llm_harness import storage

        monkeypatch.setattr(storage, "mint_plan_id", lambda **kwargs: PLAN_A)
        (memory_root / PLAN_A).mkdir()
        with pytest.raises(HarnessArtifactError):
            PlanDirectory.create(memory_root)

    def test_role_defaults_to_the_orchestrator(self, memory_root: Path):
        assert PlanDirectory(memory_root / PLAN_A).role == Role.ORCHESTRATOR

    def test_repr_names_the_plan_and_the_role(self, archivist: PlanDirectory):
        assert PLAN_A in repr(archivist)
        assert Role.ARCHIVIST in repr(archivist)

    def test_finding_path_slugifies_a_topic(self):
        assert (
            PlanDirectory.finding_path("Write-tool selection RCA")
            == "findings/write-tool-selection-rca.md"
        )

    def test_finding_path_refuses_a_topic_with_no_slug(self):
        with pytest.raises(HarnessArtifactError):
            PlanDirectory.finding_path("   ***   ")

    def test_checkpoint_path_uses_the_protocol_shape(self):
        assert PlanDirectory.checkpoint_path(0, 1) == "checkpoints/cp-000-iter1.md"
        assert PlanDirectory.checkpoint_path(12, 3) == "checkpoints/cp-012-iter3.md"

    def test_checkpoint_path_refuses_negative_indices(self):
        with pytest.raises(HarnessArtifactError):
            PlanDirectory.checkpoint_path(-1, 1)


# ---------------------------------------------------------------------------
# PlanDirectory -- confinement and ownership, through PlanMemory
# ---------------------------------------------------------------------------


class TestPlanDirectoryAuthorisation:
    def test_an_owned_write_lands_on_disk(self, memory_root: Path):
        explorer = PlanDirectory(memory_root / PLAN_A, role=Role.EXPLORER)
        explorer.write_text(PlanDirectory.finding_path("tool scope"), "# Tool scope\n")
        assert (memory_root / PLAN_A / "findings" / "tool-scope.md").read_text() == (
            "# Tool scope\n"
        )

    def test_an_unowned_write_is_refused(self, memory_root: Path):
        explorer = PlanDirectory(memory_root / PLAN_A, role=Role.EXPLORER)
        with pytest.raises(HarnessOwnershipError):
            explorer.write_text(ArtifactNames.PLAN, "# Plan\n")

    def test_a_refused_write_leaves_no_file(self, memory_root: Path):
        explorer = PlanDirectory(memory_root / PLAN_A, role=Role.EXPLORER)
        with pytest.raises(HarnessOwnershipError):
            explorer.write_text(ArtifactNames.PLAN, "# Plan\n")
        assert not (memory_root / PLAN_A / ArtifactNames.PLAN).exists()

    def test_a_refused_write_leaves_no_temp_file(self, memory_root: Path):
        explorer = PlanDirectory(memory_root / PLAN_A, role=Role.EXPLORER)
        with pytest.raises(HarnessOwnershipError):
            explorer.write_text(ArtifactNames.PLAN, "# Plan\n")
        assert _tmp_files(memory_root / PLAN_A) == []

    def test_a_path_escaping_the_memory_root_is_refused(self, orchestrator):
        with pytest.raises(HarnessConfinementError):
            orchestrator.write_text("../../escape.md", "x")

    def test_a_non_artifact_path_is_refused(self, orchestrator):
        with pytest.raises(HarnessOwnershipError):
            orchestrator.write_text("scratch/notes.txt", "x")

    def test_lessons_is_refused_to_a_non_archivist(self, orchestrator):
        with pytest.raises(HarnessOwnershipError):
            orchestrator.write_text(ArtifactNames.LESSONS, LESSONS_MD)

    def test_lessons_is_granted_to_the_archivist(self, archivist, memory_root: Path):
        archivist.write_text(ArtifactNames.LESSONS, LESSONS_MD)
        assert (memory_root / ArtifactNames.LESSONS).read_text() == LESSONS_MD

    def test_enforce_lessons_cap_is_refused_to_a_non_archivist(
        self, orchestrator, memory_root: Path
    ):
        _seed(orchestrator, ArtifactNames.LESSONS, LESSONS_MD)
        with pytest.raises(HarnessOwnershipError):
            orchestrator.enforce_lessons_cap(cap=1)


# ---------------------------------------------------------------------------
# PlanDirectory -- typed reads and writes
# ---------------------------------------------------------------------------


class TestPlanDirectoryArtifacts:
    def test_read_artifact_picks_the_model_from_the_path(self, archivist):
        _seed(archivist, ArtifactNames.LESSONS, LESSONS_MD)
        assert isinstance(archivist.read_artifact(ArtifactNames.LESSONS), LessonsDoc)

    def test_read_artifact_refuses_a_non_artifact_path(self, orchestrator):
        _seed(orchestrator, ArtifactNames.STATE, STATE_MD)
        with pytest.raises(HarnessArtifactError):
            orchestrator.read_artifact("scratch/notes.txt")

    def test_read_text_reports_a_missing_file_as_an_artifact_error(self, orchestrator):
        with pytest.raises(HarnessArtifactError):
            orchestrator.read_text(ArtifactNames.STATE)

    def test_write_artifact_round_trips(self, orchestrator):
        doc = StateDoc.from_markdown(STATE_MD)
        orchestrator.write_artifact(ArtifactNames.STATE, doc)
        assert orchestrator.read_artifact(ArtifactNames.STATE) == doc

    def test_write_artifact_refuses_the_wrong_model(self, orchestrator):
        with pytest.raises(HarnessArtifactError) as excinfo:
            orchestrator.write_artifact(
                ArtifactNames.STATE, LessonsDoc.from_markdown(LESSONS_MD)
            )
        assert "StateDoc" in str(excinfo.value)

    def test_write_artifact_refuses_an_over_cap_system_atlas(self, archivist):
        with pytest.raises(HarnessArtifactError) as excinfo:
            archivist.write_artifact(
                ArtifactNames.SYSTEM, _system(extra_components=400)
            )
        assert "rewritten, not truncated" in str(excinfo.value)

    def test_write_artifact_accepts_an_under_cap_system_atlas(self, archivist):
        archivist.write_artifact(ArtifactNames.SYSTEM, _system())
        assert archivist.exists(ArtifactNames.SYSTEM)

    def test_append_text_creates_then_extends(self, orchestrator):
        orchestrator.append_text(ArtifactNames.CHANGELOG, "line one\n")
        orchestrator.append_text(ArtifactNames.CHANGELOG, "line two\n")
        assert orchestrator.read_text(ArtifactNames.CHANGELOG) == "line one\nline two\n"

    def test_list_dir_sees_what_was_written(self, memory_root: Path):
        explorer = PlanDirectory(memory_root / PLAN_A, role=Role.EXPLORER)
        explorer.write_text("findings/alpha.md", "# Alpha\n")
        explorer.write_text("findings/beta.md", "# Beta\n")
        assert sorted(explorer.list_dir("findings")) == ["alpha.md", "beta.md"]


# ---------------------------------------------------------------------------
# Resumable run state
# ---------------------------------------------------------------------------


class TestRunState:
    def test_round_trips_through_state_md(self, orchestrator):
        original = RunState(plan_id=PLAN_A, doc=StateDoc.from_markdown(STATE_MD))
        orchestrator.save_run_state(original)
        assert orchestrator.load_run_state() == original

    def test_survives_a_fresh_plan_directory_object(self, memory_root: Path):
        """Resumability: a new process, a new object, the same position."""
        first = PlanDirectory(memory_root / PLAN_A)
        first.save_run_state(
            RunState(plan_id=PLAN_A, doc=StateDoc.from_markdown(STATE_MD))
        )
        resumed = PlanDirectory(memory_root / PLAN_A).load_run_state()
        assert resumed is not None
        assert resumed.state == "execute"
        assert resumed.iteration == 1
        assert resumed.current_step == "8 (storage.py)"
        assert resumed.fix_attempts == 2

    def test_absent_state_reads_as_none_not_as_a_default(self, orchestrator):
        assert orchestrator.load_run_state() is None

    def test_saving_another_plans_state_is_refused(self, orchestrator):
        foreign = RunState(plan_id=PLAN_B, doc=StateDoc.from_markdown(STATE_MD))
        with pytest.raises(HarnessArtifactError) as excinfo:
            orchestrator.save_run_state(foreign)
        assert PLAN_B in str(excinfo.value)

    def test_a_non_orchestrator_may_not_save_run_state(self, memory_root: Path):
        executor = PlanDirectory(memory_root / PLAN_A, role=Role.EXECUTOR)
        with pytest.raises(HarnessOwnershipError):
            executor.save_run_state(
                RunState(plan_id=PLAN_A, doc=StateDoc.from_markdown(STATE_MD))
            )

    def test_a_save_never_leaves_a_partial_state_file(
        self, orchestrator, memory_root: Path, monkeypatch: pytest.MonkeyPatch
    ):
        good = RunState(plan_id=PLAN_A, doc=StateDoc.from_markdown(STATE_MD))
        orchestrator.save_run_state(good)
        monkeypatch.setattr(
            os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom"))
        )
        broken = good.model_copy(deep=True)
        broken.doc.iteration = 99
        with pytest.raises(HarnessArtifactError):
            orchestrator.save_run_state(broken)
        monkeypatch.undo()
        assert orchestrator.load_run_state() == good


# ---------------------------------------------------------------------------
# PlanDirectory -- size policies end to end
# ---------------------------------------------------------------------------


class TestPlanDirectoryPolicies:
    def test_lessons_under_cap_is_not_rewritten(self, archivist, memory_root: Path):
        archivist.write_text(ArtifactNames.LESSONS, LESSONS_MD)
        report = archivist.enforce_lessons_cap()
        assert report.changed is False
        assert (memory_root / ArtifactNames.LESSONS).read_text() == LESSONS_MD
        assert not (memory_root / ArtifactNames.LESSONS_ARCHIVE).exists()

    def test_lessons_over_cap_is_trimmed_on_disk(self, archivist):
        archivist.write_text(ArtifactNames.LESSONS, LESSONS_MD)
        report = archivist.enforce_lessons_cap(cap=14)
        assert report.changed is True
        surviving = archivist.read_artifact(ArtifactNames.LESSONS)
        assert isinstance(surviving, LessonsDoc)
        assert surviving.line_count <= 14 or surviving.over_cap

    def test_evicted_lessons_are_appended_to_the_archive(self, archivist, memory_root):
        archivist.write_text(ArtifactNames.LESSONS, LESSONS_MD)
        report = archivist.enforce_lessons_cap(cap=14)
        archive = (memory_root / ArtifactNames.LESSONS_ARCHIVE).read_text()
        for evicted in report.evicted:
            assert evicted.split("\n")[0] in archive

    def test_system_cap_is_measured_from_disk(self, archivist):
        archivist.write_text(ArtifactNames.SYSTEM, SYSTEM_MD)
        report = archivist.enforce_system_cap()
        assert isinstance(report, CapReport)
        assert report.over_cap is False

    def test_sliding_window_rewrites_the_consolidated_file(
        self, archivist, memory_root
    ):
        archivist.write_text(ArtifactNames.CROSS_DECISIONS, DECISIONS_MD)
        report = archivist.apply_sliding_window(ArtifactNames.CROSS_DECISIONS)
        assert report.trimmed_plans == (PLAN_E, PLAN_F)
        rewritten = (memory_root / ArtifactNames.CROSS_DECISIONS).read_text()
        assert "oldest background" not in rewritten
        assert compression_marker_issues(rewritten) == []

    def test_sliding_window_leaves_an_in_window_file_alone(
        self, archivist, memory_root
    ):
        archivist.write_text(ArtifactNames.CROSS_FINDINGS, FINDINGS_MD)
        report = archivist.apply_sliding_window(ArtifactNames.CROSS_FINDINGS, keep=10)
        assert report.changed is False
        assert (memory_root / ArtifactNames.CROSS_FINDINGS).read_text() == FINDINGS_MD

    def test_sliding_window_refuses_an_unwindowed_artifact(self, archivist):
        with pytest.raises(HarnessArtifactError):
            archivist.apply_sliding_window(ArtifactNames.LESSONS)

    def test_sliding_window_is_refused_to_a_non_archivist(
        self, orchestrator, memory_root: Path
    ):
        _seed(orchestrator, ArtifactNames.CROSS_DECISIONS, DECISIONS_MD)
        with pytest.raises(HarnessOwnershipError):
            orchestrator.apply_sliding_window(ArtifactNames.CROSS_DECISIONS)
