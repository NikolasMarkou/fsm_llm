"""Tests for ``fsm_llm_harness.artifacts``.

The fixtures below are deliberately NOT minimal.  Every one is shaped from a
real artifact in this repository's own ``plans/`` tree and keeps the awkward
details that a synthetic stub would smooth away -- because those details are
exactly where a parser breaks (``plans/LESSONS.md`` [I:5], fixture audit):

* ``DECISIONS_MD`` keeps the leading HTML schema-example comment, whose body
  contains a literal ``## D-001 | ... | YYYY-MM-DD`` header.  A parser that
  does not mask comments reads it as a real entry with an unparseable date.
* ``DECISIONS_MD`` keeps a multi-line ``**Root Cause Analysis**:`` block and a
  bold-but-not-a-field line (``**Outcome ...** Budgets raised``), the two
  shapes that break a naive ``**Name**:`` field scanner.
* ``CHANGELOG_MD`` keeps the header line that QUOTES the pipe-delimited format.
  It contains eight ``|``-separated tokens and would be read as a malformed
  ledger line by an entry-first parser.
* ``PLAN_MD`` keeps a ``4b.`` step (a non-numeric step label) and puts the
  ``[RISK:]``/``[deps:]`` annotations on continuation lines, where they
  actually live.
* ``VERIFICATION_MD`` carries both acceptable and rejectable Evidence cells, so
  the evidence rule is exercised in both directions rather than only passing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fsm_llm_harness.artifacts import (
    ARTIFACT_MODELS,
    DECISION_ENTRY_SCHEMAS,
    MANDATORY_ADDITIONAL_CHECKS,
    PRESENTATION_CONTRACTS,
    VERDICT_BULLETS,
    VERDICT_RECOMMENDATIONS,
    Artifact,
    ChangelogDoc,
    ChangelogEntry,
    ChecklistItem,
    CheckpointDoc,
    ConsolidatedDoc,
    DecisionEntry,
    DecisionsDoc,
    FindingsIndexDoc,
    FindingsTopicDoc,
    IndexDoc,
    IndexRow,
    LessonsDoc,
    PlanDoc,
    ProgressDoc,
    Section,
    StateDoc,
    SummaryDoc,
    SystemAtlasDoc,
    VerificationDoc,
    compression_marker_issues,
    evidence_is_acceptable,
    lesson_importance,
    missing_entry_fields,
    missing_floor_fields,
    parse_changelog_line,
    parse_markdown_table,
)
from fsm_llm_harness.constants import ArtifactNames, Defaults, PlanSchema
from fsm_llm_harness.exceptions import HarnessArtifactError

# ---------------------------------------------------------------------------
# Fixtures -- shaped from this repository's real plan directory
# ---------------------------------------------------------------------------

STATE_MD = """# Current State: EXECUTE
*Skill: iterative-planner v2.56.0*
## Iteration: 1
## Current Plan Step: 7 (harness tier, parallel) — role layer held for RCA
## Pre-Step Checklist (reset before each EXECUTE step)
- [x] Re-read state.md (this file)
- [x] Re-read plan.md
- [ ] Re-read decisions.md (if fix attempt)
## Fix Attempts (resets per plan step)
- Step 4b, attempt 1: raised turn budgets — bytes on disk still 0/5
- Step 4b, attempt 2: added a writable path shape — bytes on disk still 0/5
## Change Manifest (current iteration)
- step 1 (`f63104f`): `src/fsm_llm_agents/native_fc.py` (+59/-10, net +49 source),
  `tests/test_fsm_llm_agents/test_native_fc.py` (+228/-1, 11 → 19 tests).
- step 3 (`9101369`): `src/fsm_llm_harness/tools.py` (CODE +24)
## Last Transition: REFLECT → EXECUTE (2026-07-22T00:45:00Z)
## Transition History:
- INIT → EXPLORE (task started)
- EXPLORE → PLAN (gathered enough context, 2026-07-21T19:52:00Z)
  - confidence: scope=deep, solutions=constrained, risks=clear
- PLAN → EXECUTE (user approved)
"""

_PLAN_SECTION_BODY = {
    "Goal": "Make `src/fsm_llm_harness` run the protocol on `:4b`.",
    "Problem Statement": "The two agent arms have COMPLEMENTARY failures.",
    "Context": "Read `findings.md` and the four detail files.",
    "Files To Modify": (
        "| File | Change | Reason |\n"
        "|---|---|---|\n"
        "| `src/fsm_llm_harness/artifacts.py` | new | artifact models |"
    ),
    "Assumptions": "- **A1.** `tools=` and `response_format=` are not stacked.",
    "Failure Modes": (
        "| Dependency | Slow | Bad data | Down |\n"
        "|---|---|---|---|\n"
        "| Ollama | 1-18s/call | empty content | steps 2, 5 unverifiable |"
    ),
    "Pre-Mortem & Falsification Signals": (
        "1. **The repair turn will not work.** → **STOP IF** <4/5 runs validate."
    ),
    "Success Criteria": "1. A role dispatch that calls a tool AND writes a file.",
    "Verification Strategy": (
        "| # | Criterion | Method |\n|---|---|---|\n| 1 | tool + file | live |"
    ),
    "Complexity Budget": "| Metric | Budget |\n|---|---|\n| Files added | 4/4 |",
}

PLAN_STEPS_BODY = """*Marker: steps 1-3 complete. Next: step 4.*

1. [x] **`native_fc` Ollama-helper repair.** Apply `apply_ollama_params` gated
   entirely by `is_ollama_model(self.config.model)`.
   [RISK: medium — it edits a shipped agent] [deps: none]

4b. [ ] **Role iteration budget + stopping condition (inserted at EXECUTE).**
   Right-size the per-role iteration budget.
   [RISK: medium — prompt+budget shape] [deps: 1, 2, 4]

9. [ ] **Drop the legacy dispatch table.** [IRREVERSIBLE]
   [RISK: high — no rollback once the table is gone] [deps: 8]
"""


def _plan_md(*, omit: str | None = None, swap: tuple[str, str] | None = None) -> str:
    names = list(PlanSchema.SECTIONS)
    if omit is not None:
        names.remove(omit)
    if swap is not None:
        first, second = (names.index(name) for name in swap)
        names[first], names[second] = names[second], names[first]
    blocks = ["# Plan v1: Make the harness actually run on `:4b`"]
    for name in names:
        body = PLAN_STEPS_BODY if name == "Steps" else _PLAN_SECTION_BODY[name]
        blocks.append(f"## {name}\n{body.strip()}")
    return "\n\n".join(blocks) + "\n"


PLAN_MD = _plan_md()

DECISIONS_MD = """# Decision Log
*Plan: plan-2026-07-21T191807-bf7ffe24*
*Skill: iterative-planner v2.56.0*

<!-- Schema example — DO NOT REMOVE. Real entries follow this shape.

## D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background>
**Trade-off**: <X> **at the cost of** <Y>
-->

## D-001 | EXPLORE → PLAN | 2026-07-21
**Title**: `native_fc` is the role arm, not stock ReAct
**Context**: The two agent arms have complementary failures.
**Decision**: Back harness roles with `NativeFunctionCallingReactAgent`.
**Trade-off**: Reliable, measured tool selection **at the cost of** losing the
free constrained decoding ReAct gets from `output_schema`.
**Reasoning**: A role that never calls a tool cannot write an artifact.

## D-002 | REFLECT → PIVOT | 2026-07-22
**Context**: Two attempts moved `concluded` 0/5 → 2/5, bytes stayed 0/5.
**What Failed**: The model reads repeatedly and never selects a write tool.
**What Was Learned**: Prompt wording cannot carry this obligation.
**Root Cause Analysis**:
1. **Immediate cause**: `:4b` does not select a write tool in this shape.
2. **Contributing factor**: the dispatch declares 9 tools.
3. **Failed defense**: A3's falsifier was aimed two steps too late.
4. **Prevention**: aim a falsifier at the FIRST step that could break it.
**Complexity Assessment**:
- Lines added in failed attempt: 77
- New abstractions added: 0
**Decision**: Add a mechanical cross-check at the worker-factory layer.
**Trade-off**: A gate that cannot be opened by a confident sentence **at the
cost of** more dispatches reporting failure in the short term.
**Reasoning**: The evidence that wording cannot carry this is repeated.
**Outcome (step 4b, iter 1) — TARGET NOT MET.** Budgets raised 8 → 14; the
prompt gained a HOW TO FINISH section.
**Anchor-Refs**: `src/fsm_llm_harness/roles.py:268`
"""

FINDINGS_MD = """# Findings
*Summary and index of all findings. Detailed files go in findings/ directory.*

## Index

1. `native_fc.py` live repair — `findings/native-fc-live-repair.md`
2. ReAct tool selection on `:4b` — `findings/react-tool-selection-live.md`
3. `:4b` decoding capability — `findings/qwen4b-decoding-capability.md`

| # | Topic | File |
|---|-------|------|
| 1 | live repair | `findings/native-fc-live-repair.md` |

## Key Constraints

**HARD**
- Structured output and Ollama thinking mode cannot coexist on `:4b`. [finding 3]

**GHOST**
- "`:4b` under-calls tools" — **FALSIFIED**, see Corrections. [finding 3]

## Corrections
- **[CORRECTED iter-0]** `plans/FINDINGS.md` SOFT constraint is falsified.
"""

FINDINGS_TOPIC_MD = """# Finding: remaining harness tier

## Summary
The remaining tier is independent of the role layer at two named seams.

## Key Findings
- `harness.py:1416-1443` `_pre_step_gate` is the swap point.
- `rules.py:109-133` `OWNERSHIP` is read-only from both readers.

## Constraints
| Constraint | Class | Source |
|---|---|---|
| `MANIFEST.in` omits `fsm_llm_monitor` | HARD | `MANIFEST.in:1` |
| review-iter-1's C2 is closed | GHOST | `tools.py:668` |

## Code Patterns
- `[REUSE]` `FileSessionStore.save` mkstemp+`os.replace` (`session.py:151-173`).

## Risks & Unknowns
- Whether a harness plan dir passes the EXTERNAL `validate-plan.mjs`.

## Atlas Contradictions
- `plans/SYSTEM.md` "Known Patterns" describes `tool_name -> "none"` as fixed.
"""

PROGRESS_MD = """# Progress

## Completed
- [x] EXPLORE: four parallel investigations, all live-measured (EXPLORE, iter 0)
- [x] **Step 1** — `native_fc` Ollama-helper repair (`f63104f`).
  Tests 11 → 19 (8 new, 6 confirmed RED first). Source delta **+49**.

## In Progress
- [ ] Step 4b's measured target (bytes on disk > 0/5) is UNMET after 2 attempts.

## Remaining
- [ ] 5. **LIVE FALSIFICATION SPIKE**, n≥5/arm [RISK: high]
- [ ] 7. `artifacts.py` — 11 artifact kinds + 6 Presentation Contracts

## Blocked
*Nothing currently.*
"""

VERIFICATION_MD = """# Verification Results (Iteration 1)
*Rewritten each iteration — not append-only.*

## Criteria Verification
| # | Criterion (from plan.md) | Method | Command/Action | Result | Evidence |
|---|--------------------------|--------|----------------|--------|----------|
| 1 | Full suite holds baseline | Automated | `pytest -q` | PASS | 3745/3745 passed, 0 failures |
| 2 | Package builds | Automated | `python -m build` | PASS | exit 0; "Successfully built fsm_llm" |
| 3 | Prompt renders the path shape | Manual | read the rendered prompt | PASS | manual review — observed `findings/<topic>.md` in the WRITES block |
| 4 | Leash halts at exactly 2 | Automated | `pytest -k leash` | FAIL | looks good |

## Additional Checks
| Check | Command/Action | Result | Details |
|-------|----------------|--------|---------|
| Regression | `pytest -q` (full re-run) | PASS | 3745 passed |
| Scope drift | manifest vs Files To Modify | CLEAN | 2 files, both planned |
| Diff review | `git diff` for debug code | CLEAN | none |
| Lint | `make lint` | PASS | 0 offenses |

## Not Verified
| What | Why |
|------|-----|
| Stacked `tools=` + `response_format=` | Deliberately never attempted (A1, D-002) |

## Concerns
- Live criteria rest on small n against a shared, contended Ollama instance.

## Verdict
- Criteria passed: 3/4
- Regressions: none
- Scope drift: none
- Simplification blockers: none
- Recommendation: → EXECUTE
"""

CHANGELOG_MD = """# Changelog
*Append-only per-edit ledger. One line per file edit. Owner: ip-executor.*
*Format: `UTC | iter-N/step-M | commit | path | OP(+N,-M) | radius:TIER(score) | D-NNN-or-dash | reason`*
*Step 4b is recorded as `iter-1/step-18`: the ledger schema requires numeric M.*
2026-07-21T20:10:30Z | iter-1/step-1 | f63104f | src/fsm_llm_agents/native_fc.py | EDIT(+59,-10) | radius:LOW(0) | D-003 | apply ollama call prep behind is_ollama_model
2026-07-22T00:05:00Z | iter-1/step-3 | 9101369 | src/fsm_llm_harness/tools.py | EDIT(+77,-2) | radius:MED(3) | D-006 | repair sentinel-prefixed absolute paths | keeps D-032 ordering
2026-07-22T01:20:00Z | iter-1/step-18 | 751fda0 | src/fsm_llm_harness/roles.py | CREATE(+77) | radius:HIGH(6) | - | raise per-role turn budgets
2026-07-22T01:25:00Z | iter-1/step-18 | uncommitted | docs/api_reference.md | DELETE(-12) | radius:UNKNOWN(script-missing) | - | drop the stale section
"""

CHECKPOINT_MD = """# Checkpoint cp-000-iter1

## Created
2026-07-21T20:10:00Z — EXECUTE, iteration 1, before step 1.

## Reason
Nuclear fallback / full-revert restore point for the whole iteration.

## Git State
- Commit: `14e27a18d9c21ee68127670bb74312dcd3763dad` (`14e27a1`)
- Working tree: clean

## Lockfiles snapshotted:
- none (no package manager touched)

## Rollback:
```bash
git -C /repo reset --hard 14e27a1
```
`plans/` is gitignored and is NEVER reverted (D-009).
"""

SUMMARY_MD = """# Summary
*Plan: plan-2026-07-21T125237-191b2eb2*

## Outcome
Partially complete. Closed at the user's explicit direction.

## Iterations
- v1 (iter 1): partially succeeded, then intentionally halted.

## Key Decisions
65 decisions recorded (D-001..D-065) across PLAN and EXECUTE/REFLECT.

## Files Changed
`src/fsm_llm_harness/` (11 files) + `tests/test_fsm_llm_harness/` (5 files).

## Decision Anchors Registry
- `src/fsm_llm_harness/tools.py:428` — `plan-2026-07-21T125237-191b2eb2/D-032`

## Lessons
See `plans/LESSONS.md` (rewritten at this CLOSE) for the importance-tagged set.
"""

LESSONS_MD = """# Lessons Learned
*Cross-plan lessons. Max 200 lines — rewrite, don't append forever.*

## Recurring Patterns
- A remediation that fixes one code path must sweep ALL sibling call sites. [I:5]
- Deep-review-then-remediate plans should split into risk-ordered iterations. [I:4]
- Untagged bullets are treated as the default importance.

## Failed Approaches (+ why)
- Adapters/wrappers as fixes — they accumulate and obscure the problem. [I:5]
- Dual-write strategies with long TTLs — storage grows unbounded. [I:3]

## Successful Strategies
- Run EXPLORE even when "I already know this". [I:4]

## Codebase Gotchas
- `retries=` is a documented no-op for `ollama_chat/*`. [I:5]
- `plans/` is gitignored, so protocol memory has no VCS backstop. [I:2]
"""

SYSTEM_MD = """# System Atlas
*Last refreshed: plan-2026-07-21T125237-191b2eb2 | 2026-07-21*
*Domain-neutral system map. Rewritten at CLOSE — max 300 lines.*

## Identity
FSM-LLM (v0.5.0): a Python framework for stateful conversational AI.

## Components
- `fsm_llm` (core, 23 files) — FSM orchestration and the 2-pass pipeline.
- `fsm_llm_harness` (11 files, PARTIAL) — protocol emulation.

## Boundaries
- In scope: `src/`, `tests/`. Out of scope: `examples/`.

## Invariants
- Core never imports `fsm_llm_agents` (see plan-2026-07-21T125237-191b2eb2/D-002).

## Flows
- turn: extract → evaluate transitions → generate response.

## Known Patterns
- FSM-driven agent loop with JsonLogic hard gates.

## Codebase Specialization
- Module map: `src/fsm_llm*/`; tests mirror it under `tests/`.
"""

INDEX_MD = """# Plan Index
*Topic-to-directory mapping. Updated on close. Survives sliding window trim.*

| Plan | Date | Goal | Key Topics |
|------|------|------|------------|
| plan-2026-07-21T125237-191b2eb2 | 2026-07-21 | Build `src/fsm_llm_harness` | harness, protocol |
| plan-2026-07-21T110044-ed1ae68b | 2026-07-21 | Prototype the planner in agents |  |
"""

CONSOLIDATED_MD = """# Consolidated Decisions
*Cross-plan decision archive. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed from 2600 lines (4 plan sections).*

### Key Outcomes
- **plan-2026-07-21T125237-191b2eb2** — PARTIALLY COMPLETE, closed at direction.
<!-- /COMPRESSED-SUMMARY -->

## plan-2026-07-21T125237-191b2eb2
### D-001 | EXPLORE → PLAN | 2026-07-21
**Trade-off**: Fastest path **at the cost of** ignoring the other stores

## plan-2026-07-21T110044-ed1ae68b
### D-001 | EXPLORE → PLAN | 2026-07-21
**Trade-off**: Safe rollback **at the cost of** doubled storage
"""

#: Every artifact kind, with a realistic document for it.
ARTIFACT_CASES: tuple[tuple[type[Artifact], str], ...] = (
    (StateDoc, STATE_MD),
    (PlanDoc, PLAN_MD),
    (DecisionsDoc, DECISIONS_MD),
    (FindingsIndexDoc, FINDINGS_MD),
    (FindingsTopicDoc, FINDINGS_TOPIC_MD),
    (ProgressDoc, PROGRESS_MD),
    (VerificationDoc, VERIFICATION_MD),
    (ChangelogDoc, CHANGELOG_MD),
    (CheckpointDoc, CHECKPOINT_MD),
    (SummaryDoc, SUMMARY_MD),
    (LessonsDoc, LESSONS_MD),
    (SystemAtlasDoc, SYSTEM_MD),
    (IndexDoc, INDEX_MD),
    (ConsolidatedDoc, CONSOLIDATED_MD),
)

_CASE_IDS = [model.__name__ for model, _ in ARTIFACT_CASES]


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """``from_markdown(to_markdown(model)) == model`` -- the core property."""

    @pytest.mark.parametrize(("model", "text"), ARTIFACT_CASES, ids=_CASE_IDS)
    def test_round_trip_is_lossless(self, model: type[Artifact], text: str) -> None:
        parsed = model.from_markdown(text)
        assert model.from_markdown(parsed.to_markdown()) == parsed

    @pytest.mark.parametrize(("model", "text"), ARTIFACT_CASES, ids=_CASE_IDS)
    def test_round_trip_is_idempotent(self, model: type[Artifact], text: str) -> None:
        once = model.from_markdown(text).to_markdown()
        assert model.from_markdown(once).to_markdown() == once

    @pytest.mark.parametrize(("model", "text"), ARTIFACT_CASES, ids=_CASE_IDS)
    def test_every_artifact_ends_with_a_newline(
        self, model: type[Artifact], text: str
    ) -> None:
        assert model.from_markdown(text).to_markdown().endswith("\n")

    def test_round_trip_of_a_hand_built_model(self) -> None:
        """A model built in code, never read from text, still round-trips."""
        doc = StateDoc(
            state="reflect",
            skill_version="iterative-planner v2.56.0",
            iteration=3,
            current_step="2 of 5",
            checklist=[ChecklistItem(checked=True, text="Re-read state.md")],
            fix_attempts=["Step 2, attempt 1: reverted middleware — still fails"],
            change_manifest=["`src/a.py` — MODIFIED (step 2, uncommitted)"],
            last_transition="EXECUTE → REFLECT (tests failing)",
            transition_history=["EXPLORE → PLAN (enough context)"],
        )
        assert StateDoc.from_markdown(doc.to_markdown()) == doc

    def test_empty_fix_attempts_round_trip_through_the_placeholder(self) -> None:
        doc = StateDoc(state="execute", iteration=1, current_step="1 of 3")
        rendered = doc.to_markdown()
        assert "(none yet for current step)" in rendered
        assert StateDoc.from_markdown(rendered).fix_attempts == []


# ---------------------------------------------------------------------------
# state.md
# ---------------------------------------------------------------------------


class TestStateDoc:
    def test_fields_are_parsed(self) -> None:
        doc = StateDoc.from_markdown(STATE_MD)
        assert doc.state == "execute"
        assert doc.iteration == 1
        assert doc.skill_version == "iterative-planner v2.56.0"
        assert doc.current_step.startswith("7 (harness tier")
        assert [item.checked for item in doc.checklist] == [True, True, False]
        assert len(doc.transition_history) == 3

    def test_a_multiline_manifest_entry_stays_one_entry(self) -> None:
        doc = StateDoc.from_markdown(STATE_MD)
        assert len(doc.change_manifest) == 2
        assert "test_native_fc.py" in doc.change_manifest[0]

    def test_fix_attempt_count_uses_the_protocol_line_grammar(self) -> None:
        assert StateDoc.from_markdown(STATE_MD).fix_attempt_count == 2
        assert Defaults.MAX_FIX_ATTEMPTS == 2

    @pytest.mark.parametrize(
        "line",
        [
            "Step 2, attempt 1: reverted the middleware change",
            "Step 2 attempt 1: reverted the middleware change",
            "Step 4b, attempt 2: added a writable path shape",
            "Attempt 1: tried the obvious thing",
            "Attempts 2: plural spelling",
        ],
    )
    def test_every_attempt_line_shape_is_counted(self, line: str) -> None:
        """Over-counting only halts sooner; under-counting would pass the leash."""
        doc = StateDoc(
            state="execute", iteration=1, current_step="1 of 1", fix_attempts=[line]
        )
        assert doc.fix_attempt_count == 1

    def test_a_narrative_fix_attempt_line_is_not_counted(self) -> None:
        doc = StateDoc(
            state="execute",
            iteration=1,
            current_step="1 of 1",
            fix_attempts=["Step 2: LEASH HIT. Transitioned to REFLECT."],
        )
        assert doc.fix_attempts and doc.fix_attempt_count == 0

    def test_an_unknown_state_is_rejected(self) -> None:
        with pytest.raises(HarnessArtifactError):
            StateDoc.from_markdown(STATE_MD.replace("EXECUTE\n", "DAYDREAM\n", 1))

    def test_a_non_numeric_iteration_is_rejected(self) -> None:
        with pytest.raises(HarnessArtifactError):
            StateDoc.from_markdown(
                STATE_MD.replace("## Iteration: 1", "## Iteration: n")
            )

    def test_a_missing_iteration_line_is_rejected(self) -> None:
        with pytest.raises(HarnessArtifactError):
            StateDoc.from_markdown(STATE_MD.replace("## Iteration: 1\n", ""))

    def test_a_wrong_h1_is_rejected(self) -> None:
        with pytest.raises(HarnessArtifactError):
            StateDoc.from_markdown(STATE_MD.replace("# Current State:", "# State:", 1))


# ---------------------------------------------------------------------------
# plan.md
# ---------------------------------------------------------------------------


class TestPlanDoc:
    def test_the_required_sections_are_the_constants_table(self) -> None:
        """No second copy of the section list; ``PlanSchema`` is the one fact."""
        assert PlanDoc.REQUIRED_SECTIONS == PlanSchema.SECTIONS
        assert len(PlanSchema.SECTIONS) == 11

    def test_all_eleven_sections_parse_in_order(self) -> None:
        doc = PlanDoc.from_markdown(PLAN_MD)
        assert [section.name for section in doc.sections] == list(PlanSchema.SECTIONS)
        assert doc.section_issues() == []

    @pytest.mark.parametrize("omitted", PlanSchema.SECTIONS)
    def test_a_plan_missing_any_section_is_rejected(self, omitted: str) -> None:
        with pytest.raises(HarnessArtifactError, match="missing required section"):
            PlanDoc.from_markdown(_plan_md(omit=omitted))

    def test_a_plan_with_sections_out_of_order_is_rejected(self) -> None:
        with pytest.raises(HarnessArtifactError, match="out of order"):
            PlanDoc.from_markdown(_plan_md(swap=("Goal", "Complexity Budget")))

    def test_a_duplicated_section_is_rejected(self) -> None:
        with pytest.raises(HarnessArtifactError, match="duplicate section"):
            PlanDoc.from_markdown(PLAN_MD + "\n## Goal\nsecond goal\n")

    def test_steps_are_parsed_with_their_annotations(self) -> None:
        steps = PlanDoc.from_markdown(PLAN_MD).steps()
        assert [step.number for step in steps] == ["1", "4b", "9"]
        assert [step.done for step in steps] == [True, False, False]
        assert steps[0].risk == "medium — it edits a shipped agent"
        assert steps[0].deps == "none"
        assert steps[1].deps == "1, 2, 4"

    def test_a_non_numeric_step_label_is_preserved(self) -> None:
        """`4b` is a legal PLAN label even though the changelog needs a number."""
        assert PlanDoc.from_markdown(PLAN_MD).steps()[1].number == "4b"

    def test_the_irreversible_tag_is_detected(self) -> None:
        steps = PlanDoc.from_markdown(PLAN_MD).steps()
        assert [step.irreversible for step in steps] == [False, False, True]


# ---------------------------------------------------------------------------
# decisions.md
# ---------------------------------------------------------------------------


def _decisions(body: str, plan_id: str = "plan-2026-07-21T191807-bf7ffe24") -> str:
    return f"# Decision Log\n*Plan: {plan_id}*\n\n{body}"


_GOOD_ENTRY = (
    "## D-001 | EXPLORE → PLAN | 2026-07-21\n"
    "**Context**: something was discovered\n"
    "**Decision**: do the thing\n"
    "**Trade-off**: speed **at the cost of** generality\n"
    "**Reasoning**: because\n"
)


class TestDecisionsDoc:
    def test_entries_and_preamble_parse(self) -> None:
        doc = DecisionsDoc.from_markdown(DECISIONS_MD)
        assert doc.plan_id == "plan-2026-07-21T191807-bf7ffe24"
        assert [entry.id for entry in doc.entries] == ["D-001", "D-002"]
        assert doc.entries[0].phase == "EXPLORE → PLAN"
        assert doc.entries[1].date == "2026-07-22"

    def test_the_schema_example_in_an_html_comment_is_not_an_entry(self) -> None:
        """The commented ``## D-001 | ... | YYYY-MM-DD`` must not be parsed."""
        doc = DecisionsDoc.from_markdown(DECISIONS_MD)
        assert len(doc.entries) == 2
        assert "YYYY-MM-DD" in doc.preamble

    def test_a_multiline_field_keeps_its_whole_body(self) -> None:
        entry = DecisionsDoc.from_markdown(DECISIONS_MD).entry("D-002")
        assert entry is not None
        rca = entry.field("Root Cause Analysis")
        assert rca is not None
        assert "Immediate cause" in rca and "Prevention" in rca

    def test_a_bold_line_that_is_not_a_field_continues_the_previous_field(self) -> None:
        entry = DecisionsDoc.from_markdown(DECISIONS_MD).entry("D-002")
        assert entry is not None
        names = [name for name, _ in entry.fields]
        assert "Outcome (step 4b, iter 1) — TARGET NOT MET." not in names
        reasoning = entry.field("Reasoning")
        assert reasoning is not None and "TARGET NOT MET" in reasoning

    def test_anchor_refs_are_kept_as_an_ordinary_field(self) -> None:
        entry = DecisionsDoc.from_markdown(DECISIONS_MD).entry("D-002")
        assert entry is not None
        assert entry.field("Anchor-Refs") == "`src/fsm_llm_harness/roles.py:268`"

    # -- header grammar ---------------------------------------------------

    def test_a_trailing_title_on_the_header_line_is_rejected(self) -> None:
        """A documented real-world gotcha: the header is strictly positional."""
        broken = _GOOD_ENTRY.replace(
            "| 2026-07-21\n", "| 2026-07-21 | native_fc is the role arm\n"
        )
        with pytest.raises(HarnessArtifactError, match="entry header must be exactly"):
            DecisionsDoc.from_markdown(_decisions(broken))

    def test_trailing_prose_after_the_date_is_rejected(self) -> None:
        broken = _GOOD_ENTRY.replace("| 2026-07-21\n", "| 2026-07-21 the role arm\n")
        with pytest.raises(HarnessArtifactError, match="entry header must be exactly"):
            DecisionsDoc.from_markdown(_decisions(broken))

    @pytest.mark.parametrize(
        "header",
        [
            "## D-1 | EXPLORE → PLAN | 2026-07-21",
            "## D-0001 | EXPLORE → PLAN | 2026-07-21",
            "## d-001 | EXPLORE → PLAN | 2026-07-21",
            "## D-001 | EXPLORE → PLAN | 21-07-2026",
            "## D-001 | EXPLORE → PLAN",
            "## D-001 EXPLORE → PLAN 2026-07-21",
            "## D-001 |  | 2026-07-21",
        ],
    )
    def test_malformed_headers_are_rejected(self, header: str) -> None:
        broken = _GOOD_ENTRY.replace(
            "## D-001 | EXPLORE → PLAN | 2026-07-21", header, 1
        )
        with pytest.raises(HarnessArtifactError):
            DecisionsDoc.from_markdown(_decisions(broken))

    # -- required content -------------------------------------------------

    def test_a_missing_trade_off_line_is_rejected(self) -> None:
        broken = "\n".join(
            line for line in _GOOD_ENTRY.split("\n") if not line.startswith("**Trade")
        )
        with pytest.raises(HarnessArtifactError, match="Trade-off"):
            DecisionsDoc.from_markdown(_decisions(broken))

    def test_a_hard_wrapped_at_the_cost_of_still_satisfies_the_rule(self) -> None:
        """D-002's fixture wraps the phrase mid-line, exactly as real entries do."""
        entry = DecisionsDoc.from_markdown(DECISIONS_MD).entry("D-002")
        assert entry is not None
        trade_off = entry.field("Trade-off")
        assert trade_off is not None
        assert "**at the\ncost of**" in trade_off

    def test_a_trade_off_without_at_the_cost_of_is_rejected(self) -> None:
        broken = _GOOD_ENTRY.replace(
            "**Trade-off**: speed **at the cost of** generality",
            "**Trade-off**: speed, and it is worth it",
        )
        with pytest.raises(HarnessArtifactError, match="at the cost of"):
            DecisionsDoc.from_markdown(_decisions(broken))

    def test_the_missing_preamble_is_rejected(self) -> None:
        with pytest.raises(HarnessArtifactError, match="preamble"):
            DecisionsDoc.from_markdown(f"# Decision Log\n\n{_GOOD_ENTRY}")

    def test_a_gap_in_the_decision_sequence_is_rejected(self) -> None:
        second = _GOOD_ENTRY.replace("D-001", "D-003")
        with pytest.raises(HarnessArtifactError, match="no gaps"):
            DecisionsDoc.from_markdown(_decisions(f"{_GOOD_ENTRY}\n{second}"))

    def test_text_before_the_first_field_of_an_entry_is_rejected(self) -> None:
        broken = _GOOD_ENTRY.replace(
            "2026-07-21\n**Context**", "2026-07-21\nstray prose\n**Context**"
        )
        with pytest.raises(HarnessArtifactError, match="text before"):
            DecisionsDoc.from_markdown(_decisions(broken))

    # -- the 9 entry-type field sets --------------------------------------

    def test_all_nine_entry_types_are_declared(self) -> None:
        assert len(DECISION_ENTRY_SCHEMAS) == 9
        for required in DECISION_ENTRY_SCHEMAS.values():
            assert "Context" in required
            assert "Trade-off" in required

    def test_a_complete_pivot_entry_reports_no_missing_fields(self) -> None:
        entry = DecisionsDoc.from_markdown(DECISIONS_MD).entry("D-002")
        assert entry is not None
        assert missing_entry_fields(entry, "reflect-to-pivot") == ()

    def test_an_explore_entry_is_incomplete_as_a_pivot_entry(self) -> None:
        entry = DecisionsDoc.from_markdown(DECISIONS_MD).entry("D-001")
        assert entry is not None
        assert missing_entry_fields(entry, "explore-to-plan") == ()
        missing = missing_entry_fields(entry, "reflect-to-pivot")
        assert "Root Cause Analysis" in missing
        assert "Complexity Assessment" in missing

    def test_a_prefixed_field_name_satisfies_the_requirement(self) -> None:
        """``**3-STRIKE TRIGGERED on `roles.py`**:`` still counts."""
        entry = DecisionEntry(
            id="D-001",
            phase="EXECUTE",
            date="2026-07-22",
            fields=[
                ("Context", "the same seam broke three times"),
                ("3-STRIKE TRIGGERED on `roles.py`", "yes"),
                ("Three Attempts", "budget, prompt, path shape"),
                ("Decision", "stop and pivot"),
                ("Trade-off", "an honest stop **at the cost of** an unbuilt tier"),
            ],
        )
        assert missing_entry_fields(entry, "three-strike") == ()

    def test_an_unknown_entry_type_raises(self) -> None:
        entry = DecisionsDoc.from_markdown(DECISIONS_MD).entries[0]
        with pytest.raises(KeyError):
            missing_entry_fields(entry, "not-a-type")


# ---------------------------------------------------------------------------
# findings.md + findings/{topic}.md
# ---------------------------------------------------------------------------


class TestFindings:
    def test_the_index_is_counted_for_the_explore_gate(self) -> None:
        doc = FindingsIndexDoc.from_markdown(FINDINGS_MD)
        assert doc.findings_count == 3
        assert doc.findings_count >= Defaults.FINDINGS_THRESHOLD

    def test_the_index_table_rows_are_not_counted_as_findings(self) -> None:
        """Only list entries count; the summary table would double-count."""
        assert FindingsIndexDoc.from_markdown(FINDINGS_MD).findings_count == 3

    def test_the_topic_five_section_schema(self) -> None:
        doc = FindingsTopicDoc.from_markdown(FINDINGS_TOPIC_MD)
        assert FindingsTopicDoc.REQUIRED_SECTIONS == (
            "Summary",
            "Key Findings",
            "Constraints",
            "Code Patterns",
            "Risks & Unknowns",
        )
        assert doc.section_issues() == []

    def test_the_optional_sixth_section_is_allowed(self) -> None:
        doc = FindingsTopicDoc.from_markdown(FINDINGS_TOPIC_MD)
        assert doc.section(FindingsTopicDoc.OPTIONAL_SECTION) is not None
        assert doc.section_issues() == []

    def test_a_missing_topic_section_is_reported_not_raised(self) -> None:
        """Topic sections are a WARN in the source protocol, not an ERROR."""
        text = FINDINGS_TOPIC_MD.replace(
            "## Risks & Unknowns\n- Whether a harness plan dir passes the "
            "EXTERNAL `validate-plan.mjs`.\n",
            "",
        )
        doc = FindingsTopicDoc.from_markdown(text)
        assert doc.section_issues() == [
            "missing required section '## Risks & Unknowns'"
        ]

    def test_constraints_are_classified_hard_soft_ghost(self) -> None:
        body = FindingsTopicDoc.from_markdown(FINDINGS_TOPIC_MD).body_of("Constraints")
        classes = {row[1] for row in parse_markdown_table(body)}
        assert classes <= {"HARD", "SOFT", "GHOST"}
        assert classes == {"HARD", "GHOST"}


# ---------------------------------------------------------------------------
# progress.md
# ---------------------------------------------------------------------------


class TestProgressDoc:
    def test_the_four_sections(self) -> None:
        doc = ProgressDoc.from_markdown(PROGRESS_MD)
        assert ProgressDoc.REQUIRED_SECTIONS == (
            "Completed",
            "In Progress",
            "Remaining",
            "Blocked",
        )
        assert doc.section_issues() == []
        assert len(doc.items("Completed")) == 2
        assert len(doc.items("Remaining")) == 2

    def test_a_missing_section_is_reported(self) -> None:
        doc = ProgressDoc.from_markdown(
            PROGRESS_MD.replace("## Blocked\n", "## Done\n")
        )
        assert doc.section_issues() == ["missing required section '## Blocked'"]

    def test_sections_out_of_order_are_reported(self) -> None:
        reordered = ProgressDoc.from_markdown(PROGRESS_MD)
        reordered.sections = list(reversed(reordered.sections))
        assert any("out of order" in issue for issue in reordered.section_issues())


# ---------------------------------------------------------------------------
# verification.md
# ---------------------------------------------------------------------------


class TestVerificationDoc:
    def test_criteria_rows_parse(self) -> None:
        rows = VerificationDoc.from_markdown(VERIFICATION_MD).criteria()
        assert [row.number for row in rows] == ["1", "2", "3", "4"]
        assert [row.result for row in rows] == ["PASS", "PASS", "PASS", "FAIL"]

    def test_the_three_evidence_shapes_are_accepted_and_lgtm_is_not(self) -> None:
        rows = VerificationDoc.from_markdown(VERIFICATION_MD).criteria()
        assert [row.evidence_ok for row in rows] == [True, True, True, False]

    @pytest.mark.parametrize(
        "evidence",
        [
            "47/47 passed, 0 failures",
            "3/3 specs",
            'exit 0; "Build succeeded in 12.4s"',
            "exit code 2; GATE:FAIL [leash-cap]",
            "manual review — observed the token field in the response",
            "manual review - observed 9 declared tools",
        ],
    )
    def test_accepted_evidence(self, evidence: str) -> None:
        assert evidence_is_acceptable(evidence)

    @pytest.mark.parametrize(
        "evidence",
        ["looks good", "seems to work", "LGTM", "", "   ", "yes", "done", "-", "ok"],
    )
    def test_rejected_evidence(self, evidence: str) -> None:
        assert not evidence_is_acceptable(evidence)

    def test_a_prose_cell_with_no_measurement_is_rejected(self) -> None:
        assert not evidence_is_acceptable("the suite ran and nothing exploded")

    def test_the_three_mandatory_additional_checks_are_present(self) -> None:
        doc = VerificationDoc.from_markdown(VERIFICATION_MD)
        assert MANDATORY_ADDITIONAL_CHECKS == (
            "Regression",
            "Scope drift",
            "Diff review",
        )
        assert doc.missing_additional_checks() == ()

    def test_a_missing_mandatory_check_is_reported(self) -> None:
        doc = VerificationDoc.from_markdown(
            VERIFICATION_MD.replace("| Diff review |", "| Doc review |")
        )
        assert doc.missing_additional_checks() == ("Diff review",)

    def test_the_verdict_has_five_bullets_in_order(self) -> None:
        doc = VerificationDoc.from_markdown(VERIFICATION_MD)
        assert [label for label, _ in doc.verdict_bullets()] == list(VERDICT_BULLETS)
        assert doc.verdict_issues() == []

    def test_a_missing_verdict_bullet_is_reported(self) -> None:
        doc = VerificationDoc.from_markdown(
            VERIFICATION_MD.replace("- Scope drift: none\n", "")
        )
        assert "Verdict is missing the 'Scope drift' bullet" in doc.verdict_issues()

    def test_verdict_bullets_out_of_order_are_reported(self) -> None:
        swapped = VERIFICATION_MD.replace(
            "- Criteria passed: 3/4\n- Regressions: none\n",
            "- Regressions: none\n- Criteria passed: 3/4\n",
        )
        issues = VerificationDoc.from_markdown(swapped).verdict_issues()
        assert any("out of the required order" in issue for issue in issues)

    def test_an_unknown_recommendation_is_reported(self) -> None:
        doc = VerificationDoc.from_markdown(
            VERIFICATION_MD.replace(
                "Recommendation: → EXECUTE", "Recommendation: ship it"
            )
        )
        assert any("is not one of" in issue for issue in doc.verdict_issues())

    @pytest.mark.parametrize("target", VERDICT_RECOMMENDATIONS)
    def test_every_legal_recommendation_is_accepted(self, target: str) -> None:
        doc = VerificationDoc.from_markdown(
            VERIFICATION_MD.replace("→ EXECUTE", f"→ {target}")
        )
        assert doc.verdict_issues() == []

    def test_not_verified_is_a_required_section(self) -> None:
        assert "Not Verified" in VerificationDoc.REQUIRED_SECTIONS
        doc = VerificationDoc.from_markdown(
            VERIFICATION_MD.replace("## Not Verified", "## Untested")
        )
        assert doc.section_issues() == ["missing required section '## Not Verified'"]


# ---------------------------------------------------------------------------
# changelog.md
# ---------------------------------------------------------------------------

_GOOD_LINE = (
    "2026-07-22T01:20:00Z | iter-1/step-18 | 751fda0 | src/a.py | "
    "EDIT(+12,-3) | radius:LOW(1) | D-013 | tighten the budget"
)


class TestChangelog:
    def test_entries_parse(self) -> None:
        doc = ChangelogDoc.from_markdown(CHANGELOG_MD)
        assert len(doc.entries) == 4
        assert doc.entries[0].commit == "f63104f"
        assert doc.entries[3].commit == "uncommitted"

    def test_the_format_documenting_header_is_not_read_as_an_entry(self) -> None:
        """The header QUOTES the 8-field format; it must stay a header."""
        doc = ChangelogDoc.from_markdown(CHANGELOG_MD)
        assert "Format:" in doc.header
        assert len(doc.entries) == 4

    def test_exactly_eight_fields(self) -> None:
        entry = parse_changelog_line(_GOOD_LINE)
        assert len(entry.to_markdown().split(" | ")) == 8

    def test_seven_fields_are_rejected(self) -> None:
        with pytest.raises(HarnessArtifactError, match="of 8 fields"):
            parse_changelog_line(_GOOD_LINE.rsplit(" | ", 1)[0])

    def test_pipes_in_the_reason_are_tolerated(self) -> None:
        entry = parse_changelog_line(f"{_GOOD_LINE} | and a second clause")
        assert entry.reason == "tighten the budget | and a second clause"

    def test_a_numeric_step_is_required(self) -> None:
        """The `4b` plan label must be recorded positionally, e.g. `step-18`."""
        with pytest.raises(HarnessArtifactError, match="step"):
            parse_changelog_line(_GOOD_LINE.replace("step-18", "step-4b"))

    @pytest.mark.parametrize(
        ("bad", "replacement"),
        [
            ("2026-07-22T01:20:00Z", "2026-07-22 01:20:00"),
            ("751fda0", "751fd"),
            ("EDIT(+12,-3)", "EDITED"),
            ("radius:LOW(1)", "LOW(1)"),
            ("radius:LOW(1)", "radius:LOW"),
            ("D-013", "D13"),
            ("iter-1/step-18", "iter-1-step-18"),
        ],
    )
    def test_a_malformed_field_is_rejected(self, bad: str, replacement: str) -> None:
        with pytest.raises(HarnessArtifactError):
            parse_changelog_line(_GOOD_LINE.replace(bad, replacement))

    @pytest.mark.parametrize(
        "op",
        [
            "CREATE(+40)",
            "EDIT(+12,-3)",
            "DELETE(-9)",
            "RENAME(old.py→new.py)",
            "REVERT(src/a.py)",
        ],
    )
    def test_every_documented_op_shape_is_accepted(self, op: str) -> None:
        assert parse_changelog_line(_GOOD_LINE.replace("EDIT(+12,-3)", op)).op == op

    @pytest.mark.parametrize(
        "radius",
        [
            "radius:LOW(0)",
            "radius:MED(3)",
            "radius:HIGH(6)",
            "radius:UNKNOWN(script-missing)",
            "radius:UNKNOWN(script-error)",
        ],
    )
    def test_every_documented_radius_shape_is_accepted(self, radius: str) -> None:
        line = _GOOD_LINE.replace("radius:LOW(1)", radius)
        assert parse_changelog_line(line).radius == radius

    def test_a_dash_decision_ref_is_legal(self) -> None:
        assert (
            parse_changelog_line(_GOOD_LINE.replace("D-013", "-")).decision_ref == "-"
        )

    def test_compression_notes_survive_a_round_trip(self) -> None:
        text = CHANGELOG_MD.replace(
            "2026-07-21T20:10:30Z",
            "- (compressed: 14 low-decision-impact edits, iter-1/step-3, files: 4)\n"
            "2026-07-21T20:10:30Z",
            1,
        )
        doc = ChangelogDoc.from_markdown(text)
        assert doc.notes and doc.notes[0].startswith("- (compressed:")
        assert ChangelogDoc.from_markdown(doc.to_markdown()) == doc

    def test_a_directly_constructed_entry_is_validated(self) -> None:
        with pytest.raises(ValidationError, match="malformed"):
            ChangelogEntry(
                timestamp="2026-07-22T01:20:00Z",
                step="iter-1/step-4b",
                commit="751fda0",
                path="src/a.py",
                op="EDIT(+1,-0)",
                radius="radius:LOW(0)",
                decision_ref="-",
                reason="x",
            )


# ---------------------------------------------------------------------------
# checkpoints, summary, cross-plan tier
# ---------------------------------------------------------------------------


class TestCheckpointAndSummary:
    def test_the_checkpoint_sections(self) -> None:
        doc = CheckpointDoc.from_markdown(CHECKPOINT_MD)
        assert doc.section_issues() == []
        assert "none (no package manager touched)" in doc.body_of(
            "Lockfiles snapshotted"
        )

    def test_a_checkpoint_without_the_lockfile_section_is_reported(self) -> None:
        doc = CheckpointDoc.from_markdown(
            CHECKPOINT_MD.replace("## Lockfiles snapshotted:", "## Lockfiles:")
        )
        assert doc.section_issues() == [
            "missing required section '## Lockfiles snapshotted'"
        ]

    def test_an_inline_heading_value_still_matches_its_section(self) -> None:
        """``## Git State: commit abc123f`` is the same section as ``## Git State``."""
        doc = CheckpointDoc.from_markdown(
            CHECKPOINT_MD.replace("## Git State\n", "## Git State: commit 14e27a1\n")
        )
        assert doc.section_issues() == []

    def test_the_summary_carries_the_anchor_registry(self) -> None:
        doc = SummaryDoc.from_markdown(SUMMARY_MD)
        assert doc.section_issues() == []
        assert "D-032" in doc.body_of("Decision Anchors Registry")


class TestCrossPlanTier:
    def test_lessons_importance_tags(self) -> None:
        lessons = LessonsDoc.from_markdown(LESSONS_MD).lessons()
        assert len(lessons) == 8
        assert sorted({level for level, _ in lessons}) == [2, 3, 4, 5]

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("- a critical lesson [I:5]", 5),
            ("- a default lesson", 3),
            ("- explicitly ordinary [I:3]", 3),
            ("- low signal [I:1]", 1),
            ("- out of range [I:9]", 3),
        ],
    )
    def test_lesson_importance(self, line: str, expected: int) -> None:
        assert lesson_importance(line) == expected

    def test_the_line_caps_are_carried_as_data(self) -> None:
        assert LessonsDoc.LINE_CAP == Defaults.LESSONS_LINE_CAP == 200
        assert SystemAtlasDoc.LINE_CAP == Defaults.SYSTEM_LINE_CAP == 300
        assert LessonsDoc.PROTECTED_IMPORTANCE == 5

    def test_a_document_under_its_cap_is_not_over_cap(self) -> None:
        assert not LessonsDoc.from_markdown(LESSONS_MD).over_cap()
        assert not SystemAtlasDoc.from_markdown(SYSTEM_MD).over_cap()

    def test_an_over_cap_lessons_file_is_detected(self) -> None:
        doc = LessonsDoc.from_markdown(LESSONS_MD)
        section = doc.section("Codebase Gotchas")
        assert section is not None
        section.body += "\n" + "\n".join(f"- filler {n} [I:1]" for n in range(250))
        assert doc.over_cap()

    def test_the_atlas_six_sections_in_order(self) -> None:
        doc = SystemAtlasDoc.from_markdown(SYSTEM_MD)
        assert SystemAtlasDoc.REQUIRED_SECTIONS == (
            "Identity",
            "Components",
            "Boundaries",
            "Invariants",
            "Flows",
            "Known Patterns",
        )
        assert doc.section_issues() == []
        assert doc.section(SystemAtlasDoc.OPTIONAL_SECTION) is not None

    def test_the_index_rows(self) -> None:
        doc = IndexDoc.from_markdown(INDEX_MD)
        assert [row.plan for row in doc.rows] == [
            "plan-2026-07-21T125237-191b2eb2",
            "plan-2026-07-21T110044-ed1ae68b",
        ]
        assert doc.rows[0].date == "2026-07-21"
        assert doc.rows[1].topics == ""

    def test_an_index_row_added_in_code_round_trips(self) -> None:
        doc = IndexDoc.from_markdown(INDEX_MD)
        doc.rows.append(
            IndexRow(
                plan="plan-2026-07-22T000000-deadbeef", date="2026-07-22", goal="x"
            )
        )
        assert IndexDoc.from_markdown(doc.to_markdown()) == doc

    def test_the_consolidated_sliding_window_and_plan_sections(self) -> None:
        doc = ConsolidatedDoc.from_markdown(CONSOLIDATED_MD)
        assert ConsolidatedDoc.WINDOW == Defaults.SLIDING_WINDOW_PLANS == 4
        assert doc.plan_ids() == [
            "plan-2026-07-21T125237-191b2eb2",
            "plan-2026-07-21T110044-ed1ae68b",
        ]

    def test_balanced_compression_markers_report_nothing(self) -> None:
        assert ConsolidatedDoc.from_markdown(CONSOLIDATED_MD).marker_issues() == []

    @pytest.mark.parametrize(
        ("text", "fragment"),
        [
            ("<!-- COMPRESSED-SUMMARY -->\nbody\n", "unclosed"),
            ("<!-- /COMPRESSED-SUMMARY -->\n", "unmatched closing"),
            (
                "<!-- COMPRESSED-SUMMARY -->\n<!-- COMPRESSED-SUMMARY -->\n"
                "<!-- /COMPRESSED-SUMMARY -->\n<!-- /COMPRESSED-SUMMARY -->\n",
                "nested",
            ),
            (
                "<!-- COMPRESSED-SUMMARY -->\n<!-- /COMPRESSED-SUMMARY -->\n"
                "<!-- COMPRESSED-SUMMARY -->\n<!-- /COMPRESSED-SUMMARY -->\n",
                "2 compression blocks",
            ),
        ],
    )
    def test_broken_compression_markers_are_reported(
        self, text: str, fragment: str
    ) -> None:
        assert any(fragment in issue for issue in compression_marker_issues(text))


# ---------------------------------------------------------------------------
# Presentation Contracts
# ---------------------------------------------------------------------------


class TestPresentationContracts:
    def test_all_six_are_declared(self) -> None:
        assert list(PRESENTATION_CONTRACTS) == [
            "PC-EXPLORE",
            "PC-PLAN",
            "PC-EXECUTE-STEP",
            "PC-EXECUTE-LEASH",
            "PC-REFLECT",
            "PC-PIVOT",
        ]

    @pytest.mark.parametrize(
        ("name", "count"),
        [
            ("PC-EXPLORE", 4),
            ("PC-PLAN", 12),
            ("PC-EXECUTE-STEP", 5),
            ("PC-EXECUTE-LEASH", 5),
            ("PC-REFLECT", 5),
            ("PC-PIVOT", 5),
        ],
    )
    def test_required_field_counts(self, name: str, count: int) -> None:
        assert len(PRESENTATION_CONTRACTS[name].required) == count

    @pytest.mark.parametrize("name", list(PRESENTATION_CONTRACTS))
    def test_the_floor_is_a_subset_of_required(self, name: str) -> None:
        contract = PRESENTATION_CONTRACTS[name]
        assert contract.floor <= set(contract.required)
        assert contract.floor

    @pytest.mark.parametrize(
        "name", ["PC-EXECUTE-STEP", "PC-EXECUTE-LEASH", "PC-REFLECT"]
    )
    def test_the_all_mandatory_contracts_floor_on_every_field(self, name: str) -> None:
        contract = PRESENTATION_CONTRACTS[name]
        assert contract.floor == set(contract.required)

    def test_the_two_partial_floors(self) -> None:
        assert PRESENTATION_CONTRACTS["PC-EXPLORE"].floor == {
            "findings-index",
            "key-constraints",
        }
        assert PRESENTATION_CONTRACTS["PC-PIVOT"].floor == {
            "checkpoints",
            "candidate-directions",
        }

    def test_the_plan_floor_is_the_five_named_sections(self) -> None:
        assert PRESENTATION_CONTRACTS["PC-PLAN"].floor == {
            "steps",
            "success-criteria",
            "verification-strategy",
            "failure-modes",
            "assumptions",
        }

    def test_missing_floor_fields_are_reported_in_contract_order(self) -> None:
        missing = missing_floor_fields("PC-PLAN", ["goal", "steps", "assumptions"])
        assert missing == ("failure-modes", "success-criteria", "verification-strategy")

    def test_a_complete_emission_reports_nothing_missing(self) -> None:
        contract = PRESENTATION_CONTRACTS["PC-EXECUTE-STEP"]
        assert missing_floor_fields(contract.name, contract.required) == ()

    def test_dropping_one_mandatory_field_is_caught(self) -> None:
        contract = PRESENTATION_CONTRACTS["PC-EXECUTE-STEP"]
        assert missing_floor_fields(contract.name, contract.required[:-1]) == (
            "next-preview",
        )

    def test_an_unknown_contract_raises(self) -> None:
        with pytest.raises(KeyError):
            missing_floor_fields("PC-NOPE", [])

    def test_contracts_are_frozen(self) -> None:
        with pytest.raises(ValidationError, match=r"frozen|immutable"):
            PRESENTATION_CONTRACTS["PC-PLAN"].name = "PC-OTHER"


# ---------------------------------------------------------------------------
# Registry and fail-closed behaviour
# ---------------------------------------------------------------------------


class TestRegistryAndFailClosed:
    def test_every_per_plan_artifact_has_a_model(self) -> None:
        for name in ArtifactNames.PER_PLAN:
            assert name in ARTIFACT_MODELS

    def test_every_cross_plan_artifact_has_a_model(self) -> None:
        for name in ArtifactNames.CROSS_PLAN:
            assert name in ARTIFACT_MODELS

    def test_the_two_subdirectories_map_to_their_per_file_model(self) -> None:
        assert ARTIFACT_MODELS[ArtifactNames.FINDINGS_DIR] is FindingsTopicDoc
        assert ARTIFACT_MODELS[ArtifactNames.CHECKPOINTS_DIR] is CheckpointDoc

    def test_the_registry_covers_the_summary(self) -> None:
        assert ARTIFACT_MODELS[ArtifactNames.SUMMARY] is SummaryDoc

    @pytest.mark.parametrize("model", [model for model, _ in ARTIFACT_CASES])
    @pytest.mark.parametrize("garbage", ["", "not markdown at all", "## No H1 here\n"])
    def test_garbage_raises_the_artifact_error(
        self, model: type[Artifact], garbage: str
    ) -> None:
        with pytest.raises(HarnessArtifactError):
            model.from_markdown(garbage)

    @pytest.mark.parametrize("model", [model for model, _ in ARTIFACT_CASES])
    def test_the_error_names_the_artifact(self, model: type[Artifact]) -> None:
        with pytest.raises(HarnessArtifactError) as excinfo:
            model.from_markdown("garbage")
        assert excinfo.value.artifact == model.ARTIFACT

    def test_a_parse_failure_never_returns_a_partial_model(self) -> None:
        """No half-populated document escapes: the funnel raises or returns whole."""
        with pytest.raises(HarnessArtifactError):
            PlanDoc.from_markdown(_plan_md(omit="Steps"))

    def test_a_section_body_is_stripped_on_both_paths(self) -> None:
        doc = ProgressDoc(
            title="Progress",
            sections=[
                Section(name=name, body="\n\n- item\n\n")
                for name in ProgressDoc.REQUIRED_SECTIONS
            ],
        )
        assert all(section.body == "- item" for section in doc.sections)
        assert ProgressDoc.from_markdown(doc.to_markdown()) == doc

    def test_parse_markdown_table_drops_header_and_separator(self) -> None:
        rows = parse_markdown_table("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n")
        assert rows == [["1", "2"], ["3", "4"]]

    def test_parse_markdown_table_on_no_table(self) -> None:
        assert parse_markdown_table("just prose\n") == []
