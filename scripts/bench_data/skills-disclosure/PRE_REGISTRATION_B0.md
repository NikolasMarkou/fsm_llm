# Pre-registration, skills-disclosure B0 (decision block)

Written and committed BEFORE row 1, together with the manifests
`manifest_hint_on.json` and `manifest_hint_off.json`. Nothing in this file or in
`scripts/skills_bench.py` between its BEGIN FIXTURE and END FIXTURE markers may
change after this commit; a new question needs a new block with a new manifest.

Question: does `qwen3.5:4b` ReAct complete the two-step disclosure (skill_load of
the right skill, then skill_read of the right file) more often with the read hint
(`constants.SKILL_READ_HINT`) on the first skill_load page than without it?
Instrument: `scripts/skills_bench.py`, bench id `skills-disclosure`, block B0, 16
skills (10 targets, 6 distractors), 12 questions (10 read-needed, 2 controls whose
answer is in the SKILL.md body), 2 trials per question per arm, n = 48 rows.
Run order: question-major; for each question and trial the two arms alternate and
the arm that goes first flips from question to question (`plan("B0")`).

## The rule, verbatim from decisions D-013 and D-014 of plan 93d78536

D-013

**Context**: I-132/I-133/I-097. B2 (5/5) is a before/after across the whole completion-fix round on three skills and one question, and the ReAct agent decides at temperature 0, so the five trials were one observation. The hint is `SKILL_READ_HINT` (`constants.py:503`).
**Decision**: Pre-register B0 (qwen3.5:4b ReAct, `hint_on` vs `hint_off`, 12 questions x 2 trials, question-major, arms alternating) and B1 (informational: native_4b, react_9b, native_9b, hint on, model-major), 16 skills, fixture hashed, grading offline. Primary unit: first-trial G1 over the 10 read-needed questions per arm. Rule fixed before row 1: effect shown iff on minus off >= 4 cells AND Fisher exact two-sided p < 0.05 -> KEEP; no effect shown iff |on minus off| <= 1 -> REMOVE the hint; harmful iff off minus on >= 4 AND p < 0.05 -> REMOVE; every other outcome (a gap of 2 or 3 cells either way, or a gap of 4 or more with p >= 0.05) is inconclusive -> KEEP and say so. Both arms at ceiling (10/10 v 9/10 or 10/10) counts as "no effect shown", not as inconclusive. Rows whose `hint_seen` contradicts the arm are invalid and reported; more than 2 invalid cells invalidates the block (hint kept). No rewording branch: no arm tests a rewording.
**Trade-off**: A decision with a written revert path **at the cost of** possibly removing a line on a null result from one 4b model and one fixture (the hint costs about 76 characters per first page), with B1 giving no hint-off counterpart for the other models.
**Reasoning**: Fisher power for n=10 per arm: 10/10 v 4/10 p=0.011, 9/10 v 3/10 p=0.020, 8/10 v 2/10 p=0.023, so a 4-cell gap is the smallest detectable one. Removal on a null is allowed because the lesson "a speculative narrowing is where fixes of fixes come from" is answered by the scope sentence in the docs and by the revert path (the constant and three tests come back in one commit). The READ-HINT DECISION entry is appended after the numbers exist.

D-014

**Context**: Findings offered an optional length-matched `hint_neutral` placebo arm (+16 min) and a B1 gate.
**Decision**: No placebo arm; B1 is informational (Wilson intervals, no ship rule).
**Trade-off**: 48 plus 72 rows, about 1.5 h **at the cost of** the B0 claim being "removing this line", not "this instruction content", and of B1 supporting no decision.
**Reasoning**: The decision at stake is whether the line stays, and the removal arm answers that directly; a placebo would answer a content question nobody has asked. B1 exists to show whether the disclosure completes on other models (I-133), which a rate with an interval states fully.

## Amendment A1, 2026-10-23 (decision D-032 of plan 93d78536): the FLOOR clause

Made BEFORE registration and before row 1. Reason: live smoke trials (scratch, never
rows; progress.md and ledger rows N-004, N-005) showed the ReAct agent on
`qwen3.5:4b` with the shipped hint reading the exact target file in 0 of 8 trials at
16 skills: it loads one skill and stops, so the hint, which appears only in a
`skill_load` reply, has no step to act on. Read literally, D-013 would then say
`|on - off| <= 1` -> "no effect shown" -> REMOVE, on a measurement that could not
see the hint.

Clause, checked first (after the validity checks, before every other branch): if
k(hint_on) <= 1 AND k(hint_off) <= 1, that is, both arms at 1 of 10 or below on the
primary unit, the block is INCONCLUSIVE and the hint is KEPT. The scope is stated
with the result: at this floor the runs did not reach `skill_read`, so the hint
could not act and the block says nothing about its value.

Every other branch of D-013 is unchanged and decides every other pair of counts:
0 v 4 is the normal path (inconclusive, p = 0.087); 2 v 1 and 3 v 3 are "no effect
shown" -> REMOVE; 10 v 10 and 10 v 9 are "no effect shown" -> REMOVE (ceiling case);
a gap of 4 or more with p < 0.05 is KEEP (on ahead) or REMOVE (off ahead). The floor
stops at 1 on purpose (a pair of 2 v 1 reads as before). The rule is read with this
clause in front of the D-013 text above; `decide_b0` implements it and `report`
prints the applied branch ("floor, inconclusive" for this one).

Row counts: B0 = 2 arms x 12 questions x 2 trials = 48 rows. Runtime estimate: about
15 minutes (48 rows x 18.7 s, the mean of 8 smoke trials), up to about 25 minutes if
trials run to their step budget. B1 (own pre-registration) has 96 rows, about 17
minutes; D-014 above quotes the earlier "48 plus 72 rows, about 1.5 h", superseded by
these figures.

## Amendment A2, 2026-10-23 (decision D-034 of plan 93d78536): a reachable null

Made after registration commit 0ac943b and BEFORE row 1: no row of B0 or B1 exists,
no `run` was started, and the manifests (fixture hash, models and digests, src
hashes, git commit) are unchanged and are NOT re-registered. A1 above stays as
written; A2 replaces its floor clause (k <= 1 for both arms) and nothing else.
Reason: the A1 floor has a cliff. 1 v 1 was a floor KEEP but 2 v 1 read as "no
effect shown" and would REMOVE the hint, and ReAct on `qwen3.5:4b` reached
`skill_read` in 0 of 8 smoke trials, so outcomes of 0 to 3 of 10 are the likely ones
and a 2 v 1 result is plausible. Removing a line on that repeats the defect A1 was
written to stop, one step higher.

Rule order now: (1) validity checks (incomplete block, more than 2 invalid cells:
KEEP); (2) effect shown, on minus off >= 4 AND p < 0.05 -> KEEP; (3) harmful, off
minus on >= 4 AND p < 0.05 -> REMOVE; (4) if max(k_on, k_off) <= 4, the block is
INCONCLUSIVE and the hint is KEPT, with the scope stated: fewer than half of the
questions reached `skill_read` in the better arm, so the hint could not be tested;
(5) otherwise |on minus off| <= 1 -> no effect shown, REMOVE; anything else ->
inconclusive, KEEP. "No effect shown -> REMOVE" therefore needs max(k_on, k_off) >= 5
of 10. Branches (2) and (3) are unchanged and cannot fire at a floor anyway.

Cases: 0 v 0, 1 v 1, 2 v 1, 1 v 2, 4 v 4, 4 v 3 -> inconclusive, KEEP; 0 v 4 (a gap of
4 but p = 0.087) -> inconclusive, KEEP; 5 v 5, 5 v 4, 10 v 10, 9 v 10 -> no effect
shown, REMOVE; 10 v 4 (p = 0.011) -> effect shown, KEEP; 4 v 10 -> harmful, REMOVE.
`decide_b0` implements this order and `report` prints the applied branch.

## Operational definitions (they only make the text above computable)

- Cell: the first-trial row of one of the 10 read-needed questions of one arm
  (20 cells in the block). k is the number of an arm's cells with G1 true. Trial 2
  rows and the 2 control questions are reported (pass^2, controls) and are NOT in
  the rule.
- G1 (primary): the answer, with commas and whitespace removed and lower-cased,
  contains the question's token (a number that appears only in the target file),
  AND the tool trace holds skill_load of the target skill, AND skill_read of the
  target skill with the exact target path. G2 token only, G3 the exact skill_read,
  G4 G1 and no other question's token. Grading is offline, no judge.
- "on minus off" is k(hint_on) - k(hint_off). p is the Fisher exact two-sided p of
  [[k_on, 10 - k_on], [k_off, 10 - k_off]] (`harness_bench.fisher_exact_two_sided`).
- A row is valid iff `hint_seen` and `hint_in_reply` both equal the expectation:
  true for `hint_on` when the run loaded at least one skill that bundles files,
  false in every other case (including every `hint_off` row). `hint_seen` is read at
  core's send binding (`fsm_llm.llm.completion`): the hint line occurs in a request
  message. An invalid cell is a cell whose row is invalid; more than 2 invalid cells
  over both arms invalidate the block. Invalid rows are reported and never dropped.
- A block that did not finish (status aborted), or an arm with fewer than 10 cells,
  is reported with its reason and the hint stays.
- `hint_off` wraps `fsm_llm.agents.agent_skills._first_page` and drops exactly the
  line equal to `SKILL_READ_HINT`; it does not set the constant to an empty string.
- The rule is computed by `scripts/skills_bench.py report skills-disclosure`
  (`decide_b0`), from the raw rows, not from the summary files.
- Nothing is tuned inside the block and the block is never re-run.
