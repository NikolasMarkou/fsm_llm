# Pre-registration, skills-disclosure B1 (informational block)

Written and committed BEFORE row 1, together with the manifests of the arms
`native_4b`, `native_4b_hint_off`, `react_9b` and `native_9b`. After that commit
this file and the fixture may change only by dated amendments, each made before any
row existed; the only one that touches B1 is A3 of PRE_REGISTRATION_B0.md.

Question (I-133): does the two-step disclosure also complete on other models and
agent patterns? Arms: `native_4b` (native_fc on
`ollama_chat/qwen3.5:4b`, hint on), `native_4b_hint_off` (the same, hint removed by
the same `_first_page` wrapper as B0's `hint_off`; added 2026-10-23 by decision D-032,
before registration), `react_9b` (ReAct on `ollama_chat/qwen3.5:9b-q8_0`, hint on),
`native_9b` (native_fc on the 9b, hint on). Same fixture, grader and limits as B0 (the
manifests pin the fixture hash), 12 questions x 2 trials per arm, 4 arms, n = 96 rows.
Run order: model-major, so no model reloads: first the two native 4b arms, which
alternate for each question and trial (the first of the pair flips from question to
question, as in B0), then `react_9b`, then `native_9b` (`plan("B1")`).
Runtime estimate: about 17 minutes (native 4b pair 48 rows x about 1.8 s; `react_9b`
24 rows x about 35 s; `native_9b` 24 rows x about 2 s; one 9b load), up to about 25
minutes with step-budget stops.

## Status of the result: informational, no gate

This block supports no decision and has no ship rule (decisions D-013 and D-014 of
plan 93d78536: "B1 is informational (Wilson intervals, no ship rule)"). `report`
prints, per arm, first-trial G1 over the 10 read-needed questions with its 95%
Wilson interval, G2 to G4, pass^2, the controls, the failure shapes and the
invalid-cell count; arms are not tested against each other, with ONE
exception that is itself informational: `report` prints the Fisher exact two-sided p
of the native 4b pair (`native_4b` v `native_4b_hint_off`, first-trial G1 cells), a
hint on/off view where the model does reach `skill_read`. It is labelled
"informational, no gate", it feeds no rule and it does not change the B0 decision.
Rows whose hint state contradicts the arm are reported, never dropped.
A block that did not finish is reported with its reason.

## Pinned and not pinned (amendment A3 of PRE_REGISTRATION_B0.md, 2026-10-23, before any row)

The run-time drift check hashes `src/fsm_llm/agent_skills.py`,
`src/fsm_llm/agents/agent_skills.py` and `src/fsm_llm/constants.py` only. The ReAct
agent, native_fc, `okf.text_page` and `llm.py` are not hashed, so a later run must
record the git commit it ran at in GRADING.md. The primary unit and the report
inputs are pinned by `tests/test_skills_bench.py`, not by a hash.

## Operational definitions

Cell, G1 to G4, validity and the rows are defined as in PRE_REGISTRATION_B0.md. The
numbers are recomputed from the raw rows by `scripts/skills_bench.py report`.
Nothing is tuned inside the block and the block is never re-run.
