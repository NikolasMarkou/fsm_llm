# bench_data: committed bench artifacts

Only the synthesis bench remains here (plan ace7a8c8, D-039): `agents-react`,
`skills-disclosure` and the harness blocks (`l4-execute-write`, `l6-e2e`,
`l7-explore-coldstart`, `l8-explore-loop`, `seed-probe`) were deleted with their
scripts `scripts/agents_bench.py` and `scripts/skills_bench.py`; their rows stay in
the git history. The stdlib helpers of the synthesis bench live in
`scripts/bench_common.py`.

The directory is tracked and no gate guards it. The blocks below were pre-registered
and run once at a fixed n; every manifest pins pre-restructure source paths, so
`report` refuses by design and `REPORT_<block>.txt` is the record.

## synthesis (scripts/synthesis_bench.py)

Bench status: blocks `s0`, `s1` and `s2` are registered, run and recorded
(144 rows each; manifests, rows, summaries, `REPORT_<block>.txt` and
`PRE_REGISTRATION_<block>.md`). A fourth block was planned and NOT run (the
session was wrapped up). All three verdicts are INCONCLUSIVE; see the results
table in `docs/synthesis.md`. The rows of one block are never edited or re-run.

Question: does a procedure a model COMPILED into an FSM (`synth-fsm`) beat the
same model as an in-context ReAct agent (`react`) on a conversational
procedure, with a hand-written FSM (`hand-fsm`) as the check that the harness
can pass at all? Every role of every arm is `ollama_chat/bonsai-2:27b`, one
2-bit 27B model: there is NO strong/cheap split, so no cost asymmetry between a
strong compiler and a cheap executor is tested, and a simulation pass is partly
the model agreeing with itself. One run, one model.

- Tasks: 16, four families of four, all judge-free: `collect` (gather three
  fields), `route` (one label from a fixed set), `branch` (ask a different
  follow-up per branch), `retry` (ask again until a number is in range). A
  scripted user (a rule table with a fixed fallback line that is counted as a
  confused turn; no model plays the user) answers. Gold values are in the
  script's hashed fixture region.
- Arms: `synth-fsm` compiles the goal through `FSMSynthesizer.obtain` with a
  fresh temporary library per task (trials 2 and 3 of a task reuse the stored
  procedure, so reuse is measured), with gates G1 to G6b; `hand-fsm` seeds a
  hand-written child into the library through the same gates and runs it on the
  same host FSM and runner, with no compile call; `react` is
  `create_agent("react")` with an `ask_user` tool backed by the same scripted
  user.
- Rule: written as `RULE` in the script, inside the hashed fixture region, and
  quoted here. Unit: the first trial of each task, all required keys correct;
  `s`, `r`, `h` are the tasks passed by `synth-fsm`, `react`, `hand-fsm` (of 16).
  H1: `s > r` with Fisher exact two-sided p < 0.05. SUPPORTED if valid, p < 0.05
  and `s > r`; REFUTED if valid, p < 0.05 and `r > s`; NOT SUPPORTED if valid and
  p >= 0.05; INCONCLUSIVE if invalid (an arm block aborted or incomplete, `h`
  below 10, `r` and `s` both at least 15, or more than 2 first-trial `synth-fsm`
  failures that are provider or host phases). A null or refuted result is
  reported as it is. `report` prints the smallest `s` that beats each `r`.
- Run order: task-major, the arms alternate and the first arm flips from task
  to task; trials per task are the script constant `TRIALS` (3 today, fixed at
  registration).
- Rows: `task_id`, `family`, `arm`, `trial`, `correct`, per-key correctness, a
  derived `phase` of a failure (`compile`, `provider`, `run_budget`, `contract`,
  `wrong_value`, `wrong_route`, `host`), `source` (`compiled` or `library`),
  `compile_rounds`, `runner_reason`, `confused_turns`, per-role call and token
  meters and per-request latency. Raw requests and replies go to the gitignored
  `logs/synthesis-bench/` (read them before any claim).
- Pins: the manifest holds the sha256 of the script, `bench_common.py`,
  `src/flow_analysis.py` and the synthesis, agents and eval modules the verdict
  depends on (`SRC_PINS`) plus the fixture region. `run` refuses and `report`
  flags an edit to one of them, so registration is the last source-touching
  commit.
  Plan 401d0471 step 16 edited five pinned files (`synthesis/runner.py` and
  `agents/{base,constants,definitions,tools}.py`: `timeout_s` to
  `timeout_seconds`, `conv_id` to `conversation_id`) after every manifest was
  already void, so it voids nothing that still ran.
- Preflight and per-trial guard: `smoke` and `run` refuse while another model is
  loaded, while another client used the server in the last 10 minutes, or when
  timing probes queue. During a run a changed loaded context or a second model
  aborts the block with status `aborted: foreign-load`; rows are kept. The bench
  never kills, pauses or reconfigures another job or Ollama.

```
.venv/bin/python scripts/synthesis_bench.py report s0   # recounts from rows
```
