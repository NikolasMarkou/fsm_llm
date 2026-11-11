# Pre-registration, synthesis block s0 (decision block)

Written and committed BEFORE row 1, together with the manifests
`s0/manifest_synth-fsm.json`, `s0/manifest_hand-fsm.json` and `s0/manifest_react.json`.
After that commit nothing in this file, in `scripts/synthesis_bench.py` (its fixture
region and verdict rule included), in any file listed in its `SRC_PINS`, or in the
fixture may change, except dated amendments listed at the end and made before any row
existed. A new question needs a new block with a new manifest. A block is run once and
never re-run.

Instrument: `scripts/synthesis_bench.py`, bench id `synthesis`, block `s0`, plan
2cefeb61 (decisions D-039, D-040, D-054, D-055, D-065 to D-068, D-075 to D-077, D-083
to D-085).

## Question and thesis

Does a procedure that a model COMPILED into a finite state machine (arm `synth-fsm`,
S) beat the same model run as an in-context ReAct agent (arm `react`, R) on a
conversational procedure that collects values from a scripted user? A hand-written FSM
(arm `hand-fsm`, H) runs on the same host and runner as S and is the check that the
harness, the scripted user and the model can finish a procedure at all.

Hypothesis H1 (falsifiable): S passes more tasks than R, `s > r`, with Fisher exact
two-sided p < 0.05.

## One model, no strong/cheap split (mandatory wording)

Compiler, scenario writer, executor, ReAct agent and hand-written arm are ONE 2-bit 27B
model (`ollama_chat/bonsai-2:27b`, Q2_0, digest in the manifests). There is NO
strong/cheap split: the comparison is the same model under three scaffolds, so no
strong-compile/cheap-run cost asymmetry is tested. A G6b simulation pass is partly the
model agreeing with itself, a weaker signal than a separate judge. One run, one model,
n = 16 tasks. The roles are separate interfaces on the same model so that each role's
calls and tokens are metered.

## Design

- 16 tasks in four families of four, all judge-free, with the gold and the scripted
  user's rule tables inside the hashed fixture region: field collection (c1 to c4),
  routing (r1 to r4), branching (b1 to b4), validate-and-retry (y1 to y4). No
  tool-lookup task: a procedure cannot call tools (the runner refuses `completion`
  states).
- The scripted user is a rule table with a fixed fallback line, counted as
  `confused_turns`. No LLM plays the user.
- Arms. S: the host FSM's `synthesize` state compiles the child through `obtain` (G1 to
  G6a, then G6b) into a fresh temporary library per task; trials 2 and 3 of a task reuse
  the stored procedure, so reuse is measured. H: a hand-written child seeded through
  `run_gates` and `FSMLibrary.put` (gates G1 to G6a recorded, never G6b), same host,
  same runner, zero compile calls (a trial with one is an errored row). R:
  `create_agent("react")` with an `ask_user` tool backed by the same scripted user.
- 3 trials per task, order task-major, arms alternate, the first arm flips per task.
  Limits, temperature 0.5, timeouts (compile 600 s, others 300 s), G6b bounds and the
  policy are in the manifests; no `num_ctx` is sent.
- One grader for every arm (D-065, D-077): every required key equals its gold after
  case, whitespace and punctuation are dropped, plus the family check (branch: no
  question about the other branch after the branch value was given; retry: the rule was
  answered twice).

## The rule, fixed in advance

Primary unit: the first trial of each task, all required keys correct, 16 units per arm.
s, r, h are the tasks passed by S, R, H.

- SUPPORTED if the block is valid, p < 0.05 and s > r.
- REFUTED if the block is valid, p < 0.05 and r > s.
- NOT SUPPORTED (the honest null) if the block is valid and p >= 0.05. This includes s
  nominally above r.
- INCONCLUSIVE if the block is invalid: an arm block aborted or incomplete; or h < 10 of
  16 (the scripted user, runner and model cannot finish a hand-written procedure); or r
  and s are both at least 15 of 16 (ceiling); or more than 2 of 16 first-trial S
  failures have phase `provider` or `host`.

The p value is `bench_common.fisher_exact_two_sided`, never a second implementation.
The smallest gap that reaches p < 0.05 at n = 16 against 16, computed by that function
at registration: r = 0 needs s >= 5; r = 1, s >= 7; r = 2, s >= 9; r = 3, s >= 10; r = 4,
s >= 11; r = 5, s >= 12; r = 6, s >= 13; r = 7, s >= 14; r = 8 or 9, s >= 15; r = 10 or
11, s = 16; r >= 12 cannot reach it. So a verdict other than NOT SUPPORTED needs a gap of
at least 5 tasks, and a 2-bit model may well put the result inside that band.

Secondary, no verdict: S against H (cells, Fisher, Wilson), pass^3 and per-family
tables, tokens and wall time per arm and per role, S per trial with and without the
compile cost, the library-reuse trials (zero compile calls). Whatever happens is
reported, a refuted or null thesis included. No document says synthesis helps before
the rule is computed. S is never reported without its compile cost.

## What was seen before registration (disclosure)

The bench has not run. The live smoke (`synthesis_bench.py smoke`, three invocations,
raw files in the gitignored `logs/synthesis-bench/smoke/`, never rows) ran on the free
server on tasks c1, r1, b1, y1 and y2, and its outcomes were read before registration:

- Smoke 1 (c1): S, H, R passed. Smoke 2 (r1, b1, y1): S passed r1 and b1 and failed y1
  (run budget); H passed r1 and y1 and failed b1 (wrong value); R failed r1 (run budget)
  and passed b1, y1.
- After smoke 2 the compile prompt was changed once (D-083): the correction rule now
  says the state-level extraction instruction must name the exact key, and the scenario
  writer's prompt says every turn is a message the user types. After smoke 3 the repair
  text for an unparseable reply was given the parser position (D-085).
- Smoke 3 (y1, y2, c1) after those edits: S failed c1 and y1 at compile (four
  byte-identical unparseable replies, a `]` where `}` belonged) and failed y2 at run
  budget (the key-naming text landed in the per-field instruction, so the correction was
  lost); H failed y1 and passed y2, c1; R passed all three. Single trials of the 2-bit
  model are noisy: H passed y1 in smoke 2 and failed it in smoke 3.
- No task-shaped tuning followed (D-084): the compiled-arm prompts and the compile loop
  were left as they are after the generic edits above, the S arm is registered as built,
  and a deterministic normaliser that would add a key-naming extraction instruction to a
  state with a value gate is recorded as a candidate for a later block, not applied.
- Smoke observations that bear on interpretation: compile plus G6b simulation costs
  about 5 to 6 minutes per synthesized first trial (about 110 simulation calls); H and R
  take 10 to 60 seconds per trial, ReAct up to about 200 seconds when it loops. The G6b
  gate passed the y1 compile that then failed at run time, so G6b is a weaker check than
  `verified` suggests.

## Calibration status

Trials per task (3), the G6b bounds, `min_pass_rate` 0.5, the simulation trial count and
every limit are the values in the manifests. They were not calibrated against a
distribution: the smoke measured cost only. Projected block time from the smoke figures:
about 3 hours on a free server, under the 6 hour time box. The preflight refuses to run
unless no other model is loaded, the default context is loaded and the Ollama journal is
quiet; a foreign model or context during the run aborts the arm block as incomplete,
which makes the verdict INCONCLUSIVE rather than a silent bias.

## Amendments

None.
