# Pre-registration, synthesis block s2 (decision block, loop 2 of 3)

Written and committed BEFORE row 1 of s2, together with the manifests
`s2/manifest_synth-fsm.json`, `s2/manifest_hand-fsm.json` and `s2/manifest_react.json`.
After that commit nothing in this file, in `scripts/synthesis_bench.py` (its fixture
region and verdict rule included) or in any file of its `SRC_PINS` may change, except
dated amendments listed at the end and made before any s2 row existed. A block is run
once and never re-run. `s0` and `s1` (`PRE_REGISTRATION_s0.md`, `PRE_REGISTRATION_s1.md`,
their `REPORT_*.txt`, commits 869de48b and 9de9e935) are never edited.

Instrument: `scripts/synthesis_bench.py`, bench id `synthesis`, block `s2`, plan
2cefeb61 iteration 3 (decisions D-106, D-107).

## Why a third block, and what it is

The user directed a loop of three: read the results, diagnose, fix, run again. s2 is the
second re-run. It is NOT an independent test: the 16 tasks are the same 16 tasks s0, s1
and the smokes already showed, and the two fixes were chosen from failures on those tasks.
A better result than s1 is evidence that the diagnosed defects were real and fixed, not
that the approach generalises.

## Question, thesis and the rule (unchanged from s0 and s1)

Does a procedure that a model COMPILED into an FSM (`synth-fsm`, S) beat the same model
run as an in-context ReAct agent (`react`, R) on the 16 conversational tasks? A
hand-written FSM (`hand-fsm`, H) is the check that the harness can finish a procedure.
H1: s > r with Fisher exact two-sided p < 0.05. The verdict rule, its thresholds and its
decision function are byte-identical to s0's and s1's: a test pins the sha256 of `RULE`,
the source of `decide` and the five constants
(693ace887c19b520c8ee77fb8ab8a64487b96bf2cd5aecbfadbe55f0a638abbe).

- SUPPORTED if valid, p < 0.05 and s > r. REFUTED if valid, p < 0.05 and r > s. NOT
  SUPPORTED if valid and p >= 0.05. INCONCLUSIVE if invalid: an arm aborted or
  incomplete; or h < 10 of 16; or r and s both at least 15 of 16 (the ceiling); or more
  than 2 of 16 first-trial S failures have phase `provider` or `host`.
- Primary unit: the first trial of each task, 16 per arm.

Known before the run: s1 ended at S 15, H 15, R 16, so the ceiling clause already fired.
If s and r are again both at least 15 the verdict is INCONCLUSIVE by the same clause; a
different verdict needs r to fall to 14 or lower. These 16 tasks no longer separate the
arms for this model. The block is still run because the directive asks for it.

## One model, no strong/cheap split (mandatory wording)

Compiler, scenario writer, executor, ReAct agent and hand-written arm are ONE 2-bit 27B
model (`ollama_chat/bonsai-2:27b`, Q2_0, digest in the manifests). There is NO
strong/cheap split: the comparison is the same model under three scaffolds, so no
strong-compile/cheap-run cost asymmetry is tested. A G6b simulation pass is partly the
model agreeing with itself. One run, one model, n = 16 tasks.

## What s1 showed (the input to the fixes; `findings/s1-diagnosis.md`)

s1 (144 rows, complete): first trial S 15, H 15, R 16; all 48 trials 45, 45, 45; pass over
three S 14, H 15, R 14; verdict INCONCLUSIVE (ceiling). S failed one first-trial compile (r3,
two `no-progress` rounds: the gate-owned happy path says "topic is sample" for an edge with
no value gate and the model correctly answers null for a closed label set), and ran out of
budget on r2 trials 2 and 3 (the compiled instruction "the priority the user gives": the
user describes an outage, the model answers null). H failed r4 in all 3 trials (the hint
"exactly one of: hr, legal, it, finance" gives null for a request that quotes no label).
R ran out of iterations on r1 (2 trials) and r3 (1 trial).

## What changed since s1 (each fix, with its decision and commit)

- Compile prompt, one rule line (generic, no task text; D-106, 619d30b0): a key the goal
  asks the procedure to decide from what the user describes gets extraction text that says
  to choose the allowed value that best fits, with the values listed, and an edge that
  gates it with `in` over the same values (which also gives G6a a real label for the
  happy path).
- Harness v3, allowed only in this new block (D-107, 344a4419): the hand-written route
  hint says "the one of these that fits what the user needs: <labels>". s0, s1 and s2
  are NOT comparable for H on the route family. Fixture hash now
  60fb0b7a847fade49ab1baa6794bafb76fb04b897d6a39a3b4ac3d402f23a198 (s1:
  f66a35b47ab3faf8bd374e71c43be19ad5b7eb2009786246145e5f4a96929d1b).

Not changed: the 16 tasks and their gold, the grader, the scripted user, the host FSM,
the limits, the temperature, the timeouts, the G6b bounds, `min_pass_rate`, the rule.
Not fixed, recorded: the G6b happy-path false rejection (the gate's filler value against
a label-restricted instruction, one of 16 first trials in s1) and the ReAct arm's
re-asking.

## What the smoke showed before registration (disclosure)

One smoke invocation on r2, r3 and r4 (raw files in the gitignored
`logs/synthesis-bench/smoke/`, never rows): all 9 trials passed. S r3 compiled with the
text "Choose the allowed value that best fits what the user described" after one repair
round; S r2 and r4 compiled with a separate first question and passed. Single trials of
this model are noisy.

## Prediction (written before any row, to be scored, not tuned to)

S first trial at least 14 of 16 (s1: 15), H at least 14 of 16 (s1: 15), R at least 14 of 16
(s1: 16). Expected verdict INCONCLUSIVE (ceiling). The prediction is not part of the rule.

## Time box

About 2 to 3 hours of machine time on a free server (s1: about 2.1 hours). The preflight
refuses unless no other model is loaded, the default context is loaded and the Ollama
journal is quiet. A foreign model or context during the run aborts the arm block as
incomplete, which makes the verdict INCONCLUSIVE.

## Amendments

None.
