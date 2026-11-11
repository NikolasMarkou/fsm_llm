# Pre-registration, synthesis block s1 (decision block, loop 1 of 3)

Written and committed BEFORE row 1 of s1, together with the manifests
`s1/manifest_synth-fsm.json`, `s1/manifest_hand-fsm.json` and `s1/manifest_react.json`.
After that commit nothing in this file, in `scripts/synthesis_bench.py` (its fixture
region and verdict rule included) or in any file of its `SRC_PINS` may change, except
dated amendments listed at the end and made before any s1 row existed. A block is run
once and never re-run. `s0` (registered and run before, `PRE_REGISTRATION_s0.md`,
`s0/REPORT_s0.txt`, commit 869de48b) is never edited.

Instrument: `scripts/synthesis_bench.py`, bench id `synthesis`, block `s1`, plan
2cefeb61 iteration 3 (decisions D-087 to D-105).

## Why a second block, and what it is

The user directed a loop of three: read the results, diagnose, fix, run again. s1 is the
first re-run. It asks the s0 question again after the fixes the s0 raw calls justify.
It is NOT an independent test: the 16 tasks are the same 16 tasks s0 and the smokes
already showed, the fixes were chosen from failures on those tasks, so a better result
than s0 is evidence that the defects diagnosed were real and fixed, not that the
approach generalises to other tasks.

## Question, thesis and the rule (unchanged from s0)

Does a procedure that a model COMPILED into an FSM (`synth-fsm`, S) beat the same model
run as an in-context ReAct agent (`react`, R) on the 16 conversational tasks? A
hand-written FSM (`hand-fsm`, H) is the check that the harness can finish a procedure.
H1: s > r with Fisher exact two-sided p < 0.05. The verdict rule, its thresholds and its
decision function are byte-identical to s0's: a test pins the sha256 of `RULE`, the
source of `decide` and the five constants to the values at commit b08fc429
(693ace887c19b520c8ee77fb8ab8a64487b96bf2cd5aecbfadbe55f0a638abbe).

- SUPPORTED if valid, p < 0.05 and s > r. REFUTED if valid, p < 0.05 and r > s. NOT
  SUPPORTED if valid and p >= 0.05. INCONCLUSIVE if invalid: an arm aborted or
  incomplete; or h < 10 of 16; or r and s both at least 15 of 16; or more than 2 of 16
  first-trial S failures have phase `provider` or `host`.
- Primary unit: the first trial of each task, 16 per arm. The smallest gap that reaches
  p < 0.05 at 16 against 16: r = 0 needs s >= 5; 1, 7; 2, 9; 3, 10; 4, 11; 5, 12; 6, 13;
  7, 14; 8 or 9, 15; 10 or 11, 16; r >= 12 cannot reach it.

## One model, no strong/cheap split (mandatory wording)

Compiler, scenario writer, executor, ReAct agent and hand-written arm are ONE 2-bit 27B
model (`ollama_chat/bonsai-2:27b`, Q2_0, digest in the manifests). There is NO
strong/cheap split: the comparison is the same model under three scaffolds, so no
strong-compile/cheap-run cost asymmetry is tested. A G6b simulation pass is partly the
model agreeing with itself. One run, one model, n = 16 tasks.

## What s0 showed (the input to the fixes)

s0 (144 rows, status complete, 12:59Z to 15:33Z): first trial of 16 tasks, S 6, H 8,
R 14; verdict INCONCLUSIVE (h = 8 of 16, below 10). S failed 8 first trials at compile,
1 at run budget, 1 on a wrong value; pass over three trials 6; 21 reuse rows made no
compile call. The raw calls (`findings/s0-diagnosis.md`) showed: a compile JSON slip
(the `states` object closed with `]`) in about 8 first-trial compiles; invented JsonLogic
operators (`truthy`, `eq`, `not`, `any`, `all`, `&`) in 9; the repair text turning `<`
and `>` into look-alike characters that the model then copied; the retry family lost
because the correction text did not name the key (S 0 of 4); the hand-written arm failing
all four branch tasks (its value instruction was weak) and c2 and c3; the grader failing
right values ("double room" against gold "double", "the router" against "router") for H
and R.

## What changed since s0 (each fix, with its decision and commit)

Synthesis feature (generic, no task text in `src/synthesis/`, a hygiene test checks it):

- Compile-time key-naming extraction instruction for a state whose edge gates a key it
  extracts, appended after the model's text, hash-neutral when already present (D-091,
  D-096, e8cbf6ab).
- Deterministic repair of a mismatched closer in a compile reply, all gates still run on
  the parsed result (D-092, D-097, c1962e81).
- The compile prompt lists the loader's allowed JsonLogic operators, read from the
  loader's own constant, and says a condition that needs only a key present has no
  `logic` (D-098, 99d8dfa8).
- The repair text keeps the allowed operators in plain ASCII (D-099, e5f3b369) and says
  what a key name is when a gate reads a computed one (D-103, D-104, f818cb6a).

Harness v2, allowed only in this new block (D-093, D-100 to D-102, D-105; 71bda75b,
bb0558fb): one grader for all arms accepts a value equal to the gold after squash, or
containing the gold as a contiguous run of whole word tokens with at most two other
tokens, none a negator, a choice word or one of the task's own labels; the hand-written
branch children take the detail from the state's own question; rows keep the extracted
values. s0's H and R counts used the stricter grader and the old instruction, so s0 and
s1 are NOT comparable for H and R. Fixture hash now
f66a35b47ab3faf8bd374e71c43be19ad5b7eb2009786246145e5f4a96929d1b (s0:
bff1cbd152d73fbbdfbc77c68fff14688b3240e0996dbd47a176c68911a30349).

Not changed: the 16 tasks and their gold, the scripted user, the host FSM, the limits,
the temperature, the timeouts, the G6b bounds, `min_pass_rate`, the verdict rule.

## What the smokes showed before registration (disclosure)

Two smoke invocations after the fixes (raw files in the gitignored
`logs/synthesis-bench/smoke/`, never rows), on tasks c2, c3, r1, b1 and y1: S passed c2,
c3, b1 and y1 (y1 after one repair round) and failed r1 (the model extracted
`department = other` for an invoice complaint: a model judgement, no generic fix); H
passed all but one b1 smoke with the generic hint, which led to the state's-own-question
instruction, after which H b1 passed; R passed all five. The earlier smokes (c1, r1, b1,
y1, y2 before the fixes) are described in `PRE_REGISTRATION_s0.md`. Single trials of this
model are noisy.

## Prediction (written before any row, to be scored, not tuned to)

S first trial at least 10 of 16 (s0: 6), H at least 12 of 16 (s0: 8), R at least 13 of
16. If S reaches 10 or more while R stays at 13 or more, the expected verdict is NOT
SUPPORTED. A SUPPORTED verdict would need r to fall or s to reach 15. The prediction is
not part of the rule.

## Time box

About 3 hours of machine time on a free server (s0: 2.6 hours). The preflight refuses
unless no other model is loaded, the default context is loaded and the Ollama journal is
quiet. A foreign model or context during the run aborts the arm block as incomplete, which
makes the verdict INCONCLUSIVE.

## Amendments

None.
