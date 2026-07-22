# FSM-LLM Harness

> The iterative-planner protocol as a real FSM: six states, hard gates that read the filesystem, and an autonomy leash that cannot talk its way past two fix attempts.

---

## Overview

`fsm_llm_harness` is an FSM-LLM-native emulation of the iterative-planner
protocol. It runs the protocol as a genuine finite state machine rather than a
prompt-driven loop:

- **6 states**: EXPLORE, PLAN, EXECUTE, REFLECT, PIVOT, CLOSE.
- **9 transitions**, of which **4 are HARD-gated**. Every gate is a JsonLogic
  `TransitionCondition` term, so a gated edge is DETERMINISTIC or BLOCKED —
  never an LLM judgement call.
- **Memory is a directory of Markdown artifacts** (`state.md`, `plan.md`,
  `decisions.md`, `findings/`, ...) with an ownership table saying which role
  may write which file.
- **Gate values are DERIVED FROM THE FILESYSTEM**, not read from the model's
  report. `findings_count` counts non-empty `findings/*.md` files; a dispatch
  that claims a write must show a tool call whose target now carries bytes.
- **The autonomy leash halts at exactly 2 fix attempts** and cannot be reset
  from inside an approving callback.
- **Confinement flows through one `resolve()` chokepoint**: every path a role
  touches is resolved-then-compared against its confinement root, so a
  sentinel-prefixed or `../`-laden path escapes nothing.

The one idea worth carrying away: **a gate reads the filesystem, never the
model's account of the filesystem.** Both disk-derivation and the leash exist
because measurement caught 4B models asserting completed work over an empty
directory.

## Installation

```bash
pip install fsm-llm[harness]
```

The `harness` extra pulls in `fsm-llm[agents]` (the package imports
`fsm_llm_agents`). It has no third-party dependencies of its own.

**Requirements**: Python 3.10+

## Quick Start

### CLI

```bash
# Mint a plan directory and drive the protocol
fsm-llm-harness new "add a retry to the uploader"

# Mint and seed the plan directory, then stop (no LLM is called)
fsm-llm-harness new "add a retry to the uploader" --create-only

# Continue an existing plan directory
fsm-llm-harness resume plans/plan-2026-07-22T101500-1a2b3c4d

# Report a plan directory's position in the protocol
fsm-llm-harness status plans/plan-2026-07-22T101500-1a2b3c4d

# Audit a plan directory (optionally scan a source tree for decision anchors)
fsm-llm-harness validate plans/plan-... --workspace .

# Audit, then apply the CLOSE size policies (dry-run without --apply)
fsm-llm-harness close plans/plan-... --apply
```

Model resolution for the driving subcommands is `--model` > `$LLM_MODEL` >
`Defaults.MODEL`. Exit codes are exactly three: `0` pass, `1` a negative or
absent answer (an `audit()` ERROR, a failed run, a missing goal), and `2`
RESERVED for a HARD `pre_step_gate` refusal.

### Programmatic

```python
from fsm_llm_harness import HarnessAgent, ContextKeys

agent = HarnessAgent(
    worker_factory=my_worker_factory,       # Callable[[RoleRequest], AgentResult]
    approval_callback=lambda req: ...,      # consulted at every human gate; defaults to DENY
    revert_callback=None,                   # None => compute the revert directive, execute nothing
    findings_threshold=3,
    max_fix_attempts=2,
    max_leash_grants=2,
    iteration_hard_cap=6,
    max_explore_redispatches=9,
)

result = agent.run(
    "add a retry to the uploader",
    initial_context={
        ContextKeys.PLAN_DIR: "plans/plan-...",
        ContextKeys.WORKSPACE_ROOT: ".",
    },
)
```

- **`worker_factory=None` is a DIAGNOSTIC mode, not a way to run the protocol.**
  The FSM still turns and Pass 2 still answers, but no gate ever opens — a gate
  flag records worker or human evidence, and there is no worker to produce any.
  Expect a stall halt naming the first shut gate.
- **`approval_callback` defaults to a callback that DENIES.** An unattended run
  cannot approve its own plan or close itself.
- **`revert_callback=None` is not a degraded mode**: the `leash-cap`
  `RevertDirective` is always computed, scoped (never the plan directory) and
  reported; only its EXECUTION is deferred to a confirmed caller. `git` is
  deliberately absent from the command allowlist.
- **One run per instance**: `run()` takes a lock, and a worker that re-enters
  `run`/`api`/`conversation_id` gets `HarnessReentrancyError`.

## Architecture

### The protocol graph (`fsm_definition.py`)

`build_harness_fsm()` returns an FSM-JSON `dict` consumable by
`fsm_llm.API.from_definition`. **Lower `priority` wins**, and slots are spaced
>= 150 apart so two passing edges never fall inside the ambiguity threshold — a
gate decision is never routed to the LLM classifier.

| Edge | Priority | Gate (JsonLogic) |
|---|---|---|
| EXPLORE -> PLAN | 10 | **HARD**: `findings_count >= threshold` |
| PLAN -> EXECUTE | 10 | **HARD**: `plan_approved AND iteration < cap` |
| PLAN -> EXPLORE | 200 | `needs_explore` |
| EXECUTE -> REFLECT | 10 | `execute_complete` |
| REFLECT -> CLOSE | 10 | **HARD**: `close_confirmed AND all_criteria_pass` |
| REFLECT -> EXECUTE | 200 | **HARD**: `completion_fix AND fix_attempts < cap` |
| REFLECT -> PIVOT | 400 | `needs_pivot` |
| REFLECT -> EXPLORE | 600 | `needs_explore` |
| PIVOT -> PLAN | 10 | `pivot_resolved` |

Every condition declares `requires_context_keys`, so a garbled or missing
worker reply leaves the edge **BLOCKED** rather than accidentally satisfied.

### Disk-derived gates (`tools.py`)

Gate values are computed from the filesystem, not the worker's report:

- `derive_disk_counts` / `DISK_DERIVED_COUNTS` — the mapping from gate key to
  its disk derivation.
- `count_gate_files` / `gate_files` — count and enumerate the non-empty files a
  gate reads (e.g. `findings/*.md` for `findings_count`).
- `has_bytes` — the non-empty predicate. The gate value, the number the model
  is told, and the redispatch loop's condition are one derivation.

### The autonomy leash (`harness.py`)

Executor dispatches on ONE plan step are bounded by
`max_fix_attempts * (1 + max_leash_grants)` for **any** sequence of approvals.
The approval callback cannot raise it — an earlier version reset `fix_attempts`
on every grant and the leash was decorative. Both counters reset together only
when the driver advances to a new plan step.

### The confinement chokepoint (`tools.py`)

`Workspace` (the source tree) and `PlanMemory` (one plan directory, also
ownership-scoped) route every path through ONE `resolve()` method:
**RESOLVE FIRST, COMPARE SECOND**. A sentinel-prefixed absolute path
(`/workspace/uploader.py`) is rewritten to root-relative before the
resolve-and-compare; `/etc/passwd`, `../outside.txt`, symlink escapes and
shared-prefix cases all raise `HarnessConfinementError`.

### Ownership model (`rules.OWNERSHIP`)

A table mapping 16 artifacts to the roles permitted to WRITE them.
`PlanMemory.authorise` reads it directly, so an edit to the table changes what a
live role can write. See `src/fsm_llm_harness/CLAUDE.md` for the full ownership
table, the per-state role/tool map, and the artifact grammars.

## Key API Reference

### HarnessAgent (`harness.py`)

The driver. A `fsm_llm_agents.BaseAgent` subclass that builds the harness FSM,
registers handlers at 6 state entries plus a pre-step gate, and dispatches one
worker per state entry. Constructor keywords include `worker_factory`,
`approval_callback`, `revert_callback`, `config`, `findings_threshold`,
`max_fix_attempts`, `max_leash_grants`, `iteration_hard_cap`,
`max_explore_redispatches`, `max_plan_redispatches`, and `max_stall_turns`.
Public surface: `run()`, `api`, `conversation_id`, `presentations`, `reverts`,
`audit_issues`, `on_leash_cap`.

### PlanDirectory (`storage.py`)

The driver's accessor for one plan directory: plan-id minting, atomic writes,
the CLOSE size policies (LESSONS eviction, SYSTEM cap, the 4-plan cross-plan
sliding window), and `RunState` persistence. Reads go through a 4 MB
driver-facing path; a worker's `PlanMemory.read_text` keeps a 64 KB cap.

### plan_validator (`plan_validator.py`)

```python
from fsm_llm_harness import pre_step_gate, audit

gate = pre_step_gate("plans/plan-...")                 # -> GateResult
issues = audit("plans/plan-...", workspace_root=".")   # -> list[Issue]
```

`pre_step_gate` evaluates 4 ordered slugs (`no-plan`, `wrong-state`,
`leash-cap`, `iteration-cap`) and short-circuits on the first failure (every
failure is HARD, exit code 2). `audit` runs 30 checks and NEVER raises for a
finding — a check that raises is itself reported as an ERROR.

## Bench Methodology

Capability claims are backed by pre-registered fixed-n benches from
`scripts/harness_bench.py`, with raw evidence committed under
`scripts/bench_data/` (git-tracked on purpose, so every number can be
recomputed). See `scripts/bench_data/README.md` for the full protocol. In
brief:

- **Pre-registration**: a block's manifest is written before dispatch 1. A
  block runs ONCE at its fixed n — no interim looks, no re-rolls.
- **6 required manifest fields**: `prompt_bytes_sha256`, `tool_surface`,
  `fixture_hash`, `model_digest`, `arm`, `git_commit`.
- **Append-only rows** (`rows_<arm>.jsonl`), one raw row per dispatch;
  `content_matched` is sha256-based, never a stat.
- **Statistics**: Wilson 95% CI per arm and Fisher exact two-sided between
  arms, with per-row seeds recorded.

```bash
.venv/bin/python scripts/harness_bench.py report <bench-id>
```

## Status — what is measured, and what is not

Measured live on `ollama_chat/qwen3.5:4b` (digest `2a654d98e6fb`). Small n is
reported as k/n, and the bars are the ones the plans set in advance. Numbers
below are freshly recomputed from the committed raw rows.

### Model-level single-state bars (MET)

| Criterion | Bar | Measured |
|---|---|---|
| L4 write tool issued AND workspace bytes on disk | >= 4/5 | **5/5 issued, 5/5 bytes — MET** |
| L4 strict sha256 content-hash match of the requested edit | >= 4/5 | **4/5 — MET** (react control 0/5) |
| L5 >= 3 distinct non-empty `findings/*.md` from dispatches | >= 4/5 | **5/5 — MET** |

L4 was moved by a **measured structural fix, not prompt wording**: the driver
now reads `plan.md`'s Files To Modify and names the exact target path + tool in
the EXECUTE dispatch. The pre-registered bench `l4-execute-write` measured the
effect directly — native EXECUTE dispatches content-matched the requested edit
**2/40** before the fix (B0) and **40/40** after (B1), Fisher p ≈ 1.6e-20; the
ReAct control arm was 0/40 in both blocks. Caveat: the content-match metric
shares vocabulary with the fix's own prompt text, so treat a PASS as
target-selection compliance, not proven code correctness (`content_matched_ast`
is the vocabulary-decoupled successor for future blocks).

### End-to-end bar (NOT MET)

`TestL6EndToEndRealWorkers` is the package's first graded end-to-end criterion
on REAL role workers (n=3 per block). It is NOT MET in either committed block:

| Block | Result | Shape |
|---|---|---|
| L6 B0 | **0/3 — NOT MET** | 2 honest explore-cap halts, 1 slugless PLAN stall over an empty plan-writer reply |
| L6 B1 | **0/3 — NOT MET** | 3/3 `furthest_state=explore`, `halt_slug=explore-cap`, `honest_halt=true`, zero slugless stalls |

The measured blocker is **EXPLORE cold-start over an empty plan directory** —
one state earlier than B0's mixed picture. The PLAN-and-later machinery (a PLAN
redispatch budget halting on the honest `plan-cap` slug; a driver-named
`plan.md` deliverable line; a verified-write floor tightened to an EXECUTE-state
workspace write) is offline-verified but live-unexercised, because no B1 run
left EXPLORE.

### Cold-start lever (REFUTED)

The `l7-explore-coldstart/B0` block tested whether seeding a zero-byte protocol
skeleton into the plan directory would move first-dispatch EXPLORE cold-start.
Two arms, n=12 each, byte-identical prompts, differing only in on-disk
population:

| Arm | bytes on disk |
|---|---|
| `bare` (plain `mkdir`) | **5/12** (Wilson 95% [0.193, 0.680]) |
| `seeded` (skeleton stubs) | **7/12** (Wilson 95% [0.320, 0.807]) |

Fisher two-sided **p = 0.6843**. The pre-registered rule (VALIDATED iff
`k_seeded > k_bare` AND p < 0.05) yields **NOT VALIDATED** — a positive but
non-significant delta. The zero-byte seeding capability exists in the code but
ships UNWIRED. A second finding falls out of the same block and reframes the
end-to-end blocker: a single cold-start EXPLORE dispatch over a bare directory
scores ~5/12, far above L6's e2e 0/3, so **L6's blocker is the multi-dispatch
redispatch-loop / structured-output-parse collapse, not first-dispatch cold
start.**

## What is NOT claimed

- The harness is **NOT production-ready**.
- A 4B model is **NOT claimed to drive it unattended to a useful result** — the
  L6 0/3, measured twice, reinforces this, it does not soften it.
- **L6 end-to-end is measured 0/3 (unmet)** in both committed blocks.
- All PLAN-and-later machinery is **offline-proven but live-unexercised** — no
  committed live run reached PLAN.
- The **zero-byte cold-start seeding lever was refuted** (Fisher p = 0.6843) and
  ships unwired.

What IS claimed is narrow and mechanical: the gates read the filesystem, and a
confident sentence cannot open one.

For the full protocol graph, ownership table, artifact grammars, exact test
counts and the complete measured-vs-not-measured writeup, see
[`src/fsm_llm_harness/CLAUDE.md`](CLAUDE.md).

## CLI Tools

| Command | Description |
|---|---|
| `fsm-llm-harness new <goal> [--plans-dir DIR] [--create-only]` | Mint a plan directory and drive the protocol |
| `fsm-llm-harness resume <plan_dir> [--goal GOAL]` | Continue an existing plan directory |
| `fsm-llm-harness status <plan_dir>` | Report a plan directory's position |
| `fsm-llm-harness validate <plan_dir> [--workspace DIR]` | Audit a plan directory |
| `fsm-llm-harness close <plan_dir> [--workspace DIR] [--apply]` | Audit, then apply the CLOSE size policies |

The driving subcommands (`new`, `resume`) also accept the shared model options
(`--model`, resolved as `--model` > `$LLM_MODEL` > `Defaults.MODEL`).

## Exception Hierarchy

```
FSMError
└── HarnessError
    ├── HarnessArtifactError(artifact, message, cause=None)  # unreadable/unparseable/over-cap
    ├── HarnessOwnershipError(artifact, role, owner)         # OWNERSHIP denies this role the write
    ├── HarnessReentrancyError(role)                         # a worker re-entered the driver
    └── HarnessConfinementError(path, root)                  # a path escaped its root
```

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE) for details.
