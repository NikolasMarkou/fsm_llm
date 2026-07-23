# fsm_llm_harness -- Iterative-Planner Protocol Harness

An FSM-LLM-native emulation of the iterative-planner protocol: a 6-state
EXPLORE / PLAN / EXECUTE / REFLECT / PIVOT / CLOSE machine whose hard gates are
JsonLogic `TransitionCondition` terms, whose memory is a directory of Markdown
artifacts on disk, and whose autonomy leash halts at exactly 2 fix attempts.

- **Version**: 0.5.0 (synced from fsm_llm)
- **Extra deps**: none of its own; the extra pulls `fsm-llm[agents]` because the
  package imports `fsm_llm_agents`
- **Install**: `pip install fsm-llm[harness]`
- **CLI**: `fsm-llm-harness` (`python -m fsm_llm_harness`)

**The one idea worth carrying away**: a gate reads the FILESYSTEM, never the
model's account of the filesystem. `findings_count` is a count of non-empty
`findings/*.md` files, not the integer the worker reported; a dispatch that holds
a write tool and claims a write must show a tool call whose target now carries
bytes. Both mechanisms exist because measurement found 4B models asserting
completed work over an empty directory 5/5.

## File Map

```
fsm_llm_harness/
├── harness.py          # HarnessAgent -- the driver. 6 state-entry handlers, the pre-step
│                       #   gate, worker dispatch, the leash, Presentation Contracts,
│                       #   state.md read/write, resume. (3,089 lines -- the biggest file)
├── artifacts.py        # Pydantic models + Markdown (de)serializers for 15 artifact kinds,
│                       #   the 9 decision entry-type schemas and the 6 Presentation Contracts
├── storage.py          # PlanDirectory: plan-id minting, atomic writes, LESSONS eviction,
│                       #   SYSTEM cap, the 4-plan cross-plan sliding window, RunState
├── plan_validator.py   # pre_step_gate() (4 slugs, ordered, short-circuit) + audit() (30 checks)
├── tools.py            # Workspace (confined source tree) + PlanMemory (confined AND
│                       #   ownership-scoped plan directory) + the 13 agent-facing tools
├── roles.py            # RoleSpec x6, role prompt builders, build_default_worker_factory
├── rules.py            # OWNERSHIP, ROLE_BY_STATE, per-state StateRules, EXPLORE_TOPICS
├── fsm_definition.py   # build_harness_fsm() -- 6 states, 9 transitions, the JsonLogic gates
├── constants.py        # HarnessStates, Role, ContextKeys, ArtifactNames, GateSlug,
│                       #   Severity, PlanSchema, Defaults, DRIVER_OWNED_SEEDS
├── hardening.py        # Small-model reply recovery: strip_model_noise, parse_json_payload,
│                       #   parse_role_output, coerce_worker_output, retry
├── exceptions.py       # HarnessError -> Artifact / Ownership / Reentrancy / Confinement
├── __main__.py         # main_cli(): new / resume / status / validate / close, exit 0/1/2
├── __version__.py      # Imports from fsm_llm.__version__
└── __init__.py         # 118 public exports in one literal __all__
```

## The Protocol Graph (`fsm_definition.py`)

6 states, 9 transitions. **Lower `priority` wins** (`TransitionEvaluator` derives
confidence as `max(0.1, 1.0 - priority/1000)`), and slots are spaced >= 150 apart
so two passing edges never fall inside the 0.1 ambiguity threshold -- a gate
decision must never be routed to the LLM classifier.

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

Every condition declares `requires_context_keys`, so a garbled or missing worker
reply leaves the edge **BLOCKED** rather than accidentally satisfied: the
evaluator fails a condition whose key is absent before it evaluates the logic.

**No harness state carries `extraction_instructions`.** The field was deleted
from `StateRules` outright, because `pipeline.py`'s additive bulk-extraction pass
fires on `bool(state.extraction_instructions)` alone and costs one extra LLM call
per turn. Measured live: 2.000 -> 1.000 core LLM calls per FSM turn.

## Key Classes

### HarnessAgent (`harness.py`)

The driver. A `fsm_llm_agents.BaseAgent` subclass that builds the harness FSM,
registers handlers at 6 state entries plus a pre-step gate, and dispatches one
worker per state entry.

```python
from fsm_llm_harness import HarnessAgent, ContextKeys

agent = HarnessAgent(
    worker_factory=my_worker,               # Callable[[RoleRequest], AgentResult]
    approval_callback=lambda req: ...,      # defaults to a callback that DENIES
    revert_callback=None,                   # None => compute the revert, execute nothing
    findings_threshold=3,
    max_fix_attempts=2,
    max_leash_grants=2,
    iteration_hard_cap=6,
    max_explore_redispatches=9,
)
result = agent.run(
    "add a retry to the uploader",
    initial_context={ContextKeys.PLAN_DIR: "plans/plan-...", ContextKeys.WORKSPACE_ROOT: "."},
)
```

- **Public surface**: `run()`, `api`, `conversation_id`, `presentations`,
  `reverts`, `audit_issues`, `on_leash_cap`.
- **`worker_factory=None` is a DIAGNOSTIC mode, not a way to run the protocol**:
  the FSM still turns, but no gate ever opens because a gate flag records worker
  or human evidence and there is no worker to produce any. Expect a stall halt.
- **`approval_callback` defaults to DENY.** An unattended run cannot approve its
  own plan or close itself.
- **`revert_callback=None` is not a degraded mode**: the `leash-cap`
  `RevertDirective` is always computed and scoped (never the plan directory) and
  reported in the leash block; only its EXECUTION is deferred to a confirmed
  caller. `git` is deliberately absent from `COMMAND_ALLOWLIST`, so the driver
  never shells out to it.
- **One run per instance**: `run()` takes a `threading.Lock`; a worker that
  re-enters `run`/`api`/`conversation_id` gets `HarnessReentrancyError`.

**Leash arithmetic.** Executor dispatches on ONE plan step are bounded by
`max_fix_attempts * (1 + max_leash_grants)` = 6 for **any** sequence of
approvals. The approval callback cannot raise it -- an earlier version reset
`fix_attempts` on every grant and the leash was decorative.

**Driver-owned context.** `constants.DRIVER_OWNED_SEEDS` seeds 16 driver-owned
keys with falsy values before turn 1 -- the nine gate flags plus the counters and
rollups -- and `DRIVER_OWNED_UNSET` names 5 more that must stay ABSENT. This is
not tidiness: core's
`_build_field_configs_from_state` mints a REQUIRED extraction config for every
key named in a transition condition's `requires_context_keys`, so an unseeded
gate key is a key the LLM is asked to invent every turn. Measured before the
seeds existed: an LLM emitting `{"plan_approved": true, "close_confirmed": true}`
drove a full traverse to CLOSE while every worker dispatch failed and a DENYING
approval callback was never consulted.

### PlanDirectory (`storage.py`)

The driver's accessor for one plan directory. Composes `PlanMemory`, so it
inherits confinement and ownership rather than restating them; what it adds is
atomicity, the path layout and the three size policies.

- `PlanDirectory(plan_dir, role=Role.ORCHESTRATOR)` / `PlanDirectory.create(parent)`
- Reads: `read_text`, `read_artifact`, `list_dir`, `exists`, `finding_path`,
  `checkpoint_path`, `load_run_state`
- Writes: `write_text`, `append_text`, `write_artifact`, `save_run_state`
- CLOSE policies: `enforce_lessons_cap`, `enforce_system_cap`, `apply_sliding_window`
- Module functions: `mint_plan_id()`, `evict_lessons()`, `check_system_cap()`,
  `apply_sliding_window()`
- **`path` vs `root`**: `.path` is the plan directory; `.root` is its PARENT,
  the confinement root `PlanMemory` was constructed with. Pass `.path` to
  anything that expects a plan directory.

**Atomicity is load-bearing, not belt-and-braces**: a torn `state.md` still
PARSES, and a truncated Fix-Attempts section reads as a smaller
`fix_attempt_count` -- i.e. a leash that resets itself on a crash. Writes go
through a module-local `_atomic_write_text` that creates its temp file in
`target.parent` (`os.replace` is atomic only within one filesystem) and
`os.replace`s it into position.

**Two read paths, on purpose.** `PlanMemory.read_text` -- the tool a ROLE calls
-- keeps a 64 KB cap, because that cap bounds what an untrusted worker can pull
into an LLM context window. `PlanDirectory.read_text` -- the DRIVER's accessor,
never handed to a worker -- reads directly with a separate
`DRIVER_READ_MAX_BYTES` of 4 MB, because a real `decisions.md` outgrows 64 KB and
the audit checks that matter most were going dark on the biggest artifact.

**LESSONS is EVICTED, SYSTEM is REFUSED -- the caps are not symmetric.**
`LESSONS.md` carries a protocol-defined eviction order (the `[I:N]` tag, 5
protected) and is trimmed, but only for a section the parser can reproduce BYTE
FOR BYTE from its own parse; anything else raises. `SYSTEM.md` carries no
ordering and all six of its sections are required, so its cap is measured and an
over-cap atlas is refused, never trimmed.

### plan_validator (`plan_validator.py`)

```python
from fsm_llm_harness import pre_step_gate, audit

gate = pre_step_gate("plans/plan-...")        # -> GateResult(passed, slug, detail, exit_code)
issues = audit("plans/plan-...", workspace_root=".")   # -> list[Issue]
```

- **`pre_step_gate`** evaluates 4 slugs in `GateSlug.ORDER` -- `no-plan`,
  `wrong-state`, `leash-cap`, `iteration-cap` -- and the FIRST failure returns.
  Every failure is HARD (`exit_code == 2`). It reads exactly one file,
  `state.md`, with a plain `Path.read_text` and writes nothing: routing it
  through `PlanMemory` would `mkdir` the very directory whose absence `no-plan`
  exists to report.
- **`audit`** runs 30 checks (`CHECKS`) and NEVER raises for a finding -- a check
  that raises is itself reported as an ERROR, so one unreadable artifact cannot
  suppress the others.
- The two-tier leash thresholds are deliberately NOT the gate thresholds: 2
  attempts is legal, 3 is a WARNING (the gate was passed), 4+ is an ERROR.
  Iteration: WARN at 5, ERROR at 6+.

`CHECKS` (30): `anchor-badprefix`, `anchor-orphan`, `anchor-refs-missing`,
`anchor-refs-stale`, `anchor-unqualified`, `atlas-absent`, `atlas-cap`,
`changelog-dref-orphan`, `changelog-malformed`, `checkpoints`, `complexity`,
`compress-markers`, `decisions-schema`, `evidence`, `findings`,
`findings-index`, `findings-topic`, `iteration`, `leash`, `lessons-absent`,
`lessons-cap`, `lessons-eviction`, `ownership`, `plan`, `plan-section`,
`preamble-mismatch`, `preamble-missing`, `progress`, `state`, `verdict`.

### Workspace / PlanMemory (`tools.py`)

Two confined roots, two vocabularies, ONE `resolve()` chokepoint.

| | `Workspace` | `PlanMemory` |
|---|---|---|
| Root | the source tree being edited | one plan directory |
| Checks | confinement | confinement **+** `rules.OWNERSHIP` |
| Tools | `read_file`, `write_file`, `append_file`, `delete_file`, `list_dir`, `path_exists`, `grep_files`, `run_command` | `read_plan_file`, `write_plan_file`, `append_plan_file`, `list_plan_dir`, `plan_path_exists` |

- **RESOLVE FIRST, COMPARE SECOND.** A model-emitted sentinel-prefixed absolute
  path (`/workspace/uploader.py`) is rewritten to root-relative BEFORE the
  unchanged resolve-and-compare. `/etc/passwd`, `../outside.txt`,
  `a/../../outside.txt`, symlink escapes and the shared-prefix `ws-evil` case all
  still raise `HarnessConfinementError` (16 escape shapes pinned, and a 43-case
  attack found zero escapes).
- The sentinel LISTS are split (`_WORKSPACE_SENTINELS` vs `_PLAN_SENTINELS`) --
  a shared list let `Workspace.resolve("/plan/findings/x.md")` write protocol
  memory into the user's source tree, confined but into the wrong root.
- `run_command` is **disabled by default**; `COMMAND_ALLOWLIST` is
  `cat grep head ls tail wc` and `git` is deliberately not in it (it executes
  repo-local hooks, aliases and pagers). `VERIFICATION_COMMANDS`
  (`git make mypy pytest ruff`) is a NAMED set a caller may opt into, not a
  default.
- Caps: `MAX_READ_BYTES` 64 KB, `MAX_OUTPUT_BYTES` 8 KB, `MAX_LIST_ENTRIES` 200,
  `MAX_GREP_HITS` 50.
- **Corrective tool feedback, never re-routing.** A FAILED cross-root call is
  ANNOTATED with the counterpart tool's name ("that path belongs to the plan
  directory: use `write_plan_file`"); a FAILED `read_plan_file` on a
  not-yet-existing protocol artifact is annotated with "`write_plan_file`
  creates the file, and any missing folder, in one call". Neither converts a
  failure into a success, and a failed WRITE is never told "write it" -- that
  would make an ownership refusal read as encouragement.
- Disk-derived gate values: `gate_files`, `count_gate_files`, `has_bytes`,
  `derive_disk_counts`, `DISK_DERIVED_COUNTS`. The gate value, the number the
  model is told, and the re-dispatch loop's condition are one derivation by
  construction.

### Roles (`roles.py`)

Six frozen `RoleSpec`s, one per state, all derived from `rules.OWNERSHIP` so tool
scope, prompt text and `owned_artifacts` are the same fact read three times.

| State | Role | Owns | Loop budget | Writable gate keys |
|---|---|---|---|---|
| EXPLORE | `explorer` | `findings/` | 14 | `findings_count`, `needs_explore` |
| PLAN | `plan-writer` | `plan.md`, `decisions.md`, `verification.md` | 10 | `needs_explore`, `total_steps` |
| EXECUTE | `executor` | `decisions.md`, `changelog.md`, `checkpoints/` | 14 | *(none -- `summary` is schema-visible only)* |
| REFLECT | `verifier` | *(nothing)* | 12 | `all_criteria_pass`, `needs_pivot`, `completion_fix`, `needs_explore`, `criteria_pass_count`, `criteria_total` |
| PIVOT | `reviewer` | `findings/` | 10 | `pivot_resolved`, `pivot_reason` |
| CLOSE | `archivist` | `decisions.md`, `summary.md` + all 6 cross-plan files | 10 | `halt_reason` |

The verifier owns nothing on purpose: a verifier RETURNS results and the driver
merges them into `verification.md`.

**Prompt placement is a MEASURED result, not a preference.** `build_role_prompt`
is three filters over one ordered block list:
`build_role_system_prompt` returns the STANDING blocks (identity, exit gate,
operative rules, held tools, write scope, stop rule, reply shape),
`build_role_task_prompt` returns the per-dispatch blocks (goal, position, context
snapshot, assigned topic), and `build_role_prompt` returns all of them in the
original order, byte-identical to before the split. On `:4b`, EXECUTE, n=5 per
arm with workspace bytes stat'd: whole prompt in the user turn wrote 0/5;
deleting the entire rules block reached 2/5; moving the standing half to the
SYSTEM message reached 4/5. Nothing was deleted or softened -- every byte the
model was told before, it is still told.

**Every role schema carries `message: str`.** Without it, core's
`_parse_response_generation_response` replaces any brace-wrapped reply lacking a
`message` key with `_GENERIC_FALLBACK_MESSAGE`, and a schema-constrained role
reply is structurally guaranteed to be exactly that shape. Supplying the key
makes core's EXISTING rescue fire two rungs earlier; the terminal guard is NOT
weakened. `message` and `summary` are schema-visible, absent from
`_WORKER_WRITABLE`, dropped by `coerce_worker_output`, and not required by
`parse_role_output`.

**`build_default_worker_factory(workspace, ...)`** builds the stock worker.
`native_function_calling=True` is the default: roles are backed by
`NativeFunctionCallingReactAgent`, which issues provider-native `tool_calls`.
The shipped ReAct alternative collapses under role-weight prompts -- measured
`Stall detected: 3 consecutive iterations with no tool selected`, zero tool
calls -- so every live number this package carries was taken on the native arm.

### hardening (`hardening.py`)

Small-model reply recovery, all fail-CLOSED.

- `strip_model_noise(text)` -- removes `<think>` blocks and fences.
- `parse_json_payload(text)` -- prefers the cleaned text, retries the RAW text
  when the cleaned text yields no object (so a payload that lives only inside a
  `<think>` block is still recovered; a real payload outside always outranks it).
- `parse_role_output(...) -> RoleOutput` -- required keys, never `message`/`summary`.
- `coerce_worker_output(...)` -- exact-type filter down to the state's writable keys.
- `type_matches`, `as_int`, `retry` (strict allowlist: a garbled reply is NOT
  retried, it fails closed), `RETRYABLE_EXCEPTIONS`.

**Two pinned-not-endorsed behaviours** (documented in their tests): a payload
whose own STRING VALUE contains `<think>` or a fence comes back with that value
blanked; and `parse_json_payload`'s raw-text retry means a `<think>`-wrapped
draft can win when nothing else parses. Both are currently harmless; neither is
a guarantee.

## Artifacts (`artifacts.py`)

15 artifact kinds with pydantic models and Markdown (de)serializers. **Aim is
isomorphism with the source protocol's format -- same section names, same order,
same strict grammars -- not byte-identical Markdown.**

Per-plan: `state.md`, `plan.md`, `decisions.md`, `findings.md`, `findings/`,
`progress.md`, `verification.md`, `changelog.md`, `summary.md`, `checkpoints/`.
Cross-plan: `FINDINGS.md`, `DECISIONS.md`, `LESSONS.md`, `SYSTEM.md`, `INDEX.md`.

Strict grammars the validator leans on:
- `plan.md`: 11 `##` sections in exact order (`PlanSchema.SECTIONS`), positional.
- `decisions.md`: `## D-NNN | PHASE | YYYY-MM-DD` header, a `**Trade-off**:`
  field containing `at the cost of`, and 9 entry-type field sets
  (`DECISION_ENTRY_SCHEMAS`).
- `verification.md`: a criteria table, 3 mandatory additional-check rows
  (`MANDATORY_ADDITIONAL_CHECKS`), a 5-bullet verdict (`VERDICT_BULLETS`), a
  recommendation from `VERDICT_RECOMMENDATIONS`, and evidence-shape rules
  (`evidence_is_acceptable` / `REJECTED_EVIDENCE`).
- `changelog.md`: 8 pipe-delimited fields, each regex-validated
  (`parse_changelog_line`).

`PRESENTATION_CONTRACTS` carries the 6 contracts (`PC-EXPLORE`, `PC-PLAN`,
`PC-EXECUTE-STEP`, `PC-EXECUTE-LEASH`, `PC-REFLECT`, `PC-PIVOT`) as
required-field / floor data, checked by `missing_floor_fields`.

## Ownership Model (`rules.OWNERSHIP`)

16 artifacts -> the roles permitted to WRITE them. `PlanMemory.authorise` reads
this table directly, so an edit here changes what a live role can write.

Two entries look like transcription slips and are not:
1. `findings/` is `(EXPLORER, REVIEWER)` -- PIVOT's operative rules order the
   reviewer to correct stale findings in place, so removing REVIEWER would order
   a role to write a file it holds no tool for.
2. `decisions.md` has FOUR owners -- the writes are disjoint and sequenced by the
   driver (one appended entry per phase), and the `## D-NNN | PHASE | date`
   header records which phase wrote each one.

## CLI (`__main__.py`)

```bash
fsm-llm-harness new "add a retry to the uploader"        # mint + drive
fsm-llm-harness new "..." --create-only                  # mint + seed, no LLM call
fsm-llm-harness resume plans/plan-2026-07-22T101500-1a2b3c4d
fsm-llm-harness status   plans/plan-...
fsm-llm-harness validate plans/plan-... [--workspace .]
fsm-llm-harness close    plans/plan-... [--apply]
```

**Exit codes -- exactly three, and the third is a contract.**

| Code | Meaning |
|---|---|
| `0` | pass |
| `1` | a negative answer, or no answer: an `audit()` ERROR, a failed run, a missing goal, a broken install |
| `2` | **RESERVED**: a HARD `pre_step_gate` refusal |

Because `2` is reserved, a `_Parser` subclass overrides `argparse`'s `error()` to
exit `1` instead of argparse's conventional `2` -- so `fsm-llm-harness --nope`
reports 1 where `ls --nope` reports 2. The reason is asymmetric risk: a wrapper
that retries on "usage error" would otherwise silently retry past a `leash-cap`.
`--help` and `--version` are untouched and exit 0.

`status` calls the gate TWICE on purpose: once with the defaults (which is what
answers `no-plan` without constructing a `PlanDirectory`, whose `PlanMemory`
would `mkdir` the directory), then again with `expected_state` read back from
`state.md`, so a healthy plan sitting in EXPLORE is not reported `wrong-state`.

`close` opens the directory as `Role.ARCHIVIST` (the CLOSE policies act on
archivist-owned cross-plan files), is DRY-RUN unless `--apply` is passed, and
REFUSES to compress at all when `audit()` finds ERROR-severity issues.

Model resolution: `--model` > `$LLM_MODEL` > `Defaults.MODEL`.

## Testing

```bash
pytest tests/test_fsm_llm_harness/          # 1,923 tests, 10 test files
```

| File | Tests |
|---|---|
| `test_roles_and_tools.py` | 469 |
| `test_harness_agent.py` | 311 |
| `test_artifacts.py` | 273 |
| `test_hardening.py` | 258 |
| `test_plan_validator.py` | 191 |
| `test_cli.py` | 103 |
| `test_storage.py` | 115 |
| `test_fsm_definition.py` | 87 |
| `test_live_ollama.py` | 92 (17 live, gated off by default) |
| `test_extraction_cost.py` | 24 |

**Live tests are DOUBLE-gated** and auto-skip: they need both
`FSM_LLM_HARNESS_LIVE=1` and a reachable Ollama, with the env term checked FIRST
so `make test` never pays a socket timeout at collection.

```bash
FSM_LLM_HARNESS_LIVE=1 pytest tests/test_fsm_llm_harness/test_live_ollama.py
```

The live file splits FIDELITY by what each criterion is a claim about. L1/L2/L3
are claims about the HARNESS (an audit verdict, a counter, an FSM edge), so the
FSM runs live while the role workers are SCRIPTED -- they still write REAL
artifacts through a role-scoped `PlanMemory`, so `OWNERSHIP` authorises them
exactly as it would a live role. L4/L5 are claims about the MODEL, so they run
the real `build_default_worker_factory` and report raw k/n. Running everything
end-to-end would make L2 unfalsifiable: "the leash halted at exactly 2" only
means something when the executor is GUARANTEED to fail.

## Status -- what is measured, and what is not

Measured live on `ollama_chat/qwen3.5:4b` (digest `2a654d98e6fb`). Small n is
stated as k/n, not as a rate, and the bars are the ones the plans set in
advance. The two model-level rows (L4/L5) were re-measured ONCE after the
driver-assigned EXECUTE target fix, bars and assertions byte-untouched
(`MODEL_BAR=4` / `RUNS_MODEL=5`).

| Criterion | Bar | Measured |
|---|---|---|
| L1 full EXPLORE->CLOSE traverse, `audit()` zero ERRORs | pass | **3/3** |
| L2 leash halts at exactly 2 fix attempts, not resettable by an approving callback | pass | **6/6** |
| L3 REFLECT -> PIVOT -> PLAN loop-back completes | pass | **3/3** |
| L4 write tool issued AND workspace bytes on disk | >= 4/5 | **5/5 issued, 5/5 bytes -- MET** |
| L4 strict sha256 content-hash match of the requested edit | >= 4/5 | **4/5 -- MET** (react control 0/5) |
| L5 >= 3 distinct non-empty `findings/*.md` on disk from dispatches | >= 4/5 | **5/5 -- met** |
| L6 B0 end-to-end REAL workers: 3/3 runs reach >= EXECUTE, >= 1 verified write, honest halt | 3/3 | **0/3 -- NOT MET** (2 explore-cap, 1 slugless PLAN stall) |
| L6 B1, same >= EXECUTE / honest-halt clauses, verified-write TIGHTENED to EXECUTE-state workspace write | 3/3 | **0/3 -- NOT MET** (3/3 furthest=explore, slug=explore-cap, honest; zero slugless stalls) |
| L6 B2, same floor, forced-write fix live (floor sha256-identical to B1) | 3/3 | **0/3 -- NOT MET at floor, but all 3/3 now reach PLAN** (EXPLORE blocker FIXED; new PLAN-writer blocker) |
| L6 B3, same floor, scaffold+honest-approval fix live (floor sha256-identical) | 3/3 | **0/3 -- NOT MET at floor** (S2 slugless-stall FIXED: 3/3 honest plan-cap; plan-writer now writes 15-18KB; but scaffold+append refuted -- content doesn't distribute into sections) |

This is the first time L4 has MET the standing bar. (The strict row's in-test
assertion is existential -- >= 1 content-matched dispatch across both arms,
measured 4/10; the native arm's 4/5 also clears the >= 4/5 `MODEL_BAR` reading,
which is the bar reported here.)

**What moved L4: a measured structural fix, not prompt wording.** A durable
bench now exists -- `scripts/harness_bench.py`, with blocks committed under
`scripts/bench_data/l4-execute-write/{B0,B1}` (n=40/arm, 6-field manifests +
raw jsonl rows; `seed` is honored by ollama for `:4b` -- probe committed under
`scripts/bench_data/seed-probe/` -- and per-row seeds are recorded). B0
measured the wrong-ROOT defect: native EXECUTE dispatches content-matched the
requested edit **2/40**. The fix extends the driver-assigned-target pattern to
EXECUTE: the driver reads plan.md's Files To Modify and names the exact target
path + tool in the dispatch. B1, same manifest: **40/40** (Fisher p=1.6e-20).
The ReAct control arm measured 0/40 in both blocks -- its failure mode is
upstream of target selection. Caveat: the content-match/content-hash metric
shares vocabulary with the fix's own prompt text ("retry"/"backoff" appear in
the task prose) -- treat a PASS as target-selection compliance, not proven
code correctness; `content_matched_ast` (AST-structural, vocabulary-decoupled,
additive) exists for future blocks.

**L6 is the open one, and it is reported as it measured -- twice.**
`TestL6EndToEndRealWorkers` is the package's first graded end-to-end criterion
on REAL role workers (n=3 per block, disk-derived rubric vectors, DENY-default
disk-bound approval stub). Block B0 (frozen under
`scripts/bench_data/l6-e2e/B0/`) measured **0/3, NOT MET**: two honest
explore-cap halts, and one run that reached PLAN and stalled SLUGLESSLY after
an empty plan-writer reply -- B0's verified-write clause held 3/3, but it
scored True off EXPLORE findings writes alone, so it was near-vacuous. Both
defects were fixed structurally (PLAN redispatch budget with exhaustion
halting on the honest `plan-cap` slug; driver-named `plan.md` deliverable
line) and block B1 was pre-registered with the verified-write clause
TIGHTENED to require an EXECUTE-state WORKSPACE write -- `>= EXECUTE` and
honest-halt clauses byte-identical to B0's, n=3, same model digest. B1
measured **0/3, NOT MET**, in a different and cleaner shape: every run
`furthest_state=explore`, `halt_slug=explore-cap`, `honest_halt=true` (wall
clocks 344.6/298.5/357.9 s vs the 1800 s ceiling; findings on disk 0/0/2).
Zero slugless stalls in B1's rows. Scope that claim precisely: the
redispatch budget covers worker-failure replies ONLY -- it retries when
`result is None or not result.success` and halts on `plan-cap` when spent
-- so the B0 worker-failure stall shape is structurally closed. The residual
`success=True`-but-empty-plan.md slugless PLAN stall (a reply that skipped the
budget, fell through to approval, and stalled slug=None on the denial) is NOW
closed too, this iteration (D-005): a disk-derived empty-`plan.md` check
(`_plan_has_content`, `tools.has_bytes` over the driver's own uncapped reader)
folded into the SAME budget condition consumes the budget for that shape, so
exhaustion halts on the honest `plan-cap` slug rather than slug=None -- no
generic `STALL` slug was minted (predecessor D-003 respected). Offline-verified,
still live-unexercised. No B1 run reached PLAN, so the redispatch budget and deliverable line
are offline-verified (unit-proven) but live-unexercised, and the tightened
verified-write clause measured false 3/3 because nothing reached EXECUTE.
The end-to-end blocker was probed by a dedicated L7 A/B
(`l7-explore-coldstart/B0`, committed under `scripts/bench_data/`): a SINGLE
cold-start EXPLORE dispatch over a bare `mkdir` scored **bare 5/12 vs seeded
7/12** (Fisher two-sided p=0.6843 -- the zero-byte protocol-skeleton cold-start
lever is **NOT VALIDATED**; a positive-but-non-significant delta). What this
measures precisely: a single first EXPLORE dispatch over a bare dir is NOT
impossible (5/12), so first-dispatch impossibility is ruled out as the
mechanism. It does NOT measure the multi-dispatch traverse -- all 24 L7 rows
carry `assigned_topic="problem-scope"` (one topic, one dispatch, no redispatch
loop), and one dispatch's `bytes_on_disk` success is NOT the same event as
clearing the EXPLORE 3-findings gate, so the 5/12-vs-0/3 magnitudes are not
directly comparable. The LEADING SUCCESSOR HYPOTHESIS -- consistent with, but
not established by, this block -- is that L6's 0/3 is a multi-dispatch
redispatch-loop / structured-output-parse (`objects=0`, `empty-reply`)
failure rather than a first-dispatch one; that mechanism remains UNMEASURED.
(The seeded arm's `empty-reply` count shifted 1->3, but at n=12 that is
within noise and is not read as a signal; the NOT-VALIDATED verdict rests on
the primary p=0.6843 alone.) L6's honest 0/3, every row naming its blocking
state and slug, is the package's end-to-end status.

**L8 (`l8-explore-loop/B0`, NEW this iteration) MEASURES the traverse mechanism
the L7 hypothesis only pointed to -- and REFUTES parse-collapse as the primary
driver.** The named successor -- a single-state EXPLORE redispatch-LOOP bench
(not L7's single dispatch) instrumented with a per-tool-call spy on
`ToolRegistry.execute` -- has now been built and RUN once (n=10 loops, 100
dispatches, `ollama_chat/qwen3.5:4b` digest `2a654d98e6fb`, one look, committed
under `scripts/bench_data/l8-explore-loop/B0/` with a tracked `PRE_REGISTRATION.md`
fixing n + the mechanism vocabulary + the W1->W2 decision rule before the run).
A deterministic classifier partitioned every FAILED dispatch into exactly one
of six buckets (partition hard-gate passed). Pooled result: `gate_cleared`
**0/10** (Wilson95 [0.000, 0.278] -- no run reached PLAN, consistent with L6),
and of 89 failed dispatches **`never-called` (family i) = 75 (84%)**,
**`empty-reply` (family iii) = 14 (16%)**, `wrong-root` (ii) / `accepted-no-bytes`
/ `unparseable` = **0**. The per-tool-call trace resolves what the
dispatch-boundary log could not: the dominant `reason=unverified-write objects=1`
signature (a PARSEABLE answer claiming work) has **`write_calls=0`** -- the
explorer issues only read/list tools (with heavy wrong-root READ churn,
`read_file`<->`read_plan_file`) and NEVER calls a write tool. So the dominant
mechanism is **(i) never-called-a-write-tool**, measured, NOT the
`empty-reply`/parse-collapse the leading hypothesis predicted (which is real but
secondary at 16%). Per the pre-registered rule this AIMS the single W2 follow-on
at a driver-side FORCED-WRITE EXPLORE target (mirroring the EXECUTE 2/40->40/40
structural fix), NOT `response_format`-primary (which would target only the 16%
tail). The forced-write fix + a fresh L6 B2 are the named successor, a LATER
iteration -- NOT executed here (D-004). A secondary, UNMEASURED contributing
hypothesis the trace surfaces: the explorer may burn its 14-turn budget on
failed wrong-root READ calls before ever reaching a write. Scope of the L8
claim, precisely: it is measured on a SINGLE seeded exploration workspace/goal
(the retry-backoff-uploader fixture) with 3 rotating sub-topics, and n is ~10
runs (89 dispatches clustered within them, empty-reply concentrated in runs
3/4/6), not 89 independent trials. The DIRECTION (never-called dominant) is
robust to classifier precedence -- `empty-reply` is ranked ABOVE `never-called`
(the ordering most generous to the parse-collapse hypothesis) and never-called
still wins, so the 16% is the MAXIMAL empty-reply attribution, not a floor; and
robust at the run level too (8/10 runs never-called-dominant).

**L8 B1 + L6 B2 (NEW this iteration) VALIDATE the forced-write fix -- the
EXPLORE blocker of the last four iterations is FIXED, and the wall MOVED a full
state to PLAN.** The fix (D-003) is an additive, default-off forced-write
finalization: `AgentConfig.force_final_tool` plus a post-loop forced-`tool_choice`
turn in `native_fc.py` (it mirrors the D-002 repair turn -- `tools=`+`tool_choice`,
never `response_format=`; it fires at most once, after the read loop), wired
EXPLORE-only in `roles.py` (`Role.EXPLORER` -> `write_plan_file`). Crucially the
MODEL issues the real `write_plan_file` call (recorded in the trace);
`_verified_writes` stays honest, there is NO driver salvage, and the founding
"a confident sentence cannot open a gate" ethos is INTACT. **L8 B1**
(`l8-explore-loop/B1`, n=10, one look, committed): `gate_cleared`
**0/10 -> 9/10** (Wilson95 [0.596, 0.982]; Fisher two-sided p=**0.00012** vs
B0); failed dispatches **89/100 -> 8/37**; `empty-reply` (family iii)
**14 -> 0**; residual `never-called` **75 -> 8** absolute -- the forced write
converts the 84%-never-called collapse into gate clearance. **L6 B2**
(`l6-e2e/B2`, n=3, one look, committed; floor sha256 verified IDENTICAL to B1
`cbeeb6aa...` before AND after the block-constant edit): **all 3/3 runs now
reach PLAN** (`furthest_state=plan`) -- vs the rigorous adjacent baseline **B1,
which was 0/3 reaching PLAN, every run stuck at EXPLORE** (`furthest=explore`,
`explore-cap`); B0 was 0/3 at the >= EXECUTE floor too but 1/3 DID reach PLAN
(run 3, an empty-`plan.md` slugless stall) -- each writing **3 real,
substantive findings** (`problem-scope.md` 1262, `affected-files.md` 1304,
`constraints-and-patterns.md` 1809 bytes) via the forced write. BUT the e2e
FLOOR (>= EXECUTE + verified_write + honest_halt) is still **0/3 -- the floor
test FAILS as measured** -- because a NEW, deeper blocker emerged at PLAN: the
4b plan-writer cannot emit a valid 11-section `plan.md`. Runs 2 & 3: the
plan-writer returned an EMPTY `plan.md`, consumed the redispatch budget
(`plan_redispatches=3=MAX_PLAN_REDISPATCHES`), and halted HONESTLY on
`plan-cap` (`honest_halt=true`). Run 1: a NON-EMPTY (4153 bytes) but
SCHEMA-INVALID `plan.md` (missing all 11 `## `-sections; `PlanDoc` validation
failed), which the DENY-default disk-bound approval correctly REFUSED, after
which the run stalled `slug=None` (`honest_halt=false`) -- a slugless stall.
This is the biggest capability advance in the package's history AND an honest
floor FAIL: the fix did exactly what it was built to do (the p=0.00012 L8 shift
proves it), the founding e2e end-goal is NOT YET achieved, and the wall is now a
DIFFERENT failure class -- structured-output conformance, not
never-called-a-write-tool (run 1 proves forcing the write alone yields an
invalid plan.md, not a valid one). **W3 gets its first live evidence.**
`MAX_PLAN_REDISPATCHES=3` had ZERO live evidence before this block; runs 2 & 3
both show `plan_redispatches=3` (== the cap) with `plan-cap` + `honest_halt=true`
-- the PLAN redispatch budget's worker-failure branch is now live-exercised 2/3
(run 1 wrote a plan.md, `plan_redispatches=0`, and stalled on the approval-denial
path). The falsifier is NOT refuted. Two named successors, deferred (NOT fixed
here): **S1** -- the PLAN-writer valid-`plan.md` blocker (the new dominant e2e
wall; needs its own investigation: a `response_format` plan schema, a plan.md
scaffold the model fills, or a looser accepted schema); **S2** -- the
non-empty-but-schema-invalid-`plan.md` slugless stall (run 1): predecessor
D-005 (`plan-2026-07-22T212329-16de43da`) closed the EMPTY-plan.md slugless
stall via `_plan_has_content` (a BYTES check), but a non-empty-but-INVALID
plan.md passes the bytes check, is denied at approval, and stalls `slug=None`;
`_plan_has_content` should check plan VALIDITY (parseable `PlanDoc`), not just
bytes.

**L6 B3 (NEW this iteration) FIXES S2 and VALIDATES the honest-approval /
aligned-gates machinery LIVE, but REFUTES this iteration's chosen scaffold+append
S1 mechanism -- the floor stays 0/3 and the wall advances from "empty/invalid
plan" to "content-not-distributed".** The user-chosen fix (D-001) is a
scaffold + honest-approval design: at PLAN entry the driver seeds `plan.md` with
the 11 `PlanSchema.SECTIONS` headers (structure only), the plan-writer fills
sections via append (`force_final_tool=append_plan_file` for PLAN + a
deliverable-line instruction), and `_plan_has_content` + the approval gate now
share ONE bar (`_plan_is_approvable` = valid `PlanDoc` AND every section
non-placeholder), closing the slugless-stall gap. **L6 B3** (`l6-e2e/B3`, n=3,
one look, committed; floor sha256 verified IDENTICAL to B1/B2 `cbeeb6aa...`
before AND after; honest-approval confound DECLARED in `PRE_REGISTRATION_B3.md`)
measured **0/3 at the floor -- the floor test FAILS as measured**, all 3 runs
`furthest_state=plan`, `verified_write=false`. But the shape changed decisively.
(1) **S2 is FIXED**: all 3 runs halt on the honest `plan-cap` slug with
`honest_halt=true`; the slugless `slug=None` stall (B2 run 1) NO LONGER OCCURS.
(2) The plan-writer now writes **15-18 KB** of real content (`plan_md_bytes`
17122 / 15592 / 18455, up from B2's 0 / 0 / 4153) -- 4b IS capable of producing
substantial plan content. (3) BUT the **scaffold+append mechanism is REFUTED**:
`append_plan_file` appends to the FILE END, so content does not distribute into
the 11 ordered sections -- run 1's plan.md parses as a valid `PlanDoc` (no
`plan-section` ERROR) yet has **10 placeholder sections** (all ~17 KB
concentrated in one; audit `plan-section` WARNING x10), and runs 2/3 hit a
`plan-section` ERROR (the appended content included its own `## ` headers ->
duplicate sections -> invalid). Appending cannot fill in-place sections. (4) The
**aligned strict bar held the line LIVE (via the BUDGET gate, not the approval
stub)**: run 1's valid-but-1-section plan made `_plan_has_content` return False
(through the shared `_plan_is_approvable` = all-non-placeholder), so it consumed
the redispatch budget and halted on the honest `plan-cap` -- it was NEVER
approved. Under a loose not-all-placeholder bar that hollow plan (1 real + 10
empty) would have been substantive → reached approval → passed to EXECUTE; the
aligned strict bar prevented exactly that hollow gate. IMPORTANT precision: all
3 B3 rows carry `approvals: []` -- `DiskEvidenceApprovals` was NEVER invoked live
(approval is only reachable once `_plan_has_content` is True), so the honest
APPROVAL stub is UNIT-validated only; the gate that held the hollow plan out live
was the aligned BUDGET gate. A corollary: the declared honest-approval B2↔B3
confound turned out INERT (no B3 plan was substantive enough to reach approval),
so B2↔B3 comparability is cleaner than declared. The ethos held live via the
shared predicate. Hedge honestly: the S2 honesty fix and the
aligned-gates machinery are permanent, VALIDATED wins that ship regardless; the
S1 CAPABILITY goal (a full run reaching >= EXECUTE) is NOT met; the wall
advanced from "empty/invalid plan" to "content authored but not distributed
into the right headers." Diagnosis confidence is STRONG but not byte-confirmed
(the L6 plan dirs are pytest `tmp_path`, deleted post-session, so run 1's raw
17 KB plan.md is gone; the append-to-end conclusion rests on the audit signature
-- valid-`PlanDoc` + 10 placeholder + 17 KB = concentrated in one section for
run 1, `plan-section` ERROR = duplicate headers for runs 2/3 -- which is
decisive but derived; a future bench should retain FAILED plan.md artifacts).
**Aimed successor (grounded, not guessed)**: the model produces ample content
but cannot place it under the right headers by appending, so the next mechanism
must DISTRIBUTE content into sections. Strongest = the **`response_format`
structured plan** (EXPLORE candidate B, NOT chosen this iteration): the model
authors 11 fields, the driver RENDERS them into correctly-structured Markdown
(distribution by construction, no append-to-end flaw). Secondary = seed the
scaffold as a readable TEMPLATE and have the model OVERWRITE the full plan
(riskier -- reintroduces free-text-structure risk).

Known gaps, standing after L6 B3 (the scaffold+honest-approval fix):
- **#1 gap -- DISTRIBUTE plan content into the 11 sections**: L6 B3 proved 4b
  CAN write substantial plan content (15-18 KB) but the scaffold+append
  mechanism cannot place it under the right headers -- `append_plan_file`
  appends to the file END, so content concentrates in one section (valid but
  hollow) or duplicates headers (invalid). The scaffold seed + append steering
  are SHIPPED but MEASURED-INEFFECTIVE for this goal (the S2/honesty fixes they
  came bundled with DO stand). The aimed fix is the **`response_format`
  structured plan** (the model authors 11 fields, the driver renders correctly
  structured Markdown -- distribution by construction); a secondary is a
  template-overwrite scaffold. UNMEASURED how far a fix here would carry a run
  past PLAN toward the >= EXECUTE floor.
- **The slugless PLAN stall (S2) -- CLOSED/FIXED this iteration (L6 B3, 3/3)**:
  predecessor D-005 closed the EMPTY-plan.md slugless stall via a BYTES-only
  `_plan_has_content`, but a non-empty-but-INVALID plan.md (B2 run 1) still
  passed the bytes check, was denied at approval, and stalled `slug=None`. The
  aligned `_plan_is_approvable` bar (valid `PlanDoc` AND all-non-placeholder,
  shared by `_plan_has_content` and the approval stub) now consumes the
  redispatch budget for that shape too, so exhaustion halts on the honest
  `plan-cap` slug -- L6 B3 shows 3/3 `plan-cap` + `honest_halt=true`, zero
  slugless stalls. No generic `STALL` slug was minted (predecessor D-003
  respected).
- **PLAN redispatch budget**: `MAX_PLAN_REDISPATCHES=3` now has its FIRST live
  evidence (L6 B2 runs 2/3: `plan_redispatches=3=MAX` + `plan-cap` +
  `honest_halt=true`) -- the worker-failure branch is live-exercised, NOT
  refuted. The `success=True`-but-EMPTY-plan.md slugless-stall residual is CLOSED
  (D-005); the non-empty-but-invalid variant is S2 (above). Still unbudgeted:
  REFLECT/PIVOT/CLOSE worker-failure stalls (no generic `STALL` slug is minted,
  per predecessor D-003).
- **Bare `/workspace` sentinel** -- CLOSED this iteration (D-004): a bare
  `/workspace` (and bare `/plan`) sentinel now maps to the confinement root
  inside the single `_strip_root_sentinel` chokepoint instead of being refused;
  `/` alone still raises.
- **Wrong-root reads**: a live explorer read a plan-dir findings path through
  the workspace-rooted `read_file` and was rejected -- the tool surface
  separates the two roots but the model conflated them.

The test suite itself has been audited adversarially, by execution: 5/5
load-bearing guard mutations (leash-cap boundary, writable-key allowlist,
empty-file gate counting, ownership deny branch, live-gate short-circuit) each
flipped tests red in a scratch copy (93 red total), and `test_cli.py`'s
exit-code 0/1/2 contract close-read verdict was CLEAN.

Offline, the package is green: 1,923 tests, `ruff` clean, `mypy` 0 errors.

**Not claimed**: that the harness is production-ready, or that a 4B model
drives it unattended to a useful result -- the L6 floor is still **0/3 at
>= EXECUTE**, measured FOUR times now (B0, B1, B2, B3), which REINFORCES this
claim's absence, it does not soften it. What iteration 6 DID advance, honestly:
the slugless PLAN stall (S2) is FIXED (L6 B3 3/3 halt on the honest `plan-cap`
slug, zero slugless stalls), and 4b was confirmed to write substantial plan
content (15-18 KB). But the plan-writer still cannot produce an APPROVABLE
(all-sections-filled) plan -- the chosen scaffold+append mechanism was REFUTED
(append-to-end concentrates content in one section or duplicates headers, it
does not distribute into the 11 sections), so the wall advanced from
"empty/invalid plan" to "content-not-distributed" but was not cleared. Through
all of this the gates stayed mechanical: the honest approval DENIES a hollow
(1-real-section) plan (validated live in B3 run 1), the MODEL performs the write
(no driver salvage), the gate reads the filesystem, and a confident sentence
still cannot open one.

## Exceptions

```
FSMError
└── HarnessError
    ├── HarnessArtifactError(artifact, message, cause=None)  # unreadable/unparseable/over-cap
    ├── HarnessOwnershipError(artifact, role, owner)         # OWNERSHIP denies this role the write
    ├── HarnessReentrancyError(role)                         # a worker re-entered the driver
    └── HarnessConfinementError(path, root)                  # a path escaped its root
```

## Conventions specific to this package

- **Constants live in `constants.py`**; `rules.py` owns protocol CONTENT (prose,
  ownership, topics) and `fsm_definition.py` owns only graph shape and gate logic.
- **One literal `__all__`** in `__init__.py` -- no dynamic extend/append.
- **Anchored decisions**: non-obvious code carries a
  `# DECISION plan-<full-plan-id>/D-NNN` comment stating what NOT to do and why.
  The full plan-id keeps the `THHMMSS` segment; the commit-tag form drops it.
  Writing the tag form into an anchor makes it invisible to the anchor audit.
- **Interface contracts** on shared helpers name their call sites, so a reader
  can see at the definition whether a change is local.
- **Evidence over testimony.** If a number can be derived from the filesystem,
  derive it. If it can only be claimed, treat it as advisory and record it as
  such.
