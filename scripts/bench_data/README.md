# bench_data -- committed harness bench artifacts

Raw, append-only evidence produced by `scripts/harness_bench.py` and
`scripts/agents_bench.py`. This
directory is GIT-TRACKED on purpose: the predecessor plan's bench scripts and
jsonl traces lived in gitignored scratch directories and are gone, so none of
its live numbers can be diffed or recomputed (plans/LESSONS.md [I:4]). Nothing
under here may be moved to a gitignored path.

## Layout

```
bench_data/
├── <bench-id>/               # e.g. l4-execute-write
│   └── <block>/              # B0, B1, ... one pre-registered block each
│       ├── manifest_<arm>.json   # written BEFORE the first dispatch
│       ├── rows_<arm>.jsonl      # one raw row per dispatch, append-only
│       └── summary_<arm>.json    # k/n + Wilson CI, recounted from rows
├── l6-e2e/                   # per-run e2e rubric vectors, graded not binary
│   └── <block>/              # B0, B1, ... one pre-registered block each;
│                             #   manifest.json + rows.jsonl (no arm suffix:
│                             #   L6 is always native and test-embedded)
├── l7-explore-coldstart/     # EXPLORE cold-start A/B, one dispatch per row
│   └── <block>/              # B0, ... manifest_<arm>.json + rows_<arm>.jsonl
│                             #   + summary_<arm>.json, arm in {bare, seeded}
├── agents-react/             # agents_bench.py: ground-truth tool-loop tasks,
│   └── <block>/              #   one row per (task, trial); manifest_<arm>.json
│                             #   + rows_<arm>.jsonl + summary_<arm>.json
└── seed-probe/               # probe-seed records (plan step 2)
```

Arms carry TWO different meanings depending on the bench, and they must not be
read as the same axis:

- `l4-execute-write`: `native` (`native_function_calling=True`, the shipped
  default) vs `react` (the opt-in control) -- these are AGENT shapes.
- `l7-explore-coldstart`: `bare` (a plain `mkdir` plan directory, L6's own
  population) vs `seeded` (the same `mkdir` plus the product's
  `PlanDirectory.seed_protocol_skeleton()`) -- these are PLAN-DIRECTORY shapes,
  and BOTH run `native_function_calling=True`. The independent variable is the
  on-disk population, not the agent.
- `agents-react`: agent engines on the same task set. A later block uses a
  new arm label for changed code, never an old one; `report` reads arm labels
  from the `rows_<arm>.jsonl` file names, so rows of an arm retired from
  `ARMS` (or whose code was deleted) still recount.
  - B0 (d4b1626 agents code; arms retired from `ARMS`): `legacy` is
    `create_agent("react", tools, config=...)` (the FSM-driven ReactAgent),
    `native_fc` is `NativeFunctionCallingReactAgent`.
  - B1 (plan 07ad3f8c D-052): `fsm_advance` is the same
    `create_agent("react", tools, config=...)` construction on the migrated
    code, where core's `advance`/`run_until_terminal` steps the agent FSM.
    Same tasks, trials, limits, model digest and meter as B0; paired with
    `B0/legacy` under the pass rule written in D-052 before the run.

## Pre-registration rule (D-002)

- A block is pre-registered: its manifest is written before dispatch 1.
- A block runs ONCE, at its fixed n. No interim looks, no re-rolls; `run`
  refuses a block that already carries rows.
- An interrupted block is committed as-is with `"status": "aborted"` in its
  summary, then ONE fresh complete block may be pre-registered. No decision is
  ever taken on a partial block.
- Any additional block requires a NEW manifest plus a new decisions.md entry.

## The 6 manifest fields (all REQUIRED)

| Field | Pins |
|---|---|
| `prompt_bytes_sha256` | rendered EXECUTE system+task prompts (fixed placeholder paths) |
| `tool_surface` | worker-factory kwargs + the exact tool names the dispatch holds |
| `fixture_hash` | sha256 of EXECUTE_PLAN_MD + EXECUTE_STATE_MD + SEED_FILES |
| `model_digest` | the digest Ollama actually served at block start (queried, never assumed) |
| `arm` | `native` boolean + display label |
| `git_commit` | the source commit the block ran against |

A summary without its manifest is NOT evidence; the writer refuses to emit
one. `report` refuses to Fisher-compare two blocks whose manifests pin
different model digests.

## Row schema (rows_<arm>.jsonl)

`bench_id`, `block`, `arm` (display), `native` (bool), `run`, `ts`,
`elapsed_s`, `tool_calls`, `write_tool_issued`, `bytes_on_disk`,
`content_matched` (sha256-based, never stat), `success`,
`tool_trace` (`[{tool, ok}]`), `seed` (int or null).

Recompute everything from the raw rows:

```
.venv/bin/python scripts/harness_bench.py report <bench-id>
```

## agents-react (scripts/agents_bench.py)

38 deterministic tasks in 7 categories (`single_tool`, `multi_step_chain`,
`no_tool_needed`, `error_recovery`, `typed_args`, `distractor_tools`,
`unanswerable`) with pure in-file tools and non-LLM graders; each task runs
for `trials` (3) fresh trials, trial-major. `list-tasks --verify` grades every
reference answer. `register` writes a manifest alone, so it can be committed
before row 1; `run` then checks it (task hash, trials, model, limits, wrapper,
served digest) and refuses on drift.

Manifest: the six fields above (`fixture_hash` equals `tasks_sha256`,
`prompt_bytes_sha256` hashes the task prompts, `tool_surface` holds the arm,
limits and per-task tool names, `arm` is `{name, factory}`) plus
`tasks_sha256` (sha256 of the file region between the `BEGIN TASKS` and
`END TASKS` markers: tools, tasks, graders, reference solvers; arms and limits
sit outside it), `trials`, `temperature`, `limits`, `wrapper_version`,
`n_tasks`, `n_preregistered`, `order`, `model`.

Row schema (rows_<arm>.jsonl): `task_id`, `category`, `arm`, `trial`,
`correct` (grader on `answer`), `success`, `stop_reason` (`timeout`/`error`
when the run raised), `iterations`, `tool_calls`, `tools_used`, `llm_calls`,
`llm_errors`, `usage_missing`, `prompt_tokens`, `completion_tokens`,
`total_tokens` (a bench-local wrapper on `litellm.completion`,
`litellm.acompletion` and `fsm_llm.llm.completion`; B0 also wrapped
`fsm_llm.classification.completion`, which no longer exists because the
classifier sends through `fsm_llm.llm.completion`: one count per provider
call either way, `wrapper_version` "1"), `latency_s`, `error`, `answer`
(first 500 chars), `bench_id`, `block`, `ts`.

Summary `metrics` (all recounted by `report`): first-trial pass@1 (primary,
one independent unit per task) with Wilson, mean over trials, pass^k, the
success x correct cross-tab, error count, llm calls and tokens mean/median,
tool calls mean, nearest-rank p50/p95 latency, stop_reason histogram, and a
per-category table.

```
.venv/bin/python scripts/agents_bench.py report agents-react \
    --blocks B0 B1 --pair B1/fsm_advance:B0/legacy
```

`--pair A:B` (each side `ARM` or `BLOCK/ARM`) prints Fisher two-sided on
first-trial pass@1 and pass^k, only when both manifests pin the same model
digest.
