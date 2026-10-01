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
  `native_fsm` (B2, plan 944e2692 D-049) is the same `native=True` dispatch
  on the code where native_fc runs as an FSM on core's completion state
  instead of its private loop; a new label because the code changed, paired
  with `B1/native`.
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
  - B2 (plan 944e2692 D-010, re-recorded in D-049 before row 1):
    `fsm_toolcall` is `create_agent("native_fc", tools, config=...)`, B0
    `native_fc`'s construction and limits on the code where
    `NativeFunctionCallingReactAgent` is an FSM driven by core's completion
    state. Same 38 tasks, `tasks_sha256`, 3 trials and model digest as B0;
    meter "2". Paired with `B0/native_fc`.

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

Blocks run after B1 also carry the request disclosures `llm_request` (the
first request's kwargs with values, secrets redacted) and `first_request`
(digests of its system text, other messages, tools and settings, plus
`n_requests` and `run_error`), read at core's send binding by one EXECUTE
dispatch whose first request is refused (`request_disclosure`; same rules as
the agents-react disclosures below). Recorded B0/B1 manifests predate them.
Their summaries carry `run: {git_commit, git_dirty}` read at run start.

`register` (B2 onwards) writes the manifest alone, so it is committed before
dispatch 1; `run` then keeps it and refuses on drift in `n_preregistered`,
`seed`, `model`, `prompt_bytes_sha256`, `tool_surface`, `fixture_hash`,
`arm`, `llm_request`, `first_request` or the served model digest (values
compared as JSON text). B0/B1 manifests were written by `run` itself.

```
.venv/bin/python scripts/harness_bench.py register --bench-id l4-execute-write \
    --block B2 --arm native_fsm --n 40 --seed 20260722000
```

Only the FIRST request of the dispatch is disclosed: the capture stub
refuses it, so request 2 (the tool-result turns, the assistant `tool_calls`
echo, a forced or repair turn) is never built. A change in the follow-up
turns moves no manifest field and is disclosed only by `git_commit` and the
summary's `run.git_commit` (review pass 12 W1 of plan 944e2692).

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
.venv/bin/python scripts/harness_bench.py report l4-execute-write \
    --blocks B1 B2 --pair B2/native_fsm:B1/native
```

Two blocks compare each arm label present in both. `--pair
BLOCK/ARM:BLOCK/ARM` compares two arms whose labels differ (a B2
`native_fsm` against a B1 `native`): it prints every differing manifest field
first (values compared as JSON text, a field one side lacks as "not
recorded"), then Fisher two-sided per metric both blocks carry, only when the
two manifests pin the same model digest.

A B2-vs-B1 pair (`B2/native_fsm:B1/native`) must disclose, beyond the
printed lines (B1's manifest lacks `llm_request` and `first_request`, so the
pair prints them as an opaque "not recorded"):

- per-request timeout: B1 sent none; every B2 request carries
  `timeout: 120.0` (core's `LiteLLMInterface` default);
- tool-schema bytes: step 14 of plan 944e2692 (exact `args_model` schemas)
  added `"default": "."` to the `path` parameter of `list_dir`,
  `grep_files` and `list_plan_dir` (+48 bytes of `tools`); B1's tool bytes
  were not recorded, so the pair cannot show it;
- request settings: B2 records `temperature` 0.3, `max_tokens` 2000,
  `reasoning_effort: "none"`, `tool_choice: "auto"` and the row `seed`; B1
  recorded none of them;
- the code path: B1 ran native_fc's private loop at 2a89226 (and native_fc
  changed in 11 commits between 2a89226 and agents B0's 73d7a6c); B2 runs
  the FSM on core's completion state. Core's request building for native
  turns reproduces e1f63a9's bytes in every key except `timeout` and
  `max_retries` (the golden test of plan 944e2692 step 1), not 2a89226's;
- only the first request is digested (above): follow-up turns are disclosed
  by the commit hash alone;
- the tool trace: the row's `tool_trace` comes from the live test's tool
  spy, which since plan 944e2692 D-034 forwards `gated` (native never
  passes it; the react arm did not run its tools under the old spy);
  native_fc trace entries now also carry `tool_status`;
- harness rows have no LLM call meter (only `tool_calls` from the spy), so
  there is no meter difference to read; the agents-react pair below has one;
- the git commit and date (the model digest must still match).

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

Request disclosures (blocks registered after B1; B0/B1 manifests predate them
and are never edited, so `report` prints them as "not recorded"). They are
computed offline at `register`/`run` by running the arm for every task on the
trial interface with core's one send binding (`fsm_llm.llm.completion`)
replaced by a stub that records the FINAL request and refuses it
(`harness_bench.captured_wire`; no provider call). So everything core adds
on the way out counts: Ollama preparation (`reasoning_effort`, `/nothink` on
the last user turn, a schema echo), the temperature rules, `max_tokens`,
timeout, retries and any extra provider kwarg (D-040 of plan 944e2692):

| Field | Pins |
|---|---|
| `llm_request` | the first task's first request kwargs except `messages` and `tools`, WITH values (`model`, `temperature`, `max_tokens`, `timeout`, `max_retries`, `reasoning_effort`, `tool_choice`, `response_format`, a `seed`, an `api_base`, ...); a secret-looking entry keeps its key with the value `"<redacted>"` (core's `is_forbidden_context_entry` via `redact_secret_entries`) |
| `agent_class` | `module.qualname` of the agent the arm factory builds (a rewrite under the same name shows only in `git_commit` and the request digests) |
| `run_cap` | the agent's core step ceiling for `max_iterations` (`max_steps`, `formula`) and `max_seconds` |
| `tool_schemas_sha256` | per task, sha256 of `json.dumps(registry.get_json_schemas())` with key order kept (never sorted): the exact `tools=` bytes a native arm sends and the one schema a prompt-mode arm renders |
| `first_request` | per task, digests of the first FINAL request: `system_sha256` (system texts), `user_sha256` (every other message, role and content), `tools_sha256` (the `tools` bytes or null), `settings_sha256` (the `llm_request` shape of that request), plus `run_error`. The `Today's date: YYYY-MM-DD` line and the task text are masked (`<DATE>`, `<TASK>`; `tasks_sha256` pins the task text), so the digests move only with the code, not with the day or the time zone |

`run` refuses a registered block whose `llm_request`, `agent_class`,
`run_cap`, `tool_schemas_sha256` or `first_request` changed since
registration; values are compared as JSON text, so a type change (`120` vs
`120.0`, `0` vs `false`) is drift. The commit that actually ran is recorded
at run start in the summary as `run: {git_commit, git_dirty}` (the
manifest's `git_commit` is the registration commit).

Only the FIRST request per task is digested: the capture stub refuses it, so
request 2 and later (tool-result messages, the assistant `tool_calls` echo,
forced, repair and conclude turns) are never built. A change confined to
later turns moves no manifest field and `run` accepts it; such a change is
disclosed only by the commit hashes (`git_commit`, the summary's
`run.git_commit`). Read a pair across code versions with that in view: for
B2-vs-B0 the private-loop-vs-core difference lives mostly in those turns
(review pass 12 W1 of plan 944e2692).

Meter (`wrapper_version`): "1" (B0, B1) patched the litellm completion
bindings and counted every provider call through them; "2" (every later
block) reads core's own counters, `LiteLLMInterface.usage()`, off one
interface per trial that the bench builds with the arm's settings and
injects as `llm_interface=`. Row fields and meanings are the same, so
`report` recounts both; `meter_parity` runs tasks under both meters at once
and must show them equal before a "2" block is registered (D-004 of plan
944e2692).

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
.venv/bin/python scripts/agents_bench.py report agents-react \
    --blocks B0 B2 --pair B2/fsm_toolcall:B0/native_fc
```

`--pair A:B` (each side `ARM` or `BLOCK/ARM`) first prints every manifest
field that differs between the two blocks (nested keys as dotted paths, a
field one side lacks as "not recorded", values compared as JSON text so a
type change prints), then Fisher two-sided on
first-trial pass@1 and pass^k, only when both manifests pin the same model
digest. A pair across code versions is read with those differences in view.
A B2-vs-B0 pair must disclose, beyond the printed lines: meter "2" vs "1";
per-request timeout 120 s vs none; tool-schema bytes (step 14 of plan
944e2692 added `"additionalProperties": true` to `order_total.quantities` and
reordered `list_stats.numbers` to `{"items", "type"}`, tasks ty-order,
ch-order-local, ty-mean, ty-range); the code path (B0 native_fc's private
loop at 73d7a6c vs the core completion state) and its step ceiling; provider
errors now wrapped as `AgentError`; the git commit and date (the model
digest must still match). Also: only the first request per task is digested
(above), so the follow-up turns are disclosed by the commit alone; core's
Ollama preparation of native requests (`reasoning_effort`, `/nothink` on
the last user turn, temperature, `max_tokens`) is byte-identical to
e1f63a9's private loop except `timeout`/`max_retries` (golden test of plan
944e2692 step 1), while B0 ran at 73d7a6c; native_fc trace entries now
carry a `tool_status` key (agent-side, not sent to the model); a
`register`-time `llm_request` records `tool_choice: "auto"`, which B0 did
not record. Rules and the full list: decisions.md D-049 of plan 944e2692.
