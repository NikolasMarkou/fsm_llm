# bench_data -- committed harness bench artifacts

Raw, append-only evidence produced by `scripts/harness_bench.py`. This
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
├── l6-e2e/                   # per-run e2e rubric vectors (plan step 7)
└── seed-probe/               # probe-seed records (plan step 2)
```

Arms: `native` (`native_function_calling=True`, the shipped default) and
`react` (the opt-in control).

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
