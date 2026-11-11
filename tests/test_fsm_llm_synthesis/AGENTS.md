# test_fsm_llm_synthesis

Path: `tests/test_fsm_llm_synthesis`
Purpose: Offline pytest suite for `fsm_llm.synthesis` (gates G1 to G7, library, runner, tools, traces, diagnosis, patch and repair) and for `scripts/synthesis_bench.py`. Counts live in the root `AGENTS.md`, not here.

## Scope

Covers `fsm_llm.synthesis` (`policy`, `gates`, `dryrun`, `simulation`, `synthesizer`, `prompts`, `library`, `runner`, `tools`, `traces`, `diagnose`, `patch`, `repair`, `constants`, `exceptions`) and the bench script. Core pieces it relies on are tested elsewhere: `State.synthesize` and `SynthesizeConfig` in `tests/test_fsm_llm/test_synthesize_field.py`, `fsm_llm.flow_analysis` in `tests/test_fsm_llm/test_flow_analysis.py`, the `stop` hook of `eval.cases.run_case_trial` in `tests/test_fsm_llm_eval/test_cases.py`.

No test calls a real LLM, a model server or the network. Not covered here: a live run of the bench (blocks s0 to s2 were run by hand, all INCONCLUSIVE), the rest of `fsm_llm`.

## Architecture

Doubles and helpers (`conftest.py` unless named):

| Double | What it does |
| --- | --- |
| `ScriptedStrongLLM(replies)` | The compiler and scenario writer. `complete` pops a queue (text, `CompletionResponse` or an exception to raise); keeps `requests`; an exhausted queue raises `IndexError` |
| `ScriptedRunnerLLM(extraction_data, responses)` | The cheap model under a pushed child and on the host. Extracts a fixed dict, replies per state (`<current_state>` tag), `usage()` counts recorded requests, `count(kind, state)` and `shown(kind, state)` read the history |
| `ByTurn` | An honest executor: extracts a key only if the current turn says `<key> is <value>`, typed by the field type |
| `good_fsm()`, `valid_fsm()`, `stuck_fsm()` | A fresh compile that passes G1 to G5 for `return_keys=("answer",)`; a stuck one whose edge needs a value nothing extracts |
| `host_fsm(...)` | A host: greet, a `synthesize` state, ok on `status == ok`, fail on `status in [failed, aborted]` |
| `_offline_network` (autouse) | `tests.conftest.block_network`: refuses every IPv4 and IPv6 connect, loopback too |
| `foreign_calls` | Fails on any `LiteLLMInterface` built or provider call made; the test asserts the list is empty |

Layout: most files test one module, named `test_<module>.py`. Cross-module files are `test_concierge.py` (the whole pattern on real `API` objects, both ways the runner drives a conversation), `test_happy_path.py` (G6b with an honest executor) and `test_synthesis_bench.py`.

## Key files

| File | Covers |
| --- | --- |
| `test_policy.py` | `SynthesisPolicy` validation, G5 allow-list (a table pinning allowed plus refused keys against the core model), abort injection through the real `API` |
| `test_gates.py` | gate order proven with spies, `GateReport.verdicts`, `repair_text`, `verify_stored`, hostile names |
| `test_dryrun.py` | G6a fixture table (what is reported and, as hard, what stays silent), seeded differential against a brute-force oracle, the over-cap witness search, a speed bound |
| `test_simulation.py` | G6b verdict rule, stuck and forged procedures, bounds read between turns, replay |
| `test_happy_path.py` | the happy path built from the definition, an honest executor, value-typed edges |
| `test_synthesizer.py` | compile and repair rounds, cut or malformed replies, outage, the goal fence read from the REAL request, approval, reuse |
| `test_library.py` | digest, lifecycle `_EDGES`, atomic write, corruption never overwritten, search, traces, scenarios, lineage |
| `test_runner.py` | `run` and `converse`, abort at `max_turns`, an outage is no procedure fault, return keys, every failure the parent sees as a fixed reason, nesting, trace events |
| `test_tools.py` | the tool pair through `ToolRegistry.execute`: labels, caps, codes, `max_obtains` |
| `test_concierge.py` | end to end, both runner modes, plus the tool pair |
| `test_traces.py` | `RunTrace.__post_init__` (value-free), `TraceRecorder` folding |
| `test_diagnose.py` | the rule table in order, outages never a procedure fault |
| `test_patch.py` | RFC 6902 Appendix A examples under `/states/s`, scope, bounds before the walk, atomicity |
| `test_repair.py` | `propose_patch` fence, `repair_procedure` check order, decision rule, `write=False` default, never `trusted` |
| `test_synthesis_bench.py` | fixture hash, scripted user and grader, phases, verdict rule, preflight with a fake client, wire spy, pins, smoke refusal, hygiene (no network, no process control, ASCII) |

## Public interface under test

The package exports (`fsm_llm.synthesis.__all__`), the reason codes (`constants.REASONS`), finding codes per gate, `PatchError` codes, `PROMOTION_CODES`, `DIAGNOSIS_CODES`, the result shape `{status, procedure_id, version, reason}`, tool reply shapes, and the bench script's commands `register`, `smoke`, `run`, `report` (tested with fakes only).

## Data shapes

- A `SimulationReport` is a frozen dataclass; the tests build procedures that are stuck, forged or half-finished and read its finding codes.
- A library root holds `definitions/`, `records/`, `traces/`, `scenarios/`, `patches/`; tests use `tmp_path` and never share a directory between tests.
- Bench rows carry a derived `phase`; the tests feed rows to `metrics`, `decide` and `report` and read the verdict.

## Invariants and constraints

- Every gate test has a case that must pass and one that must fail, so a gate that checks nothing is caught. A silent case (what a gate must NOT report) is as hard a pin as a finding.
- Fence and defang tests read the REAL request (`CompletionRequest.model_dump_json()`), not the helper output.
- No test puts a goal, reply or provider text where an exception message, finding, result or trace is read without asserting it is absent.
- The bench script must not import `fsm_llm` or `litellm` at import time, must not contain process control (`kill`, `signal`), a non-ASCII character or a second copy of the shared stats helpers (`TestHygiene`).
- The expected fixture hash in `test_synthesis_bench.py` and the pins test (`TestPins`) must change together with an intended edit to a pinned file; a change voids a registered block.
- No test uses a marker; none is `slow`.

## Dependencies

- `fsm_llm.synthesis` and its submodules, core (`definitions`, `api`, `validator`, `flow_analysis`, `transition_evaluator`), `fsm_llm.agents` (`ToolRegistry`, `ToolCall`), `fsm_llm.eval`.
- `tests.conftest` (`MockLLM2Interface`, `block_network`), `scripts/synthesis_bench.py`, `scripts/bench_common.py`.
- pytest only; no network.

## Failure modes

- Run outside the repository root: `from tests...` imports and the `scripts/` path fail.
- A bench test fails on the fixture hash after an edit to the script or a pinned file: intended edit, update the hash and say in the commit that a registered block would be void; unintended edit, revert it.
- A test that builds a real `LiteLLMInterface` fails through `foreign_calls`; give it a double.

## Working here

- Name a test after the defect it guards; many docstrings start `Guards:` or name a review finding.
- Prefer REAL objects (`API`, `FSMLibrary`, `ToolRegistry`) with scripted models over mocks of internals.
- A new gate needs a test where it fails a compile and one where it stays silent. A new reason code goes in `constants.REASONS` and gets a test that branches on the code, not the message.
- After adding or removing tests, re-measure with `.venv/bin/python -m pytest tests/test_fsm_llm_synthesis --collect-only -q | tail -1` and update the root `AGENTS.md` lines; `tests/test_packaging.py` checks those literals. Do not write a count in this folder's docs unless the doc-count script reads it.
- Run: `.venv/bin/python -m pytest tests/test_fsm_llm_synthesis/`; lint `.venv/bin/python -m ruff check tests/test_fsm_llm_synthesis`.
