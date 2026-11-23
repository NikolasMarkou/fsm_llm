# test_fsm_llm_synthesis

Path: `tests/test_fsm_llm_synthesis`
Purpose: Offline pytest suite for `fsm_llm.synthesis` (gates G1 to G7, library, runner, tools, traces, diagnosis, patch and repair).

## Scope

Covers `fsm_llm.synthesis` (`policy`, `gates`, `dryrun`, `simulation`, `synthesizer`, `prompts`, `library`, `runner`, `tools`, `traces`, `diagnose`, `patch`, `repair`, `constants`, `exceptions`). Core pieces it relies on are tested elsewhere: `State.synthesize` and `SynthesizeConfig` in `tests/test_fsm_llm/test_synthesize_field.py`, `fsm_llm.core.verify.flow` in `tests/test_fsm_llm/test_flow_analysis.py`, the `stop` hook of `eval.cases.run_case_trial` in `tests/test_fsm_llm_eval/test_cases.py`.

No test calls a real LLM, a model server or the network. Not covered here: the retired synthesis bench (blocks s0 to s2 were run by hand, all INCONCLUSIVE, and deleted with their rows), the rest of `fsm_llm`.

## Architecture

Doubles and helpers (`conftest.py` unless named):

| Double | What it does |
| --- | --- |
| `fsm_llm.testing.ScriptedLLM(replies)` (not in `conftest.py`) | The compiler and scenario writer. `complete` pops a queue (text, `CompletionResponse` or an exception to raise); keeps `requests`; an exhausted queue raises `UnscriptedCallError`; metered (set `llm.usage = None` for a model with no meter) |
| `ScriptedRunnerLLM(extraction_data, responses)` | The cheap model under a pushed child and on the host (a `MockLLM2Interface` subclass on purpose: no bulk extraction, no `complete`). Extracts a fixed dict, replies per state (`<current_state>` tag), `usage()` counts recorded requests, `count(kind, state)` and `shown(kind, state)` read the history |
| `ByTurn` | An honest executor: extracts a key only if the current turn says `<key> is <value>`, typed by the field type |
| `good_fsm()`, `valid_fsm()`, `stuck_fsm()` | A fresh compile that passes G1 to G5 for `return_keys=("answer",)`; a stuck one whose edge needs a value nothing extracts |
| `host_fsm(...)` | A host: greet, a `synthesize` state, ok on `status == ok`, fail on `status in [failed, aborted]` |
| `_offline_network` (autouse) | `tests.conftest.block_network`: refuses every IPv4 and IPv6 connect, loopback too |
| `foreign_calls` | Fails on any `LiteLLMInterface` built or provider call made; the test asserts the list is empty |

Layout: most files test one module, named `test_<module>.py`. Cross-module files are `test_concierge.py` (the whole pattern on real `API` objects, both ways a host drives a conversation: `converse` and `run_until_terminal`) and `test_happy_path.py` (G6b with an honest executor).

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
| `rig.py` | The rig of the driver rows: a real `API` on a host with a `synthesize` state and `SynthesizeExecutor` registered, and `ExecutorDriver` (`run`, `converse`, `end_conversation`: a thin wrapper over the `API` that steps the parent after a pop, which the engine does not do), `make_rig` (`host_edit`) and the helpers `make_host`, `make_procedure`, `serve_procedure`, `Metered` |
| `test_executor.py` | What only `SynthesizeExecutor` does: a Pass 2 failure after an obtain, the child greeting, the compile under the turn gate, a cancel and an interrupt during `obtain`, a forged running record, the declared inputs only and no grant, a restart proved by the frame's `fsm_id`, the persona rule, trace mode, one lock-order row, the layering |
| `test_driver_rows.py` | What a host sees of a run: the ok path, abort at `max_turns`, an outage is no procedure fault, return keys, every failure the parent sees as a fixed reason, nesting, trace events that carry no value |
| `test_executor_stream.py` | The `synthesize` kind on core's streams over a bare `API` with a `SynthesizeExecutor`: `run_until_terminal_stream` and `converse_stream` give the text and end state of the plain call, the arguments are checked at call time, a stream closed early leaves the ended child for the next entry (settle is never a closing action) |
| `test_executor_bounds.py` | The rows of the deleted runner file that carried an invariant of the engine or the trace, on the executor: the engine bounds the child (`max_turns`, a spent host budget is not the child's), a refused start is the same turn's failure edge in `converse` and `run`, a settled result is never obtained twice, a conversation ended under a child counts no run, the trace is value free |
| `test_pushed_child_result.py` | A pushed child owns its synthesis result (`inherits_result_on_push`, D-022 of plan d454fd62), on a passive executor |
| `test_tools.py` | the tool pair through `ToolRegistry.execute`: labels, caps, codes, `max_obtains` |
| `test_concierge.py` | end to end on a bare `API` with a `SynthesizeExecutor`, both modes (`converse` and `run_until_terminal`), plus the tool pair |
| `reference_tasks.py`, `test_reference_tasks.py` | the 16 reference tasks of the retired synthesis bench, their `hand_fsm` and `host_definition`, the scripted `User`, the grader and `drive_fsm` (arms S and H on one host, a bare `API` with a `SynthesizeExecutor`, and library); the hand arm runs every task offline with no strong call |
| `test_traces.py` | `RunTrace.__post_init__` (value-free), `TraceRecorder` folding |
| `test_diagnose.py` | the rule table in order, outages never a procedure fault |
| `test_patch.py` | RFC 6902 Appendix A examples under `/states/s`, scope, bounds before the walk, atomicity |
| `test_repair.py` | `propose_patch` fence, `repair_procedure` check order, decision rule, `write=False` default, never `trusted` |

## Public interface under test

The package exports (`fsm_llm.synthesis.__all__`), the reason codes (`constants.REASONS`), finding codes per gate, `PatchError` codes, `PROMOTION_CODES`, `DIAGNOSIS_CODES`, the result shape `{status, procedure_id, version, reason}` and tool reply shapes.

## Data shapes

- A `SimulationReport` is a frozen dataclass; the tests build procedures that are stuck, forged or half-finished and read its finding codes.
- A library root holds `definitions/`, `records/`, `traces/`, `scenarios/`, `patches/`; tests use `tmp_path` and never share a directory between tests.

## Invariants and constraints

- Every gate test has a case that must pass and one that must fail, so a gate that checks nothing is caught. A silent case (what a gate must NOT report) is as hard a pin as a finding.
- Fence and defang tests read the REAL request (`CompletionRequest.model_dump_json()`), not the helper output.
- No test puts a goal, reply or provider text where an exception message, finding, result or trace is read without asserting it is absent.
- No test uses a marker; none is `slow`.

## Dependencies

- `fsm_llm.synthesis` and its submodules, core (`core.definitions`, `engine.api`, `core.verify` (the validator and `flow`), `core.transition_evaluator`), `fsm_llm.agents` (`ToolRegistry`, `ToolCall`), `fsm_llm.eval`.
- `tests.conftest` (`block_network`), `fsm_llm.testing` (`MockLLM2Interface`, `ScriptedLLM`, `state_in_prompt`).
- pytest only; no network.

## Failure modes

- Run outside the repository root: `from tests...` imports fail.
- A test that builds a real `LiteLLMInterface` fails through `foreign_calls`; give it a double.

## Working here

- Name a test after the defect it guards; many docstrings start `Guards:` or name a review finding.
- Prefer REAL objects (`API`, `FSMLibrary`, `ToolRegistry`) with scripted models over mocks of internals.
- A new gate needs a test where it fails a compile and one where it stays silent. A new reason code goes in `constants.REASONS` and gets a test that branches on the code, not the message.
- No document states this suite's test count; measure it with `.venv/bin/python -m pytest tests/test_fsm_llm_synthesis --collect-only -q | tail -1`.
- Run: `.venv/bin/python -m pytest tests/test_fsm_llm_synthesis/`; lint `.venv/bin/python -m ruff check tests/test_fsm_llm_synthesis`.
