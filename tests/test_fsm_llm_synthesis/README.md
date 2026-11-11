# test_fsm_llm_synthesis

The pytest suite for `fsm_llm.synthesis`, the runtime FSM synthesis package of the FSM-LLM repository, and for the pre-registered bench script `scripts/synthesis_bench.py`. It lives at `tests/test_fsm_llm_synthesis/`. Its test counts are not written here: the repository root `AGENTS.md` carries them, and `tests/test_packaging.py` checks them against collection.

## What it is for

`fsm_llm.synthesis` has a model write a finite state machine (FSM) for a goal, judges it with gates, stores it in a library and runs it as a child of a host conversation. A wrong judgement is costly: a procedure that passes the gates is reused by many conversations. These tests check each gate against hand-built procedures that are good, stuck, forged or hostile, the library and its lifecycle, the runner, the agent tools, the traces, the diagnosis and the repair loop, the goal fence, and the bench script. No test calls a real model and none opens a network connection.

## How it works

- The strong model is `ScriptedStrongLLM` (in `conftest.py`): it answers `complete` from a queue of replies, keeps every request, and raises on an exhausted queue.
- The cheap model that runs the host and the pushed child is `ScriptedRunnerLLM` (a subclass of the repository's `MockLLM2Interface`): it extracts a fixed dict, writes a reply per state, and has a `usage()` meter that a simulator can read. `ByTurn` is the honest variant: it extracts a key only if the user's turn says `<key> is <value>`.
- Most tests run on REAL `API`, `FSMLibrary`, `ProcedureRunner`, `Simulator` and `ToolRegistry` objects, so a claim such as "a model cannot forge `procedure_abort`" is proven through the real pipeline.
- An autouse fixture blocks every IPv4 and IPv6 connect for the test (`tests.conftest.block_network`), loopback included, because the default model is a local Ollama. The `foreign_calls` fixture also fails a test that builds a `LiteLLMInterface` or makes a provider call.
- The bench tests import `scripts/synthesis_bench.py` by adding `scripts/` to `sys.path`, and drive its host FSM, scripted user, grader and run loop with scripted models and a fake model-server client. They never register, run or smoke the bench against a server.

## Files

- `__init__.py` - package marker.
- `conftest.py` - the network block, `foreign_calls`, `ScriptedStrongLLM`, `ScriptedRunnerLLM`, `ByTurn`, and the FSM builders `good_fsm`, `stuck_fsm`, `valid_fsm`, `host_fsm`.
- `test_policy.py` - `SynthesisPolicy`, G5 (`check_policy`, `needs_approval`), abort injection and stripping.
- `test_gates.py` - gate order, `GateReport`, the repair text, `verify_stored`.
- `test_dryrun.py` - G6a: the fixture table of reported and silent edges, the search against a brute-force oracle, and a speed bound.
- `test_simulation.py` - G6b: the verdict rule over trial results, stuck and forged procedures.
- `test_happy_path.py` - the G6b happy path built from the definition, with an honest executor.
- `test_synthesizer.py` - the compile and repair loop, the strong-model contract, the goal fence, approval.
- `test_library.py` - the content-addressed store, the lifecycle edge table, atomic writes, corruption handling, signature-first search, lineage.
- `test_runner.py` - `ProcedureRunner` on a host FSM with a `synthesize` state, both `run` and `converse`, and every failure the parent can see.
- `test_tools.py` - the agent tools `obtain_procedure` and `run_procedure` through `ToolRegistry.execute`.
- `test_concierge.py` - the whole pattern end to end, offline.
- `test_traces.py` - the value-free `RunTrace` and `TraceRecorder`.
- `test_diagnose.py` - the fixed rule table.
- `test_patch.py` - the RFC 6902 subset against the RFC's own examples and hostile input.
- `test_repair.py` - `propose_patch` and `repair_procedure`: the promotion order, the decision rule, the poisoning guards.
- `test_synthesis_bench.py` - the bench script: fixture hash, scripted user and grader, phases, verdict rule, preflight, pins, hygiene.

## How to use it

Run from the repository root with the project virtualenv:

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_synthesis/
.venv/bin/python -m pytest tests/test_fsm_llm_synthesis/test_gates.py -v
```

## Things to know

- Run from the repository root. Several tests import `tests.conftest`, and the bench tests read `scripts/synthesis_bench.py`, `scripts/bench_common.py` and the source files the bench pins.
- Editing a source file the bench pins (`src/synthesis/{dryrun,simulation,traces,gates,policy,synthesizer,runner,library,prompts,repair,constants}.py`, `src/flow_analysis.py`, `src/eval/cases.py` and others listed in `SRC_PINS`) or the fixture region of the script changes a hash. `test_synthesis_bench.py` holds the expected fixture hash; a change there voids a registered block, so change it only knowingly. Blocks s0 to s2 are registered and run; the s2 manifests already predate two later source edits (D-108, D-109), so `report s2` refuses by design and `s2/REPORT_s2.txt` is the record.
- Do not weaken a test to make a gate pass. Every gate test has a case that must pass AND one that must fail, so a gate that checks nothing is caught.
- No test uses a marker (`slow`, `integration`, `real_llm`): the whole suite runs in the default and the CI selection.
