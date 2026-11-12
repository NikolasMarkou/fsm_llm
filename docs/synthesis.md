# Runtime FSM Synthesis

> Covers FSM-LLM v0.12.0

Some conversations cannot be written ahead of time. "Collect what this visa form needs" depends on a goal that only shows up at run time. `fsm_llm.synthesis` lets a host conversation ask for a procedure by goal: a model writes a finite state machine (FSM) for it, a stack of checks decides whether that FSM may run, a library stores it, and a runner executes it as a child of the host conversation.

**Status of the evidence.** The code is tested offline with scripted models. The pre-registered live comparison, `scripts/synthesis_bench.py` (compiled FSM against an in-context ReAct agent, with a hand-written FSM as a harness check), ran three blocks, `s0`, `s1` and `s2`, and every verdict is INCONCLUSIVE. Every role and every arm used the same model, `ollama_chat/bonsai-2:27b` (Q2_0, 2-bit): there is NO strong/cheap split, so no strong-compile/cheap-run cost asymmetry was tested. No result in these docs says a compiled procedure helps; see "Live results" below.

## When to use it

Use it when the shape of a sub-conversation is known only at run time and you want the host's own, usually cheaper, model to run it with guardrails. Do not use it when you can write the FSM yourself: a hand-written FSM is cheaper, easier to review and is what the bench will compare against. A procedure cannot call tools (the runner refuses a `completion` state), cannot set a persona and cannot be longer than the policy allows (12 states by default).

`import fsm_llm` loads no synthesis code. Import it by name:

```python
from fsm_llm.synthesis import FSMLibrary, FSMSynthesizer, ProcedureRunner
```

The `synthesis` extra adds no dependency, there is no console script, and importing `fsm_llm.synthesis` also imports `fsm_llm.agents` (and through it `fsm_llm.reasoning`) and `fsm_llm.eval`. Nothing in core imports it back.

## The concierge pattern

A host FSM names a procedure with a `synthesize` state. The state is silent (no `response_instructions`), carries the goal and the key names, and routes on `synthesis_result.status`. Save this as `concierge.json`:

```json
{
  "name": "Concierge",
  "description": "Greets a guest and books a table with a procedure written on demand",
  "initial_state": "greet",
  "persona": "A calm, friendly concierge",
  "states": {
    "greet": {
      "id": "greet",
      "description": "Welcome the guest",
      "purpose": "Greet the guest",
      "response_instructions": "Welcome the guest to the venue.",
      "transitions": [{"target_state": "gather", "description": "Always continue"}]
    },
    "gather": {
      "id": "gather",
      "description": "Run the booking procedure",
      "purpose": "Collect the party size and the date",
      "response_instructions": "",
      "synthesize": {
        "goal": "Book a table: ask the party size and the date.",
        "inputs": ["venue"],
        "return_keys": ["party_size", "visit_date"],
        "on_failure": "sorry"
      },
      "transitions": [
        {
          "target_state": "done",
          "description": "The procedure returned the booking",
          "priority": 1,
          "conditions": [{
            "description": "The procedure finished",
            "requires_context_keys": ["synthesis_result"],
            "logic": {"==": [{"var": "synthesis_result.status"}, "ok"]}
          }]
        },
        {
          "target_state": "sorry",
          "description": "The procedure failed or was stopped",
          "priority": 2,
          "conditions": [{
            "description": "The procedure did not finish",
            "requires_context_keys": ["synthesis_result"],
            "logic": {"in": [{"var": "synthesis_result.status"}, ["failed", "aborted"]]}
          }]
        }
      ]
    },
    "done": {
      "id": "done",
      "description": "Booking confirmed",
      "purpose": "Confirm the booking",
      "response_instructions": "Confirm the booking using party_size and visit_date."
    },
    "sorry": {
      "id": "sorry",
      "description": "Booking failed",
      "purpose": "Apologise",
      "response_instructions": "Apologise and say the booking could not be made."
    }
  }
}
```

The loader checks a synthesize state against the real transition evaluator, so a wrong edge fails at load, not in production:

- Every edge of the state must read `synthesis_result` (or a `handler_only_keys` entry), or it would fire before the runner has produced a result.
- No edge may pass while the result is absent or `running`.
- The edge to `on_failure` must pass for `status` `failed` AND for `aborted`. Write it as `status in [failed, aborted]`. A bare `== "failed"` is refused, because an aborted procedure would otherwise strand the host.
- A synthesize state declares no extraction and no `completion`, and `on_failure` must be one of its own transition targets.

The runner drives the host. This runs offline with scripted interfaces (`Strong` stands in for the strong model, `Host` for the host's cheap model); with real models you pass `FSMSynthesizer(model="...")` and an `API` built the usual way:

```python
import json
import tempfile

from fsm_llm import API
from fsm_llm.definitions import (
    CompletionResponse,
    FieldExtractionResponse,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.synthesis import FSMLibrary, FSMSynthesizer, ProcedureRunner
from fsm_llm.synthesis.prompts import EXAMPLE_FSM

with open("concierge.json", encoding="utf-8") as f:
    host = json.load(f)


class Strong(LLMInterface):  # the compiler: returns a ready FSM as JSON text
    calls = 0

    def complete(self, request):
        self.calls += 1
        return CompletionResponse(kind="final", text=json.dumps(EXAMPLE_FSM))

    def generate_response(self, request):
        raise NotImplementedError


class Host(LLMInterface):  # the cheap model that runs the host and the child
    answers = {"party_size": 4, "visit_date": "Friday"}

    def generate_response(self, request):
        return ResponseGenerationResponse(message="ok", message_type="response", reasoning="")

    def extract_field(self, request):
        value = self.answers.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name, value=value, is_valid=value is not None,
            confidence=1.0 if value is not None else 0.0, reasoning="",
        )

    def complete(self, request):
        raise NotImplementedError


strong = Strong()
api = API.from_definition(host, llm_interface=Host())
with tempfile.TemporaryDirectory() as root:
    synth = FSMSynthesizer(strong, library=FSMLibrary(root))
    runner = ProcedureRunner(api, host, synth)
    for _ in range(2):
        conv_id, _ = api.start_conversation({"venue": "Luna"})
        runner.run(conv_id, max_steps=40)          # message-free; or runner.converse(conv_id, "four, Friday")
        print(api.get_current_state(conv_id), api.get_data(conv_id)["synthesis_result"]["status"])
    print(strong.calls)                            # 1: the second booking reused the stored procedure
```

The output is `done ok` twice, then `1`. Drive the host through the runner (`run`, `converse`, `end_conversation`), not through `API` alone: a bare `API.converse` on a synthesize state finds no extraction and no passing edge, so it stays in the state. The runner is the thing that gives the field its meaning.

## The synthesizer

`FSMSynthesizer(llm=None, *, model=None, policy=None, library=None, reviewer=None, embed=None, simulator=None)` takes exactly one of `llm` (anything implementing `complete`) or `model` (it builds a `LiteLLMInterface`). Two methods matter:

- `compile(goal, *, inputs=(), return_keys) -> CompileResult` asks the strong model for FSM JSON, runs the gates and, when they fail, sends the failures back as text for up to `policy.max_repair_rounds` more rounds. It raises `SynthesisError` when no round passes. A provider outage raises it too, chained from `LLMResponseError`, with no retry. A reply cut at `max_tokens` is a failed round and is never parsed.
- `obtain(goal, *, inputs=(), return_keys, min_lifecycle="verified") -> Procedure` searches the library first and compiles on a miss. `Procedure.source` is `library` or `compiled`.

### The gates

Every compile runs these checks, in this order, and reports all failures, not just the first:

1. **G1, loads.** The JSON loads as a core `FSMDefinition`, exactly as `API` would load it.
2. **G2, structure.** The core `FSMValidator` reports no error. Its warnings never fail a compile, and G2 is never stricter than the loader.
3. **G3, data flow** (`fsm_llm.flow_analysis`). A key an edge reads must be extracted earlier on every path, or be a declared input. Each return key must be guaranteed when the procedure ends.
4. **G4, liveness** (`fsm_llm.flow_analysis`). No trap, no loop without an exit that can fire, no edge hidden behind an unconditional edge of lower priority number, no permanent tie, no path longer than `policy.max_turns`.
5. **G5, policy** (`policy.check_policy`). Limits on states, edges, fields and text; reserved names; no persona; no secret-looking or internal key names; no model setting on a classification entry; no regular expression in a validation rule; completion tools only from `policy.allowed_tools`; no `synthesize` state inside a procedure unless `max_synthesis_depth` is above 1 (the default is 1). It is an allow-list of fields, so a field nobody thought of is refused.
6. **G6a, dead edges** (`dryrun.py`). It tries a small pool of values per key through the real evaluator and reports an edge no assignment can fire. No model call. It is coarse on purpose: anything outside the fragment it understands counts as possibly firing, so it never rejects a good procedure for what it cannot read.
7. **G6b, simulation** (`simulation.py`), only when a `Simulator` is given. A scripted happy path built from the definition, plus scenarios of four kinds written by one model call (`missing_info`, `correction`, `off_topic`, `adversarial`), are played against the procedure on the simulator's interface. It passes when the happy path and the overall pass rate both reach `policy.min_pass_rate` (default 0.5) and the happy path moves past the first state. `require_simulation=True` makes it mandatory.
8. **G7, approval.** Required when `policy.require_approval` is set or a `completion` state lists tools. See Approval below.

When the gates pass, an abort path is injected: a silent `abort` terminal state and a priority 0 edge to it from every non-terminal state, gated on the handler-only key `procedure_abort`. The runner sets that key when the child spends its turn budget, so no model reply can forge or block it. G1 and G2 run again on the injected form. The names `abort`, `procedure_abort` and `synthesis_result` are reserved: a compile may not declare, read or write them.

### What `verified` means

A compiled procedure is stored as `verified`. That means the gates above passed: static checks, a dead-edge check and, if you passed a simulator, a bounded simulation over scenarios that a model wrote. It does not mean the procedure is correct. If the simulator runs on the same model that wrote the procedure, a passing simulation is partly the model agreeing with itself, which is a weaker signal than a separate judge. Review what matters before you trust it (see Approval).

## The library

`FSMLibrary(path)` is a directory. Each procedure is stored under the sha256 of its canonical definition: `definitions/<sha256>.json` and `records/<sha256>.json`, with `traces/`, `scenarios/` and `patches/` beside them. Every write goes to a temp file, is synced, then replaced into place; a corrupt file raises `LibraryError` and is never overwritten, and a symlink is refused. Two processes on one directory are last writer wins on the run counters.

A record has a lifecycle: `draft`, `verified`, `trusted`, `deprecated`. A compile is stored `verified`. A human moves it to `trusted` with `library.approve(digest, by)`, or run counts do when `policy.trust_after_successes` is set (never for a repaired child). Nothing leaves `deprecated`. A synthesize state's `min_lifecycle` (`verified` or `trusted`) says which stored procedures it will run.

### Reuse

`search` first filters on the signature (every required input available, the return keys covered, the tools allowed, the lifecycle high enough), then scores the survivors against the stored goal:

- With no embedder, only an identical goal reuses (lowercased, whitespace normalised; punctuation counts).
- With an embedder (a function from a list of texts to vectors, `embed=`), the score is a cosine and a stored procedure is reused at or above `policy.reuse_min_score` (default 0.8). That is a similarity claim: cosine ignores negation, so "book a table" and "do not book a table" can match. For exact reuse, leave `embed=None` or set `reuse_min_score=1.0`.

A hit is re-verified (G1 to G6a on the stored form) before it runs. A hit that fails is skipped with a warning and a compile follows.

## Running a procedure

`ProcedureRunner(api, definition, synthesizer, *, trace_sink=None)` serves a host definition. Per conversation it settles three things as the host runs: it pops an ended child, aborts a child that spent `policy.max_turns` steps (it sets `procedure_abort`), and starts a pending synthesize state (write `running`, `obtain`, check that the procedure may run, push it as a child with no inherited context).

- `run(conversation_id, *, max_steps, max_seconds=None)` runs message-free on core `run_until_terminal`. A spent budget raises `RunBudgetExceededError`.
- `converse(conversation_id, message) -> str` is one user turn.
- `end_conversation(conversation_id)` closes the conversation and drops the runner's bookkeeping; use it instead of `API.end_conversation`.

The child gets only the declared `inputs` and runs on the host's own interface. When it ends, only the declared `return_keys` come back, and only if every one is present and the child was not aborted. The result `synthesis_result` is `{status, procedure_id, version, reason}`: `status` is `ok`, `failed`, `aborted` or `running`, and `reason` is a fixed code or `None`, never exception text. The codes are `inputs_missing`, `return_keys_missing`, `provider_error`, `library_error`, `compile_failed`, `push_failed`, `approval_required`, `approval_denied`, `reviewer_error`, `depth_refused`, `completion_unsupported`, `lifecycle_too_low`, `not_simulated`, `budget_spent` and `unexpected_error`. Branch on the code, not on any message.

Things to know when you run one:

- The result key is skip-if-set: a state that must run again on re-entry has to clear `synthesis_result` on entry (core `clear_keys_on_entry`).
- One thread per conversation, like `API.converse`. The runner holds no lock.
- A nested synthesize state (depth 2) works only for a procedure that returns one of its inputs.
- `policy.max_turns` is the only turn bound. Core has none.
- Two interfaces are metered separately: the strong one inside the synthesizer, the cheap one on the host `API`.

## Agent tools

`create_procedure_tools(synthesizer, *, runner_llm, registry=None, max_obtains=20) -> ToolRegistry` gives an agent two tools, for example `ReactAgent(tools=registry)`:

- `obtain_procedure(goal, return_keys, inputs=[])` compiles or reuses and replies with a receipt: `procedure_id`, `version`, `lifecycle`, `source`, `states`, `gates`.
- `run_procedure(procedure_id, inputs={})` runs a stored procedure on `runner_llm` with no user messages and replies with its values, or `failed: <code>`. A procedure that needs a user answer replies `failed: needs_user_input`.

Both replies start with a fixed untrusted label, are capped and control-stripped, and carry codes, ids and defanged values only. The tool descriptions are module constants, never built from a goal or a record. Neither tool is flagged `requires_approval`: that flag makes four agent patterns refuse the registry, so approval happens inside the synthesizer instead. `max_obtains` caps compiles per registry, because a compile is the slow, costly call.

## Approval

A reviewer is a callable `reviewer(request: ProcedureApprovalRequest) -> bool`. It is asked on a compile only, never for a library hit, and it fails closed: a missing reviewer when approval is required, a reviewer that raises, and a result that is not exactly `True` all refuse the procedure, which is then not stored. `request.text()` shows a Mermaid diagram of the procedure, the gate verdicts, the goal and a size summary, and starts with a fixed untrusted label. `reviewer_from_hitl(hitl)` adapts a `fsm_llm.agents.HumanInTheLoop`.

```python
from fsm_llm.synthesis import FSMSynthesizer, SynthesisPolicy

synth = FSMSynthesizer(
    strong,
    library=library,
    policy=SynthesisPolicy(require_approval=True),
    reviewer=lambda request: input(request.text() + "\nrun it? ") == "yes",
)
```

Approval records who approved (`approved_by`) as provenance. It does NOT make the record `trusted`: that needs `library.approve(digest, by)`, a separate human edge.

## Traces, diagnosis and repair

Procedures that fail in use can be improved, and the loop is built so that a user cannot talk it into weakening a shared procedure.

- **Traces.** `TraceRecorder(library.add_trace)` is a `trace_sink` for the runner. A `RunTrace` holds names, type names, state ids, counts and codes, never a value, user text or reply. `library.read_traces(digest)` returns `(traces, skipped)`; a torn or hostile line is skipped and counted, and the file keeps the last 200 traces per procedure.
- **Diagnosis.** `diagnose(traces, definition, policy, *, return_keys=())` applies nine rules in a fixed order and returns `Diagnosis` objects with a code, the state, the keys and the share of runs. Only a run that failed for the procedure's own reasons (`budget_spent`, `return_keys_missing`) can justify a rewrite; an outage or a refusal is `not-procedure-fault` and is never patched.
- **Patch.** `propose_patch(diagnoses, definition, policy, llm)` asks the model for RFC 6902 operations (`add`, `remove`, `replace`, `test` only), scoped to `/states/<diagnosed state>/...`, never the abort state, `initial_state`, `persona` or the handler-only list. `apply_patch` applies them to a copy and checks bounds before it walks anything.
- **Repair.** `repair_procedure(parent_digest, ops, *, library, synthesizer, simulator, replay_scripts=None, write=False)` re-runs G1 to G6b on the patched child (and asks the approval reviewer again if the child now needs it), replays the parent's stored failing scenarios on parent and child, and returns a `PromotionResult` with the decision and the counts: `promote`, `regressed` or `no-improvement`, or a refusal code. By default it writes nothing. Only `write=True` with a `promote` verdict stores the child, as `verified`, never `trusted`, never with the parent's approver, and a `trusted` parent is never deprecated. Replay scripts come from the parent's stored scenarios plus the ones you pass; production traces are never used as scripts.

## Limits

`SynthesisPolicy` is a frozen dataclass; a bad value raises `SynthesisError` at construction and is never clamped. The defaults are in `docs/api_reference.md` (Synthesis). The ones to know:

- Size of a procedure: `max_states` 12, `max_transitions_per_state` 6, `max_fields_per_state` 10, `max_text_chars` 600.
- Compile: `max_repair_rounds` 3, `max_compile_tokens` 8192, `max_synthesis_seconds` 300.
- Run: `max_turns` 40 steps of the child in total.
- Simulation: `max_sim_turns` 12, `sim_trials` 2, `min_pass_rate` 0.5, `max_simulation_calls` 400.
- Repair: `repair_min_failures` 3, `max_repairs_per_lineage` 3 (counted over the whole tree from the root).

## Live results

Three blocks of the comparison in `scripts/synthesis_bench.py` ran against `ollama_chat/bonsai-2:27b` on a local Ollama server (bench id `synthesis`; rows, summaries, reports and pre-registrations in `scripts/bench_data/synthesis/`). The compiler, scenario writer, executor, ReAct agent and hand-written arm are ONE model: there is no strong/cheap split, a G6b pass is partly the model agreeing with itself, and no cost saving from a cheaper runner could appear. 16 judge-free tasks in four families (collect, route, branch, retry), 3 trials each; the primary unit is the first trial of each task. The verdict rule (SUPPORTED needs S above R with Fisher p under 0.05) is byte-identical in all three blocks.

| Block | S first trial (Wilson 95%) | H first trial | R first trial | All trials S / H / R of 48 | pass^3 S / H / R | S first-trial failures | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s0 | 6 of 16 (0.18 to 0.61) | 8 of 16 | 14 of 16 | 27 / 26 / 40 | 6 / 8 / 12 | 10 (8 compile) | INCONCLUSIVE (h = 8 of 16, below 10) |
| s1 | 15 of 16 (0.72 to 0.99) | 15 of 16 | 16 of 16 | 45 / 45 / 45 | 14 / 15 / 14 | 1 (compile) | INCONCLUSIVE (ceiling: s = 15 and r = 16) |
| s2 | 15 of 16 (0.72 to 0.99) | 16 of 16 | 16 of 16 | 47 / 47 / 46 | 15 / 15 / 15 | 1 (compile) | INCONCLUSIVE (ceiling: s = 15 and r = 16) |

Cost per block (prompt plus completion tokens over all 144 rows, wall time of the arm):

| Block | S tokens | S wall | H tokens | H wall | R tokens | R wall |
| --- | --- | --- | --- | --- | --- | --- |
| s0 | 826,182 + 186,111 | 5,433 s | 183,186 + 27,663 | 945 s | 536,759 + 92,680 | 2,869 s |
| s1 | 616,130 + 133,987 | 3,923 s | 175,211 + 26,435 | 869 s | 536,759 + 93,411 | 2,848 s |
| s2 | 613,642 + 130,506 | 3,991 s | 157,326 + 23,863 | 822 s | 536,759 + 92,601 | 2,902 s |

How to read this:

- No block shows a compiled procedure beating the ReAct agent. In s1 and s2 the three arms are within one task of each other on 16 tasks that this model has saturated, so the ceiling clause fires and the data cannot separate them. S is also the most expensive arm here: it pays a compile call and a bounded simulation (over 1,000 simulation calls per block), and with one model a reused procedure saves no cheaper-model cost. Reuse itself worked: 31 of 48 S rows in s1 and s2 made no compile call.
- s0 to s1 to s2 are NOT independent tests. After each block the failures were read in the raw calls and fixed, and the next block ran on the same 16 tasks. The gain of S from 6 to 15 of 16 shows the diagnosed defects were real and fixed on those tasks; it does not show the approach generalises.
- The hand-written arm and the ReAct arm are not comparable across blocks: the grader (s1) and two hand-written hints (s1, s2) changed between blocks, each disclosed in the pre-registration of the block that first used it.
- Single trials of this model are noisy: S r2 passed trial 1 and failed trials 2 and 3 of s1 on the same stored procedure; the ReAct arm (r1, two trials) and the hand-written arm (y2, one trial) ran out of turns re-asking in s2.

What each round changed (generic changes under `src/synthesis/`, no task text there; a hygiene test checks it):

- s0 to s1 (D-091 to D-105): a compile-time instruction that names the key a gated state extracts; deterministic repair of a mismatched closing bracket in a compile reply; the loader's allowed JsonLogic operators in the compile prompt and in plain ASCII in the repair text; a note on what a key name is when a gate reads a computed one. Harness v2 (one grader for all arms, hand-written branch detail from the state's own question).
- s1 to s2 (D-106, D-107): a compile prompt rule that a key the goal says to decide needs a value gate over the allowed values; the hand-written route hint says to pick the label that fits the need.
- After s2 (D-108, D-109), NOT covered by any registered block: the compile prompt says `missing` is true only when a key is absent; a dead-edge repair line says the same once, and a failing happy-path repair line says that a key chosen from a list needs an `in` gate. They were checked by short smokes only (y2 passed without a repair; y3 passed after one round; S r4 failed twice before the second note existed; the smoke meant to check that second note was stopped at close before it ran a trial, so D-109 has no live check at all). No fourth block was run, so their effect is unmeasured.

Which rows predate which fix: every s0 row predates all of the above; s1 rows predate D-106 to D-109; s2 rows predate D-108 and D-109. The manifests of a block pin the source it ran against, so after later source edits `report <block>` refuses by design and the committed `REPORT_<block>.txt` is the record.

## Safety model

1. The goal is untrusted caller text. It reaches the strong model only inside a fenced block, cut to 2,000 characters, control characters stripped and the characters `[ ] < >` replaced, so it cannot spell a marker. It is never a setting or an instruction. System prompts are built from the policy alone.
2. Model-written text never becomes a setting. G5 is an allow-list, and the persona, model, prompt settings and regular expressions of a compile are refused. Text from a compile, a receipt or an approval request reaches a model or a person under a fixed untrusted label.
3. Return keys are the only way data leaves a child, and only on success. No goal, compile text, reply or provider text goes into an exception message, a finding, a result or a trace.
4. The abort path hangs on a key no extraction can write.
5. Repair needs positive evidence, a replay and, if asked, an approval; it is propose-only by default.

## Known limits

- The live comparison is inconclusive: three blocks, one model for every role and arm, 16 tasks the model has saturated, the later two in-sample after fixes made from earlier failures, and no fourth block. There is no measured claim that a compiled procedure beats a ReAct agent or saves anything; see "Live results".
- Open defects seen live: a G6b false rejection when a label-restricted extraction instruction meets an edge with no value gate (the gate feeds the filler "sample"; D-106 and D-109 push the model toward an `in` gate, unmeasured); G6b passes some compiles that fail at run time; the 2-bit compiler repeats a dead edge when a repair line gives no reason; the ReAct and hand-written arms sometimes re-ask past `max_iterations`; G6a has an over-cap residual and the outage detection has blind spots (see above).
- `verified` is not proof of correctness (see above). G6a is coarse and G6b is only as good as the scenarios a model wrote.
- Procedures cannot call tools. A compile may list `completion` tools (and then needs approval), but the runner and `run_procedure` refuse to run one.
- A corrupt library file blocks retrieval for the whole library until a human fixes it.
- Two signals of misbehaviour (an out-of-scope intent and a raw malformed extraction) are not recorded by core, so `diagnose` cannot use them.
- The runner marks a provider outage from the host interface's error counter (`usage().errors`). A host that calls `reset_usage()` during a run, or an interface with no meter, gets the outage recorded as `budget_spent`, which `diagnose` reads as a procedure fault. A shared interface can give a false outage mark, which fails closed.
- Importing the package loads `fsm_llm.agents`, `fsm_llm.reasoning` and `fsm_llm.eval`.

See `src/synthesis/README.md` and `src/synthesis/AGENTS.md` for the file list and the contracts, and `docs/api_reference.md` for signatures.
