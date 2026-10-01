# fsm_llm.agents: audit record and roadmap

Status: design record for the `fsm_llm.agents` audit, 2026-09-29.

- Code range: `c632893..HEAD` (plan `plan-2026-09-29T103145-06a5ec0a`, 27 steps plus eight completion fixes: 15.1, 20.1, 13.1, 3.1, 21.1, 21.2, 24.1, 20.2).
- Version: unreleased changes on top of `0.11.0`. See `CHANGELOG.md` `## [Unreleased]`.
- Scope: Track A of the audit (test infrastructure E3/E4, Phase 0 hotfixes, Phase 1 pattern correctness). Track B (Phases 2-6), the live agent bench (E1/E2), E5 and the D5 deprecations are deferred.
- Current contracts: `src/fsm_llm/agents/README.md` (user guide), `src/fsm_llm/agents/CLAUDE.md` (maintainer contracts), `docs/api_reference.md` (Agents section).

The audit was written against `4a5af49` without running tests. Every claim was re-checked at `c632893` before any fix. Verdicts: TRUE (holds as stated), PARTIAL (part holds), GHOST (does not hold or was already fixed), NOT VERIFIED (outside Track A, not traced).

## Commits

| Step | Commit | Summary |
| --- | --- | --- |
| 1 | `6ef307c` | `PromptGroundedLLM` test fake (E3) |
| 2 | `d2df2a6` | Offline network block for agents and meta tests; swallowed asserts fixed (E4, TEST-03) |
| 3 | `1672257` | Caller context cannot forge the approval grant or run outputs (SEC-01, LOOP-12) |
| 3.1 | `c35d407` | Completion fix: per-pattern run outputs unforgeable; flagged tools with no approver fail closed; review hardening |
| 4 | `79d465c` | Misplaced constructor kwargs raise (SEC-02, PAT-12, API-02) |
| 5 | `c889be4` | `requires_approval` is the default policy for a callback-only HITL (SEC-03, REACT-08) |
| 6 | `bd284e5` | Patterns without HITL refuse approval-gated tools (SEC-04) |
| 7 | `b040e49` | Tool arguments bind before the call; a tool never runs twice (TOOL-01/02/03) |
| 8 | `427fe10` | Sub-agent failure propagates; malformed native calls skipped; `AgentServer` bounded (TOOL-14, REACT-11, SEC-09) |
| 9 | `9b887bb` | Secrets kept out of memory listings, observations, traces and logs (SEC-05, SEC-06) |
| 10 | `9863675` | Semantic memory reloads, finds unembedded entries, persists `max_entries` (MEM-01..04) |
| 11 | `663ad23` | An approval grant is spent before its tool runs (LOOP-14) |
| 12 | `fe2c9a5` | ADaPT propagates budget errors, caps fan-out, reports failed decompositions (PAT-04) |
| 13 | `eea02f8` | One success contract with `stop_reason` (API-04, PAT-11 forced pass) |
| 13.1 | `ed6f69d` | Completion fix: success reflects who concluded; forced keys handler-only |
| 14 | `ef68b57` | `make_fresh_keys_handler` and typed field helpers (LOOP-02, LOOP-08, LOOP-11) |
| 15 | `6f704f3` | ReAct loop: typed fields, think-turn counting, `agent_feedback` (LOOP-01/04/05/06/09/10/16/17) |
| 15.1 | `901a8ee` | Completion fix: envelope-name collision, permissive terminate, core `_field_value` guard |
| 16 | `d6732cc` | `run_stream` yields only model text (LOOP-13) |
| 17 | `f5ce156` | Reflexion records each episode's grounded reflection (REACT-01/02) |
| 18 | `3340203` | PlanExecute replans on failed steps with typed plans (PAT-01/02) |
| 19 | `5805c42` | Debate rounds grounded; SelfConsistency votes on the answer (PAT-03/05) |
| 20 | `06e2dd0` | PromptChain gates stop the chain; MakerChecker/EvalOpt drafts grounded (PAT-06/11) |
| 20.1 | `65cc68e` | Completion fix: generated artifacts keep native JSON; an extraction envelope never ships |
| 20.2 | this commit | Completion fix: lossless envelope salvage, visible truncation, empty artifacts are no answer; D-050 rationale corrected |
| 21 | `e6a975d` | REWOO and Orchestrator report real outcomes and propagate budget errors (PAT-09/10) |
| 21.1 | `5cd416f` | Completion fix: Orchestrator judges only real worker results; narrowed planner prompts |
| 21.2 | `36b0b1e` | Completion fix: the Debate judge sees the debate history |
| 22 | `467df80` | AgentGraph topological order; Swarm keeps the task and exact handoff budget (PAT-07/08) |
| 23 | `01d433e` | VerifiedReact fails closed; `reason` tool gets the task; dead classification override removed (REACT-03/04/05) |
| 24 | `0146925` | `create_agent` pattern first; `AgentConfig.instructions`, `extra="forbid"`, `LLM_MODEL` (API-01/03, PAT-13) |
| 24.1 | `7696061` | Completion fix: actionable prompt-size errors, narrower core envelope guard, pattern-name normalisation, docs |
| 25 | `1fc874b` | Static `__all__`, dead constants removed, valid meta-builder example (API-05/06, META-06) |
| 26 | `a8fa500` | This record, docs sync, changelog, test counts (API-07) |
| 27 | `5d1f423` | Core persona limit raised to 4,000 characters (D-048) |

## Verified status and outcome per audit id

Loop (RC1, the conversational pipeline used as an agent engine):

| Id | At `c632893` | Outcome |
| --- | --- | --- |
| LOOP-01 | TRUE | Fixed, steps 14-21: every generated loop value is a typed per-field extraction with task and observations in the prompt; state-level bulk instructions are empty (except ReAct/Reflexion `think` with `use_classification=True`, D-019 of plan 21cd7f8e) |
| LOOP-02 | TRUE | Fixed, step 14 (`make_fresh_keys_handler`) plus each pattern step 15, 17-20 |
| LOOP-03 | TRUE | Partly fixed: the terminal reply is the answer channel (Debate, step 19; PromptChain, step 20). Core still never extracts on a terminal state; no core change |
| LOOP-04 | TRUE | Fixed, step 15: `max_iterations` counts think turns in `AgentHandlers.check_iteration_limit` only |
| LOOP-05 | PARTIAL | Fixed, step 15: no tool runs after the forced-stop flag |
| LOOP-06 | TRUE | Fixed, step 15: `agent_feedback` key carries executor warnings and HITL denials; a denial is not an observation |
| LOOP-07 | TRUE | Partly fixed, step 15: no blind bulk call per think turn; one call per typed field remains |
| LOOP-08 | TRUE | Fixed, steps 14-15: `agent_trace` kept out of per-field prompts by `context_keys`; not capped (declined) |
| LOOP-09 | PARTIAL | Fixed, steps 15, 17-21: intermediate states have empty `response_instructions`. PromptChain step replies stay user-owned |
| LOOP-10 | PARTIAL | Fixed, steps 15, 17, 18, 20: routing fields typed `bool`/`float`/`list` |
| LOOP-11 | PARTIAL | Fixed, steps 14-21: per-field wording points to the task and observations; generated text fields open with a "compose it yourself" sentence |
| LOOP-12 | TRUE | Fixed, step 3: run-output keys stripped from caller context; `observation_count` seeded 0. Fix 3.1 (D-052): each pattern's own drafts, verdicts and answers (`_run_output_keys`) are stripped too, also on AgentGraph edges and Swarm hand-offs |
| LOOP-13 | PARTIAL | Fixed, step 16 |
| LOOP-14 | PARTIAL (needs `handler_timeout`) | Fixed, step 11 |
| LOOP-15 | NOT VERIFIED | Deferred to Track B |
| LOOP-16 | TRUE | Step numbers fixed, step 15. The per-turn thought is still empty: the typed `reasoning` field was removed in 15.1 (open) |
| LOOP-17 | TRUE | Message fixed, step 15 (cites the loop ceiling). No partial result on `BudgetExhaustedError` (deferred) |

Security:

| Id | At `c632893` | Outcome |
| --- | --- | --- |
| SEC-01 | TRUE (bypass reproduced; public `approval_granted` alone PARTIAL) | Fixed, step 3, also through `AgentServer`, SelfConsistency, Swarm and AgentGraph |
| SEC-02 | TRUE | Fixed, step 4 (denylist, not a whitelist); fix 3.1 adds every `HumanInTheLoop` constructor name |
| SEC-03 | TRUE | Fixed, step 5 |
| SEC-04 | TRUE | Mitigated, step 6: the four patterns refuse gated tools. Real HITL there is Track B |
| SEC-05 | TRUE | Fixed, step 9 |
| SEC-06 | TRUE | Fixed, step 9; `ApprovalRequest.parameters` stays raw on purpose |
| SEC-07 | NOT VERIFIED | Deferred to Track B |
| SEC-08 | NOT VERIFIED | Deferred to Track B |
| SEC-09 | TRUE | Fixed, step 8 (`max_concurrent`, 503, generic errors with `error_id`) |
| SEC-10 | NOT VERIFIED | Deferred to Track B |
| SEC-11 | TRUE | Declined in Track A (D-030 of plan c1d5bfbc kept); needs ToolSpec side-effect metadata |
| SEC-12 | NOT VERIFIED | Deferred to Track B (step 24 changed only the judge's model resolution) |

Tools:

| Id | At `c632893` | Outcome |
| --- | --- | --- |
| TOOL-01 | TRUE (reproduced) | Fixed, step 7 |
| TOOL-02 | TRUE (reproduced) | Fixed, step 7 |
| TOOL-03 | TRUE (reproduced) | Fixed, step 7; fix 3.1: partials, callable objects, async callables and positional-only parameters |
| TOOL-04 to TOOL-13 | NOT VERIFIED | ToolSpec code subset ported in plan 944e2692: TOOL-04 (exact schemas from `args_model`) and TOOL-07 (no retry of a granted call) fixed; tool annotations and an enforced `timeout_s` added; the rest stays in Track B |
| TOOL-14 | TRUE | Fixed, step 8 |

ReAct family:

| Id | At `c632893` | Outcome |
| --- | --- | --- |
| REACT-01 | TRUE | Fixed, step 17 |
| REACT-02 | TRUE | Fixed, step 17 (`evaluation_fn` on `evaluate` entry, not PRE_TRANSITION) |
| REACT-03 | TRUE | Fixed, step 23: the dead `classification_tool_override` is removed |
| REACT-04 | TRUE | Fixed, step 23 |
| REACT-05 | TRUE | Fixed, step 23 (task text, no shadowing, registry subclasses kept). Engine input not length-capped (open) |
| REACT-06 | NOT VERIFIED | Deferred to Track B |
| REACT-07 | NOT VERIFIED | Deferred to Track B |
| REACT-08 | TRUE | Fixed, step 5 (construction warning); fix 3.1 (D-052): a flagged tool nobody can approve raises `AgentError` at construction and `run()` |
| REACT-09 | NOT VERIFIED | Deferred to Track B |
| REACT-10 | NOT VERIFIED | Deferred to Track B |
| REACT-11 | TRUE (thinking blocks not verified) | Malformed arguments fixed, step 8. The rest closed by plan 944e2692: native_fc is an FSM on core, `initial_context` goes through `_init_context`, `api_kwargs` reach `API`, and `complete_fn` is gone (the seam is `llm_interface=`) |

Patterns:

| Id | At `c632893` | Outcome |
| --- | --- | --- |
| PAT-01 | TRUE ("pre-tool guess" GHOST, fixed by D-024 of plan 21cd7f8e) | Fixed, step 18 |
| PAT-02 | PARTIAL | Fixed, step 18 |
| PAT-03 | TRUE | Fixed, step 19 |
| PAT-04 | TRUE (and `agents/CLAUDE.md` wrongly said the errors propagate) | Fixed, step 12 |
| PAT-05 | TRUE | Fixed, step 19 |
| PAT-06 | TRUE | Fixed, step 20 |
| PAT-07 | TRUE | Fixed, steps 3 (edge strip) and 22 (Kahn order) |
| PAT-08 | TRUE | Fixed, step 22 (task, budget). Transfer tools deferred (D-018) |
| PAT-09 | TRUE ("one bad step loses all" GHOST) | Fixed, step 21 |
| PAT-10 | TRUE | Fixed, step 21 (skips recorded, budget errors propagate). Parallel workers deferred |
| PAT-11 | TRUE | Fixed, steps 13 (forced pass reports failure) and 20 (grounded drafts) |
| PAT-12 | TRUE | Fixed, step 4 |
| PAT-13 | TRUE | Fixed, step 24 |

Memory, integrations, meta-builder:

| Id | At `c632893` | Outcome |
| --- | --- | --- |
| MEM-01 | TRUE | Fixed, step 10 |
| MEM-02 | TRUE | Fixed, step 10 |
| MEM-03 | TRUE | Fixed, step 10 (dir, fsync, `max_entries`). O(n^2) rewrite per `add` deferred |
| MEM-04 | TRUE | Fixed, step 10 (length mismatch scores 0.0). Embedding under the lock deferred |
| MEM-05 to MEM-07 | NOT VERIFIED | Deferred to Track B (memory v2) |
| INT-01 to INT-06 | NOT VERIFIED | Deferred to Track B (MCP v2, A2A v1.0) |
| META-01 to META-05, META-07 | NOT VERIFIED | Deferred to Track B |
| META-06 | TRUE | Fixed, step 25 |
| META-08 | GHOST (no private meta method with at most one reference) | Nothing to do |

API, observability, tests:

| Id | At `c632893` | Outcome |
| --- | --- | --- |
| API-01 | TRUE | Fixed, step 24 |
| API-02 | TRUE (plus broken `model=` snippets in `docs/api_reference.md`) | Fixed, step 4 |
| API-03 | TRUE | Fixed, step 24 |
| API-04 | TRUE | Fixed, step 13 |
| API-05 | TRUE | Fixed, step 25 |
| API-06 | TRUE | Fixed, step 25 (constants). `DecompositionError`/`ToolValidationError` removed in plan 07ad3f8c (no raise site) |
| API-07 | TRUE | Fixed, steps 4 and 26 |
| OBS-01 to OBS-05 | NOT VERIFIED | Deferred to Track B |
| TEST-01 | TRUE | Fixed, step 1 (`PromptGroundedLLM`); used by every Phase-1 step |
| TEST-02 | GHOST (fixed by `f4e9d53`) | Nothing to do |
| TEST-03 | PARTIAL | Fixed, step 2 |
| TEST-04, TEST-05 | NOT VERIFIED | Deferred with E1/E2/E5 |

E-track: E3 done (step 1), E4 done (step 2, scoped to the agents and meta suites). E1, E2 and E5 deferred.

## Adjustments to the original roadmap

Where verification or a consumer contradicted the proposed fix, the fix was changed. Ids refer to `plans/plan-2026-09-29T103145-06a5ec0a/decisions.md`.

- SEC-01 (D-002): `_init_context` strips only the driver grant `_approval_granted` and `RUN_OUTPUT_KEYS`, not every internal-prefix key. `_sensitive` policy inputs and harness roots are legitimate. `AgentServer` is the trust boundary and drops every internal-prefix key, logged, not rejected with 400.
- SEC-02 (D-003): no kwargs whitelist. A denylist (`hitl`, `tools`, `evaluation_fn`, `approval_callback`, plus config-owned `model`, `temperature`, `max_tokens`) raises `TypeError`; litellm and `API` passthrough (`seed`, `timeout`, `handlers`, `llm_interface`, ...) stays open. The monitor passes `tools=` only to classes that take it.
- SEC-03 (D-004): the flag is a default policy only for a callback-only HITL; with a policy, the policy alone decides and is never ANDed with the flag. Fix 3.1 (D-052): with neither a callback nor a policy, a flagged tool fails closed on the ReAct family like SEC-04; `RetryingToolRegistry` never retries a flagged tool (one approval, one execution); ReasoningReact's registry is a live view of the caller's, so a flag added later is seen.
- SEC-04 (D-005): refusal at construction and at `run()` instead of new HITL paths.
- SEC-11, roadmap D7 (D-006): D-030 empty-input recovery kept; no half-designed `read_only` flag ahead of ToolSpec.
- LOOP-04 (D-007, D-028): counting changed only in `AgentHandlers.check_iteration_limit`; the shared `make_iteration_limiter` is untouched. For N >= 2, `max_iterations=N` gives N think turns and N - 1 tool turns (N = 1 behaves like N = 2; Reflexion closes a cycle on the act exit, so its counts differ), plus a forced stop a few transitions before the loop ceiling. Fix 13.1 (D-051): the model is still asked on the last think turn and its own evidence-backed conclusion there reports success.
- LOOP-02 (D-008): one `make_fresh_keys_handler` for producing-state entry; never clears a forced `True` verdict.
- LOOP-01 (D-009): typed per-field extraction, but `think` with `use_classification=True` keeps its bulk fill.
- LOOP-06 (D-021, D-029): `agent_feedback` is cleared on think exit, not think entry (an entry clear erased it unread).
- REACT-02, PAT-01 (D-031, D-032): core picks the edge before PRE_TRANSITION runs, so the verdict handlers run on state entry, not at PRE_TRANSITION as planned.
- LOOP-14 (D-015): the grant is spent before the tool runs and recorded call-locally; handlers are not made `.critical()`.
- LOOP-08 (D-017): `agent_trace` is not capped or renamed (a data path, and core reads it); it is kept out of prompts instead.
- REACT-03 (D-010): the dead override is deleted, not repaired (a repair needs a core channel).
- API-04 (D-011, D-027): additive `stop_reason`; a public `forced_stop_reason` context key tells forced passes apart, because `API.get_data` drops internal keys. Fix 13.1 (D-051): a forced reason only when a handler overrode the verdict (EvalOpt, MakerChecker read only the recorded reason, not the bare limiter flag); Reflexion's reflection cap and Debate's forced consensus are forced; framework keys are core `handler_only_keys` on every agent FSM; an AgentGraph node with `success=False` takes no edge; a Swarm handoff to an unknown agent is `no_result`.
- API-01 (D-012, D-047): `create_agent` kept a legacy positional prompt with a `DeprecationWarning`; plan 07ad3f8c removed it (D-041 there): a first argument that names no pattern raises `ValueError`. `AgentConfig.instructions` is prefixed as an `Agent instructions:` block to every non-empty state and field instruction, not written into `persona` (persona reaches only Pass 2).
- D-048: the core persona cap is now 4,000 characters (`MAX_PERSONA_LENGTH`), but `AgentConfig.instructions` still uses the instructions block (D-047): a longer persona still never reaches the Pass-1 field prompts.
- PAT-13 (D-013): `AgentConfig.model` stays a `str` with an env-reading default factory; more than 20 readers use it directly.
- PAT-04, PAT-10 (D-014, D-042): budget errors from sub-runs go into a call-local `RunEndingErrorHolder` and are re-raised after the loop; core's handler `continue` mode would swallow a plain raise.
- PAT-08 (D-018, D-045): Swarm transfer tools deferred; `next_agent` stays caller- or tool-written.
- PAT-07 (D-044): local Kahn sort in `AgentGraph`, not `fsm_llm.workflows.DependencyResolver` (agents does not depend on workflows).
- PAT-03, PAT-05 (D-036, D-037): the Debate answer is the conclude reply, `proposition` is only the success key; SelfConsistency votes on the last `Answer:` line.
- Generated text fields (D-036, D-043): live `qwen3.5:4b` returned null for text fields worded as "extract"; every generated text field now opens with a sentence telling the model to compose the value.
- Envelope-name collision (D-034, D-035): a typed field named `reasoning` was filled by core `llm._field_value` from the extraction envelope's own `reasoning`. The think `reasoning` field was removed, `_typed_field_extraction` rejects envelope names, and core `_field_value` no longer falls back to an envelope key in an envelope-shaped reply (one carrying `value` or `field_name`; fix 24.1 keeps a flat `{"confidence": 0.8}`). This, the fix 20.1/20.2 envelope salvage (with `constants.TRUNCATED_SALVAGE_CONFIDENCE`) and the step 27 persona limit are the plan's core changes.
- E4 (D-019): the network block covers only the agents and meta suites.
- Generated artifacts (D-050, D-056): fixes 20.1 and 20.2. EvalOpt, MakerChecker and PromptChain artifacts are typed `any` again, which restores their baseline type (step 20 had made them `str`); on a provider without a grammar the model can then return a JSON deliverable as a native object. On Ollama the `any` grammar has no object branch, so a JSON deliverable is still an escaped string and the protection is core `llm.py`'s envelope salvage: the envelope text never lands in context, a complete value is kept whole (an undefined escape such as `\d` is kept literally), and a value cut off by `max_tokens` keeps its prefix, is logged as a WARNING and returns at confidence 0.3. ADaPT `attempt_result` stays `str` (D-057: a short answer the answer path reads only as a str). `artifact_text` treats `False`, `0`, `{}`, `[]` as no answer.
- Debate judge (D-053, D-054): fix 21.1 narrowed the judge's `consensus_reached` prompt and dropped `debate_rounds`; fix 21.2 restored it (bounded by `num_rounds`), `agent_trace` stays out.
- API-01 limits (fix 24.1): instructions plus a tool catalogue that overflow core's 5,000-character instruction slot raise `AgentError` naming the slot, the instructions length and the tool count (at `run()`, when the FSM loads; the limit is read from core's error, not copied).
- D5 deprecations (D-020): not done. Debate and SelfConsistency were fixed instead, since they returned wrong answers while shipping.

## Live evidence (G3)

`fsm-llm-eval examples --category agents`, model `ollama_chat/qwen3.5:4b`, 4 workers, N=1 heuristic score (a coarse proxy, not a success rate).

| Point | Commit | Health | Notes |
| --- | --- | --- | --- |
| Baseline | `c632893` | 175/192 = 91.1% | Non-PASS: concurrent_react; timeouts maker_checker, plan_execute, plan_execute_recovery, orchestrator_specialist, supply_chain_optimizer |
| Mid-plan probe | `6f704f3` (after step 15) | 153/192 = 79.7% | 13 timeouts; falsification signal fired (more than 5 pp down) |
| After fix 15.1 | `901a8ee` | 174/192 = 90.6% | Within noise. Newly non-PASS: adapt, hierarchical_orchestrator, maker_checker_code, react_hitl_combined; recovered: plan_execute, plan_execute_recovery, supply_chain_optimizer, concurrent_react |

Root cause of the step-15 drop (D-034): the new typed `reasoning` field collided with the extraction envelope's `reasoning` key, so the model's meta-commentary was stored as context and fed into later prompts, where it talked the model out of terminating. A stricter `should_terminate` wording and LOOP-04's doubled tool budget made the non-terminating runs time out. A call probe on `reasoning_tool` went from 10 calls / 22 s (baseline) to 29 calls / 52 s and a wrong answer; after 15.1 it took 8 calls / 17.7 s with the right answer. The prompt-grounded fake could not see this; only the live run could.

Accepted cost (D-040): a run that never concludes by itself now takes about twice as long before the forced stop, because `max_iterations` counts think turns.

Final G3 (`7696061`, after completion fixes; `fsm-llm-eval examples --category agents`, `ollama_chat/qwen3.5:4b`, 4 workers, N=1): 181/192 = 94.3% vs 91.1% baseline at `c632893`. Envelope leaks: 0. Structured outputs validated: 2 (= baseline). Non-PASS: orchestrator_specialist, react_hitl_combined, supply_chain_optimizer (TIMEOUT), concurrent_react (PARTIAL). The 0-4 score is a heuristic: it cannot see answer quality, so the review rounds (D-050, D-056) are the stronger evidence.

Fix 20.2 (`9607e56`) changed only the envelope salvage (lossless for complete values, visible for truncated ones) and was not re-run live.

Plan scale: 7 files added (tests and docs only, 0 source files); source net +3,123 lines. Both are over the plan's own budget (D-058).

## Known open items

- LOOP-16: the per-turn thought is not extracted, so `AgentStep.thought`, `ToolCall.reasoning` and `ApprovalRequest.reasoning` are empty unless `use_classification=True`.
- Typed per-field extraction costs one LLM call per field per turn, plus one retry per null required field.
- The ReAct-family success rule counts a failed tool call as success evidence. ParallelReact's empty batch loops until the limiter.
- HITL: the policy sees the call before empty-input recovery and a shallow context; a forced stop reached in `await_approval` skips the approved call (fails closed). REWOO, PlanExecute, ParallelReact and native_fc refuse gated tools instead of asking. With a policy set, a flagged tool the policy does not gate runs unasked (the policy owns the decision). (`RetryingToolRegistry` no longer retries a granted call, and retries only tools annotated `idempotent` or `read_only`: plan 944e2692.)
- Swarm never hands off by itself: nothing shipped writes `next_agent`.
- SEC-11: D-030 recovery can pass the task text to a side-effecting single-parameter tool.
- REACT-05: reasoning-engine input is not length-capped. MEM-03/04: O(n^2) rewrite per `add` and embedding under the store lock.
- Examples to re-check in the final G3 (D-040): adapt, hierarchical_orchestrator, maker_checker_code.
- Review pass 2 residue (D-056), shipped as known limits:
  - A long generated artifact can still be cut off by the per-field `max_tokens` budget; it ships truncated, marked only by a WARNING and confidence 0.3, and an unjudged PromptChain run can report `success=True`. Needs an output-budget change (Track B).
  - Orchestrator: subtasks dropped over `max_workers` are invisible to the collect judge and do not affect `success` (D-049 traded re-delegation for bounded calls).
  - AgentGraph has no failure routing: a node with `success=False` takes no edge, so no fallback branch can be built.
  - `FSMValidator` flags the framework `handler_only_keys` on agent FSMs as possible typos (false alarm in `fsm-llm-validate`).
  - The constructor denylist is derived from `HumanInTheLoop.__init__`'s signature; a new generic HITL parameter name would reject a litellm kwarg of that name.
  - Core: concurrent first-use `FSMDefinition` validation can race (pre-existing at `c632893`).
- `PromptGroundedLLM` grounds on scripted evidence; it cannot show whether a real model will terminate. Every extraction change still needs a live probe.
- The offline network block covers only the agents and meta suites.

## Track B (deferred)

Track B is the redesign part of the roadmap. It needs live measurement first, so it is split into follow-up plans.

Entry gates, in order:

1. E1: an agent bench (`scripts/agents_bench.py`, live, stdlib-only like `scripts/harness_bench.py`) and E2: its pre-registered baseline block.
2. G1: the new runtime's ReAct beats the legacy loop on the harness L4 EXECUTE bench.
3. G2: the agent bench shows no regression against the E2 baseline.
4. G3: `fsm-llm-eval examples --category agents` shows no regression.

E5 (the examples scorer reads `result.success`) lands with E1.

Work deferred to Phases 2-6 of the original roadmap:

- Runtime and accounting: cost capture (OBS-01; per-interface call and token counters exist since plan 944e2692), monitor event bursts (OBS-02), timeouts inside a turn (OBS-03), `arun` and checkpoints (OBS-04), a richer `AgentTrace` (OBS-05), `Budget`/`RunState`/events, a partial result on budget errors (LOOP-17), LOOP-15.
- ToolSpec: the rest of TOOL-05 to TOOL-13 (plan 944e2692 ported annotations, exact schemas and an enforced timeout; SEC-11 (roadmap D7) can now read the read-only and side-effect hints).
- Async core (roadmap D3) with a per-model executor (D6); parallel Orchestrator workers (PAT-10); ParallelReact stall detection (REACT-06); real HITL for REWOO, PlanExecute, ParallelReact and native_fc (lifts the SEC-04 refusal).
- Integrations: MCP v2 (INT-01 to INT-05), A2A v1.0 on the official SDK (INT-06, roadmap D4), AG-UI, OTEL.
- Guardrails: SEC-07, SEC-08, SEC-10, SEC-12.
- Memory v2: MEM-05 to MEM-07.
- Meta-builder: META-01 to META-05 and META-07.
- Swarm transfer tools (PAT-08), AutoMemory `respond` tool without mutating the caller's registry (REACT-07), REACT-09, REACT-10.
- Harness migration to the new runtime after Phase 4 (roadmap D8).

Deprecations (roadmap D5), deferred to Track B: ParallelReact, Debate and SelfConsistency, once the runtime offers replacements (D-020). The positional `create_agent(system_prompt, ...)` shim, `DecompositionError` and `ToolValidationError` were removed in plan 07ad3f8c.

## Recorded baselines at `d4b1626`

Two "before" records for later agent changes. Both measure the agents code of commit `d4b1626`.

### G3 baseline

`fsm-llm-eval examples --category agents` run from a clean worktree at `d4b1626`. N=1 heuristic score (a coarse proxy, not a success rate).

| Point | Commit | Model | Workers | Health | Envelope leaks | `Success:` lines (True / False) | stop_reason lines | Output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | `d4b1626` | `ollama_chat/qwen3.5:4b` | 4 | 178/192 = 92.7% | 0 | 20 / 9 | 0 | `evaluation/2026-09-29_22-12_d4b1626_g3-baseline/` (gitignored, local only) |

Score distribution: 43 PASS, 1 PARTIAL, 4 BROKEN. Wall time 1,517.5 s for 48 examples.

Non-PASS:

| Example | Score | Failure | Time |
| --- | --- | --- | --- |
| concurrent_react | 2 (PARTIAL) | F-EXTRACT (1/3 fields; `all_tasks_succeeded` and `all_answers_present` reported False) | 108.0 s |
| hierarchical_orchestrator | 1 (BROKEN) | F-LOOP, timeout | 300.1 s |
| orchestrator_specialist | 1 (BROKEN) | F-LOOP, timeout | 300.1 s |
| plan_execute | 1 (BROKEN) | F-LOOP, timeout | 180.1 s |
| supply_chain_optimizer | 1 (BROKEN) | F-LOOP, timeout | 300.1 s |

How the counts were read (raw logs, not the score):

- Envelope leaks: no log contains `"field_name"` or `extracted_data`.
- `Success:` lines: 18 `Success: True` and 7 `Success: False` lines, plus `Pipeline success: True` in react_structured_pipeline and three panel lines in multi_debate_panel (1 `success=True`, 2 `success=False`). 21 examples print no success line at all.
- All 7 examples that print `Success: False` still score 4 (PASS): architecture_review, debate, eval_opt_structured, legal_document_review, maker_checker_code, reflexion, regulatory_compliance. The scorer does not read `result.success`.
- No example prints a `stop_reason`, so the logs cannot show why a run stopped.
- The four timed-out examples left empty STDOUT and STDERR in their logs (output is lost when the process is killed), so they give no raw evidence beyond the timeout.

Compared with the G3 at `7696061` (181/192 = 94.3%): 3 points lower, within N=1 noise. concurrent_react, orchestrator_specialist and supply_chain_optimizer are non-PASS in both runs. Of the D-040 re-check list, adapt and maker_checker_code now pass and hierarchical_orchestrator times out.

### Agent bench block B0

`scripts/bench_data/agents-react/B0/` (tracked; never edited or re-run). 38 tasks, 3 trials, `ollama_chat/qwen3.5:4b` (digest `2a654d98...`), max_iterations 8, timeout 180 s, temperature 0.5. The rows were produced at commit `73d7a6c`, whose `src/` is identical to `d4b1626`. Recount offline with `.venv/bin/python scripts/agents_bench.py report agents-react`.

| Arm | Engine | First-trial pass@1 (Wilson 95%) | pass^3 | Rows correct | success but incorrect | LLM calls mean | Latency p50 | Stop reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy` | `create_agent("react", ...)` (FSM ReactAgent) | 28/38 = 73.7% (0.580-0.850) | 26/38 | 81/114 | 15 | 11.5 | 12.2 s | answered 93, stalled 15, max_iterations 6 |
| `native_fc` | `NativeFunctionCallingReactAgent` | 37/38 = 97.4% (0.865-0.995) | 36/38 | 110/114 | 4 | 2.4 | 0.95 s | answered 114 |

The `legacy` arm loses mostly on `multi_step_chain` (2/6 first trial) and `no_tool_needed` (2/5). A comparison block must use a new block id, a new arm label, the same task hash and limits, and the same model digest.

## Measured after the core step driver (plan `07ad3f8c`, iteration 1)

Agent loops are driven by core's `advance`/`run_until_terminal` (no synthetic "Continue." turn). Both measurements below were taken on 2026-10-01, `ollama_chat/qwen3.5:4b`, Ollama digest `2a654d98e6fb...` (same as B0 and the G3 baseline), one Ollama workload at a time.

### Agent bench block B1

`scripts/bench_data/agents-react/B1/` (tracked; run once, 05:52-06:22 UTC, never edited or re-run). Arm `fsm_advance` = `create_agent("react", tools, config=...)` on the migrated code (rows produced at `0f0789c`; `src/` identical to the manifest's `1c8572e`). Same 38 tasks (`tasks_sha256` `831a7cd5...`), 3 trials, max_iterations 8, timeout 180 s, temperature 0.5, max_tokens 1000, same call meter as B0. Pass rule (D-052) fixed before the run.

```bash
.venv/bin/python scripts/agents_bench.py run --bench-id agents-react --block B1 --arm fsm_advance --trials 3
.venv/bin/python scripts/agents_bench.py report agents-react --blocks B0 B1 --pair B1/fsm_advance:B0/legacy
```

| Block / arm | First-trial pass@1 (Wilson 95%) | pass^3 | Rows correct | success but incorrect | fail but correct | LLM calls mean / median | Tokens mean | Latency p50 / p95 | Stop reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 `legacy` (`d4b1626` code) | 28/38 = 73.7% (0.580-0.850) | 26/38 | 81/114 | 15 | 3 | 11.5 / 9.0 | 10,948 | 12.2 s / 69.2 s | answered 93, stalled 15, max_iterations 6 |
| B1 `fsm_advance` | 32/38 = 84.2% (0.696-0.926) | 32/38 | 96/114 | 12 | 9 | 10.97 / 8.0 | 9,463 | 10.1 s / 42.7 s | answered 99, stalled 12, max_iterations 3 |

Fisher two-sided, B1 vs B0 first-trial pass@1: p = 0.399 (pass^3: p = 0.176).

| Category | Tasks | B0 first trial | B1 first trial | B0 rows | B1 rows |
| --- | --- | --- | --- | --- | --- |
| single_tool | 6 | 5 | 6 | 15/18 | 18/18 |
| multi_step_chain | 6 | 2 | 2 | 6/18 | 6/18 |
| no_tool_needed | 5 | 2 | 4 | 6/15 | 12/15 |
| error_recovery | 5 | 5 | 5 | 15/15 | 15/15 |
| typed_args | 5 | 4 | 5 | 12/15 | 15/15 |
| distractor_tools | 6 | 5 | 5 | 15/18 | 15/18 |
| unanswerable | 5 | 5 | 5 | 12/15 | 15/15 |

Read from the raw rows:

- Envelope leaks (`"field_name"`, `"extracted_data"` in an answer): 0 of 114 (B0: 0). `Continue.` in an answer: 0 (B0: 1). Error rows: 0. Empty answers: 0.
- Forced stops: 15 of 114 (B0: 21). All 12 `stalled` rows are the four no-tool tasks `nt-ready`, `nt-reverse`, `nt-paint`, `nt-rgb` (17 calls each, every trial); `nt-ready`, `nt-paint` and `nt-rgb` still carry the correct answer with `success=False`. The 3 `max_iterations` rows are `di-currency` (30 calls).
- First-trial flips: 4, all B0-incorrect to B1-correct, none the other way: `st-capital` ("The capital of Veloria is Maskett." to "Maskett"), `ty-repeat`, `nt-ready`, `nt-paint`. Each task asks for the bare value; the B1 answers follow that instruction. Over all three trials `un-employee` (1/3 to 3/3) and `un-founder` (2/3 to 3/3) also improved.
- Remaining incorrect tasks (incorrect in all 3 trials): `ch-density`, `ch-manager-budget`, `ch-salaries`, `ch-order-local`, `nt-reverse`, `di-currency`. Per task, B1 is either correct in all 3 trials or in none.

### Agent bench block B2 and harness L4 block B2 (plan `944e2692`)

`NativeFunctionCallingReactAgent` runs as an FSM on core's completion state instead of its private loop. Both blocks ran once on 2026-10-01 at `955f189` (clean tree; manifests registered at `38d4357`), `ollama_chat/qwen3.5:4b`, digest `2a654d98e6fb...`, one Ollama workload at a time. Pass rules (D-010, re-recorded in D-049 before row 1); result in D-050. Both PASS.

```bash
.venv/bin/python scripts/agents_bench.py run --bench-id agents-react --block B2 --arm fsm_toolcall --trials 3
.venv/bin/python scripts/agents_bench.py report agents-react --blocks B0 B2 --pair B2/fsm_toolcall:B0/native_fc
.venv/bin/python scripts/harness_bench.py run --bench-id l4-execute-write --block B2 --arm native_fsm --n 40 --seed 20260722000
.venv/bin/python scripts/harness_bench.py report l4-execute-write --blocks B1 B2 --pair B2/native_fsm:B1/native
```

`scripts/bench_data/agents-react/B2/` (run 18:28-18:31 UTC). Arm `fsm_toolcall` = `create_agent("native_fc", tools, config=...)` with B0 `native_fc`'s limits (max_iterations 8, timeout 180 s, temperature 0.5, max_tokens 1000), same 38 tasks and `tasks_sha256`, 3 trials, call meter "2" (core usage; B0 used meter "1").

| Block / arm | First-trial pass@1 (Wilson 95%) | pass^3 | Rows correct | success but incorrect | fail but correct | LLM calls mean / median | Tokens mean | Latency p50 / p95 | Stop reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 `native_fc` (`d4b1626` code) | 37/38 = 97.4% (0.865-0.995) | 36/38 | 110/114 | 4 | 0 | 2.39 / 2.0 | 1,432 | 0.95 s / 3.19 s | answered 114 |
| B2 `fsm_toolcall` | 37/38 = 97.4% (0.865-0.995) | 35/38 | 108/114 | 6 | 0 | 2.39 / 2.0 | 1,423 | 0.90 s / 3.21 s | answered 114 |

Fisher two-sided, B2 vs B0 first-trial pass@1: p = 1.000 (pass^3: p = 1.000). Rule: pass@1 >= 35/38 PASS, calls <= 4.0 PASS, Fisher not required at 37/38, 0 envelope leaks PASS.

- Envelope leaks (`"field_name"`, `"extracted_data"`, `"extracted_value"` in an answer): 0 of 114 (B0: 0). `Continue.`: 0. Error rows: 0. Empty answers: 0. Max latency 5.4 s.
- First-trial flips: none. Incorrect rows, all `success=True`: `nt-reverse` in all 3 trials (as in B0), `nt-paint` t2 and t3 (a full sentence where the task asks for one word; B0 3/3), `un-station` t2 ("cannot be provided because there is no weather station", outside the grader's not-found phrases; B0 3/3). `st-capital` went from 2/3 to 3/3.
- `--pair` prints 12 differing manifest fields; the request disclosures (timeout 120 s, step-14 schema bytes, settings, code path) are listed in `scripts/bench_data/README.md`.

`scripts/bench_data/l4-execute-write/B2/` (run 18:31-18:33 UTC). Arm `native_fsm` = the `native=True` EXECUTE dispatch on the new code, n=40, seeds 20260722000-20260722039 (B1's).

| Block / arm | content_matched (Wilson 95%) | write_tool_issued | bytes_on_disk | success | Tool calls mean (distribution) | Elapsed mean / max |
| --- | --- | --- | --- | --- | --- | --- |
| B1 `native` (`2a89226` code) | 40/40 (0.912-1.000) | 40/40 | 40/40 | 40/40 | 2.70 (2: 34, 3: 3, 9: 1, 11: 2) | 8.5 s / 20.3 s |
| B2 `native_fsm` | 40/40 (0.912-1.000) | 40/40 | 40/40 | 40/40 | 2.33 (2: 38, 6: 1, 11: 1) | 2.8 s / 6.4 s |

Fisher two-sided on every metric: p = 1.000. Rule: >= 38/40 verified writes PASS. Failed tool calls 6 (B1: 10); the 11-call row (run 2) retried `read_file` four times, then wrote, then appended to a plan file.

### Full examples evaluation, `0f0789c` against `d4b1626` (same day)

`fsm-llm-eval examples`, all 101 examples, 4 workers, default timeout 120 s with the built-in per-example and per-category overrides (`src/fsm_llm/eval/` is unchanged since `d4b1626`, and so is `examples/`). The `d4b1626` run used an isolated worktree with `PYTHONPATH=<worktree>/src`; the example subprocesses inherited it (checked in a live subprocess environment), and its `results.json` records `git_commit` `d4b1626`. N=1 heuristic score each.

```bash
LLM_MODEL=ollama_chat/qwen3.5:4b .venv/bin/fsm-llm-eval examples --workers 4 --model ollama_chat/qwen3.5:4b
cd <worktree at d4b1626> && PYTHONPATH=$PWD/src LLM_MODEL=ollama_chat/qwen3.5:4b \
  <repo>/.venv/bin/python -m fsm_llm.eval examples --workers 4 --model ollama_chat/qwen3.5:4b \
  --output-dir <repo>/evaluation/2026-10-01_09-54_d4b1626_qwen3.5-4b_same-day-baseline
```

| Point | Commit | Started (UTC) | Health | agents | Distribution (4/2/1) | Timeouts | F-CODE | Wall time | Output (gitignored, local only) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HEAD | `0f0789c` | 06:23 | 388/404 = 96.0% | 180/192 | 95 / 2 / 4 | 4 | 0 | 1,897 s | `evaluation/2026-10-01_09-23_0f0789c_qwen3.5-4b/` |
| Same-day baseline | `d4b1626` | 06:55 | 391/404 = 96.8% | 183/192 | 96 / 2 / 3 | 3 | 0 | 2,030 s | `evaluation/2026-10-01_09-54_d4b1626_qwen3.5-4b_same-day-baseline/` |
| G3 baseline (2026-09-29, agents only) | `d4b1626` | | | 178/192 | | 4 | 0 | 1,518 s | `evaluation/2026-09-29_22-12_d4b1626_g3-baseline/` |

All non-agent categories score the same at both commits (advanced 64/68, basic 56/56, classification 20/20, intermediate 12/12, meta 20/20, reasoning 4/4, workflows 32/32; the two advanced PARTIALs `multi_level_stack` and `support_pipeline` are F-EXTRACT at both).

Examples whose score changed:

| Example | `d4b1626` same day | `0f0789c` | G3 baseline | Cause |
| --- | --- | --- | --- | --- |
| agents/plan_execute_recovery | 4 (136.6 s) | 1, F-LOOP, timeout at 180 s | 4 (123.4 s) | Code, not model noise: run alone 3 times per commit, `0f0789c` makes a 7-step plan, 15 iterations, 24 LLM calls, 24,368 prompt tokens (35-36 s) in every run; `d4b1626` makes a 4-step plan, 9 iterations, 16 calls, 14,884 prompt tokens (25.7 s) in every run. Both answer with `Success: True` when run alone. |
| agents/concurrent_react | 4 | 4 | 2 (F-EXTRACT) | Same at both commits today |
| agents/hierarchical_orchestrator | 4 | 4 | 1 (timeout) | Same at both commits today |

Read from the raw logs:

- Envelope leaks (`"field_name"`, `"extracted_data"`): 0 logs at either commit. `Continue.`: 0 logs at either commit. Tracebacks: 0 at either commit.
- `Success: True` / `Success: False` lines in agents logs: `0f0789c` 22 / 3 (architecture_review, eval_opt_structured, regulatory_compliance); `d4b1626` 22 / 4 (eval_opt_structured, legal_document_review, reflexion, regulatory_compliance). Every example printing `Success: False` still scores 4. No example prints a `stop_reason`.
- Timeouts at both commits: plan_execute (180 s), orchestrator_specialist (300 s), supply_chain_optimizer (300 s); plan_execute_recovery only at `0f0789c`. Timed-out logs hold no output.
- Total agents wall time: 4,597 s at `0f0789c`, 5,157 s at `d4b1626`.
- Step 24.1 (`e9ef064`, D-055): after the PlanExecute planner fix (no tool-less, no confirm/wait plan steps) the agents category alone scored 177/192 (`evaluation/2026-10-01_10-50_e9ef064_qwen3.5-4b/`, EVALUATE.md Run 008); `plan_execute_recovery` run alone fell from 24 to 20 LLM calls but still times out at 180 s under 4 workers, so D-044 (b) stays FAILED; `hierarchical_orchestrator` (code unchanged) timed out at 300 s this run.
