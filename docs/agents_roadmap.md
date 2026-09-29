# fsm_llm.agents: audit record and roadmap

Status: design record for the `fsm_llm.agents` audit, 2026-09-29.

- Code range: `c632893..HEAD` (plan `plan-2026-09-29T103145-06a5ec0a`, 26 steps plus one completion fix).
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
| 14 | `ef68b57` | `make_fresh_keys_handler` and typed field helpers (LOOP-02, LOOP-08, LOOP-11) |
| 15 | `6f704f3` | ReAct loop: typed fields, think-turn counting, `agent_feedback` (LOOP-01/04/05/06/09/10/16/17) |
| 15.1 | `901a8ee` | Completion fix: envelope-name collision, permissive terminate, core `_field_value` guard |
| 16 | `d6732cc` | `run_stream` yields only model text (LOOP-13) |
| 17 | `f5ce156` | Reflexion records each episode's grounded reflection (REACT-01/02) |
| 18 | `3340203` | PlanExecute replans on failed steps with typed plans (PAT-01/02) |
| 19 | `5805c42` | Debate rounds grounded; SelfConsistency votes on the answer (PAT-03/05) |
| 20 | `06e2dd0` | PromptChain gates stop the chain; MakerChecker/EvalOpt drafts grounded (PAT-06/11) |
| 21 | `e6a975d` | REWOO and Orchestrator report real outcomes and propagate budget errors (PAT-09/10) |
| 22 | `467df80` | AgentGraph topological order; Swarm keeps the task and exact handoff budget (PAT-07/08) |
| 23 | `01d433e` | VerifiedReact fails closed; `reason` tool gets the task; dead classification override removed (REACT-03/04/05) |
| 24 | `0146925` | `create_agent` pattern first; `AgentConfig.instructions`, `extra="forbid"`, `LLM_MODEL` (API-01/03, PAT-13) |
| 25 | `1fc874b` | Static `__all__`, dead constants removed, valid meta-builder example (API-05/06, META-06) |
| 26 | this commit | This record, docs sync, changelog, test counts (API-07) |

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
| LOOP-12 | TRUE | Fixed, step 3: run-output keys stripped from caller context; `observation_count` seeded 0 |
| LOOP-13 | PARTIAL | Fixed, step 16 |
| LOOP-14 | PARTIAL (needs `handler_timeout`) | Fixed, step 11 |
| LOOP-15 | NOT VERIFIED | Deferred to Track B |
| LOOP-16 | TRUE | Step numbers fixed, step 15. The per-turn thought is still empty: the typed `reasoning` field was removed in 15.1 (open) |
| LOOP-17 | TRUE | Message fixed, step 15 (cites the loop ceiling). No partial result on `BudgetExhaustedError` (deferred) |

Security:

| Id | At `c632893` | Outcome |
| --- | --- | --- |
| SEC-01 | TRUE (bypass reproduced; public `approval_granted` alone PARTIAL) | Fixed, step 3, also through `AgentServer`, SelfConsistency, Swarm and AgentGraph |
| SEC-02 | TRUE | Fixed, step 4 (denylist, not a whitelist) |
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
| TOOL-03 | TRUE (reproduced) | Fixed, step 7 |
| TOOL-04 to TOOL-13 | NOT VERIFIED | Deferred to Track B (ToolSpec) |
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
| REACT-08 | TRUE | Fixed, step 5 (construction warning) |
| REACT-09 | NOT VERIFIED | Deferred to Track B |
| REACT-10 | NOT VERIFIED | Deferred to Track B |
| REACT-11 | TRUE (thinking blocks not verified) | Malformed arguments fixed, step 8. Ignored `initial_context`/`api_kwargs` and `response_format` for a custom `complete_fn` are open |

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
| API-06 | TRUE | Fixed, step 25 (constants). `DecompositionError`/`ToolValidationError` kept (exported, no raise site): Track B deprecation |
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
- SEC-03 (D-004): the flag is a default policy only for a callback-only HITL; with a policy, the policy alone decides and is never ANDed with the flag.
- SEC-04 (D-005): refusal at construction and at `run()` instead of new HITL paths.
- SEC-11, roadmap D7 (D-006): D-030 empty-input recovery kept; no half-designed `read_only` flag ahead of ToolSpec.
- LOOP-04 (D-007, D-028): counting changed only in `AgentHandlers.check_iteration_limit`; the shared `make_iteration_limiter` is untouched. `max_iterations=N` gives N think turns and N - 1 tool turns, plus a forced stop a few transitions before the loop ceiling.
- LOOP-02 (D-008): one `make_fresh_keys_handler` for producing-state entry; never clears a forced `True` verdict.
- LOOP-01 (D-009): typed per-field extraction, but `think` with `use_classification=True` keeps its bulk fill.
- LOOP-06 (D-021, D-029): `agent_feedback` is cleared on think exit, not think entry (an entry clear erased it unread).
- REACT-02, PAT-01 (D-031, D-032): core picks the edge before PRE_TRANSITION runs, so the verdict handlers run on state entry, not at PRE_TRANSITION as planned.
- LOOP-14 (D-015): the grant is spent before the tool runs and recorded call-locally; handlers are not made `.critical()`.
- LOOP-08 (D-017): `agent_trace` is not capped or renamed (a data path, and core reads it); it is kept out of prompts instead.
- REACT-03 (D-010): the dead override is deleted, not repaired (a repair needs a core channel).
- API-04 (D-011, D-027): additive `stop_reason`; a public `forced_stop_reason` context key tells forced passes apart, because `API.get_data` drops internal keys.
- API-01 (D-012, D-047): `create_agent` keeps a legacy positional prompt with a `DeprecationWarning`. `AgentConfig.instructions` is prefixed as an `Agent instructions:` block to every non-empty state and field instruction, not written into `persona` (500-char core cap, and persona reaches only Pass 2).
- PAT-13 (D-013): `AgentConfig.model` stays a `str` with an env-reading default factory; more than 20 readers use it directly.
- PAT-04, PAT-10 (D-014, D-042): budget errors from sub-runs go into a call-local `RunEndingErrorHolder` and are re-raised after the loop; core's handler `continue` mode would swallow a plain raise.
- PAT-08 (D-018, D-045): Swarm transfer tools deferred; `next_agent` stays caller- or tool-written.
- PAT-07 (D-044): local Kahn sort in `AgentGraph`, not `fsm_llm.workflows.DependencyResolver` (agents does not depend on workflows).
- PAT-03, PAT-05 (D-036, D-037): the Debate answer is the conclude reply, `proposition` is only the success key; SelfConsistency votes on the last `Answer:` line.
- Generated text fields (D-036, D-043): live `qwen3.5:4b` returned null for text fields worded as "extract"; every generated text field now opens with a sentence telling the model to compose the value.
- Envelope-name collision (D-034, D-035): a typed field named `reasoning` was filled by core `llm._field_value` from the extraction envelope's own `reasoning`. The think `reasoning` field was removed, `_typed_field_extraction` rejects envelope names, and core `_field_value` no longer falls back to an envelope key. This is the one core change of the plan.
- E4 (D-019): the network block covers only the agents and meta suites.
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

Final G3: see REFLECT

## Known open items

- LOOP-16: the per-turn thought is not extracted, so `AgentStep.thought`, `ToolCall.reasoning` and `ApprovalRequest.reasoning` are empty unless `use_classification=True`.
- Typed per-field extraction costs one LLM call per field per turn, plus one retry per null required field.
- The ReAct-family success rule counts a failed tool call as success evidence. ParallelReact's empty batch loops until the limiter.
- HITL: the policy sees the call before empty-input recovery and a shallow context; a forced stop reached in `await_approval` skips the approved call (fails closed). REWOO, PlanExecute, ParallelReact and native_fc refuse gated tools instead of asking.
- native_fc ignores `initial_context` and `api_kwargs`, and a custom `complete_fn` never gets `response_format`.
- Swarm never hands off by itself: nothing shipped writes `next_agent`.
- SEC-11: D-030 recovery can pass the task text to a side-effecting single-parameter tool.
- REACT-05: reasoning-engine input is not length-capped. MEM-03/04: O(n^2) rewrite per `add` and embedding under the store lock.
- `DecompositionError` and `ToolValidationError` are exported but never raised.
- Examples to re-check in the final G3 (D-040): adapt, hierarchical_orchestrator, maker_checker_code.
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

- Runtime and accounting: usage and cost capture (OBS-01), monitor event bursts (OBS-02), timeouts inside a turn (OBS-03), `arun` and checkpoints (OBS-04), a richer `AgentTrace` (OBS-05), `Budget`/`RunState`/events, a partial result on budget errors (LOOP-17), LOOP-15.
- ToolSpec: TOOL-04 to TOOL-13, including side-effect and read-only hints that SEC-11 (roadmap D7) needs.
- Async core (roadmap D3) with a per-model executor (D6); parallel Orchestrator workers (PAT-10); ParallelReact stall detection (REACT-06); real HITL for REWOO, PlanExecute, ParallelReact and native_fc (lifts the SEC-04 refusal).
- Integrations: MCP v2 (INT-01 to INT-05), A2A v1.0 on the official SDK (INT-06, roadmap D4), AG-UI, OTEL.
- Guardrails: SEC-07, SEC-08, SEC-10, SEC-12.
- Memory v2: MEM-05 to MEM-07.
- Meta-builder: META-01 to META-05 and META-07.
- Swarm transfer tools (PAT-08), AutoMemory `respond` tool without mutating the caller's registry (REACT-07), REACT-09, REACT-10.
- Harness migration to the new runtime after Phase 4 (roadmap D8).

Deprecations (roadmap D5), deferred to Track B: ParallelReact, Debate and SelfConsistency, once the runtime offers replacements (D-020). Also candidates: the positional `create_agent(system_prompt, ...)` shim (warns since this change), `DecompositionError`, `ToolValidationError`.
