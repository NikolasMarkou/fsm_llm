# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — new package: `fsm_llm_harness` (extra: `pip install fsm-llm[harness]`)

An FSM-LLM-native emulation of the iterative-planner protocol: a 6-state
EXPLORE / PLAN / EXECUTE / REFLECT / PIVOT / CLOSE machine with mechanically
enforced gates, filesystem-as-memory artifacts, per-role file ownership, a
2-attempt autonomy leash, and a small-model hardening layer. Additive and
backward-compatible: **no Python file under `src/fsm_llm/` was modified**, and the
2-pass core contract is unchanged. The only other packages touched are
`fsm_llm_agents` (two files, listed under *Changed* below) and packaging/CI.

- **`HarnessAgent`** — the protocol driver. Builds the harness FSM, registers a
  handler per state entry, dispatches one worker role per entry, and owns all nine
  gate flags. Executor dispatches on one plan step are bounded by
  `max_fix_attempts * (1 + max_leash_grants)` for **any** sequence of approvals;
  the approval callback cannot raise it. The default approval callback DENIES, so
  an unattended run cannot approve its own plan or close itself.
- **`build_harness_fsm()`** — 6 states, 9 transitions. Every gate is a JsonLogic
  `TransitionCondition`, so a gated edge is DETERMINISTIC or BLOCKED, never an LLM
  judgement call, and every condition declares `requires_context_keys` so a
  garbled worker reply leaves the edge blocked rather than accidentally satisfied.
- **`artifacts`** — pydantic models and Markdown (de)serializers for 15 artifact
  kinds, the 9 decision entry-type schemas and the 6 Presentation Contracts, with
  strict grammars (`plan.md`'s 11 ordered sections, `decisions.md`'s
  `## D-NNN | PHASE | YYYY-MM-DD` header, `changelog.md`'s 8 pipe-delimited fields).
- **`storage.PlanDirectory`** — plan-id minting, atomic artifact writes
  (`mkstemp` in the target's own directory + `os.replace`), LESSONS `[I:N]`
  eviction, the SYSTEM line cap, the 4-plan cross-plan sliding window, and
  resumable run state read back from `state.md` itself.
- **`plan_validator`** — `pre_step_gate()` (4 slugs, ordered, short-circuit, all
  HARD) and `audit()` (30 structural checks; a check that raises is reported as an
  ERROR rather than suppressing the rest).
- **`tools`** — `Workspace` (confined source tree) and `PlanMemory` (confined
  **and** ownership-scoped plan directory) sharing one `resolve()` chokepoint,
  plus 13 agent-facing tools. `run_command` is off by default and `git` is
  deliberately absent from the command allowlist.
- **`roles`** — six frozen `RoleSpec`s derived from a single `OWNERSHIP` table, so
  a role's tool scope, its prompt text and its owned artifacts are one fact read
  three times. `build_default_worker_factory()` builds the stock worker, backed by
  `NativeFunctionCallingReactAgent`.
- **`hardening`** — `strip_model_noise`, `parse_json_payload`,
  `parse_role_output`, `coerce_worker_output`, `retry`. All fail CLOSED: a garbled
  reply is never retried into a pass.
- **`fsm-llm-harness` CLI** — `new` / `resume` / `status` / `validate` / `close`,
  with exactly three exit codes (`0` pass, `1` negative answer, `2` RESERVED for a
  HARD gate refusal). `close` is dry-run unless `--apply` and refuses to compress
  a directory with audit ERRORs.

**Gates read the filesystem, not the model's report.** `findings_count` is a count
of non-empty `findings/*.md` files; a dispatch that holds a write tool and claims a
write must show a tool call whose target now carries bytes; and a failed
observation leaves a gate value unchanged rather than writing a zero. This is the
package's central design commitment, and it is a response to measurement: a small
local model asserted completed code changes over an untouched workspace, and
claimed three findings over an empty directory. Prompt wording did not fix it.

**Status, stated as measured.** Offline the package is green (1,793 tests, `ruff`
clean, `mypy` 0 errors). Live on a local 4B model (`ollama_chat/qwen3.5:4b`), the
harness-level criteria pass — a full EXPLORE→CLOSE traverse whose plan directory
audits with zero ERRORs (3/3), the leash halting at exactly 2 attempts and not
resettable by an approving callback (6/6), a REFLECT→PIVOT→PLAN loop-back (3/3) —
and after the driver-assigned EXECUTE target fix (next section) the single-state
model-level criteria are MET at the untouched bars for the first time: write tool
issued 5/5 and workspace bytes 5/5 (bar >=4/5), strict sha256 content-hash match
4/5 (vs >=4/5), findings 5/5 (bar >=4/5). The new graded END-TO-END criterion on
real workers (L6, n=3) measured **0/3 against its floor and is NOT met**: two runs
halted honestly at the EXPLORE redispatch cap, one reached PLAN and stalled
sluglessly after an empty plan-writer reply (verified writes 3/3; no crashes).
This is recorded rather than rounded up: the package is not claimed to be
production-ready, and a small model is not claimed to drive it unattended to a
useful result.

### Added — harness measurement iteration: durable bench, driver-assigned EXECUTE targets, e2e criterion

- **`scripts/harness_bench.py` + `scripts/bench_data/`** — a durable, powered
  bench for harness capability claims: pre-registered fixed-n blocks (n=40/arm),
  6-field manifests (prompt-bytes hash, tool surface, fixture hash, model digest,
  arm, git commit), append-only raw jsonl rows, and a `report` subcommand that
  recomputes every k plus Wilson CI and Fisher exact (stdlib-only math) from the
  committed rows. Blocks are committed under `scripts/bench_data/` (tracked), so
  future numbers can be diffed — earlier benches lived in a gitignored scratch
  directory and no longer exist.
- **Seed determinism dispositioned by probe** — ollama honors `seed` for `:4b`
  (same seed byte-identical at temperature 0.7, different seed diverges; raw
  probe committed at `scripts/bench_data/seed-probe/probe.json`). `seed` is
  plumbed as an optional keyword-only parameter through
  `build_default_worker_factory` → `NativeFunctionCallingReactAgent`'s
  `litellm.completion` call site (default `None` = key absent, byte-identical
  call shape to before); per-row seeds are recorded in every bench row.
- **Driver-assigned EXECUTE write target** — baseline block B0 measured the
  wrong-ROOT defect: native EXECUTE dispatches content-matched the requested
  edit **2/40**. The fix extends the driver-assigned-target pattern to EXECUTE:
  `derive_execute_target` reads plan.md's Files To Modify and the dispatch names
  the exact target path + tool; an unparseable plan falls back to the previous
  prompt byte-identically. Post block B1, same manifest: **40/40** (Fisher
  p=1.6e-20). The ReAct control arm measured 0/40 in both blocks. The armed
  standing-bar classes were then re-run ONCE: L4 MET for the first time (write
  tool 5/5, bytes 5/5, strict content-hash 4/5 vs >=4/5), L5 MET 5/5;
  `MODEL_BAR=4` / `RUNS_MODEL=5` unchanged.
- **`TestL6EndToEndRealWorkers`** — the first graded end-to-end criterion on
  REAL role workers (n=3, disk-derived rubric vectors committed under
  `scripts/bench_data/l6-e2e/`, DENY-default disk-bound approval stub). Floor
  **NOT MET, 0/3** (two honest explore-cap halts at EXPLORE, one slugless PLAN
  stall; verified writes 3/3). Two structural findings recorded: EXPLORE over an
  empty plan directory clears the 3-findings gate ~1/3 of the time vs 5/5 on a
  seeded corpus, and PLAN has no redispatch budget, so one empty reply becomes a
  stall.
- **Adversarial audit executed, not just read** — 5/5 load-bearing guard
  mutations (leash-cap boundary, writable-key allowlist, empty-file gate
  counting, ownership deny branch, live-gate short-circuit) each flipped tests
  red in a scratch copy (93 red total); `test_cli.py`'s exit-code 0/1/2 contract
  close-read verdict: CLEAN.
- **Count-pinning tests** (`tests/test_packaging.py`) — documented test-count
  literals are checked against one `pytest --collect-only` subprocess, so doc
  drift now fails the suite.
- **Anchor hygiene** — 18 dead plan-ids retired via the skill's `retire` tool;
  plan-validator `[anchor-unknown-plan]` errors 155 → 0.

### Added — packaging and CI

- `harness` extra (pulls `fsm-llm[agents]`; no third-party dependencies of its
  own), included in `all`, in `make install-dev`, in `tox`'s `extras`, and in the
  CI install list.
- `fsm_llm_harness` added to the mypy target list, the coverage target list, the
  ruff `known-first-party` list, `package-data`, and `MANIFEST.in`.
- **`tests/test_packaging.py`** — derives the package list from the filesystem
  (`src/*/__init__.py`) and asserts every package appears in all 14 build/CI slots,
  so a future package that misses a slot fails loudly instead of silently. The one
  pre-existing gap (`fsm_llm_monitor` in `MANIFEST.in`) is a named, ratcheted
  exception rather than a weakened assertion.

### Changed — `fsm_llm_agents`

Both changes are gated so they are provably inert off Ollama; all pre-existing
`test_native_fc.py` tests pass unmodified in substance.

- **`NativeFunctionCallingReactAgent`** now applies `apply_ollama_params` /
  `prepare_ollama_messages` behind an `is_ollama_model` gate, recovers content from
  the reasoning trace only when there are no `tool_calls`, absorbs a malformed
  tool-call turn as a failed TURN (the loop breaks; the trace, the bytes already
  written and any answer survive) instead of losing the whole run, and gained an
  optional `system_policy` appended to its system prompt.
- **`AgentResult.success` is honest for `native_fc`** — it now requires a final,
  tool-call-free answer AND a loop that did not exhaust `max_iterations`. It
  previously returned `True` for a run that called tools, produced nothing and
  concluded nothing, which made it useless as a caller's failure signal.
- **`NativeFunctionCallingReactAgent` structured output** — when
  `AgentConfig.output_schema` is set and the free-text answer does not validate,
  exactly ONE additional completion is made carrying `response_format=` and **no**
  `tools=`. The two are never stacked in one call. Previously `output_schema` was
  silently inert on this agent's `run()` path.
- **`base._output_response_format(schema)`** extracted from `_init_context` and
  shared with the above, so there is one response-format envelope builder rather
  than two that can drift.

## [0.5.0] - 2026-07-21

### Added — agent layer additive improvements (`fsm_llm_agents`)
All of the following are **additive and backward-compatible**: existing agents,
examples, signatures, and the 2-pass core contract are unchanged. New optional
`AgentConfig` fields default to prior behavior.

- **`ToolRegistry.get_json_schemas()`** — OpenAI-compatible function-calling
  tool schemas (closes a documented-but-missing method gap).
- **`CachingToolRegistry` / `RetryingToolRegistry`** — drop-in `ToolRegistry`
  subclasses adding result memoization and retry-on-failure.
- **`AgentConfig`** new optional fields: `max_history_size`, `enable_prompt_cache`
  (litellm response caching), `reflect_every_n`, `auto_summarize_after`,
  `verification_fn`.
- **`SelfConsistencyAgent(max_workers=...)`** — opt-in parallel sampling
  (default 1 = unchanged serial; results assembled in order, deterministic).
- **`SemanticMemoryStore` + `create_semantic_memory_tools`** — embedding-backed
  long-term memory with cosine recall, JSON persistence across sessions, and an
  offline substring fallback.
- **`AutoMemoryReactAgent`** (+ `augment_task_with_memories`,
  `remember_interaction`) — automatic recall-before / remember-after at the
  `run()` boundary, removing the model-must-call-the-tool dependency.
- **`MemorySessionStore` + `save_working_memory` / `load_working_memory`** —
  persist `WorkingMemory` alongside FSM session state.
- **`BaseAgent._standard_run_stream` + `ReactAgent.run_stream`** — stream the
  final answer token by token via `API.converse_stream`.
- **`ParallelReactAgent`** — ReAct variant that extracts and dispatches multiple
  tool calls per step concurrently.
- **`VerifiedReactAgent`** — verify-and-retry via `config.verification_fn` plus
  periodic self-reflection via `config.reflect_every_n`.
- **`make_observation_summarizer`** — condenses old observations instead of
  hard-dropping them (wired in when `config.auto_summarize_after` is set).
- **`react_worker_factory` + `default_llm_judge`** — composition helpers
  (Orchestrator+ReAct worker; built-in LLM-as-judge `evaluation_fn`).
- **`NativeFunctionCallingReactAgent`** — self-contained ReAct loop using
  provider-native `tools=`/`tool_calls` (litellm) instead of JSON-in-prompt.

## [0.4.0] - 2026-05-29

### Security
- **litellm supply chain compromise**: litellm versions 1.82.7 and 1.82.8 were compromised
  with credential-stealing malware via `.pth` file injection. These versions are now
  explicitly excluded from the dependency specification (`!=1.82.7,!=1.82.8`).
  - **Impact**: Any Python invocation in an environment with the compromised versions would
    exfiltrate environment variables, SSH keys, AWS credentials, Kubernetes configs, and git
    credentials to an attacker-controlled server. No import of litellm was required.
  - **Action for users**: If you installed litellm 1.82.7 or 1.82.8 at any time, treat all
    credentials in that environment as compromised and rotate them immediately.
  - **Current status**: PyPI has quarantined the entire litellm package. Existing installs of
    safe versions (<=1.82.6) continue to work.
- Added `.pth` file audit in CI pipeline and local `make audit` / `scripts/audit_pth.py`
- Added `constraints.txt` for dependency version locking in dev/CI builds

### Changed
- **Skip Pass 2 for intermediate agent states** — States with `response_instructions=""` now skip
  the response generation LLM call entirely. The pipeline sends a minimal sentinel to the LLM
  interface (for cycle tracking) and the real LLM returns immediately without an API call. This
  halves the number of LLM calls for agent iterations, cutting wall time ~50% and eliminating
  all F-LOOP timeout failures. Applied to: think/act (ReAct, Reflexion), evaluate (EvalOpt),
  check (MakerChecker).
- **Stall detection threshold** reduced from 3 to 2 consecutive no-tool iterations before
  forced termination, saving ~20s per stall event.

### Fixed
- **MakerChecker quality_score extraction** — When the LLM embeds quality_score inside the
  checker_feedback dict instead of as a separate context field, `_track_revisions` now recovers
  it from the dict. Previously quality_score defaulted to 0.0, forcing max revisions.
- Evaluation health score improved from 95.7% to **100%** (70/70 PASS) on `ollama_chat/qwen3.5:4b`.
- **Comprehensive code-review hardening** — four deep static-review passes across all five packages
  fixed ~75 issues: 2-pass / locking / streaming-rollback bugs in core; handler-timing and
  budget/iteration-limiter bugs across the 12 agent patterns; async workflow-engine event/timeout/retry
  races; meta-builder, reasoning, and monitor fixes; sibling-class propagation of budget/timeout re-raise
  guards; and recursion-safety (recursive→iterative DFS) in workflow, agent-graph, and FSM-validator
  cycle detection. Tracked via the `*-NEW-*`, `AI3-*`, `RW3-*`, `AG-*`, `RWM-*`, and `FA-*` issue IDs in
  the git history.

### Added
- **BaseAgent ABC** for all 12 agent implementations — shared conversation loop, budget enforcement,
  answer extraction, trace building, context filtering, and `__call__` syntax (`agent("task")`)
- **Enhanced `@tool` decorator** — supports bare `@tool` (no parentheses) with auto-schema inference
  from type hints (`str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`, `dict→object`).
  Supports `typing.Annotated[T, "description"]` for per-parameter descriptions. Backward compatible
  with explicit `parameter_schema` overrides.
- **Structured output** — `AgentConfig(output_schema=PydanticModel)` validates agent answers against
  Pydantic models. Parsed result stored in `AgentResult.structured_output`. Graceful fallback on
  validation failure.
- **`create_agent()` factory** — create agents in one line: `create_agent(tools=[search], pattern="react")`
- **`ToolRegistry.register_agent()`** — register agents as tools for supervisor/orchestrator patterns
- **`AgentResult.__str__`** — returns structured_output if available, else raw answer
- `ollama.py` module — centralized Ollama helpers for structured output compatibility
  - `is_ollama_model()` — model detection
  - `apply_ollama_params()` — disables thinking via `reasoning_effort="none"`, forces `temperature=0` for structured calls
  - `build_ollama_response_format()` — builds `json_schema` response format with extraction/transition schemas
  - `EXTRACTION_JSON_SCHEMA`, `TRANSITION_JSON_SCHEMA` — JSON Schema constants for structured output
- `fsm_llm_agents` extension package for ReAct and Human-in-the-Loop agentic patterns
  - `ReactAgent` — ReAct loop agent with auto-generated FSM from tool registry (think → act → observe → conclude)
  - `ToolRegistry` — tool management with schema descriptions, prompt generation, and execution
  - `HumanInTheLoop` — configurable approval gates, confidence-based escalation, and human override
  - `@tool` decorator for simple tool registration
  - Pydantic models: `ToolDefinition`, `ToolCall`, `ToolResult`, `AgentStep`, `AgentTrace`, `AgentConfig`, `AgentResult`, `ApprovalRequest`
  - `AgentError` exception hierarchy (7 error types)
  - 109 unit tests across 8 test files
- `has_agents()` / `get_agents()` extension checks in `fsm_llm`
- `MessagePipeline` class extracted from FSMManager — encapsulates all 2-pass message processing
- `context.py` module extracted from FSMManager — stateless context cleaning utilities
- `ConversationStep` added to workflows — embeds full FSM conversations within workflow steps
- Handler execution timeout support (`DEFAULT_HANDLER_TIMEOUT = 30s`)
- Workflow step async timeout support (`DEFAULT_STEP_TIMEOUT = 120s`)
- Workflow-level timeout, conversation timeout, and event listener expiration
- `critical` flag on `BaseHandler` — errors always raise regardless of error_mode
- `FORBIDDEN_CONTEXT_PATTERNS` enforcement for password/secret/token key filtering
- 5 new examples combining sub-packages (reasoning, workflows, classification)
- 20 new complex examples (70 total) focused on agentic patterns and meta builders:
  - **Agents (14)**: debate_with_tools (evidence-based debate), reflexion_code_gen (self-improving code
    generation with test runner), orchestrator_specialist (multi-specialist ReactAgents), pipeline_review
    (PromptChain + MakerChecker QA), adapt_with_memory (ADaPT + WorkingMemory), rewoo_multi_step (complex
    multi-dependency planning), eval_opt_structured (EvaluatorOptimizer + Pydantic validation),
    plan_execute_recovery (replanning on tool failure), consistency_with_tools (SelfConsistency for
    multi-step reasoning), maker_checker_code (code review pattern), hierarchical_orchestrator (nested
    multi-level delegation), agent_memory_chain (multi-task continuity via WorkingMemory),
    react_structured_pipeline (ReAct → structured output → PromptChain), multi_debate_panel (parallel
    debates with synthesis)
  - **Meta (4)**: build_workflow (interactive workflow builder), build_agent (interactive agent builder),
    meta_review_loop (FSMBuilder + MakerChecker quality review), meta_from_spec (programmatic
    FSM/workflow/agent from text specs)
  - **Workflows (2)**: conditional_branching (condition-based routing), workflow_agent_loop (quality-gated
    agent execution with retry)
- Automated evaluation baseline: 95.7% health score (70 examples, ollama_chat/qwen3.5:4b)
- Tests for MessagePipeline, handler timeout, step timeout, context, logging, runner, LiteLLMInterface
- Audit verification tests across all packages

### Changed
- Ollama structured output uses `json_schema` response format instead of `json_object` for grammar-constrained output
- Ollama thinking mode disabled via `reasoning_effort="none"` (litellm >=1.82 maps this to Ollama's `think: false`)
- Ollama structured calls (data extraction, transition decision) force `temperature=0` for deterministic output
- Classification `Classifier._call_llm()` now applies Ollama params via shared `fsm_llm.ollama` helpers
- Minimum litellm version bumped from 1.68.1 to 1.82.0 (required for proper Ollama `think` parameter forwarding)
- FSMManager delegates message processing to MessagePipeline
- `push_fsm`/`pop_fsm` decomposed into focused sub-methods
- `evaluate_logic()` refactored with dispatch pattern
- Runner refactored to use API; workflows drops phantom FSMManager dependency
- Exception handling standardized across codebase (chaining with `from e`)
- Regex patterns pre-compiled for performance
- Test fixtures deduplicated across test suites
- mypy enforcement enabled in CI with pydantic plugin
- All 118 mypy errors fixed

### Removed
- `fsm_llm_classification` deprecation shim package (use `from fsm_llm import Classifier` directly)
- `LLMInterface.decide_transition()` deprecated method
- `LLMInterface.extract_data()` deprecated method
- `FSMManager` `transition_prompt_builder` parameter
- `WorkflowEngine` `fsm_manager` and `llm_interface` parameters
- `DataExtractionRequest` class
- `State._coerce_and_warn()` boolean coercion for `transition_classification`
- `State` `instructions` field deprecation warning
- `has_classification()` and `get_classification()` helper functions
- Empty `fsm_llm_workflows.handlers` compatibility shim
- 7 forwarding methods from FSMManager (moved to MessagePipeline)
- Dead workflow handler code (AutoTransitionHandler, EventHandler, TimerHandler)
- Dead code and empty extras across multiple packages

### Fixed
- Ollama/Qwen3 thinking mode corrupting structured JSON output (ollama/ollama#10538)
- Integration test `test_pre_processing_handler_fires` using wrong HandlerBuilder API (`.on_timing()` → `.at()`, `.execute().build()` → `.do()`)
- Race condition in conversation lock retrieval
- Conversation lock leak with cleanup methods
- Event listener race condition in workflows
- Confidence collapse with additive boost in classification
- MockLLM2Interface crash on empty transitions
- Classifier thinking hacks and multi-intent prompt mismatch
- Classification confidence handling and dead code
- Workflow step error paths and type safety
- Workflow engine safety issues
- Security gaps in handlers, context, and prompts
- JSON regex fallback validation (requires meaningful keys)
- Multi-key JsonLogic expression error
- Reasoning engine bugs and magic number extraction
- Algorithm and logic issues across codebase

### Security
- Safety limits and validation guards added
- Security gaps fixed in handlers, context, and prompts
- Context key security filtering (internal prefixes, forbidden patterns)

## [0.3.0] - 2026-03-19

### Added
- `fsm_llm_classification` extension package for LLM-backed structured classification
  - `Classifier` for single-intent and multi-intent classification
  - `HierarchicalClassifier` for two-stage domain-then-intent classification (>15 classes)
  - `IntentRouter` for mapping classified intents to handler functions with low-confidence fallback
  - Pydantic models: `ClassificationSchema`, `IntentDefinition`, `ClassificationResult`, `MultiClassificationResult`, `HierarchicalSchema`
  - Prompt and JSON schema builders with reasoning-first field ordering (mitigates constrained-decoding distortion)
  - Structured output support via `response_format` when the LLM provider supports it
- `has_classification()` / `get_classification()` extension checks in `fsm_llm`
- 39 unit tests for classification package
- Classification extension documentation (README, examples, architecture docs)
- `timeout` parameter on `LiteLLMInterface` (default 120s) to prevent indefinite hangs on network issues
- `pytest-mock` added to dev extras in pyproject.toml
- `[tool.ruff.lint]` configuration in pyproject.toml to suppress false E402 from `__future__` annotations
- 21 regression tests for codebase review fixes (`test_regression_review.py`)
- 15 new `ContextKeys` constants for reasoning sub-FSM result keys (deductive, inductive, abductive, analogical, critical, hybrid)

### Fixed
- Version number aligned to 0.3.0 across `pyproject.toml` and `__version__.py` (was still 0.2.1)
- Context pruning log now reports actual new size instead of repeating the original size
- Hard-coded context keys in `merge_reasoning_results` replaced with `ContextKeys` constants (prevents silent `None` on key mismatch)
- Duplicate `import re` removed from `llm.py` `_make_llm_call()` (leftover from Qwen3.5 workaround)
- Extraction parse failure now returns `confidence=0.0` instead of `0.5` (callers can distinguish failure from low-confidence extraction)
- `requirements.txt` aligned with `pyproject.toml` core deps (removed dev deps, fixed `python-dotenv` version pin)

### Removed
- Unused async handler support from `handlers.py` (asyncio import, `AsyncExecutionLambda` type, `is_async` detection, ThreadPoolExecutor fallback) — no async handlers existed in the codebase
- `MergeStrategy` alias from `reasoning/constants.py` — engine now imports `ContextMergeStrategy` directly
- Dynamic `__all__.extend()` / `__all__.append()` calls from `__init__.py` — consolidated into single `__all__` definition
- Dead `[testenv:docs]` sphinx environment from `tox.ini`

## [0.2.1] - 2026-03-19

### Added
- `[tool.pytest.ini_options]` in pyproject.toml
- `[tool.mypy]` configuration in pyproject.toml
- Python 3.12 support in CI and tox
- CHANGELOG.md (this file)
- examples/README.md with example index and learning path

### Changed
- Python minimum version updated to 3.10 (was 3.8)
- Package-data now includes `fsm_llm_reasoning`
- Pre-commit hooks replaced: pytest-on-commit removed, ruff + standard hooks added
- Makefile expanded from 3 to 8 targets (added help, lint, format, type-check, install-dev)
- CI workflow installs from pyproject.toml instead of requirements.txt
- tox.ini aligned with CI (consistent flake8 config, added mypy env)

### Fixed
- CLI entry point now correctly resolves `fsm-llm` command
- Exception chaining (`from e`) added to all catch-and-reraise blocks for proper traceback preservation
- `__main__.py` docstring placement (was after imports, not recognized by Python)
- Workflows package version now imported from main package instead of hardcoded
- LLM interface log levels demoted from INFO to DEBUG (less noisy)
- Input validation added to `LiteLLMInterface` (model, temperature, max_tokens)

## [0.2.0] - 2026-03-18

### Changed
- Project renamed from `llm-fsm` to `fsm-llm` across all packages, tests, docs, and examples

## [0.1.0] - 2026-03-07

### Added
- Initial release with 2-pass architecture
- FSM stacking with push/pop operations
- Handler system with builder pattern
- JsonLogic expression evaluator
- LiteLLM multi-provider support
- CLI tools: fsm-llm, fsm-llm-visualize, fsm-llm-validate
- 7 examples (basic, intermediate, advanced)
- Comprehensive documentation

[0.5.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NikolasMarkou/fsm_llm/releases/tag/v0.1.0
