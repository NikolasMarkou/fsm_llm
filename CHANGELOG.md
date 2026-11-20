# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An opt-in durable event log (roadmap rows 1.1 to 1.3, first slice, plan 21e6f00d), in the new layer package `fsm_llm.runtime` (imported by `fsm_llm.engine`, so `import fsm_llm` now loads it; it imports only `core`, `llm` and the kernel, and the 32 import contracts are kept with no new debt). Nothing changes while the log is off: no file is written, no model call is added, the snapshots and the lock order are the same. No wire string changed, so there is no `**BREAKING:**` entry.
  - `API(event_log=)` takes an `EventLog` (shared, left open by `close`) or a directory path (the API owns the log and closes it); `APIBuilder.set_event_log` sets it. Every turn of a conversation tree is appended to one file, `logs/<root conversation id>.jsonl`, as versioned events (`v`, `seq`, `ts`, `conversation_id`, `frame`, `type`); a payload over 8,192 bytes is a content-addressed blob, `blobs/<aa>/<62 hex>`. `seq` is the order, never `ts`. The vocabulary (`runtime.EventType`, 21 declared types, 13 written today) includes `turn/start` and `turn/end` with an outcome (`committed`, `rolled_back`, `abandoned`, `interrupted`), `llm/request`, `llm/response`, `provider/error`, `executor/start`, `executor/result`, `transition`, `stack/push`, `stack/pop`, `context/record`, `tool/call` and `tool/result`.
  - `fsm_llm.runtime` holds `EventLog` (append, batch, read and query, `tool_call`), the JSONL store (unbuffered append, torn-tail recovery on open, an advisory lock per log, a bounded reader), `BlobStore` (sha256, write-if-absent, re-hash on read), the checkpoint rule and the turn scope the engine reaches the log through. The log is written and fsynced before every model call and before a side-effecting tool call; a log that cannot be written refuses the call instead of running it unlogged.
  - The `llm` layer gains a request observer (`fsm_llm.llm.observer`: the `RequestObserver` Protocol and a scoped `ContextVar`) called from `LiteLLMInterface._send`, so the log holds the exact provider request of every call that reaches the one send path (the Ollama rewrites and the apology retry as their own requests), through an allow-list of keys that never carries credentials, headers or timeouts. The observer is not a counter: the meter and its order in `_send` are unchanged. An interface that never reaches `_send` (a custom one, a `fsm_llm.testing` double, a mock) is logged by the engine at the call site instead, never twice.
  - `fsm_llm.engine.session.EventLogSessionStore(log, *, legacy_dir=None)` and `fsm_llm.runtime.stores.session.SessionProjection` (version 2, with one adjacent migration from version 1 and a reader for old `FileSessionStore` files, bounded to 32 MiB): `API(session_store=EventLogSessionStore(log)).restore_session(old_id)` starts a new conversation from the state of the last committed turn of the log, or the pre-turn state of a turn that never ended (a `kill -9` in the middle of a turn). A completion state restored mid tool exchange continues from the transcript the log carries (only the transcript and response-format keys the current definition declares; any other internal key, above all the approval grant, is refused), and a tool call with no result is reported as `unknown` in the session metadata and is never retried. `FileSessionStore` snapshots are unchanged and a restore from a snapshot still refuses a mid exchange transcript.
  - Known limits: the greeting of a new conversation is not logged as a request (its conversation id is minted after the call; the start record carries the state after the greeting, so the greeting text is in the log, D-021). The log sits beside the snapshots: snapshots stay authoritative (a snapshot restores the root frame only; restoring a pushed FSM stack, `fork` and `resume` from the log are the next entry, roadmap row 1.4). User text and prompts are stored as the model saw them, so keep the log directory as private as the session directory; only context entries that are secret-looking by the repository rule (`is_forbidden_context_entry`) are redacted, and a `*_token` key whose value is not token-shaped is not secret-looking by that rule. Power-loss durability (fsync of the file and, on POSIX, of the directory) is argued, not tested. The lock, the raw file descriptor and `Popen.kill()` are proven on Linux only; their Windows behavior is covered by the CI Windows legs. No product tool call writes `tool/call` by itself yet: `ToolRegistry` and `native_fc` are pinned by a recorded bench, so a consumer calls `EventLog.tool_call` for now. Rebuilding a prompt from the layout is not claimed (rows 1.9 and 2.3).
  - More known limits: a secret-looking value is stored as the placeholder `<redacted>` and a restore from the log puts the placeholder back as the value, so a `missing` gate or a `has_context` check sees the key as present with a fake value. A JSON schema stored in the request blob of an interface that does not reach `_send` has the `properties` it names like a secret renamed by the same filter (a restore never replays a request blob, so only the audit copy is affected). Tool-call arguments in a stored transcript are redacted by entry name when they are a JSON object; a restored transcript then replays the placeholder. Every conversation that is appended to keeps two file descriptors and the file's lock until it ends (`end_conversation`), its id is released, the API or log is closed, or the dropped writer is collected (its finalizer closes it): a long-running host must end its conversations or close the API. A failed write poisons the writer of that tree until the id is released or the log is closed, and a late append of a straggler thread after the release reopens a writer. A reopen right after a crash can briefly see the log locked on Windows, which frees the lock of a terminated process asynchronously (the tests retry there; unverified until the Windows run). A turn is labelled `rolled_back` or `abandoned` by comparing the state at its end with its start, history and summary included. Power loss is not covered beyond the fsync argument above.
- `API.fork`, `API.resume` and stack restore from the event log (roadmap row 1.4, plan 5a6a1f80; both need `API(event_log=)` and raise `FSMError` without one). `api.fork(conversation_id, *, at_turn)` starts a NEW conversation (a minted id) from the state of the source at the end of its Nth turn and returns `(new_conversation_id, SessionState)` like `restore_session`; `at_turn` counts the `turn/end` events of every frame in log order (a rolled-back turn counts, the greeting is not a turn, `0` is the state before the first turn), and an `at_turn` that is not an int of 0 or more, is past the last finished turn or names an unfinished one raises `ValueError`. `api.resume(conversation_id)` continues the SAME conversation on the SAME log after a crash or a close (same ids, `seq` continues, the stack rebuilt with its own ids): the writer lock is taken before the log is read (another process or `EventLog` holding it is `LogLockedError`, an id live in this API is refused), an unknown id is plain `FSMError` and creates no file, tool calls the log could not resolve are written once as `tool/result` with `status: "unknown"`, listed in `state.metadata["unknown_tool_calls"]` and never re-run, and a killed turn gets no closing `turn/end` (the next `turn/start` supersedes it). Both rebuild a pushed FSM stack (`SessionState.stack_depth` is the real depth) through the one restore body `restore_session` uses, and `stack/push` and `stack/pop` now carry what the rebuild needs (the child's post-greeting `state_ref`, `return_context_ref`, `shared_context_keys`, `preserve_history`, the parent's post-merge `state_ref`); the new `fsm_llm.runtime.stores.stack.fold_stack` is depth-indexed with a written recovery rule (a duplicate or lost record never shifts a frame, a frame with no covering push is `LogCorruptError`) and `turn_cut` is the turn index. A frame above the root is trusted only through its definition: it must load and its recomputed `fsm_id` must equal the logged one, only keys the definition declares are seeded, a transcript is accepted only under the default `messages_key` name and `response_format_key` values are not restored, and every internal-prefixed key a restore reads from the log (a frame's return context and shared key names too) is dropped, so a forged log cannot carry the driver-only grant `_approval_granted` or `_run_output_keys`; public keys of a forged root blob are seeded as given, like a plain snapshot, because the log is trusted as much as the session directory (a child mid structured-output turn does not continue). A pushed frame's own `max_steps` and `max_seconds` ride its `stack/push`, so a restored frame keeps them with a fresh count, and a symlinked log is refused by `resume`. The fork's own log is self-sufficient (a `fork` record and a `restored` push per frame), so a fork of a fork resumes. New names: `API.fork`, `API.resume`, the read-only `API.event_log` property, `EventLog.open_existing` and `has_writer`. Proof: a sweep over every event offset and every torn-tail length of a stacked scenario against the state observed live, one real `Popen.kill()` resume at frame 1, two processes resuming one id (exactly one wins) and a fork while the source appends; the Windows behavior of the lock after a kill is unverified until the Windows legs.
- Core turn budgets (roadmap row 1.7): a lifetime `max_steps` and `max_seconds` per conversation and per pushed frame, off by default (`None` changes nothing). New keyword-only arguments: `API(max_steps=, max_seconds=)` (the default for the root and every frame of that API), `API.start_conversation`, `API.push_fsm`, `FSMManager.start_conversation`, and `APIBuilder.set_max_steps` and `set_max_seconds`; `run_until_terminal(max_steps=)` stays the per-call budget. One private record per manager conversation id lives outside `context.data`, `metadata`, handlers and the rollback snapshot, so nothing a model can extract and no rollback can reset or refund it; it is charged in the turn entry before any model call (`converse`, `advance` and both streams; a rolled-back turn counts, the greeting does not, a stream counts at its first `next()`). A spent frame raises the existing `RunBudgetExceededError` on its next turn, undoes its claim (no lock or active-turn residue), appends one `decision` event (`kind: "budget_spent"`; `decision` is now a produced event type, 14 of the 21 declared) and is never virtually ended or popped by the engine. `FSMManager.turn_budget_spent` and `turn_budget_steps` read the record. A spent count is NOT persisted: `fork` and `resume` rebuild a pushed frame with the limits its push logged and a fresh count (the API default fills a limit it never had), and `restore_session` starts fresh from the API defaults (carried: persisting a spent count).
- Usage as a projection (roadmap row 1.13): `fsm_llm.runtime.usage` folds the `llm/response` and `provider/error` events of a log into the same `LLMUsage` the interface meter reports, through one record shape and one `fold` (`fsm_llm.llm.usage.UsageRecord`, `fold`, `usage_kind`, `usage_counts`), and answers "provider errors since sequence N for this root and frame". `llm/response` events carry inline `prompt_tokens`, `completion_tokens` and `total_tokens` integers and a `stream` flag. A reader takes ONE source (the log or the ledger, never their sum). Cost is not here (the model catalog is roadmap row 2.1). Known differences: an interface with no meter and no log still reads 0 outages, a stream that errors mid-read is one error in the projection and an answered call in the meter, and engine-logged `greeting` calls fold as kind `complete`.
- Privacy levels and a redaction hook for the event log (roadmap row 1.11): `EventLog(root, *, privacy=PrivacyLevel.VALUES, redact=None)`. Level `shape` (`fsm_llm.runtime.PrivacyLevel`) writes no blob and no `*_ref` and keeps only a per-type allow-list of fields (ids, names, counts, outcomes, codes, sorted `argument_keys`), marks every event with the reserved field `lvl: "shape"`, and still tells `rolled_back` from `abandoned`; a shape log, and a values log a shape writer continued, are unrestorable, unforkable and unresumable by design (`fold_session`, `fold_stack`, `fork` and `resume` raise the new `LogPrivacyError`, an `FSMError`). The default stays `values`, so the log holds user text and prompts as the model saw them. The hook `redact(kind, value) -> value` (`RedactionKind`: `prompt`, `response`, `user_message`, `state`, `delta`, `arguments`, `definition`) runs inside the log after the built-in secret filters and before encoding, gets a deep copy, must be pure and deterministic, and fails closed: an exception, a non-JSON or too deep result refuses the checkpoint and so the model call, and `put_blob` of raw bytes is refused while a hook is set. Honest limits: a free-text secret (a card number typed into a message) reaches disk at level `values` unless the hook handles it; a hook that rewrites a `state` or `definition` blob can break a later fork or resume; a shape log still says which states a conversation visited; keep the log directory as private as the session directory at either level.
- `EventLog.prune(*, max_age_seconds=None, max_logs=None, blob_grace_seconds=3600.0, dry_run=False) -> PruneReport` (roadmap row 1.11): explicit retention, no thread and no automatic call, and with no criterion nothing is read or deleted. A log's age is its file mtime (never `ts`); a log whose writer lock is held is skipped and counted (`logs_busy`); `.lock` files are never unlinked. Blobs go afterwards by mark and sweep: any blob a remaining log or a live blob names stays (so blobs a fork shares survive the source log), the rest older than the grace goes, and a re-put of an old blob refreshes its mtime. `PruneReport` is frozen (`logs_removed`, `logs_busy`, `blobs_removed`, `blobs_young`, `bytes_freed`). A resumable crashed conversation that is old enough IS deleted when the caller asks; an old blob re-put between the sweep's stat and its unlink is lost and the reader then raises `LogCorruptError` (fail closed); a refused unlink (`PermissionError`) is skipped and counted, and its Windows behavior is unverified until the Windows legs. The log stays JSONL with versioned events: compression of blobs (zstd) and an SQLite store are deferred (D-030), so row 1.2 stays "JSONL, versioned, migrations".
- `ReasoningEngine.close()` and the context-manager protocol (plan 52ce5464, audit item F1): `close()` ends every conversation still open on the orchestrator and the classifier `API` (and closes an event log the engine opened itself) and is safe to call twice; `with ReasoningEngine(...) as engine:` calls it on exit. An interrupt during `solve_problem` now cleans up on its own, see Fixed.
- `EventCollector.forget_conversations(conversation_ids)` (monitor, plan 52ce5464, audit item F6): drops conversation ids from the active-conversation metric under the collector's lock; `InstanceManager.attach_api` calls it for the conversations of the API it replaces.
- Node probes for the monitor frontend (plan 52ce5464, audit items G1 to G6): `tests/test_fsm_llm_monitor/test_static_js.py` runs small ES-module probes (`static_probes/*.mjs`) on a DOM stand-in, with no bundler, npm or network. They skip when `node` 18 or newer is not on PATH, and a test fails on a Linux GitHub Actions leg that has no usable `node`, so the probes cannot go quiet in CI.
- `fsm_llm.aio`, for interface authors (plan 51fde95b, async-first layer, no behaviour change for a sync caller beyond two noted below, the stream close and the `function` field of JSON log lines): `run_sync(coroutine)` runs an `async def` body to its end on the calling thread with no event loop (a coroutine that really suspends there raises `SuspendedInSyncModeError`), `iterate_sync(async_generator)` does the same for a stream, `aiterate(source)` is the reverse, `acall(obj, "name", *args)` calls the sync method in sync mode and awaits the `a` twin in async mode (a sync-only object runs on a worker thread), and `SYNC` is the flag between them. It is a kernel module (standard library only), so `import fsm_llm` now loads it, and the import contract and the laziness test list it.
- Non-abstract async twins on `LLMInterface`: `acomplete`, `agenerate_response`, `agenerate_response_stream`, `aextract_field` and `aextract_bulk_data`. The default of each runs the sync method on a worker thread (`asyncio.to_thread`, inline in sync mode so `run_sync` and `iterate_sync` work on it; the stream steps its generator one `next` per worker call and closes it when the stream ends, fails or the reader stops early, and when a cancel or a timeout lands during a step it closes it once that step ends), so a sync-only interface already works from async code. A cancelled await returns at once but does not stop that thread, so a late result can land after the caller gave up, and concurrent awaits run the sync method on several worker threads at once. An interface that implements only the `a*` methods cannot serve sync callers: the sync methods stay abstract or raise `NotImplementedError`. A `Mock(spec=LLMInterface)` makes every `a*` an `AsyncMock` whose awaited result is a child mock that ignores `complete.return_value`: set the `a*` return value, or use a double of `fsm_llm.testing`.
- Native async on `LiteLLMInterface` over litellm's `acompletion`: `acomplete`, `agenerate_response`, `agenerate_response_stream`, `aextract_field` and `aextract_bulk_data` are the one body of each method, and the sync name is the thin `run_sync` (or `iterate_sync`) wrapper over it, so a sync caller sees the same results, errors, request observer events and usage meter, and a patch of `completion` still intercepts the sync names. Both transports go through one send family that feeds the meter and the request observer; a request cancelled in flight is reported to the observer (`provider/error`) and counted on the meter as an error, and `CancelledError` is re-raised. Closing an async stream early closes the provider stream, and the sync `generate_response_stream` now closes the provider stream too when the stream ends, fails or is closed early (it only iterated it before). A subclass of `LiteLLMInterface` that overrides a sync twin (for example `generate_response`) without its async twin gets a `RuntimeWarning` at its class statement, because an async caller reaches the inherited native twin and bypasses the override; overriding an `a` twin alone is not warned.
- `LiteLLMEmbedder.aembed` over litellm's `aembedding` (`embed` is the `run_sync` wrapper over it, and a patch of `embedding` still intercepts `embed`), and `Classifier.aclassify`, `Classifier.aclassify_multi` and `HierarchicalClassifier.aclassify` (the sync names are `run_sync` wrappers over them; a patch of `Classifier._call_llm` or of the interface's `complete` still intercepts the sync names). An async caller bypasses a subclass override of `classify`, `_call_llm` or `embed`. The `function` field of a log record (the JSON log sink writes it) for these bodies now reads the async name (`agenerate_response`, `aclassify`, `_acall_llm`, `aembed`), also for a sync caller.
- The public test doubles `MockLLM2Interface`, `PromptGroundedLLM` and `ScriptedLLM` answer the async twins on the calling thread (no worker thread), with the same results, records, meter counts and errors as their sync methods.
- `ProcedureRunner.run_stream(conversation_id, *, max_steps, max_seconds=None)` and `ProcedureRunner.converse_stream(conversation_id, message)` (plan 51fde95b step 11.2): the streaming forms of `run` and `converse`, thin wrappers over core's `run_until_terminal_stream` and `converse_stream` that give the same text, end state and errors (`RunBudgetExceededError` propagates, the arguments of `run_stream` are checked at call time). Closing `run_stream` early pops a child that has ended and writes its result (it starts nothing), so the library counts the run once. The edit voids the registered synthesis bench blocks that pin `runner.py` by design (already void).
- `SynthesisPolicy(allow_classification=False)` (plan 51fde95b step 11.4, D-030): when True the compile prompt (`prompts.system_prompt`, one more rule `prompts.CLASSIFICATION_RULE`) tells the strong model it may route a state on what the user means with `classification_extractions` (at least two intents, a fallback, never `model` or `prompt_config`). It is off by default, the default prompt is the same text as before and no G5 verdict changes in either setting; set it only when the interface that runs the procedures implements `complete`, because a classifying state on one that does not fails G6b with an outage message and `run_procedure` with `push_failed`. The edit voids the registered synthesis bench blocks that pin `policy.py` and `prompts.py` by design (already void).
- A near miss in the synthesis compile request (plan 51fde95b step 11.5, D-027): `FSMLibrary.near_misses(goal, *, inputs_available, return_keys, tools_allowed, min_lifecycle, embed=None, top_k=1)` returns the stored records that `search` refuses only for their input or return key names (the same lifecycle and tool filter and scoring), and `FSMSynthesizer.obtain`, after a library miss, shows the best one that scores as a reuse would (`policy.reuse_min_score` for an embedder, an identical goal without one) to the strong model as one fenced block, `[EARLIER PROCEDURE BEGIN ...]`, under the untrusted procedure label, abort path stripped, as data: the record is valid compact JSON in a code fence (control characters and non-ASCII text escaped, no bracket look-alikes, so a model can copy its operators; a record over 6000 characters is not shown). With an embedder, each library miss makes one more embedding request, for scoring the records `search` refused. It is never reused or remapped, `compile()` is unchanged, and a library that cannot give one does not fail the compile. The compile that follows passes every gate and approval as before; the compile request with no near miss is the same as ever. The edit voids the registered synthesis bench blocks that pin `library.py`, `prompts.py` and `synthesizer.py` by design (already void).
- `MetricSnapshot.usage` (monitor, plan 51fde95b step 12.2): the `LLMUsage` summed over the interfaces of the managed FSMs and the attached API, read through their `usage()` only (the monitor never calls `reset_usage()`; one count per interface; what ended FSM instances used is kept in the total; `None` when no interface has a meter, as for a test double without one). It is in `GET /api/metrics` and the `/ws` metrics frame, and the dashboard shows LLM call, token and error tiles when it is set. Agents report no usage because `AgentResult` carries none. The OTEL exporter also exports the `workflow_advanced` (with `step_id`) and `workflow_event_delivered` events as `lifecycle.*` spans, ids only; no `latency_ms` is added, because `pre_processing` to `post_processing` times Pass 1 and not the turn.
- `ConversationSnapshot.classification_results`, `.transition_classification` and `.cut_off` (monitor, plan 51fde95b step 12.3a): the latest classification record per field and the classifier record of the last ambiguous tie (both through `redact_context`), and `{reply, fields}` when the last turn's reply or an extraction was cut at `max_tokens` (read from the internal `_cut_off` key before the internal keys are dropped; `None` when nothing was cut). They are in `GET /api/conversations/{id}` and the conversation drawer.
- A read-only Event log browse in the monitor (plan 51fde95b step 12.3b, a first slice of roadmap row 1.10): `fsm-llm-monitor --event-log DIR`, `InstanceManager.attach_event_log(log)` and the `InstanceManager.event_log` property (the explicit log, else the attached `API`'s `event_log`), the routes `GET /api/eventlog` and `GET /api/eventlog/{conversation_id}` (`since_seq`, `limit`), and an Event log panel on the Logs page. It uses only the public readers of `EventLog`; an event shows its envelope plus the fields `SHAPE_FIELDS` allows for its type, never a `*_ref`, a blob or text, at either privacy level; a log file that is a link is skipped; the log's usage fold is shown beside the meter and not added to it. Time travel, fork and resume from the dashboard stay with row 1.10.
- JSON colouring of the Visualizer FSM source editor (monitor): `attachJsonOverlay` in `utils/json_highlight.js` puts a coloured `pre` behind the textarea, whose text stays the editable and copyable source (the tokenizer gains a lenient mode, so half-typed JSON is coloured as far as it reads). A value set from code fires an `input` event to refresh it.
- The Builder page of the monitor (plan 51fde95b, step 13): a Max Tokens field; a Cancel button that aborts the request and drops the session, and an elapsed-seconds status while a build runs; `DELETE /api/builder/{session_id}?force=true`, which drops a session even with a send in flight (it only abandons the worker thread: the model call runs on, the id stays busy until it ends and the session is never put back); Edit JSON on the result panel, then Launch sends the edited artifact (the text is only parsed, a parse error is shown, and the launch route that loads the definition is the validator); `GET /api/capabilities` gains `agent_types`, the class names the monitor can launch. Launching a built FSM opens its first conversation in the drawer.
- JSON syntax colouring in the monitor (plan 51fde95b, step 14, owner request): the JSON text of the Event log fields, the conversation drawer (object context values and the extraction, transition, response and classification panels), the Builder result and internal state, and the workflow and agent drawers (workflow step data and context, tool-call parameters, agent context values) is coloured by token (keys, strings, numbers, booleans, `null`, punctuation). One shared module, `static/utils/json_highlight.js`, scans the text once (no recursion, no regex) and builds one `span` per token with `textContent` only, never `innerHTML`, so a string holding `<script>` stays text and selecting or copying still gives the plain JSON. A text that is not valid JSON, or longer than 200000 characters, is shown as it is. The colours are the `--json-*` variables in `:root` of `style.css`, each at least 4.5:1 against every surface JSON appears on (the dashboard has one dark theme). The Builder Edit JSON textarea, the Visualizer and Launch textareas and the free-text values stay plain. No route and no wire change.
- The async turn surface of the core (plan f150ec28, roadmap row 1.5 part B): `API.aconverse(user_message, conversation_id) -> str`, `API.aadvance(conversation_id) -> AdvanceResult`, `API.astart_conversation(initial_context=None, *, max_steps=None, max_seconds=None)`, the lazy async iterators `API.aconverse_stream` and `API.aadvance_stream`, and `API.arun_until_terminal` and `API.arun_until_terminal_stream` (the async forms of `run_until_terminal` and its stream; `before_step` may be a coroutine function, which is awaited; the sync run loops step an async hook with `run_sync`, so one that really suspends raises `SuspendedInSyncModeError`). A turn is written once, as an `async def` body in the engine, and the sync names (`converse`, `advance`, `start_conversation`, `converse_stream`, `advance_stream`, `run_until_terminal` and its stream) are thin `run_sync` or `iterate_sync` wrappers over it, so a sync caller still needs no event loop and a patch of a sync name still intercepts it. While a model call is suspended the loop runs other tasks, so two conversations interleave on one loop. Handlers still run inline on the loop (an async handler is not awaited, as before) and a sync-only `LLMInterface` runs on a worker thread through the default twins. `push_fsm` stays sync (`apush_fsm` is not added: a greeting call inside it blocks the loop, carried).
  - A conversation is guarded by a `TurnGate` (`fsm_llm.engine.fsm.gate`), owned by the turn and not by a thread. A second turn on a conversation whose turn is in flight is still refused with the same `FSMError` (never waited for), from another task, another thread or a handler, and a task started by the turn (or a `to_thread` worker of it) may re-enter. A reader or `end_conversation` called from a SIBLING task of the loop during an async turn now raises `ConversationBusyError` at once instead of reading half a turn; from another thread it waits for the turn as before, in arrival order.
  - A cancelled async turn is rolled back whole: one `asyncio.CancelledError` arm per turn body restores the snapshot taken at the start (Pass 1 included, so a committed transition or an extraction of a cancelled turn does not survive), the user message is popped, the gate is released, the turn budget stays counted, a greeting that was cancelled leaves no instance registered, and no ERROR handler runs. A stream whose consumer is cancelled while the stream awaits its next chunk rolls back the same way and persists no partial reply; `aclose` (abandonment) keeps the partial reply as before, and `athrow(CancelledError)` into a suspended stream is treated as `aclose`. A cancel or `break` in the consumer's own code never reaches the stream: that is abandonment, and the conversation stays claimed until the generator is closed, so wrap a stream you may leave early in `contextlib.aclosing`.
  - Known limits of the async turn, each pinned by a test: a handler or tool that outlived its timeout keeps the turn's ticket and may still read or end its conversation during the turn; a sync-only custom `StateExecutor` runs `extract` on a worker thread with the live conversation and may write into it after a cancel and its rollback (override `aextract`); a subclass of `ConversationExecutor` or `CompletionExecutor` that overrides only the sync `extract` or `after_transition` is bypassed by the async turn (override the `a` twin too, no warning is raised); between chunks the task that ran a stream's first step still owns the conversation; a second `arun_until_terminal` on a busy conversation fails with `ConversationBusyError` and not the "already being processed" text. `API.close()` called while a turn is in flight (a sibling task) refuses that conversation's end and leaves an owned event log open, so the turn can write its end; call it again after the turn.
  - Not breaking, so no `**BREAKING:**` entry: every sync call, every patch of a sync name and every concurrent-turn refusal behave as before. The one behaviour to know is the reader above: a sync reader (`get_data`, `get_current_state`, `end_conversation` and the like) called from a sibling task during an async turn raises `ConversationBusyError`, so read after the turn or from the turn's own task.
  - Known limit, carried (step 6 of plan f150ec28 was cut by its size rule): with the event log on, the writes and fsyncs still run inline on the loop thread in async mode, about 29 fsync calls and about 76 ms of blocking I/O per logged turn on the measured NVMe against about 0.5 ms with the log off. A loop that serves many logged conversations stalls for that time on every turn; run them with the awareness that a turn is not free for the other tasks. `FileSessionStore.save` is also inline, and a burst of conversations over sync-only interfaces can exhaust the default executor.
- `apply_json_patch(document, ops, *, max_ops, max_chars, check_path=None)` and `JsonPatchError(FSMError)` in `fsm_llm.core.definitions` (roadmap E1.2): the in-house RFC 6902 subset (`add`, `remove`, `replace`, `test`) that `fsm_llm.synthesis.apply_patch` was built on, with no knowledge of what the document is. `check_path(tokens) -> bool` is the one way a caller narrows where an op may land (a false return refuses the whole patch with `out-of-scope` at that op); every other bound, the refusal order and the codes are the ones of `apply_patch`. `apply_patch`, `PatchError` (still a `SynthesisError`) and the 13 codes of `PATCH_CODES` behave as before. The ops of the subset and its four limits (`PATCH_OPS`, `MAX_PATCH_DEPTH`, `MAX_POINTER_TOKENS`, `MAX_POINTER_TOKEN_CHARS`, `MAX_INDEX_DIGITS`) are exported from `fsm_llm.core.definitions` too.

### Changed

- **BREAKING:** `fsm_llm.synthesis.constants.PATCH_OPS`, `MAX_POINTER_TOKENS`, `MAX_POINTER_TOKEN_CHARS` and `MAX_INDEX_DIGITS` are removed; they live in `fsm_llm.core.definitions` (same values). `MAX_PATCH_DEPTH` is still readable from `fsm_llm.synthesis.constants` (a re-export of the core name); `PATCH_CODES` and `PATCH_MAX_TOKENS` stay there.
- The synthesis runner no longer counts child steps: `_Run.child_steps`, `_count`, `_count_child_step`, the `aborting` flag and the `procedure_abort` context write are gone. The runner pushes the child with `max_steps=policy.max_turns` and the engine budget counts for it (a hand-driven `API.advance` on the child now counts too, which the old counter missed), and a spent child frame, told apart from the host's own run budget by asking the manager and not by the exception text, ends as `{status: aborted, reason: budget_spent}` with no merge. A trace `stepped` event now means "the engine counted new turns of the child" (several turns between two ticks are one event with the engine's count), and an aborted run no longer runs the child's abort-edge turn (3 child extractions fewer in the pinned case). `inject_abort`, `strip_abort`, the G5 reservation and the library digests are unchanged and stay inert until roadmap P7.1. The registered synthesis bench blocks were already void; this edit changes a pinned file again.
- The synthesis runner decides `provider_error` from a position it takes when it pushes the child (the event log of the API through `API.event_log`, else the interface ledger, else the interface's own count, else 0; a source that raises is skipped) and no longer from `usage().errors` before and after, so `reset_usage()` during a run and an interface with no meter (with the log on) no longer hide an outage. `fsm_llm.synthesis.runner` now imports `fsm_llm.runtime.usage`, an edge the layers contract already allowed.
- `reset_usage()` of the interface meter moves a marker instead of clearing history: the cumulative per-kind counters only grow and `usage()` reports the counters since the marker, so every `usage()` and `reset_usage()` value is byte-identical to before (checked on a recorded scenario), while a position taken earlier (`mark()`, `since(mark)`) stays valid across a reset. No public signature changed and `LLMInterface` gets no default `usage()`; the `_usage_of` helper of `fsm_llm.llm.usage` is now the public `usage_counts`.
- **BREAKING:** a `field_extractions` or `classification_extractions` entry whose `field_name` is blank or internal-prefixed (`_x`, `system_x`, `"   "`) is refused at load, by `load_fsm_definition` and `fsm-llm-validate` with the same message (`State '<id>': field_extractions ['_x'] are empty or internal-prefixed and can never be extracted`). Before, the definition loaded and the entry could never extract: an internal key was stripped after a wasted model call and a blank name was committed as the context key `"   "`. The rule already held for `required_context_keys`. Remove or rename the field; a name with surrounding spaces (`" pad "`) is still accepted (plan 52ce5464, audit item B5).
- **BREAKING:** `validation_rules` with inverted bounds (`min_length` above `max_length`, `min_value` above `max_value`) or a negative `max_length` are refused at load, by the loader and `fsm-llm-validate` alike. Before, they loaded and rejected every value the model extracted for that field. Equal bounds and a negative `min_length` stay valid (plan 52ce5464, audit item B6).
- A framework `result_key` is closed to every channel, an author-listed `handler_only_keys` entry stays classifiable (plan 52ce5464 D-002, the owner's "option A", audit item B1). A classification field named like the `result_key` of a completion or synthesize state is no longer classified in Pass 1, in its retry or in the pass after a transition; before, it was written and the completion state then made no call (its result key was already set), now the completion state makes its call and the field stays unset. A `handler_only_keys` entry that a `classification_extractions` entry owns stays classifiable in Pass 1 and is now also classified after a transition, one more classifier call per unset listed classification field of the new state (this supersedes the clause "or `handler_only_keys`" of D-006 of plan 8a03483a). `analyze_data_flow` and `analyze_liveness` count such a listed classification field as produced, and `fsm-llm-validate` warns when an extraction channel of a state names a synthesize `result_key`, as it already did for a completion one. No `fsm_id` of an example moved.
- `fsm-llm-validate` warns on an unknown key of a `field_extractions` entry, a `classification_extractions` entry or an intent (`confidence_treshold`, `require`, `desc`); the loader still ignores it, so this is a WARNING only (plan 52ce5464, audit item B4).
- `<`, `>`, `<=` and `>=` coerce a string operand to a number only when it is a plain decimal or scientific literal (surrounding padding is trimmed, so `" 25"` still orders): `"inf"`, `"Infinity"`, `"1_000"` and non-ASCII digits no longer pass a numeric gate such as `>= 18`, as `==` already refused them (D-048); two non-numeric strings still compare as strings. This is a behaviour change of the evaluator, not marked BREAKING because no documented form is refused (plan 52ce5464, audit item C2).
- `ProcedureRunner.converse` steps the host on along its `on_failure` edge in the same call when a start is refused before any push (`inputs_missing`, `compile_failed`, `approval_denied`, `push_failed` and the other refusals), as `run` does; the call returns the failure state's reply. Before, it returned `""` and the next user message was eaten by the silent synthesize state (plan 52ce5464, audit item E4).
- `repair_procedure` returns `no-replay-set` right after `child-exists`, before it asks the approval reviewer or runs G6b. A caller with an empty replay set and a child that would also be refused for approval or simulation now sees `no-replay-set` instead of `approval-refused` or `simulation-failed`, and the reviewer and the simulator are not spent; `PROMOTION_CODES` and `synthesis/AGENTS.md` list the new order (plan 52ce5464, audit item E7).
- The `fsm_id` of the reasoning orchestrator moves: its `handler_only_keys` gained `hybrid_loop_count`, so a model claim of that key no longer lands in `final_context` (plan 52ce5464, audit item F3, D-015). The orchestrator is not an example FSM and no session of it outlives a solve; a log written by an older version for such a conversation would show the old `fsm_id`. The kind-equality baseline fixture was regenerated for this one case.
- The synthesis fixes of this release (audit items E1 to E8, and the core items B1, B3 and B4) edit files that the registered synthesis bench blocks pin (`SRC_PINS` of `scripts/synthesis_bench.py`: the runner, simulation, synthesizer, traces, policy, repair, constants, library, `core/verify/flow.py` and `core/verify/structure.py`), so they void bench pins. The `s0`, `s1` and `s2` manifests were already void (D-012 of plan 5a6a1f80), so nothing runnable changed: `report` still refuses by design and the recorded `REPORT_s2.txt` stays the record.
- **BREAKING:** `python -m fsm_llm.reasoning --list-types` prints its table on stdout, and every error of the reasoning CLI (a bad `--context`, `--quiet` with `--verbose`, an unexpected failure) is one line `python -m fsm_llm.reasoning: error: ...` on stderr. Before, all of it went through the library logger: the table reached stderr with timestamped log prefixes (the `README.md` said so), the error lines carried the same prefixes, and a `main()` call from Python printed nothing at all. A script that read the table or an error from stderr must read stdout or the bare line; exit codes are unchanged (plan 51fde95b, second-pass reasoning audit R1).
- `simple_calculator` of `fsm_llm.reasoning` computes in Python (plan 51fde95b step 10): `extract_elements` asks the model for ONE arithmetic expression (`calculation_expression`, a `str` field, so a bare number is read as its text) and the new `CalculatorExecutor` handler, a critical handler on entry to `perform_calculation`, evaluates it with `fsm_llm.reasoning.calculator.evaluate_expression`; `calculation_result` and `calculation_error` are `handler_only_keys` of the FSM and no longer `required_context_keys`, `perform_calculation` and `extract_elements` are silent, so `perform_calculation` makes no model call, a calculator solve makes fewer model calls than before, and the result is exact (the model no longer computes). The evaluator is total and bounded (text over 200 characters, more than 100 expression nodes, `|exponent|` over 1000, an integer over 2048 bits, `bool`, complex, non-finite and non-arithmetic text are refused with a fixed message, never a Python exception). A new `CalculatorFinalizer` handler on entry to `final_answer` sets `final_solution` to the result of a `simple_calculator` solve, so the model can no longer cut or rewrite a long number (`2 ** 100` keeps every digit). A refused expression such as `10 / 0` is reported as `calculation_error_details`; `validate_refine` retries it at most 3 times like any attempt with no solution. The `fsm_id` of `simple_calculator` and its entry in the kind-equality record move. A small model can mis-translate a problem (see `src/fsm_llm/reasoning/AGENTS.md`).
- **BREAKING:** the `simple_calculator` strategy no longer extracts `operand1`, `operand2` and `operator` and no longer asks the model for `calculation_result`: it extracts `calculation_expression` (one expression such as `0.15 * 240`) and a handler writes `calculation_result` and `calculation_error`. A caller that read `operand1`, `operand2` or `operator` from the strategy's context, or that set or expected a model-written `calculation_result`, must read `calculation_expression` or the result. `ContextKeys.OPERAND1`, `ContextKeys.OPERAND2` and `ContextKeys.OPERATOR` of `fsm_llm.reasoning.constants` are removed with no alias (see Removed); they were named only in `test_constants.py` and the test suite's `AGENTS.md`, and no example, doc or evaluation file read them. The orchestrator-level keys are unchanged: `proposed_solution`, `calculation_result` and `calculation_error_details` still come back from the merge.
- The monitor's workflow run list (`GET /api/workflow/{id}/instances`, `InstanceManager.get_workflow_instances`) lists at most the newest `MAX_WORKFLOW_RUNS_LISTED` (200) runs, still in start order and still with their redacted context and history (plan 51fde95b step 12.1): a long-lived engine no longer answers with every run it ever started. The list shape is unchanged, so this is not a wire break.
- `fsm-llm-eval` (`run` and `examples`, `fsm_llm.eval.resolve_model`, `run_dataset` and `run_cases` without a factory) and the monitor launch routes (FSM, agent and builder) use `ollama_chat/bonsai-2:27b` (the new constant `fsm_llm.constants.DEFAULT_TOOL_MODEL`) when no model is named, as the last step after the argument, `LLM_MODEL` and the `--env-file` file; before, an unset model was an error (`ModelNotConfiguredError`, exit 1 in the eval CLI, HTTP 400 on the monitor launch routes). The core is unchanged: `llm.resolve_model`, `API`, `Classifier`, the agents, `ReasoningEngine` and the other CLIs still raise `ModelNotConfiguredError` when nothing names a model, and an eval run with an injected `llm_interface_factory` is still labelled `injected`. The fallback is `fsm_llm.llm.interface.resolve_tool_model` (not exported by `fsm_llm.llm`); the examples default to the same model when `LLM_MODEL` is unset (plan 51fde95b D-036).
- Monitor Builder (plan 51fde95b, step 13): the default `max_tokens` of `BuilderStartRequest` is 4096 (was 2000), the meta-builder library default; a build cut at the limit is still not flagged. The Builder disables Launch, with a note, for a built agent type the monitor cannot run (EvaluatorOptimizer, MakerChecker, PromptChain) instead of failing after the user typed a task. The launch form of a built agent has editable Model, Label, Max Iterations and Timeout fields. `InstanceManager.get_capabilities()` returns `dict[str, Any]` (the three flags plus `agent_types`); the public API snapshot records the new return type.

### Removed

- `LlmCall.stream_from` of `fsm_llm.runtime.scope` (plan f150ec28, step 3.2): the lazy `make` form of a streamed model call. It had no caller once Pass 2 reached the model through `acall_stream`, which is itself lazy; `LlmCall` is an internal recorder ("not for construction", reached through `llm_call` and `engine/`), so this is not a `**BREAKING:**` entry. `LlmCall.stream` and `LlmCall.astream` are unchanged.
- `make counts`, `scripts/doc_counts.py` and the documented-count tests of `tests/test_packaging.py` (the three count classes `TestTheMakeTestPinLivesInContributing`, `TestDocumentedTestCountsMatchCollection` and `TestEveryDocumentedCountMatchesCollection`, with the unit tests of the script) are gone, and so is the README test that looked for a test count line. Documents state no count of tests, test files, modules or lines: a reader measures one with `make test` or `pytest --collect-only -q`, and a ratchet test that failed when a tracked `.md` file gained one was removed again in plan ace7a8c8 (step 4), so the rule is now a review rule. The rule that a tests README row names a real test file, and that every test file has a row, went with the script.
- `ReasoningTrace.remove_computed_fields_from_input`, a `model_validator(mode="before")` that dropped the four computed names (`total_steps`, `unique_states_visited`, `reasoning_complexity`, `average_step_time`) from a dict input. It was redundant: `TimestampedModel` has `extra="ignore"`, so those names are still accepted and ignored on input and nothing changes for a caller that never called the method (plan 51fde95b, reasoning audit R5).
- `ContextKeys.OPERAND1`, `ContextKeys.OPERAND2` and `ContextKeys.OPERATOR` of `fsm_llm.reasoning.constants` (`"operand1"`, `"operand2"`, `"operator"`), unread since the calculator FSM extracts one expression; no alias. The `ContextKeys` of the agents package keep their own `OPERATOR`.
- `scripts/agents_bench.py`, `scripts/skills_bench.py` and the recorded blocks they and the removed harness produced under `scripts/bench_data/` (`agents-react`, `skills-disclosure`, `l4-execute-write`, `l6-e2e`, `l7-explore-coldstart`, `l8-explore-loop`, `seed-probe`), plus the dead helpers of `scripts/bench_common.py` (`wire_digest`, `sha256_json`, `MANIFEST_FIELDS` and the masks they used) (plan ace7a8c8, step 5.2, D-039). Nothing live read them: the bench tests went in step 4, no module of `src/` imports a script, and the rows stay in git history. `scripts/synthesis_bench.py`, `scripts/bench_common.py` and `scripts/bench_data/synthesis/` stay because `test_synthesizer.py` and `kind_corpus.py` read the 16 reference tasks. Not a public API change.
- **BREAKING:** `fsm_llm.reasoning.SolutionResult` (and its `__all__` entry) is removed (plan ace7a8c8, step 5.3, D-031, D-032). No code built it: `ReasoningEngine.solve_problem` returns a tuple of the answer and the context, and the engine builds only `ReasoningTrace`, `ValidationResult` and `ReasoningClassificationResult`. Use the `solve_problem` tuple (the one result envelope is planned as `RunResult`).
- **BREAKING:** `fsm_llm.quick_start(path, model)` is removed. It was `API.from_file(path, model=model)` and nothing else; call that.
- **BREAKING:** `fsm_llm.agents.with_auto_memory(tools, config, memory, **kwargs)` is removed. It was `AutoMemoryReactAgent(tools=tools, config=config, memory=memory, **kwargs)`; call that.
- **BREAKING:** `fsm_llm.logging.register_stream_handler(handler_id, sink, format, context)` is removed. Nothing in the repository called it; `setup_logging` tracks the stream handlers it adds.
- `fsm_llm.core.verify.flow._result_key` (a private one-line wrapper of `kind_of(state).result_key(state)`, called by nothing in `src`) and the unused `outside_scope`, `CommentShare.percent` of `scripts/decision_index.py` and six dead helpers of `tests/test_fsm_llm_regression/workflow_support.py` (plan ace7a8c8, step 5.3).

### Fixed

Findings of six external audits of the engine, core, runtime, synthesis, monitor and reasoning code, each reproduced by a probe before the fix and covered by a test that fails on the parent commit (plan 52ce5464, audit item ids A1 to G6 in brackets).

- A plain `restore_session`, and the frames that `fork` and `resume` rebuild, seed only the public keys of a saved session (A1): a tampered session file carrying `_approval_granted` or `_run_output_keys` in `context_data` no longer reaches a live context, and the public keys are kept. This closes the hole that stayed open since the event log, where only the log-backed folds filtered those keys. `start_conversation` itself still filters nothing (agents seed `_max_iterations`).
- A failed `restore_session`, `fork` or `resume` no longer leaves its half-built conversation cached as ended: `has_conversation_ended` read True for an id that was never restored, and an entry the id already had was replaced (A2).
- `pop_fsm` keeps the frame when ending the child is refused (a turn holds the child's lock past `END_CONVERSATION_LOCK_TIMEOUT_SECONDS`), so the child is no longer orphaned and ending the root still ends it; `push_fsm` and `pop_fsm` log the stack depth read in the same hold that changed the stack, so two racing calls no longer log the depth of the other. On `ConversationBusyError` from a refused pop the parent may already hold the merged return keys, and a retry merges again (A3).
- A `KeyboardInterrupt` or `SystemExit` during `start_conversation` releases the conversation's instance, lock and turn budget (A4).
- The event-log producers raise `FSMError` ("Conversation not found") instead of `KeyError` for a frame that ended just before its record (A5).
- `push_fsm` is annotated to accept `os.PathLike[str]`, like `process_fsm_definition`; type only, a path already worked (A6).
- A condition no longer fails or raises on a very large number or a wrong operand shape (C1): `<`, `<=`, `>`, `>=`, `!=` and a mixed `==` answer correctly when an operand is an integer past float range (a 400-digit integer an extraction can carry; `float()` raised `OverflowError`, the condition failed and `{"<": [1, 10**400]}` was False); a two-argument `has_context` with a non-string key (a list or a dict) logs a WARNING and is False instead of raising `TypeError` out of `evaluate_logic`; `missing_some` with a names argument that is not a list raises `TransitionEvaluationError` (the condition fails) instead of a bare `TypeError`.
- `build_fsm_graph`, `to_mermaid`, `to_dot` and `visualize_fsm_ascii` raise `ValueError` for `states` that is not a mapping (`[]`, `None`), a `transitions` that is not a list (`{}`, `""`; `null` still means terminal) and a `conditions` that is not a list of mappings; before, the answer was an `AttributeError` or a silent empty diagram (C3).
- `create_okf_handler` refuses a whitespace-only `query_key` (`"   "`) with `OKFError`, as it refuses `""` (C4).
- `EventLogSessionStore.exists` reads membership (a log file or a legacy file of that id) instead of folding the whole log: it is True for a corrupt log and for a shape-level log that `load` refuses, consistent with `list_sessions`; before, it raised `LogCorruptError` or `LogPrivacyError`. An id outside the log's id rule still raises `EventPayloadError` unless a legacy directory can hold it (C5).
- The extraction retry loop asks again for a required field whose key is present in the context as `None` (from `initial_context`, `update_context` or a restored null), like the first Pass 1 filter (B2).
- `analyze_liveness` reports `permanent-tie` when a lower-ranked edge whose gate reads an undeclared key used to hide the tie of the unconditional edges below it (B3).
- A symbolic link at `logs/<id>.jsonl` (dangling or to an existing file) or at `<id>.lock` no longer redirects or destroys a file outside the log root (D1). The writer refuses a log path that is not a regular file in both modes (`LogCorruptError`) and opens the log and the lock with `O_NOFOLLOW` (`LogWriteError`); before, a link to a file without a trailing newline cut that file to zero bytes. Windows has no `O_NOFOLLOW`, so the file-type check is the only defense there (unverified until the Windows legs run).
- Rebuilding a conversation from the event log (`fork`, `resume`, `EventLogSessionStore`) raises `LogCorruptError` for a `working_memory` that is not an object, `hidden_buffers` that are not a list of names or a delta that is not an object, instead of a raw `ValueError` or a silently wrong set (D2); a delta part that is not an object is still skipped, so a log written with a redaction hook still restores.
- A streamed reply logs the request it really made, and a provider stream is closed when its reader stops early (D3). A stream that failed before sending no longer links the previous call's request; a custom interface that reads the provider stream eagerly is logged once; a custom interface that raises before it returns its iterator writes `provider/error` (it was an abandoned reply and the outage was not counted). A `KeyboardInterrupt` or `SystemExit` mid-stream stays an abandoned reply and an interrupted turn with no `provider/error`, now pinned by a test.
- `EventLog.append_batch(id, [])` creates no file, lock or writer, and tool-call arguments nested 31 or 32 levels deep are stored whole through a blob instead of being cut inline (D4). The module docstrings of the blob store and the JSONL store now say that the directory entries of `logs/` and `blobs/` are not fsynced (D5).
- `FSMSynthesizer.obtain` no longer returns a `deprecated` procedure after a paid compile that reproduced its digest (it raises `SynthesisRefusedError` with `lifecycle_too_low`, and a runner result carries that reason), and it returns a reproduced `draft` digest `verified` (E1).
- The G6b happy path says the keys a state mints from `required_context_keys` and from edge `requires_context_keys`, not only its declared extractions; a procedure that extracted through those alone no longer fails G6b with `pass-rate-low` and `no-progress` (E2).
- `run_procedure` reports a provider outage that stalled the child as `failed: provider_error` and counts no run (it was `needs_user_input` and a failed run). A procedure that cannot be built on the `runner_llm` (a classifying state and an interface without `complete`) is `failed: push_failed` instead of a failed tool result carrying state and field names, and the runner counts no run for an outage either (E3).
- `TraceRecorder` keeps nested runs of one procedure (`max_synthesis_depth` above 1) apart; they were merged into one trace and an empty one (E5).
- `check_policy` and `run_gates` end on a hand-built cyclic structure instead of looping forever, and `apply_patch` raises `PatchError("bad-scope")` for a `scope_states` that is not an iterable of names (it raised `TypeError`) (E6).
- A library record whose `provenance` lacks a key (for example `approved_by` or `gates_passed`) is a corrupt record (`LibraryError`) instead of a bare `KeyError` at the first read (E8).
- A `KeyboardInterrupt` or `SystemExit` during `ReasoningEngine.solve_problem` ends the orchestrator conversation and any pushed strategy frame (they stayed registered, with their event-log writer) and reaches the caller unwrapped (F1).
- A validator verdict given before `key_insights` arrived no longer costs a whole strategy pass and a retry slot; the attempt is judged at `validate_refine` entry (F2).
- `ReasoningEngine.model` reports the injected interface's model when it states one; `ReasoningStep.context_keys_used` and `SolutionResult.used_context_keys` write a sorted list in JSON, so the same result gives the same bytes under any hash seed; a model claim of `hybrid_loop_count` no longer lands in `final_context` (F3).
- The monitor's workflow instances list (`GET /api/workflow/{id}/instances`) shows every run, finished ones included; it showed none for a preset that completes inside `start` (F4).
- A request body over 1 MiB with no `Content-Length` (chunked) is refused with 413 and the security headers before the API key check, and the app buffers no more than the cap plus one chunk of it; uvicorn still receives and discards the rest of the upload. The check runs before the Host check, so an oversize chunked body from a refused Host now gets 413, not 400 (F5).
- Leaving the monitor server's lifespan closes the `InstanceManager` (conversations end, handlers detach), `InstanceManager.close()` runs once, and `attach_api` stops counting the previous API's live conversations (F6).
- A cancelled builder send releases its session when the worker thread ends (F7).
- `POST /api/dashboard/config` answers 400 for a panel or alert that is not an object and for a `config` that holds a dashboard (the whole `GET` answer), instead of dropping items or storing an empty config; the OTEL exporter emits a `lifecycle.agent_cancelled` span, and `OTELExporter.enable` on a collector without `record_event` leaves the exporter disabled (F8).
- The monitor's Markdown renderer no longer rewrites the inside of a code block or inline code (`x = a*b*c` became emphasis) and renders `1. 2. 3.` runs as an ordered list instead of bare list items; it still escapes first (G1).
- The monitor's chat no longer applies a reply (state badge, hidden input) to another conversation when the drawer was switched during the send; clicking an activity row opens the conversation without the 200 ms timer chain; the workflow Send Event target list offers only runs that can still take an event (G2).
- Applying a new dashboard config rebuilds the custom panels section: the old title, description and alerts row no longer stay, and a config with no panels shows no section (G3).
- The Launch button stays disabled until the modal closes, so a second click inside the 500 ms window no longer starts a second agent, FSM or workflow (G4).
- A refused agent launch from the Builder keeps the launch form (task and stub responses) open for a retry (G5).
- The Logs page has TRACE and SUCCESS pills (SUCCESS on, TRACE off by default) and the minimum-level request uses the full level order, so TRACE and SUCCESS records are no longer filtered out; TRACE records appear only when the monitor's log level is set to TRACE (G6).
- `FSMSynthesizer.obtain` refuses a stored `deprecated` digest with `SynthesisRefusedError` (`lifecycle_too_low`) before the reviewer is asked and before `FSMLibrary.put` can touch it, so a deprecated record no longer gains an approver or gates and a human is no longer asked about a procedure that is then refused; the `draft` to `verified` step is one lock hold (the new `FSMLibrary.ensure_verified`), so a concurrent `verify` or `deprecate` no longer makes `obtain` raise `LibraryError: illegal lifecycle edge` (plan 51fde95b, second synthesis audit).
- The G6a dead-edge gate (`analyze_dryrun`, `edge_witnesses`, `run_gates` and so `FSMSynthesizer.compile`) no longer raises `OverflowError` for an integer literal outside float range (a model reply with `10**400` crashed `obtain`), and no longer rejects an ordering gate on a literal of 2**53 or more (`x > 10**17` was reported `edge-unfireable` because a float absorbs `x + 1`): the pool now steps beyond a large end in the integer domain (plan 51fde95b, second synthesis audit).
- `FSMSynthesizer.compile` no longer raises a bare `ValueError` for a procedure with an integer literal of 4300 digits (the most `json.loads` accepts): the G6a pool stepped beyond it with a 4301 digit witness that the G6b happy path could not turn into text; such a value now says the key's sample, so the happy trial fails with a typed `SynthesisError`. The G6a pool also holds the integer midpoint of two huge integer literals, so a band such as `n > 10**400 and n < 10**400 + 10` is no longer reported `edge-unfireable`. The best-effort witness search (`edge_witnesses(best_effort=True)`) spends at most `MAX_FALLBACK_ASSIGNMENTS` assignments over a whole definition (a hostile 12-state definition took about 12 s before any trial, past `max_simulation_seconds`), and `dryrun.py` no longer calls the gate cost milliseconds (plan 51fde95b, review of steps 8 to 11).
- `ProcedureRunner.converse` pops every ended child in the same call: with `max_synthesis_depth=2`, a child that a refused nested start leaves on a terminal state stayed on the stack, so each further user turn raised `FSMError: Conversation has ended` and the host reached its terminal state only inside the raising call, losing the final reply; the settle pass now repeats until no ended child is left, as `run` already behaved (plan 51fde95b, second synthesis audit).
- `apply_patch(..., scope_states=)` answers `bad-scope`, as for a bare string, for `bytes` and for any iterable with a name that is not a `str` (they were reported `out-of-scope`, still refused); the `FSMSynthesizer` module docstring no longer names schema version 4.1 (plan 51fde95b, second synthesis audit).
- `--verb`, an argparse abbreviation of `--verbose`, now also prints the traceback of an unexpected failure of the reasoning CLI: the decision read `sys.argv` for the exact spelling and ignored the parsed flag (plan 51fde95b, reasoning audit X1).
- `ReasoningHandlers.prune_context` keeps the first 5 and the last 5 steps of `reasoning_trace` (the rule `update_reasoning_trace` already used past 50 steps) instead of the last 10: with a context over 8000 characters a retried solve lost its first step, which lowered `total_steps` and the reasoning types read from the trace. The other pruned lists keep their last 10 (plan 51fde95b, reasoning audit R2).
- `ReasoningHandlers.validate_solution` reads a non-`str` `problem_statement` (a number, a list, a dict in the context) as empty, as it already did for `problem_type`, instead of raising `AttributeError` from the critical validator and failing the solve after its model calls (plan 51fde95b, reasoning audit R4).
- A `simple_calculator` solve whose expression the calculator refuses (division by zero, `^`, anything that is not arithmetic) no longer ends with the model's own arithmetic: a present `calculation_error_details` counts as no solution, so the bounded retry (3 attempts) runs, and after the budget the solve returns the fixed message `The calculation could not be completed: <reason>` (new `ErrorMessages.CALCULATION_REFUSED`) instead of a number the model wrote as `proposed_solution` or `final_solution`. `calculation_result` and `calculation_error_details` are now handler-only keys of the orchestrator FSM (its `fsm_id` moves) and are cleared on every attempt, so a bulk reply can no longer claim a result that becomes the answer and a stale error no longer survives a later success (plan 51fde95b, review of steps 8 to 11).
- Run as `python -m fsm_llm.reasoning`, a failed solve, a failed `--save` and Ctrl-C each print exactly one line on stderr, as documented, instead of a timestamped logger record plus the error line plus a `use --verbose` hint line; the hint is inside the one line (plan 51fde95b, review of steps 8 to 11).
- `run_procedure` (`create_procedure_tools`) stops a procedure that waits for a user instead of spending its whole turn budget on it (plan 51fde95b step 11.3): it runs with a `before_step` hook that raises once `policy.dwell_threshold` steps in a row left the state and the public data unchanged, and replies `failed: needs_user_input` with no failed run counted in the library. A procedure that needs a user answer used to take every one of `policy.max_turns` steps (121 cheap-model calls for one run in the reproduction, now about ten) and count a failed run; a stall caused by a provider outage is still `provider_error`, and a run that spends its budget while still changing something still counts one failure. The edit to `tools.py` is not in the bench pins.
- The G6b happy path finds a value for an edge outside the G6a fragment (plan 51fde95b step 11.1): `edge_witnesses(definition, *, inputs=, best_effort=False)` fills each edge that G6a cannot judge (arithmetic such as `a + b > 10`, one key against another) from the new `fsm_llm.synthesis.dryrun.fallback_witness`, a search of at most `MAX_ASSIGNMENTS` assignments through the real evaluator over the edge's literals and their sums and differences, typed by the same narrowing and judged by the whole state; `happy_path` asks with `best_effort=True`. Before, such a valid compile got the fixed filler for every key and failed G6b (`pass-rate-low`, `no-progress`). The G6a verdict is unchanged (`analyze_dryrun` never calls it, and a dead edge stays dead); a dotted read keeps the filler. The edit voids the registered synthesis bench blocks that pin `dryrun.py` and `simulation.py` by design (already void, see above).
- Three monitor remainders of plan 52ce5464 (plan 51fde95b step 12.1, each with a test that fails on the parent). A manager closed by a first server shutdown is no longer served by the next run of the same app in one process: the lifespan start replaces a closed manager with a fresh one (it answered from a manager with no log sink and no handlers before); `InstanceManager.closed` is the new read-only flag. The Markdown renderer opens a paragraph for text that follows a list, code block or rule (the reply ended in a stray `</p>`). The chat message input of the drawer is disabled only for the conversation whose reply is pending, not for another conversation shown meanwhile; a send in another conversation during that wait shows a toast and keeps the typed text instead of silently doing nothing.
- Monitor Builder: a Cancel (a forced delete of a session whose send is still running) no longer takes the build out of the session cap. The abandoned worker thread cannot be stopped, so it now counts against the 50-session cap and against a new bound of 3 (`MAX_ABANDONED_BUILDERS`) until it ends, and `POST /api/builder/start` answers 429 beyond either; start, send and Cancel in a loop used to run any number of hidden model loops and could fill the thread pool every dashboard route shares. A second server start in one process keeps the configured `MonitorConfig`, the attached Event log and the dashboard config instead of building a default manager (plan 51fde95b, review of steps 12 to 14).
- Monitor Event log browse: a page no longer folds the whole log for `usage` or lists and `lstat`s the log directory on every request. The usage fold is cached per log until its file changes (at most 32 logs kept), the id is checked and `lstat`ed alone, the list stops at its cap and the timeline route has the 504 bound of the other worker routes; a log with one damaged line is no longer a 500 for every page: the answer shows the events before the damage and the usage of that prefix, with the new fields `skipped` (1) and `corrupt_after_seq`, and the page prints it. A page deep into a large log still scans the file from the start (plan 51fde95b, review of steps 12 to 14).
- Monitor Builder (plan 51fde95b, review of steps 12 to 14): `OrchestratorAgent` is not launchable from the dashboard. Step 13 had added it to the launchable set, but with no `worker_factory` every subtask is a placeholder with `success: False`, so every run ended `failed` with `stop_reason=no_result`; it leaves `GET /api/capabilities` `agent_types`, the launch form and `POST /api/agent/launch` (400), and the Builder disables Launch for a built Orchestrator with its note (a monitor worker factory that runs a real sub-agent is carried). The Builder's agent type lookup uses an own-key check, so an edited artifact with `agent_type` `constructor` or `__proto__` is no longer mapped through `Object.prototype` (the launch request used to drop the type and run a `ReactAgent`), and a capabilities reply for the pre-edit artifact can no longer enable or disable Launch for the edited one.
- The example scripts that printed or documented `export LLM_MODEL=ollama_chat/qwen3.5:4b` now name `ollama_chat/bonsai-2:27b`, the default their code already uses, and the five meta examples (`build_agent`, `build_fsm`, `build_workflow`, `meta_from_spec`, `meta_review_loop`) read an empty `LLM_MODEL` as unset again (`os.environ.get("LLM_MODEL") or "ollama_chat/bonsai-2:27b"`) instead of passing an empty model name on (plan 51fde95b, review of steps 12 to 14; examples edited on the owner's request, text only). A regression row now pins that only `fsm-llm-eval` and the monitor read `resolve_tool_model` and `DEFAULT_TOOL_MODEL`.
- Monitor Builder (plan 51fde95b, step 13): a launched built agent always ran `gpt-4o-mini`. The Builder took `artifact.config.model`, a fixed template value of every built agent (and model-authored text), before the Builder Model field, so a local-model dashboard launched an OpenAI model and `DEFAULT_TOOL_MODEL` never applied. The model is now the Builder Model field, then the Model field of the launch form, then the server default, never the artifact; the label, max iterations and timeout come from the visible launch fields (pre-filled from the artifact, bounded by the launch route).

## [0.13.0] - 2026-11-16

**Highlights**

The first release of the project on PyPI as `llm-fsm` (`pip install llm-fsm`). The import package is still `fsm_llm` and the console scripts keep their names. This block is the short version: the sections below list every change, and every change that breaks a caller is an entry marked BREAKING that names the old form and the new one.

- Install and support: `pip install llm-fsm` (extras such as `llm-fsm[agents]` or `llm-fsm[all]`), Python 3.11 to 3.15 (3.10 is dropped), an upper bound on every dependency range. The licence is Apache-2.0; an older 0.1.1 upload under the same name on PyPI is GPLv3 and was left in place.
- New layout: the flat modules of 0.12.0 became layer packages (`fsm_llm.core`, `engine`, `llm`, `context`, `handlers`, `security`, `tools`, `policy`, `okf`, `skills`), held by import-linter contracts. The old import paths do not resolve and there is no shim: the BREAKING entries under Changed name the new place of every moved module.
- Safe defaults: there is no built-in default model (pass `model=` or set `LLM_MODEL`, else `ModelNotConfiguredError`), and importing the package reads no `.env` file (call `fsm_llm.load_env()` or pass `--env-file`). The BREAKING entries under Changed say what to set.
- `fsm_llm.testing` is a public subpackage: scripted and prompt-grounded model doubles, reply builders, a scripted user and state assertions for code built on `fsm_llm`, with no network and no `pytest` import.
- State kinds: every state is a `conversation`, `completion` or `synthesize` state, read through one internal registry that the pipeline, the loader, the validator and the analyses share (`State.kind`). Definition version 4.2 is the new default; a definition that omits `version` keeps its `fsm_id` and one that declares `"version": "4.1"` gets a new one.
- Supply chain and CI: dependencies pinned in `constraints.txt` and refreshed weekly under a 7-day cooldown, `pip-audit` and a `.pth` audit in CI, a seven-leg test matrix (Linux on Python 3.11 to 3.15, Windows on 3.11 and 3.15; macOS is not tested in CI) and a coverage gate that never decreases.
- Releases and docs: one pushed tag `v*` builds and checks the sdist and the wheel, writes an SBOM and publishes to PyPI by trusted publishing with attestations; a documentation site (mkdocs-material, versioned with mike) is built from `docs/`.
- Also new: Open Knowledge Format bundles (`fsm_llm.okf`, `fsm-llm-okf`), Agent Skills (`fsm_llm.skills`, `fsm-llm-skills`), `HandlerErrorMode`, `fsm-llm-validate --upgrade`, `--env-file` on every model command, and a generated public API snapshot (`tests/fixtures/api_snapshot.txt`).

Upgrading from 0.12.0: read the entries marked BREAKING, which name the old form and the new one. The list of every change follows.

### Added

- Python 3.13, 3.14 and 3.15 are supported (roadmap row 0.2, plan d454fd62 step 115): `pyproject.toml` classifies them next to 3.11 and 3.12 (`requires-python = ">=3.11"` is unchanged and has no upper bound), `tox.ini` runs `py311` to `py315`, and the README and `AGENTS.md` state the range. Evidence before the declaration: on each of the three interpreters, in a scratch virtual environment with the CI install line against `constraints.txt` and `mcp`, `import fsm_llm`, `mypy -p fsm_llm` (no issues), `scripts/audit_pth.py`, the import contracts (32 kept), the same 17,358 collected tests as on 3.12, the API snapshot byte-equal to `tests/fixtures/api_snapshot.txt` and the whole non-slow suite (17,344 passed, 12 skipped and one failure on each, fixed below). PyYAML 6.0.3 has no cp315 wheel, so on 3.15 pip builds it from the sdist (about 5 seconds, Cython 3.3.1 pulled at build time, pure Python without libyaml headers, the C extension also builds and loads when the headers exist). New tests: `TestClassifiersMatchFloor` pins the newest classifier, the `tox.ini` envlist and the `AGENTS.md` range to the classifiers, the CI Linux legs equal the classified versions, and the README badge and note are derived from them. Windows stays unverified until the first matrix run, and macOS is not tested in CI. `setuptools` is now in the `dev` extra: Python 3.12 and later no longer seed it, so `test_packages_find_matches_disk` failed on every leg from 3.12 up (no BREAKING).
- Release workflow (roadmap row 0.1): `.github/workflows/release.yml` starts on a pushed tag `v*` and on nothing else, builds the sdist and the wheel once (`python -m build` with the backend under the pins of `constraints.txt`, `twine check --strict`), refuses a tag that differs from `src/fsm_llm/__version__.py`, a pre-release version, a missing CHANGELOG section (the release notes come from `scripts/changelog_check.py extract` and are cut at the 125000-character limit of a GitHub release body) and a built file name that does not match `[project] name`, writes an SBOM (`sbom.cdx.json`, CycloneDX 1.6, from a fresh environment with the wheel installed with every extra, `cyclonedx-bom` pinned), then publishes to PyPI by trusted publishing with PEP 740 attestations (no API token exists; the `publish` job, in the GitHub environment `pypi` that needs a reviewer, holds the only `id-token: write`) and creates the GitHub release with the distributions, the SBOM and the notes. `twine` and `cyclonedx-bom` are pinned in `constraints.txt` (CI-only, in no extra). The owner steps are in `CONTRIBUTING.md`, "Releasing"; the distribution is named `llm-fsm` now (see Changed), so the publish job needs only the owner's trusted-publisher setup for that PyPI project.
- Install from PyPI (roadmap row 0.1, plan d454fd62 step 114, D-020): the README, `docs/quickstart.md`, `docs/README.md`, the package READMEs and `AGENTS.md` install with `pip install llm-fsm` and `pip install "llm-fsm[extra]"` (the extras are `workflows`, `reasoning`, `agents`, `eval`, `synthesis`, `testing`, `monitor`, `mcp`, `otel`, `a2a`, `all` and `dev`), tell a reader who installed the unrelated `fsm-llm` project to run `pip uninstall fsm-llm` first, and no longer tell anyone to install from a clone; the editable install is in `CONTRIBUTING.md` only (`make install-dev`). The README extras table no longer states dependency versions (it said `mcp >=1.0.0`, which was stale) and describes `all` and `dev` as they are (`all` is every extra, `dev` is the contributor tools). `SECURITY.md` states the dependency ranges as they are (a floor and an upper bound inside the tested major, with the build backend as the one named exception) and how to check the installed distribution. `tests/test_fsm_llm/test_readme_pypi.py` pins the README install text against the extras of `pyproject.toml`, refuses a clone-install instruction in the README, the docs and the package READMEs, a relative link in `README.md`, `CONTRIBUTING.md` and `SECURITY.md`, and an emoji in the README and `docs/`, and ties the README version badge to `__version__`.
- Licence note: version 0.1.1 of `llm-fsm` on PyPI (2025-05-19) carries the GPLv3 licence classifier; this project is Apache-2.0, so 0.13.0 is licensed differently from that earlier upload under the same name. Nothing was yanked or deleted.
- `scripts/changelog_check.py` (stdlib only, read-only, no network; roadmap row 0.5): `check` reads this file in the Keep a Changelog form and fails on a `###` heading that is not Added, Changed, Deprecated, Removed, Fixed or Security (or is repeated or out of that order), a missing or stale `[Unreleased]: .../compare/v<newest release>...HEAD` reference, sections that are not newest first, a spelling of the marker other than the exact `**BREAKING:**` (`**Breaking:**`, `**breaking for ...**`, the prose form), and a `**BREAKING:**` marker outside a list item; it judges `[Unreleased]` and, while that is empty, the newest release (history is frozen; `--from VERSION` widens the scope). `extract VERSION` prints one section for release notes (`0.13.0`, `v0.13.0` or `Unreleased`, `--with-heading` for the heading line). `make changelog-check` runs the check, and so does the lint job of CI.
- Scheduled pin refresh (roadmap row 0.10): `.github/workflows/refresh-pins.yml` runs weekly (Mondays 05:17 UTC) and on demand and, through the new stdlib-only `scripts/refresh_pins.py`, proposes the newest `litellm` and `mcp` release that is stable, not yanked, at least 7 days old (a cooldown, after the litellm 1.82.7 and 1.82.8 incident), admitted by `Requires-Python` for every Python of the CI matrix and inside the major of the current pin (it never downgrades and never crosses a major). When a pin moves, the `test` job (read-only token, no credentials, because it runs the new third-party release) installs it, runs the non-slow suite, mypy, the `.pth` audit and `pip-audit`, and only then does the `propose` job push a branch and open a pull request (a pull request opened with the default token triggers no workflow, so the gate runs in the workflow); the `test` job uploads the three proposal files as an artifact before it installs anything, and `propose` holds `contents: write` and `pull-requests: write` (registered in the allowlist of `test_supply_chain.py`), installs nothing, runs no project code and refuses anything but those three files and the litellm and mcp pin lines; no job that holds a write scope installs a package or runs project code, with the docs deploy as the one named waiver; `SECURITY.md` describes it. The owner enables "Allow GitHub Actions to create and approve pull requests" once (CONTRIBUTING.md). `tests/test_fsm_llm_regression/test_refresh_pins.py` (canned JSON, no network; 33 one-edit mutants of the script) and `test_refresh_workflow.py` (105 mutants of the workflow, the guard and pull request steps replayed in bash on scratch repositories) pin both.
- Supply-chain checks in CI (roadmap row 0.10): the job `supply-chain` of `.github/workflows/python-package.yml` installs the package with every extra and runs `pip-audit --skip-editable` over the installed environment, so a known-vulnerable dependency fails the build, then a canary step that runs `pip-audit` over the new `tests/fixtures/canary-vulnerable-pins.txt` (old releases with published advisories, audited and never installed) and passes only when the audit exits 1 with a non-empty report; the job `dependency-review` runs `actions/dependency-review-action` on pull requests (`fail-on-severity: high`, `pull-requests: write` on that job only); the `lint` job also runs `actionlint` and `zizmor` at the pins of `constraints.txt` over `.github/workflows`; the workflow has a concurrency group (a new push cancels the run in progress on a pull request, never on `main`). `pip-audit` joins `constraints.txt` and the `dev` extra (`pip-audit>=2.9,<3`), `zizmor` and `actionlint-py` join `constraints.txt`. `tests/test_fsm_llm_regression/test_supply_chain.py` reads every file under `.github/workflows` (a full-SHA pin with a release-tag comment on every `uses:`, no `pull_request_target`, a read-only top-level token, a write scope only from its allowlist), runs the canary script against a stub `pip-audit` and pins the policy text of `SECURITY.md`; 64 one-edit mutants each return their own rule code.
- CI test matrix (roadmap rows 0.1 and 0.2): `.github/workflows/python-package.yml` now runs a `lint` job once (ruff, format check, mypy, the `.pth` audit, the import contracts through pytest and the decision-anchor gate with its report as the step summary) beside a `test` matrix of seven legs, `fail-fast: false`: Linux on Python 3.11, 3.12, 3.13, 3.14 and 3.15, and Windows on 3.11 and 3.15 (macOS is not tested in CI), all with `mcp` installed, `PYTHONUTF8=1`, a pip cache and a 45-minute timeout. The Linux 3.12 leg measures branch coverage in place of the plain run and enforces the ratchet (`scripts/coverage_ratchet.py check`, and `guard` on pull requests). The token is read-only, runners are pinned and every action is a commit SHA with its release tag in a comment; `tests/test_fsm_llm_regression/test_ci_workflow.py` pins the workflow and kills 68 mutants of it. `coverage.json` is git-ignored. macOS, Windows and Python 3.13 to 3.15 are unverified until the first run of the matrix.
- Documentation site skeleton (roadmap row 0.7): `mkdocs.yml` (mkdocs-material, `include-markdown`, the mermaid fence, `theme.font: false` so the site makes no web-font request, `strict: true` and a `validation` block, a `nav` of the guides, the roadmaps and the wrapper pages) and `docs/index.md`, the landing page. The tools are the dependency group `docs` in `pyproject.toml` (a PEP 735 group, not an extra: `mkdocs>=1.6.1,<2` and Material need Python 3.11 or newer), pinned exactly in `constraints.txt`; install them with `pip install --group docs -c constraints.txt`. New Makefile targets `docs` (a strict build into `site/`), `docs-serve` and `docs-clean`; `make clean` removes `site/`; `check-yaml` of pre-commit skips `mkdocs.yml`. `tests/test_fsm_llm/test_docs_site.py` checks the configuration without importing mkdocs.
- A command line page for the documentation site (roadmap row 0.7): `docs/cli.md` lists every console script of `pyproject.toml` with its entry point, usage, `--env-file` and `--model` rules and exit codes, and `python -m fsm_llm.reasoning`. `tests/test_fsm_llm/test_docs_content.py` fails when a console script has no section or the page names one that is gone.
- Content pages and navigation of the documentation site (roadmap row 0.7): `docs/concepts.md` (why, how a message travels through the two passes, the five pillars, status and references, written as standalone docs prose), `docs/examples.md` (the examples table, linked as absolute GitHub URLs because the examples are not part of the site), one-line wrapper pages that include the root files `CONTRIBUTING.md`, `SECURITY.md`, `EVALUATE.md` and `CHANGELOG.md` (`docs/contributing.md`, `security.md`, `evaluate.md`, `changelog.md`; the changelog is kept out of the search index), and the `nav` grouped Home, Getting started, Guides, Reference, Packages, Architecture, Decisions, Project (both roadmaps published unedited). `tests/test_fsm_llm/test_docs_links.py` checks every relative link and anchor of the pages under `docs/` without importing mkdocs (anchors by the `toc` rules, cross-checked against Python-Markdown when the docs tools are installed) and `tests/test_fsm_llm/test_docs_content.py` pins the nav groups, the wrapper pages, the unedited roadmaps and the examples links. `CONTRIBUTING.md` now links `AGENTS.md` by its GitHub URL (a relative link out of `docs/` is not a site page).
- `.github/workflows/docs.yml` (roadmap row 0.7): the documentation site is built with `mkdocs build --strict` on every pull request to `main` and every tag `v*`, and a tag deploys it to the `gh-pages` branch with `mike` under the version of the tag (a final release also takes the `latest` alias, a pre-release with an `a`, `b`, `rc` or `.dev` segment does not). A guard step fails the deploy unless the tag is `v` plus `src/fsm_llm/__version__.py`. The token is read-only except in the deploy job, every action is pinned to a 40-character commit SHA (`actions/checkout` v7.0.1, `actions/setup-python` v7.1.0), the docs tools install from the `docs` group against `constraints.txt`, and CI never runs `mike set-default`. `docs/README.md` has the owner steps (enable Pages from `gh-pages`, run `mike set-default --push latest` once) and records the Material for MkDocs end of life on 2027-05-05 with its escape hatches (ProperDocs, Zensical). Pinned by `tests/test_fsm_llm_regression/test_docs_workflow.py`.
- `executors=` on `API`, `FSMManager` and `MessagePipeline` (keyword-only, passed down; internal, not exported from `fsm_llm` and without a builder setter until the kind extension point is public): a `{kind_name: StateExecutor}` map for that one instance, validated and copied read-only (`TypeError` for a value that is not a mapping of `StateExecutor`, `ValueError` for a kind name no registered `KindSpec` carries). It adds or replaces the Pass-1 executor of a state kind for that `API` only; another `API` and the module table of `fsm_llm.engine.kinds.registry` are not affected. A kind added at runtime (a `KindSpec` plus an executor) runs a full conversation with no edit in core (`tests/test_fsm_llm/test_executor_map.py`). Only the signatures of `API.__init__` and `FSMManager.__init__` in `tests/fixtures/api_snapshot.txt` change.
- `State.kind`: a read-only property returning the kind name of a state (`conversation`, `completion` or `synthesize`), backed by the new internal module `fsm_llm.core.state_kinds` (a registry of the state kinds and their declarative rules). It is not a field: it is absent from `model_dump`, `model_fields` and the JSON schema, so no `fsm_id` or library digest moves, and a definition cannot declare it. The engine side is the new internal package `fsm_llm.engine.kinds` (the flat `fsm_llm.executors` until step 80) (`StateExecutor`, one executor per kind); neither module is exported from `fsm_llm`, and there is no public way to add a kind until the plugin work (roadmap C10).
- `scripts/coverage_ratchet.py` (stdlib only, no network) and the branch-coverage configuration behind a never-decreasing coverage gate: `check COVERAGE_JSON [--baseline FILE] [--tolerance 0.10]` fails when the combined or branch percent of a `coverage json` report is below the committed `coverage-baseline.json` minus 0.10 percentage points, `update COVERAGE_JSON` only raises that file (each number floored to 0.01, never lowered), and `guard` compares it with `origin/main` and fails on any decrease; exit 0 or 1. `pyproject.toml` gains `[tool.coverage.run]` (`branch = true`, `source = ["fsm_llm"]`) and `[tool.coverage.report]` (`precision = 2`, `show_missing = true`), `coverage` joins the `dev` extra (`>=7.10.6,<8`) and is pinned in `constraints.txt` (7.13.5), and `make coverage` and the tox coverage env add `--cov-branch`. The baseline file `coverage-baseline.json` is written from the canonical leg only (Linux, Python 3.12, `mcp` installed) and is committed.
- `fsm_llm.testing`, the seventh subpackage (import it by name; `fsm_llm/__init__.py` does not load it and no library module imports it): the home of the public test doubles, the scripted user and the assertions for code built on `fsm_llm`. It exports `MockLLM2Interface` (fixed answers, a plain mutable and subclassable `LLMInterface`) and `PromptGroundedLLM` (a fact comes back only when the prompt shows its evidence), the two doubles promoted unchanged from the repository's own `tests/conftest.py`; the dead `transition_target` argument of `MockLLM2Interface` (stored, never read) is dropped. It also exports `ScriptedLLM` (a queue of `complete` replies as text, `CompletionResponse` or an exception to raise; fields from a dict; the Pass-2 reply per state; `usage()` that counts errors like the real interface; streaming chunks; a call nothing scripted is kept in `unexpected`, raised as `UnscriptedCallError` and failed by `assert_exhausted()`, because the pipeline swallows a raised extraction error), the reply builders `final_reply`, `tool_call_reply` and `malformed_reply`, `state_in_prompt` (the one reader of the state tag of a response prompt, drift-tested against the real prompt builder) and `hash_embed` (a pure, deterministic embedder). It also exports `ScriptedUser` (sends fixed messages, then the answer of the first rule whose regex or callable fits the last reply, then a fallback, bounded by `max_turns`), `advance_until_terminal` (the same for a message-free FSM) and the frozen `ConversationRun` record they return, whose `states` is the visited path (the state after the start and after each reply); exceptions propagate. It exports the plain assertions `assert_state(source, state_id)`, `assert_context(source, **expected)` (a subset check), `assert_path(source, expected, *, exact=True)` (`exact=False` is an ordered subsequence) and `assert_result(result, *, kind=None, status=None, reason=...)`; `source` is a `ConversationRun` or an `(api, conversation_id)` pair, they raise `AssertionError` explicitly (never `assert`, so `python -O` changes nothing), hide their own frames from the pytest traceback and name the first divergence; expected `kind`, `status` and `reason` go through `fsm_llm.core.codes` `from_string` (a typo raises `ValueError` listing the vocabulary) and `reason=None` means "no reason" while leaving it out checks nothing. `PathRecorder` is an opt-in handler that records the states code under test visits, per conversation id; registering any handler makes the pipeline deep-copy the context at every handler timing (D-022), which its docstring states. Extra `testing` (no dependency), requested by `all`, `make install-dev`, tox and CI; it imports the core and `fsm_llm.core.codes` only, never `pytest`, `litellm` or another subpackage. Documentation: the guide `docs/testing.md` (every Python block in it is run by a test, offline; it covers the `pytest.register_assert_rewrite` note, the `PathRecorder` deep-copy caveat and a migration note for projects that copied the doubles out of `tests/conftest.py`), the Testing section of `docs/api_reference.md`, `src/fsm_llm/testing/README.md` and `src/fsm_llm/testing/AGENTS.md`. The repository's own tests use the public module (`tests/conftest.py` no longer exports the doubles; `tests/test_examples/test_example_scripted_paths.py` drives four shipped example FSMs offline with it); the suite is `tests/test_fsm_llm_testing/`.
- `--env-file PATH` on `fsm-llm-eval` (both commands), `fsm-llm-meta`, `python -m fsm_llm.reasoning` and `fsm-llm-monitor`, next to the one `fsm-llm` already has (shared as `cli_support.add_env_file` and `apply_env_file`). The file fills only variables the process does not have (an empty or whitespace-only one counts as not set, as `resolve_model` treats a blank `LLM_MODEL`, so `LLM_MODEL=""` in the shell no longer blocks the file), so the model is the `--model` flag, then the process environment, then the file; with none each CLI prints one stderr line and exits 1, and a missing file does the same (`fsm-llm-eval` keeps exit 1 for every failure and 2 only below `--fail-under`). The monitor starts without a model; its launch routes read `LLM_MODEL` from the file. `fsm-llm-okf`, `fsm-llm-skills`, `fsm-llm-validate` and `fsm-llm-visualize` call no model and take no such flag.
- `fsm_llm.load_env(path=None, *, override=False) -> Path | None`, exported from `fsm_llm` (module `fsm_llm.env`): the one explicit way to read a `.env` file. With no `path` it searches for `.env` from the current working directory upward (never from the package location); an explicit `path` that is not a file raises `FileNotFoundError`; a variable the process already has is not replaced unless `override=True`, except that a blank one (empty or whitespace only) counts as unset and is filled when the file gives it a non-blank value (a `${VAR}` in that value resolves as it does for an unset variable; to keep a variable deliberately empty, do not name it in the file); it returns the file read, or `None` when none was found, and never logs a value. `fsm-llm` gains `--env-file PATH` (calls it before anything else) and `--model/-m NAME` (the model for the run: `--model`, then the process environment, then the file, then `ModelNotConfiguredError`, which prints one `Error:` line and exits 1). The other CLIs get `--env-file` in the entry above.
- `tests/fixtures/api_snapshot.txt` is a generated snapshot of the public API (a test fixture, not a documentation page): every name in the static `__all__` of the eight packages and of `fsm_llm.okf`, `fsm_llm.engine.okf_bridge`, `fsm_llm.skills` and `frontmatter`, plus `core.definitions.schema`, `core.codes` and the five public `core.verify.flow` names, with signatures, fields and enum values (`__version__` excluded). `scripts/api_snapshot.py` writes it (`make api-snapshot`, or `--write PATH`; stdout by default) and the non-slow test `tests/test_fsm_llm_regression/test_api_snapshot.py` fails when it differs from the code, so a removed or renamed name in those lists, or a changed signature, model field, default or enum member, fails the build until the file is regenerated and reviewed in the same commit. It does not cover module functions outside `__all__` (for example `fsm_llm.core.verify.upgrade_fsm_file`), `model_config`, the values of plain constant classes, the contents of factory defaults or property types. A class lists the members it inherits from a private (`_`-prefixed) base or from a same-package `<Group>Mixin`, and its class line names only the other bases (`LiteLLMInterface` and `ScriptedLLM` list `usage` and `reset_usage`, the request models list `checked`), so splitting a class into mixins leaves the file unchanged. The text is byte-identical on Python 3.10 to 3.12.
- `fsm_llm.core.codes` is the shared vocabulary of run outcomes: `Status` (`ok`, `failed`, `aborted`, `running`, `abstained`), `Reason` (the 15 synthesis reason codes in their pinned order plus `needs_user_input`) and `CompletionKind` (`calls`, `final`, `malformed`), each a `str, Enum` with `from_string`, plus three pure mappers, `status_of_stop_reason`, `status_of_tool_status` and `approval_code`, that translate the agents' stop reasons, tool statuses and approval outcomes (a tool status `unknown` is refused, never mapped to `failed`). Imported by name like `fsm_llm.okf`, not exported from `fsm_llm`; standard library only. Every wire string is unchanged, and nothing in `fsm_llm.synthesis` or `fsm_llm.agents` imports it: a test reads their constants and fails when one drifts. `fsm_llm.definitions` now takes its four status literals from `Status`. `abstained` is new vocabulary with no earlier source string: only `status_of_tool_status("skipped")` produces it, and the four `STATUS_*` constants of `fsm_llm.synthesis` map to the other four.
- `fsm_llm.core.definitions.schema` records the definition version that introduced each field (`FIELD_VERSIONS`, `parse_version`, `newer_fields`, `SchemaFinding`). The loader and `fsm-llm-validate` now warn when a definition declares a version older than the fields it uses (for example `"version": "4.1"` with a `synthesize` state, which needs 4.2): one log line from the loader per definition object, not per conversation (library logging is off by default) and a WARNING in the validator report, never an error, so nothing that loaded before is refused. A declared version below 4.0 or not of the form `N.M` is not checked.
- `fsm-llm-validate --upgrade` validates the file first, then rewrites it in place to the current definition version (`4.2`), only when it loads as an FSM definition and declares an `N.M` version from `4.0` up to (not including) the current one (so `"4.0"` and `"4.1"` change). Everything else is left byte-identical and the command exits 1 (the JSON report is still printed): invalid JSON, a non-object, a JSON object that is not an FSM (a `package.json`), duplicate keys, a non-string `version`, a version below `4.0` or not of the form `N.M` (`"3.0"`, `"1.3.0"`, `"v4.1"`: it may be the author's own revision string) and an I/O failure; a current, newer (`"5.0"`) or omitted `version` (an omitted one already means the default) is nothing to do and the exit code is the validation's, so it never downgrades; a file declaring `"3.0"` (six of the shipped examples do) must have its `version` edited by hand, because the registry does not know the shape below `4.0`. A deeply nested file is a refusal, never a traceback: a file about to be rewritten that nests containers more than 256 levels deep is refused with its bytes untouched (Python 3.14 and later accept a 100,000-deep value, whose indented dump needs gigabytes, and 3.12 raised `RecursionError` from the dump for a few thousand levels; the deepest definition the loader accepts is about 108 levels deep). A file with nothing to change is not written. The write is atomic (a temp file in the same directory, then a replace; the file mode is kept, a symlink keeps its link and its target is rewritten). A rewritten file is normalised: indent 2, LF line endings, numbers re-serialised, key order and non-ASCII text kept; the new version is the default, so the file's `fsm_id` and library digest change. stdout stays the JSON report. Also `fsm_llm.core.verify.upgrade_fsm_file(path)` (returns an `UpgradeResult` with `rewritten` and `refusal`), `fsm_llm.core.definitions.schema.SCHEMA_VERSION_OLDEST` and `main(fsm_path, upgrade=False)`.
- `HandlerErrorMode(str, Enum)` (`CONTINUE="continue"`, `RAISE="raise"`, with `from_string`) in `fsm_llm.handlers`, exported from `fsm_llm`: the closed vocabulary of `handler_error_mode`. `HandlerSystem`, `API`, `FSMManager` and the builders' `set_handler_error_mode` accept a member or its string, coerced once in `HandlerSystem`; the strings `"continue"` and `"raise"` keep working, and anything else (including `None` and a number) raises `ValueError`.
- `OKFError` and `SkillError` are exported from `fsm_llm` next to the other exceptions (`from fsm_llm import OKFError, SkillError`); they are the same classes as in `fsm_llm.definitions`, and `OKFError` is also in `fsm_llm.okf.__all__` and `SkillError` in `fsm_llm.agent_skills.__all__`. Additive: `fsm_llm/__init__.py` still imports no subpackage and not `fsm_llm.okf` or `fsm_llm.agent_skills`. The private `okf._FrontmatterTooLarge` is now `okf._FrontmatterTooLargeError` (internal, no alias).
- Open Knowledge Format (OKF) support, following OKF v0.2 (SPEC.md at commit `ad30107`, 2026-08-21; the spec is pre-1.0 and mutable, upstream issue #24). A bundle is a directory of markdown files with YAML frontmatter; the concept id is the file path minus `.md`.
  - Core module `fsm_llm.okf` (import it by name; `fsm_llm/__init__.py` does not load it): `load_bundle` (bounded by `OKFLimits`, including a `max_seconds` load deadline of 120 s (a plain 33 MB bundle loads in about 63 s on Python 3.11 and 92 s on 3.10; a slower machine can hit it, which yields a partial bundle with a `load-deadline` error), raised with `OKFLimits(max_seconds=...)`; the check phase is bounded too on its own clock; the four incomplete-bundle stop issues are kept first under the 5,000-issue cap; symlinks skipped, hardlinks not detected (an accepted residual); lookalike ids load and get an `id-collision` warning, nothing is dropped; never raises on bundle content, so an unexpected failure on one file is a `parse-failed` issue), `OKFBundle` (lookup, `list`, lexical `search`, `outgoing`, `backlinks`, link and resource resolution), `OKFConcept`, `OKFIssue`, `TrustTier`, `parse_concept_text` (never raises: every PyYAML constructor failure is a `frontmatter-invalid` issue), `check_bundle` (spec section 11; `check-deadline` on its own `max_seconds` clock, `link-broken` capped at 20 per concept), `write_bundle` (every file validated and encoded before the first byte is written, refusing what it cannot render within the caps (frontmatter 16,384 bytes, file 262,144 bytes, depth 64, 16,384 nodes) with an error that names the concept, so a loaded bundle is not always writable (five edge shapes, see `docs/api_reference.md` Known limits); `index.md` is cut at 196,608 bytes with a last `* ... and N more concepts` line; atomic per file but not for the whole bundle, unknown frontmatter keys kept, explicit UTC offsets), `OKFError`, and the `OKF_*` limits in `constants.py`. The memory export and the plain-FSM handler are in `fsm_llm.okf_bridge` (next bullet).
  - Module `fsm_llm.engine.okf_bridge` (an engine module since step 83 of plan d454fd62; import it by name; `fsm_llm/__init__.py` does not load it; it imports `fsm_llm.okf` by public names only): `bundle_from_memory_entries` (lone surrogates replaced by U+FFFD with a `text-sanitized` warning; one `metadata-truncated` warning per entry names every metadata cut: depth, items, text, values or size; `int` and `float` subclasses such as `IntEnum` become plain numbers, other non-JSON types (`Decimal`, `UUID`, `datetime`, `set`, `bytes`, `Path`) are dropped as a `values` cut; metadata then tags are dropped when the frontmatter would be too large), `create_okf_handler` (capped concept summaries, with a `trust_claimed` field, into a public context key of a plain FSM).
  - Console script `fsm-llm-okf <validate|list|show|search|export>` (`fsm_llm.okf.cli`, entry `fsm_llm.okf.cli:main_cli`); `validate`, `list`, `show` and `search` take `--max-seconds S` for the load budget. Text the console cannot encode is written as `\uXXXX` escapes instead of a traceback; an incomplete bundle prints a note on stderr. `main()` sets the process's stdout and stderr error handling to `backslashreplace` and does not restore it. Exit codes: 0 ok, 1 failure, 2 usage error, 130 Ctrl-C.
  - Agents integration in `fsm_llm.agents`: `create_okf_tools` (four read-only tools `okf_list`, `okf_read` with paging sized to the agent loop's observation cap minus the wrapper and a 300-character margin for model-added input keys, so the loop may still cut an entry when the echoed input is long and the cut is visible, `okf_search`, `okf_links`; when the bundle was loaded incompletely each reply carries one fixed note line; handler records and `OKFMemoryBackend` summaries do not, `OKFMemoryBackend.incomplete` is non-empty and `bundle.issues` has the reason), `OKFMemoryBackend` (a read-only memory backend for `AutoMemoryReactAgent`; every summary starts with `OKF_SUMMARY_LABEL`), and the exporters `export_semantic_memory` and `export_agent_result`.
  - Safety model: bundle text is untrusted and reaches a model in three ways, all capped and control-stripped: tool output, public-context summaries from `create_okf_handler`, and the labelled `OKFMemoryBackend` summaries that `augment_task_with_memories` concatenates into the task string (callers should pass an untrusted `header`; `AutoMemoryReactAgent` has none, so the per-line label is the protection). It is never an instruction, description or persona; nothing a bundle names is executed; the trust tier is shown as `trust (claimed by bundle)` and is a claim, never authorization.
- `security.resolve_within(root, relative)`: a shared path-confinement helper used by the OKF loader and writer and the Agent Skills reads (resolve first, compare path components); other confinement sites keep their own checks.
- OKF documentation in `docs/api_reference.md`, `README.md`, `AGENTS.md` and the package guides.
- Agent Skills support: folders with a `SKILL.md` (YAML frontmatter plus a markdown body, and optional bundled files), following the open standard at agentskills.io as read and re-checked on 2026-10-23 (unversioned and mutable: name 1 to 64 characters, description 1 to 1,024, compatibility 1 to 500, the keys `name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools`). Read-only: there is no writer, exporter, memory backend or plain-FSM handler.
  - Layer package `fsm_llm.skills` (import it by name; `fsm_llm/__init__.py` does not load it; the package imports no okf module, no engine module and no subpackage; litellm is loaded by the `fsm_llm` package itself, not by this module): `load_skills(paths)`, `SkillCatalog`, `AgentSkill`, `SkillIssue`, `SkillLimits`, `parse_skill_text`, `incomplete_note`, `skipped_note`. Loading is tolerant (content never raises, a defect is a `SkillIssue`; only a missing or non-directory root is a `SkillError`) and bounded (`SkillLimits`: 500 skills, 262,144 bytes per file, 1,000 resources per skill, 60 s; a spent bound makes the catalog `incomplete`). The catalog keeps metadata only and re-reads each body or resource page from disk, confined with `security.resolve_within` plus a lexical-equality check (symlinks inside a skill and dot entries are skipped, never read). Discovery is explicit paths at depth 1. `render_catalog()` gives a labelled block closed by a fixed end line, for a caller's task string, then the incomplete note and a skipped note (a count of skill folders skipped for an error, a lower bound, saying the user can check them with `fsm-llm-skills validate`) when they apply; in names and descriptions `[`, `]`, `<` and `>` become `(`, `)` and the single angle quotes U+2039 and U+203A and `&` is escaped, so skill text cannot spell the label, the closing tag, the end line or a note, even after the agent prompt sanitizer; it is `""` for a catalog with no loaded skill. One skill adds at most 25 warnings to the issue list, then one `issues-truncated` note.
  - Console script `fsm-llm-skills <validate|list|show>` (`fsm_llm.skills.cli`); `validate --strict` is a portability check against the Anthropic API rules, so extra keys such as `version`, `released` and `commit` warn. `list` says `no skills found` on stderr for an empty result. Exit codes 0, 1, 2 (usage), 130. `main()` sets the process's stdout and stderr error handling to `backslashreplace` and does not restore it, as `fsm-llm-okf` does.
  - Agents integration in `fsm_llm.agents`: `create_skill_tools(catalog, registry=None)` registers three read-only tools, `skill_list`, `skill_load` and `skill_read`. Every reply starts with a fixed untrusted label, is control-stripped and sized to fit the agent loop's 2,000-character observation entry; `skill_load` shows what the skill claims in `license`, `compatibility` and `allowed-tools` as claims, not enforced; `skill_read` reads a bundled text file or, for an empty path or a directory, lists the bundled files; only the first `skill_load` page carries the description, claims and file list, and for a skill that bundles files it ends the list with a fixed hint naming `skill_read`. `next offset: N` is never cut off, offsets are character offsets clamped to the text, and `skill_list` shows the skipped note. Live checks with `ollama_chat/qwen3.5:4b` (n=5 each, one run each, rule written before trial 1): before the hint 0 of 5 trials read a bundled file (the model repeated `skill_load` and never called `skill_read`); after the hint and the paging fixes 5 of 5 did, on the same fixture and pass rule. That is a before and after comparison of the whole round, not of the hint alone, on one model, three small skills and one question. The catalog is not injected into a prompt automatically: put `render_catalog()` in the task string, or the model calls `skill_list` (a small model may not).
  - Safety model: skill text is untrusted and reaches a model only as labelled tool output or inside `render_catalog()` output that the caller places; never as a tool description, instruction, persona or state instruction. Nothing in a skill folder is executed, imported or shelled out (scripts are listed and readable as text only), and `allowed-tools` is reported, never enforced. Names are ASCII. Known limits and accepted debt (the private helpers `agents/agent_skills.py` imports from `agents/okf.py`) are listed in `docs/api_reference.md`.
  - `SkillError(FSMError)` in `fsm_llm.definitions` and the `SKILL_*` limits, caps and fixed texts in `fsm_llm.constants`.
  - `okf.capped_issues(issues, cap, stops, kind)` and `okf.text_page(text, offset, size)` are now public helpers shared by OKF and Agent Skills (the issue cap and the paging); `okf.body_page` calls `text_page`. `okf.iso_utc`, `okf.has_lone_surrogate` and `okf.writes_within_caps` are public too, so `okf_bridge` imports no private name. `okf.capped_note(emit, cap, what)` is the one per-document warning cap (25 warnings, then one `issues-truncated` info) used by the OKF parser and the Agent Skills parser.
  - Measured on seven real skills of a local `~/.claude/skills`: all load with 0 errors (three `key-unknown` warnings for the `version`, `released` and `commit` extras). The largest, `iterative-planner` (a 50 KB body), takes 37 `skill_load` pages, a consequence of the 2,000-character observation cap.
- Agent Skills documentation in `docs/api_reference.md`, `README.md`, `AGENTS.md` and the package guides.
- `fsm-llm-skills show NAME PATH --file DIR` lists the bundled files under a directory (with or without a trailing `/`; `--file ""` lists every file) with the header and `--offset` paging of the agents tool `skill_read`, instead of refusing it. The prefix rule is now `SkillCatalog.files_under(name, path)` in core (`()` for a file, an unknown directory or skill, or a non-string; never raises), and `skill_read` calls it.
- `fsm-llm-okf` and `fsm-llm-skills` share one implementation of the exit-code mapping, the BrokenPipe parking, the unencodable-output escape and the `--max-seconds` type in the new core module `fsm_llm.cli_support` (import it by name; `fsm_llm/__init__.py` does not load it). `fsm-llm-skills` no longer imports private helpers from `okf_cli.py`. No behaviour change.
- Agent Skills catalog notes (unreleased feature, no removal): `SkillCatalog.render_catalog()` of a catalog with no loaded skill but with a skipped or incomplete note is now the label, an empty `<available_skills>` block, the end line and the notes (it is `""` only when there is neither a skill nor a note); the "more" line is bracketed, `[N more, use skill_list]`, so skill text (where `[ ]` become `( )`) cannot spell it; a `SKILL.md` that is itself a symlink is now the error `skill-md-unreadable` (was the warning `symlink-skipped`), is counted by the skipped note and makes `fsm-llm-skills validate` exit 1 for that tree (it exited 0). A resource link inside a loaded skill is still the warning `symlink-skipped`, and a later duplicate name is still a warning, not a skipped folder.
- `PythonToolLoader.load_directory(path, pattern, category)` returns a frozen `PythonToolLoadReport` (`tools`: a tuple of `PythonToolDefinition`; `errors`: a tuple of `(file, message)` pairs, message `ExcType: text`), one error for every file it could not load (an exception in the file, a file over 1 MB, an unloadable module spec, an unreadable file stat) and one for a missing directory. `PythonToolLoader.from_directory` is built on it: it still returns a plain list (the tools of the report) and still logs one warning per failed file, and it now also calls `warnings.warn(..., UserWarning, stacklevel=2)` once per error, so the caller sees a broken tool file without enabling logging and `fsm_llm.disable_warnings()` cannot hide it (`python -W error` turns it into an exception, by the caller's choice). `PythonToolLoadReport` is exported from `fsm_llm.agents`. `SystemExit` and `KeyboardInterrupt` raised by a tool file still propagate. The "removed name" test of the Python-tool rename is edited on purpose: the file is still skipped quietly and the result is `[]`, and it now also warns.
- `SOPRegistry.register_directory` now also emits one `UserWarning` per SOP file that fails to load, attributed to the caller, beside its existing log line; the other files register and the return value is still the count.
- `fsm_llm.logging.replace_default_handler()`, called by `setup_logging()` and `enable_debug_logging()`: after either call a library record printed twice (once by loguru's default stderr handler, once by the library's own); it now prints once. loguru's default handler (id 0) is replaced by a sink with loguru's own `add` defaults (`LOGURU_LEVEL`, `LOGURU_FORMAT` and the other `LOGURU_*` settings still apply) that writes to whatever `sys.stderr` is at write time (not the stream of the moment of the call, so a closed capture stream or `redirect_stderr` is not kept) and a filter that drops `fsm_llm` and `fsm_llm.*` records, so the host application's own records keep printing in loguru's default format. Does nothing when id 0 is already gone (a second call, a host that removed it) or when `LOGURU_AUTOINIT` is off (id 0 is then the host's own sink). `setup_cli_logging` keeps removing id 0 outright. The replacement runs after the sink is added, so a call that raises for a bad argument leaves id 0 alone. A host record also printed twice (I-149: the library's plain stderr handler had no name filter); that handler, and the one `enable_debug_logging()` registers, now pass `fsm_llm` records only while the replacement default sink is registered, so a host record prints once; if a host removed loguru's default sink (so none was installed, or it was removed later) they pass host records too, as before, and a host's own extra sink still sees its records twice (unchanged). JSON, file and stdout sinks stay unfiltered and keep host records. Behaviour change: after `setup_logging(sink="file")` library records no longer also appear on stderr (before, loguru's id 0 printed them); they go to the file only.
- `scripts/doc_counts.py` and `make counts`: one `pytest --collect-only` child (a plugin reads markers) is compared with the count tokens of the tracked `.md` files in the shapes it knows (`N tests` and `N collected`, `-m "not slow"` counts, marker counts, `N slow` and `(N slow)` in a table cell, `N live` and `N gated` (a `real_llm` test with a `skipif`), `N offline tests`, `tests in N files`, test-file counts, per-file and per-suite rows), not only the count literals of the two files (`AGENTS.md` and `README.md`) that `tests/test_packaging.py` pins; a token of one of those shapes that no rule names fails too, a number in prose with no such shape (a singular `1 test`, run outcomes such as `2,011 passed, 17 skipped`, `77 offline checks`) is not read, dated records (this file, `plans/`, `evaluation/`, `scripts/bench_data/`, `docs/agents_roadmap.md`) are skipped, and with the optional `mcp` package importable (the documented counts are the no-`mcp` counts) it skips and says so. A second slow class in `tests/test_packaging.py` runs it.
- `scripts/skills_bench.py`, a pre-registered ablation bench of the Agent Skills read hint (`SKILL_READ_HINT`, the line `skill_load` adds for a skill with bundled files): bench id `skills-disclosure`, block B0 (ReAct 4b, hint on versus off, 12 questions over 16 skills, rule written before any row, with two dated amendments) and block B1 (native 4b hint on and off, ReAct 9b, native 9b, informational), with `tests/test_skills_bench.py` (37 offline tests, no model call) and the manifests and rule files under `scripts/bench_data/skills-disclosure/`. The blocks are registered and NOT RUN: the live runs were declined, so the hint stays as a recorded default, not a measurement, and the earlier 5 of 5 result stays a before and after comparison of a whole round. A later run refuses to start if `agent_skills.py`, `agents/agent_skills.py` or `constants.py` changed since registration.
- `fsm_llm.ModelNotConfiguredError` (module `fsm_llm.definitions`; an `FSMError` and a `ValueError`): raised when no model is named. Its one-line message names the three ways to set one (`model=`, `LLM_MODEL`, `fsm_llm.load_env()`); a CLI prints it and exits 1, the monitor launch routes answer HTTP 400 with it. Catch it where you accept user input; to avoid it set `LLM_MODEL` or pass `model=`.
- `fsm_llm.llm.resolve_model(model=None)`, the one model resolver (argument, then env `LLM_MODEL`, else `ModelNotConfiguredError`, see the BREAKING entries of this release; the value is stripped and a blank `LLM_MODEL` is unset). `API`, `agents.resolve_agent_model` and `eval.resolve_model` call it. `DEFAULT_MAX_TOKENS`, `DEFAULT_LLM_TIMEOUT_SECONDS` and `DEFAULT_EMBEDDING_MODEL` are defined once in `fsm_llm.llm`.
- Shared helpers made public or new, each replacing copies: `utilities.redact_context` (moved from `fsm_llm.runner`, with `REDACTED` and `CYCLE_PLACEHOLDER`), `utilities.read_json_object`, `utilities.balanced_json_objects`, `utilities.coerce_int`/`coerce_float`, `session.SESSION_ID_RE`, `definitions.reachable_from`, `definitions.MAX_CLASSIFICATION_INTENTS`, `security.checked_api_key`/`presented_api_key`/`api_key_matches`, `visualizer.ASCII_STYLES`, `agents.tools.registry_from`/`call_with_timeout`/`callable_name`/`tool_call_from_context`, `agents.handlers.record_tool_step`, `agents.definitions.config_with`/`READ_ONLY_ANNOTATIONS`, `reasoning.constants.STRATEGIES`, `MetaDefaults.SUMMARY_LINE_WIDTH`.

- Runtime FSM synthesis: the sixth subpackage, `fsm_llm.synthesis` (import it by name; `fsm_llm/__init__.py` does not load it; extra `synthesis`, which adds no dependency; no console script; importing it also imports `fsm_llm.agents` and `fsm_llm.eval`, and none of them imports it). A host FSM names a goal in a `synthesize` state, a model compiles it into a child FSM, gates decide whether it may run, a file-backed library stores it and a runner executes it as a child of the host conversation. Documentation in `docs/synthesis.md`, `docs/api_reference.md`, `src/fsm_llm/synthesis/README.md` and `src/fsm_llm/synthesis/AGENTS.md`.
  - Compile and judge: `FSMSynthesizer` (`compile`, `obtain`; the strong model is any `LLMInterface.complete`, no `litellm` import in the package), `SynthesisPolicy` (a frozen, validated set of limits), and `run_gates` and `verify_stored` over eight gates in a fixed order: G1 loads, G2 structure, G3 data flow, G4 liveness, G5 policy allow-list (`check_policy`), G6a dead edges through the real transition evaluator (`analyze_dryrun`), G6b bounded simulation (`Simulator`), G7 approval (`reviewer`, `reviewer_from_hitl`; absent, raising or non-`True` refuses). `verified` means G1 to G6 passed over a model-written spec and model-written scenarios; it is not a proof of correctness, and `trusted` needs a human approval or `trust_after_successes`. The goal is untrusted text that reaches a model only through a fixed fence; model-authored text never becomes a setting (a `model`, `prompt_config`, regex pattern, persona or unknown key is refused by G5).
  - Store and run: `FSMLibrary` (content-addressed definitions and records, lifecycle `draft`, `verified`, `trusted`, `deprecated`, signature and goal search, optional embedder, atomic writes, a corrupt file raises `LibraryError` and is never overwritten) and `ProcedureRunner` (pushes the procedure as a child of the host conversation through core's `push_fsm`, `pop_fsm` and the `before_step` hook of `run_until_terminal`, aborts it over `max_turns`, returns the return keys only on success, writes `{status, procedure_id, version, reason}` to the result key). `create_procedure_tools` gives an agent `obtain_procedure` and `run_procedure`.
  - Observe and repair: `TraceRecorder` and `RunTrace` (value-free by construction), `diagnose`, `apply_patch` (an RFC 6902 subset, scoped to the diagnosed state), `propose_patch` and `repair_procedure` (propose-only and replay-gated; nothing is written without `write=True` and a `promote` verdict; a repaired child is `verified`, never `trusted`).
  - Exceptions rooted at `FSMError`: `SynthesisError`, `SynthesisRefusedError`, `PatchError`, `LibraryError`.
- `State.synthesize` (`SynthesizeConfig{goal, inputs, return_keys, on_failure, result_key="synthesis_result", min_lifecycle}`), an optional, additive state field. Adding it moves no `fsm_id`, because the id hashes only the fields that differ from their defaults; it is the field that introduced definition version 4.2, and the move of the default `version` to 4.2 is the separate BREAKING entry under Changed. Core only loads and validates it: the loader runs the real transition evaluator on sample results and refuses a state whose edge passes before a runner wrote a result, or whose edge to `on_failure` does not pass for `failed` and `aborted` (write `status in [failed, aborted]`). The pipeline mints no extraction for such a state and makes its result key handler-only; a bare `API.converse` on it stays. `TransitionEvaluator.transition_passes` is the one "does this edge pass" call the loader and the evaluator share.
- Core module `fsm_llm.core.verify.flow` (import it by name; it imports only `core.definitions` and `core.state_kinds`; first added as `fsm_llm.flow_analysis`): `analyze_data_flow` (G3) and `analyze_liveness` (G4) over a typed `FSMDefinition`, `declared_available`, frozen `FlowFinding` rows and `strongly_connected_components`. It never logs and never raises on a definition the loader accepted. `FSMValidator.validate()` calls none of the analyses (it is never stricter than the loader); `fsm-llm-validate` is unchanged.
- `fsm_llm.eval.cases.run_case_trial(..., stop=)`: an optional callable checked before every turn; a true result ends the trial there, like an ended conversation. Default none, so existing callers are unchanged.
- `scripts/synthesis_bench.py` (bench id `synthesis`, blocks `s0` to `s2`) and its offline tests in `tests/test_fsm_llm_synthesis/`: a pre-registered comparison of a compiled FSM (`synth-fsm`), a hand-written FSM (`hand-fsm`) and a ReAct agent (`react`) on 16 judge-free conversational tasks, one model (`ollama_chat/bonsai-2:27b`) for every role and arm, with a preflight that refuses to run while the shared model server is busy. Bench status: three blocks run on one model, all INCONCLUSIVE; no result favours synthesis. See Known open below and `docs/synthesis.md`.
- A `## Naming conventions` section in `AGENTS.md`: the rules for new code (spelled-out ids, seconds in the name, constant prefixes, `str, Enum` vocabularies, `create_`/`build_`/`with_`, exception shape, lifecycle verbs, one parameter name per concept, wire strings never renamed, the renaming procedure) and the naming decisions that stay closed. `CONTRIBUTING.md` points to it.
- `scripts/decision_index.py` (stdlib only, never imports `fsm_llm`): `scan(root)` finds every `# DECISION [plan-id/]D-NNN [STALE]` anchor of the repository with the grammar of the planner's `validate-plan.mjs`, `load_manifest(path)` parses `plans/ANCHORS.md`, and `classify` sorts each anchor into `resolved` (a manifest line exists), `retired` (`[STALE]` and no line: resolved by tag), `unresolved` or `badprefix` (the prefix is not a plan id, or a bare `D-NNN`); a record names the path and the enclosing definition and carries no line number; `python scripts/decision_index.py --report` prints the counts. The anchor gate is the next bullet.
- The anchor resolution gate (roadmap 0.5): `python scripts/decision_index.py --check` takes no argument, reads `plans/ANCHORS.md` only and exits 1 when any anchor is `unresolved` or `badprefix`; `--write PATH` renders every `# DECISION` anchor as a report on demand, grouped by plan (the manifest date and rationale, the path and enclosing definition of each place, never a line number, a rule summary from the anchor comment, a Retired section for `[STALE]` anchors, the manifest entries with no anchor), deterministic and ASCII only, and nothing of it is committed; `--report` adds `resolved + retired`, the report-only `resolved-local` tier (a local `plans/<id>/decisions.md`, only while `CI` is unset) and the comment share of `src` (reported, not gated). `make decision-check` runs the gate and the lint job of CI runs it; `make decision-index` writes the report to `build/decision-index.md` (ignored by git). `CONTRIBUTING.md` (Decision anchors) and `AGENTS.md` describe the gate, the manifest and the `[STALE]` (resolved by tag) rule.

### Changed

- Documentation reorganised (roadmap row 0.7, plan d454fd62 step 123): `docs/` is a small hand-written set: the landing page, Quick Start, Concepts, the FSM design, handlers, synthesis and testing guides, a command line page (`cli.md`), the API reference, the architecture page, the examples page and the wrappers of CONTRIBUTING, SECURITY, EVALUATE and this changelog, beside the roadmaps. The published pages `reference/` (the generated module pages, the API snapshot page and the old CLI page, now `cli.md`), `packages/` (includes of the subpackage READMEs, which stay at `src/fsm_llm/<package>/README.md`), `upgrade-guide/` and `decisions/` (the generated decision index) are removed, and so are `docs/api.txt` and `docs/agents_roadmap.md` (a dated audit record of 0.12.0: its open items are in the 2026 roadmap and the agents README). The API snapshot is now the test fixture `tests/fixtures/api_snapshot.txt` (`make api-snapshot`, the same guard); `scripts/decision_index.py --check` takes no argument and judges only that every anchor resolves to a line of `plans/ANCHORS.md`; `scripts/changelog_check.py` no longer reads guide links (CL006, CL007, CL008, CL011 and the options `--docs-root` and `--next-version` are gone; the marker spelling, the section form and order and `extract` stay); the `mkdocstrings` plugin and its two pins left the `docs` dependency group. The migration record of this release is this changelog: each BREAKING entry names the old form and the new one, and no upgrade guide with code before and after is published.
- The README is refreshed for 0.13.0 and stays the feature-forward page (key features, why, installation from PyPI, six Quick Starts, how it works with the three state kinds and the layers, documentation, examples and command-line tools, CI and release, status, references), with absolute URLs for the logo and every link so the PyPI page renders and a link only to pages that exist. What CONTRIBUTING.md now holds stays out of it: the `make` block, the full project structure tree and the open items. The pinned `Run full test suite (N tests)` line lives in `CONTRIBUTING.md`, and `tests/test_packaging.py` reads it from there. `docs/README.md` and `docs/AGENTS.md` no longer list the deleted `strands_features*.md` files. `tests/test_fsm_llm/test_readme_pypi.py` keeps the README free of relative URLs, loads every full FSM block, runs the offline examples and checks every count the README states (agent patterns, reasoning strategies, step types, examples, console scripts) against the code.
- Pass 1 of a turn is dispatched through the executor of the state's kind (`fsm_llm.engine.kinds`) instead of testing `state.completion`; behaviour, request bytes and every refusal text are unchanged (checked against a recorded corpus of 111 definitions and four scripted turns). Private names moved with it: `MessagePipeline._execute_completion_turn` is now `CompletionExecutor.turn`, and the `pipeline._record_cut_off` and `pipeline._was_cut_off` functions are `context.record_cut_off` and `context.was_cut_off`; `flow_analysis._minted_keys` and `validator._validate_completion_states` are gone (the minting rule is `KindSpec.minted_keys`, the raw lint is `FSMValidator._validate_kind_states`). The conversation field, bulk and classification extraction methods moved to the conversation executor at step 80 (see the Changed entry on `fsm_llm.engine.kinds`).
- **BREAKING:** every declared dependency range now has an upper bound that contains its exact pin in `constraints.txt`, and two floors rise: `litellm>=1.109.0,<2.0` (was `>=1.83.0,<2.0`) and the `mcp` extra `mcp>=2.2,<3` (was `>=1.0.0`, which admitted the 1.x line the code no longer supports). The other ranges keep their floors and gain an upper bound at the next major (`pydantic<3`, `loguru<1`, `python-dotenv<2`, `tenacity<10`, `pyyaml<7`, `fastapi<1`, `uvicorn<1`, `jinja2<4`, `opentelemetry-api` and `-sdk<2`, `httpx<1`, and the dev tools). `constraints.txt` pins every direct dependency and the build backend exactly (it held eight names), chosen with `uv pip compile` on Python 3.11 and 3.15 for linux, macos and windows; `tests/test_packaging.py::TestDeclaredRangesMatchPins` fails on an unbounded range, a missing pin or a pin outside its range. An environment holding litellm below 1.109 or mcp 1.x must upgrade.
- **BREAKING:** no declared dependency range admits a major version other than the tested one (roadmap row 0.10, final REFLECT concern 3): the first version component of every floor equals that of its exact pin in `constraints.txt`. The runtime dependency `tenacity` is now `>=9.0,<10` (was `>=8.0,<10`), so an environment that pins tenacity 8 cannot install this release until it lifts that pin; the dev tools `pytest>=9.0,<10` (was `>=7.4.0,<10`), `pytest-asyncio>=1.0,<2` (was `>=0.21.0,<2`) and `pytest-cov>=7.0,<8` (was `>=4.0.0,<8`) follow, and `[build-system] requires` is `setuptools>=77,<85` (was `setuptools>=77`, with no upper bound), the one named exception: its floor is a feature floor (PEP 639 license expressions, 77), not the pinned major. `0.x` packages stay bounded by `<1`. `tests/test_packaging.py::TestNoRangeAdmitsAnUntestedMajor` enforces the rule.
- CI and the pin refresh install every extra, `mcp`, `otel` and `a2a` included (one identical `pip install -c constraints.txt -e` line in the `lint`, `test`, `supply-chain` and refresh `test` jobs), so the OpenTelemetry exporter tests run instead of skipping and `pip-audit` audits their packages; `make install-dev` and `tox` add `otel` and `a2a` (`mcp` stays out of the local lists on purpose). `tests/test_packaging.py::TestEveryInstallListRequestsEveryDeclaredExtra` pins it (the older rule judged only the extras that have a subpackage). The owner commands of `CONTRIBUTING.md` "Releasing" were corrected before the first release: the `pypi` environment is created with its required reviewer and the tag restriction by one JSON `PUT` with real booleans and one `POST`, with a verification command, "require a pull request before merging" (the coverage `guard` runs on pull requests only), "Re-run failed jobs" only after a partial release and `mike delete` for an abandoned tag; `tests/test_fsm_llm_regression/test_owner_commands.py` pins the text. No BREAKING.
- **BREAKING:** the minimum Python version is 3.11 (`requires-python = ">=3.11"`, was `>=3.10`); the Python 3.10 classifier, the tox `py310` environment and the 3.10 CI leg are removed, and the build backend requires `setuptools>=77,<85`.
- `[build-system] requires` is now just `setuptools>=77,<85`: the unbounded `wheel` entry is dropped (setuptools 70.1 and later build wheels themselves; `build_wheel`, `build_editable` and `build_sdist` were run on a setuptools 84.2.1 environment with no `wheel` installed) and `wheel` leaves `constraints.txt`. Building from source no longer installs `wheel`.
- **BREAKING:** `llm.resolve_model` (and so `API`, `AgentConfig()` and `eval.resolve_model`) no longer falls back to a built-in default model: with no `model=` argument and no `LLM_MODEL` environment variable (a blank value counts as unset) it raises the new `ModelNotConfiguredError` (exported from `fsm_llm`; an `FSMError` and a `ValueError`) whose one-line message names `model=`, `LLM_MODEL` and `fsm_llm.load_env()`. Before, such a call used `ollama_chat/bonsai-2:27b` silently. Pass `model=`, set `LLM_MODEL`, or inject an `llm_interface`. The remaining resolvers (`Classifier`, `ReasoningEngine`, the monitor) and the removal of the `DEFAULT_LLM_MODEL` constant follow in the next entries of this release.
- **BREAKING:** `Classifier(schema)` and `HierarchicalClassifier(schema)` with no `model=` and no `llm_interface=` no longer default to `DEFAULT_LLM_MODEL` (and no longer ignore the environment): they resolve the model through `resolve_model` (`model=`, then `LLM_MODEL`) and raise `ModelNotConfiguredError` when neither is set. Inside a conversation the classifier is unchanged (it uses the conversation's own interface). Agents: `AgentConfig()` and `default_llm_judge()` raise the same error when unset (`default_llm_judge(complete_fn=...)` does not: the custom backend owns the model); an agent, `create_agent` or `ConfiguredAgentBuilder` that holds an injected `llm_interface` needs no model (`default_agent_config` takes the interface's own `model`, else `LLM_MODEL`, else a placeholder that is never sent); `AgentArtifactBuilder._config_problems` reports a missing model as a problem instead of raising. Pass `model=` or set `LLM_MODEL` (or inject an interface).
- **BREAKING:** `fsm_llm.constants.DEFAULT_LLM_MODEL` (`"ollama_chat/bonsai-2:27b"`), `fsm_llm.agents.constants.Defaults.MODEL` and `fsm_llm.reasoning.constants.Defaults.MODEL` are removed, with no alias: the package has no default model. The signatures change with them: `ReasoningEngine(model=None)` (was `model=Defaults.MODEL`; an `llm_interface` in its keyword arguments needs no model), `InstanceManager.launch_fsm(model=None)` and `launch_agent(model=None)`, and the `model` field of the monitor requests `LaunchFSMRequest`, `LaunchAgentRequest` and `BuilderStartRequest` is `None` by default (was the default model; a blank value counts as unset). Every one resolves the model through `resolve_model` and raises `ModelNotConfiguredError` when neither the argument nor `LLM_MODEL` is set: the monitor routes `POST /api/fsm/launch`, `/api/agent/launch` and `/api/builder/start` answer HTTP 400 with the error message; `python -m fsm_llm.reasoning` (its `--model` default is now `None`, so the engine reads `LLM_MODEL`, which it ignored before), `fsm-llm-eval examples` and `run`, `fsm-llm-meta` and `fsm-llm` print the message as one stderr line (`fsm-llm`: `Error: ...`) and exit 1, with no traceback. `fsm-llm-eval` and `fsm-llm-meta --help` no longer name a default. Pass `model=`/`--model`, set `LLM_MODEL`, or inject an `llm_interface`. Anyone who imported `DEFAULT_LLM_MODEL` for its value should name the model they want.
- **BREAKING:** importing `fsm_llm` no longer loads a `.env` file. litellm reads one when it is imported (python-dotenv searches upward from litellm's own directory, or from the working directory in `python -c`, a REPL or a debugger, so a stale `.env` could fill `LLM_MODEL` or a key behind the caller's back); `fsm_llm.llm` now sets `LITELLM_MODE=PRODUCTION` around its one `import litellm` when the caller has not set the variable, and restores the environment right after, so `LITELLM_MODE` is never left in `os.environ` for child processes. A caller that set `LITELLM_MODE` itself is respected (`DEV` opts back in), and a host that imports litellm before `fsm_llm` has already loaded the file (that cannot be undone). Export the variables, or read a file on purpose with `fsm_llm.load_env()` or `fsm-llm --env-file PATH` (see the next entries).
- **BREAKING:** the `fsm-llm` command no longer reads a `.env` file by itself. It called `dotenv.load_dotenv()` on every run, which searched upward from the package directory (a checkout's own `.env`, never the file in the directory you ran it from) and let a stale file supply `LLM_MODEL` and provider keys unasked. It now reads a file only through `--env-file PATH`, with the same rule as `fsm_llm.load_env` (a variable the process already has is not replaced, except an empty or whitespace-only one, which counts as unset and is filled when the file gives it a value; to keep a variable deliberately empty, for example `HTTP_PROXY=`, do not name it in the file). `fsm_llm.runner.dotenv` is gone (the `runner` module no longer imports python-dotenv; `src/fsm_llm/env.py` is the one module that does), so a test or a host that patched `fsm_llm.runner.dotenv.load_dotenv` must patch `fsm_llm.runner.load_env` instead. A missing `--env-file` path, or a missing `--fsm` file, now prints one `Error:` line on stderr and exits 1 instead of a traceback.
- **BREAKING:** `Classifier(llm=)` and `HierarchicalClassifier(llm=)` are renamed `llm_interface=`, the spelling of every other core entry point (`API`, `MessagePipeline`, the agents), with no alias and no forwarding. Behaviour is unchanged. Because both constructors take `**llm_kwargs` (connection settings for a private interface), a caller still writing `llm=x` would have been swallowed silently and a private provider interface built that ignores `x`; the old keyword is therefore refused with a `TypeError` that names `llm_interface`. `ContextCompactor.summarize(conversation, llm_interface)` is now typed `LLMInterface | None` (its name did not change). The private attribute `Classifier._llm` keeps its name. `FSMSynthesizer(llm=)` and `Simulator(llm)` are unchanged.
- **BREAKING:** the path parameter of `utilities.load_fsm_from_file` (was `file_path`), of `utilities.read_json_object` (was `file_path`) and of the package-level `quick_start` (was `fsm_file`) is now `path`, the spelling of `API.from_file`, with no alias and no forwarding. Behaviour is unchanged. No call in this repository passed it by keyword; a caller that does (`load_fsm_from_file(file_path=...)`, `quick_start(fsm_file=...)`) now gets a `TypeError`. `load_fsm_definition(fsm_id_or_path)` keeps its name (it takes a path or an id and refuses an id), and the `fsm_file` field of the workflows `ConversationStep` and `conversation_step` is unchanged.
- **BREAKING:** the seven text helpers of `fsm_llm.visualizer` are renamed from `create_*` to `build_*`, with no alias: `create_fancy_header`, `create_metadata_section`, `create_persona_section`, `create_states_section`, `create_transitions_section`, `create_legend` and `create_state_boxes` are now `build_fancy_header`, `build_metadata_section`, `build_persona_section`, `build_states_section`, `build_transitions_section`, `build_legend` and `build_state_boxes`. They build text and return it, like `build_graph_representation` in the same module; `create_*` stays for the functions that make an object. None is in `__all__` and none is exported from the package, but each was importable from `fsm_llm.visualizer`.
- **BREAKING:** `utilities.ContextFilterWorkError` is renamed `ContextFilterError` (no alias) and now subclasses `FSMError` as well as `ValueError`, like `BuildError`. It is the refusal `filter_context_tree` raises for a cyclic context whose aliasing would unfold past its work ceiling (`get_data`, `save_session`, the extracted-data commit, the monitor display filter). `except ValueError` keeps catching it, and `except FSMError` now does too; the old name no longer imports. It was never in `fsm_llm.__all__`, so only a caller that imported it from `fsm_llm.utilities` is affected. `API` behaviour is unchanged: `handle_conversation_errors` already re-raised both `ValueError` and `FSMError` unchanged.
- **BREAKING:** `HandlerSystem(error_mode=)` is renamed `handler_error_mode=`, and its attribute `HandlerSystem.error_mode` is now `HandlerSystem.handler_error_mode` (a `HandlerErrorMode`, which still compares equal to `"continue"` and `"raise"`), with no alias: `API`, `FSMManager` and the builders already used that name, so every entry point now spells it one way. A `HandlerSystem(error_mode=...)` call raises `TypeError`, and the `ValueError` for an unknown value now reads `Invalid handler_error_mode: ...`.
- **BREAKING:** the workflows time limits carry their unit and the step budget has one name, with no alias: `WorkflowEngine.start_workflow(workflow_timeout=)` and the `WorkflowInstance.workflow_timeout` field are renamed `max_seconds` (core calls a run budget `max_seconds`; the field appears in `model_dump()`), and the base `WorkflowStep.timeout` field, with the `timeout=` parameter of every step factory in `fsm_llm.workflows.dsl` (`auto_step`, `api_step`, `condition_step`, `llm_step`, `parallel_step`, `conversation_step`, `agent_step`, `retry_step`), is renamed `timeout_seconds`, the name `wait_event_step` and `WaitEventConfig` already use. `serialize()` therefore writes `timeout_seconds` instead of `timeout` for a step. `WorkflowTimeoutError.timeout_seconds` and `ConversationStep.conversation_timeout` keep their names (the latter is a second limit: the effective one is the smaller of it and `timeout_seconds`). A caller still passing `workflow_timeout=` or a factory `timeout=` now gets a `TypeError`, and a step class, `WorkflowInstance` or `WaitEventConfig` built directly with `timeout=` or `workflow_timeout=` gets a `ValidationError`.
- **BREAKING:** `WorkflowEngine(max_steps_per_run=)` is renamed `max_steps` (also the engine attribute and the `ValueError` text), and the constant `MAX_STEPS_PER_RUN` (in `fsm_llm.workflows`, `.constants` and `.engine`) is renamed `DEFAULT_MAX_STEPS`, with no alias. The default (1000) and the behaviour are unchanged; `max_steps` now matches the core `run_until_terminal(max_steps=)` budget it resembles.
- **BREAKING:** the eight engine context-key constants of `fsm_llm.workflows.constants` are renamed `KEY_*` -> `CONTEXT_KEY_*`, the spelling of core (`CONTEXT_KEY_CUT_OFF`), with no alias: `KEY_WAITING_INFO`, `KEY_TIMER_INFO`, `KEY_WORKFLOW_INFO`, `KEY_TIMER_EXPIRED`, `KEY_LAST_EVENT`, `KEY_USER_INPUT` and `KEY_CANCELLATION_REASON` become `CONTEXT_KEY_WAITING_INFO` and so on, and `KEY_TIMEOUT` becomes `CONTEXT_KEY_EVENT_TIMEOUT` (it holds `{event_type, timeout_at}`, not a limit). The values (`"_waiting_info"`, `"_timeout"`, ...) and so every stored context are unchanged. None was in `fsm_llm.workflows.__all__`; only a caller that imported one from `fsm_llm.workflows.constants` is affected. The monitor now reads the cancellation reason through `CONTEXT_KEY_CANCELLATION_REASON` instead of the literal.
- **BREAKING:** the `WorkflowEngine` methods no longer repeat the class name, with no alias: `start_workflow` -> `start`, `advance_workflow` -> `advance`, `cancel_workflow` -> `cancel`, `get_workflow_instance` -> `get_instance`, `get_workflow_definition` -> `get_definition`, `get_workflow_status` -> `get_status` and `get_workflow_context` -> `get_context`. `register_workflow`, `remove_instance`, `add_hook`, `process_event` and `get_statistics` keep their names, and so do the monitor's own `InstanceManager.advance_workflow`, `cancel_workflow`, `get_workflow_status`, `start_workflow_instance` and `send_workflow_event` (different methods). The monitor and the nine workflow examples call the new names.
- **BREAKING:** `WorkflowEngine.get_active_workflows` is renamed `list_active_instances` (it returns instance ids, like the core `list_*` readers) and `WorkflowEngine.shutdown` is renamed `close` (the verb of core `API.close`), with no alias. The `WorkflowResourceError` raised by `start` on a closed engine now has `resource_id="closed"` and the message `Engine has been closed` (was `"shutdown"` and `Engine has been shut down`).
- **BREAKING:** the four workflow step classes are renamed to match their factories, with no alias: `AutoTransitionStep` -> `AutoStep` (`auto_step`), `APICallStep` -> `APIStep` (`api_step`), `LLMProcessingStep` -> `LLMStep` (`llm_step`) and `WaitForEventStep` -> `WaitEventStep` (`wait_event_step`); the factories keep their names, and the old names no longer import from `fsm_llm.workflows`. `WorkflowDefinition.serialize()` writes the class name as each step's `"type"`, so the tag changes with the class (`"AutoStep"`, `"APIStep"`, `"LLMStep"`, `"WaitEventStep"`); the output of `serialize()` is the only place the tag appears (no loader reads it). The error texts of `APIStep` and `LLMStep` name the new class. `steps.STEP_EXECUTOR` is now `_STEP_EXECUTOR` (internal), and `fsm_llm.workflows.engine` no longer defines `__all__` (its names, such as `DEFAULT_MAX_STEPS`, `Timer` and `WorkflowEngine`, stay importable from the module and `fsm_llm.workflows`).
- **BREAKING:** the four workflow pattern helpers of `fsm_llm.workflows` take the `create_` prefix of core's factories, with no alias: `workflow_builder` -> `create_workflow_builder`, `linear_workflow` -> `create_linear_workflow`, `conditional_workflow` -> `create_conditional_workflow`, `event_driven_workflow` -> `create_event_driven_workflow` (also in `fsm_llm.workflows.__all__`). The `<kind>_step` factories, `create_workflow` (which returns a `WorkflowDefinition`), `with_step`, `with_initial_step` and `serialize()` keep their names.
- **BREAKING:** `WorkflowDefinition.validate()` is renamed `validate_structure()`, with no alias. The old name shadowed pydantic's own `BaseModel.validate` (it needed a `# type: ignore[override]`); core names its checks `validate_<what>`. It still raises `WorkflowValidationError` with every error in `validation_errors`; `WorkflowEngine.register_workflow` and `WorkflowBuilder.build()` call it.
- **BREAKING:** the eight `Workflow*Error` constructors of `fsm_llm.workflows.exceptions` take the message first and the identifying data as keyword-only arguments, like core (`StateNotFoundError(message, state_id=None)`), with no alias and no forwarding: `WorkflowDefinitionError(message, *, workflow_id=None, details=None)`, `WorkflowInstanceError(message, *, instance_id=None, details=None)`, `WorkflowStepError(message, *, step_id=None, cause=None, details=None)`, `WorkflowTimeoutError(message, *, operation=None, timeout_seconds=None, instance_id=None, details=None)`, `WorkflowValidationError(message, *, validation_errors=None, details=None)`, `WorkflowStateError(message, *, current_step_id=None, operation=None, details=None)`, `WorkflowEventError(message, *, event_type=None, details=None)` and `WorkflowResourceError(message, *, resource_type=None, resource_id=None, details=None)`. The attributes keep their names and the engine's messages read as before (an id still prefixes the text, `"Step 'a': ..."`; an id left out adds no prefix). A caller that built one with the old positional shape (`WorkflowStepError("s", "msg")`) now gets the first argument as the message and the second as a `TypeError`. `WorkflowValidationError` no longer builds its own `Validation failed with N error(s)` headline: the caller passes the message and the per-error lines are appended to it. `WorkflowTimeoutError` no longer builds its text from `operation` and `timeout_seconds`: the caller passes the message.
- **BREAKING:** `WorkflowStateError.current_state` is renamed `current_step_id` (the attribute, the constructor argument and the `details["current_step_id"]` key), with no alias: it always held a step id, never a run status. The class name is unchanged. A caller reading `err.current_state` now gets an `AttributeError`.
- **BREAKING:** `fsm_llm.monitor.MonitorError` now subclasses `FSMError` (was `Exception`), so `except FSMError` also catches `MonitorError` and its four subclasses (`MonitorInitializationError`, `MetricCollectionError`, `MonitorConnectionError`, `MonitorCapacityError`); the constructors are unchanged (a message, plus `FSMError`'s optional `details`). Code that relied on a `MonitorError` escaping an `except FSMError` must reorder its handlers.
- **BREAKING:** the `InstanceManager` lifecycle methods follow the vocabulary of core and the workflow engine, with no alias: `shutdown` -> `close` (it stops the manager, like `API.close`) and `destroy_instance` -> `end_instance` (it finishes one managed instance of any kind, like `end_conversation`; `cancel_*` stays the abort verb). The REST route `DELETE /api/instances/{id}`, the event type `instance_destroyed` and the `OTELExporter.shutdown` method (which wraps the OpenTelemetry provider) are unchanged.
- **BREAKING:** the workflow routing fields end in `_step_id`, with no alias and no second suffix: `next_state` -> `next_step_id`, `success_state` -> `success_step_id`, `error_state` -> `error_step_id`, `failure_state` -> `failure_step_id`, `timeout_state` -> `timeout_step_id`, `default_state` -> `default_step_id`, `true_state` -> `true_step_id`, `false_state` -> `false_step_id`. This covers the step classes and their factory keywords (`auto_step(next_step_id=)`, `condition_step(true_step_id=, false_step_id=)`, `api_step`, `timer_step`, `wait_event_step`, `switch_step`, ...), `WaitEventConfig`, `EventListener`, `Timer`, and the `register_event_listener(success_step_id=, timeout_step_id=)` and `schedule_timer(next_step_id)` parameters. A state is an FSM concept; a workflow step names a step. An old keyword now raises a `TypeError` (the factories) or a pydantic `ValidationError` (the step classes, `WaitEventConfig`, `EventListener`, `WorkflowStepResult`, which refuse any unknown keyword), so a step that still passes `next_state=` fails at construction. Before this fix a pydantic model dropped an unknown keyword silently, so `AutoStep(error_state=...)` ran with no error route.
- **BREAKING:** `WorkflowStepResult.next_state` is renamed `next_step_id`, and so is the `next_state` keyword of `WorkflowStepResult.success_result` and `failure_result`, and the `next_state` key of the `step_completed` hook data (`{step_id, success, next_step_id, error}`), with no alias. A custom step returning `success_result(next_state=...)` or a hook reading `data["next_state"]` must change. Terminal semantics are unchanged: an empty `next_step_id` on any successful result ends the workflow.
- **BREAKING:** the workflow models `WorkflowStep` (so every step class), `WorkflowStepResult`, `WorkflowInstance`, `EventListener` and `WaitEventConfig` now refuse unknown keywords (`extra="forbid"`, like `AgentConfig`): pydantic used to drop them, so a removed spelling such as `error_state=`, `timeout=`, `default_state=` or `workflow_timeout=` was silently ignored. Every unknown keyword on them, a typo included, now raises a `ValidationError`; the same holds for `ToolDefinition` (see the `timeout_seconds` bullet).
- **BREAKING:** the dict keys inside the `_waiting_info` and `_timer_info` context values follow the same rename, with no alias: `success_state` -> `success_step_id` and `timeout_state` -> `timeout_step_id` in `_waiting_info` (written by `WaitEventStep`, read by the engine), and `next_state` -> `next_step_id` in `_timer_info` (written by `TimerStep`). Both keys are internal context values a custom step may write itself (they are in `STEP_INTERNAL_WHITELIST`): such a step must use the new keys. An old-shaped marker is not refused: a `_waiting_info` whose `success_state` key was not renamed reads `success_step_id` as empty, so the workflow COMPLETES when the event arrives instead of routing to the old target; an old `timeout_state` key is ignored, so a timed-out wait FAILS instead of routing to its timeout step; and an old-shaped `_timer_info` makes the instance FAILED with the error text `'next_step_id'` (a `KeyError` that the engine records) when the timer fires.
- **BREAKING:** `WorkflowStep.referenced_states()` is renamed `referenced_step_ids()`, with no alias, and its single discovery rule is now the field-name suffix `_step_id` (it was `_state`). A custom `WorkflowStep` subclass must name its routing fields with that suffix (`next_step_id`, `error_step_id`, ...) or override `referenced_step_ids()`; a field still named `next_state` is no longer found, so validation, reachability and terminal detection no longer see it.
- **BREAKING:** `fsm_llm.reasoning.ReasoningEngineError` is renamed `ReasoningError` (no alias), the `<Package>Error` spelling of every other root; `ReasoningExecutionError` and `ReasoningClassificationError` keep their names and now derive from `ReasoningError`. The export in `fsm_llm.reasoning.__all__` moves with it.
- **BREAKING:** the reasoning step budgets of `fsm_llm.reasoning.constants.Defaults` end in `_STEPS`, with no alias, because each one is a ceiling on core steps (`run_until_terminal(max_steps=...)` or the sub-FSM step counter): `MAX_TOTAL_ITERATIONS` -> `MAX_TOTAL_STEPS`, `MAX_SUB_FSM_ITERATIONS` -> `MAX_SUB_FSM_STEPS`, `MAX_CLASSIFICATION_ITERATIONS` -> `MAX_CLASSIFICATION_STEPS`. Values unchanged; `MAX_SOLVE_STEPS` and `MAX_HYBRID_LOOPS` keep their names. Internal: the reasoning engine's private `conv_id` parameters are now `conversation_id`.
- **BREAKING:** the per-tool time limit carries its unit, with no alias: `timeout_s` is renamed `timeout_seconds` on `ToolDefinition`, `PythonToolDefinition`, `@tool(timeout_seconds=)`, `ToolRegistry.register_function(..., timeout_seconds=)` and the two helpers `check_timeout` and `call_with_timeout` (and in the `TOOL_BAD_TIMEOUT` error text and a serialized `ToolDefinition` dump). `@tool(timeout_s=...)` is public API for tool files loaded by `PythonToolLoader`: such a file must now say `@tool(timeout_seconds=...)`, otherwise the file fails to load and is reported in `PythonToolLoadReport.errors`. `Defaults.MAX_TOOL_TIMEOUT_S` and `AgentConfig.timeout_seconds` are unchanged; `get_json_schemas()` (and so `tool_schemas_sha256`) does not carry the field. `ToolDefinition` now sets `extra="forbid"`: a call that still passes `timeout_s=` raises a `ValidationError` (pydantic used to drop the keyword, so the tool ran with no time limit), and so does any other unknown keyword.
- **BREAKING:** the `conv_id` parameters of `ProcedureRunner` (`hook`, `run`, `converse`, `end_conversation`) and of the agent loop hooks `BaseAgent._last_turn_cut_off`, `_end_run_conversation`, `_on_loop_iteration` and `_handle_hitl_approval` (also `VerifiedReactAgent._inject_reflection`) are renamed `conversation_id`, the spelling of core. Callers that pass it by position are not affected; a keyword call `runner.run(conv_id=..., ...)` or a subclass override that keeps the old parameter name must follow. Core calls the runner hook as `before_step(step)` only, so the rename cannot break it. The pinned synthesis bench manifests were already void; see `src/fsm_llm/synthesis/AGENTS.md`.
- Contributors: an existing editable clone must run `pip install -e .` again, because the console scripts `fsm-llm-validate` (`fsm_llm.core.verify.cli:main_cli`), `fsm-llm-visualize` (`fsm_llm.core.visualize.cli:main_cli`), `fsm-llm-okf` (`fsm_llm.okf.cli:main_cli`) and `fsm-llm-skills` (`fsm_llm.skills.cli:main_cli`) point at new modules and the old stubs in `.venv/bin` import modules that no longer exist (`fsm-llm-okf` and `fsm-llm-skills` did not exist in 0.12.0). The directory layout is that of 0.12.0 (`src/fsm_llm/`, found by `[tool.setuptools.packages.find]` with `where = ["src"]`), so nothing changes for a user of the installed package: the import name, every public import path that is not listed as BREAKING and the wheel's non-module files are unchanged. What changes for a contributor: `mypy_path = "src"` lets `python -m mypy -p fsm_llm` run directly (`make type-check`, tox and CI call it), `tests/conftest.py` no longer puts `src/` on `sys.path`, `make clean` also removes `src/*.egg-info`, and `scripts/audit_pth.py` needs no allowlist entry for the editable install (the install writes a plain `<repo>/src` path entry; the audit flags the code-bearing finder hook of an earlier flat-layout checkout until `pip install -e .` is re-run). Bench pins (`scripts/synthesis_bench.py` `SRC_PINS`, `scripts/skills_bench.py`) name the files at their new paths.
- `max_history_size` (`API`, `FSMManager`, `AgentConfig.max_history_size`, `fsm-llm -n`) now sets how many messages the extraction and response prompts show, at most N messages and within the history token budget (default 3000), which can cut the window shorter (with 400-character messages and size 20, 8 are shown): the three default prompt builders are built with `max_history_messages` set to the validated `max_history_size`, passed unchanged (not doubled), so the stored unit is exchanges and the shown unit is messages. Before, the knob only sized storage and the prompt always showed the last 5 messages. A caller passing a non-default size now sends a different prompt (size 0 shows no history, sizes above 5 show more, a size of 1 to 4 shows fewer than before); the default size 5 gives byte-identical prompts. `API` no longer builds its own prompt builders; `FSMManager` does. The `fsm-llm -n` help text now says so.
- A context key named exactly `system` is no longer dropped from the data that feeds the per-field extraction and classifier prompts (`FSMContext.get_user_visible_data()`); it is an ordinary key, shown there as it already was in the Pass 2 prompt and `get_data`. The one key policy stays in `security.py` (`_`, `system_`, `internal_`, `__` prefixes, secret-looking entries). A handler that stores metadata under a bare `system` key now reaches extraction prompts: store it under `_` or `system_` instead. No code in this repository wrote such a key.
- A request whose system prompt is over 30000 characters or whose user message is over 10000 (the cap also holds when `max_message_length` is set higher, up to 50000) now raises one `LLMResponseError` that names the field, its length, the cap and the three largest public context keys with their character counts (key names only, never values; internal-prefix and secret-looking entries are skipped), instead of a pydantic `ValidationError` that named no key. The two caps are the named values `PROMPT_MAX_CHARS` and `USER_MESSAGE_MAX_CHARS` in `fsm_llm.definitions` (the workflows LLM step reads the same value). The advice to shrink context entries is given only when entries were named; otherwise the message says to shorten the prompt or the history. The soft bulk-extraction site still falls back and shows the new text in its WARNING; the other sites still fail the turn and roll it back. Prompts under the cap are unchanged. An over-cap value is not truncated.
- **BREAKING:** the default embedder is now `ollama/embeddinggemma-2:740m` (`DEFAULT_EMBEDDING_MODEL`, was `ollama/qwen3-embedding:0.6b`); it returns 768-dimensional vectors, so vectors embedded with the old default are not comparable with it: a `SemanticMemoryStore` file records its own `embedding_model` and keeps using it, but a legacy file with no stored `embedding_model` now falls to the new default and must be re-embedded. (This entry first also moved the default chat model to `ollama_chat/bonsai-2:27b`; that default no longer exists, see the BREAKING entries above, so name the chat model yourself with `model=` or `LLM_MODEL`.) The eval baselines (Run 006 and Run 007 in `EVALUATE.md`, measured on the 4b) are not comparable with a run on another model. The registration of the `skills-disclosure` bench is voided: its manifests pin the sha256 of `constants.py`, so `register` and `run` refuse after this edit (the bench was registered, never run, and records `qwen3.5:4b`; it is not re-registered).
- `SemanticMemoryStore(embedding_model=...)` now defaults to `None` (it was `DEFAULT_EMBEDDING_MODEL`). With `persist_path=` pointing at an existing store file, an omitted `embedding_model` adopts the model that file recorded (as `load()` and `from_dict()` already did), so reopening a store through its constructor keeps its vector space; the file is read before the embedding backend is built, once. A file with no recorded model, a blank file, a missing file and no `persist_path` use `DEFAULT_EMBEDDING_MODEL`. An explicit `embedding_model=` always wins and still logs the mismatch WARNING when it differs from the recorded one. Before, the constructor ignored the recorded model and ran on the current default, so a store saved under another embedder scored `0.0` on every recall.
- Docs only (plan 0068b055 step 12.18, live finding D1), superseded by the BREAKING entries above: a `.env` file used to be loaded when `fsm_llm` was imported (litellm did it) and could fill `LLM_MODEL` before any default was read, so a stale `.env` silently chose the model. Neither the default nor the implicit load exists now: the model is the `model=` argument, then `LLM_MODEL`, and a `.env` is read only through `fsm_llm.load_env()` or `--env-file`.
- The YAML frontmatter code of `okf.py` (the frontmatter split, the safe loader that refuses aliases and anchors, the bounded load, the no-follow read, the lone-surrogate walk) moved to the new core module `fsm_llm.frontmatter`, which Agent Skills share: one YAML loader in `src/`. No public change: OKF behaviour and messages are unchanged, and `okf.py` keeps its old private names as imports. `okf.py` is about 1,320 lines instead of 1,650 (the memory export and the handler then moved to `fsm_llm.okf_bridge`).
- `pyyaml>=6.0` is now a declared core dependency (it parses OKF and Agent Skills frontmatter; it used to arrive only as a transitive dependency).
- **BREAKING:** the Python-tool API in `fsm_llm.agents` is renamed, with no alias, no shim and no deprecation period, because the old names collide with Agent Skills (SKILL.md folders, which are read and never executed, while this loader executes the files it loads). Behaviour is unchanged. Old -> new:
  - module `fsm_llm.agents.skills` -> `fsm_llm.agents.python_tools`
  - `SkillDefinition` -> `PythonToolDefinition`
  - `SkillLoader` -> `PythonToolLoader`
  - `ToolRegistry.register_skill(skill)` -> `ToolRegistry.register_python_tool(tool)`
  - the parameter `skills` of `PythonToolLoader.to_tool_registry` and `PythonToolLoader.by_category` -> `python_tools` (a call that passes it by keyword breaks)
  - the `SKILLS = [...]` list convention in a `.py` file that `from_directory` loads -> `PYTHON_TOOLS = [...]`
  - **Any tool file that still uses an old name (a `SKILLS = [...]` list, a `SkillDefinition` or `SkillLoader` import) loads no tools and raises nothing: `PythonToolLoader.from_directory` skips a file that fails to load, logs a warning (library logging is off by default) and now also emits one `UserWarning` per failed file, attributed to the caller (see the loader report entry below). A `SKILLS` list in a file that already imports the new names is skipped without any warning (its `@tool`-decorated tools still load); a file that imports an old name is skipped whole. Rename the list to `PYTHON_TOOLS` and the imports to the new names.**
  - Log and warning texts follow the new noun ("Python tool directory not found", "Loaded N Python tools from ..."). The example `examples/agents/skill_loader/` is updated to the new names (token-level edits only, same directory and file names).
- `ClassificationPromptConfig` moved from `fsm_llm.prompts` to `fsm_llm.definitions` (still importable from `fsm_llm` and `fsm_llm.prompts`). `ClassificationExtractionConfig` is now a subclass of `ClassificationSchema` and adds only `field_name`, `required`, `model`, `prompt_config` and `context_keys`. The conversation error decorators moved from `fsm_llm.logging` to `api.py` and `fsm.py`, so `logging` imports only `constants`.
- Numeric extraction coercion is stricter: an `int` field rejects a `bool`, a non-integral float and a non-finite one; a `float` field rejects a `bool`, `inf` and `nan` (also the strings `"nan"`, `"inf"`, `"1e999"`, and a number too large for a float such as `10**400` or `Decimal('Infinity')`, which raised `OverflowError` through `coerce_float`/`coerce_int`). The `min_value`/`max_value` checks compare a number too large for a float exactly instead of through `float()`, so such a value is simply out of range (an invalid extraction) and one inside an enormous bound still passes. A rejected value is an invalid extraction (retried or left unset), not a truncated or infinite number. The raw-text rung of `llm.py` uses the same rule. A confidence of that size reads as the default like `inf` (`coerce_confidence`), and an embedding reply holding one is refused as non-finite.
- `advance`/`advance_stream` on a terminal state raise before any work (no LLM call, history untouched). The greeting of a terminal initial state honours the `output_response_format`. `API`/`FSMManager`/`APIBuilder` validate `max_history_size` and `max_message_length` at construction (`ValueError` naming the argument). `API.start_conversation` lets a `ValueError` through like `push_fsm`/`pop_fsm`, and refuses a falsy non-dict `initial_context` (`[]`, `""`, `0`) with `FSMError`. `ContextCompactor` validates `transient_keys` and `prune_on_entry` (a bare string raises `ValueError`) and no longer counts a key holding `None` as cleared.
- Blank `LLM_MODEL` is unset: `API.from_definition` under `LLM_MODEL=""` used to raise `ValueError`. `fsm-llm-meta --help` names the env default. `FSMInstance.persona` defaults to `None`, so a directly built instance no longer out-ranks the definition persona.
- Logging: `setup_logging` reads a level and a format in any case (blank is unset) and raises `ValueError` for an unknown sink or an unknown explicit `format` or `level` argument before registering anything (an unknown `FSM_LLM_LOG_FORMAT` or `FSM_LLM_LOG_LEVEL` in the environment is reported on stderr once per distinct value and ignored by every command: a command uses its own default level and a direct `setup_logging()` uses the human format and, for a level such as `WARN` or `FATAL` that is not a loguru level, `WARNING` (an unset variable still gives `DEBUG`). For the format that is what `setup_logging` did before; for the level it raised `ValueError` before); `setup_file_logging` returns the handler id. `enable_debug_logging` now lives in `fsm_llm.logging` on the `setup_logging` path, so its output uses `LOG_HUMAN_FORMAT` (date and padded level) instead of `HH:mm:ss | LEVEL | name:function:line`. JSON log lines label a host record as such and an extra cannot overwrite a core field. `fsm-llm-validate` and `fsm-llm-visualize` print no DEBUG record.
- `fsm-llm-validate` wording and checks: a transition to an unknown state reads `Invalid transition from ...`; each structural defect is reported once; a mistyped `validation_rules` bound is refused at load; `transition_classification` keys that name no transition target warn; four more loader error types are in the allow-list; a non-object or non-UTF-8 file gets a clean message. The visualizer's `Maximum Path Length` is the longest path from the initial state with back edges not followed: exact for an FSM without loops, a lower bound of the longest simple path (never above it, and it can depend on the order of the transitions) for one with loops; an unknown ASCII style raises `ValueError` and a malformed definition is refused the same way by every style; a nameless dict is titled `Unnamed FSM`. `visualizer.find_longest_path(graph, initial_state)` and the signatures of `create_transitions_section`, `generate_minimal_ascii_diagram`, `sort_states_by_depth` and `create_state_boxes` lost unread parameters.
- `LambdaHandler` accepts a bare `str` for all six set arguments (it was a substring or character test). The monitor handler states survive `handler_timeout` (a `ContextVar` copied into the timed worker).
- Agents: `create_agent` accepts a `ToolDefinition` and `tools=[]` is no tools; `create_memory_tools` and `create_semantic_memory_tools` build their tools with `@tool` (the native function-calling parameter schemas of `remember`, `recall`, `forget` and `list_memories` now carry `"default": "core"` / `"all"` on `buffer`, which the model sees; the three read tools also gain read-only `annotations`); a bad `approval_timeout` raises `ToolExecutionError` when `HumanInTheLoop` is constructed (a timeout is a `float` or an integer type such as `np.int64`, finite and above 0; `Decimal`, `Fraction` and `np.float32` were a `TypeError` in `thread.join` before and stay refused); the empty-registry guard of ReAct, Reflexion, REWOO, ParallelReact and native_fc is one `_require_tools`; `AgentConfig(auto_summarize_after=n)` raises when `n` is at least `Defaults.MAX_OBSERVATIONS` (20) and `make_observation_summarizer` refuses `keep_last >= threshold`; ADaPT's no-answer text is the shared `ErrorMessages.NO_ANSWER`. `MetaDefaults.AGENT_*` alias `Defaults.*`.
- Monitor: the `/ws` tick ships only what the SPA reads (running workflow ids in `workflow_updates`, an agent progress summary with `log_entries`, `dashboard_config_version`); it no longer carries `log_count`, the config body, `conversation_log`, `agent_types` or `workflow_ids`, and the visualize payload carries no server `x`/`y`. The visualize routes work before any `get_manager()`; a guard refusal (400, 403, 413) carries the security headers; workflow launch runs in a worker thread; a destroy of an unknown instance answers `not found` like the other routes; `fsm-llm-monitor` exits 130 on Ctrl-C.
- Reasoning: the "direct computation" special case is gone, so the classified type wins for every spelling; a missing strategy FSM raises `KeyError` at engine construction; detailed output states the summary facts once and `--quiet` prints the stripped solution; nine strategy tables derive from `STRATEGIES` (all 11 FSM hashes unchanged). `fsm-llm-meta` runs on `cli_support`: unencodable stdout is escaped, an error is one stderr line `fsm-llm-meta: error: ...` with exit 1, `main(argv)` is callable; `save_artifact` writes through a temp file and `os.replace`.
- `MessagePipeline.execute_handlers(instance, timing, *, current_state=None, target_state=None, updated_keys=None, error_context=None)` lost its third positional parameter `conversation_id` (unread) and everything after `timing` is keyword-only: a caller that passed `current_state` or `target_state` by position 3 or 4 now gets a `TypeError` instead of the id bound to `current_state`. `FSMManager` and the pipeline already passed keywords.
- **BREAKING:** Agents tools (plan 0068b055 step 5), for a `@tool` with a lone `dict[...]` parameter (whose typed schema the model no longer sees) and for a function that relied on `**kw` dropping extra keys: `@tool` and `ToolRegistry.register_function` infer a schema by one rule, `infer_schema_from_hints` (the second pre-check in `register_function` is gone). A lone positional dict-style parameter (`dict`, `dict[str, int]`, its string form or no annotation) gives the empty schema on both (it is what `register_function` always did): the model sees an empty object schema and the whole arguments object is the dict, where a `@tool` function with `tags: dict[str, int]` used to show the typed `tags` property (a typed schema for this form is carried, F-AB-05b); a keyword-only dict parameter (plain `dict`, `dict[...]` or `= None`, also on a bound method) keeps its named schema and is called by keyword (step 12.5: the dict used to be passed by position, which raised `TypeError` on every call) and its `dict[...]` annotation is advertised as `object` (step 12.12, below); `*args` and `**kwargs` are never schema properties and never required (`a: int, *args: int` listed `args` as required and could not bind; a lone `**kwargs` listed one and lost every argument, `{"q": 1}` returned `{}`). A function that names a parameter and takes `**kw` receives the extra keys the model sends (they were dropped, with a warning that they were ignored); a lone `**kw` with an explicit schema (an MCP tool) still filters by that schema. `register_function` now lists the named parameters of `a: int, **kw` for the model. The description rule (first docstring line for `@tool`, the whole docstring for `register_function`) is unchanged.
- Agents ADaPT (plan 0068b055 step 6): `ADaPTAgent` executes no tool, so it no longer advertises one: the `attempt_result` instruction no longer appends "Available tools" next to "written from your own knowledge", and the agent always logs `started in LLM-only mode` (it said `started with N tools` when a registry was given). `tools=` is still accepted and stored (three frozen examples pass it) but is reserved, as `OrchestratorAgent(tools=)` is; `AgentResult.tools_used` stays empty. `build_adapt_fsm` no longer takes `registry` and `build_adapt_field_instructions()` takes no arguments (both under Removed). The FSM built for an ADaPT run with tools is now the one built without, so the `adapt_tools` golden hash case is gone.
- Agents Orchestrator and ADaPT (plan 0068b055 step 7, D-013): the intermediate states are silent. Orchestrator `orchestrate`, `delegate` and `collect` and ADaPT `attempt`, `assess` and `decompose` have no `response_instructions`, so they make no reply call (a silent state returns `""` and adds nothing to history) and only `synthesize` and `combine` speak, as in React, PlanExecute and REWOO. A run makes 3 to 5 fewer model calls (mocked: Orchestrator one round 6 to 3, ADaPT direct success 5 to 3, ADaPT depth-1 decomposition 15 to 10), and the typed-field judges (`attempt_succeeded`, `all_collected`, `subtasks`, `operator`) no longer see the pre-verdict narration under "Recent conversation" (the `assess` reply used to be written before `attempt_succeeded` was judged); the evidence they weigh (`attempt_result`, `worker_results`) is unchanged. The six prompt functions that only fed these states are removed (see Removed). The golden hashes of `adapt`, `adapt_depth`, `orchestrator` and `orchestrator_keys` are re-pinned after a dict diff showed only those six fields moved. Evidence level: mocked (`PromptGroundedLLM`) and, later in this plan, one live check on `ollama_chat/bonsai-2:27b` (step 12.11, below). Its revert rule: a same-day N>=3 `fsm-llm-eval examples --category agents` run on the model under test, both arms on that model, the parent of this change (`8a62490`) against `c73dab7`; revert `c73dab7` if the after-arm median is lower by more than the run-to-run spread (no model is named: the default is now `ollama_chat/bonsai-2:27b` and the 4b baselines are not comparable). Live result, the 7 agents examples that use the two patterns, same day, runs interleaved, N=3 per arm, `8a62490` against `8da28be`: health before 67.9 / 67.9 / 67.9 (median 67.9), after 78.6 / 67.9 / 78.6 (median 78.6, spread 10.7), so the rule says do not revert; the whole difference is `hierarchical_orchestrator` (timeout at 300 s in every before run, finished in two of three after runs), the examples that finish ran in about half the wall time, and `orchestrator_specialist` and `supply_chain_optimizer` time out in both arms. The 0-4 scorer saturates (6 of 7 examples score 4 in both arms), so this shows the change did not lower the score and does not measure the answers' quality (carried, below).
- Agents Orchestrator and ADaPT run ended from outside (plan 0068b055 step 7): a run ended from outside (its conversation closed, `stop_reason="ended"`) before `synthesize` or `combine` spoke now returns `ErrorMessages.NO_ANSWER` ("Agent could not determine an answer.", `success=False`); before, `OrchestratorAgent` shipped the last non-empty intermediate reply and `ADaPTAgent` its responses fallback as the answer, which was an intermediate narration. Pinned by a test. A limiter or budget stop is not this case: those raise (`BudgetExhaustedError`) or route to the speaking state. The revert rule is the one stated in the Orchestrator and ADaPT entry above.
- Agents constructor limits (plan 0068b055 step 8, D-008): `ReflexionAgent(max_reflections=...)` below 1 and `PlanExecuteAgent(max_replans=...)` below 0 now raise `AgentError` naming the value and the minimum, in the shape of the existing `ParallelReactAgent(max_parallel=...)` check (minimum 1, minimum 0 meaning no replans). Before, `max_reflections` 0 and -1 behaved exactly like 1 and `max_replans=-1` like 0, silently. `max_replans=0` is unchanged. The checks were bare `< minimum` comparisons like the siblings', so `True`, `1.0` and NaN passed; step 12.8 (below) replaces them with one stricter helper and validates the other limits too. `ParallelReactAgent` now logs `Agent started with N tools, model=...` at construction, as its siblings do.
- `MetaBuilderConfig` refuses the inherited `AgentConfig` fields the meta-builder never reads (plan 0068b055 step 9, D-010): `enable_prompt_cache`, `max_history_size`, `transition_config`, `output_schema`, `reflect_every_n`, `auto_summarize_after`, `verification_fn` and `force_final_tool` raise a `ValidationError` naming the field when set to anything but their default. Before, a non-default value was accepted and ignored (the meta agent builds its own API from `model`, `temperature`, `max_tokens`, `timeout_seconds` and `max_turns` only). The check compares each value with the field default and never reads `model_fields_set`, so the fluent builder (which marks every field as set at its default) and `create_agent("meta_builder", config=AgentConfig(...))` with defaults still build. `max_iterations` and `instructions` stay accepted and ignored (D-012; two frozen examples pass `max_iterations`). No in-repo caller set a refused field.
- `AgentArtifactBuilder` (the meta-builder's agent artifact, plan 0068b055 step 10, D-011): `set_config` validates each value through `AgentConfig` (`max_iterations` and `max_tokens` at least 1, `temperature` 0 to 2, `timeout_seconds` positive) (finite floats, no `bool` and a non-blank `model` are `AgentConfig` rules since step 12.7); a bad value logs a warning and is skipped, as a wrong type already was, so the `set_config` tool no longer stores `temperature=9`. `validate_complete` re-checks the final config dict and lists each invalid key. Valid configs and stored artifacts are unchanged. `MetaDefaults.AGENT_MODEL` (`gpt-4o-mini`) stays a fixed template value, not the runtime default; changing it is carried.
- `AgentConfig` (plan 0068b055 step 12.7, D-020, F-AM-06b): refuses a NaN, infinite or non-positive `timeout_seconds` (NaN passed the old `<= 0` test), a `bool` for every numeric field (`max_iterations`, `timeout_seconds`, `temperature`, `max_tokens`, `max_history_size`, `reflect_every_n`, `auto_summarize_after`: `temperature=True` ran as a silent 1.0) and a blank `model`; an int too large for a float field was already refused. Valid values are unchanged (ints for float fields, digit strings and integral floats still coerce). `MetaBuilderConfig` inherits the rules. `AgentArtifactBuilder.set_config` and `validate_complete` follow `AgentConfig` and keep one extra rule of their own: `max_iterations` and `max_tokens` above 2**63-1 are skipped with a warning, so the artifact always serialises (`json.dumps` raises on an int past 4300 digits). A caller that passed `True` or NaN now gets a `ValidationError`.
- **BREAKING:** Agents constructor limits (plan 0068b055 step 12.8, D-008 and D-020, F-AR-13b), for a caller that relied on a silent fix-up: every `max_*` / `num_*` limit goes through one helper, `fsm_llm.agents.exceptions.require_int_limit(name, value, *, minimum, error=AgentError)`, which accepts an `int` (or a numpy integer, stored as a plain `int`) from the minimum to `Defaults.MAX_INT_LIMIT` (2**63-1) and refuses a `bool`, a `float` (also `1.0`, NaN, inf), a `str`, `None`, a value below the minimum and a larger int, with a message that never holds more than a short prefix of the value (before, `num_rounds=10**5000` raised a bare `ValueError` from a log line and the old messages formatted the value in full). Minimums: `max_reflections` 1, `max_parallel` 1, `num_rounds` 1, `max_workers` (Orchestrator and SelfConsistency) 1, `num_samples` 1, `max_replans` 0, `max_refinements` 0, `max_revisions` 0, `max_depth` (ADaPT) 0, `max_handoffs` 0, `max_verify_retries` 0 (0 means none of that work, so it stays valid). `DebateAgent(num_rounds=0)` and a negative value are now an `AgentError` (they were clamped to 1 silently). The same rule, with `ValueError` kept, now covers `AgentServer(max_concurrent=, max_input_chars=)` (minimum 1), `RetryingToolRegistry(max_retries=)` (0), `CachingToolRegistry(max_entries=)` (1; 0 used to raise `StopIteration` from the first cached call) and `SemanticMemoryStore(max_entries=)` (0). The tool-timeout error message (`TOOL_BAD_TIMEOUT`) also shows a bounded value, so a 5000-digit `timeout_s` is a `ToolExecutionError`, not a `ValueError`. The artifact builder's int bound (step 12.7) uses the same `Defaults.MAX_INT_LIMIT`.
- Agents tools (plan 0068b055 step 12.12, D-020, live finding D2): a `dict[...]` parameter that is not the lone positional dict-style one (keyword-only, or beside another parameter), bare or inside `Optional`, is now advertised as `{'type': 'object'}` in the inferred `parameter_schema` (it was `{'type': 'string'}`; bare `dict` was already `object`, and the model-visible args-model schema already said `object`, so only a tool without an args model, e.g. with an unannotated sibling parameter, showed the model a string). No example tool changes (every example tool takes the lone `params: dict`). The runtime is unchanged: a JSON string sent for an `object` (or `array`) parameter still reaches the function as a `str`, as before; the registry has no JSON-string coercion for lists, so none was added for dicts.
- `fsm-llm-eval examples` scoring (plan 0068b055 step 12.19, D-020, live finding D1): a run that exits 0 but prints a line-anchored `Error:` (or `Error during ...`) or `Success: False` is now capped at score 2 (failure codes `F-CODE` and `F-MODEL`), and an extraction rate of at most 50% is a partial failure (`extracted * 2 <= total`; exactly 50% used to pass because the test was a strict `< 50%`). Scores of examples that report failure are therefore lower than before, so earlier eval scores and baselines (Runs 001 to 009, the Run 006 and 007 figures) are not comparable with new ones; the heuristic already overstated. `Error:` text inside prose mid-line, `Error Rate:` metric lines and stderr are not read.
- `fsm-llm-eval examples` timeouts (plan 0068b055 step 12.21, D-020, live finding D5): 23 examples whose agents run on the default `AgentConfig.timeout_seconds` (300 s) had a 300 s harness timeout, so the agent's own timeout and the harness kill fired together (`supply_chain_optimizer`: an `Error: Agent timed out after 300.0 seconds` line scored 4, or a score-1 kill, by chance). Their `EXAMPLE_TIMEOUTS` entries are now 360 (budget + 60 s) in `fsm_llm.eval.constants`: `agents/adapt_with_memory`, `agent_as_tool`, `agent_memory_chain`, `architecture_review`, `debate`, `eval_opt_structured`, `evaluator_optimizer`, `hierarchical_orchestrator`, `investment_portfolio`, `legal_document_review`, `medical_literature`, `memory_agent`, `multi_debate_panel`, `orchestrator`, `orchestrator_specialist`, `pipeline_review`, `react_structured_pipeline`, `regulatory_compliance`, `security_audit`, `supply_chain_optimizer`, and `workflows/customer_onboarding`, `release_management`, `workflow_agent_loop`. A hung example in this group now takes 60 s longer to be killed. A test scans `examples/` and fails when a harness timeout ties an agent budget.
- `fsm-llm-eval examples` stdin scripts (plan 0068b055 step 12.22, D-020, live finding D6): `agents/react_hitl_combined` was sent `Search for weather in Paris\ny\nquit\nquit`; that task calls no gated tool, so no approval was asked and the `y` ran as a second task. It now gets a task that names the gated tool, `Use the publish_report tool to publish a one-paragraph summary of recent AI papers\ny\nquit` (the `y` answers the `publish_report` approval prompt; a model that skips the gated tool still turns it into a second task). The first wording of this step, `Research recent AI papers and publish a summary report`, did not make the default model call the tool (3 of 3 live runs in `findings/live-runs-3.md`: no prompt, the model claimed the publication in prose); the explicit wording did (step 12.31, 3 of 3 live runs on `ollama_chat/bonsai-2:27b`). The three `meta/build_*` scripts lose their trailing `quit` (`run_interactive()` reads to EOF and sent it to the builder as a message, one more model call), and five scripts for examples that never read stdin (`classification/intent_routing`, `classification/classified_transitions`, `agents/reasoning_tool`, `agents/tool_decorator`, `agents/skill_loader`) are removed. A test reads every script against its example's `input()` calls.
- Agents: a final reply cut by `max_tokens` ends the run with `success=False` and the new `stop_reason` `"truncated"` (`StopReason.TRUNCATED`, one of `StopReason.FORCED`; plan 0068b055 step 12.28, D-025). Before, the cut-off message of step 12.23 (or a cut prose reply) was an agent's final answer with `success=True` and, in `fsm-llm-eval`, a score 4 (live: `supply_chain_optimizer`, `legal_document_review`). Core now writes one internal context key, `fsm_llm.definitions.CONTEXT_KEY_CUT_OFF` (`"_cut_off"`), for the turn whose Pass-2 reply, field extraction or completion-state call was cut (`{"reply": True}` and/or `{"fields": [names]}`; absent when nothing was cut; removed at the start of every turn, so it describes the last turn only; internal, so in no prompt, not in `API.get_data` and not in a saved session), from the new `truncated: bool = False` field of `ResponseGenerationResponse` and `FieldExtractionResponse` that `llm.py` sets from the provider's `finish_reason` (a custom `LLMInterface` may set it). `BaseAgent._run_conversation_loop` reads it once and hands it to the success seam (`_run_outcome`, which applies it only when the answer text was cut, step 12.33), so every pattern that runs on `_standard_run` and ADaPT report it; the answer text still ships, and a stop that was already forced (`max_iterations`, `forced_pass`, `stalled`, `gate_failed`, `ended`) keeps its own reason. REWOO stops on a cut `plan_blueprint`: no tool runs, `success=False`, `stop_reason="truncated"`, the answer says the plan was cut off at `max_tokens` (it used to run an empty plan with no cause visible), and `final_context["plan_cut_off"]` is `True`; its `solve` call still happens. A streamed reply (`converse_stream`, `run_stream`) is never flagged. The completion-state result stays `{kind, text, calls}`. Revert rule: none; the change only turns a cut run from success to failure.
- Agents: `stop_reason="truncated"` now applies only when the text that ships as the answer was cut (plan 0068b055 step 12.33, D-027, review pass 3). Step 12.28 flagged any cut in the last turn, so EvaluatorOptimizer, MakerChecker, ADaPT and PromptChain, whose answer is a context key (`generated_output`, `draft_output`, `attempt_result`, the last chain result), read `success=False, truncated` when only the unused closing reply (`output`, `combine`) was cut, although the answer was whole. The verdict moved out of `_run_conversation_loop` into the one success seam: the loop hands the cut-off record on in `final_context`, `BaseAgent._answer_was_cut_off` reads it against `BaseAgent._answer_source_keys` (where `_extract_answer` read the answer; ADaPT and PromptChain override both together) and `_run_outcome(answer_cut_off=)` returns `(False, "truncated")`. A reply answer (ReAct, Reflexion, Debate, ...) is flagged by any cut of the last turn as before; a key answer only by a cut of that key's own field. `_run_outcome(..., answer_cut_off=False)` is a new optional keyword. EvaluatorOptimizer and MakerChecker (`judged=True`) no longer lose the check on the limiter's last round: the bare `max_iterations_reached` flag, which they ignore for success, used to suppress it. Revert rule: none; a whole answer is no longer failed.
- `fsm-llm-eval examples` SIGTERM (plan 0068b055 step 12.34, D-028, review pass 3 NOTE 5): the process now exits right after the interrupt. The report was written at once, but the process stayed alive until every running example subprocess finished (10.4 s with a 15 s example, up to the 360 s harness timeout in real runs), because the pool threads are non-daemon and SIGTERM sent to the parent pid reaches no child. Examples now run through `fsm_llm.eval._pool.run_tracked` (`Popen` and `communicate`, same arguments, timeout, partial output and `PYTHONUNBUFFERED` as before): on POSIX each child starts in its own session, an interrupt (Ctrl-C, SIGTERM, an exception from the progress callback) terminates every running child's process group (SIGTERM, `CHILD_TERMINATE_GRACE_SECONDS` = 2.0, then SIGKILL) and a child started after the interrupt is refused, so the example's own subprocesses go too; a timeout now kills the group as well, not only the child. `fsm-llm-eval run` (cases) starts no subprocess: its trials are threads of the process and still finish in the background after an interrupt (a hung provider call is bounded by the request timeout).
- Agents (plan 0068b055 step 12.35, D-030, review pass 3 notes 7 to 9): `AgentConfig` refuses an int field above `Defaults.MAX_INT_LIMIT` (2**63-1; `max_iterations`, `max_tokens`, `max_history_size`, `reflect_every_n`; `MetaBuilderConfig` inherits it): `AgentConfig(max_iterations=10**5000)` was accepted and its `repr()` raised `ValueError` (the 4300-digit limit), the bug class of `num_rounds` in step 12.8. The reason text is `must be at most 9223372036854775807`; `AgentArtifactBuilder.set_config` reports it through `AgentConfig` (its own copy of the bound is gone). A tool parameter annotated `Mapping[...]`, `MutableMapping[...]` (also bare or in `Optional`) or a `TypedDict` is advertised as `{"type": "object"}` by `@tool` and `register_function`, as `dict[...]` was in step 12.12 (it was `string`); the lone positional dict-style call stays `dict` only. A model that sends the flat object `{"x": 1}` for a keyword-only mapping parameter `payload` no longer runs the tool with `payload=1` (the fallback put the first stray value into the dict parameter): the call fails with the "Tool requires parameters: ['payload']" message so the model can name it; the whole flat object is NOT passed in as the dict (that is the bare-dict rule, for a lone positional parameter only). A scalar parameter keeps the fallback.
- Docs (plan 0068b055 step 12.35): `src/fsm_llm/README.md` said "a half JSON answer is never shown as the reply"; it now says an unreadable cut envelope is replaced by a cut-off message while an envelope whose `message` can be read ships as read and may be partial (for example `Use the config {\\`), and that the reply is `truncated` on every rung (a test pins it).
- The package now has six subpackages (`reasoning`, `workflows`, `agents`, `monitor`, `eval`, `synthesis`): the extra `synthesis` is requested by `all`, `make install-dev`, tox and CI, `tests/test_packaging.py` pins it, and the architecture diagrams and subpackage lists of `README.md`, `AGENTS.md`, `src/` and `tests/` say six. The synthesis suite is `tests/test_fsm_llm_synthesis/`.
- `FSMValidator` finds trap components through `flow_analysis.strongly_connected_components` (the iterative Tarjan moved there; behaviour kept, pinned by a golden table written before the move). `definitions.logic_referenced_keys` is built on a shared logic walk, and `definitions.handler_only_names` is the one answer to "which keys no extraction may write" for the pipeline and `flow_analysis`. No public name changed or was removed.
- **BREAKING:** the default definition `version` is now `"4.2"` (was `"4.1"`; `fsm_llm.core.definitions.schema.SCHEMA_VERSION_DEFAULT`, which `FSMDefinition.version` and `FSMArtifactBuilder.to_dict()` follow), the version that introduced the `synthesize` state field. A definition that omits `version` or declares `"4.2"` keeps its `fsm_id` and library digest (the id hashes `model_dump(exclude_defaults=True)`, so the default is not part of it). A definition that declares `"version": "4.1"` explicitly gets a new `fsm_id` and a new library digest: a session saved under the old id only logs the existing id-mismatch WARNING when restored, and a stored procedure that declares `"4.1"` is stored again under its new digest (a duplicate, not a break). Definitions declaring `"3.0"` or `"4.0"` already hashed their version and do not move. `examples/` are not edited.
- Layout (plan d454fd62, step 65, the first of the 0.4 module splits): `fsm_llm.reasoning.reasoning_modes` is now a package with one module per FSM (`orchestrator`, `classifier`, `simple_calculator`, `analytical`, `deductive`, `inductive`, `creative`, `critical`, `abductive`, `analogical`, `hybrid`), `registry` (`ALL_REASONING_FSMS`) and `_phrases` (the shared no-questions sentences). The import path and every public name are unchanged (`from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS, hybrid_fsm` still works, the package `__init__` re-exports them); the FSM definitions are value-identical. Only the file path `reasoning/reasoning_modes.py` and the private names `_NO_QUESTIONS`, `_IMPORTANT_NO_QUESTIONS`, `_PRESENT_FINDINGS`, `_PRESENT_FINAL` (now in `reasoning_modes._phrases`) moved, so nothing here is a break. A mock patch aimed at one of these names must aim at the defining module, for example `fsm_llm.reasoning.reasoning_modes.registry.ALL_REASONING_FSMS`; a new guard test fails a patch aimed at a package that only re-exports the name.
- Layout (plan d454fd62, step 66, the second of the 0.4 module splits): `fsm_llm.agents.constants` is now a package of six modules, `core` (`ContextKeys`, `Defaults`, `ErrorMessages`, `LogMessages`), `states` (the per-pattern state classes and the native_fc names), `keys` (the run-output, framework-only and dropped key sets, the misplaced and config-owned kwarg sets, `ReasoningIntegrationKeys`), `outcomes` (`StopReason`, `ToolRunStatus`, `ToolObservationPrefix`, `OBSERVATION_ENTRY`), `handler_names` (`HandlerPriorities`, `HandlerNames`) and `meta` (the meta-builder classes and `META_*` constants). The import path and every public name are unchanged (`from fsm_llm.agents.constants import ContextKeys, Defaults, StopReason` still works, the package `__init__` re-exports all 53 names with a static `__all__`); every class and constant is value-identical (checked per definition and by value and `repr` against the old module), class names and members are unchanged. Only the file path `agents/constants.py` moved, so nothing here is a break. The agents modules now import it by its absolute path (`from fsm_llm.agents.constants import ...`, the cross-package form). The synthesis bench pins (`SRC_PINS` in `scripts/synthesis_bench.py`) name the seven new files instead of the old one; the frozen manifests under `scripts/bench_data/` are untouched.
- Layout (plan d454fd62, step 67, the third of the 0.4 module splits and the first new layer package): `fsm_llm.security` is now a package of six modules, `names` (`INTERNAL_KEY_PREFIXES`, `has_internal_prefix`, `is_forbidden_context_entry`, the name-scan data and the value judges the decision calls), `paths` (`resolve_within`), `api_keys` (`checked_api_key`, `presented_api_key`, `api_key_matches`), `patterns` (layer 1: `FORBIDDEN_CONTEXT_PATTERNS`, `COMPILED_FORBIDDEN_CONTEXT_PATTERNS` and the regex fragments), `shape_data` (layer 2 tables) and `shapes` (layer 2 detectors). The import path and every public name are unchanged (`from fsm_llm.security import has_internal_prefix, is_forbidden_context_entry, resolve_within` still works, the package `__init__` re-exports the 9 public names with a static `__all__`; `fsm_llm.constants` still re-exports its five); every table, regex and function is value-identical (checked per definition with `ast.dump` and at run time, regex patterns and flags included, plus 6,954 calls of the decision functions on a corpus against the old module). The 61 private names (`_looks_like_credential_value`, `_CREDENTIAL_VALUE_PREFIXES`, `_TOKEN_VALUE_SCAN_NAME_RE` and the rest) are no longer attributes of `fsm_llm.security`: import one from the module that defines it (`fsm_llm.security.shapes`, `shape_data`, `patterns` or `names`); no importer outside the tests used one, so nothing here is a break for the public API. A mock patch or an assignment aimed at a private name must aim at the module that performs the lookup (for example `fsm_llm.security.shapes._CREDENTIAL_VALUE_PREFIXES`, not `shape_data`). `security.hmac` is gone with the module attribute: the two tests that spied on `hmac.compare_digest` read it from `fsm_llm.security.api_keys`. `fsm_llm.security` is a layer package: it ships with the core and needs no extra, `tests/test_packaging.py` pins it in `LAYER_PACKAGES`, and `.importlinter` gains the contract `security-is-a-leaf` (it imports the standard library only).
- Layout (plan d454fd62, step 68, the fourth of the 0.4 module splits): `fsm_llm.agents.fsm_definitions` is now a package with one module per pattern builder (`react`, `reflexion`, `plan_execute`, `parallel_react`, `rewoo`, `prompt_chain`, `self_consistency`, `debate`, `orchestrator`, `adapt`, `evalopt`, `maker_checker`, `native_fc`, `meta_builder`), `_common` (the state, edge and condition constructors every builder shares, `_finalize_fsm` and `_flag_condition` included) and `_loop` (the think, act and conclude pieces of the loop patterns, `conclude_on_evidence_logic`, `loop_field_context_keys` and the typed-field helper `_field`); the largest module is 360 lines (the file had 2,492). The package `__init__` re-exports the 14 `build_*_fsm` builders and the two public helpers with a static `__all__`, so every public import path (`from fsm_llm.agents.fsm_definitions import build_react_fsm`, used by an example) is unchanged and no builder output moved (51 golden cases equal in value and `repr` to the old file; `fsm_id`s unchanged). Not BREAKING: no public name moved; the 21 private names (`_state`, `_edge`, `_field`, `_finalize_fsm`, `_await_approval_state` and the rest) are no longer attributes of the package and are imported from their defining module (`fsm_llm.agents.fsm_definitions._common`, `_loop`, `native_fc`); no non-test importer used one.
- **BREAKING:** the module `fsm_llm.definitions` moves to the new layer package `fsm_llm.core.definitions` (plan d454fd62, step 69, the fifth of the 0.4 module splits and the first of the `core` layer; `fsm_llm.core` itself is an empty package). The old path no longer imports, with no alias and no shim: `from fsm_llm.definitions import X` becomes `from fsm_llm.core.definitions import X`, and every public name (the 74 names of the old module that do not start with an underscore: the models, the exceptions, the constants `ASCII_IDENTIFIER_PATTERN`, `MAX_PRIORITY`, `PROMPT_MAX_CHARS`, `CONTEXT_KEY_CUT_OFF` and the rest, and the helpers `logic_referenced_keys`, `reachable_from`, `typed_field_extraction`, `handler_only_names`) is re-exported by the new package with a static `__all__`. Names imported from `fsm_llm` (`from fsm_llm import State, FSMError, ...`) are unchanged. The package has eleven modules (`limits`, `errors`, `completion`, `extraction`, `pass_models`, `classification`, `transition`, `kinds`, `state`, `results`, `fsm`), none over 800 lines. What a user can see: `type(exc).__module__`, `Model.__module__`, tracebacks and `repr` of a class now show `fsm_llm.core.definitions.<module>` (for example `fsm_llm.core.definitions.state.State`) instead of `fsm_llm.definitions`; a pickle of a model made by an earlier version names the old path and no longer loads (the library itself pickles nothing: sessions and snapshots are JSON); the 13 private names of the old module (`_PromptRequest`, `_coerce_entity_values`, `_REPORTED_DEFINITIONS`, `_SYNTHESIZE_SAMPLES` and others) are attributes of the module that defines them, not of the package, and a `patch("fsm_llm.definitions._x")` must name that module. `fsm_id`, `model_dump`, `model_json_schema`, the validators and their order, and every error message are unchanged (proved on the 14 example FSMs and the 114 cases of the kind-equality corpus against the old file); one pydantic detail changes: `FSMInstance` fields that named classes defined later in the old file (`ClassificationResult`) are resolved when the class is created, not at its first use.
- **BREAKING:** the module `fsm_llm.transition_evaluator` (present in 0.12.0) moves to `fsm_llm.core.transition_evaluator` (plan d454fd62, step 70, with three modules of this unreleased train that were never published at the old path: `fsm_llm.codes` is now `fsm_llm.core.codes`, `fsm_llm.state_kinds` is `fsm_llm.core.state_kinds` and `fsm_llm.schema_registry` is `fsm_llm.core.definitions.schema`, so the Added bullets above name the new paths and need no BREAKING bullet of their own). The old paths no longer import, with no alias and no shim: `from fsm_llm.transition_evaluator import TransitionEvaluator` becomes `from fsm_llm.core.transition_evaluator import TransitionEvaluator`, and `TransitionEvaluator` and `TransitionEvaluatorConfig` are still exported from `fsm_llm`. Files are moved whole, the code of the four modules is unchanged (their import lines and two docstring paths aside); `tests/fixtures/api_snapshot.txt` changes only in the path prefix of the `codes` and `schema_registry` lines. `core/definitions/` now holds a twelfth module, `schema`.
- **BREAKING:** the module `fsm_llm.expressions` (present in 0.12.0) moves to the package `fsm_llm.core.expressions` (plan d454fd62, step 71, the seventh of the 0.4 module splits). The old path no longer imports, with no alias and no shim: `from fsm_llm.expressions import evaluate_logic` becomes `from fsm_llm.core.expressions import evaluate_logic`, and the 13 public names of the old module (`evaluate_logic`, `is_missing`, `get_var`, `missing`, `missing_some`, `soft_equals`, `hard_equals`, `less`, `less_or_equal`, `greater`, `greater_or_equal`, `cat`, `operations`) are re-exported by the new package `__init__` with a static `__all__`. `from fsm_llm import evaluate_logic` is unchanged. The package has seven modules, one per operator group (`comparison`, `arithmetic`, `logic_and_text`, `access` for variable access and the "missing" predicate, `registry` for the eager operator table, `evaluator` for the recursive evaluator, `undefined` for the private UNDEFINED sentinel and the None / UNDEFINED rule sets), the largest 317 lines (the file had 1,118); `ALLOWED_JSONLOGIC_OPERATIONS` is still defined once, in `fsm_llm.constants`. No behaviour changed: every function body is `ast.dump`-equal to the old file and 6,029 generated expressions (every allowed operator, edge operands, errors and log records) give the same result, exception type and message and the same log records as the old module. What a user can see: `__module__` and tracebacks name `fsm_llm.core.expressions.<module>`; the 22 private names of the old module (`_numeric_equal`, `_UNDEFINED`, `_data_operators`, `_SHORT_CIRCUIT_OPERATORS`, `_evaluate_short_circuit` and the rest) are attributes of the module that defines them, not of the package, and a mock patch such as `patch("fsm_llm.expressions.logger")` must name the module that performs the lookup (for example `fsm_llm.core.expressions.evaluator.logger`); `fsm_llm.core.expressions.operations` is the same dict object the evaluator reads, so adding an entry to it still takes effect.
- **BREAKING:** the module `fsm_llm.utilities` (present in 0.12.0) is split by layer and no longer exists (plan d454fd62, step 72, the eighth of the 0.4 module splits). The old path does not import, with no alias and no shim; each name moves to the module that defines it: `resolve_reasoning_trace`, `remove_think_blocks`, `strip_think_and_fences` to `fsm_llm.llm.text`; `extract_json_from_text`, `balanced_json_objects` to `fsm_llm.llm.json_extraction`; `ContextFilterError`, `filter_context_tree`, `redact_non_json_leaf`, `redacting_json_default` to `fsm_llm.context.filtering`; `redact_context`, `REDACTED`, `CYCLE_PLACEHOLDER` to `fsm_llm.core.redaction` (first placed in `context.filtering` at step 72, moved down at step 95); `coerce_confidence`, `coerce_int`, `coerce_float`, `parse_bool_word` to `fsm_llm.core.coerce`; `read_json_object`, `load_fsm_from_file`, `load_fsm_definition`, `get_fsm_summary` to `fsm_llm.core.loaders` (the 13 private helpers moved with the code that uses them, for example `_cycle_reaching_containers` to `context.filtering`, `_redact_mapping` to `core.redaction` and `_match_brace_partners` to `llm.json_extraction`). Names imported from the package root (`from fsm_llm import extract_json_from_text, get_fsm_summary, load_fsm_definition, load_fsm_from_file`) are unchanged. `fsm_llm.llm` and `fsm_llm.context` keep their dotted paths but are now packages (the layer packages `llm` and `context`; `llm/__init__.py` and `context/__init__.py` hold the code of the old `llm.py` and `context.py` unchanged apart from import lines), so `from fsm_llm.llm import LiteLLMInterface` and a patch such as `patch("fsm_llm.llm.completion")` keep working; only a read of the file paths `src/fsm_llm/llm.py` and `src/fsm_llm/context.py` changes. `ContextFilterError` is now defined in `fsm_llm.context.filtering` (its `__module__`, its traceback line and its repr say so; it is still an `FSMError` and a `ValueError`). The code of every moved function is unchanged (checked per definition and against the old module on more than 54,000 generated inputs); a mock patch or `monkeypatch` aimed at a moved private name must aim at the new defining module (for example `fsm_llm.core.redaction.logger` for the redaction warning, `fsm_llm.llm.json_extraction.logger` for the extraction logs). `fsm_llm.reasoning.utilities` is a different module and is unchanged.
- **BREAKING:** the module `fsm_llm.ollama` (present in 0.12.0) moves to `fsm_llm.llm.ollama` (plan d454fd62, step 73, the ninth of the 0.4 module splits). The old path does not import, with no alias and no shim: `from fsm_llm.ollama import is_ollama_model` becomes `from fsm_llm.llm.ollama import is_ollama_model`; the four public helpers (`is_ollama_model`, `apply_ollama_params`, `prepare_ollama_messages`, `build_ollama_response_format`) are unchanged. `fsm_llm.llm.ollama` is not re-exported by `fsm_llm.llm`.
- **BREAKING:** the module `fsm_llm.llm` (a single module in 0.12.0, a package since step 72) is split by concern (plan d454fd62, step 73): `fsm_llm.llm.interface` (`LLMInterface`, `tool_exchange`, `check_tool_transcript`, `implements_complete`, `interface_model`, `resolve_model`, `PLACEHOLDER_MODEL`, the `DEFAULT_*` connection defaults), `fsm_llm.llm.usage` (the usage meter), `fsm_llm.llm.parsing` (the reply parsers), `fsm_llm.llm.providers.litellm` (`LiteLLMInterface`), `fsm_llm.llm.providers.request` (the request builder) and `fsm_llm.llm.providers.embedder` (`LiteLLMEmbedder`). Every public name still imports from `fsm_llm.llm` (and `from fsm_llm import LiteLLMInterface` is unchanged), so a caller of the public API changes nothing. Three things break: the names `completion`, `embedding` and `get_supported_openai_params` are no longer attributes of `fsm_llm.llm`, so a test or host that patched `fsm_llm.llm.completion` (or `.get_supported_openai_params`) must patch `fsm_llm.llm.providers.litellm.completion` (or `.get_supported_openai_params`) and a patch of `fsm_llm.llm.embedding` becomes `fsm_llm.llm.providers.embedder.embedding`, the modules whose code looks them up (a patch of the old name raises `AttributeError`); private helpers are no longer importable from `fsm_llm.llm` and come from their defining module (`_Metered`, `_meter_of`, `_UsageMeter` from `fsm_llm.llm.usage`, `_finish_reason` and `_GENERIC_FALLBACK_MESSAGE` from `fsm_llm.llm.parsing`, `_completion_response` from `fsm_llm.llm.interface`, `_fill_empty_user_turns` from `fsm_llm.llm.providers.request`, `_embedding_vectors` from `fsm_llm.llm.providers.embedder`); and litellm is now imported in exactly one module, `fsm_llm.llm.providers.litellm` (importing `fsm_llm.llm` still sets `litellm.suppress_debug_info = True` once and still skips a `.env` file). The reply parsers and the request builder are mixins (`ParseMixin`, `RequestMixin`) of `LiteLLMInterface`; its public methods and the API snapshot are unchanged.
- **BREAKING:** the module `fsm_llm.prompts` (present in 0.12.0) is split by pass into the package `fsm_llm.context.layout` and no longer exists (plan d454fd62, step 74, the tenth of the 0.4 module splits). The old path does not import, with no alias and no shim. Every public name imports from the package `fsm_llm.context.layout`, so `from fsm_llm.prompts import BasePromptBuilder` becomes `from fsm_llm.context.layout import BasePromptBuilder`: `BasePromptConfig`, `BasePromptBuilder`, `HistoryManagementStrategy` (defined in `fsm_llm.context.layout.base`), `DataExtractionPromptConfig`, `DataExtractionPromptBuilder`, `FieldExtractionPromptConfig`, `FieldExtractionPromptBuilder` (`fsm_llm.context.layout.extraction`), `ResponsePromptConfig`, `ResponseGenerationPromptBuilder` (`fsm_llm.context.layout.response`), `build_classification_json_schema`, `build_classification_system_prompt`, `build_classification_context_block`, `sanitize_text_for_prompt` (`fsm_llm.context.layout.classification`). Three names the package surface does not carry come from their defining module: the private helpers (`_cap_history_line`, `_history_messages` from `fsm_llm.context.layout.base`), and `ClassificationPromptConfig`, which `fsm_llm.prompts` only imported and now comes from `fsm_llm.core.definitions` (where it is defined). The prompt text is byte-identical: a differential run of every builder over generated inputs matched the old module. The names `from fsm_llm import DataExtractionPromptBuilder, DataExtractionPromptConfig, FieldExtractionPromptBuilder, FieldExtractionPromptConfig, ResponseGenerationPromptBuilder, ResponsePromptConfig, build_classification_json_schema, build_classification_system_prompt` are unchanged.
- **BREAKING:** the package `fsm_llm.context` (the module `context.py` of 0.12.0, a package since step 72) is split (plan d454fd62, step 74): its code moves to `fsm_llm.context.compactor` (`ContextCompactor`, `clean_context_keys`) and `fsm_llm.context.commit` (`commit_with_key_rollback`, `was_cut_off`, `record_cut_off`), and `clear_keys_delta` moves down from `fsm_llm.handlers` to `fsm_llm.context.commit` so the `context` layer imports no engine module. Every one of those names still imports from `fsm_llm.context`, and `from fsm_llm.handlers import clear_keys_delta` and `from fsm_llm import clear_keys_on_entry, ContextCompactor` are unchanged (`handlers` binds the moved function). What breaks: names `fsm_llm.context` only imported are no longer its attributes (`MAX_CONTEXT_FILTER_DEPTH` comes from `fsm_llm.constants`, `logger` from `fsm_llm.logging`), and a patch of `fsm_llm.context.logger` must aim at the module that logs, `fsm_llm.context.compactor.logger`; `fsm_llm.handlers.clear_keys_delta` is the same function object as `fsm_llm.context.commit.clear_keys_delta`, so a patch of the old name changes only the `handlers` binding (`clear_keys_on_entry` looks the name up there), not `ContextCompactor` or the agents.
- **BREAKING:** the module `fsm_llm.validator` (present in 0.12.0) is split into the package `fsm_llm.core.verify` and no longer exists (plan d454fd62, step 75, the eleventh of the 0.4 module splits). The old path does not import, with no alias and no shim. The validator names import from the package: `from fsm_llm.validator import FSMValidator` becomes `from fsm_llm.core.verify import FSMValidator`, and the same for `FSMValidationResult`, `validate_fsm_from_file`, `upgrade_fsm_file` and `UpgradeResult` (defined in `fsm_llm.core.verify.validator`, `.result`, `.upgrade`); `from fsm_llm import FSMValidator, FSMValidationResult, validate_fsm_from_file` is unchanged. The CLI functions `main` and `main_cli` are not re-exported: import them from `fsm_llm.core.verify.cli`, and the console script `fsm-llm-validate` now points at `fsm_llm.core.verify.cli:main_cli` (was `fsm_llm.validator:main_cli`; an editable install must run `pip install -e .` once to refresh the script, a wheel install gets it with the new release). The private helpers move with their code (`_string_literals`, `_condition_keys`, `_RESULT_KINDS` to `fsm_llm.core.verify.structure`; `_no_duplicate_keys`, `_refuse` to `fsm_llm.core.verify.upgrade`). `FSMValidator` is composed of `StructureMixin` (`structure.py`) and `GraphMixin` (`graph.py`) over a shell, its method bodies unchanged: `validate()` returns the same errors, warnings and info in the same order (a differential run over the recorded corpus and 1,600 generated definitions matched the old module), `fsm-llm-validate` prints the same report and exits with the same codes, and a patch of a name inside the validator aims at the module that performs the lookup (`fsm_llm.core.verify.validator.FSMDefinition`, `fsm_llm.core.verify.cli.setup_cli_logging`).
- `fsm_llm.flow_analysis` (added in this release, never in a tagged version, so no BREAKING entry) is placed as `fsm_llm.core.verify.flow`, next to the validator that shares its component primitive: `analyze_data_flow`, `analyze_liveness`, `declared_available`, `FlowFinding` and `strongly_connected_components` keep their signatures (`tests/fixtures/api_snapshot.txt` differs only in the module header lines), and the old path does not import.
- **BREAKING:** the module `fsm_llm.visualizer` (present in 0.12.0) is split into the package `fsm_llm.core.visualize` and no longer exists (plan d454fd62, step 76, the twelfth of the 0.4 module splits). The old path does not import, with no alias and no shim. The public names import from the package: `from fsm_llm.visualizer import build_fsm_graph` becomes `from fsm_llm.core.visualize import build_fsm_graph`, and the same for `FSMGraph`, `FSMGraphEdge`, `FSMGraphNode`, `to_mermaid`, `to_dot`, `visualize_fsm_ascii`, `visualize_fsm_from_file`, `ASCII_STYLES` and `OUTPUT_FORMATS`; `from fsm_llm import build_fsm_graph, to_mermaid, to_dot, visualize_fsm_ascii, visualize_fsm_from_file, FSMGraph, FSMGraphNode, FSMGraphEdge` is unchanged. The other names import from the module that defines them: the glyph tables and the ASCII geometry (`BOX_STYLES`, `ARROW_STYLES`, `ICONS`, `build_graph_representation`, `calculate_depths`, `find_longest_path`, `sort_states_logically`, `sort_states_by_depth`, `build_state_boxes`, `detect_loops` and the private row helpers such as `_fit`) from `fsm_llm.core.visualize.ascii_layout`; the section and diagram builders (`build_fancy_header`, `build_metadata_section`, `build_persona_section`, `build_states_section`, `build_transitions_section`, `build_legend`, `generate_enhanced_ascii_diagram`, `generate_compact_ascii_diagram`, `generate_minimal_ascii_diagram`) from `fsm_llm.core.visualize.ascii_render`; the shape checks (`_require_definition_shapes` and its siblings) from `fsm_llm.core.visualize.shapes`; the console-script functions `main` and `main_cli` from `fsm_llm.core.visualize.cli` (they are not re-exported). The console script `fsm-llm-visualize` now points at `fsm_llm.core.visualize.cli:main_cli` (was `fsm_llm.visualizer:main_cli`; an editable install must run `pip install -e .` once to refresh the script, a wheel install gets it with the new release), and `fsm-llm --mode visualize` is unchanged. The function bodies are the old ones, moved: the ASCII, Mermaid and DOT output is byte-identical for every recorded definition and for generated ones (a differential run against the old module compared the three formats, every style, the file and CLI paths and the helpers), and the one change inside a body is the lazy `cli_support` import in `main`, now absolute. The loguru record name of the CLI diagnostics reads `fsm_llm.core.visualize.cli` instead of `fsm_llm.visualizer`.
- Layout (plan d454fd62, step 77, the thirteenth of the 0.4 module splits and the fifth layer package): `fsm_llm.handlers` is now a package of seven modules at the same dotted path, `protocol` (`HandlerTiming`, `HandlerErrorMode`, `FSMHandler`, `ExecutionLambda`, `ConditionLambda`), `errors` (`HandlerSystemError`, `HandlerExecutionError`), `base` (`BaseHandler`), `system` (`HandlerSystem`), `lambda_handler` (`LambdaHandler`), `builder` (`HandlerBuilder`, `create_handler`) and `helpers` (`clear_keys_on_entry`). The import path and every public name are unchanged (`from fsm_llm.handlers import HandlerTiming, HandlerSystem, HandlerBuilder, create_handler, clear_keys_on_entry, clear_keys_delta, ...` still works, the package `__init__` re-exports the 14 names with a static `__all__`, and `from fsm_llm import HandlerTiming, create_handler, clear_keys_on_entry` is unchanged), so nothing here is a break for the public API. Every class and function body is the old one, moved (checked per definition with `ast.dump`, and at run time against the old module: seeded generated handler systems, builder programs and executions compared in every outcome, log record and handler trace). The private name `_as_set` and the module attributes the old file carried by accident (`copy`, `logger`, `threading`, `traceback`, `contextvars`, `math`, `numbers`) are no longer attributes of `fsm_llm.handlers`: import `_as_set` from `fsm_llm.handlers.lambda_handler`, and aim a mock patch at the module that performs the lookup (for example `fsm_llm.handlers.system.copy.deepcopy`, not `fsm_llm.handlers.copy.deepcopy`). The classes report a new `__module__` (`fsm_llm.handlers.system`, `fsm_llm.handlers.errors`, ...), and the loguru record name of the handler system reads `fsm_llm.handlers.system` (and `fsm_llm.handlers.lambda_handler`) instead of `fsm_llm.handlers`; an exception pickled with the old class path still loads, because the old path re-exports it.
- **BREAKING:** the module `fsm_llm.fsm` (present in 0.12.0) is split into the package `fsm_llm.engine.fsm` and no longer exists (plan d454fd62, step 78, the fourteenth of the 0.4 module splits and the sixth layer package, `fsm_llm.engine`, whose `__init__` is empty). The old path does not import, with no alias and no shim. `FSMManager` is defined in `fsm_llm.engine.fsm.manager`: `from fsm_llm.fsm import FSMManager` becomes `from fsm_llm.engine.fsm.manager import FSMManager` (or `from fsm_llm.engine.fsm import FSMManager`, the package re-exports it with `with_conversation_context`), and `from fsm_llm import FSMManager` is unchanged. `with_conversation_context` and the private helpers come from `fsm_llm.engine.fsm.helpers` (`_strip_internal_mapping`, `_unexpected_turn_failure`, `_conversation_entry`); the two TypeVars `_TurnT` and `_SnapshotT` live in `fsm_llm.engine.fsm.turns` and `fsm_llm.engine.fsm.snapshots`. Names the old module only imported (`FSMError`, `FSMDefinition`, `HandlerTiming`, `threading`, `logger`, ...) were never part of its API: import them from where they are defined (`from fsm_llm import FSMError`), and aim a patch at the module that performs the lookup, for example `fsm_llm.engine.fsm.snapshots.END_CONVERSATION_LOCK_TIMEOUT_SECONDS` (read by `_acquire_or_refuse`; was `fsm_llm.fsm.END_CONVERSATION_LOCK_TIMEOUT_SECONDS`). `FSMManager` is composed of two mixins, `TurnsMixin` (`turns.py`: the turn guard, the locked sync turn, the lazy locked stream, the ERROR handlers) and `SnapshotsMixin` (`snapshots.py`: the reads, snapshots and the restore seed), over the shell in `manager.py`; every method body is the old one, moved (the same signatures, `tests/fixtures/api_snapshot.txt` is byte-equal; the one annotation change is the `manager` parameter of the private `_conversation_entry`, typed by a small `_ConversationRegistry` protocol so the mixins can pass `self`), and the lock order (`_lock` before `conv_lock`, the lazy `conv_lock` of the stream closure, `_fsm_cache_lock` a leaf) is unchanged and now pinned by `tests/test_fsm_llm/test_fsm_lock_order.py`. The loguru record name of the manager reads `fsm_llm.engine.fsm.manager` (`.turns`, `.snapshots`, `.helpers` for the turn, read and helper code) instead of `fsm_llm.fsm`.
- **BREAKING:** the module `fsm_llm.pipeline` (present in 0.12.0) is split into the package `fsm_llm.engine.turn` and no longer exists (plan d454fd62, step 79, the fifteenth of the 0.4 module splits, the second module of the `fsm_llm.engine` layer). The old path does not import, with no alias and no shim. `MessagePipeline` is defined in `fsm_llm.engine.turn.message_pipeline`: `from fsm_llm.pipeline import MessagePipeline` becomes `from fsm_llm.engine.turn.message_pipeline import MessagePipeline` (or `from fsm_llm.engine.turn import MessagePipeline`, the package re-exports it; it was never exported from `fsm_llm` itself). The private helpers come from the module that defines them: `_TurnSnapshot`, `_TurnRecord`, `_NOT_CAPTURED`, `_record_transition_classification`, `_is_silent` and `_response_temperature` from `fsm_llm.engine.turn.records`, and, since the conversation executor owns the conversation Pass 1 (plan d454fd62, step 80), `_BulkFailed` from `fsm_llm.engine.kinds.conversation.bulk`, `_coerce_bool`, `_coerce_str`, `_coerce_list`, `_coerce_dict`, `_number_for_range` and `_TYPE_COERCERS` from `fsm_llm.engine.kinds.conversation.fields`, and `_CLASSIFICATION_SOFT_FAIL_EXCEPTIONS`, `_record_provenance`, `_record_classification_result`, `_classification_record`, `_is_agent_managed`, `_json_native_values`, `_drop_forbidden_entry` and `_value_digest` from `fsm_llm.engine.kinds.records` (the other helpers of that list stay in `fsm_llm.engine.turn.records`: `_TurnSnapshot`, `_TurnRecord`, `_NOT_CAPTURED`, `_record_transition_classification`, `_is_silent`, `_response_temperature`). Names the old module only imported (`Classifier`, `logger`, `copy`, `threading`, `FSMError`, ...) were never part of its API, and a patch aims at the module that performs the lookup, for example `fsm_llm.engine.turn.transitions.Classifier` (read by `_get_classifier`; was `fsm_llm.pipeline.Classifier`). `MessagePipeline` is composed of four mixins over the shell in `message_pipeline.py` (`__init__`, `get_state`, `execute_handlers`): `DriverMixin` (`driver.py`: `process`, `advance`, the stream variants), `Pass1Mixin` (`pass1.py`: the one Pass-1 seam and the executor dispatch), `TransitionsMixin` (`transitions.py`) and `Pass2Mixin` (`pass2.py`); the conversation kind's Pass-1 machinery (the private methods that `MessagePipeline` carried in 0.12.0) is no longer a method of the pipeline, see the Changed entry on `fsm_llm.engine.kinds`; every method body is the old one, moved (the same signatures, `tests/fixtures/api_snapshot.txt` is byte-equal), and the new private `_PipelineSurface` base only declares, for mypy, the attributes and methods the mixins share. The loguru record name of the pipeline reads `fsm_llm.engine.turn.<module>` (`driver`, `pass1`, `transitions`, `pass2`, `message_pipeline`) instead of `fsm_llm.pipeline`, and that of the conversation extraction `fsm_llm.engine.kinds.conversation.<module>` (`extraction`, `bulk`, `fields`, `classification`, `post_transition`).
- The conversation kind's Pass 1 belongs to the conversation executor (plan d454fd62, step 80, the sixteenth of the 0.4 module splits, D-004 stage 2 and D-030). The flat module `fsm_llm.executors` is the package `fsm_llm.engine.kinds` (`base`, `registry`, `completion`, `services`, `records`, `conversation/`); it was added in this release and never in a tagged version, so its move has no BREAKING entry, and the names `StateExecutor`, `ConversationExecutor`, `CompletionExecutor`, `executor_for`, `register_executor` and `check_executors` import from `fsm_llm.engine.kinds` (a test that patches the module table patches `fsm_llm.engine.kinds.registry._EXECUTORS`). The private methods that `fsm_llm.pipeline.MessagePipeline` carried in 0.12.0 for the conversation kind (`_extract_after_transition`, `_execute_data_extraction`, `_build_field_configs_from_state`, `_bulk_extract_from_instructions`, `_execute_field_extractions`, `_validate_field_extraction`, `_execute_classification_extractions`, `_handler_only_keys`; the post-transition block was inline then) are methods of `fsm_llm.engine.kinds.conversation.executor.ConversationPass1`, which `ConversationExecutor` builds over the pipeline's services for each call; their bodies are the old ones (a service of the pipeline is read as `self._services.<name>` through the private `_PipelineServices` protocol, so the executors never import the turn pipeline). `engine/turn/` holds no conversation Pass-1 method and reads no extraction config. Behaviour, request bytes and the equality corpus are unchanged (a 2,000-program differential over a recording model); only the loguru record names of the moved code change (`fsm_llm.executors` became `fsm_llm.engine.kinds.completion`). `tests/fixtures/api_snapshot.txt` is byte-equal.
- **BREAKING:** the module `fsm_llm.api` (present in 0.12.0) is split into the package `fsm_llm.engine.api` and no longer exists (plan d454fd62, step 81, the seventeenth of the 0.4 module splits, the fourth package of the `fsm_llm.engine` layer). The old path does not import, with no alias and no shim. `API` is defined in `fsm_llm.engine.api.conversation`: `from fsm_llm.api import API` becomes `from fsm_llm.engine.api.conversation import API` (or `from fsm_llm.engine.api import API`, the package re-exports it), and `from fsm_llm import API` is unchanged. `FSMStackFrame` and `ContextMergeStrategy` come from `fsm_llm.engine.api.stack` (`from fsm_llm import ContextMergeStrategy` is unchanged), `handle_conversation_errors` and `llm_settings_for` from `fsm_llm.engine.api.helpers`, `MAX_ENDED_CONVERSATIONS_CACHE` from `fsm_llm.engine.api.lifecycle`; the package `__init__` re-exports all six with a static `__all__`. The private helpers come from the module that defines them: `_unexpected_error`, `_read_or_ended`, `_require_classifier_interface` and `_ReadT` from `fsm_llm.engine.api.helpers`, `_turn_errors` and `_auto_save_session` from `fsm_llm.engine.api.conversation`, and `_check_run_budgets`, `_seconds_exempt_set`, `_run_rounds`, `_stack_depth`, `_hook_popped_ended_frame`, `_check_exempt_states` and `_closed_conversation_ends_run` from `fsm_llm.engine.api.run`. Names the old module only imported (`FSMManager`, `LiteLLMInterface`, `time`, `logger`, `threading`, `FSMError`, ...) were never part of its API: import them from where they are defined, and aim a patch at the module that performs the lookup, for example `fsm_llm.engine.api.conversation.LiteLLMInterface` (read by `API.__init__`; was `fsm_llm.api.LiteLLMInterface`) or the `time` binding of `fsm_llm.engine.api.run` (the seconds budget), `.stack`, `.lifecycle` and `.conversation` (the idle clock; was one `fsm_llm.api.time`). `API` is composed of four mixins over the shell in `conversation.py` (the constructor, the factories, `start_conversation`, `converse`, `advance` and the streams, handler registration, `close`): `RunMixin` (`run.py`: `run_until_terminal` and its stream, the budgets, the `before_step` rounds), `StackMixin` (`stack.py`: `push_fsm`, `pop_fsm`, the context merge, the top-of-stack lookup), `LifecycleMixin` (`lifecycle.py`: the ended-conversation cache, the getters, `end_conversation`, the idle sweep) and `SessionsMixin` (`sessions.py`: `save_session`, `load_session`, `restore_session`); every method body is the old one, moved (the same signatures, `tests/fixtures/api_snapshot.txt` is byte-equal; the one annotation change is the `api` parameter of five private helpers, typed by the new private mypy-only `_ApiSurface` so a mixin can pass `self`), and the lock order (`FSMManager._lock` before `conv_lock`, `API._stack_lock` never held across an `FSMManager` call) is unchanged. The loguru record name of the API reads `fsm_llm.engine.api.<module>` (`conversation`, `run`, `stack`, `lifecycle`, `sessions`, `helpers`) instead of `fsm_llm.api`.
- **BREAKING:** the module `fsm_llm.classification` (present in 0.12.0) moved to `fsm_llm.engine.classification` and the old path no longer exists (plan d454fd62, step 82, the eighteenth of the 0.4 module splits, a pure move of one file into the `fsm_llm.engine` layer). The old path does not import, with no alias and no shim: `from fsm_llm.classification import Classifier, HierarchicalClassifier, IntentRouter, HandlerFn` becomes `from fsm_llm.engine.classification import Classifier, HierarchicalClassifier, HandlerFn` and `from fsm_llm.engine.intent_router import IntentRouter` (`IntentRouter` has its own module since plan 51fde95b, step 6.2, in the same unreleased window), and `from fsm_llm import Classifier, HierarchicalClassifier, IntentRouter, HandlerFn` is unchanged. Every definition body of the move is the old one (only the import lines of the file changed: relative to absolute); plan 51fde95b then rewrote `classify`, `classify_multi`, `_call_llm` and `HierarchicalClassifier.classify` as `run_sync` wrappers over new async bodies. Names the old module only imported (`Callable`, `time`, `logger`, `LiteLLMInterface`, ...) were never part of its API, and a patch aims at the module that performs the lookup, for example `fsm_llm.engine.classification.<name>`. The loguru record name of the classifier reads `fsm_llm.engine.classification` instead of `fsm_llm.classification` (and `fsm_llm.engine.intent_router` for `IntentRouter`); the `package` extra of the one "Classifier initialized" line still reads `fsm_llm.classification` (a log field, left as it was).
- **BREAKING:** the module `fsm_llm.session` (present in 0.12.0) moved to `fsm_llm.engine.session` and the old path no longer exists (plan d454fd62, step 82, a pure move). The old path does not import, with no alias and no shim: `from fsm_llm.session import SessionState, SessionStore, FileSessionStore` becomes `from fsm_llm.engine.session import ...` (also `SESSION_ID_RE`), and `from fsm_llm import FileSessionStore, SessionState, SessionStore` is unchanged. Every definition body is the old one. A patch aims at the module that performs the lookup, for example `fsm_llm.engine.session.os.replace` (was `fsm_llm.session.os.replace`). The loguru record name of the store reads `fsm_llm.engine.session` instead of `fsm_llm.session`.
- **BREAKING:** the module `fsm_llm.builders` (present in 0.12.0) moved to `fsm_llm.engine.builders` and the old path no longer exists (plan d454fd62, step 82, a pure move; the module now sits on top of the engine layer, above `fsm_llm.engine.api` and `fsm_llm.engine.fsm`, which it builds and which never import it). The old path does not import, with no alias and no shim: `from fsm_llm.builders import APIBuilder, FSMManagerBuilder` becomes `from fsm_llm.engine.builders import ...` (also the unexported `refuse_named_options`, which `fsm_llm.agents.builders` imports), and `from fsm_llm import APIBuilder, FSMManagerBuilder` is unchanged. Every definition body is the old one. The loguru record name reads `fsm_llm.engine.builders` instead of `fsm_llm.builders` (the module logs nothing today). `fsm_llm.__init__` still imports no subpackage, `tests/fixtures/api_snapshot.txt` is byte-equal, and the import contracts name the new modules (a new contract, `engine-classification-and-session-are-leaves`, holds the classifier and the session store free of every other engine module).
- Open Knowledge Format is a layer package and its text helpers are a kernel module (plan d454fd62, step 83, the nineteenth of the 0.4 module splits, D-014, D-032 and D-033). The flat `fsm_llm.okf` (about 1,350 lines) is the package `fsm_llm.okf` at the same dotted path: `models`, `parser`, `links`, `bundle`, `loader`, `checker`, `writer`, `reading` (`body_page`, `close_ids`) and `cli`, no module over 800 lines, and its `__init__` keeps every name the module had in a static `__all__` (`from fsm_llm.okf import load_bundle` and every other public name still work; a private name such as `_link_candidates` or `_render_concept` is imported from its module, `fsm_llm.okf.links`, `fsm_llm.okf.writer`). `fsm_llm.okf_cli` became `fsm_llm.okf.cli` (the console script is `fsm-llm-okf = "fsm_llm.okf.cli:main_cli"`; after pulling this change run `pip install -e .` once to refresh the console script of an editable install) and `fsm_llm.okf_bridge` became `fsm_llm.engine.okf_bridge` (it needs `handlers`, an engine module). Neither old path was ever in a tagged version (OKF was added in this release), so there is no BREAKING entry for them. Six helpers moved to the new kernel module `fsm_llm.text` (standard library and `constants` only, no OKF model): `strip_unsafe_text`, `capped_line`, `capped_issues`, `capped_note`, `text_page` and `iso_utc`; an AST scan showed they read nothing okf-specific, while `has_lone_surrogate` (raises `OKFError`) and `writes_within_caps` (renders a concept) stayed in `okf/writer.py`. `fsm_llm.okf` re-exports all six, so the documented names are unchanged, and the modules that only need them (`cli_support`, `agent_skills_cli`, `agents/okf.py`, `engine/okf_bridge.py`, `okf/cli.py`) import them from `fsm_llm.text`. `cli_support` therefore no longer imports `okf`: the named import debt `core.visualize.cli` to `cli_support` is retired and `.importlinter` gained contracts for the `okf` package, for the kernel leaves `text` and `cli_support`, and for the isolation of `engine/okf_bridge.py` in both directions (the one open debt is `okf.cli` to `engine.okf_bridge`, retired by step 95). Behaviour, bytes written and read, issue texts, the product string `fsm-llm/<version>` and every cap are unchanged (a differential over generated bundles, the memory export, the handler and the five CLI commands, byte for byte); only the loguru record names of the moved code change (`fsm_llm.okf` became `fsm_llm.okf.parser` and `fsm_llm.okf.loader`). `tests/fixtures/api_snapshot.txt` changes only the module name of the two bridge functions (`fsm_llm.engine.okf_bridge`).
- Agent Skills is a layer package (plan d454fd62, step 84, the twentieth of the 0.4 module splits, D-034). The flat `fsm_llm.agent_skills` (704 lines) became `fsm_llm.skills.catalog` and `fsm_llm.agent_skills_cli` became `fsm_llm.skills.cli`; the package `__init__` is the one import surface with a static `__all__` of the nine names the module exported (`from fsm_llm.skills import load_skills, SkillCatalog` and the rest; a private name such as `_MESSAGES` or `_confined` is imported from `fsm_llm.skills.catalog`). The module stays whole (under 800 lines, one reader, one issue table and one "nothing is executed" anchor). The console script is `fsm-llm-skills = "fsm_llm.skills.cli:main_cli"` (after pulling this change run `pip install -e .` once to refresh the console script of an editable install). Neither old path was ever in a tagged version (Agent Skills was added in this release), so there is no BREAKING entry for them. The package imports the kernel (`constants`, `frontmatter`, `security`, `text`, `cli_support`) and `core` only: it no longer imports `okf` (the five text helpers it used come from `fsm_llm.text` now), and `.importlinter` holds that with a new contract (no debt); `fsm_llm.agents.agent_skills` (the agents tools) stays at its path and imports `fsm_llm.skills`. Behaviour, issue texts, caps and every byte of every reply and CLI output are unchanged (a differential over generated skill folders and the CLI, byte for byte); only the `__module__` of the eight public classes and functions changes (`fsm_llm.skills.catalog`; the moved code logs nothing, so no loguru record name changes). The two bench-pinned skills files had their import lines edited, `scripts/skills_bench.py` `SRC_PINS` names the new path of the first and keeps its manifest key, and `scripts/bench_data/` is untouched; the `skills-disclosure` manifests already differed from source, so a run refuses by design as before. `tests/fixtures/api_snapshot.txt` changes only in the module name of the nine public names.
- The tool layer is a layer package (plan d454fd62, step 85, the twenty-first of the 0.4 module splits, D-035 and D-036). `ToolRegistry`, the tool models (`ToolDefinition`, `ToolCall`, `ToolResult`, `ToolAnnotations`, `READ_ONLY_ANNOTATIONS`), the `@tool` decorator, `registry_from`, the schema inference, `redact_secret_entries`, the per-call timeout (`ToolTimedOut`, `check_timeout`, `call_with_timeout`), `ToolExecutionError`, `ToolNotFoundError`, `require_int_limit`, `bounded_repr` and `smart_truncate` moved down from `fsm_llm.agents` into the new package `fsm_llm.tools` (12 modules: `models`, `registry`, `decorator`, `schema`, `arguments`, `hints`, `timeouts`, `errors`, `limits`, `truncation`, `constants` and the facade `__init__`, whose static `__all__` is the one import surface); the 1,399-line `agents/tools.py` is 144 lines (`register_agent`, `normalize_tool_input`, `tool_call_from_context`, `refuse_execute_without_gated`: what only an agent run needs). `fsm_llm.agents` re-exports every name it exported before, so `from fsm_llm.agents import ToolRegistry, ToolDefinition, tool` and `from fsm_llm.agents.tools import register_agent` are unchanged, and `tests/fixtures/api_snapshot.txt` is byte-identical. `AgentError` moved down into `fsm_llm.core.definitions.errors` (beside `OKFError` and `SkillError`) so the tool errors can subclass it without importing `agents`: `except AgentError` still catches `ToolExecutionError` and `ToolNotFoundError`, and `fsm_llm.agents.AgentError` and `fsm_llm.core.definitions.AgentError` are one class. The tool-layer subset of the agents constants (`ContextKeys.NO_TOOL`, `TOOL_NAME`, `TOOL_INPUT`, four `Defaults`, four `ErrorMessages`, `LogMessages.TOOL_LATE_RESULT`) is defined in `fsm_llm.tools.constants` and assigned from there by the agents classes, each in its old position: `vars()` of `ContextKeys`, `Defaults`, `ErrorMessages` and `LogMessages` is unchanged and each value has one source at import; the classes that share a name are linked (the metaclass `MirroredConstants`, D-045), so setting or deleting a shared name on ANY of them (the agents class, the tool class or the policy class) reaches every reader of every class, as when one class held it (`Defaults.MAX_OBSERVATION_LENGTH = 5000` on `fsm_llm.agents.constants.Defaults` lifts the cut of `ToolResult.observation` too); `ToolRunStatus` and `ToolObservationPrefix` moved there whole and are still exported by `fsm_llm.agents.constants`. A new import contract holds the layer (`fsm_llm.tools` imports no agents, engine, llm, handlers, okf, skills, context or subpackage module; one named debt, `tools.arguments` to `context.filtering.redact_context`, retired by step 95), and the layer contracts of `core`, `llm`, `context`, `okf` and `skills` forbid it. Behaviour, messages, schemas, timeouts and prompts are unchanged (a differential over 3,250 generated programs, old tree against new, found no difference); what changes is observable only through `__module__` (`fsm_llm.tools.registry`, `.models`, `.errors`, ...) and the loguru record `name` of the registry (`fsm_llm.tools.registry`, `.arguments`, `.schema`, `.timeouts` and `.models`, was `fsm_llm.agents.tools` and `fsm_llm.agents.definitions`). The remaining `synthesis` to `agents` imports are `agents.constants` (`OBSERVATION_ENTRY`, `Defaults`), `agents.hitl` (retired by step 86) and `agents.semantic_tools`; the six imports of `agents.definitions`, `agents.exceptions` and `agents.tools` now read `fsm_llm.tools`. Eighteen bench-pinned agents files (and `agents/agent_skills.py`, pinned by the skills bench), `synthesis/{library,policy,synthesizer}.py`, `scripts/bench_common.py` and `scripts/synthesis_bench.py` (`SRC_PINS` now names the twelve `tools/` modules and drops `agents/truncation.py`) were edited (import lines, and the moved definitions leaving `agents/{tools,definitions,exceptions}.py` and `agents/constants/{core,outcomes}.py`); every synthesis manifest already differed from source, so no registered block is newly voided, and `scripts/bench_data/` is untouched.
- The policy layer is a layer package (plan d454fd62, step 86, the twenty-second of the 0.4 module splits, D-037). `HumanInTheLoop`, the approval vocabulary (`ApprovalRequest`, `ApprovalPolicy`, `ApprovalCallback`, `EscalationCallback`) and `ApprovalDeniedError` moved down from `fsm_llm.agents` into the new package `fsm_llm.policy` (`approval`, `hitl`, `constants` and the facade `__init__`, whose static `__all__` of six names is the one import surface); the 211-line `agents/hitl.py` is gone. `fsm_llm.agents` re-exports all six names, so `from fsm_llm.agents import HumanInTheLoop, ApprovalRequest, ApprovalPolicy, ApprovalDeniedError` is unchanged, `fsm_llm.synthesis` imports `HumanInTheLoop` from `fsm_llm.policy` (the last `synthesis` to `agents.hitl` edge: three `synthesis` to `agents` imports remain, `agents.constants` once and `agents.semantic_tools` twice), and `tests/fixtures/api_snapshot.txt` is byte-identical. `make_hitl_checker`, the `CONTEXT_UPDATE` gate handler that writes `approval_required`, is agent glue (it reads the run's `tool_name` and writes an agents key) and moved to `fsm_llm.agents.tools` beside `tool_call_from_context`; the driver-only `_approval_granted` grant and its checks stay in `agents/base.py` and `agents/handlers.py`, untouched. The HITL subset of the agents constants (`Defaults.CONFIDENCE_THRESHOLD`, `LogMessages.APPROVAL_REQUESTED`, `APPROVAL_RESULT` and `ESCALATION`) is defined in `fsm_llm.policy.constants` and assigned from there by the agents classes, each in its old position: `vars()` of the agents classes is unchanged and the shared names are linked as above, so an override of the agents or the policy class reaches every reader. `fsm_llm.policy` imports the kernel, `core` (`AgentError`) and `fsm_llm.tools`, never `agents`, the engine, `llm` or a subpackage: it sits above `tools`, a deviation from the roadmap Section 5.2 row that lists the two as siblings (step 95 amends the `layers` contract); a new import contract holds it (with the `tools` debt repeated, because it is reached through `tools`), and the contracts of `core`, `llm`, `context`, `okf`, `skills` and `tools` forbid `fsm_llm.policy`. Behaviour, messages, timeouts, redaction and every approval decision are unchanged (a differential over 1,800 generated programs, old tree against new, found no difference); what changes is observable only through `__module__` (`fsm_llm.policy.hitl`, `fsm_llm.policy.approval`) and the loguru record `name` of the manager (`fsm_llm.policy.hitl`, was `fsm_llm.agents.hitl`). Bench pins: `agents/hitl.py` leaves `scripts/synthesis_bench.py` `SRC_PINS` and the four `policy/` modules join it (`_POLICY`); `agents/{base,handlers,react,reflexion,reasoning_react,tools,definitions,exceptions,__init__}.py`, `agents/constants/core.py` and `synthesis/synthesizer.py` were edited (import lines, and the moved definitions leaving them); every synthesis manifest already differed from source, so no registered block is newly voided, and `scripts/bench_data/` is untouched.
- `fsm_llm.agents.base` is a package at the same dotted path (plan d454fd62, step 87, the twenty-third of the 0.4 module splits, D-039). The 1,842-line module, `BaseAgent` and twelve module-level helpers, is eight modules, none over 800 lines: `agent` (`BaseAgent`, the shell, and `accepts_tools`), `hitl_gate` (`HitlGateMixin`: the one HITL predicate, the gate, the refusals of flagged tools, the call-local `AgentHandlers` and the driver that writes the driver-only `_approval_granted` grant), `outcome` (`OutcomeMixin`: answer extraction, with `_answer_source_keys` beside `_extract_answer`, the cut-off verdict, the evidence rule, the trace, the context filter and the structured-output parse), `runs` (`RunsMixin`: the conversation loop on core's bounded run, the budgets, the API factory and the standard run and its stream form), `registration` (`RegistrationMixin`: the iteration limiter, the tool executor, the think-loop and lifecycle handlers), `helpers` (the module-level functions), `surface` (`_AgentSurface`, the declarations the mixins share, for mypy only) and the facade `__init__`, whose static `__all__` names the eight public names the module had (`BaseAgent`, `accepts_tools`, `artifact_text`, `caller_prompt_keys`, `flagged_tool_names`, `pattern_run_output_keys`, `reject_misplaced_kwargs`, `strip_caller_context`). Every import of a public name is unchanged (`from fsm_llm.agents.base import BaseAgent`), so there is no BREAKING entry, and `tests/fixtures/api_snapshot.txt` is byte-identical. What does change: a private name is imported from the module that defines it (`_output_response_format`, `_with_instructions`, `_prompt_overflow_error`, `_flagged_tool_policy`, `_HITL_CTOR_KWARGS` and `_INSTRUCTION_SLOTS` from `fsm_llm.agents.base.helpers`); the names the old module only imported (`API`, `AgentHandlers`, `time`, `logger`, `ContextCompactor`, ...) are no longer attributes of `fsm_llm.agents.base`, so a patch aims at the module that performs the lookup (`fsm_llm.agents.base.runs.API.from_definition`, `fsm_llm.agents.base.hitl_gate.AgentHandlers`; the run clock `time` is read in `fsm_llm.agents.base.runs` only); `BaseAgent.__module__` is `fsm_llm.agents.base.agent`; and a loguru record of the base is named after its module (`fsm_llm.agents.base.runs`, `.helpers`, `.outcome`, `.agent`, `.registration`, `.hitl_gate`), not `fsm_llm.agents.base`. `BaseAgent` is `BaseAgent(HitlGateMixin, OutcomeMixin, RunsMixin, RegistrationMixin, ABC)`; the mixins are `_AgentSurface` subclasses, a name resolves on every one of the 17 `BaseAgent` subclasses to the same body as before (a comparison of 987 attribute resolutions found no difference), and the `super()` calls and private-method calls of the subclasses are unchanged. `_run_outcome` and `accepts_tools` stay in `agent.py` because they name `BaseAgent` itself. Method and function bodies are the old ones (compared per definition with `ast.dump`); the HITL security boundary (the `_approval_granted` grant, `TestOnlyTheValueTrueApproves`, the refusal of caller `initial_context` run-output keys) and the `AgentConfig(model=...)` injected-interface rule are untouched. Behaviour is unchanged: a differential over 3,000 generated programs on all 18 patterns (`run`, `run_stream`, tools, approval callbacks that approve, deny, raise or answer truthy, forged `initial_context`, cut replies, outages, spent budgets), old tree against new, compared 422,760 log records and every result, error, request and approval and found no difference. Two new import contracts hold the package (no agent pattern, meta-builder or transport module imports from `fsm_llm.agents.base`, and the mixins, helpers and surface never import the shell). Bench pins: `base.py` leaves `scripts/synthesis_bench.py` `SRC_PINS` and the eight modules join it (`_AGENTS`); every synthesis manifest already differed from source, so no registered block is newly voided, and `scripts/bench_data/` is untouched. Tests added where the mutation proof of the split found none: the reader of core's cut-off record, the priorities of the limiter and tool-executor registration, and the clock binding of the agents' run budget (the old binding of `fsm_llm.agents.base.time` was inert: no test failed without it).
- `fsm_llm.agents.handlers` is a package at the same dotted path (plan d454fd62, step 88, the twenty-fourth of the 0.4 module splits, D-040). The 967-line module (nineteen public names, one private table) is eight modules, none over 380 lines: `executor` (`AgentHandlers`: the ReAct-family tool executor and the HITL approval refusal, `approval_refusal`, `spend_grant` and `consume_approval`), `limiters` (`ThinkTurnLimiter` and `make_iteration_limiter`), `approval` (the driver-grant helpers `approval_grant`, `call_label`, `call_ran`, `refusal_record`), `forced` (`forced_stop_skip`, `recorded_forced_reason`, `concluded_on_evidence`, `is_forced_verdict`), `recording` (`record_tool_step`, `next_step_number`, `with_feedback`), `freshness` (`make_redraft_handlers`, `make_fresh_keys_handler`), `run_guards` (`run_evaluation_fn`, `RUN_ENDING_ERRORS`, `RunEndingErrorHolder`) and the facade `__init__`, whose static `__all__` names the nineteen public names. Every import of a public name is unchanged (`from fsm_llm.agents.handlers import AgentHandlers`; the agents patterns import them as before), so there is no BREAKING entry, and `tests/fixtures/api_snapshot.txt` is byte-identical. What does change: the private `_CONCLUDE_VERDICTS` (importable from the old module in 0.12.0) is imported from `fsm_llm.agents.handlers.forced`; the names the old module only imported (`logger`, `clear_keys_delta`, `ToolCall`, `ToolRegistry`, ...) are no longer attributes of `fsm_llm.agents.handlers`, so a patch aims at the module that performs the lookup (`fsm_llm.agents.handlers.executor.logger`); `AgentHandlers.__module__` is `fsm_llm.agents.handlers.executor` (and so on per name); and a loguru record of the handlers is named after its module (`fsm_llm.agents.handlers.executor`, `.limiters`, `.run_guards`, `.forced`, `.recording`, `.freshness`), not `fsm_llm.agents.handlers`. Function and class bodies are the old ones (compared per definition with `ast.dump`); the HITL security boundary (the driver-only `_approval_granted` grant, the refusal of a gated call without it, the spent-grant record that survives a discarded delta, caller `initial_context` run-output keys) is untouched. Behaviour is unchanged: a differential over 3,000 generated programs, 2,174 across all 18 agent patterns (`run`, `run_stream`, tools, approval callbacks that approve, deny, raise or answer truthy, forged `initial_context`, cut replies, outages, spent budgets) and 826 that drive the handlers directly (the approval refusal, grant spend and consume, the limiters, `record_tool_step`, the freshness handlers, the evaluation guard, the run-ending holder, over 4,649 steps with random and hostile contexts), old tree against new, compared 317,461 log records and every result, error, request, tool execution and approval request and found no difference. Two new import contracts hold the package (no agent pattern, meta-builder, transport or `agents.base` module is imported from `fsm_llm.agents.handlers`, and its helper modules never import the executor). Bench pins: `handlers.py` leaves `scripts/synthesis_bench.py` `SRC_PINS` and the eight modules join it (`_AGENTS`); every synthesis manifest already differed from source, so no registered block is newly voided, and `scripts/bench_data/` is untouched. Tests added where the mutation proof of the split found none: the strict-`True` reads of `forced_stop_skip` and `concluded_on_evidence`, the observation-count floor, the verdict key of each state, the text form of the grant's tool name, `call_ran` on a non-list trace, and the limiter's rule for a recorded forced reason (six old-suite gaps, each seen only by the differential before); and the layout test of the package.
- `fsm_llm.agents.meta_builders` is a package at the same dotted path (plan d454fd62, step 89, the twenty-fifth of the 0.4 module splits, D-041). The 1,158-line module (the abstract `ArtifactBuilder` and the three artifact builders) is one module per builder class, none over 420 lines: `base` (`ArtifactBuilder`, the ABC with the lazy per-instance `_warnings` property, `take_warnings`, `build`, `get_progress` and the shared summary helpers), `fsm` (`FSMArtifactBuilder`), `workflow` (`WorkflowArtifactBuilder`), `agent` (`AgentArtifactBuilder`, with `_config_problems`, the `MetaDefaults.AGENT_MODEL` template read and the anchor that says not to switch that value, so the sentence and the value it protects stay in one module) and the facade `__init__`, whose static `__all__` names the four classes. Every import of a public name is unchanged (`from fsm_llm.agents.meta_builders import FSMArtifactBuilder`; `agents/__init__.py`, `meta_builder.py`, `meta_prompts.py` and `meta_tools.py` import them as before), so there is no BREAKING entry, and `tests/fixtures/api_snapshot.txt` is byte-identical. What does change: the names the old module only imported (`logger`, `MetaDefaults`, `BuildError`, `FSMDefinition`, `ModelNotConfiguredError`, `reachable_from`, `AgentConfig`, `PLACEHOLDER_MODEL`, `copy`, ...) are no longer attributes of `fsm_llm.agents.meta_builders`, so a patch aims at the module that performs the lookup (`fsm_llm.agents.meta_builders.agent.AgentConfig`); a class's `__module__` is its own module (`fsm_llm.agents.meta_builders.fsm`, ...), so a builder pickled from now on names that module (a pickle written before this change still loads, the facade keeps the class names); and the loguru record of `FSMArtifactBuilder` (the only builder that logs) is named `fsm_llm.agents.meta_builders.fsm`, not `fsm_llm.agents.meta_builders`. Class bodies are the old ones (compared per definition with `ast.dump`). Behaviour is unchanged: a differential over 5,000 generated programs, old tree against new (every builder class on random and hostile calls with and without `LLM_MODEL`, the three tool factories through the real registry, and `MetaBuilderAgent` runs and interactive sessions through the meta-builder FSM with scripted model replies), compared 149,816 log records and every result, error, artifact, summary, tool reply and request and found no difference. Two new import contracts hold the package (it imports no agent pattern, meta-builder peer, transport, `agents.base` or `agents.handlers` module, and no builder module imports another). Bench pins: `meta_builders.py` leaves `scripts/synthesis_bench.py` `SRC_PINS` and the five modules join it (`_AGENTS`); every synthesis manifest already differed from source, so no registered block is newly voided, and `scripts/bench_data/` is untouched. Tests added where the mutation proof of the split found none (nine edges, each seen only by the differential or by nothing before): `to_dict` returns copies of the top-level containers (`states`, `steps`, `tools`, `config`), `get_progress` never reports a negative `completed` count when more fields are missing than the total, the minimal summary counts edges, `update_state` strips, a graph with no terminal state is reported missing one, `remove_transition` removes only the named target, the standard workflow summary cuts a step name, every agent-type alias resolves to its pattern, and a config with no model still reports its other bad keys without the pydantic prefix; and the layout test of the package.
- `fsm_llm.agents.meta_builder` is a package at the same dotted path (plan d454fd62, step 90, the twenty-sixth of the 0.4 module splits). The 1,101-line module (`MetaBuilderAgent`, the agent behind `fsm-llm-meta`, and its helpers) is eleven modules, none over 220 lines: `agent` (`MetaBuilderAgent`, the shell: construction and the session attributes, composed of six mixins), `hints` (`HintsMixin`: the keyword hints, pure functions of one message, with `_normalize_message`, the patterns and the alias table `_TYPE_ALIASES`), `handlers` (`HandlersMixin`: `_create_api` and the four handlers that write the build request and judge the build reply), `assembly` (`AssemblyMixin`: one build reply assembled on a fresh builder, `_create_builder`), `runs` (`RunMixin`: `run` and the `build` step), `session` (`SessionMixin`: `start`, `send`, one user turn, the build turn, the canned fallbacks, and `_with_build_prompt`), `results` (`ResultsMixin`: `is_complete`, `get_result`, `get_internal_state`, `run_interactive`, `_result_from`), `specs` (the typed specs of a build reply), `names` (the three aliases `_S`, `_K`, `_TYPE_VALUES`), `surface` (`_MetaBuilderSurface`, the declarations the mixins share, for mypy only) and the facade `__init__`, whose static `__all__` names the two public names the module had (`MetaBuilderAgent`, `MetaBuilderConfig`). Every import of a public name is unchanged (`from fsm_llm.agents.meta_builder import MetaBuilderAgent, MetaBuilderConfig`; `agents/__init__.py`, `meta_cli.py` and the monitor import them as before, and `fsm-llm-meta` is unchanged), so there is no BREAKING entry, and `tests/fixtures/api_snapshot.txt` is byte-identical. What does change: a private name is imported from the module that defines it (`_normalize_message` and `_TYPE_ALIASES` from `fsm_llm.agents.meta_builder.hints`, `_with_build_prompt` from `fsm_llm.agents.meta_builder.session`, the reply specs and `_given` from `fsm_llm.agents.meta_builder.specs`); the names the old module only imported (`MetaBuilderResult`, `AgentConfig`, `API`, `logger`, `re`, `clear_keys_delta`, ...) are no longer attributes of `fsm_llm.agents.meta_builder` (`MetaBuilderResult` imports from `fsm_llm.agents`), so a patch aims at the module that performs the lookup (`fsm_llm.agents.meta_builder.handlers.API`); `MetaBuilderAgent.__module__` is `fsm_llm.agents.meta_builder.agent`; and a loguru record of the agent is named after its module instead of `fsm_llm.agents.meta_builder`: `fsm_llm.agents.meta_builder.handlers` (the classify and build entries), `.assembly` (the assembly warnings), `.runs` (the run start) and `.session` (a failed collect reply, a failed build turn). The `clear_keys_delta` import of the agent now names its defining package (`fsm_llm.context`; `fsm_llm.handlers` still re-binds it). Two import contracts hold the package (`agents-meta-builder-imports-no-pattern`: no agent pattern, transport or consumer of the agent; `agents-meta-builder-mixins-do-not-import-the-shell`). Behaviour, request bytes and results are unchanged (a differential over scripted models: runs, sessions, `run_interactive`, handlers and assembly on hostile replies, the keyword hints).
- `fsm_llm.workflows.engine` is a package at the same dotted path (plan d454fd62, step 91, the twenty-seventh of the 0.4 module splits). The 1,570-line module (`WorkflowEngine`, the async in-memory workflow engine, with `Timer` and a few helpers) is eleven modules, none over 402 lines: `engine` (`WorkflowEngine`, the shell: constructor and storage, composed of seven mixins), `status` (`StatusMixin`: the hooks, the per-instance lock registry, `_set_status`, `_fail`), `lifecycle` (`LifecycleMixin`: `close`, `register_workflow`, `start`, `advance`, `cancel`, instance creation), `driver` (`DriverMixin`: the step loop, the run of one step, result handling, the transitions into the next step or a wait), `events` (`EventsMixin`: `register_event_listener`, `process_event`, delivery, the event timeout, the buffers of early targeted events), `timers` (`TimersMixin`: `_arm_timer`, `schedule_timer`, timer expiry, the deadline watchdog), `resources` (`ResourcesMixin`: release of wait and instance resources, `remove_instance`, the purge of terminal instances), `queries` (`QueriesMixin`: the getters and `get_statistics`), `helpers` (`Timer`, `LifecycleHook`, `_CancelRequest`, `_timer_key`, `_timeout_error`), `surface` (`_EngineSurface`, the declarations the mixins share, for mypy only) and the facade `__init__`, whose static `__all__` names `DEFAULT_MAX_STEPS`, `LifecycleHook`, `Timer` and `WorkflowEngine`. Every import of a public name is unchanged (`from fsm_llm.workflows.engine import WorkflowEngine`, `from fsm_llm.workflows import Timer, WorkflowEngine`; the monitor and the workflow examples import them as before), so there is no BREAKING entry, and `tests/fixtures/api_snapshot.txt` is byte-identical. What does change: a private name is imported from the module that defines it (`_timer_key`, `_timeout_error` and `_CancelRequest` from `fsm_llm.workflows.engine.helpers`); the names the old module only imported (`asyncio`, `logger`, `datetime`, the constants, the models and the exceptions) are no longer attributes of `fsm_llm.workflows.engine`, so a patch aims at the module that performs the lookup (`fsm_llm.workflows.engine.timers.asyncio`, `fsm_llm.workflows.engine.driver.datetime`); `WorkflowEngine.__module__` is `fsm_llm.workflows.engine.engine`; and a loguru record of the engine is named after its module instead of `fsm_llm.workflows.engine`: `fsm_llm.workflows.engine.driver` (the step loop and its transitions), `.events`, `.timers`, `.lifecycle` (`start`, `register_workflow`, `cancel`), `.status` (a failed hook, a terminal instance that is not overwritten), `.resources` (purge, removal) and `.engine` (the constructor). Method bodies are the old ones (compared per definition with `ast.dump`, 6 units and 61 members). Behaviour, task creation, timer scheduling, lock and await order and event-listener order are unchanged: a differential over 10,000 generated programs, old tree against new, on a virtual-time event loop with a fake clock and an inline executor (workflows with all eleven step kinds and custom steps, conditions, retries, parallel branches, event waits and `process_event` with correlation, buffered and cancelled deliveries, timers and deadlines, concurrent cancel, advance and start, hooks, limits and purges, the whole state of every engine table and instance after every operation), compared 311,499 log records and the whole observable state and found no difference; the loop iteration counts match too. One import contract holds the package (`workflows-engine-mixins-do-not-import-the-shell`). Tests added where the mutation proof of the split found none (each of 30 edges was seen only by the differential, or by nothing): the exact concurrency and step limits, the hook emission rules, the guards under the lock of the timeout, timer and deadline handlers, the buffers of targeted events, the release of an instance's resources, the statistics, `Timer.cancel` on its own task and the facade, and the layout test of the package. The old suite pinned none of them.
- `fsm_llm.workflows.steps` is a package at the same dotted path (plan d454fd62, step 92, the twenty-eighth of the 0.4 module splits). The 1,208-line module (`WorkflowStep`, the 11 step classes and a few helpers) is eight modules and a facade, none over 291 lines, grouped by what a step does: `base` (`WorkflowStep`, the one base class, whose `model_config` carries `extra="forbid"`, so every kind inherits the refusal of an unknown or removed keyword from one definition), `helpers` (`_STEP_EXECUTOR`, `_is_async_callable`, `_call_user_callable`, `_error_text`), `routing` (`AutoStep`, `ConditionStep`, `SwitchStep`), `calls` (`APIStep`, `LLMStep`, `_lookup_path`), `waits` (`WaitEventStep`, `TimerStep`), `conversation` (`ConversationStep`), `agent` (`AgentStep`), `composite` (`ParallelStep`, `RetryStep`) and the facade `__init__`, whose static `__all__` names the 12 public classes. Every import of a public name is unchanged (`from fsm_llm.workflows.steps import AutoStep`, `from fsm_llm.workflows import AutoStep`; the `dsl` factories, `WorkflowDefinition` and the nine workflow examples import them as before), so there is no BREAKING entry, `tests/fixtures/api_snapshot.txt` is byte-identical, and the wire strings do not move: a step class keeps its name, so the `"type"` tag of `WorkflowDefinition.serialize()` is the same class name; only `__module__` changed (`fsm_llm.workflows.steps.routing.AutoStep`), and nothing serialized or persisted holds a module path. What does change: a private name is imported from the module that defines it (`_STEP_EXECUTOR`, the one `ContextVar` the engine driver sets and `ConversationStep` and `_call_user_callable` read, from `fsm_llm.workflows.steps.helpers`; it is defined once and never rebound); the names the old module only imported (`API`, `asyncio`, `logger`, `datetime`, the constants, the models, the exceptions) are no longer attributes of `fsm_llm.workflows.steps`, so a patch aims at the module that performs the lookup (`fsm_llm.workflows.steps.conversation.API`, which is where `ConversationStep` reads it; a patch aimed at the package now raises `AttributeError` instead of silently doing nothing); and a loguru record of a step is named after its module instead of `fsm_llm.workflows.steps`: `fsm_llm.workflows.steps.routing`, `.calls`, `.conversation`, `.agent` and `.composite` (`waits` and `base` log nothing). Method bodies are the old ones (compared per definition with `ast.dump`, 17 units and 101 members). Behaviour is unchanged: a differential over 5,000 generated programs, old tree against new, on a virtual-time event loop with a fake clock, an inline executor and a fake `API` (every step kind built through its class and through its `dsl` factory with valid and invalid keywords, unknown and removed keywords included, then validated, dumped, copied, deep-copied, pickled, round-tripped and serialized inside a definition; every kind executed directly and through the engine with sync, async, awaitable-returning and raising callables, retries, parallel children and conversations; the class fingerprints (fields, validators, signatures, docstrings), `_call_user_callable`, `_lookup_path` and `_error_text` on their own), compared the whole observable result of every program and every log record (module names mapped) and found no difference: the old and the new output files are byte-identical. One import contract holds the package (`workflows-steps-import-no-driver-definition-or-dsl`). Tests added where the mutation proof of the split found none (27 of 80 mutations survived the old suites): the helpers of `_call_user_callable` (a callable object with an async `__call__`, the step executor), the text of an empty error, the switch comparison of a missing value and its referenced ids, an exact result key against a dotted path, the omitted API input key, the regex flags, the default system prompt, the named missing template and task variables, the wait and timer reports, the smaller of the two conversation limits, the user-input switch, the model passed to the API, the executor of a conversation, an agent result with no `success`, the children of a parallel step (nested wait and timer refusal, the failure of a raising or garbage child, the copy warning threshold) and the retry bounds and backoff, and the layout test of the package.
- `fsm_llm.monitor.instance_manager` is a package at the same dotted path (plan d454fd62, step 93, the twenty-ninth of the 0.4 module splits). The 1,974-line module (`InstanceManager`, the three `Managed*` classes, the observer handlers, the workflow presets and the example-preset reads) is eleven modules and a facade, none over 410 lines: `manager` (`InstanceManager`, the shell: constructor, properties, the loguru sink, `attach_api`, `close`, the instance queries, the typed getters, the two event recorders), five mixins `snapshots` (`SnapshotsMixin`, with `snapshot_from_api`), `fsm` (`FsmMixin`), `workflow` (`WorkflowsMixin`), `agent` (`AgentsMixin`, with the launchable agent classes `_AGENT_CLASSES`) and `lifecycle` (`LifecycleMixin`: registering, ending, tearing down), `managed` (`ManagedFSM`, `ManagedWorkflow`, `ManagedAgent` and their private base, kept in ONE module because each `__init__` calls `super().__init__`), `handlers` (`_MonitorHandler`, `register_monitor_handlers`, `unregister_monitor_handlers`, `monitor_handlers_on`), `workflow_presets` (`_WORKFLOW_PRESETS`), `examples` (`find_examples_dir`, `validate_preset_id`, `load_preset_json`, `read_json_file`), `surface` (declarations for mypy only; 21, each proved needed by a mypy error when removed) and the facade `__init__`, whose static `__all__` names the 13 public names. Every import of a public name is unchanged (`from fsm_llm.monitor.instance_manager import InstanceManager`, `from fsm_llm.monitor import InstanceManager`; `monitor/server.py` imports four names from it as before), so there is no BREAKING entry, `tests/fixtures/api_snapshot.txt` is byte-identical, and the REST and WebSocket field names, status strings and event names do not move. One read is fixed by hand: `find_examples_dir` counts directories from the file that performs it, and the new `instance_manager/examples.py` is one level deeper than the old module, so `Path(__file__).resolve().parents[3]` became `parents[4]` (the repository root again; a wrong depth does not raise, it returns `None` and the dashboard lists no presets; in a wheel install the result is the same directory as before, `examples` next to `site-packages`, which does not exist, so the answer is still `None`). It has a test that fails on the moved code with the old depth (`TestFindExamplesDir`: the directory found is the checkout's non-empty `examples/`, a copy of the module loaded four levels below a scratch root finds that root's `examples/` and not a decoy one level up, a layout with no `examples/` returns `None` without raising). What does change: a private name is imported from the module that defines it (`_AGENT_CLASSES` from `instance_manager.agent`, `_MonitorHandler` and `_build_context_capture_handlers` from `instance_manager.handlers`, `_WORKFLOW_PRESETS` from `instance_manager.workflow_presets`); the names the old module only imported (`API`, `logger`, `asyncio`, `uuid`, `resolve_model`, the constants, the definitions) are no longer attributes of `fsm_llm.monitor.instance_manager`, so a patch aims at the module that performs the lookup (`fsm_llm.monitor.instance_manager.lifecycle.logger` for the agent-teardown warning; a patch aimed at the package now raises `AttributeError` instead of silently doing nothing); and a loguru record of the manager is named after its module instead of `fsm_llm.monitor.instance_manager`: `fsm_llm.monitor.instance_manager.fsm`, `.workflow`, `.snapshots` and `.lifecycle` (`manager`, `agent`, `handlers`, `examples`, `managed` and `workflow_presets` log nothing through the library logger), so the `module` field of a `LogRecord` the dashboard shows (`/api/logs`, the WebSocket `logs`) for those records is that file name (`lifecycle`, ...) and no longer `instance_manager`, and its `line` is the line in the new file. Method bodies are the old ones (compared per definition with `ast.dump`: 33 units and 100 members equal; the one body that changed is `find_examples_dir`, whose only difference is the integer in `parents[...]`). Behaviour is unchanged: a differential over 6,000 generated programs, old tree against new, in two processes (the old and the new output files are byte-identical) (instance creation of the three kinds with scripted LLM doubles and stub agent classes, every lifecycle operation on a manager and through the REST routes and the WebSocket, error and capacity paths, the example-preset listing and id validation, `snapshot_from_api` over hostile contexts, the observer handlers on a real `API`, a subclass of `InstanceManager` overriding four methods, and the fingerprint of every method and property of the manager and its subclass), compared the whole observable result of every operation, the manager's collectors and every log record (module names mapped) and found no difference. Two import contracts hold the package (`monitor-instance-manager-mixins-do-not-import-the-shell`, `monitor-instance-manager-imports-no-server`). Tests added (30): the layout test of the package (10), the examples-directory tests above (7), a test that every module that logs uses the library logger itself, and 12 tests for edges of the manager that no test pinned and that the mutation proof of the split reached (47 named mutations of the new package: 35 of the first 46 were caught by the suites at once; the 11 survivors were the capabilities report, the lexical `..` check of a preset id, the eviction order of finished instances, the model resolution of an FSM and of an agent launch, the terminal event of a cancelled workflow run, the agent capacity boundary, the `cancelling` status of a cancelled agent, the fail-closed default of `snapshot_from_api`, the `include_ended=False` filter of the activity list and a wrapped logger; 10 of them are now caught and so is every later mutation; one, the first `cancel_event.set()` of `end_instance`, is repeated by `_teardown` and is an equivalent mutant).
- `fsm_llm.monitor.server` is a package at the same dotted path (plan d454fd62, step 94, the thirtieth and last of the 0.4 module splits; the 800-line limit now holds for every module and the `OVERSIZED` table of the size guard is empty). The 1,677-line module (the FastAPI `app`, its middleware, `configure`, 53 route and mount entries, the `/ws` push and the builder sessions) is nine modules and a facade, none over 432 lines: `application` (the one `app`, its three middleware and the CORS middleware, `configure`, `get_manager`, `get_flows`, `_require_api_key`, the Host and Origin checks, the WebSocket authorisation `_ws_authorized`, `STATIC_DIR` and `TEMPLATE_DIR`, and the three routes `/`, `/health` and `/api/auth` that read that state or the template directory), `support` (`_raise_http`, `_bounded`, `_json_body`, the `_GATED` dependency list), `routes_monitoring`, `routes_fsm`, `routes_workflow`, `routes_agent`, `routes_visualize` (FSM visualization, presets and the `flows.json` pattern graphs), `routes_builder` (the builder sessions) and `websocket` (`/ws`), and the facade `__init__`, whose static `__all__` names `app`, `configure`, `get_manager`, `get_flows`, `STATIC_DIR` and `TEMPLATE_DIR`. Every route is still `@app.<method>(...)` on the one `app`, so every route body and decorator is the old one; the route modules register when the facade imports them, in the order the old module declared them (FastAPI matches in registration order and `app.openapi()` lists paths in it), which `server/__init__.py` fixes with `# isort: off` and `TestRegistrationOrder` pins (the 53 entries of `app.routes`, the four middleware, the OpenAPI path order). Every import of a public name is unchanged (`from fsm_llm.monitor import app, configure`, `from fsm_llm.monitor.server import app, configure, get_manager`, `uvicorn fsm_llm.monitor.server:app`, `fsm-llm-monitor` and `python -m fsm_llm.monitor`), so the only BREAKING entry is the one for the route functions that follows this bullet, `tests/fixtures/api_snapshot.txt` is byte-identical, and the REST and WebSocket paths, field names, status strings and event names do not move. Two reads are fixed by hand, and they are the only bodies that changed: `STATIC_DIR` and `TEMPLATE_DIR` count directories from the file that defines them, and `application.py` is one level deeper than the old module, so `Path(__file__).parent / "static"` became `Path(__file__).parent.parent / "static"` (and likewise `templates`); a wrong depth is loud for the static mount (`StaticFiles` refuses a missing directory at import) but SILENT for the template (a 500 on `/`) and for `flows.json` (empty pattern graphs). `TestStaticAndTemplateDirectories` resolves what the server really uses (the mounted directory, the Jinja search path, the constants, the served bytes of `/` and of four static files, the flows) and fails on the moved code with the old depth. Module-level state lives in exactly one module each: `application` holds `_manager`, `_api_key`, `_flows`, `_CORS_ORIGINS`, `_CORS_ORIGIN_REGEX`, `_TRUSTED_HOSTS` and `_requests_processed` with every function that reads or rebinds them, `routes_builder` holds the builder store and its lock, `routes_visualize` the preset cache (a `from .application import _api_key` elsewhere would bind a copy that `configure()` no longer reaches; a layout test scans every module for an import or a `global` statement of such a name, and behaviour tests show `configure()` reaching a route of every module and the WebSocket). What does change: a private name is imported from the module that defines it (`_api_key`, `_flows`, `_require_api_key`, `_WS_CLOSE_*` from `server.application`, `_builder_sessions`, `_builder_busy`, `_get_builder_internal_state` from `server.routes_builder`, `_fsm_dict_to_snapshot`, `_graph_error_reason`, `_read_preset_json`, `_scan_fsm_presets`, `build_fsm_graph` from `server.routes_visualize`); the names the old module only imported (`logger`, `asyncio`, `InstanceManager`, the definitions, the constants) and its route functions (`index`, `api_metrics`, ...) are no longer attributes of `fsm_llm.monitor.server`, so a patch aims at the module that performs the lookup (`fsm_llm.monitor.server.support.logger` for the error mapping, `...server.routes_visualize.build_fsm_graph`; a patch aimed at the package raises `AttributeError` instead of silently doing nothing; `fsm_llm.monitor.server.configure` stays the binding `fsm_llm.monitor.__main__` reads); and a loguru record of the server is named after its module instead of `fsm_llm.monitor.server` (`fsm_llm.monitor.server.application`, `.support`, `.routes_monitoring`, `.routes_fsm`, `.routes_workflow`, `.routes_visualize`, `.routes_builder`, `.websocket`; `routes_agent` logs nothing), so the `module` field of a `LogRecord` the dashboard shows (`/api/logs`, the WebSocket `logs`) for those records is that module name and no longer `server`, and its `line` is the line in the new file. Method bodies are the old ones (compared per definition with `ast.dump`: 110 units equal; the two that differ are `STATIC_DIR` and `TEMPLATE_DIR`, by one `.parent`). Behaviour is unchanged: a differential over 3,100 generated programs, old tree against new, in two processes per range (the output files are byte-identical; a second run of the old tree over 200 programs is identical too) (every REST route with success, validation, 404, 400, 413, 429, 501 and 504 paths, API key present, wrong and absent, Host, Origin and CORS variants including preflights, body bounds, malformed bodies, the builder routes over a scripted meta-agent, static files and the page, `configure()` sequences with the environment key set and unset, WebSocket sessions with auth variants and a manager replaced mid-stream, and `app.openapi()`, the route and middleware tables and a fingerprint of every function), compared status, headers, bodies and every log record (module names mapped), and found no difference. One `layers` import contract holds the package (`monitor-server-layers`: `websocket`, then the six route modules, independent of each other, then `support`, then `application`; 20 plants break it and 6 controls stay kept). Tests added (46): `test_server_package.py` (39: the directories, the registration order, one home per global, `configure()` through every router module and the WebSocket, the request counter, the builder lock, the layout), six in `test_server_security.py` for edges of the server that no test pinned and that the mutation proof of the split reached (the Origin `null`, the 504 of `_bounded` (two cases), the stale-builder sweep keeping a session with a send in flight, a send to a busy session, `configure()` closing the previous manager) and `test_the_table_ends_empty` in the size guard (45 named mutations of the new package: 37 were caught by the suites at once; the 8 survivors were the WebSocket's starting manager, the request counter that `configure()` reads, the builder lock, the 504 of `_bounded`, the Origin `null`, the stale-builder sweep, the busy-session refusal of a send and `configure()` closing the previous manager; each is now pinned by a test and caught).
- **BREAKING:** the 49 route functions that 0.12.0 defined at module level in `fsm_llm.monitor.server` no longer import from it: `index`, `health`, `api_auth` and `no_cache_static` live in `fsm_llm.monitor.server.application`; `api_metrics`, `api_conversations`, `api_conversation`, `api_activity`, `api_events`, `api_logs`, `api_config_get`, `api_config_set`, `api_dashboard_config_get`, `api_dashboard_config_set`, `api_dashboard_config_delete`, `api_info`, `api_capabilities`, `api_instances`, `api_instance_detail`, `api_instance_events` and `api_instance_destroy` in `...server.routes_monitoring`; the five `api_fsm_launch`, `api_fsm_start_conversation`, `api_fsm_converse`, `api_fsm_end_conversation` and `api_fsm_conversations` in `...server.routes_fsm`; the seven `api_workflow_*` functions (`api_workflow_presets`, `_launch`, `_advance`, `_cancel`, `_event`, `_status`, `_instances`) in `...server.routes_workflow`; the four `api_agent_launch`, `api_agent_status`, `api_agent_result` and `api_agent_cancel` in `...server.routes_agent`; `api_fsm_load`, `api_fsm_visualize`, `api_fsm_visualize_preset`, `api_presets`, `api_preset_fsm`, `api_agent_visualize` and `api_workflow_visualize` in `...server.routes_visualize`; the four `api_builder_start`, `api_builder_send`, `api_builder_result` and `api_builder_delete` in `...server.routes_builder`; and `websocket_endpoint` in `...server.websocket`. The module attribute `templates` (the `Jinja2Templates` of 0.12.0) is now `fsm_llm.monitor.server.application.templates`. `from fsm_llm.monitor.server import api_metrics` raised nothing in 0.12.0 and raises `ImportError` now; import the function from the module named here, or call the REST route instead. Unchanged and still importable from `fsm_llm.monitor.server`: `app`, `configure`, `get_manager`, `STATIC_DIR` and `TEMPLATE_DIR` (`get_flows` is new in this release). The routes themselves (paths, methods, fields, status codes) are unchanged, the route functions keep their bodies, and a patch aims at the module that performs the lookup.
- The layered import contract of the roadmap's Section 5.2 is enforced (plan d454fd62, step 95, the first step after the 0.4 module splits, D-016 and D-042). `.importlinter` now holds 32 contracts. The `layers-section-5-2` contract orders the layer packages (`(serve) | monitor | eval`, `agents | synthesis | workflows | reasoning`, `(simulation) | (oo)`, `engine`, `handlers`, `policy`, `context | (retrieval) | okf | skills | tools`, `(runtime)`, `llm`, `core`; a name in parentheses is a roadmap package that does not exist yet): a layer imports only the layers below it. Two deviations from the roadmap table, both recorded: `policy` has a row above the L3 siblings because it imports `tools` (D-037), and `handlers` sits below `engine` instead of beside it (stricter, and it holds). A kernel contract keeps `constants`, `logging`, `security`, `text`, `env`, `frontmatter`, `memory` and `cli_support` off every layer package, and the per-package "imports no higher layer" lists of steps 69 to 86 are replaced by it (eight contracts removed, two small ones added, the engine one narrowed; proved by planted imports: all 592 imports that the old contracts forbade fail the new set). The first run on the real tree listed 10 module edges; three were fixed by moving a symbol down, bodies unchanged (checked per definition, and by a differential of 3,000 generated programs, old tree against new, with the log records, 0 differences; 18 of 19 mutants caught, the survivor changes only a resource bound that the 16 KB frontmatter cap makes unobservable): the redaction walk (`redact_context`, `REDACTED`, `CYCLE_PLACEHOLDER`, three private helpers) from `fsm_llm.context.filtering` to the new `fsm_llm.core.redaction`, which retires the debt `tools.arguments` to `context.filtering`; `bundle_from_memory_entries` with `_native` and `_slug` from `fsm_llm.engine.okf_bridge` to the new `fsm_llm.okf.memory_export` (the bridge re-exports it, so `from fsm_llm.engine.okf_bridge import bundle_from_memory_entries` still works, and `okf.cli` no longer reaches an engine module), which retires the debt `okf.cli` to `engine.okf_bridge`; `OBSERVATION_ENTRY` from `agents/constants/outcomes.py` to `fsm_llm.tools.constants` (`fsm_llm.agents.constants` re-exports the same object), which retires the import of `agents.constants` by `synthesis.tools`. No path of the 0.12.0 tree changes (`okf_bridge` and `context.filtering` are newer, and the one BREAKING bullet that named `context.filtering` for `redact_context` now names `fsm_llm.core.redaction`). Three debts remain as named `ignore_imports` lines, each with a retiring roadmap row or a written permanent reason, and a test pins the set: `synthesis.simulation` and `synthesis.repair` to `eval.{cases,config,stats}` (roadmap 6.6 and P7.6), `synthesis.library` and `synthesis.synthesizer` to `agents.semantic_tools` (roadmap 3.2), and `agents.reasoning_react` to `reasoning` (permanent: no roadmap row retires it; the owner decides). The new non-slow test `tests/test_fsm_llm_regression/test_import_contract.py` runs `lint-imports` as part of pytest (it skips only when the `dev` extra is not installed), checks that every `ignore_imports` line is a tagged debt, that every package on disk is placed in the layers or kernel contract, and that a planted import in a copy of the tree breaks the named contract. `scripts/synthesis_bench.py` `SRC_PINS` names `core/redaction.py` (98 to 99 entries) and `REQUIRED_PINS` of its test lists it; four pinned files were edited (`tools/constants.py`, `tools/arguments.py`, `agents/constants/outcomes.py`, `context/filtering.py`); every synthesis manifest already differed from source, so no registered block is newly voided, and `scripts/bench_data/` is untouched. `tests/fixtures/api_snapshot.txt` is byte-identical. The CI and tox wiring of `lint-imports` is steps 108 and 109.
- The docs describe the Section 5 layout (plan d454fd62, step 96, a docs-only step, no code behaviour changes). The root `AGENTS.md` architecture diagram shows the layer packages, the kernel and the seven subpackages with the named import debts; its key-files table gained rows for `handlers/`, `core/visualize/` and `core/redaction.py`, the engine builders and session store, and the kernel and entry modules; `docs/architecture.md` has a Package Layout section; the stale module paths (`pipeline.py`, `llm.py`, `utilities.py`, `fsm_llm.agent_skills`, `monitor/server.py` and the rest) were replaced by the new dotted paths in the `AGENTS.md` and `README.md` files, and comments of three bench-pinned files were corrected without changing their code (D-043). Two new non-slow tests: `tests/test_fsm_llm_regression/test_docs_layout_paths.py` fails when a current doc names a dotted path or an old module file that does not exist, and `tests/test_fsm_llm_regression/test_import_laziness.py` pins the exact set of packages `import fsm_llm` loads (no subpackage, no `okf`, `skills`, `tools` or `policy`). The existing laziness probes of the OKF, Agent Skills, frontmatter and CLI-support tests now also cover `synthesis` and `testing`.
- **BREAKING:** `ToolRegistry`, `tool`, `redact_secret_entries`, `tool_parameters_schema` and `schema_types` (present in `fsm_llm.agents.tools` in 0.12.0) are defined in `fsm_llm.tools` and no longer in `fsm_llm.agents.tools` (plan d454fd62, step 85), with no alias and no shim: `from fsm_llm.agents.tools import ToolRegistry` becomes `from fsm_llm.tools import ToolRegistry` (or `from fsm_llm.agents import ToolRegistry`, unchanged), and likewise for `tool`. `register_agent`, `normalize_tool_input`, `refuse_execute_without_gated` and `tool_call_from_context` stay in `fsm_llm.agents.tools`. The new helpers added to the same module after 0.12.0 (`registry_from`, `check_timeout`, `call_with_timeout`, `ToolTimedOut`, `callable_name`, `infer_schema_from_hints`) were never tagged and live in `fsm_llm.tools` only. A patch aimed at `fsm_llm.agents.tools.<name>` for a moved name or for a module attribute the old module only imported (`time`, `logger`, `redact_context`, ...) must aim at the module that performs the lookup, for example `fsm_llm.tools.registry.time` or `fsm_llm.tools.timeouts.logger`.
- **BREAKING:** `ToolDefinition`, `ToolCall`, `ToolResult` and `ToolAnnotations` (present in `fsm_llm.agents.definitions` in 0.12.0) are defined in `fsm_llm.tools.models` and no longer in `fsm_llm.agents.definitions` (plan d454fd62, step 85), with no alias and no shim: `from fsm_llm.agents.definitions import ToolCall` becomes `from fsm_llm.tools import ToolCall` (or `from fsm_llm.agents import ToolCall`, unchanged). `READ_ONLY_ANNOTATIONS` was added after 0.12.0 and lives in `fsm_llm.tools` only. `AgentConfig`, `AgentResult` and every other model of the module stay.
- **BREAKING:** `ToolExecutionError` and `ToolNotFoundError` (present in `fsm_llm.agents.exceptions` in 0.12.0) are defined in `fsm_llm.tools.errors` and no longer in `fsm_llm.agents.exceptions` (plan d454fd62, step 85), with no alias and no shim: `from fsm_llm.agents.exceptions import ToolExecutionError` becomes `from fsm_llm.tools import ToolExecutionError` (or `from fsm_llm.agents import ToolExecutionError`, unchanged). `AgentError` is now defined in `fsm_llm.core.definitions.errors` (`from fsm_llm.core.definitions import AgentError`, or `from fsm_llm.agents import AgentError`, unchanged; only its `__module__` changes, and `except AgentError` still catches every agent and tool error). `bounded_repr` and `require_int_limit` (added after 0.12.0) live in `fsm_llm.tools` only.
- **BREAKING:** the module `fsm_llm.agents.truncation` (present in 0.12.0) moves to `fsm_llm.tools.truncation` and no longer exists (plan d454fd62, step 85). The old path does not import, with no alias and no shim: `from fsm_llm.agents.truncation import smart_truncate` becomes `from fsm_llm.tools import smart_truncate`.
- **BREAKING:** the module `fsm_llm.agents.hitl` (present in 0.12.0) no longer exists (plan d454fd62, step 86), with no alias and no shim. `HumanInTheLoop`, `ApprovalPolicy`, `ApprovalCallback` and `EscalationCallback` are defined in `fsm_llm.policy` (`fsm_llm.policy.hitl`, `fsm_llm.policy.approval`): `from fsm_llm.agents.hitl import HumanInTheLoop` becomes `from fsm_llm.policy import HumanInTheLoop` (or `from fsm_llm.agents import HumanInTheLoop`, unchanged). `make_hitl_checker` moves to `fsm_llm.agents.tools` (`from fsm_llm.agents.tools import make_hitl_checker`; it was never in `fsm_llm.agents.__all__`). A patch aimed at `fsm_llm.agents.hitl.<name>` must aim at `fsm_llm.policy.hitl.<name>` (the manager's `logger`, `check_timeout`, `call_with_timeout`), and a log filter on the record name `fsm_llm.agents.hitl` now matches `fsm_llm.policy.hitl`.
- **BREAKING:** `ApprovalRequest` (present in `fsm_llm.agents.definitions` in 0.12.0) is defined in `fsm_llm.policy.approval` and no longer in `fsm_llm.agents.definitions` (plan d454fd62, step 86), with no alias and no shim: `from fsm_llm.agents.definitions import ApprovalRequest` becomes `from fsm_llm.policy import ApprovalRequest` (or `from fsm_llm.agents import ApprovalRequest`, unchanged). `AgentConfig`, `AgentResult` and every other model of the module stay.
- **BREAKING:** `ApprovalDeniedError` (present in `fsm_llm.agents.exceptions` in 0.12.0) is defined in `fsm_llm.policy.approval` and no longer in `fsm_llm.agents.exceptions` (plan d454fd62, step 86), with no alias and no shim: `from fsm_llm.agents.exceptions import ApprovalDeniedError` becomes `from fsm_llm.policy import ApprovalDeniedError` (or `from fsm_llm.agents import ApprovalDeniedError`, unchanged). It is still an `AgentError` (and so an `FSMError`): `except AgentError` catches it as before.
- **BREAKING:** the distribution is published as `llm-fsm` and no longer as `fsm-llm` (plan d454fd62, step 113, D-020, roadmap row 0.1): `[project] name` in `pyproject.toml` is `llm-fsm`, so the wheel is `llm_fsm-<version>-py3-none-any.whl`, the sdist `llm_fsm-<version>.tar.gz`, and the extras are requested as `pip install "llm-fsm[monitor]"` (the `all` extra refers to `llm-fsm[...]`). `fsm-llm` on PyPI is a different project, so never `pip install fsm-llm`. The import package `fsm_llm`, every console script (`fsm-llm`, `fsm-llm-validate`, ... ) and the OKF product string `fsm-llm/<version>` are unchanged. An environment that installed the old name from a clone must run `pip uninstall fsm-llm` and then `pip install -e .` again (the editable install of the new name does not replace the old one, and a leftover `src/fsm_llm.egg-info` keeps reporting the old distribution until it is deleted or `make clean` removes it). The "From a clone" install hints in the error messages of `fsm_llm`, `monitor`, `agents.mcp`, `agents.remote` and `monitor.otel` now name `pip install "llm-fsm[extra]"`. Also fixed: the wheel build no longer warns that `fsm_llm.monitor.static` is "absent from the `packages` configuration" (`include-package-data = false`; the wheel and sdist file lists are unchanged).
- The `dev` extra installs `numpy` (`numpy>=2.0,<3`, pinned `numpy==2.4.6` in `constraints.txt`, not on Python 3.15 where 2.4.6 has no wheel) so the two tests that `importorskip("numpy")` (a real numpy integer accepted by the tool limit readers) run in CI instead of skipping (plan d454fd62, step 120.2).

### Removed

- `docs/normalization.md` (the naming normalization report) is removed, with its row in `docs/AGENTS.md`. Its rules now live in `AGENTS.md` under `## Naming conventions`; the per-row audit trail is in the CHANGELOG bullets of the renames and in git history.
- **BREAKING:** six prompt functions are removed from `fsm_llm.agents.prompts`, with no alias: `build_orchestrate_response_instructions`, `build_delegate_response_instructions`, `build_collect_response_instructions`, `build_attempt_response_instructions`, `build_assess_response_instructions` and `build_decompose_response_instructions`. They only fed the Orchestrator and ADaPT intermediate states, which are now silent (plan 0068b055 step 7).
- **BREAKING:** `build_adapt_field_instructions(registry=None, task_description=None)` loses both parameters and takes none, with no alias: ADaPT advertises no tool, so neither was read for the typed field instructions. A caller that passed either now gets a `TypeError`.
- **BREAKING:** `AgentGraph.run(task, initial_context=None)` no longer takes `config`, with no alias and no raising stub: the argument was never read (`config=AgentConfig(max_iterations=1)` changed nothing), and a graph-level config has no meaning for member agents that are already built. Passing it is now a `TypeError`; configure each agent when you build it. No caller in the repository passed it.
- **BREAKING:** `SwarmAgent(..., **api_kwargs)` is removed, with no alias and no raising stub: a swarm builds no API (its members own theirs), so `llm_interface=`, `seed=` and even a mistyped keyword were accepted and ignored. The signature is `SwarmAgent(agents, entry_agent, max_handoffs=10, memory=None, config=None)`; any other keyword, also through `create_agent("swarm", ..., **kwargs)`, is a `TypeError`. No caller in the repository passed one.
- **BREAKING:** `build_adapt_fsm(registry=None, task_description="", max_depth=3, context_keys=())` loses its first parameter `registry`, with no alias: ADaPT runs no tool and the parameter had no effect on the FSM. The signature is `build_adapt_fsm(task_description="", max_depth=3, context_keys=())`; a positional caller that passed the registry or `None` first (`build_adapt_fsm(None, "task", 2)`) must drop it, and `registry=` is a `TypeError`. `ADaPTAgent(tools=...)` is unchanged. The FSM output is byte-identical, so no golden hash moves.
- **BREAKING:** `API.active_conversations` (a dict whose keys were always those of `conversation_stacks`) is removed, with no alias: `list_active_conversations()` is the public read, `conversation_stacks` the registry. `fsm_llm.eval.constants.EXIT_OK` and `EXIT_ERROR` are removed, with no alias: they repeated core's `CLI_EXIT_OK` and `CLI_EXIT_FAILURE` (`EXIT_BELOW_THRESHOLD = 2` stays eval's own).
- **BREAKING:** `fsm_llm.workflows.constants.PAUSING_STEP_TYPES` (a set of class names as strings) is removed, with no alias: it was redundant with the `isinstance(inner, (WaitEventStep, TimerStep))` check in `WorkflowDefinition._is_pausing_step`, which is now the only rule, and would have gone stale on the step class renames. A step type that pauses is added to that `isinstance` check.
- **BREAKING:** `fsm_llm.reasoning.utilities.load_fsm_definition` is renamed `get_reasoning_fsm` (no alias): it shared its name with the core export `fsm_llm.load_fsm_definition` and returns a different type (a deep copy of a dict from `ALL_REASONING_FSMS`). It was never in `fsm_llm.reasoning.__all__`.
- **BREAKING:** the iterative-planner harness is removed, with no alias, no shim and no deprecation period. Gone with it:
  - the package `fsm_llm.harness` (and its `README.md` and `AGENTS.md`), so `from fsm_llm.harness import ...` and `python -m fsm_llm.harness` fail; `HarnessAgent`, `HarnessAgentBuilder` and the `HarnessError` tree (`HarnessError` was a subclass of `FSMError`) no longer exist
  - the console script `fsm-llm-harness` (reinstall with `pip install -e .` to drop the stale script)
  - the extra `harness`, and its place in `all`; `pip install "fsm-llm[harness]"` now warns that the extra does not exist, and `make install-dev` no longer asks for it
  - the live-test gate `FSM_LLM_HARNESS_LIVE`, and the test suite `tests/test_fsm_llm_harness/` with the `fsm-llm-harness` tests
  - the CLI and the live half of `scripts/harness_bench.py` (`register`, `run`, `report`, `probe-seed`). Its stdlib helpers (statistics, row files, manifest and drift checks, wire capture) moved to `scripts/bench_common.py`, which has no CLI; `scripts/agents_bench.py` and `scripts/skills_bench.py` now load it, and `tests/test_fsm_llm_eval/test_bench_parity.py` still holds its mirrored helpers equal to their `fsm_llm.eval` twins. The tests of the moved helpers are `tests/test_bench_common.py` (was `tests/test_harness_bench.py`)
  - the harness blocks under `scripts/bench_data/` (`l4-execute-write`, `l6-e2e`, `l7-explore-coldstart`, `l8-explore-loop`, `seed-probe`) are kept as frozen history: the code that produced them is gone, so they cannot be re-run, and they are not edited.
- **BREAKING:** `Defaults.MIN_ANSWER_LENGTH` (`fsm_llm.agents.constants`, value 5) is removed, with no alias: nothing reads it any more (see the short-answer entry under Fixed). `EvaluatorOptimizerAgent._extract_answer` (private) is gone too; the base chain gives the same result.
- `PlanExecuteAgent` no longer registers the `AgentIterationLimiter` handler (its private `_make_iteration_limiter` method and the `_max_iterations` run-context seed are removed). No edge of the PlanExecute FSM read `max_iterations_reached`; see the PlanExecute entry under Fixed. `make_iteration_limiter` stays for the other patterns. No handler was renamed.
- **BREAKING:** `ThinkTurnLimiter.reset` and `AgentHandlers.reset` (`fsm_llm.agents.handlers`) are removed, with no alias: nothing called them (every run builds a fresh instance), and `fsm_llm.agents.base.with_instructions` / `prompt_overflow_error` are now private (`_with_instructions`, `_prompt_overflow_error`; `_create_api` is their only caller). None of the four is exported from `fsm_llm.agents`.
- **BREAKING:** `EventCollector.get_events_since`, `get_logs_since`, `get_events_by_conversation`, `handler_name` and `handler_priority`, `InstanceManager.find_instance_for_conversation` and `OTELExporter.generate_trace_id`/`generate_span_id` (`fsm_llm.monitor`) are removed, with no alias: nothing in `src/`, the SPA or the docs called them (the two `*_since` forms were the pre-cursor streaming API and drop the middle of a burst; use `events_after`/`logs_after`). `InstanceManager.launch_workflow` loses its `initial_context` parameter, which was accepted and ignored (pass the context to `start_workflow_instance`). Plan 5980fbb7 step 16.
- **BREAKING:** `fsm_llm.workflows.WorkflowValidator` is removed (also from `fsm_llm.workflows.__all__`), with no alias: nothing called it (it only wrapped the definition's own check). Use `WorkflowDefinition.validate_structure()` (renamed from `validate()` in this release), which raises `WorkflowValidationError` with `validation_errors`. `DependencyResolver` (still exported) loses `add_dependency`, `get_dependencies`, `get_dependents`, `get_all_steps` and `dependency_count`, which nothing called; `add_step`, `resolve`, `has_cycles`, `step_count`, `clear` and `from_dict` stay. `Timer.is_expired` (`fsm_llm.workflows.Timer`) is removed: only `EventListener.is_expired` was ever called.
- **BREAKING:** `AutoStep` (`AutoTransitionStep` in 0.12.0) and `ConditionStep` no longer RAISE `WorkflowStepError` from `execute()` when the action or condition fails and no `error_step_id` is set: they return a failure result with no `next_step_id` (like `APIStep`, `LLMStep` and the others), and the engine fails the instance from it. A caller that drove `execute()` itself must read `result.success` and `result.error` instead of catching the exception. `LLMStep(llm_interface=...)` is validated at construction: an object with neither a callable `generate` nor a callable `generate_response` raises `ValueError` (pydantic `ValidationError`) instead of failing the instance at run time.
- **BREAKING:** `fsm_llm.session.session_json_default` is removed, with no alias (use `fsm_llm.context.filtering.redacting_json_default`; the CLI dump of `fsm-llm` now uses it, so a secret-looking object prints as `<redacted:Type>`). `fsm_llm.logging.is_library_logging_enabled` is removed (nothing read it). `fsm_llm.expressions.JsonLogicExpression` (an alias of `Any`) is removed.
- **BREAKING:** in `fsm_llm.agents`: `MetaDefaults.MODEL` is removed (use `resolve_model`); `HandlerNames.ADAPT_ASSESSOR` is renamed `ADAPT_DECOMPOSE_TRACKER` (the handler name string `ADaPTAssessor` is unchanged); `HandlerNames.END_CONVERSATION` is removed together with the marker handler that wrote `_agent_completed` and `_agent_type` (nothing read them); `parallel_react.TOOL_CALLS_KEY` is removed in favour of `ContextKeys.TOOL_CALLS`. `meta_prompts.build_followup_message` and `build_output_message` are removed, and `meta_prompts.build_collect_response_instructions` is renamed `build_meta_collect_response_instructions` (the orchestrator one in `agents.prompts` keeps the old name).

### Fixed

- The sdist no longer ships six test files (roadmap row 0.1, plan d454fd62 step 116): setuptools adds `tests/test*.py` to every sdist by default, so `llm_fsm-<version>.tar.gz` held `tests/test_packaging.py`, `test_agents_bench.py`, `test_skills_bench.py`, `test_bench_common.py`, `test_coverage_ratchet.py` and `test_integration_ollama.py`, none of which can run outside the repository (they read the rest of `tests/`, `scripts/`, `docs/` and git). `MANIFEST.in` now has `prune tests`; the wheel is unchanged. `TestSdistContent` (slow) builds an sdist from every tracked file and pins its top-level names, its `src/fsm_llm` files (equal to the wheel's) and its example JSON files.
- `docs/testing.md`: the two not-run fences were written ```` ```python notest ```` and ```` ```diff notest ````. The documentation site's fence parser (`pymdownx.superfences`) cannot open a fence with a bare word after the language, so on the built site the `conftest.py` block turned its first comment into a page heading and swallowed the heading `Migrating from tests.conftest` and the text up to the next fence. The fences are now ```` ```python {.notest} ```` and ```` ```diff {.notest} ````; `tests/test_fsm_llm_testing/test_guide.py` follows, and `test_docs_links.py` refuses any bare-word fence info string in `docs/`.
- State-kind registry fixes from review gate 1 (plan d454fd62 step 58.1, D-026; internal, `executors=` is not public): `executor_for` resolved a kind with no executor of its own (`synthesize`, a runtime kind nobody registered) to the module's built-in conversation executor, so an `API(executors={"conversation": X})` override was silently bypassed for those states; the fallback now goes through the instance's `conversation` entry first (`synthesize` no longer has a module table entry). `CompletionExecutor.extract` raises `FSMError` for a state without a completion config instead of an `assert` that `python -O` strips.
- Safe-defaults regressions found by review gate 1 (plan d454fd62 step 58.1, D-025): with no `LLM_MODEL` set, `SOPRegistry.register` (and so `register_from_dict`, `register_from_file`, `register_directory` and `load_builtin_sops()`) raised `invalid config_overrides: No LLM model is set` for any SOP whose overrides name no model, and `SwarmAgent(config=None)` raised although a swarm builds no API and its members own their interfaces. A registry validates the overrides against `AgentConfig` with a placeholder model that is never stored or returned, and a swarm without a config gets one built with `definitions.model_or_placeholder()`. `SOPDefinition.to_agent_config` takes a keyword-only `llm_interface=` (the injected interface supplies the model, as for `default_agent_config`); without a model, an interface or `LLM_MODEL` it still raises `ModelNotConfiguredError`, because the config would run an agent; a config that holds the placeholder model (an interface that states none and no `LLM_MODEL`) is valid only next to that interface, because `llm.resolve_model` now counts the placeholder (`fsm_llm.llm.PLACEHOLDER_MODEL`, formerly defined in `fsm_llm.agents.definitions`, which still exposes the name) as unset, so used without it the agent raises `ModelNotConfiguredError` where it needs a model instead of a late provider error (D-028). `fsm_llm.eval.cases.run_cases` and `run_dataset` raised `ModelNotConfiguredError` when an `llm_interface_factory` was given and `LLM_MODEL` was unset (the run directory and report needed a model name, and the CHANGELOG advice for a missing model is to inject an interface): with a factory they need no model and label the run with the factory's own `model` attribute, else the model the config or `LLM_MODEL` names, else `injected`; without a factory and without a model they still raise it, now documented in both docstrings. The new sweep `tests/test_fsm_llm_regression/test_no_model_paths.py` deletes `LLM_MODEL` itself (the root autouse fixture sets it for every test, which hid this class of regression) and checks every public constructor and runner that takes an injected interface.
- `API.push_fsm` no longer hands a pushed child the parent's value of the child's own synthesize `result_key` (plan d454fd62 step 55, D-015). A synthesize state's edges must read its result key, so a child that inherited the parent's `synthesis_result` (a parent that had already run a procedure, `inherit_context=True`, the default) showed the stale result in `get_data` and routed on it at its first step, before any run wrote one. The key is now dropped from the inherited context exactly as the completion `result_key` already was; an explicit `context_to_pass` still wins, a key the child does not own is inherited as before, and the parent's own value is untouched by the push and the pop. The procedure runner is unaffected (it pushes with `inherit_context=False`).
- `security.resolve_within` raises on a symlink loop on Python 3.13 to 3.15 too (plan d454fd62 step 11). `Path.resolve()` stopped raising on a loop in 3.13, so the shared confinement check returned the looped path: `write_bundle` then replaced a loop symlink at its target instead of refusing it ("cannot be resolved"). A strict resolve of the result now re-raises `OSError` (ELOOP) on 3.13 and later; 3.11 and 3.12 behave as before.
- Windows static review (plan d454fd62 step 10). `fsm-llm-visualize` (and `fsm-llm --mode visualize`) wrote its box-drawing diagram straight to stdout, so a redirected Windows stdout (cp1252) or any pipe whose encoding cannot hold it ended in a `UnicodeEncodeError`; it now writes unencodable characters as `\uXXXX` through `cli_support.escape_unencodable_output`, as the other CLIs do. `fsm-llm-eval examples` now runs every example child with `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`: the runner already encoded the stdin script and decoded the pipes as UTF-8, so a child on a cp1252 host mangled every non-ASCII reply and died printing an emoji. `scripts/audit_pth.py` (`make audit`) read each `.pth` as strict UTF-8: an ANSI byte (a Windows user name in a path) crashed it with a `UnicodeDecodeError`, and a UTF-8 BOM in front of an `import` line, which `site` accepts, hid executable code from the check; it now decodes `utf-8-sig` with replacement characters and scans what it can read.
- `FileSessionStore` reads and writes session files as UTF-8 on every platform (plan d454fd62 step 7, D-011, D-018). `load` used the locale encoding, so a UTF-8 session file holding non-ASCII text came back as `None` (the file counted as unreadable) on Windows and under the C locale. Every other text file call in `src/` and `scripts/` now names `encoding="utf-8"` too, and ruff rule `PLW1514` (explicit preview) plus `tests/test_packaging.py::TestExplicitEncodings` keep it that way.
- `FileSessionStore.load` returns `None` for a session file too deep for the JSON parser (a count-based parser, Python 3.11 to 3.13, raised `RecursionError` out of a method that never raises).
- `scripts/audit_pth.py` (`make audit`, the `lint` job and every test leg) accepts `pywin32.pth` on Windows, where the `mcp` extra pulls in pywin32: the audit flagged its line `import pywin32_bootstrap` and stopped both Windows legs of the first CI run. The file is accepted by content, not by name: its name must be `pywin32.pth` and every line must be blank, a `#` comment, a path line or exactly `import pywin32_bootstrap`; any other code in it, the same line in another file name, or a byte-order mark in front of other code is still flagged.
- `scripts/audit_pth.py` (`make audit`) now judges a `.pth` the way `site` reads it on every supported Python (plan d7719b03 step 9.1). `site` splits the file with `str.splitlines()` on 3.12.15 and later, which also ends a line at a form feed, `\x0b`, `\x1c` to `\x1e`, `\x85`, `\u2028` and `\u2029`, while the audit split at the newline only: `win32<U+2028>import x` after a path or a comment passed the audit and ran on 3.12.15 to 3.15, in any file name. Every line of both splits is now judged, the content is decoded as `site` decodes it (UTF-8 with an optional BOM, then the locale encoding), and the `.start` entry-point files of Python 3.15 (PEP 829) are scanned: any entry point is flagged. Two residuals are stated in the script: a `pywin32_bootstrap.py` placed in site-packages shadows the module that the vetted `pywin32.pth` line imports, and the user site directory is not scanned.
- `fsm-llm-okf` and `fsm-llm-skills` also park stdout and exit 1 when the reader of a pipe is gone on Windows, where a write or flush to a closed pipe raises `OSError` with `EINVAL` instead of `BrokenPipeError` (`fsm_llm.cli_support`; before, the error ended in a traceback or an `error:` line). An `EINVAL` error that names a file (Windows reports a `<` or `|` in a path that way) is an ordinary error, reported on one `error:` line, not taken for a gone reader.
- Agent Skills read a bundled text file with CRLF line ends like its LF twin (plan d7719b03 step 11.1, first Windows CI run): `SkillCatalog.read_resource` (so the `skill_read` tool and `fsm-llm-skills show FILE`) passed the decoded text to `strip_unsafe_text`, which turns every `\r` into a space, so a file checked out with `core.autocrlf=true` gained a trailing space per line and its `length` and page offsets moved by one per line. `\r\n` and a lone `\r` become `\n` first, as `split_frontmatter` already did for a `SKILL.md` body; offsets and `length` count the normalised text and `max_file_bytes` still counts the file.
- A symlink swapped in after the checks is refused on Windows too (plan d7719b03 step 11.1): `fsm_llm.frontmatter.read_bytes`, the one bounded reader of OKF concepts and Agent Skills files, relied on `O_NOFOLLOW`, which Windows does not have, so the race described at `skills/catalog.py` D-023 was open there. Where the flag is missing the name is checked after the open (not a symlink, and the same file the handle holds, `OSError` with `ELOOP` otherwise); POSIX is unchanged.
- `security.resolve_within` keeps its contract on Windows with Python 3.13 and later (plan d7719b03 step 11.1): a NUL byte in the relative path raises `ValueError` on every platform (the non-strict `resolve()` of Windows 3.13 and later no longer raised), and a symlink loop reported as `winerror` 1921 (Windows has no `ELOOP`; 1920 also stands for a reparse point Windows cannot follow, so it is not read as a loop) (Windows has no `ELOOP`) raises like `ELOOP`. Before, the callers still refused such a path, but through other branches or with other messages.
- The monitor answers 400 `invalid preset ID`, not 500, when the preset id cannot be resolved (plan d7719b03 step 11.2): `validate_preset_id` (the preset routes, `/api/preset/fsm/` and `/api/fsm/visualize/preset/`, and the launch of a preset) let the `OSError` or `RuntimeError` of `security.resolve_within` (a symlink loop or an unreadable link on the way) reach the route as a failure to read the preset. It now raises `ValueError("Invalid preset ID")` like an id that escapes `examples/`.
- Concurrent saves of one file no longer fail on Windows (plan d7719b03 step 11.1): `FileSessionStore.save`, `save_working_memory` and `SemanticMemoryStore.save` write a temp file and `os.replace` it over the target, and while another writer's replace holds the target Windows refuses a second one with `PermissionError` (`WinError 5`; 367 of 1,200 saves in the first Windows CI run). On win32 only, a refused replace is retried through a backoff of about 0.3 s in all, then raises as before; other platforms raise at once. The loop is written out at the three sites (their decision anchors refuse a shared helper); the other `os.replace` sites are not written concurrently and are unchanged.
- `fsm-llm-eval examples` interrupted (SIGTERM or Ctrl-C) before the first example finished left no `logs/` directory although the closing banner printed its path (plan 0068b055 step 12.30, live finding D5). `run_examples` now creates `<run dir>/logs/` before the first example starts; `results.json` (`interrupted: true`) and `scorecard.md` were already written.
- Library logging no longer prints `Context update contains internal-prefix key: '_max_iterations'` on every agent run (plan 0068b055 step 12.29, D-026, live finding D4). `FSMContext.update` warned for any internal-prefix key, but every caller that reaches it writes the key by name through a public path (`start_conversation` seeding, `push_fsm` inheritance, `API.update_context`); an extraction result is stripped by `clean_context_keys` first and a handler delta is merged straight into `context.data`, so no accidental write ever reached the line. The write itself is unchanged (the keys are still stored, hidden from prompts and `get_data`); the debug line names them. The reserved-key warning for a handler overwrite stays.
- Docs only, no behaviour change: the `fsm_llm.expressions` module docstring and the `_numeric_equal` docstring no longer claim that `==`, `<=` and `>=` agree. They agree on a number against a numeric string (`1.0`, `"1"`); `<=` and `>=` also accept two numeric strings spelled differently (`"01" <= "1"`) and a bool against a numeric string (`true <= "1"`), where `==` is False. The behaviour is kept (dropping it would flip a numeric-string gate such as `>= "50"` on an extracted `"50.0"`); pinned by `TestNumericStringOrdering`. The formats of the seeded `_conversation_start` (naive local ISO text), `_timestamp` and `_transition_timestamp` (epoch seconds), none persisted in sessions, are pinned in `test_pipeline.py`.
- `API(path)` and `API.process_fsm_definition(path)` accept an `os.PathLike` like `API.from_file` and `APIBuilder.set_definition` always did (the builder no longer converts on its own). A failing stream turn (`converse_stream`, `advance_stream`) logs the same ERROR line with the traceback as the sync turn. `fsm-llm-eval examples` no longer crashes on an example file (or example output) that is not UTF-8.
- OKF and Agent Skills (plan 5980fbb7 step 9): `check_bundle` reports `check-deadline` once even after `backlinks` already left it in `bundle.issues` (it was listed twice and `fsm-llm-okf validate` counted two errors); the frontmatter bracket pre-check (`frontmatter._flow_depth_exceeded`) no longer counts `[` and `{` inside a quoted scalar or a `#` comment, so a flat description with over 64 literal brackets loads instead of being refused as `frontmatter-too-deep` (genuinely nested flow collections are still refused at the cap, and the brackets of all quoted spans together are bounded at 4 times the cap because the span reading is a heuristic); `fsm-llm-okf validate` hints `--max-seconds` next to a `check-deadline` issue too, not only `load-deadline`. `fsm-llm-okf validate` now also control-strips each issue line, as `fsm-llm-skills` already did. `okf.py` has a static `__all__` and no longer keeps the private aliases `_Note` and `_SURROGATE_RE`. `fsm-llm-okf validate` and `fsm-llm-skills validate` share one report loop and one `--max-seconds` argument in `cli_support` (`report_issues`, `add_max_seconds`); output is unchanged.
- Agents foundation (plan 5980fbb7 step 10): `run_evaluation_fn` (Reflexion, EvaluatorOptimizer) really never raises: with library logging on, a verifier error holding braces (`{"ok": false}`) made loguru format the message and raise `KeyError`; the six `exc_info=True` log calls (also ADaPT, SelfConsistency x2, Orchestrator, ReasoningReact) now use `logger.opt(exception=True)`, so the traceback is finally printed too. ReasoningReact's `reason` path records its step like every executor (entry cut to 2,000 characters, last 20 kept, trace `tool_input`, so `AgentResult.trace.tool_calls` carries the problem) through one `handlers.record_tool_step` shared with `AgentHandlers` and ParallelReact, and a failed `reason` call is a `[TOOL FAILED]` observation and clears `should_terminate`. `HumanInTheLoop.request_approval` with `approval_timeout` runs the callback through the tool timeout path: the caller's context variables are visible, `SystemExit` from the callback is raised instead of reported as a timeout, and a bad `approval_timeout` (0, negative, NaN) raises `ToolExecutionError` instead of denying every request. `PythonToolLoader.from_functions` skips a callable without `__name__` (a `functools.partial`) with its warning instead of raising. `create_agent(tools=[...])` accepts a `ToolDefinition` like `ConfiguredAgentBuilder.add_tool` (one `tools.registry_from`), and `tools=[]` is no tools (a tool-less pattern no longer raises `TypeError`). `create_memory_tools` and `create_semantic_memory_tools` build their tools with `@tool`: they now carry an `args_model` (the native function-calling schema shows the `buffer` defaults) and `recall` / `list_memories` are annotated read-only and idempotent, so a `RetryingToolRegistry` may retry them.
- Agents integrations (plan 5980fbb7 step 15): `SwarmAgent(memory=WorkingMemory())` and `PythonToolLoader.to_tool_registry(registry=ToolRegistry())` keep the caller's empty container (an empty one is falsy and was replaced by a private one); `OKFMemoryBackend(store=...)` falls back to the lexical search when the store yields no usable hit (its embedding provider down returned `[]`); `SwarmAgent` reports `total_iterations` as the sum of its members' iterations, as `AgentGraph` does (it counted agents run), and a failed hand-off keeps the tool calls, iterations and `_swarm_traces` of the agents that ran; a `bool` offset or limit in the OKF tools and `okf.text_page` is 0 or the default, as in the skill tools (`True` was 1); an MCP result with empty `content` shows its `structuredContent` as JSON, else `""` (it showed the pydantic repr); `SemanticToolRegistry.retrieve` raises `ValueError` for `top_k < 1` (`0` meant the default, `-1` all but one); `SOPRegistry.register_from_file` raises `ValueError` naming the file for bad JSON or YAML, a non-mapping or a missing `name` (it raised `JSONDecodeError`, `TypeError` or `KeyError`); `SemanticMemoryStore.load` reads a blank file as an empty store and names the file in an invalid-payload error, like the constructor; MCP discovery and tool calls share one `_open_session` helper (the dead `MCPToolProvider._session` is gone); `AgentServer` and `RemoteAgentTool` timeouts come from `Defaults`.
- Monitor instances (plan 5980fbb7 step 16): a launch that fails validation (bad FSM JSON, invalid agent config) no longer evicts the oldest finished instance (`_make_room` now runs after everything that can fail, in one `_register_instance` shared by `launch_fsm`/`launch_workflow`/`launch_agent`); a terminal reply whose `end_conversation` the core refuses leaves the conversation unmarked as ended (as `end_conversation` already did) instead of completing the instance; a re-enabled `OTELExporter` no longer exports every event twice after another exporter wrapped over it; the lifecycle events of an agent or workflow (`instance_launched`, `agent_*`, `workflow_started`, `workflow_event_delivered`) now also reach the instance's own collector (`/api/instances/{id}/events`); `InstanceManager.shutdown()` tears down workflow engines and signals agents like `destroy_instance` does (it only closed FSMs), also an engine launched through `POST /api/workflow/launch` (that route builds the instance in a worker thread; `start_workflow_instance` records the loop that owns the engine); presets and `flows.json` are read as UTF-8 under any locale (one `read_json_file`, one `load_preset_json` shared by the manager and the server; `/api/presets`, `/api/preset/fsm/...` and a preset launch failed or showed mojibake under a non-UTF-8 locale).
- Reasoning (plan 5980fbb7 step 18): `python -m fsm_llm.reasoning` writes its output through `cli_support.escape_unencodable_output` (as `fsm-llm` and `fsm-llm-meta` do), so a solution or a trace step the console cannot encode (and the `→` of `--output detailed`) is written as `\uXXXX` and `--save` still runs after a finished solve (it exited 1 and lost the run); `solve_problem` measures the wall-clock time and `trace_info["reasoning_trace"]["execution_time_seconds"]` (and the CLI JSON metadata and `average_step_time`) is a float instead of `null`; `reasoning_strategy == "direct computation"` no longer overrides the classified type (only that exact spelling did, and the prompts ask for `simple_calculator`; `validate_solution` no longer treats it as a simple problem either; the alias still maps to `simple_calculator` in `map_reasoning_type`); the classifier API no longer carries the `ReasoningTracer` handler (its trace was never read); the pruned-string marker is core's `... [truncated]` (was `...[truncated]`); `--output detailed` no longer repeats the steps, types and confidence in a second "Metadata:" block, `--output text` no longer appends the raw confidence a second time, and `--quiet` prints the stripped solution; a failed classifier cleanup is a WARNING like the orchestrator's (was DEBUG); an FSM definition missing from the package now raises `KeyError` at `ReasoningEngine` construction instead of logging and leaving the type out. No prompt text changed: the 11 FSM definitions hash identically (`tests/test_fsm_llm_reasoning/test_golden_tables.py`).
- Workflows (plan 5980fbb7 step 19): a step that fails by RAISING (`AutoTransitionStep`, `ConditionStep`) now leaves a `step_completed` hook, a step-history entry and the cause text in `instance.error`, like every step that returns a failure result; a `TimeoutError` raised by user code inside a step with `timeout` set keeps its own text instead of becoming "Step timed out after Ns"; `ParallelStep(aggregation_function=)` accepts an `async def` (a sync one runs in the step executor, like every other user callable); `schedule_timer` refuses a finished instance with `WorkflowInstanceError`, as `register_event_listener` already did; the first PENDING to RUNNING move is announced to hooks (`status_changed`, `running`, before `step_started`). The three timer plumbings, the eight key strings, the three "complete from a wait" copies, the eight `_get_referenced_states` branches (now `WorkflowStep.referenced_states()`), the vacuous `_listener_lock`, `_cleanup_workflow_resources` and `_handle_step_exception` are one helper each or gone; no behaviour change.
- Core (plan 5980fbb7 steps 3 to 8): `ContextCompactor` clears keys like `clear_keys_delta`; a per-field prompt context that cannot be serialised drops its section with a WARNING instead of failing the turn; the Pass-2 history keeps both messages of a multi-key entry and the field history reads the role case-insensitively; a rejected extraction logs its reason; `Conversation` clips user and system messages by one rule; memory session ids use `SESSION_ID_RE` and `save_working_memory` fsyncs before `os.replace`; reasoning-trace recovery reads the last whole JSON object with the string-aware scanner; a repeated tool-call id makes the turn `kind="malformed"`; `HierarchicalClassifier` builds one interface instead of 1+N; `security` secret patterns end on an explicit word boundary and `extract_json_from_text` strategy 4 reuses the brace partners (both pinned by exhaustive differentials, 0 mismatches).
- Agents patterns (plan 5980fbb7 steps 10 to 14): structured output reads the model's answer before context hints; ADaPT fills `structured_output` and registers the lifecycle handlers; an LLM judge reading a `NaN`, `inf` or `bool` score scores 0 (the judge fails); a raising or non-`str` `aggregation_fn` is an `AgentError`; the observation summariser never re-wraps its own summary; REWOO keeps a tool name holding a parenthesis; a failed ReasoningReact `reason` call is recorded as a `[TOOL FAILED]` observation; `fsm-llm-meta` whole-message build triggers ignore trailing punctuation (`Done.`, `build now!`); a failed `MetaBuilderAgent.start` leaves the object retryable; a meta build reply is read through `extract_json_from_text`; an orphan state in a meta FSM build is one error.

A pass over `src/fsm_llm` fixed 21 CRITICAL and HIGH findings. Each has a test that fails on the parent commit.

- `fsm-llm` (the runner) prints the greeting, each `System:` reply, the `Type 'exit'...` hint and the final redacted `Data:` dump to stdout with `print()`, and a failure to stderr. They used to go only to the logger, so after `setup_file_logging` (or with library logging off) a terminal session showed nothing. The file log keeps the `logger` records; a silent reply prints no `System:` line.
- The JSON file sink stays valid JSONL for text that looks like loguru markup (`<red>`, `</b>`) and for records with an exception: `<` is written as the JSON escape `\u003c` (it decodes back to `<`) and the `{exception}` traceback tail is dropped (the record already carries `exception_type` and `exception_message`). Before, such a record could be lost or turned into a markup error, and a traceback was appended after the JSON line. A JSON consumer decodes the file as before; a raw `grep '<'` on the bytes no longer matches.
- Secret filter (`is_forbidden_context_entry`): a digit run after `key`, `token`, `secret` or `credential` is part of the name, so `key1`, `token2`, `secret1`, `private_key2` and `ssh_key2` are stripped by name like `key`, `token`, `secret` and `private_key` (they reached the prompt before). `key_2` and `max_tokens2` are unchanged. Under an ambiguous `*_key` name, the string leaves of a list, tuple, set or dict value and a bytes value are now judged by the same value test (bounded, failing closed); `partner_key: ["sk_live_..."]` used to be kept because every non-`str` value was. A very large honest container under an ambiguous `*_key` name can now be stripped.
- The transition gate decides on the committed context. The pipeline passes no extracted data to `evaluate_transitions`, so a CONTEXT_UPDATE handler that overwrites or deletes an extracted key is what the transition reads (before, the raw extraction overlaid the handler's change). The `extracted_data` parameter stays on `TransitionEvaluator`. MakerChecker's forced pass and Debate's forced consensus now route on their own turn.
- Equal-priority transitions to the same target are one classifier candidate (one target is DETERMINISTIC, with no model call). A tie set the classifier schema cannot hold (15 or more distinct targets, or a target named `_stay_in_state`) no longer crashes the turn with a `ValidationError`: the conversation stays in the state with a WARNING, and `fsm-llm-validate` warns about the definition.
- A JsonLogic arithmetic operator that fails (`+ - * / % min max` on a non-numeric operand or with the wrong arity; a zero divisor still raises) yields the private UNDEFINED value, so a comparison never reads it as 0 and `evaluate_logic` returns `None` instead of `False` for such a bare expression.
- A `bool` field reads a closed yes/no vocabulary (`utilities.parse_bool_word`: `true/yes/y/1` and `false/no/n/0`, case-folded, surrounding quotes and trailing `.!?,;:` ignored). Any other string, such as `"maybe"`, is an invalid extraction (retried or left unset) instead of a confident `False`. The `llm.py` unstructured bool rung uses the same helper, so it now also accepts `y`, `n` and `"Yes."`. A wider vocabulary (`yeah`, `sure`) is not accepted on purpose.
- Debate and MakerChecker stop at their caps when the judge or the checker extracts null. Core skips CONTEXT_UPDATE handlers on a turn that commits no extracted key, which left the round or revision uncounted. Debate now bumps `current_round` on `propose` entry, records the round and the forced reason in a PRE_TRANSITION handler on `judge` (the `DebateRoundTracker` name is unchanged, its timing moved), and the `judge -> conclude` edge reads `current_round >= max_rounds`. MakerChecker counts revisions in a new `MakerCheckerRevisionCount` handler on `check` entry and the `check -> output` edge reads `revision_count >= max_revisions` (new optional `max_revisions` parameter of `build_maker_checker_fsm`); `_track_revisions` only records the trace and the quality auto-pass. The non-null caps are unchanged.
- PlanExecute: a normal plan is no longer a forced stop. The removed limiter counted every transition (2k+1 for a k-step plan) and set `max_iterations_reached`, so at the default `max_iterations=10` a plan of 4 or more steps skipped its later tools and reported `success=False`. The bound is core's `max_iterations * 3` step ceiling (`BudgetExhaustedError`). `trace.total_iterations` is now core's loop-turn count for this pattern. A failed step past the `max_replans` cap now records `forced_stop_reason = no_result` (`success=False`), as a timed-out step already did.
- A raising `evaluation_fn`, or one that returns something other than an `EvaluationResult`, is a failed evaluation (`passed=False`, feedback `Evaluation error: ...`, WARNING) on Reflexion and EvaluatorOptimizer, through the new `handlers.run_evaluation_fn`. Before, Reflexion's raise was swallowed by core's continue mode and the model self-evaluated, and EvaluatorOptimizer kept a wrong-typed return. Behaviour change: an EvaluatorOptimizer `evaluation_fn` that returns a bool now refines (it used to ship the first draft).
- `ParallelReactAgent`: `tool_calls` is a run output that caller `initial_context` can no longer seed, so a seeded list cannot run tools without a model choice (also through `AgentServer`).
- EvaluatorOptimizer, PromptChain and ADaPT ship a short answer: the test is "non-empty after strip", not "longer than 5 characters" (`"Paris"`, `"42"` and `"Yes"` used to be replaced by the failure sentence with `success=True`). `Defaults.MIN_ANSWER_LENGTH` is removed (see Removed). ADaPT still skips `{"extracted_data": ...}` envelopes.
- `meta_builder` built through `create_agent` or `ConfiguredAgentBuilder` gets a `MetaBuilderConfig` (new classmethod `MetaBuilderConfig.from_agent_config`): the fields you set carry over and the 0.7 temperature, 4096 max tokens and `max_turns=50` defaults hold, so `send()` has its `max_turns` limit. `set_config_option("max_turns", n)` is accepted for that pattern.
- `MCPToolProvider.register_tools` skips a tool whose name is already in the registry and warns (`UserWarning` and the logger), so a server cannot replace a local or approval-gated tool. The return value counts only registered tools; the skipped names are in `skipped_tools`. `ToolRegistry.register` itself is unchanged (last wins). The first provider wins when two servers collide.
- `PythonToolDefinition` gains `timeout_s`, `annotations` and `args_model`, and a loaded `@tool` function keeps them in `to_tool_definition()` (the loader dropped all three, so a declared timeout or read-only annotation was lost). `metadata` is not carried: `ToolDefinition` has no such field.
- The hybrid reasoning strategy's refinement back edge clears the nine hybrid body keys and `needs_refinement` on re-entry (new `HybridKeyClearer` handler and `HYBRID_CLEARED_KEYS` constant), so each refinement loop re-runs its extractions. Before, core extracted a key only while it was unset, so the loop re-ran no extraction and only repeated Pass 2. `hybrid_loop_count` is never cleared. Each loop now makes real model calls (about 5 bulk calls plus Pass-2 calls).
- `WorkflowEngine.cancel_workflow` on a RUNNING chain registers a cancel request before waiting for the instance lock, and the run loop honours it between steps (the step in flight finishes, later steps do not run). It returns `True`; before, it waited for the lock, found a terminal instance and returned `False`. A workflow that completes on its own before the loop sees the request still returns `False`.
- `timer_step(..., next_state="")` completes the workflow on expiry, like `wait_event_step(..., success_state="")`, instead of failing with `Invalid next state`.
- Monitor: `InstanceManager.destroy_instance` for a managed FSM caches the live conversations' snapshots and calls `api.close()` before it unregisters the global collector's handlers (in a `finally`), so `conversation_end` is recorded, the snapshot survives and OTEL spans end. `shutdown()` does the same for every managed FSM, so it now closes manager-owned FSM APIs (ending their live conversations). `attach_api` is unchanged: the caller's API is never closed.
- `fsm-llm` (plan 0068b055 step 12.13, live finding D3): a failure of the very first model call (`start_conversation`, e.g. a model that is not installed) now prints `Error: ...` on stderr and exits 1, like a failure of any later turn; it used to end with a raw Python traceback. Ctrl-C still exits 130.
- `fsm-llm-eval examples` (plan 0068b055 step 12.14, live finding D4): the example subprocess now runs with `PYTHONUNBUFFERED=1`, so a timed-out example's log holds the output it printed before the kill (its stdout pipe was block-buffered and lost, leaving an empty log; `run_example` documented the partial output). `TimeoutExpired` output is `bytes` on Python 3.11 and is decoded (a `str` is accepted too).
- `fsm-llm-eval examples` and `run` (plan 0068b055 step 12.15, live finding D5): SIGTERM (what `timeout(1)` and service managers send) during a run is now read like Ctrl-C: queued work is cancelled, `scorecard.md`/`results.json` (or `summary.md`) are written with `interrupted: true`, and the exit code is 130. It used to end the process with a run directory holding `logs/` only. The handler exists for the run only and the previous one is restored; it is not installed off the main thread or over an ignored (`SIG_IGN`) SIGTERM.
- Monitor (plan 0068b055 step 12.16, live finding D7): the OpenAPI document (`/openapi.json`, `/api/docs`) now reports the package version in `info.version` (it reported FastAPI's default `0.1.0`).
- litellm banner (plan 0068b055 step 12.20, D-023, live finding D3): importing `fsm_llm.llm` now sets `litellm.suppress_debug_info = True`. litellm prints a red "Give Feedback / Get Help" banner to stdout on every mapped provider exception while it is False (its default), which polluted the host's stdout and hid the real error in `fsm-llm-eval` logs. litellm's default is False, so an explicit False set before the import cannot be told from no setting; a host that wants the banner sets `litellm.suppress_debug_info = False` after importing `fsm_llm`.
- Reply cut by `max_tokens` (plan 0068b055 step 12.23, D-024, live finding D2): the provider's `finish_reason` is now read, in one place. `CompletionResponse` gains `finish_reason: str | None` and a `truncated` property (`finish_reason == "length"`); a WARNING (model, call type, the `max_tokens` in force, advice to raise `max_tokens` / `LLM_MAX_TOKENS`) is logged once per non-streamed cut reply (a streamed reply is not read). A bulk extraction whose reply was cut and is not valid JSON raises an `LLMResponseError` that says it was cut off at `max_tokens=N` (was only the JSON error); a field extraction left with no value says so in its `validation_error`; a classification parse error says so. Response generation no longer ships a half JSON envelope: a cut reply that opens with `{` or `[` and does not parse is replaced by a cut-off message (not the generic apology, so the one-shot retry does not repeat the same cut); a cut prose reply is shipped as is, and a reply with `finish_reason` `stop` is untouched. The committed completion-state result stays `{kind, text, calls}` (no `finish_reason`).

Known open after this pass, for a later pass:

- `API` stream methods are not thread-affine (F-AF-02): owner-agnostic guard versus enforced single-thread iteration is a lock design choice, and the `_lock -> conv_lock` order constrains it.
- A `PromptChainAgent` `validation_fn` that raises is not changed by this pass (F-AM-04).
- `APICallStep(failure_state="")` still FAILS the instance on an error. Completing on a failure would hide errors, and the empty failure route is the documented workflows rule. Only `timer_step` and `wait_event_step` treat `""` as completion.
- `InstanceManager.attach_api` leaves the attached API open on `destroy_instance` and `shutdown`: the API belongs to the caller and closing it is a policy choice.
- The MEDIUM, LOW and opportunity findings of this pass are open.

Known open after plan 0068b055 (each row is carried on purpose):

- F-DV-01: an over-cap prompt raises a typed error and the turn fails; truncating the largest context leaves (option B) is not built, because it silently loses data and its effect needs a live model.
- F-SU-02: `max_history_size` counts exchanges stored and messages shown; the honest `2 * max_history_size` unit is not built (it grows every default prompt from 5 to 10 messages and moves the eval baselines; needs an N>=3 live run).
- F-AF-23: `_conversation_start` stays naive local ISO text and `_timestamp`/`_transition_timestamp` stay epoch seconds, none persisted in sessions; UTC-aware values would break a naive-minus-aware subtraction in handlers and are not built (an owner product call).
- F-AM-05: `MetaBuilderConfig` refuses the eight unread `AgentConfig` fields; honouring them (copying `BaseAgent._create_api` logic) is the opposite design choice to the refusal and is not built.
- F-AM-06: `MetaDefaults.AGENT_MODEL` (`gpt-4o-mini`) is still a fixed template value, not `resolve_agent_model()`; changing it moves every built artifact and is an owner call.
- F-AP-16b: an executor for `ADaPTAgent(tools=)` on the PlanExecute `execute_step` pattern is not built (a new capability; whether a small model picks the right tool inside a decompose loop needs a live check), so the two examples that score `tools_called` stay unmet.
- F-AB-05b: a typed schema for a lone `dict[...]` `@tool` parameter is not built (unwrapping it is ambiguous and is a product choice); the model sees an empty object schema and a model that names the parameter gets it double-wrapped.
- F-AR-05 answer quality: the live check (Changed, step 7) shows the silent Orchestrator and ADaPT states did not lower the 0-4 score, but the scorer saturates and two of the seven examples time out in both arms, so whether the answers are as good is unmeasured beyond that; a measure needs graded answers, not the heuristic.
- Model speed in the agents examples: on `ollama_chat/bonsai-2:27b` (about 3 to 9 s per extraction call and 20 to 27 s for a 1000-token reply, one GPU, `OLLAMA_NUM_PARALLEL=1`) the agents category timed out in the live run only under queueing: with `--workers 2` the second worker's requests wait for the first's and the example timeout counts the wait, so 14 examples (`legal_document_review`, `pipeline_review`, `supply_chain_optimizer` and 11 more) timed out and all 14 finished alone with `--workers 1` in 11 to 307 s. `fsm-llm-eval` guidance now says to use one worker on a one-slot server (EVALUATE.md, `--workers`). `plan_execute`, `plan_execute_recovery` and `orchestrator_specialist` were slow, not looping, when run alone (the last hit its own 300 s agent budget) and were not re-run in the second session; `supply_chain_optimizer` run alone needs 306.6 s against its own 300 s budget (the orchestrator runs its three specialists one after another). The harness timeout table was raised only for the 23 examples that tied their agent budget (step 12.21); whether to raise more of it for this model is an owner choice.
- Reply cut off at `max_tokens` (steps 12.23 and 12.28, D-024, D-025): detected and carried into `AgentResult` (`stop_reason="truncated"`, `success=False`) for the LAST turn of every pattern on `_standard_run` and ADaPT when the cut text is the answer's (a whole answer read from a context key is not failed by an unused cut reply, step 12.33, D-027), and REWOO no longer runs an empty plan. Still open: (a) `native_fc` ends the loop `malformed` when a cut tool call has half arguments and then runs its `conclude` turn, which clears the record, so that run is still not flagged (the loop would have to keep the record across turns); (b) a truncated meta-builder artifact is not flagged (`MetaBuilderAgent` and the artifact builders run their own `converse` driver, not `_run_conversation_loop`); (c) `SelfConsistencyAgent` samples run their own `converse`, so a cut sample still votes; (d) a streamed reply carries no `finish_reason`, so `run_stream`/`converse_stream` never flag; (e) a cut in an EARLIER turn is not flagged even when its output (a cut field kept at reduced confidence) is what the answer was built from; (f) REWOO still makes its `solve` call after a cut plan, and does not retry at a higher `max_tokens`.
- `examples/agents/regulatory_compliance` prints `Checker passed: True` next to `Success: False` and a low quality score: the pass is forced at the revision cap, which is the documented design of `MakerCheckerForcePass` (`AgentResult.success=False` and `stop_reason=forced_pass` tell the truth; the frozen example prints the raw context key). `examples/` are frozen, so the output is not edited.
- `agents/react_hitl_combined`: the harness task names the gated tool (step 12.31) and the live model (`ollama_chat/bonsai-2:27b`) then asks for approval and the `y` answers it, 3 of 3 live runs (each 33 to 34 s, one `publish_report` call, `Approve this action? (y/n)` then `APPROVED`). That is 3 runs of one prompt: a model that skips the gated tool still turns the `y` into a second task, and the score 4 does not tell the two cases apart.
- Claimed side effect without the tool call (plan 0068b055 review pass 3 note 10, live-runs-3 D2): on `ollama_chat/bonsai-2:27b`, with a task that did not name `publish_report`, the first answer of 3 of 3 live `agents/react_hitl_combined` runs said the report had been published and no `publish_report` call was made, so no approval was asked. Step 12.31 changed the harness task to name the tool, which makes the example exercise HITL; it records the defect's cause (the model, not the repository) but does not remove it. No library check can tell a true claim from a false one in free text, and the example's own `Success`/`Error` lines cannot see it, so a HITL user whose model answers without calling a gated tool gets a confident answer and no approval prompt. Carried: model behaviour on this model, observable only by comparing the answer with `AgentResult.trace.tool_calls`.
- A bulk extraction cut off at `max_tokens` is not recorded in the cut-off record (plan 0068b055 review pass 3 note 2, step 12.35, D-030). `extract_bulk_data` raises `LLMResponseError` ("cut off at max_tokens=N") for a cut reply that is not valid JSON, but `MessagePipeline._bulk_extract_from_instructions` catches every exception, logs a WARNING and returns the `_BulkFailed` marker, so the turn does not fail: it goes on with `extraction_failed=True`. A cut reply that still parses is committed with no flag. The 12.28 line above names the Pass-2 reply, field extraction and completion-state call only. Recording it needs a structured cut signal out of `extract_bulk_data` (a field on `DataExtractionResponse` or a typed exception); matching the message text is the shape heuristic D-022 forbids. Agent FSMs use typed per-field extractions, so only core FSMs with `extraction_instructions` and no field configs (and `use_classification=True` agents) are affected.
- A POSITIONAL mapping parameter beside other parameters (for example `fn(params: dict[str, Any], note="x")`) still receives the first stray value when a model sends a flat object whose keys name no parameter: `params` gets that value (a scalar), not the object (plan 0068b055 step 12.35, D-030). The keyword-only case is refused since 12.35; widening it to positional parameters changes the behaviour 13 HITL-executor tests (`test_hitl_security.py`) depend on, so it is a separate decision: refuse it (a failed call the model can correct) or pass the whole flat object as the dict (the bare-dict rule, which is for a lone positional parameter only).
- The 27 examples that run on the default 300 s agent budget and have a harness timeout below it (120 to 240 s) keep their harness timeouts; the harness is their hang detector and fires first, which is the intended order, so they are not raised.
- `examples/agents/plan_execute/run.py` prints a `Plan steps (N):` header and then no steps (it prints only dict steps); `examples/` are frozen baselines, so it is not edited.
- 73 examples print `export LLM_MODEL=ollama_chat/qwen3.5:4b` in the hint they show when `LLM_MODEL` is missing (for example `examples/workflows/workflow_agent_loop/run.py`), while the package has no default model. Seen only when an example is run directly without `LLM_MODEL` (`fsm-llm-eval` sets it); `examples/` are frozen, so the hint is not edited.
- The `skills-disclosure` bench registration (B0 and B1) is void since the `constants.py` edit of this plan (its manifests pin that file's sha256): `register` and `run` refuse, and the bench was never run. It is not re-registered; re-registering on the new default model is a separate decision.
- A stale `.env` can no longer choose the model: importing `fsm_llm` loads no `.env` and there is no default (see the BREAKING entries of this release). Still open: a host that imports litellm before `fsm_llm` has already loaded the file, which `llm.py` cannot undo.

Known open after plan 2cefeb61 (runtime FSM synthesis; each row is carried on purpose):

- Bench status: `scripts/synthesis_bench.py` ran blocks `s0`, `s1` and `s2` on `ollama_chat/bonsai-2:27b` (every role and arm, no strong/cheap split). First trial of 16 tasks, S / H / R: s0 6 / 8 / 14 (INCONCLUSIVE, H under 10), s1 15 / 15 / 16 and s2 15 / 16 / 16 (INCONCLUSIVE, ceiling clause). The 16 tasks are saturated for this model, s1 and s2 are in-sample after fixes chosen from earlier failures, and a fourth block was not run, so "a compiled procedure beats an in-context agent" stays an open question. The two fixes after s2 (D-108, D-109) were checked by smoke only.
- `verified` is the strongest meaning the library has for a procedure the host did not review: static gates, a bounded simulation over scenarios a model wrote, no proof. `trusted` needs a human approver.
- A procedure cannot call tools (the runner refuses a `completion` state) and a nested synthesize state (depth 2) works only for a procedure that returns one of its inputs; both are limits, not options.
- Library statistics are last-writer-wins across processes, and a corrupt file blocks retrieval for the whole library until a person fixes it.

## [0.12.0] - 2026-10-02

Two sets of changes. The entries marked "Step driver" come from plan `07ad3f8c`
(2026-09-30 to 2026-10-01): core gained a message-free step (`API.advance`) and a
bounded run loop (`API.run_until_terminal`), every agent pattern and the harness now
run on them with no synthetic "Continue." message, a silent state says nothing, and
the legacy names listed under Removed are gone with no deprecation period. The other
agents entries come from the 2026-09-29 audit of `fsm_llm.agents` (Track A); its
record, with each finding's status and the deferred Track B work, is
`docs/agents_roadmap.md`.

The entries marked "One LLM layer" come from plan `944e2692` (2026-10-01 to
2026-10-02), the second half of the step driver work. Core gained one request
primitive for tool calling, plain and structured completion
(`LLMInterface.complete`), per-instance usage counters, an embedder
(`LiteLLMEmbedder`) and a general tool-calling and structured-output state (the
optional `completion` state field). Every caller that used to talk to litellm by
itself now goes through that layer: the classifier (on the conversation's own
interface), the LLM judge, semantic memory and semantic tool retrieval, the
meta-builder and `native_fc`; `litellm` is imported only in `src/fsm_llm/llm.py`.
`NativeFunctionCallingReactAgent` and `MetaBuilderAgent` are FSM definitions run by
core, ToolSpec (tool annotations, exact schemas, an enforced timeout) is ported, and
the reasoning engine runs on core's message-free run loop with no synthetic message.

Measured for One LLM layer on `ollama_chat/qwen3.5:4b` (Ollama digest `2a654d98`,
2026-10-01 and 2026-10-02, pre-registered fixed-n blocks): agent bench block
`agents-react/B2`, arm `fsm_toolcall` (`native_fc` on core), 37/38 first-trial
pass@1 at 2.39 LLM calls per task with 0 envelope leaks, against B0 `native_fc`
37/38 at 2.39 (counted by a different meter; every difference is listed in
`scripts/bench_data/README.md`); harness block `l4-execute-write/B2`, arm
`native_fsm`, 40/40 verified writes against B1 `native` 40/40. Reasoning probe (12
pre-registered problems, 2 trials each): the `e1f63a9` baseline passed 22/24 at
63.17 LLM calls per solve with 2 null solutions; the engine on core's run loop that
ships passed 24/24 at 49.71 calls with none; a further variant with typed per-key
fields and silent states passed 18/24 at 28.50 calls, failed its pre-registered rule
(logic problems answered with a bare "True"/"False", an open-ended answer with one
item) and was reverted. Full examples eval (101 examples, 4 workers, same day, Ollama
shared with another loaded model): 386/404 (95.5%) against 367/404 (90.8%) for
`e1f63a9`, 0 F-CODE, 0 envelope leaks; the one example that dropped
(`agents/reasoning_stacking`, a 180 s timeout) scores 4 in about 45 s run alone at both
commits (`EVALUATE.md` Run 009).

Measured on `ollama_chat/qwen3.5:4b` (2026-10-01, `docs/agents_roadmap.md`,
`EVALUATE.md` Runs 007 and 008): agent bench block `agents-react/B1` 32/38 first-trial
pass@1 at 10.97 LLM calls per task against B0's 28/38 at 11.5, 0 envelope leaks,
0 "Continue." in answers; full examples evaluation 388/404 against 391/404 for
`d4b1626` on the same day; agents category 177/192 after the PlanExecute planner fix.

### Added

- Builder standard (plan `89b03f61`): `BuildError(FSMError, ValueError)` in core,
  exported from `fsm_llm`, with `.errors` (the problem strings) and the domain or
  constructor error chained as `__cause__`. A `build()` failure that comes from the
  builder's own rules or from the constructor or validation it calls is a
  `BuildError` chained from the cause; a programming error such as `None` where an
  object is required may still raise its own error.
- Builder standard: `APIBuilder` and `FSMManagerBuilder` (`fsm_llm`),
  `ConfiguredAgentBuilder` (`fsm_llm.agents`, any `create_agent` pattern) and
  `HarnessAgentBuilder` (`fsm_llm.harness`). Each has `set_`/`add_` mutators that
  return the builder and only record, and a `build()` that passes only the values that
  were set to the real constructor, copies what the builder owns and raises
  `BuildError`. Open-ended input has its own setter (`set_llm_option`, `set_option`,
  `set_api_option`) that passes names through unfiltered, except that `build()`
  refuses a name that repeats a named constructor parameter (read from the
  signature) and says which typed setter to use. `ConfiguredAgentBuilder` and
  `HarnessAgentBuilder` copy the `AgentConfig` shallowly: callables inside it stay
  shared. `ConfiguredAgentBuilder.set_option` also refuses `pattern`, `tools`,
  `config` and `system_prompt`. The convention is written once in
  `src/fsm_llm/CLAUDE.md` (Invariants) and `docs/api_reference.md`.
- One LLM layer, core: `LLMInterface.complete(request: CompletionRequest) ->
  CompletionResponse`, the one request primitive for tool calling, plain completion
  and structured output. It is not abstract: the base method raises
  `NotImplementedError`, so existing `LLMInterface` subclasses keep working;
  `LiteLLMInterface.complete` builds the request with the same builder as every other
  call and sends it through the one send path. Every failure is `LLMResponseError`
  (provider error chained, empty reply, unreadable reply shape); a provider
  "malformed tool call" error on a request with tools is data, not an error.
- One LLM layer, core: `CompletionRequest{messages, tools, tool_choice,
  response_format, temperature, max_tokens, call_type}` (frozen, unknown fields
  refused; `tools` and `response_format` never together, `tool_choice` only with
  `tools`), `CompletionResponse{kind, text, calls}` (frozen; `kind` is `"calls"`,
  `"final"` or `"malformed"`) and `ModelToolCall{id, name, arguments: dict}`, all
  exported from `fsm_llm`. A reply whose tool call has no id, no name, or arguments
  that are not a JSON object is `kind="malformed"` with no calls, so none of that
  turn's calls runs; a reply with text and calls is `kind="calls"` with the text kept.
- One LLM layer, core: tool-transcript helpers in `fsm_llm.llm`: `tool_exchange(content,
  calls, results)` (exported from `fsm_llm`; the paired assistant tool-call message
  and one `tool` message per call), `check_tool_transcript(messages)` (refuses an
  unpaired or malformed transcript), `is_malformed_tool_call_error`,
  `decode_tool_arguments`, `implements_complete`, `interface_model`, and
  `constants.MALFORMED_TOOL_CALL_MARKERS`.
- One LLM layer, core: usage counters. `LiteLLMInterface.usage() -> LLMUsage` and
  `reset_usage()` (returns the snapshot it cleared). Every provider call of an instance
  is counted once on that instance, per kind (`generate`, `extract`, `classify`,
  `stream`, `complete`; `constants.USAGE_KIND_*` and `USAGE_KIND_BY_CALL_TYPE`): calls,
  errors, replies without usage, and
  prompt, completion and total tokens. A call that raises counts as a call and an
  error; a streamed call counts with usage missing (also when it fails mid-way).
  `LLMUsage(LLMCallCounts)` adds `by_kind`; both are frozen and exported from
  `fsm_llm`. There is no process-wide counter: a reader meters the interface it owns
  (inject it with `llm_interface=`).
- One LLM layer, core: `LiteLLMEmbedder(model, *, api_key=None, timeout=120.0,
  retries=0, **kwargs)` with `embed(texts) -> list[list[float]]` (one provider request
  for all texts, `[]` and no request for no texts), `usage()` and `reset_usage()`
  (kind `embed`), exported from `fsm_llm`. It has its own model and connection
  settings, never the chat model's. A reply with a missing, uneven, empty or
  non-finite vector raises `LLMResponseError`. Reserved call kwargs
  (`constants.RESERVED_EMBEDDING_CALL_KWARGS`) are ignored with a WARNING.
- One LLM layer, core: the optional `completion` state field
  (`CompletionStateConfig{tools, tool_choice, response_format_key, instructions,
  messages_key="_completion_messages", result_key="completion_result"}`, exported from
  `fsm_llm`; FSM definition format stays v4.1). A completion state's Pass 1 is one
  `complete` call over `[system(instructions)] + context[messages_key]`: a tool-calling
  turn (`tools`) or a structured turn (`response_format_key` names the internal context
  key holding the response format), never both. No prompt builder text, no history and
  no neutral user turn are added. The result `{kind, text, calls}` is written to the
  public `result_key` (handler-only for extraction) and transitions read
  `<result_key>.kind`; the call is made only while that key is unset. The transcript
  is a consumer-owned internal context list, never the conversation history; core runs
  no tool. A transcript that `check_tool_transcript` refuses, or a tool-calling turn
  whose transcript holds no user message, raises `LLMResponseError` before anything is
  sent and the turn rolls back. A completion state may not also declare extraction
  fields, `classification_extractions`, `required_context_keys` or
  `extraction_instructions`, nor be terminal; `fsm-llm-validate` warns when its
  transitions do not cover every result kind. New constants
  `DEFAULT_COMPLETION_MESSAGES_KEY`, `DEFAULT_COMPLETION_RESULT_KEY`,
  `COMPLETION_TOOL_CHOICE_KEYWORDS`.
- One LLM layer, core: `Classifier(schema, model=None, *, llm=None, api_key=None,
  config=None, **llm_kwargs)` and `HierarchicalClassifier(..., llm=None)`: with `llm`
  every classifier request goes to that interface (an `api_key`, extra kwargs, a
  different `model`, or an object that is not an `LLMInterface` implementing
  `complete` raise `ValueError`); without it the classifier builds its own interface
  as before.
- One LLM layer, core: `API.run_until_terminal(..., seconds_exempt_states=())` and the
  stream form: a round whose current state is in the set runs even when the seconds
  budget is spent (the steps budget still applies). Ids are checked against the
  running definition (`ValueError` for an unknown id or a bare `str`).
- One LLM layer, core: `ResponseGenerationRequest.temperature: float | None` (a
  per-call Pass-2 temperature, set by the pipeline from the internal context key
  `constants.CONTEXT_KEY_RESPONSE_TEMPERATURE`, `"_response_temperature"`; a custom
  interface should honour it) and `fsm_llm.api.llm_settings_for(api_kwargs,
  **settings)` (a subpackage's own LLM settings, only when no interface is injected).
- One LLM layer, core: `typed_field_extraction(field_name, field_type, instructions,
  *, context_keys, required=True)` and `clear_keys_on_entry(keys, *, state, name=None,
  priority=100)`, exported from `fsm_llm` (moved from agents); also
  `definitions.TypedFieldType`, `handlers.clear_keys_delta(keys, context)`,
  `definitions.checked_key_names(keys, *, argument)` (refuses `None`, a bare `str` and
  non-`str` members), `constants.EXTRACTION_ENVELOPE_KEYS` and
  `constants.FIELD_PROMPT_CONTEXT_LABEL`.
- One LLM layer, agents: ToolSpec. `ToolAnnotations{read_only, destructive,
  idempotent, open_world}` (exported; `retry_safe` is True for an explicit
  `idempotent` or `read_only`), `ToolDefinition.annotations`, `.timeout_s` (finite, up
  to `Defaults.MAX_TOOL_TIMEOUT_S`) and `.args_model` (the pydantic model of the
  function's parameters, giving exact JSON schemas); `@tool(annotations=,
  timeout_s=)` and `register_function(..., *, annotations=, timeout_s=)`; MCP tool
  annotations are mapped onto `ToolAnnotations`. `tools.tool_parameters_schema(tool)`
  is the one schema source for native tool schemas and prompt descriptions;
  `tools.schema_types(prop, *, root=None)` renders `anyOf`, type lists and local
  `$ref`s (an `Optional[int]` parameter reads `integer or null`, a `str` Enum
  `string`; an unresolvable reference reads `any`).
- One LLM layer, agents: `ToolResult.timed_out`, `ToolResult.status` and
  `ToolResult.observation` (the one place the model-facing text is built:
  `[TOOL FAILED]`, `[TOOL OUTCOME UNKNOWN]` for a timed-out call, or the summary);
  `constants.ToolRunStatus` (`success`, `failed`, `unknown`); every executor trace
  entry carries `tool_status` (AgentHandlers, ParallelReact, `native_fc`, REWOO; REWOO
  `evidence_status` entries too); `tools.refuse_execute_without_gated(registry)`;
  `ErrorMessages.PROMPT_CACHE_WITH_INTERFACE`.
- One LLM layer, agents: `SemanticToolRegistry(..., *, embed_fn=None)`, the same
  per-text callable seam `SemanticMemoryStore` has.
- One LLM layer, agents: the FSM definitions behind the two rebuilt agents and their
  constants: `fsm_definitions.build_native_fc_fsm`, `NativeFCStates`,
  `NativeFCContextKeys`, `NativeFCHandlerNames`; `fsm_definitions.build_meta_builder_fsm`,
  `MetaBuilderStates`, `MetaContextKeys`, `MetaBuildOutcome`, `MetaHandlerNames` and the
  `META_*` constants (`META_BUILD_PROMPT`, the keyword tables, intents and key lists).
  `BaseAgent._step_ceiling(max_iterations)` is the overridable step-ceiling hook.
- One LLM layer, reasoning: `ContextKeys.REASONING_PUSH_PENDING`, `NEEDS_REFINEMENT`,
  `HYBRID_LOOP_COUNT`; `Defaults.MAX_SOLVE_STEPS` (170) and `MAX_HYBRID_LOOPS` (2);
  `HandlerNames.RETRY_KEY_CLEARER` and `HYBRID_LOOP_COUNTER`;
  `ReasoningHandlers.count_hybrid_loop`; constants `RETRY_CLEARED_KEYS`,
  `ORCHESTRATOR_HANDLER_ONLY_KEYS`, `HYBRID_EVALUATION_STATE`, `SOLVE_DRIVER_KEYS`.
- One LLM layer, bench: `scripts/agents_bench.py` meter `wrapper_version` "2" (reads
  the injected interface's `usage()`; "1" stays for recounting B0/B1), arm
  `fsm_toolcall`, manifest disclosures (`llm_request`, `agent_class`, `run_cap`,
  `tool_schemas_sha256`, `first_request`) and the recorded block `agents-react/B2`.
  `scripts/harness_bench.py`: `register`, arm `native_fsm`, `report --blocks` and
  `--pair BLOCK/ARM:BLOCK/ARM`, the shared request-capture and digest helpers, and the
  recorded block `l4-execute-write/B2`.
- Step driver, core: `API.advance(conversation_id) -> AdvanceResult` runs one turn of
  the current state with no user message: the same turn body as `converse` (one turn
  at a time per conversation, rollback on failure, every handler timing, extraction,
  transition evaluation, Pass 2 from the post-transition state, session auto-save),
  resolved on the top of the FSM stack. Nothing is added to the history for a user.
  It raises `FSMError` on a terminal state. Also `FSMManager.advance` and
  `MessagePipeline.advance`.
- Step driver, core: `API.advance_stream(conversation_id) -> Iterator[str]`, the
  streamed form (reply text only; read the outcome afterwards with
  `get_current_state` / `has_conversation_ended`). Also `FSMManager.advance_stream`
  and `MessagePipeline.advance_stream`.
- Step driver, core: `API.run_until_terminal(conversation_id, *, max_steps,
  max_seconds=None, before_step=None) -> tuple[AdvanceResult, ...]` and
  `API.run_until_terminal_stream(...) -> Iterator[str]`. Each round asks whether the
  conversation (top of the stack) has ended, checks the seconds and steps budgets,
  calls `before_step(n)`, asks again whether the conversation ended and re-checks the
  seconds budget (the hook may end it or use up the time), then runs one step. Budgets are checked only between steps. `max_steps` must be an int of at
  least 1 and `max_seconds` `None` or a positive number (`ValueError` for a bool, a
  non-number, NaN or a value out of range).
- Step driver, core: `AdvanceResult{state_before, state_after, transition_outcome,
  response, ended}` (frozen; `response` is `None` for a silent state), exported from
  `fsm_llm`.
- Step driver, core: `RunBudgetExceededError(FSMError)` with `budget` (`"steps"` or
  `"seconds"`), `limit` and `steps_done`, exported from `fsm_llm`. The steps already
  run are kept; the error does not carry their results (read the history and the
  current state).
- Step driver, core: FSM graph data and export, exported from `fsm_llm`:
  `build_fsm_graph(fsm) -> FSMGraph{name, initial_state, nodes, edges}` (frozen
  `FSMGraphNode`, `FSMGraphEdge`; `ValueError` for a definition whose initial state or
  a transition target is missing or whose shape is malformed), `to_mermaid(graph)`
  (Mermaid `stateDiagram-v2`) and `to_dot(graph)` (Graphviz DOT).
  `fsm-llm-visualize --format ascii|mermaid|dot` (default `ascii`; `--style` applies
  to ASCII only) and `visualize_fsm_from_file(..., *, output_format="ascii")`. The
  monitor's FSM visualizer reads the same graph data.
- Step driver, core: `constants.NEUTRAL_USER_TURN` ("Proceed according to the
  instructions above.") is the provider user turn sent when there is no user message
  (a step, the greeting, a context-only classifier call); `constants.EMPTY_USER_MESSAGE_TURN`
  ("(empty message)") is sent when the user's message is empty or whitespace, so an
  empty message is never read as an instruction to proceed. Both are filled in one
  place, the request builder of `LiteLLMInterface`.
- Step driver, core: `LiteLLMInterface.complete_structured(system_prompt,
  user_message, *, json_schema, schema_name)`, the structured call the `Classifier`
  sent through (removed again in this cycle: One LLM layer replaced it with
  `LLMInterface.complete`, see Removed); `DataExtractionPromptBuilder.build_extraction_source_sections(
  instance, scoped_context)`, the context and history sections of the no-message bulk
  extraction prompt.
- Step driver, agents: `StopReason.ENDED` (`"ended"`, a forced stop): the run's
  conversation was closed from outside before a terminal state; the run returns
  `success=False` with its last output.
- Step driver, agents: `ContextKeys.REFUSED_ACTIONS` (`refused_actions`): one sentence
  per gated call the human approver refused and that did not run ("`<tool>(<redacted
  params>)`: was refused by the human approver and was not performed."). It is a run
  output (caller context cannot set it), framework-only (no extraction writes it),
  shown to the conclude prompt of approval-gated FSMs and returned in
  `final_context`. `build_conclude_response_instructions(*, refused_actions=False)`
  adds the sentence that tells the model to report those actions as not performed.
- Step driver, agents: `handlers.call_label(tool_name, tool_input)` (the one
  `<tool>(<redacted normalised params>)` label shared by the trace `action`, the
  refusal record and the ran-check), `handlers.call_ran(trace, tool_name,
  tool_input)` and `handlers.refusal_record(tool_name, tool_input)`.
- Step driver, agents: `ContextKeys.OPERATOR` (`operator`), the ADaPT `decompose`
  typed field (AND/OR; null or other text is AND), and `ReflexionStates.AWAIT_APPROVAL`
  (the shared approval state). Every `*States` class now names every state of its
  FSM and the FSM builders read the constants.
- Step driver, monitor: `InstanceManager.attach_api(api)` shows an `API` you created
  (events and conversations). One API is attached at a time; attaching another
  switches the previous one's handlers off; a failed handler registration raises
  `MonitorConnectionError` and keeps the previous API attached.
- Step driver, bench: `scripts/agents_bench.py` (`register`, `run`, `report`,
  `list-tasks`; 38 deterministic tool tasks in 7 categories with non-LLM graders) and
  `tests/test_agents_bench.py`. Recorded blocks under `scripts/bench_data/agents-react/`:
  `B0` (arms `legacy` and `native_fc`, the agents code of `d4b1626`, copied byte for
  byte) and `B1` (arm `fsm_advance`, the code of this release, pre-registered pass
  rule met). The B0 arm labels are retired: new rows cannot carry them.
- Agents: `AgentResult.stop_reason` (default `None`) and the `StopReason` constants
  (exported from `fsm_llm.agents`): `answered`, `evidence`, `max_iterations`,
  `forced_pass`, `stalled`, `verification_failed`, `no_result`, `gate_failed`.
  `AgentServer` `/invoke` and `/stream` responses include `stop_reason`.
- Agents: `AgentServer(max_concurrent=8)`. A request that finds every slot taken gets
  503; a slot is held until the agent thread ends, even after a 504.
- Agents: context keys `agent_feedback` (executor warnings and HITL denials the next
  think turn reads) and `forced_stop_reason` (written only by a forcing handler).
  Both are run-owned: caller context cannot set them.
- Agents: `Defaults.ADAPT_MAX_SUBTASKS` (8) caps one ADaPT decomposition.
- Tests: `tests.conftest.PromptGroundedLLM`, a fake LLM that answers a field only when
  the prompt contains its evidence, and `block_network`, autoused by the agents and
  meta suites so they cannot open a TCP connection.
- `docs/agents_roadmap.md`: the agents audit record and deferred roadmap.

### Changed

- Install hints: runtime install messages (lazy extension imports, monitor, OTEL,
  MCP, A2A, harness CLI) and the extras docstrings now name a clone install, because
  the `fsm-llm` PyPI name belongs to another project: `pip install -e ".[extra]"`
  where the extra adds packages (`monitor`, `otel`, `mcp`, `a2a`, `harness`) and
  `pip install -e .` where it adds none (`workflows`, `reasoning`, `agents`, `eval`,
  and the lazy-import hints in the core package).

- Builder standard: `fsm_llm.workflows.WorkflowBuilder.build()` has no `validate`
  argument (removed, not deprecated): it always validates, returns a fresh,
  isolated `WorkflowDefinition` (a later builder call no longer changes it) and
  raises `BuildError` chained from `WorkflowValidationError` or
  `WorkflowDefinitionError`. A caller that caught `WorkflowValidationError` from
  `build()` now catches `BuildError` and reads `__cause__`. `set_initial_step`
  keeps its step in call order (the last call decides the initial step, earlier
  ones stay as steps) and a malformed step or definition field raises `BuildError`
  at `build()`.
- Builder standard: `AgentGraphBuilder.add_node` with a name already used is now a
  `build()` error instead of a silent overwrite; `HandlerBuilder.build()` refuses a
  non-callable execution function. The graph, handler, workflow and meta builders
  raise `BuildError` (still a `ValueError`, so existing `except ValueError` keeps
  working). `HandlerBuilder` keeps its `at`/`on_state`/`when`/`do` vocabulary.
- Builder standard: the meta artifact builders' mutators return the builder, so
  calls chain, and the warnings they used to return as a `list[str]` are read with
  `take_warnings()` (returns and clears). A new `build()` runs `validate_complete()`
  and raises `BuildError`, else returns a deep copy of `to_dict()`. The call-time
  `BuilderError` refusals (empty id, unknown type, missing source) stay, as does
  `update_state`. `BuilderError` is now also a `BuildError`.
- Agents (fix): the meta artifact tools (`create_fsm_tools` and the workflow and
  agent siblings) serialise each call per builder, so parallel tool calls no longer
  steal each other's warnings, and warnings left by direct mutator calls no longer
  leak into the next tool reply.

- One LLM layer, core: the classifier of an AMBIGUOUS transition and of a
  `classification_extractions` entry sends its request to the conversation's own
  `LLMInterface` (`complete`), so a custom `llm_interface` given to `API` is used and
  its timeout, retries and kwargs apply (the 120 s classifier bound now comes from the
  interface's timeout). A private interface is built only when the entry names a
  different `model`, and it inherits no connection kwargs. The classifier cache key is
  the content plus the interface's identity. The pipeline's "no model available"
  skip and the `InvalidTransitionError` at a tie are gone.
- One LLM layer, core: classifier failures. Every failure of the classifier's call is
  `ClassificationError` (an unreadable reply was `ClassificationResponseError`, a dict
  `choices` leaked `KeyError`); a reply text that does not parse stays
  `ClassificationResponseError`. The turn soft-fails (stays, WARNING) only on
  `ClassificationError`: any other exception raised by a custom interface's
  `complete`, and a classifier that cannot be built (a bad `prompt_config`), now fail
  the turn instead of a silent stay. `API` (and `push_fsm`) raise `ValueError` when a
  definition's `classification_extractions` classify through an interface that does
  not implement `complete`; a tie with such an interface still soft-fails each turn.
- One LLM layer, core: `API(llm_interface=...)` raises `ValueError` when it is given any
  other LLM setting beside the interface (`api_key`, a non-None `temperature` or
  `max_tokens`, a `model` other than the interface's own, `seed`, `caching` or any
  other kwarg); they used to be dropped silently.
- One LLM layer, core: request building. `tools`, `tool_choice`, `functions`,
  `function_call` and `n` join `constants.RESERVED_LLM_CALL_KWARGS` (ignored with a
  WARNING when given to a constructor). The empty-turn filler touches only `user`
  messages, so an assistant tool-call message keeps `content: None`. On Ollama a
  `complete` call that sends a `response_format` runs at temperature 0 unless its call
  type is `response_generation`. Every Pass-2 request carries the conversation's
  temperature (`ResponseGenerationRequest.temperature`).
- One LLM layer, core: `fsm_id` hashes `model_dump(exclude_defaults=True)`, so adding an
  optional field to the models no longer changes every id and an explicit default
  hashes like its omission. Every FSM's id changes once against 0.11.0 (restoring a
  session saved under 0.11.0 logs the existing `fsm_id` mismatch WARNING).
- One LLM layer, core: `run_until_terminal` and its stream form call `before_step` once
  when the top of the stack is an ended pushed FSM (stack depth above 1), so the hook
  can pop it and the run continues on the parent; if it does not pop, the run returns
  as before. That call is not a step and is not gated by the budgets; a hook tells it
  from a normal call with `has_conversation_ended`. A pushed child FSM does not
  inherit a completion result key.
- One LLM layer, agents: the LLM judge (`default_llm_judge`) sends through a core
  `LiteLLMInterface` (temperature 0): on Ollama thinking is off; requests carry a
  120 s timeout and `max_tokens` 1000; a provider failure is still `EvaluationError`,
  now chained from core's `LLMResponseError`. `complete_fn=` stays its extension point.
- One LLM layer, agents: `SemanticMemoryStore` and `SemanticToolRegistry` embed through
  a `LiteLLMEmbedder` by default; `rebuild_embeddings` makes one batch request
  (all-or-nothing on failure, one WARNING); a registry with an empty
  `embedding_model` and no `embed_fn` raises `ValueError` at construction.
- One LLM layer, agents: `execute(tool_call, *, gated=False)` on `ToolRegistry`,
  `CachingToolRegistry`, `RetryingToolRegistry` and ReasoningReact's registry view;
  the HITL executor passes `gated=True` for a granted call. A third-party `execute`
  override must accept `gated`: the ReAct family, Reflexion and PlanExecute refuse one
  without it at the start of `run()`, ReasoningReact at construction (`AgentError`);
  REWOO, ParallelReact and `native_fc` never pass it. `RetryingToolRegistry` retries
  only a `retry_safe` tool (annotated `idempotent` or `read_only`) and never a gated
  call or a `requires_approval` tool, so unannotated tools are no longer retried.
  `CachingToolRegistry` neither serves nor stores a gated call.
- One LLM layer, agents: `ToolDefinition.timeout_s` is enforced by `ToolRegistry.execute`
  (a worker thread; the registry lock is never held across the call). A timed-out call
  returns a failed `ToolResult` with `timed_out=True` and `tool_status` `unknown`; the
  Python tool keeps running and its late result is discarded with a WARNING. PlanExecute
  sends an unknown-outcome step to synthesis (no replan, no next step) and reports
  `success=False, stop_reason="no_result"`; a ParallelReact batch with a timed-out call
  is `unknown`.
- One LLM layer, agents: prompt-mode tool descriptions read the same exact schema as
  native tool schemas (`Optional[int]` is `integer or null`, an Enum its value type);
  tools with a `dict` parameter gain `additionalProperties: true`, and harness tools
  with a defaulted path gain `"default": "."` in their schemas.
- One LLM layer, agents: `enable_prompt_cache=True` with an injected `llm_interface`
  raises `AgentError` at construction. `AgentConfig.model`, `temperature` and
  `max_tokens` configure the interface core builds and are not applied to an injected
  one. SelfConsistency sets each sample's temperature on the request, so it works with
  an injected interface.
- One LLM layer, agents: `NativeFunctionCallingReactAgent` runs as an FSM on core
  (`build_native_fc_fsm`: `call_model` and the optional `force_final` and `repair`
  completion states, `run_tools`, `conclude`) through `BaseAgent._standard_run`. The
  model-visible requests are byte-identical to the old loop on 24 recorded scenarios
  (transport keys aside). Changes: each request carries the interface timeout (120 s
  by default, none before); `initial_context` goes through `_init_context` (run-output
  keys and the approval grant stripped; the model still sees only the system message
  and transcript) and `final_context` is filtered like the other patterns; other
  `api_kwargs` reach `API`; `iterations_used` counts loop model turns; a
  whitespace-only answer is `no_result`; the third positional constructor argument is
  `system_policy` (it was `complete_fn`), and `system_policy` must be a `str` or `None`
  (`TypeError`); a forced turn with an unrunnable call runs none of its calls; an empty
  task is sent as `EMPTY_USER_MESSAGE_TURN`; a call with an empty tool name makes the
  turn malformed (it ran as an unknown tool); a provider outage is `AgentError` chained
  from core's `LLMResponseError`. The wall clock is checked only before each loop model
  turn, so the forced and repair turns still run after the deadline, as before. The
  test and caller seam is `llm_interface=`.
- One LLM layer, agents: `MetaBuilderAgent` runs an FSM through core (`classify`,
  `collect`, `build`, `build_failed`, `done`): the artifact type is a
  `classification_extractions` field (intents fsm, workflow, agent and a fallback
  `unknown`), collect replies are core Pass 2, and the build is a structured
  completion state (temperature 0 and core's Ollama preparation). Every model call
  goes to the conversation interface, so `llm_interface=` reaches all of them. Public
  API unchanged; behaviour shifts: the reply of `start(message)` is written by the
  model (canned only for `start("")`), and every collect reply ends with "Say 'build
  it' when you're ready." (appended to the returned reply when the model drops it;
  the stored history keeps the model's text); the agent pattern comes from an
  `agent_type` enum in the build schema (no second classifier; keyword fallback); a
  low-confidence or failed reclassification keeps the previous type; `start` never
  builds; keyword hints match whole words, a negation that governs the build phrase
  ("don't build it yet") blocks it, and a switch word plus a build phrase reclassifies
  before building. A malformed build reply (a JSON schema echo, or a field of the wrong
  type) is a failed build with one validation error per field in `send` and raises
  `MetaValidationError(errors=...)` from `run`; a build-call outage raises
  `BuilderError` (chained from `LLMResponseError`) from `run` and keeps the session
  open with the failed-build reply in `send`. `MetaBuilderConfig.timeout_seconds` is
  the per-request timeout; misplaced constructor kwargs (`model=`, `temperature=`,
  `hitl=`, ...) raise `TypeError`. `fsm-llm-meta` exits 130 on Ctrl-C (it exited 1).
  Prompt tokens per session rose about 43% (core's Pass-2 prompt).
- One LLM layer, reasoning: the engine sends no user message (no "Continue reasoning"
  turns). The classifier FSM runs on core `run_until_terminal` (10 steps; a spent
  budget is `ReasoningClassificationError` chained from `RunBudgetExceededError`). One
  orchestrator run (170 steps, `Defaults.MAX_SOLVE_STEPS`) drives every strategy FSM
  through its `before_step` hook: pushed by type (no FSM dict in context, so none in a
  prompt), popped when it ends or after 30 steps; a failing pop stops the solve.
  `execute_reasoning` entry clears `proposed_solution`, `key_insights` and
  `validation_result`, so a retry produces and validates a new answer. The validation
  verdict, counters, chosen strategy and confidence are handler-only (the model cannot
  open the validation gate); an attempt with no solution counts as a failed one; the
  hybrid back edge runs at most twice. The classifier, strategy executor, validator,
  retry limiter and hybrid counter are critical handlers: their failure stops the
  solve. A spent budget, and any other failed solve, raises `ReasoningExecutionError`
  with `details={conversation_id, responses_so_far, partial_context}` (it used to
  return the fallback string as the solution); ReasoningReact's `reason` tool reports
  a failed call. Malformed classifier values are normalised (a bad recommended type
  becomes `analytical` with a WARNING); a `0` or `False` answer is a solution; the
  driver keys `reasoning_push_pending`, `reasoning_type_selected` and
  `classified_problem_type` are dropped from `initial_context` with a WARNING;
  `all_responses` leaves out silent steps. Prompts reworded where they asked the model
  for handler-owned keys.
- One LLM layer, bench: `scripts/harness_bench.py probe-seed` sends through core's
  LLM layer (Ollama preparation now applies, so a new seed-probe record is not
  byte-comparable with an old one).
- Step driver, core: a silent state (empty `response_instructions`) says nothing on
  every entry. It makes no Pass-2 LLM call and no `LLMInterface` call at all;
  `start_conversation` and `converse` return `""` (they returned a `[<state>]`
  marker), `converse_stream`, `advance_stream` and the stream run loop yield nothing,
  `advance` returns `response=None`, and nothing is added to the history (no marker;
  a `converse` turn still records the user message). `MessagePipeline.process`
  returns `""` for it instead of raising. A custom `LLMInterface` is no longer called
  for silent states.
- Step driver, core: conversation history holds no synthetic entries: no `[state]`
  marker, and a message-free step adds no user exchange.
- Step driver, core: the LLM request models `ResponseGenerationRequest`,
  `FieldExtractionRequest` and `BulkExtractionRequest` carry `user_message: str |
  None` and refuse unknown fields (`extra="forbid"`). `None` means there is no user
  message; a string, `""` included, is what the user sent. A custom `LLMInterface`
  must accept `None` (it gets `None` for message-free steps, the greeting and
  context-only classifier calls).
- Step driver, core: the provider user turn is never empty. No user message is sent
  as `NEUTRAL_USER_TURN`; an empty or whitespace user message as
  `EMPTY_USER_MESSAGE_TURN` (`"(empty message)"`), which a model does not read as
  "proceed" (live: an empty message no longer fires a payment transition or leaves the
  classifier's fallback). This changes the user turn of every greeting (it was
  empty) and of every `converse("")`.
- Step driver, core: prompts have a no-message wording selected by `user_message=None`
  (a string keeps the old prompt byte for byte). Per-field extraction: "Determine the
  value of the field ..." from the instructions, context and conversation, composed
  when the instructions ask for something to be written or decided, null only when
  they cannot be followed. Bulk extraction: built from the state's `read_keys`-scoped,
  security-filtered context and history; on a message-free step it fills only unset
  keys and reports no rejected correction. Classification and ambiguity resolution:
  a context-only system prompt (`build_classification_system_prompt` and
  `build_classification_context_block` take keyword-only `user_message`, default
  `""`; `Classifier.classify`, `classify_multi` and `HierarchicalClassifier.classify`
  accept `None`). Pass 2: `build_response_prompt(user_message: str | None)`; with
  `None` the reply is the state's output, opens with its content (no greeting,
  thanks or remark about the step) and the `<transition_info>` block is dropped.
- Step driver, core: `Classifier` sends its request through its own
  `LiteLLMInterface.complete_structured` (the one request builder; `classification.py`
  no longer imports litellm). Its constructor and `classify` signatures are
  unchanged, but `retries=N` now means the SDK's `max_retries`, a model without
  `response_format` support logs a WARNING on every classifier call, whitespace-only
  user content counts as empty, and the extra keyword arguments are `**llm_kwargs`.
  Tests that patched `fsm_llm.classification.completion` must patch
  `fsm_llm.llm.completion`. (Superseded in this cycle by One LLM layer: the classifier
  sends `LLMInterface.complete` requests, inside a conversation through the
  conversation's own interface.)
- Step driver, core: `API.from_definition(fsm_definition=None, *, definition=None,
  **kwargs)`: the definition is given positionally or under either keyword, exactly
  once.
- Step driver, core: the transition-classification record of an AMBIGUOUS turn lives
  only in `context.metadata["transition_classification"]` (read it with
  `fsm_manager.get_complete_conversation(conv_id)["metadata"]`); handlers no longer
  see it in their context dict.
- Step driver, core: `fsm-llm-validate` reports a `handler_only_keys` entry that no
  state references as INFO, not WARNING (so exported agent FSMs no longer raise a
  false alarm). The `fsm-llm` runner prints no `System:` line for a silent reply.
- Step driver, core: error and log wording on the turn path: "Processing turn in
  state ...", "Failed to process message: ...", and for a step "Failed to advance
  conversation: ..." (manager) and "Failed to advance without a message: ..." (API).
- Step driver, agents: every FSM pattern runs on core's `run_until_terminal` /
  `run_until_terminal_stream` (`BaseAgent._run_conversation_loop` and
  `_standard_run_stream` are thin callers). No agent sends a synthetic "Continue."
  message, counts steps or filters markers. Core holds both budgets: `max_steps =
  max_iterations * FSM_BUDGET_MULTIPLIER` and `max_seconds` = what is left of
  `timeout_seconds`; the agents map core's `RunBudgetExceededError` to
  `BudgetExhaustedError` / `AgentTimeoutError` with the same messages, and the agent
  error's `__cause__` is now the core error. `_check_budgets(start_time)` keeps only
  the wall-clock check (SelfConsistency samples; `native_fc` used it too until One
  LLM layer moved it onto core's run loop).
- Step driver, agents: the HITL driver runs in the run loop's `before_step` hook and
  the seconds budget is checked again after it, so a slow approver uses up the
  timeout before an approved call runs (`AgentTimeoutError`; before, the call ran).
- Step driver, agents: a run whose conversation is closed from outside (by a hook or
  another thread) ends normally and reports `success=False, stop_reason="ended"` with
  its last output, instead of failing with `AgentError` "Conversation ... not found".
  Core's run loops end normally when the conversation is closed mid-run.
- Step driver, agents: a refused gated call is reported as not performed. On a denial
  the driver appends `refusal_record(tool, input)` to `refused_actions`, unless the
  same call (`call_label`) already ran in this run; `spend_grant` removes the matching
  entry when the same call is later approved and runs, and the key is dropped when no
  entry is left. The conclude prompt of approval-gated FSMs tells the model to say
  those actions were not performed (live: 8 of 8 denied runs that reached the gated
  tool answered so). A re-asked identical call is still asked again.
- Step driver, agents: `await_approval` extracts nothing (one LLM call less per visit,
  and no model channel in the state that guards the approval).
  `plan_all`, `orchestrate`, `collect` and ADaPT `attempt`, `assess`, `decompose` have
  no state-level bulk extraction (one call less per visit; their typed fields are
  every key the run reads; ADaPT `operator` is a typed `decompose` field).
  `delegation_plan`, ADaPT `confidence`, `reasoning` and `evaluation_feedback` no
  longer appear in the final context. Ten terminal states lost their dead
  `extraction_instructions`, and four (`conclude` of React, Reflexion and
  ParallelReact, ADaPT `combine`) their `required_context_keys: ["final_answer"]`, so
  the conclude prompt no longer asks to collect a "Final answer".
- Step driver, agents: `final_context["final_answer"]` is no longer an answer source.
  `BaseAgent._extract_answer`, `_completion_is_real` and the ADaPT, EvalOpt and
  PromptChain overrides read the pattern's own answer keys and the last spoken reply;
  a model-written `final_answer` is inert. `final_answer` stays a run output that
  `initial_context` cannot set.
- Step driver, agents: prompt wording without loop-signal text. The conclude
  instructions describe the reply (the run's last output from the observations, no
  further tool, no progress report, a plain statement of what could not be
  determined); the typed-field prompts drop "The user message is only a loop signal".
  PlanExecute's `plan_steps` and replan instructions add no step that needs no tool
  unless the task asks for one, and no step that confirms, waits for or asks for
  anything.
- Step driver, agents: ADaPT `operator` is stripped from caller context like the
  other run outputs. `run_stream` yields only model text from speaking states.
  `ReactAgent.run_stream` on a conversation ended from outside ends without an error
  (Known open).
- Step driver, monitor: `configure(*, manager=None, cors_origins=None, api_key=None,
  trusted_hosts=None)` is keyword-only (a positional call is a `TypeError`). The FSM
  visualize routes draw core's `build_fsm_graph` and answer 400 `failed to parse FSM
  definition: <reason>` for a definition core cannot graph (an empty `{}` or a
  dangling target used to draw a partial graph); every valid definition gives the
  same payload as before. The control page no longer labels `final_answer`.
- Step driver, workflows: `WorkflowEngine(*, max_concurrent_workflows=100,
  max_completed_instances=1000, max_steps_per_run=1000, executor=None)` takes keyword
  arguments only. `ConversationStep` publishes `last_response` / `final_answer` as the
  last reply that was actually spoken and adds neither key when nothing was spoken
  (it published `""` when the last turn ended on a silent state).
- Step driver, harness: the harness runs on core's run loop through `BaseAgent`; its
  step ceiling is unchanged (`MAX_TURNS` 60 x 3 = 180 core steps).
- Core: `FSMDefinition.persona` and `FSMInstance.persona` accept up to 4,000
  characters (was 500), one constant `fsm_llm.constants.MAX_PERSONA_LENGTH`. Persona
  is still rendered only in the Pass-2 response prompt, sanitized; a longer persona
  makes every reply call's prompt larger.
- Agents: `create_agent(pattern="react", tools=None, *, config=None, system_prompt=None,
  **kwargs)` takes the pattern first, so `create_agent("debate")` builds a
  `DebateAgent`. Pattern names are matched after `strip().lower()` (`"Debate "` is the
  debate pattern); any other first argument raises `ValueError` listing the patterns
  (the positional system prompt is removed, see Removed). `system_prompt` counts
  against the prompt limits: over 2,000 characters raises `ValidationError`, and one
  that makes an FSM instruction slot exceed core's 5,000-character limit together with
  the tool catalogue raises `AgentError` at `run()` (naming the slot, the
  instructions length and the tool count). A third positional argument is a
  `TypeError`.
- Agents: `system_prompt` (new `AgentConfig.instructions`, at most 2,000 characters)
  now reaches the model. FSM patterns prefix it to every non-empty state and per-field
  instruction (replies and per-field extractions); it does not reach
  `classification_extractions` (the `use_classification=True` think path), core's
  AMBIGUOUS transition classifier or ReasoningReact's reasoning engine.
  `NativeFunctionCallingReactAgent` uses it as its default `system_policy`. Swarm and
  meta_builder reject it in `create_agent`.
- Agents: `AgentStep.thought`, `ToolCall.reasoning` and `ApprovalRequest.reasoning`
  are now empty in the ReAct family unless `use_classification=True`: the think state
  no longer extracts a `reasoning` field (it collided with the extraction envelope's
  own key), so an approval UI no longer gets the model's stated rationale. Known open
  (LOOP-16, `docs/agents_roadmap.md`).
- Agents: `AgentConfig` and `MetaBuilderConfig` reject unknown fields
  (`extra="forbid"`). A misspelt key, the removed `MetaBuilderConfig.output_path`, or a
  typo in an SOP's `config_overrides` now raises instead of being ignored.
- Agents: `AgentConfig.model` defaults to env `LLM_MODEL` (read when the config is
  built), then `DEFAULT_LLM_MODEL`; `default_llm_judge(model=None)` resolves the same
  way. An explicit model always wins.
- Agents: `success` has one meaning, "the run reached its goal". A run that was forced
  to stop (`max_iterations_reached`, three turns with no tool, a failing
  EvaluatorOptimizer or MakerChecker verdict overridden at its revision or budget
  limit, a Reflexion run that hit `max_reflections` without a passing evaluation, a
  Debate consensus forced by `num_rounds` or the budget, a failed PromptChain gate)
  still returns its last answer but reports `success=False` with the matching
  `stop_reason`. A genuine pass, verdict or conclusion on the budget's last round
  counts as success. SelfConsistency is no longer always `True` (it needs a sample
  with text), a Debate needs a proposition (the last round that had one), and REWOO
  needs at least one tool call that succeeded. A `SwarmAgent` handoff to an unknown
  agent reports `success=False, no_result`. Monitor runs and workflow `AgentStep`s
  show these runs as failed.
- Agents: every agent FSM lists `max_iterations_reached`, `forced_stop_reason`,
  `iteration_count` and `observation_count` in core `handler_only_keys`, so no model
  extraction can write them.
- Agents: in the ReAct family (`ReactAgent`, `ParallelReactAgent`,
  `ReasoningReactAgent` and the ReAct subclasses) `max_iterations=N` now counts think
  turns: for N >= 2 a run that never concludes gets N think turns and N - 1 tool
  calls, and N = 1 behaves like N = 2 (2 think turns, 1 tool call). Before, it
  counted every FSM transition (about half as many tool calls). `ReflexionAgent`
  counts the same think turns but closes a cycle on the act exit, so its forced
  conclusion comes from `evaluate` (N = 1, 2, 3, 4 gave 1, 1, 2, 3 think turns and
  1, 1, 2, 2 tool calls). A run that never concludes by itself takes about twice as
  long before its forced stop. No tool runs after the forced-stop flag is set, and
  the model is still asked on the last think turn: its own conclusion there, backed
  by a tool result, reports `success=True`.
- Agents: `@tool(requires_approval=True)` now gates the tool when `HumanInTheLoop` has
  an approval callback and no policy (the flag is the default policy). With a policy,
  the policy alone decides, as before. `ReactAgent`, `ReflexionAgent` and
  `ReasoningReactAgent` raise `AgentError` when a flagged tool exists and nobody can
  decide on it (`hitl=None`, or a `HumanInTheLoop` with neither a callback nor a
  policy), at construction and again at the start of `run()`/`run_stream()`; before,
  the tool ran unasked after a logged WARNING. A policy without a callback still
  constructs (it logs a WARNING): the policy decides, and a call it gates raises
  `ApprovalDeniedError`.
- Agents: `REWOOAgent`, `PlanExecuteAgent`, `ParallelReactAgent` and
  `NativeFunctionCallingReactAgent` have no approval step, so they now raise
  `AgentError` when their registry holds a `requires_approval` tool (at construction
  and again at the start of `run()`).
- Agents: constructors raise `TypeError` for `hitl=`, `tools=`, `evaluation_fn=` or
  any `HumanInTheLoop` argument (`approval_policy=`, `approval_callback=`,
  `on_escalation=`, `confidence_threshold=`, `approval_timeout=`) on a pattern that
  does not take them, and for `model=`,
  `temperature=`, `max_tokens=` (set them on `AgentConfig`). They used to be forwarded
  to `litellm.completion`, so HITL was silently ignored. Other keyword arguments
  (`seed`, `timeout`, `handlers`, `llm_interface`, ...) still pass through.
  `create_agent` passes `tools` only to patterns whose constructor takes them and
  raises `TypeError` for the others.
- Agents: `initial_context` can no longer set run-owned keys (`final_answer`,
  `should_terminate`, `observation_count`, tool selection and results, approval keys)
  or the driver grant `_approval_granted`; they are dropped with a WARNING and
  `observation_count` is seeded 0. `AgentServer` also drops every internal-prefix key
  from the request context. Swarm hand-offs and AgentGraph edges strip the same keys.
  Each pattern also drops its own run outputs (drafts, verdicts, answers and progress
  keys, e.g. MakerChecker `draft_output`/`checker_passed`, EvalOpt
  `generated_output`, Debate `proposition`/`consensus_reached`, PlanExecute
  `plan_steps`) from `initial_context`, and an AgentGraph node or Swarm hand-off
  target never receives its own run outputs from a predecessor: a forged or inherited
  draft used to ship as a successful answer with no work done.
- Agents: loop values (tool selection, thoughts that route, drafts, critiques,
  verdicts, plans, reflections, step results) are extracted as typed per-field values
  with the task and results in the prompt, cleared before each round, instead of from
  a context-free bulk prompt; intermediate states no longer write a Pass-2 reply.
  `agent_trace` is kept out of these prompts. Each typed field is one LLM call.
  PromptChain no longer bulk-extracts the keys a step's `extraction_instructions`
  name; each step extracts one `chain_step_result`.
- Agents: `ReactAgent.run_stream` yields only model text and wraps errors as
  `AgentError` like `run()`.
  `VerifiedReactAgent` and `AutoMemoryReactAgent` stream by running `run()` and
  yielding the answer once, so verification and memory are kept.
- Agents: the Debate answer is the conclusion written after the last round, not the
  first round's judge verdict. SelfConsistency votes on each sample's last `Answer:`
  line (casefolded) and `confidence` is the share of samples that agree.
- Agents: `AgentGraph` runs nodes in topological order, each once, after all its
  predecessors; the answer comes from the last executed sink. A node that raises or
  returns `success=False` takes no outgoing edge. A cycle raises `ValueError` also for
  a directly constructed `AgentGraph`.
- Agents: `SwarmAgent` gives every member the original task (the hand-off message
  travels in context) and `max_handoffs=N` allows exactly N handoffs.
- Agents: `PlanExecuteAgent` replans when a step's tool fails, `max_replans=N` allows
  exactly N replans, and a new plan replaces the old one. `plan_steps` is a typed list
  capped at 10 steps.
- Agents: `VerifiedReactAgent` periodic reflection notes go to `agent_feedback`, not
  `observations` (they no longer count as evidence).
- Agents: tool calls bind their arguments against the function signature first. A
  `TypeError` raised inside a tool is a failed call, never a retry, and a named
  optional argument is never moved into a missing required one. `register_function`
  infers the parameter schema from type hints like `@tool`, also for
  `functools.partial` objects, callable instances and async callables. A
  positional-only parameter is bound by position from its named value.
- Agents: `RetryingToolRegistry` never retries a `requires_approval` tool: one approval
  covers one execution. A tool gated only by an approval policy is still retried
  (closed later in this cycle by One LLM layer: a granted call is never retried, and
  only `retry_safe` tools are).
- Agents: `ReasoningReactAgent`'s tool view follows the caller's registry live, so a
  tool re-registered there with `requires_approval=True` after construction is gated.
- Core: in an envelope-shaped single-field reply (one carrying `value` or
  `field_name`), `llm._field_value` no longer falls back to `data[field_name]` for a
  field named `reasoning`, `confidence` or `field_name`: there that key is the
  envelope's own explanation, score or echoed name, so with a null `value` the
  fallback returned it as the field's value. A flat reply without `value` or
  `field_name` (`{"confidence": 0.8}` for a field named `confidence`) keeps its value.

### Fixed

- One LLM layer, core: a custom `llm_interface` given to `API` was bypassed by the
  classifier (AMBIGUOUS transitions and `classification_extractions` ran on a private
  litellm interface).
- One LLM layer, agents (TOOL-04): typed tool parameters had coarse schemas
  (`Optional[int]` was `string`); schemas now come from the function's own model.
- One LLM layer, agents (TOOL-07): `RetryingToolRegistry` could re-run a call an
  approver granted when the tool was gated only by an approval policy.
- One LLM layer, agents: the default LLM judge ran with thinking on for `ollama_chat`
  models (live: 8.3 s per verdict, 0.43 s now, same 8/8 verdicts).
- One LLM layer, reasoning: a solve whose validation kept failing never reached its
  retry limit (the rejected solution stayed set and was never re-extracted; the
  baseline ran until core's prompt cap), the model could write the validation verdict
  and retry counters itself, the hybrid strategy's loop counter never moved, the whole
  strategy FSM dict reached a reply prompt through context, a spent budget returned a
  placeholder sentence as the solution, and a failing pop was logged and swallowed.
- Core: a single-field extraction reply that opens with this field's
  `{"field_name": ..., "value": ` envelope but does not parse now yields the
  envelope's `value` on the `str`/`any` rung, never the envelope text. Prose that
  merely contains JSON is still kept verbatim. A complete value is kept whole, also
  when it holds an escape JSON does not define (`\d` in a regex, `C:\Users`), which is
  kept literally. A value cut off by `max_tokens` keeps its decoded prefix (never
  ending in half a surrogate pair), is logged as a WARNING naming the field, and
  returns at confidence 0.3 (`constants.TRUNCATED_SALVAGE_CONFIDENCE`, below the 0.5 of
  other unstructured coercions), so a field `confidence_threshold` of 0.5 rejects it.
  On Ollama this salvage, not the field type, is what keeps an envelope out of an agent
  answer: the Ollama `any` grammar has no object branch.
- Agents: EvalOpt `generated_output`, MakerChecker `draft_output` and PromptChain
  `chain_step_result` are typed `any` again, their type before this release (step 20
  had made them `str`). On a provider without a grammar the model can return a JSON
  deliverable as a native object; a dict/list value reaches `evaluation_fn` and
  `AgentResult.answer` as indented JSON text, so `output_schema` validation parses
  it. `False`, `0`, `{}` and `[]` in an answer key count as no answer (they rendered
  as `"False"`, `"0"`, `"{}"`, `"[]"` and made an unjudged run report success).
- Reasoning: `ReasoningTrace` now dumps `reasoning_types_used` as a sorted list, so
  `python -m fsm_llm.reasoning --output json` and `--save` JSON files list the types
  instead of writing `"<redacted:set>"`. `model_dump()` returns a list too, so the
  `ReasoningReactAgent` observation shows `type=['analytical']` instead of a set repr.
- Monitor UI: the launch dialog's Max Iterations input now allows 1 to 100 like the
  server, the Settings log-level select lists SUCCESS, and SUCCESS and TRACE log lines
  get their own CSS classes. A test ties the HTML to `LOG_LEVELS` and
  `MAX_AGENT_ITERATIONS`.
- `setup_cli_logging` docstring names its three call sites (harness and eval `run()`,
  reasoning `--verbose`).
- Agents (SEC-01, LOOP-12): a caller could forge the HITL driver grant or a finished
  run through `initial_context`, directly or via `AgentServer`, SelfConsistency, Swarm
  or AgentGraph, and run a gated tool without asking.
- Agents (SEC-05, SEC-06): memory tools listed, wrote and deleted the hidden
  `metadata` buffer and printed secrets. Secret-looking tool arguments appeared in
  observations, traces, logs and the approval `context_summary`; they are now
  `<redacted>` there (the tool and `ApprovalRequest.parameters` keep the real values).
- Agents (SEC-09): `AgentServer` 500 responses and `/stream` error events no longer
  contain the exception text; they carry a generic message and an `error_id`.
- Agents (TOOL-01, TOOL-02): a tool whose body raised `TypeError` ran twice; a nested
  `tool_input` dropped its sibling arguments.
- Agents (TOOL-14): `register_agent` tools and `RemoteAgentTool` reported a failed
  sub-agent run (`success: false`) as a successful call.
- Agents (REACT-11): `NativeFunctionCallingReactAgent` ran a tool with `{}` when its
  arguments were not a JSON object; such a turn now ends the loop without running it.
- Agents (MEM-01 to MEM-04): `SemanticMemoryStore(persist_path=...)` never loaded the
  file, so the next save overwrote stored memories; entries without an embedding were
  unreachable once any had one; `max_entries` was not saved; a missing parent directory
  made saving fail; saves were not fsynced; vectors of different length were zipped.
  A corrupt store file now raises `ValueError` instead of being overwritten.
- Agents (LOOP-14): with `handler_timeout`, an approved call whose handler timed out
  could run again on the same approval.
- Agents (PAT-04, PAT-10): `BudgetExhaustedError` and `AgentTimeoutError` from an
  ADaPT subtask or an Orchestrator worker were swallowed; they now end the run. ADaPT
  answered with the failed first attempt and counted its decomposition as a tool call;
  Orchestrator silently dropped subtasks beyond `max_workers` (now listed in
  `skipped_subtasks`, kept out of `worker_results` so the collect step does not
  re-delegate them). Orchestrator, ADaPT, REWOO and the Debate judge extract their
  planning and verdict fields through typed prompts that no longer carry `agent_trace`.
- Agents (REACT-01, REACT-02): Reflexion stored an empty reflection for episode 1 and
  lagged one episode behind; `evaluation_fn` was skipped when the self-evaluation came
  back empty.
- Agents (PAT-01, PAT-02): PlanExecute never replanned, and a string plan was iterated
  per character.
- Agents (PAT-03, PAT-05, PAT-06): Debate rounds reused round 1's text; SelfConsistency
  compared whole replies, so the first sample always won; PromptChain validation gates
  stopped nothing and `chain_results` stayed empty.
- Agents (PAT-09): REWOO reported success when every tool failed, and plan id `"E1"`
  became `EE1` so `#E1` never resolved.
- Agents (PAT-11): MakerChecker drafts and revisions did not see the task or the
  checker's feedback.
- Agents (REACT-04, REACT-05): a VerifiedReact answer rejected on the last attempt kept
  its success flag and a raising verifier counted as a pass; ReasoningReact sent the
  reasoning engine `"{}"` instead of the task, shadowed a user tool named `reason`, and
  dropped the behaviour of Caching/Retrying registries.
- Agents (LOOP-06, LOOP-16, LOOP-17): executor warnings and HITL denials never reached
  the next think turn; observation step numbers repeated after 20 observations;
  `BudgetExhaustedError` cited `max_iterations` instead of the loop ceiling. An early
  `should_terminate` no longer drops an approved call.
- Agents (META-06): the meta-builder's FSM few-shot example targeted an undeclared
  `end` state.
- Agents: `ToolRegistry.register_function` raised `AttributeError` for a
  `functools.partial` or a callable object (a regression from schema inference in
  this release). The two argument-fallback DEBUG log lines printed raw tool input;
  they are redacted like the other tool-input log lines.
- Docs: `docs/api_reference.md` agent snippets passed `model=` to constructors, which
  crashed at run time.
- Agents (meta): builder subclasses that skip `ArtifactBuilder.__init__` no longer
  fail in `take_warnings()` and the `validate` and `get_summary` meta tools (the
  warnings list is created on first use).
- Core: `APIBuilder.build()` reports an open-ended option that repeats a constructor
  parameter before it reports a missing definition, so
  `set_llm_option("fsm_definition", ...)` points at `set_definition()`.

### Renamed

- Agents (meta-builder): `FSMBuilder`, `WorkflowBuilder` and `AgentBuilder` in
  `fsm_llm.agents` and `fsm_llm.agents.meta_builders` are now `FSMArtifactBuilder`,
  `WorkflowArtifactBuilder` and `AgentArtifactBuilder`, with no alias. The old
  `WorkflowBuilder` name clashed with `fsm_llm.workflows.WorkflowBuilder`, a different
  class that keeps its name. `ArtifactBuilder` is unchanged. Removed names:
  `fsm_llm.agents.FSMBuilder`, `fsm_llm.agents.WorkflowBuilder`,
  `fsm_llm.agents.AgentBuilder` and the same three in
  `fsm_llm.agents.meta_builders`. `fsm_llm.harness.AgentBuilder` (a callable alias)
  and `fsm_llm.workflows.WorkflowBuilder` are different objects and keep their names.
- Builder standard: `fsm_llm.workflows.WorkflowBuilder.build(validate=...)` lost its
  `validate` argument (see Changed).

### Removed

- One LLM layer, core: `LiteLLMInterface.complete_structured`; use
  `LiteLLMInterface.complete(CompletionRequest(..., response_format=...))`.
- One LLM layer, core: the private classifier connection-kwargs path of the pipeline
  (`MessagePipeline._classifier_connection_kwargs`, `pipeline._CLASSIFIER_BOUND_NAMES`).
  `Classifier._extract_response` now takes the `CompletionResponse` alone (it took
  `(content, response)`).
- One LLM layer, agents: the `complete_fn` constructor parameter of
  `NativeFunctionCallingReactAgent` and the `CompleteFn` type alias in
  `fsm_llm.agents.native_fc`; inject an `LLMInterface` with `llm_interface=`.
- One LLM layer, agents: the private `native_fc` loop and its helpers
  (`NativeFunctionCallingReactAgent._litellm_complete`, `_complete`,
  `_assistant_message`, module `_degrades_turn`, `_is_malformed_tool_call`,
  `_call_arguments`, `_MALFORMED_TOOL_CALL_MARKERS`; use core
  `llm.is_malformed_tool_call_error`, `llm.decode_tool_arguments`,
  `constants.MALFORMED_TOOL_CALL_MARKERS`), and the `details["malformed_tool_call"]`
  label of its `AgentError` (a malformed tool turn is now data, not an error).
- One LLM layer, agents: `MetaBuilderConfig.build_max_iterations`,
  `MetaBuilderConfig.build_timeout_seconds` and `MetaBuilderConfig.build_temperature`
  (passing one now raises `ValidationError`); use `timeout_seconds` for the request
  timeout.
- One LLM layer, agents: the `max_iterations` default override of `MetaBuilderConfig`
  (25); the field is the inherited `AgentConfig.max_iterations` (default 10), accepted
  and not read by the meta-builder.
- One LLM layer, agents: `MetaDefaults.BUILD_MAX_ITERATIONS`,
  `MetaDefaults.BUILD_TIMEOUT_SECONDS`, `MetaDefaults.BUILD_TEMPERATURE`.
- One LLM layer, agents: the meta-builder's own LLM path and Python state machine
  (`MetaBuilderAgent._llm_call`, `_get_type_classifier`, `_get_agent_type_classifier`,
  `_generate_collect_response`, `_detect_type`, `_execute_build`,
  `_run_deterministic_pipeline`, `_preseed_agent_type`, `_build_result`, the
  `_FSM_SCHEMA`/`_WORKFLOW_SCHEMA`/`_AGENT_SCHEMA`/`_ARTIFACT_SCHEMAS` class attributes
  (the schemas are `meta_prompts.artifact_schema`), and the private fields `_builder`,
  `_artifact_type`, `_complete`, `_build_error`, `_started`, `_messages`).
- One LLM layer, agents: `fsm_definitions._typed_field_extraction`,
  `fsm_definitions._EXTRACTION_ENVELOPE_KEYS` and `fsm_definitions.TypedFieldType`;
  import `typed_field_extraction` and `TypedFieldType` from core (`fsm_llm`,
  `fsm_llm.definitions`) and `EXTRACTION_ENVELOPE_KEYS` from `fsm_llm.constants`. No
  re-export from agents.
- One LLM layer, reasoning: `ContextKeys.REASONING_FSM_TO_PUSH`
  (`"reasoning_fsm_to_push"`); the strategy FSM is pushed by type
  (`ContextKeys.REASONING_PUSH_PENDING`).
- Step driver, core: `FSMManager.cleanup_stale_conversations()`; use
  `FSMManager.prune_orphaned_locks()` (drops orphaned locks) or
  `API.cleanup_stale_conversations()` (ends idle conversations; kept).
- Step driver, core: `ClassificationResult.is_low_confidence`; use
  `is_below_default_threshold` (fixed 0.6) or `Classifier.is_low_confidence(result)`
  (schema threshold; kept).
- Step driver, core: `SchemaValidationError` (nothing raised it); catch
  `ClassificationError`.
- Step driver, core: the no-op `TransitionEvaluatorConfig` fields
  `ambiguity_threshold`, `minimum_confidence` and `evidence_conditions_normalizer`
  (passing one is now a `TypeError`); transitions are ranked by `priority` alone.
- Step driver, core: `HandlerSystem.close()` (it did nothing).
- Step driver, core: the `IntentDefinition` re-export from `fsm_llm.classification`;
  import it from `fsm_llm` (or `fsm_llm.definitions`).
- Step driver, core: `constants.CONTEXT_KEY_CLASSIFICATION_RESULT` and the
  `context.data["_transition_classification_result"]` copy; read
  `context.metadata["transition_classification"]`.
- Step driver, core: the underscore-private `security` names re-exported from
  `fsm_llm.constants`, and the private `pipeline._PROVENANCE_KEY`. The public security
  names stay importable from `constants`.
- Step driver, core: `ResponseGenerationRequest.skip_generation` and the `"."`
  system-prompt sentinel: a silent state makes no interface call (passing
  `skip_generation=` is now a validation error).
- Step driver, core: the `[<state>]` marker that `start_conversation`, `converse` and
  streams returned and recorded for a silent state; they return `""` and record
  nothing.
- Step driver, core: the "Continue." de-anchor branch of the per-field extraction
  prompt (it told the model to ignore a "Continue." message); message-free steps use
  the no-message prompt instead.
- Step driver, agents: `Defaults.CONTINUE_MESSAGE` and the synthetic "Continue."
  turns of the agent loops; the private `BaseAgent._skip_marker_states`,
  `_drop_skip_marker` and `_start_conversation`. Use core's run loop.
- Step driver, agents: the positional system prompt of `create_agent`
  (`create_agent("You are ...", tools)`, deprecated in this release by Track A); any
  first argument that names no pattern raises `ValueError`. Use
  `create_agent(pattern, tools, system_prompt=...)`.
- Step driver, agents: `DecompositionError` and `ToolValidationError` (never raised).
- Step driver, agents: the module `fsm_llm.agents.meta_fsm` (`build_meta_builder_fsm`,
  an unused placeholder; the meta-builder never called it). One LLM layer later added
  a real `build_meta_builder_fsm` in `fsm_llm.agents.fsm_definitions`, which the
  meta-builder runs.
- Step driver, agents: `PromptChainStates.GATE_PREFIX` and
  `SelfConsistencyStates.AGGREGATE` (no state of that name), `Defaults.EVALUATION_THRESHOLD`,
  `ErrorMessages.BUDGET_EXHAUSTED`, `ContextKeys.DELEGATION_PLAN`.
- Step driver, agents: prompt builders whose text no run sends any more:
  `build_approval_extraction_instructions`, `build_orchestrate_extraction_instructions`,
  `build_collect_extraction_instructions`, `build_attempt_extraction_instructions`,
  `build_assess_extraction_instructions`, `build_decompose_extraction_instructions`,
  and the terminal-state builders `build_conclude_extraction_instructions`,
  `build_synthesize_extraction_instructions`, `build_rewoo_solve_extraction_instructions`,
  `build_evalopt_output_extraction_instructions`,
  `build_maker_checker_output_extraction_instructions`,
  `build_orchestrator_synthesize_extraction_instructions`,
  `build_combine_extraction_instructions`, `build_chain_output_extraction_instructions`.
- Step driver, monitor: `MonitorBridge` and the module `fsm_llm.monitor.bridge`;
  `configure(bridge=)`; `server.get_bridge` (use `server.get_manager`);
  `InstanceManager.connect_bridge` (use `InstanceManager.attach_api`). To show your own
  `API`: `manager = InstanceManager(); manager.attach_api(api);
  configure(manager=manager)`.
- Step driver, monitor: `THEME_NAME` and the `COLOR_*` constants (`COLOR_PRIMARY`,
  `COLOR_SECONDARY`, `COLOR_BACKGROUND`, `COLOR_SURFACE`, `COLOR_FOREGROUND`,
  `COLOR_ACCENT`, `COLOR_WARNING`, `COLOR_ERROR`, `COLOR_SUCCESS`, `COLOR_MUTED`,
  `COLOR_BORDER`; the colors live in `static/style.css`), `EVENT_LOG` (no event of
  that type is emitted), the no-op POST_TRANSITION entry of
  `EventCollector.create_handler_callbacks()` (7 callbacks now), and the unused
  front-end accessors `isLogPaused` and `isAuthRequired`.
- Step driver, workflows: `MAX_STEP_DEPTH` (use `MAX_STEPS_PER_RUN`),
  `WorkflowEngine(handler_system=)` and its `handler_system` attribute (never called;
  use `add_hook`), and the private `_KEY_*` / `_STEP_INTERNAL_WHITELIST` aliases in
  `engine.py`.
- Step driver, reasoning: `Defaults.MAX_CONTEXT_SIZE`, `ErrorMessages.CONTEXT_TOO_LARGE`,
  `ErrorMessages.VALIDATION_FAILED`, `ErrorMessages.CALCULATION_ERROR` (no reader).
- Step driver, harness: `Defaults.CONTINUE_MESSAGE`, `Defaults.DECISIONS_COMPRESS_LINES`,
  `CHANGELOG_COMPRESS_LINES`, `LESSONS_IMPORTANCE_MIN`, `LESSONS_IMPORTANCE_MAX`;
  `fsm_llm.harness.storage.PLAN_ID_RE` (use `fsm_llm.harness.PLAN_ID_RE`, defined once
  in `artifacts`); the private `storage._atomic_write_text` alias (use
  `fsm_llm.harness._atomic.atomic_write_text`); the read path for legacy
  `plan_YYYY-MM-DD_<hex8>` plan ids (a cross-plan section headed by one is no longer
  counted as a plan section).
- Step driver, build: `make clean` no longer deletes the pre-2026-09-29
  `src/fsm_llm_<sub>/` directories, and `tests/test_packaging.py` no longer checks for
  them; delete them by hand in an old clone.
- `fsm_llm.harness.constants.HandlerPriorities` and `HandlerNames` no longer define
  `PRE_STEP_GATE`, `END_CONVERSATION` or `ERROR`. Nothing registered them: the pre-step
  gate runs inside the EXECUTE dispatch, and end and error handlers come from
  `BaseAgent`.
- `fsm_llm.harness.PRESENTATION_CONTRACTS` no longer defines `PC-PIVOT`; the driver
  never emitted it and no artifact supplies its candidate-directions or
  ghost-constraints fields.
- Agents: `AgentHandlers.classification_tool_override` and its registration. It read
  a context key nothing wrote, so it never did anything (`AgentHandlers` is not
  exported).
- Agents: unused constants in `fsm_llm.agents.constants`: the classes
  `MetaBuilderStates` (back, with new states, for the meta FSM of One LLM layer) and
  `DecisionWords`; `MetaDefaults.BUILD_MAX_TOKENS`;
  `MetaLogMessages.ARTIFACT_CLASSIFIED`, `BUILD_COMPLETE`, `REVIEW_STARTED`,
  `REVISION_STARTED`; `MetaErrorMessages.BUILDER_NOT_INITIALIZED`,
  `INVALID_ARTIFACT_TYPE`; `ErrorMessages.APPROVAL_DENIED`, `TIMEOUT`, `NO_TOOLS`,
  `MAX_REFLECTIONS`, `MAX_REFINEMENTS`, `MAX_REVISIONS`, `MAX_DEPTH`.
- Agents: `fsm_llm.agents.__all__` is one static list; `ReasoningReactAgent` is always
  exported and `create_agent("reasoning_react")` always available.

### Known open

From One LLM layer (plan `944e2692`; review leftovers and work kept out of scope):

- Out of scope: the ReAct family still extracts `tool_name`/`tool_input` as typed
  fields (about 11 LLM calls per task against 2.4 for native tool calling) and is not
  on the completion state; the reasoning classifier FSM is not a
  `classification_extractions` field; no `parallel_tool_calls` control, no streamed
  tool calls, no per-conversation usage counters; nothing was checked live on a
  non-Ollama provider.
- Reasoning: typed per-key reasoning fields with silent intermediate states cut calls
  by a further 43% but lost answers on the pre-registered probe (18/24 against 24/24:
  logic problems answered with a bare "True"/"False", an open-ended answer with one
  item), so they were reverted; a new attempt needs its own pre-registered probe. A
  final step that fails (for example over core's 30,000-character prompt cap) loses
  that turn's `final_solution` (`partial_context` keeps the validated
  `proposed_solution`). `ReasoningClassificationError` reaches the caller only as the
  `__cause__` chain of `ReasoningExecutionError`. A `final_context` passed back as
  `initial_context` still carries the previous problem's analysis (`problem_type`,
  `reasoning_strategy`). A malformed recommended type falls back to `analytical`
  (63 scripted calls against 45 for the model's own strategy; the cost was not
  measured live). The classifier's unused final reply can approach the prompt cap with
  very long fields (live peak 8.8k of 30k characters). A stuck orchestrator state costs
  up to the 170-step budget in calls. `responses_so_far` in the error details leaves
  out step replies.
- Core run loop: the ended-frame `before_step` call repeats the next step number
  (tell the two calls apart with `has_conversation_ended`), and `seconds_exempt_states`
  ids are matched per round against the current top of the stack, so a pushed child
  state with an exempt id is exempt too. `clear_keys_on_entry` fires only on a
  transition into the state, never on start, push, pop or restore.
  `typed_field_extraction` accepts underscore field names and its preamble says "from
  the task" (the agents wording, kept byte for byte).
- Core classification: a tie between transitions with a custom interface that has no
  `complete` soft-fails every turn, so two unconditional transitions at one priority
  never resolve; a direct `Classifier(llm=...)` refuses such an interface with
  `ValueError` while the pipeline reports `ClassificationError`. Classifier calls to an
  entry's own `model` are not counted on the conversation interface's `usage()`. A
  classifier reply carrying a huge integer fails the turn (`ValueError` in parsing)
  instead of soft-failing.
- Core LLM layer: a custom `LLMInterface` may ignore
  `ResponseGenerationRequest.temperature` (SelfConsistency samples then share one
  temperature); `_response_temperature` is not saved in sessions and not passed to a
  pushed child; the apology retry's temperature is untested; `deepcopy` or pickling of
  a `LiteLLMInterface` after its first call raises `TypeError` (its meter holds a
  lock); constructor kwargs such as `parallel_tool_calls` or `logprobs` reach every
  call, classification included; list-shaped message content is sent to
  Ollama as a Python repr; a provider that omits tool-call ids makes every such turn
  `malformed` (Ollama sends ids); the validator's result-kind coverage warning misreads
  `!=` and variable comparisons.
- Core JsonLogic: `==` against `true` also passes for `"true"`, `"True"`, `1` and `1.0`
  (documented; behaviour unchanged; the meta-builder's gates use `===`).
- Core prompts: the conversational Pass-2 prompt tells the model to acknowledge a
  transition, so replies can name internal phases (meta-builder: "moving into the
  collection phase").
- Tools: a pydantic-model parameter still receives a plain dict (the schema is exact,
  the binder is unchanged); a tool that keeps hanging past its timeout keeps its thread
  (a Python thread cannot be killed), so repeated timeouts pile up threads; the MCP
  annotation mapping is untested against the real `mcp` package, MCP `$defs` are
  dropped (an Enum parameter reads `any`) and an MCP timeout reads `failed`, not
  `unknown`; `allOf`-wrapped and pathological hand-written `$ref`s read `any` or recurse
  deeply in `schema_types`; an `IntEnum` think-state example shows `0`; REWOO,
  ParallelReact and `native_fc` do not check that a registry's `execute` accepts
  `gated`; PlanExecute's `no_result` for a timed-out step does not name the timeout.
- `native_fc`: with an interface built with `timeout=None` the forced and repair turns
  after the deadline are unbounded; an agent given `seed=` beside `llm_interface=` is
  refused only when `run()` builds its `API`; `reasoning_model=` of
  `ReasoningReactAgent` is dropped beside an injected interface. The harness anchor in
  `roles.py` still names the old `native_fc` block (harness source unchanged here).
- Meta-builder: an explicit `null` in a build reply field is a malformed build; a
  one-word trigger with punctuation ("go.") is not a trigger; the negation guard lets
  some negated requests build ("don't you dare build it"); the reply returned to the
  user and the reply stored in core's history differ by the appended build sentence;
  prompt tokens per session are about 43% higher than before.
- Bench: `agents-react/B2` calls (meter "2") and B0 calls (meter "1") come from
  different meters, bridged by an offline and a live parity check; only the first
  request of each task is digested in a manifest; harness rows carry no LLM-call
  count; the L4 react control arm makes few tool calls (as in B0/B1);
  `agents_bench register`/`run` still import litellm (cost-map fetch).

From the step driver work (plan `07ad3f8c`), open until its iteration 2 or later:

- Closed by One LLM layer: every LLM caller is on core's LLM layer (`native_fc` and
  the meta-builder are FSMs on core); the classifier uses the conversation's
  `llm_interface`; the reasoning engine sends no synthetic message; ToolSpec is ported
  (TOOL-04 and TOOL-07 fixed); usage counters exist and the agent bench reads them.
- A message-free step of a non-agent FSM can repeat the same extraction prompt on
  every step.
- `RunBudgetExceededError` does not carry the results of the steps already run (read
  the history and the current state).
- `agents/plan_execute_recovery` times out at 180 s under 4 eval workers (score 1;
  4 at `d4b1626`): the planner now writes a complete 6-step plan (one fetch per
  category per tool) that costs 20 LLM calls against 16 for `d4b1626`'s copied 4-step
  plan, which collected one category only. Run alone it succeeds in about 28 s. The
  pre-registered full-evaluation rule D-044 (b) is recorded as FAILED.
  `hierarchical_orchestrator` timed out once at 300 s in the agents-only re-run (also
  a timeout in the 2026-09-29 `d4b1626` baseline; one run, load or noise).
- PlanExecute costs more calls per task than at `d4b1626` (16.7 vs 15.3 in the live
  probe), a run that completes every plan step can still report `success=False,
  max_iterations` at small limits (also REWOO; pre-existing), and a plan step that
  needs no tool costs a wasted retry call.
- React-family success gap: a run can answer that it did something it never
  attempted and report `success=True` (`hitl_search_publish` answers "published"
  though the gated tool was never selected); a refused run that then answers through
  an ungated tool reports `(True, "answered")`. Forced Reflexion answers can cite
  numbers that are in no observation (also at `d4b1626`).
- Refusal record (`refused_actions`) limits; enforcement is unaffected, only the
  conclude wording can be wrong: the same call is matched by its redacted label text,
  so dict key order or `5` vs `5.0` makes an approved-then-denied repeat look new
  (also for the removal in `spend_grant`), and two calls that differ only in a
  secret-looking value share one record; ReasoningReact's built-in `reason` tool
  builds its own trace label, so the ran-check does not cover it; a failed earlier
  call counts as "ran", so a refused retry gets no record; a call whose empty input
  the executor filled from the task is traced under the filled label. Repeated asks
  of a denied call are not reduced (3 per denied React run).
- An agent `run_stream` whose conversation is ended from outside ends silently (it
  raised before), and no test pins an outside end that lands after the last step.
- Secrets inside free-text string values are not redacted (only secret-looking keys
  and value shapes are).
- The no-message Pass-2 prompt: 4 of 94 live replies were not a clean JSON envelope
  (2 salvaged, 2 accepted as plain text). Greetings of speaking initial states keep
  the conversational Pass-2 prompt. ADaPT `attempt`, `assess` and `decompose` still
  make a Pass-2 call whose text is only a last-resort answer. With the bulk calls
  gone, a typed field that comes back null in REWOO `plan_all` or the orchestrator
  has no second chance (the state takes its fallback edge). `evaluator_optimizer`
  strict runs scored lower in one live probe (5/6 vs 6/6).
- `build_fsm_graph` raises `TypeError`, not `ValueError`, for an unhashable
  `initial_state`.
- The examples binding guard (`tests/test_examples/test_example_calls_bind.py`) does
  not check calls on names whose type it cannot infer.
- The message-free `classification_extractions` path has offline test coverage only.
- Harness live L3 runs took 19.9 to 26.5 s against 15.1 to 17.9 s at `d4b1626`.

From the agents audit reviews (`docs/agents_roadmap.md`, Known open items):

- A long generated artifact (EvalOpt, MakerChecker, PromptChain) can still be cut off
  by the per-field `max_tokens` budget. It ships as a truncated answer, marked only by
  its WARNING and confidence 0.3; PromptChain has no judge, so such a run can report
  `success=True`. Fixing it needs an output-budget change (Track B).
- Orchestrator: subtasks dropped over `max_workers` (`skipped_subtasks`) are not shown
  to the collect judge and do not affect `success`; the answer may be incomplete.
- AgentGraph: a node with `success=False` (a forced pass, a Debate without agreement,
  a budget stop) takes no outgoing edge and no edge can route on failure, so a
  fallback branch cannot be built.
- The `TypeError` denylist of agent constructors includes every `HumanInTheLoop`
  constructor parameter, derived from its signature: a new HITL parameter with a
  generic name would start rejecting a litellm passthrough kwarg of that name.
- Core: validating `FSMDefinition`s on first use from several threads at once can race
  (pre-existing at `c632893`; seen with concurrent Debate panels).

## [0.11.0] - 2026-09-29

### Added (`fsm_llm.eval`, 2026-09-29)

- New subpackage `fsm_llm.eval` (extra `eval`, no third-party dependencies, included in
  `all`) and console script `fsm-llm-eval` (also `python -m fsm_llm.eval`). It gathers
  the evaluation code that lived in `scripts/eval.py` and the generic statistics and
  row helpers of `scripts/harness_bench.py`. Docs: `src/fsm_llm/eval/README.md`, `EVALUATE.md`,
  `docs/api_reference.md`.
- `fsm-llm-eval examples`: the examples evaluator with the same flags, 0-4 scores,
  example names and output layout as `scripts/eval.py`, plus `--examples-dir`,
  `--python`, `--config FILE` and `--fail-under PCT`. A golden table built from the
  old scorer at `0facf56` pins identical scores.
- `fsm-llm-eval run DATASET`: conversation evaluations. A dataset (JSON list, JSON
  object with `config` and `cases`, or JSONL) lists cases with an FSM (path relative to
  the dataset, or inline), optional `initial_context`, user `turns` and `expect`
  checks (`final_state`, `visited_states`, `context`, `context_keys`,
  `responses_contain`, `ended`). Each case runs `trials` times (default 3) through
  `fsm_llm.API`; the run writes `rows.jsonl` (one row per trial, flushed as it
  finishes), `results.json` (per-case and overall pass rates with Wilson 95% intervals)
  and `summary.md`. Once the conversation ends, remaining turns are not sent, and that
  fails the trial unless the case declares `ended`. Sample dataset:
  `evaluation/datasets/simple_greeting_cases.json`: three plumbing cases over
  `examples/basic/simple_greeting` and `name_is_extracted` over
  `evaluation/datasets/name_capture_fsm.json`, which fails for a model that extracts
  nothing.
- `EvalConfig`: every setting has a default and can be set, in increasing precedence,
  by a dataset's embedded `config`, a `--config` JSON file, or a flag. Unknown keys are
  an error. The per-example stdin and timeout tables can be extended or overridden from
  the config file. LLM settings for `run`: `temperature`, `max_tokens` (flags
  `--temperature`, `--max-tokens`) and `llm_kwargs`, extra `fsm_llm.API` arguments
  such as `api_base` (recorded in `results.json` by key only, never by value).
- Exit codes for both commands: `0` finished, `1` usage or input error (a file that is
  not UTF-8 or not JSON included), `2` only when `--fail-under PCT` is given and the
  score is below it (compared exactly in integers, so 57 of 100 meets 57), `130`
  interrupted. Ctrl-C cancels the work not yet started and writes the report files for
  what finished, marked `"interrupted": true`.
- Python API: `run_dataset(path, *, config=None, llm_interface_factory=None,
  checks=(), **overrides)` runs a dataset in one call with the CLI's settings layers;
  `checks` are callables that get each finished trial and return failure messages
  (also on `run_cases`). Also `run_examples`, `run_cases`, `load_cases`, `check_expectations`,
  `wilson_ci`, `fisher_exact_two_sided`, `pass_rate`, `append_row`/`read_rows`,
  `open_run_dir` and more; `run_cases(..., llm_interface_factory=...)` runs offline
  with a fake LLM. Exceptions `EvalError(FSMError)` -> `EvalConfigError`,
  `EvalDatasetError`.

### Changed

- `fsm-llm-eval examples` takes the same flags as the removed `scripts/eval.py`, with
  two behaviour changes: an `--output-dir` that already holds files is now refused
  (exit 1) instead of reused, and a usage error (unknown flag, bad value) exits 1
  instead of argparse's 2, because 2 now means "below `--fail-under`".
- `scripts/harness_bench.py` is unchanged and stays stdlib-only and offline: its
  `wilson_ci`, `fisher_exact_two_sided`, `append_row`, `read_rows`, `_write_json`,
  `_utc_now` and `_git_commit` are copies of the `fsm_llm.eval` helpers, kept equal
  by `tests/test_fsm_llm_eval/test_bench_parity.py`. Delegating them was tried and
  reverted: importing `fsm_llm` pulls litellm, which opens a network socket.
- Example scorecards: "Total wall time" is now the real elapsed time of the run; the
  old value (sum of example durations) is reported as "Total example time". Compare
  old and new scorecards on "Total example time".
- Example `results.json` gains `wall_time_s`, `workers`, `default_timeout`,
  `evaluator` and `interrupted`; every old key keeps its meaning.
- Example runs use a thread pool driving one subprocess per example instead of a
  process pool.

### Fixed

- The examples evaluator ran examples with a hardcoded `.venv/bin/python`; it now uses
  the running interpreter, overridable with `--python`.
- Two runs in the same minute on the same commit and model wrote into the same run
  directory; a new run now gets a `_2`, `_3`, ... suffix and never overwrites one.
- A worker that crashed (runner bug, OS error) was dropped from the scorecard; it is
  now recorded as a score-0 result with the error text.
- A run with no results no longer crashes the scorecard timing section.
- The default model is read from `$LLM_MODEL` when the run starts, not at import.

### Removed

- `scripts/eval.py`: use `fsm-llm-eval examples` or `python -m fsm_llm.eval examples`
  (same flags) from the repository root, or pass `--examples-dir`.
- `scripts/agent_chat_harness.py`, a one-off multi-turn agent diagnostic.
- `evaluation/datasets/oolong_synth_real_subset.jsonl`, which nothing read.

## [0.10.0] - 2026-09-29

### Changed (BREAKING: `fsm_llm` is the single top-level package, 2026-09-29)

The five extension packages now live under `fsm_llm` as subpackages. `src/` holds one
package, `src/fsm_llm/`. There are no compatibility shims: the old top-level names
raise `ModuleNotFoundError`, so update your imports.

| Old import | New import |
|------------|------------|
| `import fsm_llm_agents` / `from fsm_llm_agents import X` | `from fsm_llm import agents` / `from fsm_llm.agents import X` |
| `import fsm_llm_reasoning` / `from fsm_llm_reasoning import X` | `from fsm_llm import reasoning` / `from fsm_llm.reasoning import X` |
| `import fsm_llm_workflows` / `from fsm_llm_workflows import X` | `from fsm_llm import workflows` / `from fsm_llm.workflows import X` |
| `import fsm_llm_monitor` / `from fsm_llm_monitor import X` | `from fsm_llm import monitor` / `from fsm_llm.monitor import X` |
| `import fsm_llm_harness` / `from fsm_llm_harness import X` | `from fsm_llm import harness` / `from fsm_llm.harness import X` |

Submodules move the same way, for example `fsm_llm_agents.meta_cli` is now
`fsm_llm.agents.meta_cli` and `fsm_llm_monitor.server:app` is now
`fsm_llm.monitor.server:app`. Update `mock.patch` targets and `sys.modules` keys too.

**Existing clones:** after pulling this change, run `make clean` (or delete the old
`src/fsm_llm_<sub>/` directories by hand), then reinstall with `pip install -e .`. Git
leaves those directories behind because they still hold ignored `__pycache__/`, and an
empty leftover imports as a namespace package, so `import fsm_llm_agents` would succeed
instead of raising `ModuleNotFoundError`.

- `import fsm_llm` still loads no subpackage. `from fsm_llm import agents` works through
  the normal submodule import; `fsm_llm.agents` is not an attribute of a bare
  `import fsm_llm` until something imports it.
- Console scripts keep their names and point at the new modules:
  `fsm-llm-monitor = fsm_llm.monitor.__main__:main_cli`,
  `fsm-llm-meta = fsm_llm.agents.meta_cli:main_cli`,
  `fsm-llm-harness = fsm_llm.harness.__main__:run`. Reinstall (`pip install -e .`)
  to refresh editable wrappers.
- Module entry points: `python -m fsm_llm.reasoning`, `python -m fsm_llm.agents`,
  `python -m fsm_llm.monitor`, `python -m fsm_llm.harness`.
- **(behavior)** Logging: the single `logger.disable("fsm_llm")` now covers every
  subpackage, so agents, reasoning, workflows, monitor and harness logs are silent until
  `setup_logging()` or `enable_debug_logging()` runs (agents, reasoning, harness and
  monitor logs used to reach loguru's default stderr handler). The workflows
  self-disable is gone and `LIBRARY_LOGGER_NAMES` is `("fsm_llm",)`. The
  `fsm-llm-monitor` CLI enables library logging so its Logs page keeps working; if you
  run `uvicorn fsm_llm.monitor.server:app` yourself, call `setup_logging()` first.
  `fsm-llm-harness` and `python -m fsm_llm.harness` (the new `run()` wrapper) show
  library records at WARNING and above, so worker failures, halts and audit errors
  still reach the terminal; `FSM_LLM_LOG_LEVEL` overrides the level.
  `python -m fsm_llm.reasoning --verbose` shows them at INFO. `fsm-llm-meta` and the
  other `python -m` entry points stay silent by default.
- **(behavior)** Evaluation: `scripts/eval.py` classifies failures partly from each
  example's stderr, and the examples never enable logging, so agent examples no longer
  print extension warnings and tracebacks there. Eval scores from before and after this
  change are not comparable, for agent examples especially; re-baseline before
  comparing.
- **(behavior)** `disable_warnings()` still filters `fsm_llm` and its submodules, which
  now include the five subpackages, so their warnings are silenced too (before, they
  stayed visible). A lookalike sibling such as `fsm_llm_contrib` is still not matched.
- `has_agents()`, `has_reasoning()` and `has_workflows()` keep their names and now probe
  `fsm_llm.agents` etc. The subpackages ship in every install, so they return `True`
  whenever `fsm_llm` is installed. `get_agents()` and friends still raise the install
  hint only when the subpackage itself is missing; a missing third-party dependency
  propagates.
- The extras keep their names (`agents`, `reasoning`, `workflows`, `monitor`,
  `harness`) and now only add third-party dependencies; the code ships with the core.
- Packaging: the wheel's `top_level.txt` is just `fsm_llm`. The monitor frontend's
  `static/pages/`, `static/services/` and `static/utils/` subdirectories were missing
  from the wheel and sdist (package-data was `static/*`); `static/**/*` now ships all
  of them.
- The five sub-package `py.typed` markers are removed; `fsm_llm/py.typed` covers every
  subpackage under PEP 561.
- mypy and coverage run on one root: `mypy src/fsm_llm/`, `--cov=fsm_llm`.
- Test directories keep their names (`tests/test_fsm_llm_agents/` and so on).
- Version stays 0.9.0; this change ships in the next release.

### Fixed (monitor audit, 2026-09-28)

Breaking behavior changes are marked **(behavior)**.

- **(behavior)** Security: state-changing requests from a foreign `Origin` get 403, a
  foreign `Host` gets 400 (DNS rebinding), bodies over 1 MB get 413, and every response
  carries CSP / nosniff / frame-ancestors headers. Trusted hosts and CORS origins come
  from `FSM_LLM_MONITOR_TRUSTED_HOSTS` / `FSM_LLM_MONITOR_CORS_ORIGINS`; the permissive
  CORS origin regex is gone.
- **(behavior)** With an API key set, sensitive reads (conversations, activity, events,
  logs, agent status/result, builder result) and `/ws` require it. `/ws` authenticates
  with a first message `{type: "auth", api_key}` and closes 4401 (bad key) or 4403
  (foreign origin). `FSM_LLM_MONITOR_API_KEY` applies without calling `configure()`;
  new `GET /api/auth` and `--api-key` CLI flag.
- **(behavior)** Error mapping: unknown ids 404 (were 500), busy conversation 409,
  capacity 429 (`MonitorCapacityError`), missing extension 501, bad input 400; 500
  details are generic.
- Secret-looking context entries were shown by the snapshot, workflow status, agent
  result and tool parameters; one redaction path (`collector.redact_context`) now covers
  all of them, and non-JSON values are no longer `str()`-ed.
- Monitor handlers recorded the wrong states (the core never passes `_target_state`);
  they now capture current/target state in `should_execute`. Registration is idempotent,
  handlers are unregistered on destroy, and the POST_TRANSITION duplicate is dropped (7
  handlers per FSM, was 8).
- WebSocket streaming used timestamps and list lengths and dropped or repeated events
  after bursts or ring-buffer eviction; it now uses sequence cursors.
- Workflow runs: status, completion, failure and cancellation come from engine hooks;
  every run of an instance is tracked; unknown runs are 404; launch has a timeout and
  cleans up orphans; engines are shut down on destroy.
- Agents: `max_running_agents` cap, finished instances evicted at `max_instances`, a
  late cancel keeps the result, no double cancelled event, `last_tool` read from the
  conversation log.
- Conversations: terminal conversations are ended after caching their final state;
  `end_conversation` on an unknown id raises; a completed FSM instance reopens on start.
- Config: bounds on refresh interval and buffer sizes, `log_level` validated and applied
  to the log sink, buffers resized live. Request bounds on message, task, iterations,
  timeout and tools.
- Builder: sessions capped at 50 (429), TTL counts from last use, a timed-out send keeps
  the session busy until its thread finishes, delete while busy is 409.
- Dashboard config accepts the builder's unwrapped output; malformed input is 400.
- `get_logs` level filter is ordered and case-insensitive; blocking snapshot routes run
  off the event loop.
- OTEL: bounded open conversation spans, fixed span names, lifecycle spans carry ids only,
  `disable()` restores only its own wrapper.
- Frontend: API key prompt and retry, WebSocket auth, server-side internal-key filtering
  (client `startsWith('_')` removed), workflow "Send event" and "End conversation"
  controls, readable 422 errors, and many UI fixes (stale timers, log dedup/pause,
  drawer re-render, escaping, accessibility).

### Fixed (workflows audit, 2026-09-27)

Breaking behavior changes are marked **(behavior)**.

- `process_event`: one instance's failure (deadline passed, cancelled meanwhile, bad
  state) no longer aborts delivery to the other waiting instances, whose listeners had
  already been consumed. Delivery re-checks WAITING under the instance lock and no longer
  mutates the context of a non-waiting instance.
- A follow-up wait on the same event type kept its timeout: the old code cancelled the new
  wait's timeout after the transition.
- A wait with `timeout_seconds` and no `timeout_state` now FAILS the instance on timeout
  instead of staying WAITING forever; `success_state=""` completes the instance when the
  event arrives. `timeout_state` without `timeout_seconds` is rejected.
- `LLMProcessingStep` works with a core `LLMInterface` (`generate_response`) and with a
  sync or async `generate(prompt)`; before, every core interface failed with
  `AttributeError`.
- **(behavior)** Failures never fall back to the success route: LLM, conversation,
  parallel and agent steps without `error_state` now FAIL the instance, and an agent
  reporting `success=False` fails its step.
- `ParallelStep` counts every failed child (an exception with an empty message was
  aggregated as success), drops internal keys before the `step_<i>_` prefix, rejects
  wait/timer children, keeps the successful children's data on failure, and falls back to
  a shallow copy for contexts that cannot be deep-copied.
- User callables that return a coroutine (a lambda wrapping an async function, an object
  with `async __call__`) are awaited: a `ConditionStep` no longer always takes
  `true_state` and an `APICallStep` no longer maps nothing.
- Cancelling instance `order` no longer cancels the timers of `order_2`; a custom
  `instance_id` still held by the engine is rejected instead of overwriting it.
- The workflow deadline is enforced while WAITING; `WorkflowTimeoutError.instance_id`
  names the instance when `start_workflow` raises.
- **(behavior)** Only synchronous cycles are rejected at registration: loops through a
  timer or event wait are allowed. The recursive driver with a 20-step depth cap is
  replaced by a loop with `max_steps_per_run` (default 1000).
- Events can be targeted (`WorkflowEvent.instance_id`, buffered until the instance
  waits) and correlated (`WaitEventConfig.correlation_key`).
- **(behavior)** `switch_step` defaults to `default_state=None` like `SwitchStep` (an
  unmatched value FAILS instead of completing); its outputs are now
  `switch_<id>_matched`/`_target` (the old `_switch_*` keys never reached the context).
- `AgentStep` maps `"answer"` even with an empty `final_context`, passes `input_mapping`
  as `initial_context`, accepts a plain string result, and adds per-step
  `agent_<id>_answer`/`_success`.
- `ConversationStep` honours the base `timeout`, reports `conversation_<id>_ended`, can
  `require_completion` and `use_user_input`, accepts an `FSMDefinition` object, and
  rejects a missing or doubled FSM source at construction.
- Resources: fired event timeouts leave no timer entry; timers, listeners and buffered
  events are released on every terminal status; failed instances are purged too;
  **(behavior)** `max_completed_instances` defaults to 1000 and history to 1000 entries
  per instance.
- Running instances keep the definition they started with; the engine stores a copy on
  registration.
- `fsm_llm_workflows` logging is off until `setup_logging()`/`enable_debug_logging()`,
  like the core package (new `fsm_llm.logging.enable_library_logging`).
- Smaller: `error_state` on `AutoTransitionStep`/`ConditionStep`; float timer and wait
  durations; every DSL factory exposes `timeout`; history records step errors without
  redundant "running" entries; `DependencyResolver.has_cycles` no longer reports an
  unknown dependency as a cycle; exception constructors copy `details`;
  `register_event_listener` raises `WorkflowEventError` for an empty event type;
  duplicate step ids raise `WorkflowDefinitionError`; `APICallStep` output paths
  (`"a.b"`, `""`); `serialize()` strips custom callables; `shutdown()` awaits its tasks
  and refuses new starts; `start_workflow(wait=False)`; `get_workflow_context` returns a
  copy; lifecycle hooks (`add_hook`); injectable `executor`; `WorkflowHistoryEntry`
  exported; new `constants.py`.
- Event delivery only wakes an instance still at the wait the listener was registered
  for; leaving a step drops its listeners and wait timers; a cancelled `process_event`
  restores the listeners it had not delivered; a background start never runs a step of
  an instance that was cancelled or driven first; long prompts fit the core
  `ResponseGenerationRequest` limits.
- Monitor: workflow status redacts secret-shaped context and history entries; new
  `send_workflow_event` / `POST /api/workflow/{id}/event` (API-key gated).


### Changed

- **Default model is `ollama_chat/qwen3.5:4b` again.** 0.9.0 changed `DEFAULT_LLM_MODEL`
  (and the `scripts/eval.py`, `scripts/agent_chat_harness.py` and
  `react_worker_factory` defaults) to `ollama_chat/qwen3.5:9b-q8_0`; this reverts all of
  them and the docs that state the default. Set `LLM_MODEL` or pass `model=` to use 9b.
  The 0.9.0 live results below were measured on 9b and stay as recorded.

### Live results for 0.9.0 (measured after the release)

Measured on `ollama_chat/qwen3.5:9b-q8_0` (the new default) after v0.9.0. Raw records
were kept outside the repository; numbers are reported as measured.

- `scripts/eval.py`, all 101 examples, `--workers 4`, N=1: 80.9% (327/404). Almost every
  loss is an agent example hitting its eval timeout (the per-example timeouts were tuned
  on the faster 4b model). This is not comparable to the old 95.3% 4b baseline.
- Controlled A/B on the 48 agent examples, same model, workers and timeouts, run back to
  back: pre-plan code (c6e8461) 73.4% (141/192, 17 timeouts); 0.9.0 77.1% (148/192,
  14 timeouts). Six examples improved (both debate examples, pipeline_review,
  react_hitl_combined, reasoning_stacking, concurrent_react) and three went from pass to
  timeout (plan_execute, agent_as_tool, hierarchical_orchestrator). The logs show those
  three progressing, not looping: plan_execute now runs every step's tool (it did not
  before), so a 5-step plan needs more than the 180 s example timeout.
- F-LIVE-02: `TestLiveMemoryAgent` passed 3 of 3 runs (81 s, 68 s, 83 s), up from 1 of 2
  and then 0 of 2 before the follow-up. Runs were sequential with nothing else on the
  GPU.
- FB-05 probe (one orchestrator and one debate run per side): before, the orchestrator
  never extracted `all_collected` (null) and debate stayed at `current_round=1` after 12
  iterations; after, the orchestrator extracts `all_collected=True` and debate runs its
  2 rounds to `current_round=3` in 8 iterations.

## [0.9.0] - 2026-09-24

### Agents follow-up 2026-09-24

Follow-up to the agents audit below (`plans/plan-2026-09-24T091842-c1d5bfbc`, one
commit per step or completion fix, three adversarial review passes). It closes the
audit's HITL security gaps and most of its Known open list, makes CI green on
Python 3.10, 3.11 and 3.12, and changes the default model. Every behaviour change has a
test that fails on the pre-fix code. Full suite: 7,158 tests collected (was 6,940). Core `src/fsm_llm` changed only in
`DEFAULT_LLM_MODEL` and one comment. Reviews and decisions: the plan's
`findings/review-iter-1*.md` and `decisions.md` (D-001 to D-030).

Live results (ollama_chat/qwen3.5:9b-q8_0): measured after the release, see
"Live results for 0.9.0" above.

### Security -- agents follow-up 2026-09-24

- **Approval is a driver-only grant bound to one call (D-004, D-023).** In
  `ReactAgent`, `ReflexionAgent` and `ReasoningReactAgent` with an approval policy, a
  tool the policy gates runs only when the approval driver wrote a grant for that exact
  call: the internal key `_approval_granted` (`ContextKeys.DRIVER_APPROVAL`), whose
  value is `{"tool_name", "parameters"}` of the call the human approved. Core drops
  internal-prefixed keys from every model extraction, so the model cannot write it.
  `AgentHandlers(registry, requires_approval=...)` checks it before a registered tool
  runs (`approval_refusal`): without a matching grant the tool does not run, no
  observation is recorded (a refused call is never conclude evidence), the selection
  is kept, `approval_required` is set, a model-written `approval_granted` is deleted,
  and `tool_status` is `awaiting_approval`, so the driver asks on the next loop
  iteration. The grant is spent by the call it names, so one approval still covers one
  call (D-015). A call changed after the approval is refused: pre-fix `ReactAgent` ran
  `danger(x="EVIL")` after the human approved `danger` with an empty input and the
  model filled the input on the `await_approval` turn. The public
  `approval_granted`/`approval_required` keys still route the FSM and stay
  model-writable; the refusal, not the route, is the boundary.
- **Reflexion asks before a gated tool (D-005).** `build_reflexion_fsm` takes
  `include_approval_state` and builds the same `await_approval` state as
  `build_react_fsm` (shared builders `_await_approval_state` and
  `_approval_think_transition`). Before, a gated tool ran first and the callback was
  asked afterwards, so a denial did not stop it.
- **ReasoningReact has an approval driver (D-005).** `ReasoningReactAgent` now calls
  `_handle_hitl_approval` before each turn, as `ReactAgent` does. Before, the callback
  was never asked and the model opened the gate by extracting `approval_granted`.
- **One approval predicate for all three agents (D-019).** `_hitl_active` and
  `_approval_predicate` moved to `BaseAgent`. The same predicate builds
  `await_approval`, registers the gate and feeds the refusal, so they cannot disagree.
- **The approval is a strict bool (D-023).** The driver asks unless `approval_granted`
  is exactly `True` and stores the callback's answer as `True` or `False`. A
  model-written `"yes"` or a callback returning `None` used to park the run in
  `await_approval` until `BudgetExhaustedError`.
- **The driver asks only about a real gated tool, and sees the full context (D-005).**
  A model that writes `approval_required=True` with tool `none`, an unknown tool or an
  ungated tool is no longer asked about; the driver routes the run back to `think`.
  The policy is evaluated on the same full context (internal keys included) the
  refusal sees, so a policy that reads a `_`-prefixed key asks and runs as expected.
  The approval callback still receives the filtered `get_data` view.
- **`AgentServer` opt-in API key and input limit (D-006).**
  `AgentServer(..., api_key=None, max_input_chars=100_000)`. With `api_key` set,
  `/invoke` and `/stream` need `Authorization: Bearer <key>` or `X-API-Key: <key>`
  (401 otherwise; `hmac.compare_digest` on UTF-8 `surrogateescape` bytes, a copy of the
  monitor's compare). `/health` and `/info` stay open. A request whose
  `len(task) + len(json.dumps(context, ensure_ascii=False))` exceeds `max_input_chars`
  gets 413 before the agent runs; `None` disables the check. The key is checked before
  the size and before body validation. `RemoteAgentTool(..., api_key=None)` sends the
  key as a bearer token on `invoke` and `ainvoke`. An empty or whitespace-only
  `api_key` raises `ValueError` in both classes: `compare_digest(b"", b"")` is true, so
  `api_key=os.getenv("KEY", "")` used to give an open server that looked protected.
  The server is still unauthenticated by default.
- **Monitor `configure(api_key="")` raises `ValueError` (D-006).** The same empty-key
  hole existed in the monitor's mutating routes. An empty or whitespace-only `api_key`
  now raises before any global changes, so a key already configured stays in force.
  The env var `FSM_LLM_MONITOR_API_KEY=""` keeps its documented meaning (no key).

### Behaviour changes to know about -- agents follow-up 2026-09-24

- **Default model is `ollama_chat/qwen3.5:9b-q8_0` (D-003).** `DEFAULT_LLM_MODEL`, the
  harness default (which follows it), `react_worker_factory`, `scripts/eval.py`,
  `scripts/agent_chat_harness.py` and the docs changed from `ollama_chat/qwen3.5:4b`.
  New users who rely on the default download a larger model (about 10 GB). Bench
  records and "measured on 4b" notes are unchanged.
- **Pins: `litellm==1.102.1` and `mcp==2.2.0` in `constraints.txt` (D-002, D-025).**
  litellm 1.100.1 and 1.100.2 import `typing.NotRequired` (3.11+) and fail to import on
  Python 3.10; 1.101.0 is the first fixed release. One pin for every Python version,
  no marker split. mcp is pinned to the version CI ran green, since mcp 2.x already
  renamed `Tool.inputSchema` once.
- **Reflexion and ParallelReact conclude only on tool evidence (D-008).** Their
  `think->conclude` and `act->conclude` edges now carry ReAct's guard
  (`should_terminate` and (`observation_count > 0` or `max_iterations_reached`)), built
  by `_conclude_on_evidence_logic()`. Reflexion's `evaluate->conclude` edge needs
  `evaluation_passed` plus the same evidence, or a forced stop
  (`max_iterations_reached` alone). A turn-1 answer from memory now goes through `act`;
  a Reflexion run that never calls a tool ends at the limiter's forced stop instead of
  concluding on turn 3. Reflexion at `max_iterations=1` no longer raises
  `BudgetExhaustedError`. ParallelReact's empty-batch branch does not clear
  `should_terminate`, so a model that insists on stopping with no batch loops until the
  forced stop (bounded, `max_iterations + 2`).
- **maker_checker forces a pass only in `check` (D-007, D-021).** The limiter still
  counts every turn, but the forced `checker_passed=True` is written only on the
  `check` state (new handler `MakerCheckerForcePass`), and `check->output` also passes
  on `max_iterations_reached == True`. Every shipped draft is one the checker judged;
  before, odd budgets could ship the draft written on the forced `revise` turn
  unjudged. Measured with an always-reject checker: budgets 2 to 10 take the same
  number of turns as before the plan; budget 1 ships a judged draft in 2 turns
  (before the plan it shipped an unjudged one).
- **evaluator_optimizer ends at every budget (D-022).** The limiter's forced
  `evaluation_passed=True` was overwritten by the next evaluation, so with an
  always-failing evaluator the run could raise `BudgetExhaustedError` (budgets 1, 2,
  3 and 10 with the defaults, 2 to 7 with a high `max_refinements`). `_run_evaluation` now
  treats `max_iterations_reached` like the refinement cap and ships the last evaluated
  output.
- **A forced stop reports a verdict the judge did not give.** maker_checker's forced
  pass and evaluator_optimizer's budget stop return `success=True` with
  `checker_passed`/`evaluation_passed=True` in `final_context`. Read
  `max_iterations_reached` to tell a forced stop from a real pass.
- **Orchestrator and debate decisions are typed bools (D-009).** The orchestrator
  `collect` state and the debate judge declare `all_collected` and
  `consensus_reached` as `field_type: "bool"` extractions (one extra extraction call per
  such turn). `all_collected` is cleared on entry to `orchestrate` and
  `consensus_reached` on entry to `propose`, so each round is decided again; before,
  the first round's False stuck and later values were never extracted. The debate no
  longer seeds `consensus_reached=False`, so the model's verdict is extracted: a
  round-1 consensus now ends after 1 judge (it ran 3 or 5 before). The judge->conclude
  edge reads `consensus_reached == True` or `current_round > max_rounds`, so the round
  cap holds in every round (it only worked in round 1).
- **meta_builder `step_type` is an enum (D-010).** The extraction schema sets
  `"enum": sorted(WorkflowBuilder.VALID_STEP_TYPES)`, so a live model cannot invent a
  step type. The agentic `meta_tools.add_step` parameter is still a free string.
- **List tool input stays a list (D-011).** `normalize_tool_input` turns a list into
  `{"input": list}` instead of a string repr. A list reaches a parameter whose schema
  type is `array` as a list; a string-typed or untyped parameter still gets
  `str(list)`, the pre-fix value.
- **`@tool` infers an array schema for list-like annotations (D-011).** Bare `list`,
  `list[X]`, `typing.List[X]`, `tuple[...]`, `set[...]`, `frozenset[...]`,
  `Sequence[X]`, and any of these inside `Optional`/`X | None`, give
  `{"type": "array"}`, with `items: {"type": ...}` for a parametrized generic (OpenAI
  native function calling rejects an array without `items`). They used to be
  `"string"`. The prompt now tells the model to send an array for these parameters.
  Other `Optional` types are unchanged. An empty `tool_input` on a single list-typed
  parameter is filled with `[task]` (D-030); it used to get the bare task string.
- **A `**kwargs`-only tool is called with keywords (D-024).** A tool whose one
  parameter is `**kwargs` used to receive the parameters dict positionally and fail on
  every call. This covers zero-argument MCP tools and the monitor's launched-agent stub
  tools.
- **A single `params` dict tool typed `dict[...]` still gets its dict (D-024).** Only a
  bare `dict` annotation was recognised; `dict[str, Any]`, `typing.Dict[...]` or a
  string annotation (`from __future__ import annotations`) made every call fail. Such
  a parameter is the legacy dict form only when the schema does not name it.
- **`previous_draft` and `previous_output` are dropped from `final_context` (D-012).**
  Checker and refiner prompts still see them during the run
  (`RESULT_DROPPED_CONTEXT_KEYS`).
- **`max_iterations_reached` is always present and cannot be forged (D-007).**
  `_init_context` seeds it `False`, so it appears in every prompt context and in
  `final_context` of the agents that use `_init_context`. Core bulk extraction fills
  only unset keys, so a model that writes `max_iterations_reached: true` no longer
  passes every evidence guard (it concluded React and Reflexion on turn 1 and made
  maker_checker ship its first draft). Only the limiter and stall handlers write True.
- **ReasoningReact builds `await_approval` for any approval policy (D-019).** It used
  to need a policy and at least one tool flagged `requires_approval`. A policy can gate
  any tool, including unflagged ones and the `reason` pseudo-tool.
- **A gated call with no approval callback raises in Reflexion and ReasoningReact
  too.** With a policy and no `approval_callback`, the driver's
  `request_approval` raises `ApprovalDeniedError`, which ends the run as
  `AgentError`, as `ReactAgent` already did. Before, Reflexion ran the tool and
  ReasoningReact let the model approve.

### Fixed -- agents follow-up 2026-09-24

- `AgentServer` returned 422 for every valid `/invoke` and `/stream` body: the request
  models were defined inside `_create_app`, and under `from __future__ import
  annotations` FastAPI read `request` as a query parameter. They are module-level now
  (D-020). No test had ever called the server.
- plan_execute ran each step's tool on `execute_step` entry, before the step's
  `tool_name`/`tool_input` were extracted, so a 1-step plan never called its tool and
  step N ran step N-1's input; every step also recorded `success: False` because the
  compactor had cleared `tool_status`. The executor now runs on `check_result` entry,
  and the checker runs after it in the same cascade (D-024).
- mcp 2.x renamed `Tool.inputSchema` to `input_schema`, so every discovered tool had an
  empty schema and every call failed. `MCPToolProvider` reads either name (D-025).
- An MCP tool error result (`isError`/`is_error`) is a failed tool call. It used to be
  reported as `success=True` and counted as evidence (D-026).
- `SkillLoader` loads a `@tool` reachable under two attribute names once. The first
  attribute in `dir()` order wins; a different function claiming a loaded name is
  skipped with a warning (D-012).
- CI's Python 3.10 leg was red since v0.8.0 (litellm import error, see the pin above).
  Two latent mypy errors in `mcp.py` and four OTEL tests that ran without the
  OpenTelemetry SDK surfaced once CI installed the `mcp` extra; both fixed.

### Tests -- agents follow-up 2026-09-24

- `tests/test_fsm_llm_agents/test_hitl_security.py`: forged `approval_granted` and
  `_approval_granted` with a deny-always callback (tool never runs, callback asked), a
  swapped or filled-in call after approval, Reflexion asking before the tool,
  ReasoningReact asking its callback, the driver asking only about a real gated tool,
  and a policy that reads an internal key.
- `tests/test_fsm_llm_agents/test_remote.py`: the 401/200/413 matrix on both routes,
  bearer and `X-API-Key`, non-ASCII and empty keys, at-limit and over-limit input, and
  `RemoteAgentTool` sending the key.
- `tests/test_fsm_llm_agents/test_forced_stop_flag.py`: a model-written
  `max_iterations_reached` changes nothing in React, Reflexion and maker_checker.
- `tests/test_fsm_llm_agents/test_mcp_stdio.py` with the fixture
  `mcp_fixture_server.py`: real stdio discovery and call, a slow call timing out with
  the child process gone within 5 s, a hung server, and an erroring tool. They run in
  CI, which now installs the `mcp` extra; mcp 1.x was checked only in a local Python
  3.10 venv.
- `tests/test_fsm_llm/test_turn_guard_deterministic.py`: a deterministic
  same-conversation turn-guard test (an Event-blocked mock LLM; exactly one success and
  one "already being processed"), shown to fail with the guard disabled. The live
  `TestThreadSafety` counts only guard outcomes and skips on live-model errors.
- Extended: maker_checker and Reflexion budget sweeps (1 to 10), evaluator_optimizer
  budget stops, `early=True` limiter boundaries for orchestrator, debate and
  prompt_chain, the unknown-then-real tool run asserting success, the harness
  non-native `ReactAgent` path, list and dict parameter dispatch, the SkillLoader alias
  case, the meta_builder enum, and the monitor's empty-key and stub-tool cases.

### Known open -- agents follow-up 2026-09-24

- An approved call can be dropped: the model selects a gated tool with
  `should_terminate=True`, the human approves, and `await_approval->conclude` fires
  before `act`. The tool never runs and the run reports `success=False` (fails closed,
  but the human was asked about an action that never happened).
- The per-tool `requires_approval` flag does nothing without an `approval_policy`.
  Agents with no `hitl` parameter have no approval at all: ParallelReact, REWOO,
  `NativeFunctionCallingReactAgent` and PlanExecute.
- The approval policy sees the call before empty-input recovery, so a
  parameter-sensitive policy can pass `{}` while the tool then runs with the task text
  filled in. The policy gets a shallow copy of the live context, read outside the
  conversation lock; write policies as pure predicates. The driver now calls the policy
  on every turn that selects a registered tool.
- The react-family success rule ("an answer key or at least one tool call") counts
  failed tool calls and forced stops as success: a model that never calls a tool
  reports `success=True` once the budget runs out.
- plan_execute's recorded step `result` is the model's text from before the tool ran
  (only `success` reflects the real call), and a failed tool does not set
  `step_failed`, so it does not trigger a replan.
- ParallelReact: an empty batch with `should_terminate` loops to the limiter (bounded,
  no stall detector).
- Debate: `proposition`, `critique`, `counter_argument` and `judge_verdict` stick
  across rounds (core fills only unset keys), so rounds 2+ judge round-1 text; a judge
  turn whose consensus extraction is null is not counted as a round.
- `@tool` item-type inference: `list[Any]` and nested lists give
  `items: {"type": "string"}`, and `tuple[int, str]` gives `items: {"type": "integer"}`.
  Not worse than the old `"string"` schema.
- A string (or a JSON-array string) sent to a list-typed parameter reaches the tool as
  a `str`, so a type-correct tool iterates its characters (pre-existing).
- `AgentServer` has no rate limiting (use a reverse proxy), and the HTTP body is parsed
  before the size check.
- MCP reconnects on every call, so the 30 s per-call timeout includes starting the
  server. mcp 1.x was tested only locally. `constraints.txt` does not pin pydantic
  (fresh installs resolve 2.13.5).
- Out of scope, recorded: core PRE_TRANSITION handlers do not run on a BLOCKED turn;
  `validate-plan` reports pre-existing errors (`pyproject.toml:33`,
  `fsm_llm_monitor/server.py:1382`) and orphaned decision anchors; `src` grew by about
  +570 net lines in this plan (5 pre-existing bugs found on the way), and
  `fsm_definitions.py` keeps growing.
- F-LIVE-02 live re-check on `ollama_chat/qwen3.5:9b-q8_0`: 3 of 3 passed after the
  release, see "Live results for 0.9.0" above.

### Agents audit 2026-09-24

Audit of `src/fsm_llm_agents` dated 2026-09-24 (`plans/plan-2026-09-24T045559-3e4eb3e5`,
13 steps and 9 completion fixes after review, one commit each). Finding ids below are
the audit's own. Every behaviour change has a test that fails on the pre-fix code,
except the logging and docstring items. Full suite: 6,940 tests collected (was 6,873). `ruff` and `mypy` clean across
all 6 packages. The ReAct-family routing change (first item) changes how a live model's
turns are routed, so the already stale eval baseline was not re-measured.

### Behaviour changes to know about -- agents audit 2026-09-24

- **`think` can no longer stall on a bad tool name (CR-01, the F-LIVE-02 mechanism).**
  In `build_react_fsm` (and so `ReasoningReactAgent`), `build_reflexion_fsm` and
  `build_parallel_react_fsm`, the `think->act` edge used to require a valid tool
  selection. A null or unknown tool name left `think` BLOCKED, and on a BLOCKED turn
  neither the `act` entry handler nor any PRE_TRANSITION handler runs, so the run
  burned the 3x loop ceiling and raised `BudgetExhaustedError`. `think->act` is now the
  unconditional lowest-priority (300) fallback, so the existing no-tool feedback,
  stall detector and iteration limiter run and the agent concludes. The `conclude` and
  `await_approval` edges are unchanged. Every non-concluding `think` turn now goes
  through `act`, which gives corrective feedback where the turn used to sit in `think`
  silently.
- **An unknown tool name is a no-tool turn, not evidence (D-012).** In
  `AgentHandlers.execute_tool`, a tool name that is not `none` and not registered now
  takes the no-tool path: the model gets `Unknown tool '<name>'.` plus the list of
  available tools, the turn counts toward the 3-turn stall detector, and no observation
  is recorded. Before, it recorded a `[TOOL FAILED]` observation that satisfied the
  `think->conclude` evidence guard, so a hallucinated tool plus `should_terminate=True`
  ended with `success=True`. The turn's `tool_status` is now `skipped`, not
  `failed`. Unknown-name and no-tool turns (including the literal `none`) now clear
  `tool_name` and `tool_input` as a real tool call does, and the plain warning turn
  also clears `should_terminate`. Core extracts a key only while it is unset, so a
  kept name used to block every later real tool call. `ParallelReactAgent` is
  unchanged.
- **Fallback edges out of more loop states (FB-01, D-002).** A turn where the routing
  key was never extracted used to BLOCK the state until `BudgetExhaustedError`, because
  no PRE_TRANSITION limiter runs on a BLOCKED turn. Unconditional priority-900 edges now
  continue the loop: maker_checker `check->revise`, ADaPT `assess->combine` and
  `decompose->combine`, evaluator_optimizer `generate->evaluate`. Every agent loop state
  now has an unconditional fallback except plan_execute `plan` (its key is seeded) and
  `await_approval`. In ReactAgent the approval driver always writes a bool before
  `await_approval` routes. ReasoningReactAgent uses the same state with no driver, so
  it can BLOCK there until the 3x ceiling (see Known open).
- **maker_checker and evaluator_optimizer re-judge every round (D-013).** Core extracts
  a key only while it is unset, so maker_checker judged `checker_passed` and
  `quality_score` once per run and `revise` never replaced `draft_output`; the
  evaluator_optimizer `refine` state never replaced `generated_output`, so the
  evaluation function scored the first draft on every round. Entering `revise`/`refine`
  now moves the draft to `previous_draft`/`previous_output` and clears the draft (and,
  in maker_checker, a False verdict), so the revised draft really replaces the previous
  one and is judged again. Leaving `revise` clears the consumed `checker_feedback`, so
  the next check writes fresh feedback; if no new draft was produced, the previous one
  is restored. When `checker_passed` is already True (a forced pass from the quality
  auto-pass or `max_revisions`), entering `revise` changes nothing, so the draft the
  checker judged is the one that ships; the forced pass still costs one extra
  `revise` turn. One exception (see Known open): when the iteration limiter forces
  the pass on a `revise` turn, the new draft skips `check` and ships unjudged; the run
  reports `max_iterations_reached=True`. `previous_draft`/`previous_output` stay in `final_context` and in
  later prompts (the checker sees both drafts); answer extraction never reads them.
  `max_revisions` now ends the maker_checker loop (always-False checker at
  `max_iterations=10`: 12 turns with 1 verdict before, 8 turns with 3 verdicts now).
  Two prompt lines changed to point at the new keys (`Your previous output is in
  'previous_output'.`, `Your previous draft is in 'previous_draft'.`); this was not
  measured on a live model. New handler factory `make_redraft_handlers`.
- **Iteration limiters share one factory (PT-05/PT-06/FB-06).** The 8 hand-rolled
  PRE_TRANSITION limiters (plan_execute, rewoo, evaluator_optimizer, maker_checker,
  prompt_chain, debate, orchestrator, adapt) now use
  `fsm_llm_agents.handlers.make_iteration_limiter(max, forced, *, context_max_key=None,
  early=True)`. With `early=True` it forces its keys at `count >= max_iterations - 1`
  (only adapt did this before); maker_checker passes `early=False` and triggers at
  `max_iterations` as before, so the checker judges at every budget (D-014). The early
  trigger changes routing only where a transition reads the forced keys: orchestrator,
  debate and prompt_chain now stop one iteration earlier (adapt already did). For plan_execute, rewoo
  and evaluator_optimizer the forced keys do not change routing (no transition reads
  them, the loop is linear, or `_run_evaluation` overwrites them). `DebateAgent.run()`
  no longer stores `_max_fsm_iterations` on the instance (PT-02).
- **HITL approvals are single-use (D-015).** `approval_granted` was never reset, so
  after one granted approval every later approval-required call skipped the callback
  and ran. This approval bypass predates the audit. `execute_tool` now deletes
  `approval_granted` at `act` entry once no approval is pending, so each gated call is
  asked for exactly once. This holds for ReactAgent only. It does not make approval a
  security boundary: ReflexionAgent, ReasoningReactAgent and model-written approvals
  have open gaps (see Known open).
- **Empty `tool_input` recovery checks the parameter type (CR-02).** When a tool with a
  single required parameter is called with no input, the task string is used as that
  parameter only when its schema type is absent, `"string"`, or a list containing
  `"string"`. A property schema that is not a dict (a JSON-Schema `true`, a description
  string) counts as untyped. An integer parameter used to receive prose and fail with a
  `TypeError`; the call now fails with `Tool requires parameters: [...]`, naming what
  the model must supply.
- **MCP calls time out (MM-06).** `MCPToolProvider`, `from_stdio` and `from_url` take
  `timeout` (default `Defaults.MCP_TIMEOUT_SECONDS = 30.0`; `None` means no limit). A
  discovery that runs past it raises `AgentTimeoutError`; a hung tool call raises
  `ToolExecutionError`, which `ToolRegistry.execute` turns into a failed `ToolResult`.
  Before, both could hang forever. The executor reconnects on every call, so the limit
  is per call and includes starting the server and `initialize`: a long-running tool
  that worked before can now fail by default (pass a larger `timeout` or `None`).
  Teardown of a real stdio server under cancellation is untested; on Python 3.10/3.11
  a slow teardown can run past the timeout.
- **Unknown workflow `step_type` is a validation error (MM-01/MM-02).**
  `WorkflowBuilder.validate_complete` reports a step whose `step_type` is not in
  `VALID_STEP_TYPES` as an error (`add_step` still only warns). The meta-builder's
  extraction schema types `step_type` as a free string, so a live model that invents a
  step type now gets `MetaBuilderResult.is_valid == False` where it used to get True
  (visible as `artifact_valid` in `examples/meta/build_workflow`). `is_valid` for
  workflow and agent artifacts means a structurally complete spec, not a directly
  loadable object; the docstrings now say so.
- **Duplicate tool registration warns (FB-07).** `ToolRegistry.register` logs a warning
  when a name is already registered. The last registration still wins.
- **`SkillLoader` loads a skill once (PT-07).** A name already provided through a
  module's `SKILLS` list is no longer registered a second time by the `@tool` scan.
- **HITL logs a late callback error (CR-05).** An approval callback that raises after
  the approval timeout (the request was already denied) is now logged instead of
  dropped. Two small timing changes come with it: a callback that returns between the
  join timeout and the internal lock is now honoured (it used to be denied), and a
  callback that raises a `BaseException` (not an `Exception`) is logged as a timeout
  and denied.

### Removed -- agents audit 2026-09-24

- `MetaBuilderConfig.output_path` (MM-07): declared and documented but never read. A
  caller that still passes it is ignored, as before (pydantic `extra="ignore"`).
- `fsm_llm_agents.prompts.build_think_response_instructions`,
  `build_act_response_instructions`, `build_evalopt_evaluate_response_instructions` and
  `build_checker_response_instructions` (FB-04): no caller anywhere in the repository.
- The `".." in Path(path).parts` check in `meta_output.save_artifact` (MM-03): it ran
  after `.resolve()` and so never fired. `save_artifact` trusts its caller and writes to
  the resolved path; the docstring says so. There was never any path confinement.

### Fixed -- agents audit 2026-09-24

- `test_reasoning_react.py` handler-reset tests no longer make a live LLM call when
  Ollama is reachable (RB-05). New smoke tests for `fsm-llm-meta` and
  `python -m fsm_llm_agents` (MM-09); their docstrings drop options that do not exist
  (CR-04).
- Internal cleanup with identical FSM output: `ContextKeys` constants replace 56 context
  key literals in `fsm_definitions.py` and one `_finalize_fsm` helper applies the
  description truncation rule (FB-02/FB-03); shared persona and meta-tool factories
  (FB-08/MM-08); unreachable branches removed in `native_fc.py` and `hitl.py` (CR-03).

### Known open -- agents audit 2026-09-24

Closed by the agents follow-up above: `AgentServer` auth and size limit, FB-05 typing
(the live check is still pending), CR-06, the three HITL gaps, the maker_checker
unjudged draft, the Reflexion/ParallelReact evidence guard and the `SkillLoader`
alias. Each closed item below is marked; the text is kept as it was.

- F-LIVE-02 live check, raw outcome, not a fix claim: `TestLiveMemoryAgent` on `ollama_chat/qwen3.5:9b-q8_0` ran twice on the final
  commit. Run 1 failed with `AgentTimeoutError` (120 s) after the first `remember`
  call (a cold model, with test collection running on the same machine); run 2 passed
  in 85 s. Both agents in run 2 reached `Iteration 5/5` before concluding. 1 of 2 is
  not enough to call the issue closed, and the base commit was not re-run.
- **Closed by the follow-up above (D-006).** `AgentServer` (A2A) has no authentication and no request size limit (MM-05). Fine for
  local use; it needs its own design before it is exposed.
- **Closed by the follow-up above (D-009; live check pending).** FB-05: `all_collected` and `consensus_reached` are read by transitions but not
  declared for extraction; confirming whether this bites needs a live extraction check.
- **Closed by the follow-up above (D-011).** CR-06: a list `tool_input` value is stringified (the known trade-off of the `any`
  union type).
- Core: PRE_TRANSITION handlers do not run on a BLOCKED turn. This fix works around it
  in the agents package; changing it would change handler timing for every FSM.
- **Closed by the follow-up above (D-004, D-005; remaining gaps listed there).** **HITL approval gaps (security, pre-existing, not fixed).** Do not rely on
  approval to guard a dangerous tool yet.
  (a) ReflexionAgent has no `await_approval` state: an approval-required tool runs
  first and the callback is asked afterwards, so a denial does not stop it.
  (b) ReasoningReactAgent builds `await_approval` (a policy plus a
  `requires_approval` tool) but has no approval driver: the callback is never asked,
  and the model opens the gate by extracting `approval_granted` (the state's
  extraction prompt asks for it). If the model never extracts it, the state BLOCKS
  until the 3x ceiling.
  (c) `approval_granted` is an ordinary context key the model can extract, so a bulk
  extraction in any state (for example `think`) can approve a call itself, in
  ReactAgent too; a callback that always denies is then never asked.
  Fix direction, for a follow-up plan: `execute_tool` refuses to run a real tool while
  `approval_required` is set and the grant did not come from the driver, and the
  grant moves to a driver-only internal key (for example `_approval_granted`).
- **Closed by the follow-up above (D-007).** maker_checker can ship an unjudged draft when the iteration limiter ends the run
  (D-017/D-018). If the limiter forces `checker_passed=True` on a `revise` turn, the
  draft written on that turn skips `check` and ships without a verdict; the run
  reports `max_iterations_reached=True`. It depends on the budget: with an always-False
  checker, odd `max_iterations` values ship an unjudged draft and even ones ship the
  judged draft. Fix direction, for a follow-up plan: force the pass only while the
  current state is `check`, and add a RED test at `max_iterations=3` asserting the
  answer is the first (judged) draft.
- **Closed by the follow-up above (D-008), except ReWOO.** Reflexion's and ParallelReact's `think->conclude` have no evidence guard (D-004), so
  they can conclude without a tool call (pre-existing). ReWOO has no stall guard (still open).
- **Closed by the follow-up above (D-012).** `SkillLoader`'s `@tool` scan dedupes only against `SKILLS` names: a `@tool` function
  reachable under two attribute names (an alias or a re-import) still loads twice.
- `scripts/eval.py` was not re-run; the baseline in `CLAUDE.md` is staler after the
  routing change above. The follow-up above re-runs it on the new default model.

### Core audit 2026-09-22

Core audit of `src/fsm_llm` dated 2026-09-22 (`plans/plan-2026-09-22T080837-8b258a25`,
16 steps in 26 commits, one commit per step or substep, the last five being fixes from
the iteration review). Every behaviour change has a
test in `tests/test_fsm_llm/test_audit_2026_09_22.py` that fails on the parent commit;
four cross-cutting sweeps live in `tests/test_fsm_llm/test_audit_sweeps.py`. Full suite:
6,862 tests collected (was 6,564). `ruff` and `mypy` clean across all 6 packages. No
prompt text changed, so the stale eval baseline was not re-measured.

### Behaviour changes to know about -- core audit 2026-09-22

- **Load-time validation of `prompt_config` and `transition_classification`.** A
  `classification_extractions[].prompt_config` with an unknown key or an out-of-range
  value (`max_tokens < 1`, `temperature` outside 0.0-2.0, `max_intents` outside
  1-`MAX_MULTI_INTENTS`) now fails when the FSM loads, with the same rules
  `ClassificationPromptConfig` applies at extraction time (it used to fail, softly,
  mid-conversation). `State.transition_classification` must be None or a dict: the
  reserved `confidence_threshold` key must be a non-bool number in [0, 1], and every
  other key must map to a dict whose only key is `description` (str or None). `{}` is
  accepted. `fsm-llm-validate` gives the same verdict. All shipped examples still load.
- **Session files hold placeholders for non-JSON objects.** `FileSessionStore.save`
  no longer calls `str()` on an arbitrary value. An exact `datetime`, `date`, `time`,
  `timedelta`, `Decimal` or `UUID` is still written as `str(value)` (byte-identical to
  before); anything else (`set`, `bytes`, `frozenset`, a subclass of those scalars,
  `pathlib.Path` (`"<redacted:PosixPath>"`), an `Enum` member, a numpy scalar such as
  `numpy.int64`, a custom object) is written as `"<redacted:TypeName>"`, so a restore
  reads the placeholder string. `fsm_llm_agents.memory_persistence.save_working_memory`
  (and so `MemorySessionStore`) writes WorkingMemory with the same hook, now public as
  `fsm_llm.session.session_json_default`.
- **`context_snapshot` is filtered.** `get_complete_conversation(...)["metadata"]
  ["classification_results"][field]["context_snapshot"]` drops every secret-shaped
  entry (`is_forbidden_context_entry`) at any depth, including inside lists. The verdict
  is taken on the stored JSON form (tuples as lists, non-str keys as strings), and a
  value nested deeper than `MAX_CONTEXT_FILTER_DEPTH` is dropped. A cyclic value is
  still omitted, as before.
- **`FSMStackFrame.fsm_definition` is typed `FSMDefinition`.** A frame built from an id
  string, or from anything else that does not validate as an `FSMDefinition`, is now a
  validation error.
- **Typed "unknown FSM id" error.** `load_fsm_definition("<id>")` for an id that is not
  a file path raises `FSMDefinitionNotFoundError`, which is both an `FSMError` and a
  `ValueError`; the message keeps the `Unknown FSM ID` prefix and names the missing
  registry.
- **`API(handler_timeout=..., max_fsm_cache_size=...)` are consumed.** Both used to fall
  into `**llm_kwargs` and reach the LLM interface; they now configure `HandlerSystem`
  and `FSMManager`. Defaults (`None`, 64) behave as before. `max_fsm_cache_size` below
  1 raises `ValueError` when the `API` or `FSMManager` is constructed (0 used to make
  every `start_conversation` fail with `dictionary is empty`). The
  `MAX_TIMED_HANDLER_STRAGGLERS` cap is shared by every conversation of one `API`
  (documented, unchanged).
- **Bulk extraction ignores a sanitiser override.** Bulk Pass-1 extraction sanitises the
  user message through `prompts.sanitize_text_for_prompt` (the shared default
  sanitiser), so a custom `data_extraction_prompt_builder` subclass that overrides
  `_sanitize_text_for_prompt` no longer affects that call.
- **Restore into a missing instance raises.** `restore_session` seeds summary, history,
  provenance and working memory in one `FSMManager.seed_restored_conversation` call. If
  the FSM instance is gone it raises a typed `FSMError` (and the half-restored
  conversation is torn down) instead of logging a warning and skipping the history
  replay.
- **A failed FSM load no longer evicts a cache entry.** `get_fsm_definition` evicts only
  when it inserts a loaded definition (same `len >= max_fsm_cache_size` rule). On a
  concurrent miss of the same id the loader may run twice; the first-inserted
  definition is returned to both callers.
- **Getters on ended conversations agree.** `get_data`, `get_current_state`,
  `has_conversation_ended` and `get_conversation_history` re-raise
  `ConversationBusyError`. On any other error they answer from the ended-conversation
  cache only when the conversation is gone (its stack or its FSM instance was torn
  down); on a live conversation the original error re-raises, so a failing state
  lookup is no longer reported as "not ended". A cache miss re-raises too, except that
  `has_conversation_ended` returns False for an id it does not know.
  `get_conversation_history` now returns the cached history after `end_conversation`
  (it used to raise). An id that was never started still raises `ValueError`.
- **`max_history_size=0` keeps a summary.** Exchanges are digested into
  `Conversation.summary` (same 2,000-character cap) before they are cleared.
- **`get_version_info()["architecture"]`** is `"2-pass"` (was `"improved-2-pass"`).

### Fixed -- core audit 2026-09-22

- P0-1: the ended-conversation cache (`API._ended_conversations`) is written, evicted
  and read only under `API._stack_lock`. A reader during a concurrent end no longer
  sees an evicted-but-not-inserted entry, and concurrent evictions no longer fail with
  `KeyError`.
- P0-3: `restore_session` no longer takes `FSMManager._lock` while holding a
  conversation lock (4 sites, C-NEW-007). The seeds run in the canonical order
  (`_lock` for the lookup only, then the conversation lock). `api.py` has no remaining
  access to `fsm_manager._lock`, `._conversation_locks` or `.instances`.
- P0-4: `handler_timeout` and `max_fsm_cache_size` are reachable through `API` (see
  above).
- P1-1: the Pass-1 memo name is bound on every path (defensive; no `NameError` was
  reachable).
- P1-2: `max_history_size=0` summarises before clearing (see above).
- P1-4: `get_conversation_history` falls back to the ended cache (see above).
- P1-5: `get_data` falls back to the ended cache when the conversation is gone and
  propagates `ConversationBusyError` (see above).
- P1-6: `SessionState.stack_depth` is documented as advisory (written by
  `save_session`, never read by restore).
- P1-7: the Pass-2 apology retry is counted (`LiteLLMInterface.apology_retry_count`);
  the retry itself is unchanged.
- Security: secret-shaped entries no longer reach the `context_snapshot` metadata (see
  above), and a value's `__str__` no longer reaches a saved session file or a saved
  WorkingMemory file (see above).
- `runner._redact_context` (the `fsm-llm` CLI's debug context dump) terminates on a
  self-cycle (the cycle becomes `"<redacted:cycle>"`) and runs in linear time on aliased
  acyclic input: a 3-way-aliased tower 16 levels deep went from more than 25 s to
  under 1 ms. Input that aliases every level and also cycles is not memoised and can
  still be slow; the runner only redacts `get_data()` output, which is always acyclic.
  Output on acyclic input is unchanged.
- `DEFAULT_TEMPERATURE` is the single source of the 0.5 default in `API.__init__` and
  `LiteLLMInterface.__init__` (it was defined but unused next to two literals).
- Stale comments in `fsm.py` said `get_fsm_definition` re-takes `_lock`; they now give
  the real reason (the loader runs on a cache miss).

### Added -- core audit 2026-09-22

- `src/fsm_llm/security.py`: the credential/forbidden-context filter and
  `has_internal_prefix` moved there verbatim (`constants.py` 1,978 -> 299 lines).
  `fsm_llm.constants` re-exports every public name and every private name used in
  `src/` or `tests/`; no filter verdict changed.
- `ResponseGenerationRequest.skip_generation` (default False), set by the greeting and
  synchronous Pass-2 skip sites and honoured by `LiteLLMInterface.generate_response`
  and its streaming path. The streaming skip site still makes no interface call.
- `FSMDefinitionNotFoundError` (exported, pickles with `fsm_id` and details).
- `FSMManager.seed_restored_conversation`, `FSMManager.has_instance`,
  `FSMManager.copy_raw_context_data`, `FSMManager.prune_orphaned_locks`.
- `ClassificationResult.is_below_default_threshold` (property).
- `WorkingMemory.hidden_buffers` (read-only property, returns the instance's own
  `frozenset`).
- `fsm_llm.logging.reset_handlers()` and `fsm_llm.logging.register_stream_handler()`;
  `enable_debug_logging` uses them and touches no logging private.
- `fsm_llm.prompts.sanitize_text_for_prompt(text)`, the public entry to the shared
  prompt sanitiser (output identical).
- Constants: `DEFAULT_MAX_FSM_CACHE_SIZE` (64, shared by `API` and `FSMManager`),
  `CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT`, `PROVENANCE_METADATA_KEY` (`pipeline._PROVENANCE_KEY`
  stays as an alias), `TRANSITION_CLASSIFICATION_THRESHOLD_KEY`.
- `LiteLLMInterface.apology_retry_count`.
- `API.__init__` parameters `handler_timeout` and `max_fsm_cache_size`.
- Tests: `tests/test_fsm_llm/test_audit_2026_09_22.py` and
  `tests/test_fsm_llm/test_audit_sweeps.py` (JsonLogic operator partition and dispatch,
  agreement of the four context filters, reachability of every `HandlerSystem` and
  `FSMManager` option from `API`, an exhaustive small-alphabet prompt-sanitiser sweep
  with 255/256/257/300 padding cases). The sweeps found no defect.
- Docs: `CONTRIBUTING.md` (setup, commands, count pins, frozen examples, the `plans/`
  convention, DECISION anchor format, the 6-line rule, `[STALE]` versus
  `[SUPERSEDED BY D-nnn]`); a README "Behaviour details" section (LLM calls per turn,
  `handler_timeout` and the straggler cap, what the extractor can write, JsonLogic
  differences, `extraction_retries` on Ollama).
- `expressions.py` checks at import time that no short-circuit operator (`and`, `or`,
  `if`) re-enters the eager `operations` table.

### Deprecated -- core audit 2026-09-22 (removal in 1.0)

- `FSMManager.cleanup_stale_conversations`: use `prune_orphaned_locks` (warns).
  `API.cleanup_stale_conversations` is a different method and is not deprecated.
- `ClassificationResult.is_low_confidence`: use `is_below_default_threshold` (warns).
  `Classifier.is_low_confidence` is not deprecated.
- `TransitionEvaluatorConfig.ambiguity_threshold`, `minimum_confidence` and
  `evidence_conditions_normalizer`: no effect since priority-only ranking; setting any
  of them to a non-default value emits `DeprecationWarning`. Defaults stay silent.
- The `system_prompt="."` Pass-2 skip sentinel: still sent and still honoured by
  `LiteLLMInterface`; custom interfaces should read `skip_generation` instead (no
  runtime warning).

### Removed -- core audit 2026-09-22 (breaking)

- `fsm_llm.DomainSchema`, `fsm_llm.LLMRequestType` and
  `fsm_llm.validate_json_structure` (and their `__all__` entries).
- Constants: `LOG_FIELD_TIMESTAMP`, `LOG_FIELD_LEVEL`, `LOG_FIELD_MESSAGE`,
  `LOG_FIELD_MODULE`, `LOG_FIELD_FUNCTION`, `LOG_FIELD_LINE`,
  `LOG_FIELD_CONVERSATION_ID`, `LOG_FIELD_PACKAGE`, `LOG_MESSAGE_PREVIEW_LENGTH`,
  `LOG_RESPONSE_PREVIEW_LENGTH`, `DEFAULT_STEP_TIMEOUT`, `DEFAULT_HANDLER_TIMEOUT`,
  `MIN_BASE_CONFIDENCE`, `PRIORITY_SCALING_DIVISOR`, `CONDITION_SUCCESS_RATE_BOOST`.
- `fsm_llm.constants` no longer exposes the credential filter's unreferenced private
  helpers (import them from `fsm_llm.security`): `_ACRONYM_BOUNDARY`,
  `_AMBIGUOUS_CREDENTIAL_ABBREVIATIONS`, `_CAMEL_BOUNDARY`,
  `_CREDENTIAL_MATERIAL_HEADS_SHARED`, `_CREDENTIAL_NAME_ANY_RE`,
  `_CREDENTIAL_NAME_HEAD`, `_CREDENTIAL_NAME_TERMS`, `_CRYPTO_AT_WORD`,
  `_CRYPTO_GAP_REACH`, `_ISO_DATE_RE`, `_KEY_TRIGGER`, `_MIN_CREDENTIAL_VALUE_ENTROPY`,
  `_NAME_TOKEN_SPLIT_RE`, `_PATH_SEGMENT_TRIGGER`, `_PEM_PREFIX`,
  `_POLICY_TAIL_COUNT_END_RE`, `_POLICY_TAIL_DURATION_END_RE`, `_POLICY_TAIL_MAX_COUNT`,
  `_POLICY_TAIL_RE`, `_POLICY_TAIL_STATE_WORDS`, `_POLICY_TAIL_WORD_SPLIT`,
  `_POLICY_TAIL_WORD_VALUE_MAX_CHARS`, `_PURE_HEX_RE`, `_TOKEN_ONLY_MATERIAL_HEADS`,
  `_TOKEN_TRIGGER`, `_TRIGGER_PLURAL`, `_ULID_VALUE_RE`, `_UUID_VALUE_RE`,
  `_VALUE_SCAN_NAME_RE`, `_WORD_END`, `_is_path_shaped`,
  `_policy_tail_scalar_is_credential`, `_policy_tail_value_is_credential`; nor the
  stdlib names `re`, `math`, `unquote` and `Iterable`.
- `TransitionEvaluation.confidence`, the evaluator's `"confidence"` score key and
  `"confidence_factor"` condition-result key, and the confidence computation behind
  them (an old `confidence=` kwarg is ignored).
- `ResponseGenerationRequest.context`, `.extracted_data` and `.previous_state`, and
  their per-turn computation (a caller still passing them is ignored, `extra="ignore"`).
- `DataExtractionResponse.additional_info_needed` (and the required-names scan that
  fed it).
- `ContextCompactor(summarize_on_trim=...)` and the attribute.
- `BasePromptBuilder._estimate_token_count(is_json=...)` and
  `_build_response_format(field_heading=...)` parameters (the heading is always
  `"Where:"`).
- `expressions.operations["and"]`, `["or"]`, `["if"]` (unreachable eager fallbacks) and
  `expressions.if_condition`. `_SHORT_CIRCUIT_OPERATORS` is now a `frozenset`.
- `ollama.TRANSITION_JSON_SCHEMA` and the `"transition_decision"` entry of
  `_CALL_TYPE_SCHEMAS`; `build_ollama_response_format("transition_decision")` returns
  the unknown-call-type fallback, `None`.
- Visualizer: `ICONS["note"]`, `ARROW_STYLES["down_arrow"]`, `["right_arrow"]`,
  `["diamond"]`, `BOX_STYLES["section"]`.
- `API._replay_history` (private; folded into `seed_restored_conversation`).

### Performance -- core audit 2026-09-22

- `LiteLLMInterface` calls `litellm.get_supported_openai_params` once per instance and
  model string (a raised lookup is not memoised).
- The FSM definition cache has its own leaf lock, so turn-path definition lookups no
  longer take `FSMManager._lock`; the loader runs outside the lock.
- `utilities.filter_context_tree` skips the cycle pre-scan for a plain `dict` root whose
  values are all leaves (output identical).
- Runner redaction is linear on aliased acyclic input, and its debug context dumps are lazy
  (no redaction when debug logging is off).
- The removed `ResponseGenerationRequest` fields are no longer computed on every turn.

### Fixed -- follow-up 2026-09-22

- `json.dumps(default=str)` is gone from every writer that emits context or trace values
  out of the process, closing the same `__str__`-leak class the session fix closed on
  disk. An object whose text carries a secret is now written as
  `"<redacted:TypeName>"`, and its `__str__` is never called:
  - `fsm_llm_monitor`'s dashboard websocket push, which broadcast to every connected
    browser;
  - the two `fsm_llm_reasoning` engine sites that render context into an LLM prompt;
  - the `fsm_llm_reasoning` CLI's JSON output and its saved results file.
- The hook itself moved to `fsm_llm.utilities.redacting_json_default`, since it now
  serves disk, prompt and socket writers. `fsm_llm.session.session_json_default` is the
  same object under its on-disk name, so existing imports keep working. An exact
  `datetime`/`date`/`time`/`timedelta`/`Decimal`/`UUID` is still written as its `str()`.

### Not changed (considered and declined) -- core audit 2026-09-22

- P0-2, Ollama null memo across retries: kept. Structured Ollama calls run at
  temperature 0, so a retry of the same prompt is pure waste; documented in the README
  instead (D-021).
- Unifying the 4 tie-break rules of `extract_json_from_text`: all four or none, no
  failing case, and it radiates to every package (D-022).
- One shared context walker with a budget parameter: the prompt walker truncates, the
  data walker must never truncate, the runner redacts instead of dropping (D-023).
- Splitting `pipeline.py` into three modules: high churn, no behaviour value (D-024).
- Per-FSM `secret_context_keys`/`public_context_keys`: needs plumbing into all four
  filter seams; its own plan (D-025).
- `FSMDefinition.agent_managed` instead of sniffing `agent_trace`: agent FSMs are built
  at many sites; deferred (D-026).
- Splitting `is_forbidden_context_entry` by arity: every production caller passes the
  value, and one decision point stays one name (D-027).
- `setup_logging`'s `-1` sentinel: test-pinned intended behaviour (D-028).
- `handle_conversation_errors` union parameter: local convention, no defect (D-028).
- Renaming `HandlerBuilder.critical`: not a real name collision (D-028).
- `_CLASSIFICATION_CONTEXT_BUILDER` visibility: module-private, no reach-in (D-028).
- Transition-snapshot reuse: would touch the rollback contracts (D-028).
- A classifier digest cache: minor gain (D-028).
- Session `strip_forbidden_keys` opt-in: needs a walker on the save path; the `str()`
  leak is fixed instead (D-028).
- Relocating the 281 existing DECISION anchor bodies: mass churn; the convention now
  applies to new anchors (D-028).
- Per-conversation straggler keying: would thread `conversation_id` into
  `execute_handlers`; the shared cap is documented instead (D-028).
- P1-8, one unit for `max_history_messages` (fetch `ceil(n / 2)` exchanges): changes
  the rendered history under the `TOKEN_BUDGET` strategy, so not output-identical; the
  fetch still passes an exchange count (D-038).
- P1-3, a read-only `should_execute` probe: landed in step 4 as a
  `types.MappingProxyType` view, then reverted in the same release. It broke conditions
  that only read the context but serialise, copy or type-check it (`json.dumps`,
  `copy.deepcopy`, `pickle`, `isinstance(ctx, dict)`), and under the default
  `error_mode="continue"` such a handler was silently skipped. The probe again gets the
  live context; a condition must not mutate it (the documented pure-predicate contract).
- `json.dumps(default=str)` in the three `fsm_llm_reasoning` size measurements
  (`handlers.py`): the serialized text is only measured with `len()` and never emitted,
  so no `__str__` reaches a prompt, a file or a socket. The emitting writers were fixed
  (see "Fixed", below).
- `context_snapshot` keeps internal-prefixed keys named in `context_keys`
  (pre-existing): whether `context_keys` may name internal keys needs its own decision
  (D-039).

### Core audit 2026-09-21

Core audit of `src/fsm_llm` dated 2026-09-21 (`plans/plan-2026-09-21T203800-8a03483a`,
15 fix steps, one commit per step). 45 audit ids: 42 fixed, 2 partly fixed (D9, D12),
1 skipped (D7). Every fixed id is pinned by a `test_<id>_*` regression test in
`tests/test_fsm_llm/test_audit_2026_09_21.py` that fails on the pre-fix code. Full
suite: 6,564 tests collected (was 6,177). `ruff` and `mypy` clean across all 6 packages.

### Behaviour changes to know about

- **A2: transition priority now decides.** When several transitions pass, the one
  with the unique lowest `priority` wins deterministically, whatever the gap between
  priorities or the number of conditions. Only a tie at the lowest priority is
  AMBIGUOUS and goes to the classifier (tied group only). `minimum_confidence` and
  `ambiguity_threshold` are kept as deprecated no-ops. 10 shipped example FSMs now
  resolve some formerly ambiguous turns deterministically (87 state/passing-set
  combinations). `examples/basic/simple_greeting` has two unconditioned transitions
  (farewell p0, conversation p1) and now goes to `farewell` on every turn. Authors who
  want the classifier to choose between intent-routed transitions must give them
  equal priority. The eval baseline in CLAUDE.md is stale until re-run.
- **B1-B4: one None/missing rule for JsonLogic.** Ordering (`<`, `<=`, `>`, `>=`) is
  False when any operand is `None`. Arithmetic (`+ - * / % min max`) on a `None`
  operand yields an internal "undefined" value that no comparison can satisfy:
  `==`, `!=`, `===`, `!==`, ordering, `in` and `contains` are all False on it, and
  `!`, `!!`, `and`, `or`, `if` treat it as false. So
  `{"<": [{"-": [total, discount]}, 100]}` and `{"!=": [{"-": [balance, paid]}, 0]}`
  with the second operand unset are False (the first used to compare `False` as 0
  and pass), and a bare arithmetic condition on an unset operand does not fire
  (`evaluate_logic` returns `None` for it). `-` is unary only with
  exactly one operand. `missing`, `missing_some` and `requires_context_keys`
  treat absent, `None` and `""` as missing, and an extracted `None` no longer
  overwrites a stored value during transition evaluation. `==`/`!=` coerce
  numerically only for a mixed number/string pair whose string is a plain decimal or
  scientific literal in ASCII digits (`1.0 == "1"` is True; `"1_000" == 1000`,
  `" 1 " == 1` and `"١٠٠٠" == 1000` are False); two strings are never coerced (`"01" == "1"` stays False), bools are never
  coerced, and `null == null` stays True.
- **B5, B6, B10: new load-time errors.** A JsonLogic operator object with more than one
  key, or nesting deeper than `MAX_JSONLOGIC_DEPTH`, fails to load (B5).
  `context_scope` is now a `ContextScope` model (`read_keys`/`write_keys` lists,
  unknown keys rejected); a string instead of a list or a misspelled key is an error
  (B6). A `field_name` declared twice in `field_extractions`, or twice in
  `classification_extractions`, is an error, and so is a blank or internal-prefixed
  `required_context_keys` entry (B10). `fsm-llm-validate` reports the same errors.
- **C3: framework-reserved context keys.** Handler deltas can no longer set or delete
  the keys in `constants.RESERVED_CONTEXT_KEYS` (`_conversation_id`,
  `_current_state`, `_fsm_id`, `_previous_state`, and 8 more framework-seeded keys).
  An attempted change logs a WARNING; an equal echo is ignored silently.
  Handler-owned internal keys (for example agents' `_replan_count`) still merge.
- **C8: CLI exit codes and output streams.** `fsm-llm`, `fsm-llm-validate` and
  `fsm-llm-visualize` exit 0 on success, 1 on failure (a turn error used to exit 255)
  and 130 on Ctrl-C (mid-turn Ctrl-C used to print a traceback). The validation
  report and the diagram now go to stdout; diagnostics stay on stderr.
- **C12: `end_conversation` raises on lock timeout.** If the conversation lock cannot
  be taken within `END_CONVERSATION_LOCK_TIMEOUT_SECONDS` (30 s),
  `FSMManager.end_conversation` raises `ConversationBusyError` (a new exported
  `FSMError` subclass) and changes nothing, instead of tearing the conversation down
  while a turn may still be running. Retry after the turn ends. `API.end_conversation` propagates that refusal (it used to log it and
  return) and keeps the conversation active with its stack, idle tracking and data;
  frames above the refusing one that already ended are removed from the stack. Its
  data/state/history cache read is bounded by the same timeout (it used to wait
  forever); any other failure of that read is logged and the end proceeds without
  a cache entry. `cleanup_stale_conversations` and `close()` log a refusal and continue
  with the other conversations.
- **D5: more credential names are stripped from prompts.** `passwd`, `pwd`, `pass`,
  `passcode`, `passphrase`, `pin`, `otp`, `mfa_code`, `cvv`, `ssn`, `credit_card`,
  `card_number`, `cookie`, `jwt`, `bearer`, `authorization`, `auth_header` and
  `recovery_code` (whole name segments, optional plural) are now filtered out of
  prompt context. Policy-style tails (`password_min_length`) and `bool` values are
  kept. Names such as `pass_id`, `pin_code` or `cookie_consent` are now stripped too.
  `cvc`, digit-suffixed terms (`cvv2`, `pin2`) and camelCase/acronym forms
  (`pinCode`, `PINCode`) match as well. A policy-suffix name (`pin_attempts`,
  `authorization_status`) keeps only a value that cannot be a credential: None, a
  `bool`, a dict (its keys are then filtered by their own names), a number under a
  count suffix (`pin_attempts: 3`, below 1,000) or a duration suffix
  (`cookie_max_age: 86400`), an ISO date, or a string made only of state words
  (`verified`, `locked`, `enabled`, ...). `pin_enabled: "1234"`, `cvv_status: 737`,
  `pin_status: "hunter"` and `authorization_status: "Bearer ..."` are stripped.
  `ttl`, `timeout` and `limit` join the password policy suffixes.
- **D6: no reasoning fallback.** A Pass-2 reply with an empty or missing `message`
  no longer shows the model's internal `reasoning` to the user; it produces the
  generic apology and the pipeline's one retry.
- **D10: removed dead Pass-1 prompt API.** `DataExtractionPromptBuilder.build_extraction_prompt`
  and `build_refinement_prompt` had no caller and are deleted, together with the five
  `DataExtractionPromptConfig` fields only they read (`include_context_data`,
  `include_state_instructions`, `enable_detailed_guidelines`, `enable_format_rules`,
  `enable_extraction_guidance`). Passing those fields now raises `TypeError`.
- **New export:** `fsm_llm.ContextScope`.

### Fixed

A. Decision-making (classification / memory)

- A1: the transition classifier's `confidence_threshold` is enforced; a result below it stays in the current state (WARNING, record flagged `low_confidence`).
- A2: priority spread alone no longer bypasses the classifier inconsistently; the unique lowest priority wins and only ties are ambiguous (see above).
- A3: `Classifier.classify`/`classify_multi` accept a context (last 3 exchanges with each line capped at 150 characters, state purpose, scoped visible data), rendered sanitized and security-filtered; the pipeline feeds it at both call sites. An extraction's `context_keys` only narrows the state's `context_scope.read_keys`; it can no longer expose a key `read_keys` hides.
- A4: every classification-extraction result is kept in `context.metadata["classification_results"]` and the ambiguous-transition record in `context.metadata["transition_classification"]`, readable via `get_complete_conversation`.
- A5: after a transition, the new state's unset `classification_extractions` run on the same message.
- A6: `Conversation.summary` is rendered as `<conversation_summary>` in prompts and persisted in sessions (`SessionState.conversation_summary`).
- A7: `WorkingMemory` buffers reach the transition evaluator and the Pass-2 prompt context (`FSMContext.get_merged_data`).
- A8: `_transition_classification_result` is cleared at the start of every turn.

B. Rules engine (expressions / evaluator / definitions / validator)

- B1: `-` with a `None` second operand no longer becomes unary negation.
- B2: `<=`/`>=` are False when both operands are unset.
- B3: `==` agrees with `<=`/`>=` for mixed number/string operands.
- B4: `missing` and `requires_context_keys` treat `None`/`""` as missing; an extracted `None` does not mask a stored value.
- B5: multi-key operator objects and over-deep logic are rejected at load time.
- B6: `context_scope` is validated (`ContextScope` model).
- B7: the validator detects trap cycles per strongly connected component, so interlocking cycles with no exit are reported.
- B8: `missing_some` with a non-integer minimum is a clean error; an integer needle `in` a string haystack is matched as text; `%` sign semantics are documented.
- B9: the load-time logic walk no longer recurses into data lists, and its depth is bounded.
- B10: duplicate `field_name` within one extraction list and blank or internal-prefixed `required_context_keys` are load errors.
- B11: dead "unreachable terminal" warning branch removed.
- B12: the validator's structure pass no longer crashes on type-invalid input; a non-validation loader exception is an ERROR, not a warning with `is_valid=True`; key references are read from the logic tree, including dotted `var` paths.
- B13: `strict_condition_matching` is documented as diagnostics only.

C. Handlers / sessions / logging / CLI

- C1: `register_handler` is copy-on-write under a lock; concurrent `execute_handlers` never sees an empty handler list.
- C2: a timed handler runs on its own copy of the context in a daemon thread; a timed-out handler's writes never reach later handlers and never block exit. A timed handler that finishes in time has its in-place writes adopted, so later handlers see the same context with or without `handler_timeout`. At most `MAX_TIMED_HANDLER_STRAGGLERS` (4) timed-out threads per `HandlerSystem` may still be running; past that a timed call fails at once as a timeout (WARNING) instead of starting another thread.
- C3: handler deltas cannot overwrite framework-reserved context keys (see above).
- C4: `FileSessionStore` ids use a full match; `load`/`exists`/`delete` return None/False on `OSError`; `list_sessions` returns only loadable ids; `save` fsyncs before replace.
- C5: file logging is marked initialized only after `logger.add` succeeds; the JSON sink no longer stashes the rendered line in the shared record.
- C6: non-dict handler returns log a WARNING; `priority` is validated at registration; condition errors are wrapped once; `HandlerExecutionError` pickles.
- C7: JSON log lines never `str()` unknown values (`<non-serializable: TypeName>`); the session store's lossy coercions are documented.
- C8: consistent CLI exit codes and stdout payloads (see above).
- C9: the visualizer handles `"transitions": null` and multi-line FSM names.
- C10: `get_workflows()` and siblings re-raise inner `ImportError`s; `disable_warnings()` only silences `fsm_llm` warnings.
- C11: `_sub_conversation_summary.fsm_type` is the definition name.
- C12: `end_conversation` refuses to tear down on lock timeout (see above).

D. LLM interface / prompts / context filters

- D1: the Pass-2 embedded-JSON fallback strips `<think>` blocks before scanning.
- D2: context filters redact non-JSON-native leaves as `<redacted:TypeName>` (stdlib date/time, `Decimal` and `UUID` values are kept).
- D3: context filters drop reference cycles. The prompt filter caps work at `MAX_CONTEXT_FILTER_NODES` (100,000), failing closed. `get_data`, `save_session` and the extracted-data commit never truncate: every container that cannot reach a cycle is memoised, even next to an unrelated cycle, and only a cycle that is itself aliased past the work ceiling raises `utilities.ContextFilterWorkError` instead of returning a partial value.
- D4: a flat bulk-extraction reply drops top-level `confidence`/`reasoning` instead of merging them into context.
- D5: 18 more credential names are stripped from prompts (see above).
- D6: internal `reasoning` is never shown as the reply (see above).
- D8: a non-string `reasoning` from the classifier model no longer raises a pydantic error; intent names match case-insensitively when the match is unique.
- D9 (partly): the prompt sanitizer escapes tag names starting with `_`, `!` or `?` (`<_task>`, `<!--`, `<![CDATA[`, `<?xml`) and unterminated closers at the end of text.
- D10: dead Pass-1 builders and their unread config fields removed (see above).
- D11: `{"value": null, "<field_name>": ...}` falls back to the field-name key.
- D12 (partly): `stream` and `response_format` are reserved LLM call kwargs (`constants.RESERVED_LLM_CALL_KWARGS`), ignored with a WARNING in `LiteLLMInterface` and `Classifier`.

### Not fixed (with reasons)

- D7 skipped: the `...key` value-scan names (`monkey`, `primary_key`, `cache_key`) strip only for a bare high-entropy hex value, the same verdict `passkey` gets; bare `token` is a corpus must-strip row and `secret_santa` strips by design. Changing it would loosen the shared secret filter.
- D9, attribute part skipped: escaping `<b onclick>` also escaped benign prose such as `a < b and c > d` that is pinned as safe, and attributes are inert to an LLM. The audit's `<Think>` claim was false (already escaped).
- D12, `/nothink` gating skipped: every Ollama call already sets `reasoning_effort="none"`, the harness depends on the prefix, and the effect is live-model behaviour offline tests cannot measure.
- D12, `<original_input>` truncation skipped: truncating at `max_message_length` (1,000) would cut long agent tasks and reasoning problems, and the same raw message also reaches Pass 1 and the classifier.
- B10, cross-list duplicates kept legal: a `field_name` in both `field_extractions` and `classification_extractions` is a supported fallback pattern (the explicit extractor fills the key when the classifier is below threshold), so only same-list duplicates are rejected.

## [0.8.0] - 2026-09-21

Licence change release: FSM-LLM is now distributed under the Apache License 2.0
(previously GPL-3.0-or-later). Releases up to and including 0.7.0 remain available
under GPL-3.0-or-later. No code or behaviour changes.

### Changed

- Relicensed from GPL-3.0-or-later to Apache-2.0 (`LICENSE`, `pyproject.toml` `license`, `fsm_llm.__license__`, README, CLAUDE.md).

## [0.7.0] - 2026-09-22

Second-layer core-engine audit (`plans/plan-2026-09-20T114608-a8e47b88`, 4 audit-fix
iterations, each with its own regression tests and adversarial review, building on the
0.6.0 audit release below). 9 items re-verified from that release's own "Known
limitations" list still reproduced; 1 genuinely new correctness bug and 8 smaller gaps
were found by an independent fresh sweep of `src/fsm_llm/`. Every one of the 10 items
shipped with a regression test that reproduces the defect pre-fix and closes it
post-fix (or, for one mechanical refactor, an equivalence proof both ways). Adversarial
review ran at the end of every iteration and found real, reproduced problems each time
(2 findings iteration 1, 5 iteration 2, 5 iteration 3, plus one the verifier itself
found in iteration 3) -- every one caught and closed inside that same iteration's
completion-fix, per this repo's own established audit practice. A live-Ollama
regression pass (`ollama_chat/qwen3.5:9b-q8_0`, 27 calls across 2 of the 5
pre-registered scenarios) found no regression in the sanitizer/`rejected_corrections`
code paths this plan does not touch. Full suite: 6,091 tests collected (was 6,035),
zero regressions; `ruff`/`mypy` clean across all 6 packages throughout.

### Fixed

- **`save_session` torn snapshot (D-005, D-009; `f9166f8`, `6c22714`).**
  `FSMManager.get_conversation_snapshot` now reads state, internal-key-stripped data,
  history, the working-memory dict and provenance metadata inside ONE
  `_read_under_lock` hold; `api.py`'s `save_session` calls it instead of composing from
  4 separately-locked reads. A concurrent turn landing in the gap between those reads
  could previously persist pre-transition state paired with post-transition data.
  `get_stack_depth` deliberately stays outside the snapshot (a separate, non-atomic
  call): it is structural and `restore_session` ignores a session file's persisted
  `stack_depth` entirely, so a torn pairing there is never read back.
- **Visualizer truncation-ellipsis sweep completed (D-002, D-007; `ae83494`,
  `cd26435`).** `create_state_boxes`'s STATE DIAGRAM row and `create_states_section`'s
  icon-carrying row -- the sibling call site an earlier decision's own comment named
  but never migrated -- now both route through the existing `_fit()` helper. Two
  states sharing a long common prefix previously rendered byte-identical rows with no
  truncation marker in `--style full` output; they now render distinguishably with a
  middle-ellipsis marker, matching every other bordered row in the module.
- **`restore_session`'s `fsm_id` mismatch is now observable, and `fsm_id` is
  content-derived (D-011, D-016; `44ad994`, `e5a1415`).** `restore_session` logs a
  `logger.warning(...)` (does not hard-fail, since `fsm_id` legitimately drifts across
  additive schema upgrades) when `state.fsm_id != self.fsm_id`. `fsm_id` is now one
  content hash of the parsed FSM definition (`fsm_{name}_{hash(model_dump())}`),
  computed once after the dict / `FSMDefinition` / file construction paths converge --
  replacing three separate per-branch hash inputs, one of which (the file path) carried
  no content hash at all. This closes both the silent-restore-under-a-different-FSM gap
  on `API.from_file`/the CLI path (the documented primary entry point -- exactly the
  gap the 0.6.0 "Known limitations" list below named as `restore_session does not
  compare fsm_id`) and a false-positive-mismatch-noise bug where the identical FSM
  produced different ids depending on how it was constructed or loaded.
- **`runner.py`'s CLI JSON logging survives a non-JSON-native context value (D-014,
  D-015; `0e95801`, `c9f920d`).** Both `json.dumps(...)` call sites now pass
  `default=_json_default`, a callable that emits a `<non-serializable: TypeName>`
  placeholder and logs one WARNING -- never `str(obj)`/`repr(obj)` -- mirroring
  `_redact_mapping`'s existing non-str-key WARNING pattern. A handler storing a
  `datetime` (or any non-JSON-native value) no longer crashes the CLI's debug/dump
  logging path; a secret-bearing object under a benign key no longer leaks its repr
  into persisted logs on the very redaction path meant to protect it.
- **`enable_debug_logging()` no longer duplicates log lines (D-013, D-015; `0eb3be7`,
  `07b58f1`).** It now registers its stderr handler in `logging.py`'s existing
  `_stream_handler_ids` dict, mirroring `setup_logging()`'s own registration, so a
  later `setup_logging(sink="stderr", format="human")` call no longer adds a second
  handler and every subsequent log line no longer prints twice. `src/fsm_llm/CLAUDE.md`'s
  file map corrected (`enable_debug_logging`/`disable_warnings` live in `__init__.py`,
  not `logging.py`). Three scope edges are now documented: the `fsm_llm` logger is
  disabled by default, a `level=` request is silently dropped on the short-circuit
  path, and the dedup covers only the exact `(stderr, human, context=False)` triple.
- **A deleted context key's provenance digest no longer outlives its plaintext (D-018,
  D-024; `8d4c05e`, `8bf5f18`).** `MessagePipeline`'s `merge_delta` -- the one place a
  handler's `None`-delta convention actually deletes a context key -- now also pops the
  matching entry from `context.metadata`'s provenance map, for any handler at any
  non-ERROR timing (not just `ContextCompactor.compact`/`prune`, which never had a path
  to `context.metadata` themselves). A completion-fix closed a gap this same fix
  introduced: `_execute_state_transition`'s POST_TRANSITION rollback now snapshots and
  restores `context.metadata` alongside `context.data`, so a handler-chain failure
  after a provenance-clearing deletion no longer leaves the plaintext restored with its
  digest permanently gone.
- **`create_fancy_header`'s box width now matches every sibling section box (D-019;
  `ab155a6`).** Capped to a fixed 60 columns (was `max(60, len(name) + 10)`), with the
  FSM name routed through `_fit()`. An FSM name over roughly 50 characters previously
  rendered a header box strictly wider than every box below it, with no truncation
  guard of any kind.
- **`execute_handlers` defers its context deep-copy (D-020, D-025; `7dfc60c`,
  `2154af2`).** `HandlerSystem.execute_handlers` now deep-copies `context` only once
  the first handler whose `should_execute()` passes is found, instead of
  unconditionally before the loop -- zero deep copies when no registered handler at a
  timing will run. A completion-fix hoisted the copy back above the per-handler
  execution `try:` (a regression the deferral itself introduced): a non-deep-copyable
  context value now raises a loud, correctly-attributed `TypeError` again in both error
  modes, instead of being silently swallowed in the default `error_mode="continue"` or
  misattributed to a handler that never actually ran in `error_mode="raise"`.
- **`WorkingMemory.to_dict()`/`from_dict()` round-trip `_hidden_buffers` symmetrically
  (D-021, D-026; `1a1c2f6`, `5eff8f5`, `47c081b`).** `to_dict()` now folds
  `_hidden_buffers` into its own flat return; `from_dict()` reads it from there by
  default, with an explicit kwarg still overriding. This closes a real gap in
  `fsm_llm_agents/memory_persistence.py`'s `save_working_memory`/`load_working_memory`,
  which call neither method with a `hidden_buffers` kwarg, so a custom hidden-buffer
  set previously reset to the default on every file-backed reload. **Fix, not a new
  behavior change to work around:** the reserved name `"_hidden_buffers"` is now
  rejected with `ValueError` at every buffer-creation entry point (`set`,
  `create_buffer`, `update_buffer`/`import_flat_data`, the constructor); previously a
  buffer literally named `_hidden_buffers` -- reachable from LLM-chosen input via
  `fsm_llm_agents/memory_tools.py`'s `remember(buffer=...)` tool -- had its entire
  contents silently destroyed by `to_dict()`'s unconditional overwrite of that key.
- **`llm.py`'s structural-presence `hasattr()` checks converted to `getattr()` (D-003,
  D-022; `80417cd`, `0881317`).** 3 of the 4 flagged sites (the streaming loop's
  `chunk.choices`; `_make_llm_call`'s `response.choices` and `choice.message`)
  converted to `getattr(obj, "x", None)`-style checks, proven behaviorally identical to
  their pre-conversion form for a genuinely missing (not just `None`-valued) attribute.
  The 4th (`choice.message.content`) is deliberately kept as `hasattr()` -- a
  documented exception, not an oversight: it must distinguish "attribute absent" from
  "attribute present with value `None`" (the H3 Ollama reasoning-only-reply recovery
  path needs exactly that distinction, and `getattr(x, "content", None) is not None`
  cannot make it), confirmed empirically by temporarily forcing the conversion and
  watching a real litellm-object-driven pre-existing test fail.

### Known limitations

- **7 items re-confirmed WONTFIX this plan, unchanged from the "Known limitations"
  list under 0.6.0 below** (`decisions.md` D-004 of `plan-2026-09-20T114608-a8e47b88`
  documents the reasoning for each): `handler_only_keys` stays opt-in (LV2-04); the
  Pass-1 half-committed-turn design (CF-08) and the CONTEXT_UPDATE-provenance decline
  (RB-04) still need their own superseding design, not a mechanical audit fix; the
  history-cap recall (LV2-10) and the developer-text newline-flattening (LS-10) are
  prompt-content changes gated on the still-stale `scripts/eval.py` baseline
  (re-running it is out of this plan's scope); RB-08's per-transition extra-call cost
  stays a documented-not-capped trade-off; and `EvaluatorOptimizerAgent.success` after
  the refinement cap is out of this package's scope (`fsm_llm_agents`).
- **The `scripts/eval.py` 95.3% baseline (N=3 median, Run 006) is now stale against
  five audit iterations of prompt-adjacent and core-engine changes**, not just the four
  the 0.6.0 entry below named -- this plan's own `context.py`/`pipeline.py` provenance
  fix touches the same handler-execution path a prompt-content change would. Still not
  re-measured; still a release gate before the baseline can be trusted.

Third-layer audit of `classification.py` / `memory.py` (`plans/plan-2026-09-20T165703-0d9c218e`,
iteration 1, "classification.py / memory.py audit, loop 1 of 5"). Two deep audits (16 memory
findings, 9 classification findings) drove 8 code steps, one documentation-truth step and one
NEW gated live suite; every code item ships with a regression test in `tests/test_fsm_llm/`.
Finding ids below are the audit's own (`memory #N`, `classification #N`). Full suite: 6,158
tests collected (was 6,091); `ruff`/`mypy src/fsm_llm/` clean after every code step.

### Fixed -- `classification.py` / `memory.py` audit loop 1 (plan-2026-09-20-0d9c218e, iteration 1)

- **`fsm_llm.memory.__doc__` was `None` (memory #1; `6122510`).** The module docstring sat
  BELOW `from __future__ import annotations`, so Python discarded it. Moved above the import;
  regression test asserts a non-empty `str`. The executor's sweep then found the SAME defect in
  81 modules repo-wide (D-003: `fsm_llm`, `fsm_llm_workflows`, `fsm_llm_agents`,
  `fsm_llm_monitor`); all swapped in `4aadd01`, with a filesystem-derived
  `tests/test_packaging.py` test that fails on any future module hiding its docstring behind
  the future import.
- **`WorkingMemory.from_dict` accepted malformed session data silently or with a raw stdlib
  error (memory #10, #11; `0a02716`).** A non-dict buffer body (`None`, a string, a list) now
  raises `ValueError` naming the buffer (`buffer 'core' must be a dict, got NoneType`); a
  `_hidden_buffers` value that is not a `list`/`tuple`/`set`/`frozenset` of `str` (a bare
  `str` was previously iterated character by character) raises `ValueError` naming
  `_hidden_buffers`. `API.restore_session` on a hand-corrupted session file raises with the
  buffer named and leaves no half-registered conversation.
- **`WorkingMemory.__init__` edges (memory #8, #9, #12, #13; `a311bb7`).** `buffers=[]` /
  `buffers=()` now means NO buffers (was: silently replaced by the four defaults through a
  falsy check); `initial_data` passed without a `core` buffer logs a WARNING naming the
  dropped key count instead of vanishing; `get_all_data`'s "last non-core buffer wins" shadow
  order is documented and pinned by a 3-buffer collision test; `to_dict` is documented as a
  one-level-shallow copy (nested containers are shared with the caller) at the source, not
  one file away. **Behavior change**: `WorkingMemory(buffers=[])` and
  `WorkingMemory.from_dict({})` now mean ZERO buffers (previously both silently produced
  the four default buffers); pass `buffers=None` (the default) to get the defaults.
- **`IntentRouter.validate()` dead branch and duplicated dispatch (classification #1, #7,
  #8; `345c28c`).** The unreachable "append fallback" branch is deleted with a one-line
  invariant comment (`ClassificationSchema.validate_schema` guarantees `fallback_intent in
  intent_names`); `route` and `route_multi` share one private `_resolve_handler(intent)` so
  both raise identically when handler and fallback are absent; `Classifier._kwargs` is typed
  `dict[str, Any]`.
- **`ClassificationPromptConfig` had no bounds (classification #3, #9; `ed40bea`).**
  `__post_init__` now enforces `1 <= max_intents <= MAX_MULTI_INTENTS` (new `constants.py`
  value, 5), `max_tokens >= 1` and `0 <= temperature <= 2`, each `ValueError` naming the
  field; a `prompt_config={"max_intents": 0}` on a `classification_extractions` entry is
  caught by the extraction site's existing narrow tuple, logged, and leaves the key unset.
  The field doc states that Ollama models force `temperature` to 0 under structured output
  (`apply_ollama_params`), and `ClassificationExtractionConfig.prompt_config` lists all six
  accepted keys. **Behavior change**: the bounds are enforced at construction, so
  `ClassificationPromptConfig(max_intents=6)` (previously accepted, any value above
  `MAX_MULTI_INTENTS`), `max_tokens=0` or `temperature=3.0` now raise `ValueError` instead
  of being carried into the prompt.
- **`_resolve_ambiguous_transition` caught `Exception` (classification #2; `d4acaac`).** The
  ambiguous-transition classifier site now catches the same D-004 soft-fail tuple the
  extraction site already used, hoisted to one module-level
  `_CLASSIFICATION_SOFT_FAIL_EXCEPTIONS` in `pipeline.py` (`ClassificationError`,
  `ValueError`, `TypeError`, `KeyError`, `RuntimeError`, `OSError`). Those still degrade to
  "stay in state" with `_classification_result.fallback = True`; programming errors
  (`AttributeError`, `ZeroDivisionError`) and `KeyboardInterrupt` now propagate through
  `API.converse` instead of being swallowed as a stay. **Behavior change**: a
  non-soft-fail exception (`AttributeError`, `ZeroDivisionError`, `IndexError`, ...)
  raised by `Classifier.classify` at the ambiguous-transition site now escapes
  `API.converse` as `FSMError` (`converse` wraps every non-`FSMError` exception);
  previously the turn completed as a "stay" and the caller never saw it.

### Changed

- **Classifier instances are cached per pipeline (classification #5; `7d2e81c`).** Both
  pipeline classification sites go through `MessagePipeline._get_classifier`, keyed by a
  content hash of schema + model + prompt config + connection kwargs (NOT
  `(state_id, field_name)`), bounded at `MAX_CLASSIFIER_CACHE_SIZE` (new `constants.py`
  value, 64) with oldest-entry eviction. One `Classifier` (pre-built prompts and schema, no
  per-call `self` writes) now serves every turn and conversation with the same key; a
  different schema still constructs a new instance. `patch("fsm_llm.pipeline.Classifier")`
  keeps working because construction goes through the module-level symbol.

### Added

- **Buffer-name exports (memory #15; `4aadd01`).** `BUFFER_CORE`, `BUFFER_SCRATCH`,
  `BUFFER_ENVIRONMENT`, `BUFFER_REASONING`, `DEFAULT_BUFFERS` and `DEFAULT_HIDDEN_BUFFERS`
  are importable from `fsm_llm` and listed in `__all__`.
- **`tests/test_fsm_llm/test_live_classification_memory.py` (`c462594`).** A 7-test live
  suite gated exactly like `tests/test_integration_ollama.py` (`integration` + `real_llm` +
  `slow` markers, `conftest.ollama_available()` skip), retaining raw litellm
  request/response pairs and printing them on failure: single- and multi-intent
  classification, `<intent>` injection resistance, entity extraction, a two-stage
  `HierarchicalClassifier`, a classified FSM transition with a working-memory session
  round trip, and a `ReactAgent` + `create_memory_tools` round trip. Iteration-1 result on `ollama_chat/qwen3.5:9b-q8_0` (run together with
  `test_integration_ollama.py`): **18 passed / 1 failed in 122.72s**. The one FAIL is
  F-LIVE-01 below. The 5 example runs at 9b all exited 0; `examples/classification/multi_intent`
  scoring 2/4 is an example-design artefact, not a classifier miss (OBS-LIVE-01:
  `primary_intent` is a `classification_extractions` field on the `listen` state only, so
  turns arriving in `handle_purchase`/`handle_question` keep the stale value; both turns that
  reached the classifier were classified correctly). Examples untouched.

### Docs

- **WorkingMemory's real prompt reach (memory #2-#4, #7, #14; `4aadd01`).** The
  `WorkingMemory` class docstring, `FSMContext` in `definitions.py`, `docs/architecture.md`
  and `docs/fsm_design.md` now state that working-memory data reaches ONLY the Pass-1
  per-field extraction prompt (through the default `context_keys`) and NOT the Pass-2
  response prompt, which is built from raw `context.data`; `to_scoped_view` drops its
  "scoped view for LLM prompts" claim (method kept); the `exclude=True` consequence for
  `model_dump()` is noted at the field.
- **`docs/api_reference.md` `is_low_confidence` example (classification #4; `4aadd01`).**
  The example now calls `classifier.is_low_confidence(result)` (the property on the result is
  a fixed 0.6 threshold, not the classifier's configured one).

### Known limitations -- still open for later loops of this plan

- **WorkingMemory is not wired into Pass 2 (memory #2, #5, #6).** Documented truthfully this
  loop, not changed: the response prompt never sees buffer contents, and the
  scoped-view/reasoning-buffer machinery has no consumer in the pipeline. Wiring is a
  design decision for a later loop, not a mechanical fix.
- **The classifier receives no conversation history or FSM context (classification #6).**
  `Classifier._call_llm` builds exactly two messages (static system prompt + raw user
  message); `ClassificationExtractionConfig.context_keys` only feeds a post-hoc debug
  snapshot. A bare "yes" reply whose intent is decidable only from prior turns cannot be
  classified correctly. A design gap, deferred: it changes prompt content and is gated on
  the still-stale `scripts/eval.py` baseline.
- **`WorkingMemory.search` and `fsm_llm_agents` `MemoryBackend` have different shapes
  (memory #16).** `MemoryBackend.search(query, k) -> list[(text, score, meta)]` vs
  `WorkingMemory.search(query, limit) -> list[(buffer, key, value)]`; passing a
  `WorkingMemory` to `augment_task_with_memories` raises `TypeError` inside that function's
  broad catch and silently degrades to "no recall" with a WARNING. One of three
  incompatible memory shapes in the repo; consolidation is out of this core-package loop's
  scope.
- **F-LIVE-01 -- `ReactAgent` + `create_memory_tools` on 9b never calls `remember`
  (`fsm_llm_agents`, MEDIUM).** `TestLiveMemoryAgent::test_remember_then_recall` fails
  (`remember did not write 'teal' into WorkingMemory: {}`) and the untouched
  `examples/agents/memory_agent` reproduces it (`Tools used: []`, budget exhausted, empty
  memory). Mechanism read from the 14 raw calls: the ReAct loop's per-field extraction
  (`tool_name`, `tool_input`) driven by the synthetic `Continue.` message returns `null` on
  every iteration ("contains no new information"), then `"none"` once the Pass-2 prose
  "I've noted that your favorite color is teal." has leaked into the history. A model-side
  under-call on the ReAct extraction path, the same family as the fixed 4b under-call, now
  on 9b with a different prompt; not a `classification.py`/`memory.py` defect (neither was
  invoked). The test is kept strict; the agents package is out of this plan's scope.
  (Superseded by loop 2 below: the root cause turned out to be in core `pipeline.py`, and
  is fixed.)

### Fixed -- `classification.py` / `memory.py` audit loop 2 (plan-2026-09-20-0d9c218e, iteration 2, final loop)

Iteration 2 re-audited iteration 1's own diffs (adversarial review W1-W7, N8-N12) and
root-caused F-LIVE-01 from the retained raw calls. Every code item ships with a regression
test in `tests/test_fsm_llm/`. Full suite: 6,174 tests collected (was 6,158);
`ruff`/`mypy src/fsm_llm/` clean after every code step.

- **F-LIVE-01 root cause was in core, not the agent: the greeting Pass-2 call ignored an
  empty `response_instructions` (D-006; `ead5535`).** `MessagePipeline.process_message`
  skips Pass 2 when the state's `response_instructions` is empty, but
  `generate_initial_response` (the `start_conversation` greeting) did not, so every ReAct
  agent run opened with a full prose greeting built from the `think` state's purpose; that
  prose then sat in the history and poisoned the `tool_name` extraction on the first real
  turn. `generate_initial_response` now mirrors the sync site exactly: on an empty
  `response_instructions` it makes the `"."` sentinel request (no litellm call in
  `LiteLLMInterface`), records the synthetic `[<state_id>]` marker as the first assistant
  entry and returns it. RED test written against the old code first; a
  `response_instructions=None` control still builds the full prompt.
- **W2: classifier-cache race and unguarded construction (`4047ccd`).** The check /
  construct / evict / insert sequence in `_get_classifier` is one critical section under a
  new `_classifier_cache_lock` (review reproduced `RuntimeError: dictionary changed size
  during iteration` on the unlocked code); the `_get_classifier` call at the
  ambiguous-transition site moved INSIDE the soft-fail `try`, so a classifier construction
  failure (`ValueError` from a bad `prompt_config`) degrades to "stay" at both sites
  instead of escaping only at one.
- **W1 / N11: cache key digested non-JSON-native connection kwargs via `default=str`
  (`40204c8`).** A `SecretStr` (whose `str()` is a fixed mask) or any object whose `str()`
  does not identify its value made two different credentials hash to the SAME key, so a
  cached instance built for one could be served for the other. `_get_classifier` now BYPASSES the cache for a
  non-JSON-native connection kwarg (fresh construct, no insert, no lock); `default=str` is
  gone. Documented at the D-001 anchor: cached `Classifier` instances retain `api_key` and
  other connection credentials for the pipeline's lifetime, up to
  `MAX_CLASSIFIER_CACHE_SIZE` copies, by design.
- **W4: cache-key components pinned (`1d0ad06`).** Tests assert that a changed
  `prompt_config` and a changed connection kwarg each produce a distinct cache entry, and
  that the same inputs hit the same instance.
- **N10: `Classifier._call_llm` malformed-shape wrap (`118aff1`).** The post-call parsing
  (`.choices[0].message.content` chase and `_extract_response`) is wrapped in
  `except (AttributeError, IndexError, TypeError)` re-raised as
  `ClassificationResponseError("Malformed LLM response shape: ...")`, so a choice without
  `.message` now lands in the existing `ClassificationError` family instead of leaking a
  raw `AttributeError`; the existing `"Empty response from LLM"` raise is untouched.
  Caveat: defensive, not reachable via the installed litellm's real response objects
  (those always carry `.message`); the RED test uses a `SimpleNamespace` shape.
- **N8 / W7: `restore_session` collapsed an explicit empty buffers map into the defaults
  (`3b9deac`).** `API.restore_session` now maps a missing / `None` `working_memory.buffers`
  to the default buffers and, when no `hidden_buffers` key is present, the default
  hidden set (`{"metadata"}`; an explicit list, `[]` included, is honoured as-is --
  loop-2 completion fix `iter-2/step-6.1`), and an explicit `{}` to
  `WorkingMemory.from_dict({})` (zero buffers), matching loop 1's `from_dict({})` contract;
  a `# DECISION plan-2026-09-20T165703-0d9c218e/D-001` anchor in `memory.py` pins
  `buffers if buffers is not None else DEFAULT_BUFFERS` against the tempting
  `buffers or DEFAULT_BUFFERS` simplification.
- **W3: prompt-reach docs corrected (`cc71820`).** `FSMContext` (class docstring and the
  `working_memory` Field) and `docs/architecture.md` now say where the buffer data goes:
  `get_user_visible_data()` feeds the Pass-1 per-field prompt AND the public
  `ResponseGenerationRequest.context` at the three Pass-2 sites when a response is
  generated; the shipped `LiteLLMInterface` never reads `request.context`, so no buffer
  data reaches the Pass-2 PROMPT, but a custom `LLMInterface` that reads it will see it.
  The earlier "only Pass-1" wording was wrong about the request object.
- **W5: live suite hardened (`8b33388`).** The `<intent>` injection test asserts the
  injected message still classifies as `check_balance` (it previously only asserted "no
  crash"); the raw-call recorder writes `rec.dump()` for EVERY test (pass or fail) to
  `$FSM_LLM_LIVE_RAW_DIR/<nodeid>.txt`, so passing tests can be audited from their raw
  calls; the memory-agent test mirrors the example's `temperature=0.7` /
  `max_iterations=5`.

Live evidence (iteration 2, `ollama_chat/qwen3.5:9b-q8_0`, n=1, no commit): the live suite
plus `test_integration_ollama.py` ran **18 passed / 1 failed in 246.83s**. The F-LIVE-01
greeting mechanism is GONE: `remember` fires at iteration 1 with the correct
`{key, value, buffer}` payload and the first `agent.run` reaches `conclude`. The remaining
FAIL is a NEW known limitation, **F-LIVE-02** (`fsm_llm_agents`, not fixed, out of this
plan's scope): after a tool result the ReAct loop can sit in `think` with every transition
BLOCKED until the budget is exhausted. Three framework-side links: (a) a `null`
`tool_input` falls back to the whole task string, so `recall` searches for
`"Use the recall tool: what is the user's favourite colour?"` and `WorkingMemory.search`
(substring match) finds nothing; (b) the per-field extraction prompt's closing IMPORTANT
block ("extract from the current message") overrides its own NOTE about the `Continue.`
signal, so `tool_name` comes back `null` on every continuation turn while
`should_terminate=false` stays pinned in `Already extracted:` and is never re-asked; (c)
the 3-consecutive-no-tool stall detector (`AgentToolExecutor`, POST_TRANSITION) and the
iteration limiter (PRE_TRANSITION) never run while transitions are blocked, so the net
written for exactly this case is unreachable and the run ends only on the outer
`BudgetExhaustedError`. The same mechanism drives `examples/agents/memory_agent` (1/5;
task 1 additionally sends `remember` with the whole fact stuffed into `key` and no
`value`, rejected by the tool; the example's `memory_populated` check counts the four
default buffers, so it is `True` on an empty memory). Two further raw-read observations,
recorded not fixed: the multi-intent ranking follows the order intents are mentioned in
the message rather than any salience, and a terminal reply may claim a side effect
("I've saved that") that no tool performed. `examples/classification/multi_intent` 2/4 is
unchanged from iteration 1 (OBS-LIVE-01, example design). Examples untouched.

These changes are not yet released; no version bump is cut by this plan (D-004).

## [0.6.0] - 2026-09-20

Core-engine audit release (4 audit-fix loops over `src/fsm_llm`, each verified live on
`ollama_chat/qwen3.5:9b-q8_0` and adversarially reviewed). **Read before upgrading:**
several behaviours changed on purpose (see "Changed -- public contract" below) --
notably the ERROR-handler and re-entrancy contract, provenance-gated bulk corrections,
`extract_json_from_text` returning `dict | None`, the `fsm_id` hash of
`API.from_definition(FSMDefinition(...))` (now carries `handler_only_keys: []`), extra
LLM calls on back-edge revisits and on an Ollama null-extraction memo hit/miss, and a
wider prompt sanitizer. The `scripts/eval.py` 95.3% health baseline was **not re-measured**
against these prompt-content changes and is stale. See "Known limitations" for what is
not fixed.

### Fixed -- core (`fsm_llm`) audit remediation (plan-2026-09-19-21cd7f8e, iteration 1)

Each fix was reproduced RED first and pinned in `tests/test_fsm_llm/test_audit_iter1_seam.py`
(plus updated tests where a superseded contract was encoded). Decision ids refer to
that plan's `decisions.md`.

- **LV-01, typed field-extraction schema (D-001, D-002).** On Ollama models the
  per-field extraction `response_format` now types `value` per `field_type`
  (str `[string,null]`, int/float `[number,string,null]`, bool `[boolean,string,null]`,
  list/dict/any analogous) instead of `"value": {}`, which made the model return
  dict-wrapped junk. `_coerce_str` and `_coerce_bool` now raise on `dict`/`list`
  input, so a str/bool field no longer stores the repr of a container as a valid
  value (it fails extraction instead). Supersedes the D-018 "never raise" clause of
  plan-2026-07-18T051819-80b0bd4d for those two coercers. A `confidence == 0.0`
  extraction is no longer stored. **Behavior change**: `field_type="any"` on the
  Ollama path can no longer yield an object; declare `field_type="dict"` for that.
- **LV-04, plain-text streaming prompt (D-003).** `converse_stream` builds the
  Pass-2 prompt with a plain-text response-format section
  (`build_response_prompt(..., plain_text_response=True)`), so streamed tokens and
  stored history no longer carry the `{"message","reasoning"}` envelope. The sync
  path and terminal states with an output format keep the JSON prompt. **Behavior
  change**: sync and stream Pass-2 prompts now differ.
- **LV-03, bulk correction overwrite (D-004).** A later-turn correction returned by
  the bulk extraction now overwrites an already-set key when the key is covered by
  one of the state's own field configs, was not extracted this turn, is non-null and
  differs, and the FSM is not agent-managed. Instruction-only keys and agent FSMs
  keep skip-if-set. **Behavior change**: on non-agent FSMs a handler-set value for a
  config-covered key can be overwritten by a bulk LLM value.
- **CF-05, ERROR handlers (D-009).** `HandlerTiming.ERROR` handlers now fire on
  `FSMError` (including `LLMResponseError`, the LLM-outage case) and on the streaming
  path, via one `FSMManager._fire_error_handlers`. The `FSMError` is still re-raised
  unwrapped; `KeyboardInterrupt`/`SystemExit`/`GeneratorExit` still run no handlers.
  **Behavior change**: a critical ERROR handler failure now replaces the `FSMError`
  as the raised exception.
- **CF-01, context scope enforced in prompts (D-005).** `context_scope.read_keys` is
  now applied to the context shown in the Pass-2 prompt on the turn, stream and
  greeting paths (`build_response_prompt(..., context=...)`); it previously only
  filtered `request.context`, which the LLM never read. Unscoped states are
  byte-identical.
- **CF-02, classification-owned keys (D-006).** Keys owned by a
  `classification_extractions` entry are no longer also auto-minted as plain
  `requires_context_keys` extraction configs, so a below-threshold classification can
  no longer be bypassed by the plain extractor and one LLM call per turn is saved.
- **CF-04, classifier stay is not a transition (D-007).** A classifier error or the
  fallback intent no longer counts as a transition to the current state
  (`_resolve_ambiguous_transition` returns `None`): no PRE/POST_TRANSITION handlers,
  no post-transition re-extraction. Declared self-loops are unchanged.
- **CF-03, classifier connection (D-008).** The `Classifier` now inherits
  `api_key`, `api_base`, other litellm kwargs and `timeout` from the
  `LiteLLMInterface` (`_classifier_connection_kwargs`), so proxy and self-hosted
  users' classification goes to their endpoint. A per-config `model` on a different
  model does not inherit the key.
- **DH-01 / DH-19, linear-time parsing (D-010).** `extract_json_from_text` scans
  code fences with `str.find` (was a cubic regex: 5000 spaces took about 80 s) and
  `strip_think_and_fences` strips `<think>` blocks in one linear pass (was quadratic
  on unclosed tags).
- **DH-08 / LS-05, dict-or-None JSON (D-010).** `extract_json_from_text` now honours
  its `dict | None` annotation: valid non-object JSON (`42`, `[1,2]`, `true`, `"hi"`)
  returns `None` instead of the raw value. `Classifier.classify` raises
  `ClassificationResponseError` (was `AttributeError`) on non-dict JSON, and
  `LiteLLMInterface.extract_bulk_data` returns an empty response on non-dict JSON and
  tries one `extract_json_from_text` recovery parse when `json.loads` fails.
  **Behavior change**: a caller that relied on a list being returned now gets `None`.
- **DH-02 / DH-03, null-safe validator and visualizer (D-011).** `validator.py` and
  `visualizer.py` use `.get(k) or []` for `conditions`, `requires_context_keys` and
  `required_context_keys`, so a `model_dump()`-ed FSM (explicit `null` Optionals)
  validates and renders. Sibling sites in `fsm_llm_monitor/bridge.py` and
  `fsm_llm_agents/meta_builders.py` got the same treatment.

### Fixed -- core (`fsm_llm`) and agents audit remediation (plan-2026-09-19-21cd7f8e, iteration 2)

Iteration 2 fixes the defects the iteration-1 review and the re-audit found in
iteration 1's own changes, plus the live-proven and small localised backlog items.
Each fix was reproduced RED first and pinned in
`tests/test_fsm_llm/test_audit_iter2_seam.py` (plus updated tests where a superseded
contract was encoded). Decision ids refer to that plan's `decisions.md` (D-014..D-025).

- **LS-18 / RA-05, Ollama detection by prefix (D-021, e768178).** `is_ollama_model`
  matches only the `ollama/` and `ollama_chat/` prefixes (case-insensitive). A model
  such as `azure/gpt-4o-not-ollama` no longer gets the Ollama `json_schema` grammar
  and forced temperature 0. The classifier's connection kwargs now drop `schema`,
  `model` and `config`, so a `config` kwarg no longer raises `TypeError` in both
  classification paths.
- **Bulk pass provenance and coercion (D-015, 8e178fd).** The bulk pass overwrites a
  stored key only when the stored value is still exactly what the pipeline extracted
  (a sha256 digest of the value is recorded in `context.metadata` at the two Pass-1
  commit sites), so a handler-seeded gate value is never flipped (reviewer repro5:
  `is_verified` False -> True). Bulk values for config-covered keys go through the
  same coercion and validation as per-field values, so a dict no longer lands in a
  `str` key and `24` stays an int instead of becoming `'24'` (RA-02).
- **Re-entrancy guard and ERROR handler merge (D-018, b3431f3).** A same-conversation
  `converse` or `converse_stream` called from inside a handler while a turn is in
  flight raises `FSMError` instead of nesting (reviewer repro1: depth 99 and 99
  provider calls for one user turn, now depth 1). An ERROR handler's returned dict is
  no longer merged into the conversation whose turn was just rolled back (RA-07).
- **RA-01, kwargs tool called with `input=` (D-017, 3f96567).**
  `normalize_tool_input` json-decodes a string `tool_input` that is a JSON object, so
  a keyword-argument tool is called with its arguments instead of `input=`.
- **RA-01b, ReAct tool never ran on the Ollama grammar (D-024, d22a0f2).** The
  react and reflexion think states declare `tool_name` (`str`) and `tool_input`
  (`dict`) as explicit `field_extractions`. Live on `ollama_chat/qwen3.5:9b-q8_0`
  (n=3): tool ran 3/3, versus 0/3 under the auto-minted `any` grammar.
- **RA-04 / RA-06, JSON extraction (d4ec278).** `extract_json_from_text` resumes the
  brace scan after a fenced non-object, so a fenced array's interior object is no
  longer returned, and Strategy 1 and 2 tolerate `RecursionError` on deeply nested
  input instead of raising.
- **LS-01 / LS-06, sanitizer and forbidden names (b674384).** The prompt sanitizer no
  longer passes an unterminated safe tag that swallowed a closing tag (`<b </task>`).
  `is_forbidden_context_entry` also tests the snake-cased form of camelCase key names
  (`apiKey`, `authToken`, ...). Benign prompts are byte-identical; the change for
  non-benign input is not measured by `scripts/eval.py`.
- **LS-02, `extracted_data` prompt section (5d964ec).** The Pass-2 `extracted_data`
  section goes through the security filter, so a declared secret-named field is not
  echoed, and a `datetime` value no longer drops the whole section. Benign data
  renders byte-identically.
- **CF-06, bulk-pass prompt and result filtering (D-019, b5e06b5).** The bulk
  extraction prompt sanitizes user text, and the bulk result drops the `agent_trace`
  marker and forbidden-name keys on both call sites. The sanitization of non-benign
  text is not measured by `scripts/eval.py`.
- **RA-03, classification-owned keys on the bulk pass (D-019, da4eea4).** The
  additive bulk pass no longer fills a key owned by a `classification_extractions`
  entry on a non-agent FSM, so a below-threshold classifier cannot be bypassed by
  bulk extraction. Agent FSMs keep the bulk fill.
- **LV2-01, structured terminal reply (D-020, 3d440ec).** A schema-valid JSON reply to
  a requested `response_format` that has no `message` key is the reply (verbatim JSON
  text in the response and history) instead of the generic apology. Replies over the
  5000-character cap still degrade; behaviour without a schema is unchanged.
- **LV2-02, empty greeting (D-020, 3edf412).** A Pass-2 reply that would be the
  generic apology (for example an empty `message`) is retried once and the retry
  result is returned; an error on the retry keeps the first apology. Normal replies
  make one call.
- **CF-07, stacked `save_session` (D-022, 6b924f7).** `save_session` on a stacked
  conversation saves the root frame's state, data, history and working memory, so
  `restore_session` no longer meets a sub-FSM state that does not exist in the root
  definition. Unstacked saves are unchanged.
- **DH-07 / EF-03, zero-handler turns (D-022, 470ba14).** `execute_handlers` returns
  immediately when no handler subscribes to the timing (new
  `HandlerSystem.handlers_at`), so a zero-handler advance turn drops from 14 to 4
  context deep-copies and a non-copyable context value no longer crashes handler
  timings. The pre-turn rollback snapshots are untouched.
- **LS-05 remainder, bulk confidence (D-023, 8099e54).** `extract_bulk_data` keeps the
  extracted data when the model's confidence cannot be coerced (`"high"`, `null`, an
  object); the confidence falls back to 1.0 instead of failing the whole bulk pass.
- **DH-09, `<=` and `>=` on numeric strings (D-023, d3d2b2a).** `1 <= "1.0"` and
  `"2.0" >= 2` are true (they previously fell back to string ordering).
  `==` and soft equals are unchanged.
- **DH-10, `validation_rules` typing (D-023, 896fc13).** `min_length` and
  `max_length` must be int, `allowed_values` a list, `pattern` a compilable regex
  string; a bad rule fails at FSM load and in `fsm-llm-validate` instead of as a
  `TypeError` on the first extraction turn.
- **LV2-05, anchors and pin (D-011, D-016, 0bab5a5).** Qualified DECISION anchors for the
  exact-0.0 confidence rejection (D-016) and the null-safe validator reads (D-011),
  and a seam test pinning the accepted below-threshold classifier stall. Comments and
  tests only.

### Fixed -- core (`fsm_llm`) and agents audit remediation (plan-2026-09-19-21cd7f8e, iteration 3)

Fix-of-a-fix items from the iteration-2 review and a fresh re-audit (RB-01..RB-12),
two agent-pattern fixes with live proof, and first-touch docs that load. Each fix was
reproduced RED first and pinned in `tests/test_fsm_llm/test_audit_iter3_seam.py` (or
the agent test files). Decision ids refer to that plan's `decisions.md`.

- **RB-01, uncoercible confidence (D-028, 0a5c59b).** A per-field extraction whose
  `confidence` cannot be coerced (`"high"`, `null`, an object, `"95%"`) keeps the
  returned value at confidence 0.5 at both field-extraction rungs, instead of falling
  to the unstructured rung and storing the raw JSON text as the field value.
- **RB-02, structured reply without `message` (D-028, a9dd46f).** A reply to a
  requested `response_format` whose object has no `message` key but has a `reasoning`
  key reaches the user as the JSON text, not as the reasoning alone.
- **RB-10, JSON before a fenced example (D-029, 05744ec).** `extract_json_from_text`
  skips a fenced non-object by blanking its span in place (length-preserving), so an
  object that appears BEFORE a fenced example is found again, the fenced interior is
  never recovered, and Strategy 3 and Strategy 4 read the same string.
- **RB-06, quadratic tag sanitizer (D-029, eb63863).** The prompt tag sanitizer's
  attribute tail excludes `<`, so `<a<a<a...` is linear (10k characters: 2.5 s to
  under 0.5 s) and an unterminated `<b ` can no longer swallow text; its closing tags
  stay escaped. (Iteration 4 found this change also stopped escaping a closing tag with
  a nested `<`, `</original_input <b>`; restored under D-047, see the iteration 4
  section.)
- **RB-11, duck-typed handler system (D-029, 44f89ae).** `handlers_at` is an optional
  fast-path hook, so a `handler_system` that only implements `execute_handlers` starts
  and converses again.
- **LS-09, inline fences kept (D-030, b126340).** `strip_think_and_fences` strips a
  code fence only at the START of the reply, so an inline fence in prose keeps its
  markers. JSON in a mid-text fence is still recovered by `extract_json_from_text`.
- **LS-03 / LS-04 / RB-03, plain-text rung (D-030, b1eac34).** The plain-text rung
  strips `<think>` blocks first and replaces brace-shaped text by the message or
  apology only when it parses as JSON, so `{name}, welcome! ...` and `{1, 2, 3}` reach
  the user after one provider call and a think-prefixed reply is no longer shown raw.
- **RB-05, provenance survives a restart (D-031, b2e2144).** `save_session` persists
  the provenance digests in `SessionState.metadata["pipeline_extracted"]` and
  `restore_session` re-seeds them, so a correction lands after a restart while a
  handler-seeded or `update_context` value is still never overwritten. An old session
  file restores an empty map.
- **RB-12a, refused corrections are reported (D-032, 96a42ac).** A bulk correction the
  provenance rule refuses and the user's message contains is carried on
  `DataExtractionResponse.rejected_corrections` (default `{}`); the stored value is
  unchanged. Ungrounded, landed and agent-managed cases report nothing.
- **RB-12b, reply is told (D-032, 8ee02e3).** `build_response_prompt` takes an optional
  last `rejected_corrections` argument and both Pass-2 call sites pass it, so the reply
  is told the requested value was NOT applied. A turn without a rejection builds a
  byte-identical prompt (hash pinned).
- **RB-07, opt-in `handler_only_keys` (D-033, 0086e60).** New `FSMDefinition` field
  (default `[]`, behaviour unchanged): a listed gate key is dropped from the bulk
  return, the per-field configs and the post-transition configs, so user text cannot
  write it. A handler write, `update_context` and `initial_context` still work; a
  stacked child uses its own list.
- **RB-08, back-edge correction (D-034, D-042, 765dda1).** A transition into a
  different state whose own config-covered key is already set with provenance re-runs
  the target state's Pass-1 extraction, so a same-message correction lands
  ("wait, change my name, it's Bob Jones"). A bare "go back", a self-loop, an
  agent-managed FSM, a handler-seeded key, a forward hop into an empty state and a
  state that owns `classification_extractions` make no extra call and keep the stored
  value.
- **RB-09, memoised identical null extractions (D-035, 1332ac9).** On an Ollama model
  (exact `ollama/` or `ollama_chat/` prefix, temperature 0) an identical null per-field
  extraction is memoised within one extraction call, keyed on field name plus the built
  prompt and message: 3 null keys at `extraction_retries=3` cost 3 provider calls
  instead of 12. Non-Ollama providers, exceptions and successful results are never
  memoised.
- **LV4-04, evaluator_optimizer success (D-036, 2571213).** `EvaluatorOptimizerAgent.run`
  passes `generated_output` as an extra answer key, so a run whose final context holds
  a non-empty `generated_output` and no `final_answer` reports `success=True` instead
  of reading as a prose fallback. An empty output still fails.
- **LV4-01, plan_execute plans (D-036, 4b93236; the seed removal was REVERSED in
  iteration 4, D-046).** `PlanExecuteAgent` no longer seeds
  `plan_steps` with an empty list, so the pipeline's skip-if-set filter stops reading
  `[]` as already set and the plan is extracted (live A/B on `qwen3.5:9b-q8_0`: a
  non-empty plan 3/3 under both the seed-removal and an empty-as-unset variant; the
  smaller change shipped). `success` stays False on those runs, see Known limitations.
- **LV2-05, below-threshold classification is visible (D-037, b44d223).** A
  classification result discarded for being below its `confidence_threshold` is a
  WARNING naming field, intent, confidence and threshold instead of a debug line.
  Behaviour is unchanged.
- **First-touch docs load (D-037, 6a349c7).** The FSM snippets in `README.md`,
  `docs/quickstart.md`, `src/fsm_llm/README.md` and `CLAUDE.md` now pass the loader
  (`description` is required at FSM, state and condition level; classification
  `schema` shape), and `tests/test_fsm_llm/test_docs_snippets.py` loads each one. False
  claims corrected: `required_context_keys` never blocks a transition, handlers do not
  see `_user_input`, `push_fsm` returns the greeting, `pop_fsm` takes the root id and
  merges only declared keys, ERROR handlers skip `start_conversation`, `write_keys` is
  advisory, a custom `LLMInterface` needs `extract_bulk_data`, the CLI needs
  `LLM_MODEL`.
- **Anchors (D-038, D-039, D-040, 636d7e9).** The inline LS-01, LS-02 and LS-06
  comments in `prompts.py` and `constants.py` became qualified DECISION anchors naming
  the constraint and the rejected alternative. Comment-only.

Not done in iteration 3:

- **RB-04, normalising CONTEXT_UPDATE handler (D-041, declined).** Recording the
  digest of the post-handler value cannot tell a normalisation (`blue` -> `BLUE`) from
  an authoritative override (`blue` -> `HANDLER`) and would let a later bulk turn
  overwrite a value a handler deliberately replaced. Provenance keeps recording the
  PRE-handler value (fail closed). The step-9 patch is kept as reference only.
- **GD-22, validator warning for a condition on a key nothing extracts (D-037,
  discarded by its own gate).** The prototype added warnings on 12 of 55 shipped FSMs
  (keys a handler or the engine sets) and was about 30 lines against the 15-line cap.
  Deferred: a silent-for-handler-keys warning needs a declared list of those keys on
  the FSM, a schema change.

### Fixed -- core (`fsm_llm`) and agents audit remediation (plan-2026-09-19-21cd7f8e, iteration 4, final loop)

The final loop repairs what iterations 1 to 3 introduced and closes the small gaps their
live runs and reviews proved; no feature was added. Each fix was reproduced RED first and
pinned in `tests/test_fsm_llm/test_audit_iter4_seam.py` (or the agent test files).
Decision ids refer to that plan's `decisions.md`.

- **Sanitizer nested-`<` bypass restored (D-047, d7f1679).** Iteration 3's
  denial-of-service fix (RB-06) silently stopped escaping a closing tag with a nested
  `<` (`</original_input <b>`, `</user_message <i>`, `</task <b>`), so a hostile user
  message could close the prompt's own wrapper tag and inject instructions. The tag
  tail is now `(?:[^>]{0,256}/?>|(?=[^>]{257}))`: a tag with a nested `<` or a tail longer
  than 256 characters (a padded closer) is escaped again (the overflow arm is a
  zero-width lookahead, so only the `<` and the name are escaped; the first version
  consumed 257 characters and escaped benign prose, fixed in the final review round,
  concern 1), and the scan stays linear
  (`"<a" * 10000` and `"<" + "a" * 100000` each under 0.5 s). A benign `< name`
  (`latency < threshold`) is kept raw only when NO `>` follows it anywhere in the
  text (D-054): a padded opener (`< name` + 257 or more characters + `>`) is escaped,
  as it was in ed6cffd. A differential test compares the new pattern with the
  pre-D-029 pattern on every token string of length up to 6 AND on padded shapes of
  258 to 1000 characters (the short corpus alone could not see a padded shape, it
  stops at 24 characters); the documented residual set is asserted explicitly.
  Three tests that pinned the weakened iteration 3 output were rewritten with
  annotations, and the reversed-ordering payloads were added.
- **`<rejected_corrections>` honours `context_scope.read_keys` (D-032, final review
  concern 3).** The refused-value block bypassed the scope that hides a key from
  `<current_context>`, so a value the state hides (widened in iteration 4 to every
  instruction-only key) still reached the Pass-2 prompt. The rejected dict is now
  filtered through the same scope on the turn and stream paths; a state without a
  scope, and the stored data, are unchanged. A block whose keys are all out of scope
  is omitted.
- **`plan_execute` on an unplannable task ends normally again (D-046, 4ecbfce).**
  Iteration 3 removed the `plan_steps: []` seed, so a task the model cannot plan raised
  `BudgetExhaustedError` after 36 wasted model calls. The seed is restored (it is the
  only exit of the `plan` state) and the pipeline's skip-if-set filter reads an empty
  list or dict as unset for agent-managed FSMs only, so the seeded plan is still
  extracted: the task returns `AgentResult(success=False)` after 1 `plan_steps` ask.
  A non-agent FSM that seeds `[]` for a config-covered key is unchanged.
- **A reply that ends in a code block keeps its closing fence (D-048, f736c86).**
  `strip_think_and_fences` strips the closing fence only when a leading fence was
  stripped; a fully fenced JSON reply is still unwrapped, and `extract_field` and
  `extract_bulk_data` still recover an object followed by a stray closing fence. One
  annotated `_STRIP_CORPUS` row was rewritten (a lone stray closing fence is now kept
  by the helper).
- **The "not applied" note needs a whole-token match (D-049, 007a405).** The grounding
  test for `rejected_corrections` was a raw substring, so `bored now` grounded `red`
  and `1500 items` grounded `500`. It now requires a whole token of at least 3
  characters; `make it red` and `make it red.` still ground.
- **A failed bulk extraction is surfaced to Pass 2 (D-050, 07afae1; LV5-01).** When the
  bulk extraction call raises, the reply used to claim an update the store never made.
  `DataExtractionResponse` gains `extraction_failed: bool = False` and
  `build_response_prompt` gains a last optional parameter `extraction_failed`, which
  adds one plain line saying a restated value may not have been stored. A turn without a
  failure builds a byte-identical prompt. One annotated iteration 3 test that pinned the
  exact signature tail was rewritten.
- **`fsm-llm-validate` warns about a `handler_only_keys` entry that protects nothing
  (D-051, ed39590).** One WARNING for a listed key no state references (a likely typo)
  and one for a listed key that is a `classification_extractions` field name (the
  classification channel is not covered). Warnings only, silent for an empty list, zero
  new warnings on every shipped example and agent builder.
- **An instruction-only key is reported when a correction is refused (D-052, 1b53d7c;
  LV5-03).** A bulk value for a key with no field config that already holds a different
  value is now listed in `rejected_corrections` under the same grounding test, for
  non-agent FSMs only; the stored value is still never overwritten. The change is net
  minus one source line, inside the plan's 10-line gate.
- **Docs-snippet coverage extended (D-052, ea49e53, test only).** The docs-snippet test
  also scans `docs/api_reference.md`, `docs/architecture.md`, `docs/fsm_design.md` and
  `docs/handlers.md`; none carries a full FSM (`"initial_state"`) today, so they add no
  case and the guard still requires the four first-touch files to contribute. `docs/fsm_design.md`
  carries no full FSM (no `"initial_state"` block), only fragments, and fragments are
  NOT loaded by the test: a wrong fragment there is not caught.
- **LV6-01, an applied correction listed as rejected (D-052, 2da48c4).** Introduced by
  the LV5-03 change above (1b53d7c) and found by the live audit, fixed inside this run:
  on a back-edge turn the confirm state's refusal was merged into
  `rejected_corrections` and never pruned, although the target state's re-extraction
  then applied the value, so Pass 2 was told the opposite of the store (3/3 live looks;
  0/3 in iteration 3). After the re-extraction an entry is dropped when the stored
  value now equals it under the same trimmed, case-insensitive comparison as the merge
  point; a value a handler edited or that never landed stays listed. One live look after
  the fix showed no block (n=1, an observation, not a rate).

Regressions found and fixed inside this run (stated plainly):

- **Iteration 3 introduced two regressions, both found by the iteration 3 adversarial
  review and fixed in iteration 4.** (1) The sanitizer nested-`<` bypass: the RB-06 fix
  stopped escaping a closing tag with a nested `<` (introduced in iteration 3 at
  eb63863, fixed at d7f1679). (2) The `plan_execute` exception: the LV4-01 seed removal
  made an unplannable task raise `BudgetExhaustedError` after 36 model calls instead of
  returning `success=False` (introduced at 4b93236, fixed at 4ecbfce).
- **Iteration 4 introduced one regression, LV6-01, found by the live audit and fixed
  inside the same iteration.** Introduced by the LV5-03 change (1b53d7c), fixed at
  2da48c4. The live audit found it by reading the raw prompts of the back-edge turn; the
  check booleans (5/5 in all three looks) did not see it. The offline suite did not
  either, because no test built a state whose key is config-covered only in the target
  state.

### Changed -- public contract (iterations 1 to 4)

- **Bulk overwrite is provenance-gated (D-015).** Supersedes iteration 1's D-004
  overwrite rule. A later-turn correction returned by the bulk pass replaces a stored
  key only when the key is config-covered, the FSM is not agent-managed and the
  stored value is still exactly the value the pipeline extracted. A handler-set or
  `update_context`-written value for a config-covered key is never overwritten by the
  bulk pass (iteration 1 allowed it). Iteration 3 (D-031) persists the
  provenance across `save_session`/`restore_session`; a session file from before that
  has none, so corrections fall back to skip-if-set until the pipeline re-extracts the
  key. A correction that changes only letter case or
  whitespace counts as no change.
- **Exact-0.0 confidence is rejected (D-016).** A per-field extraction the model
  reports at `confidence` exactly `0.0` is not stored, whatever
  `confidence_threshold` is. There is no knob to accept a 0.0-confidence value.
- **Re-entrant `converse` raises (D-018).** Calling `converse` or `converse_stream`
  on a conversation from inside a handler (or between `next()` calls of an open
  stream) while a turn is in flight raises `FSMError`. `update_context` stays
  allowed.
- **ERROR handler return value is not merged (D-018).** The dict an ERROR-timing
  handler returns is dropped (debug-logged). `update_context` is the supported write
  path for an ERROR handler.
- **Forbidden-name keys are not captured by the bulk pass (D-019).** An
  instruction-only field whose name matches a forbidden pattern (`password`,
  `api_key`, ...) is no longer captured by the bulk pass; declare it in
  `field_extractions` or `required_context_keys` to capture it.
- **Ollama detection by prefix (D-021).** `is_ollama_model` is true only for
  `ollama/...` and `ollama_chat/...`; a model that merely contains "ollama" in its
  name (a proxy route) now takes the non-Ollama path.
- **Bad `validation_rules` are rejected at load (D-023).** An FSM definition whose
  `validation_rules` carries a wrongly-typed value (`min_length: "abc"`,
  `allowed_values: 5`, `pattern: "["`) now raises `ValueError` when loaded, where it
  previously loaded and failed on the first extraction turn. `<=` and `>=` now agree
  with `==` on numeric strings.
- **`extract_json_from_text` returns `dict | None`.** It is exported in
  `fsm_llm.__all__`; valid non-object JSON (`42`, `[1,2]`, `true`, `"hi"`) now returns
  `None` where it returned the raw value (iteration 1, D-010).
- **`_build_response_format_section` takes a positional parameter.** The method on
  `ResponseGenerationPromptBuilder` gained `plain_text` (iteration 1, D-003). A
  subclass that overrides it with the old zero-argument signature now raises
  `TypeError` when the base calls it.
- **LV-01 re-extraction cost.** A required field that stays unset (null,
  zero-confidence or a rejected container value) is re-asked every turn, and
  `extraction_retries` makes each such field cost up to `1 + retries` extra provider
  round-trips per turn (wording corrected in iteration 3, D-027). One
  agent test went from 34 to 152 provider calls. Retries against Ollama are
  byte-identical (temperature is forced to 0).
- **React and reflexion extraction contract (D-024).** The think state extracts
  `tool_name` and `tool_input` through explicit typed configs; extraction order
  changed (`should_terminate` now precedes them). `tool_input` can still come back as
  an empty `{}`; the tool layer fills required parameters from the task text.
- **Structured terminal reply and greeting retry (D-020).** A terminal state with an
  output schema returns the schema JSON as the reply; an apology-producing Pass-2
  reply is retried once, so one turn can make at most one extra LLM call, and only
  on a turn that would otherwise have shown the apology.
- **Stacked `save_session` and zero-handler turns (D-022).** `save_session` on a
  stacked conversation saves the root frame (a session saved while stacked used to
  record the sub-FSM state). Turns with no handler for a timing skip the handler
  deep-copies.

- **`handlers_at` is an optional hook (D-029).** `execute_handlers` reads it with a
  guarded `getattr`; a duck-typed `handler_system` without it works and keeps the
  handler deep-copies (it just does not get the zero-handler shortcut).
- **`extract_json_from_text` fence handling (D-029, D-030).** A fenced non-object is
  skipped by blanking its span in place (length-preserving), not by truncating the
  text before it; `strip_think_and_fences` strips a fence only at the start of the
  reply. Six iteration-1 corpus rows in `test_audit_iter1_seam.py` were rewritten to
  the D-030 semantics.
- **Uncoercible confidence keeps the value (D-028).** A per-field `confidence` that
  cannot be coerced now means "value kept at 0.5", not "failed extraction".
- **Brace-shaped prose is no longer replaced (D-030).** The plain-text rung decides by
  parseability, not text shape. Trade-off: a malformed brace envelope such as
  `{"message": hi}` (no recoverable JSON) is now shown as text instead of the generic
  apology. `<think>` in STREAMED replies still passes through raw (buffering would
  break time-to-first-token).
- **Provenance is persisted (D-031).** Supersedes the D-015 clause "not persisted":
  the digest map is stored in `SessionState.metadata["pipeline_extracted"]`, exposed as
  `_pipeline_extracted` in `FSMManager.get_complete_conversation()['metadata']` and
  written to the session file (digests, not values; same trust domain as
  `context_data`). A session file without the key restores an empty map and fails
  closed.
- **`handler_only_keys` is a new `FSMDefinition` field (D-033).** `model_dump()` emits
  `handler_only_keys: []` for every FSM, so a snapshot that compares a dumped
  definition exactly must expect the new key.
- **Rejected corrections reach Pass 2 (D-032).** `DataExtractionResponse` gained
  `rejected_corrections` (default `{}`) and `build_response_prompt` a last optional
  argument; a turn with no rejection is byte-identical. A caller that subclasses or
  wraps `build_response_prompt` positionally is unaffected.
- **Retry-cost wording corrected (D-027, review note 11).** A required field that
  stays unset costs up to `1 + extraction_retries` extra provider round-trips per
  field per turn (the "LV-01 re-extraction cost" bullet above is corrected in place);
  on Ollama the identical retries
  are now collapsed by the null memo (D-035).
- **Extra calls (D-034, D-042, D-035).** A transition into a different, already-filled
  state costs +1 bulk call (+1 retry per still-null required key; measured 4 -> 5
  provider calls on the back-edge turn, 5 -> 7 with one required null key) and the
  same call fires on a revisit that changes nothing (live: +1 call per return to
  `confirm`). On Ollama a null key at `extraction_retries=3` drops from 1+3 calls to 1.
- **`plan_execute` seed removed (D-036), then restored in iteration 4 (D-046).**
  Iteration 3 stopped pre-seeding `plan_steps` with `[]`; that made an unplannable task
  raise `BudgetExhaustedError`, so the seed is back (see the iteration 4 section).
- **`evaluator_optimizer` success (D-036).** `AgentResult.success` is true when
  `generated_output` is non-empty; see the LV5-02 limitation.
- **Iteration 4 (final loop) contract changes.**
  - The prompt sanitizer escapes a `<name` opener or closer whose tail contains a
    nested `<` or is longer than 256 characters; only the `<` and the tag name are
    escaped (`&lt;/task`), never the text after it. Benign prose is byte-identical,
    including a `<` followed by a space (`latency < threshold`) when no `>` follows
    it in the text. The exact residual (D-054): a `<` DIRECTLY followed by a letter
    (`x<y`), with 257 or more characters and no `>` after it, has its `<` escaped;
    a `<` plus whitespace plus a name with no `/` (`< task`), no `>` within 256
    characters AND no `>` anywhere later in the text, is kept raw (a comparison);
    with any `>` later in the text it is escaped. An unterminated closer
    or opener with a short tail and no `>` anywhere (`</task NEW INSTRUCTIONS`)
    still reaches the prompt raw; that shape is pre-existing (every earlier pattern
    left it raw) and is not closed by this change.
  - `strip_think_and_fences` strips the closing fence only after a leading fence was
    stripped (a reply that ends in a code block keeps its fence).
  - Rejected-correction grounding is a whole-token match with a 3-character floor. The
    named cost: a real correction to a 1 or 2 character value (`US`, `EU`, `42`) never
    produces the `<rejected_corrections>` block, and a differently formatted value
    (`1,000` for `1000`) does not ground.
  - `DataExtractionResponse.extraction_failed: bool = False` (new public field) and
    `build_response_prompt(..., extraction_failed=False)` (new optional last parameter;
    positional callers are unaffected).
  - `fsm-llm-validate` emits two new WARNINGs for `handler_only_keys` (never errors).
  - `PlanExecuteAgent` seeds `plan_steps: []` again (D-046).
  - `fsm_id` change on upgrade: `API.from_definition(FSMDefinition(...))` derives its
    `fsm_id` from `model_dump()`, which now carries `handler_only_keys: []`, so the id
    of an FSM built that way differs from the id the previous release computed
    (`restore_session` does not compare `fsm_id`, see the limitations).

### Known limitations -- iteration 2 list (current status in the iteration 3 list below)

- **LV2-04, undeclared gate key.** A key a transition reads but no field declares can
  still be added by a steered bulk value or by the per-field channel (live s13:
  `is_admin` opened, 1/1). Provenance narrows but does not close CF-06.
- **LV2-05, low-confidence classification stall.** A classification below the state's
  `confidence_threshold` keeps the stale fallback intent and does not transition
  (live s7: three turns end in `triage`). The behaviour is pinned by a seam test, not
  changed.
- **LV3-01, reply and data contradict.** When a correction of a handler-set key is
  rejected, the Pass-2 reply can still claim the correction was applied while the
  stored data is unchanged (live s14, 2/2 looks). The data is safe; the reply is not.
- **`tool_input` quality after D-024.** The tool now runs live, but `tool_input` is
  often `{}` and parameter quality is not fixed. Reflexion is not measured live.
- **Prompt-content changes are not measured by `scripts/eval.py`.** The sanitizer
  (LS-01), the secret-name filter (LS-06), the `extracted_data` filter (LS-02) and the
  bulk-prompt sanitizer (CF-06) change prompt content only for non-benign input;
  benign prompts are byte-identical (hash test). The 95.3% eval baseline stays stale
  and must be re-run before it is trusted.
- **Live evidence is n=1 to n=3 on one model** (`ollama_chat/qwen3.5:9b-q8_0`, 106
  calls). The LV2-02 retry was not exercised live (0 empty greetings in 18 starts);
  it is proven offline only. Non-Ollama providers and non-`any` union types are not
  measured.
- **Deferred to iteration 3:** the efficiency batch (identical retries, filter
  caches), LV2-03 (form back-edge correction), provenance persistence across
  `restore_session`, and LV3-02/LV3-03 (`<information_still_needed>` listing a filled
  key; `agent_trace` visible in per-field extraction prompts).

### Known limitations -- after iteration 4 (final loop; supersedes the iteration 3 list where it differs)

Live evidence: `ollama_chat/qwen3.5:9b-q8_0` only, 1 to 4 looks per scenario, 173 calls
against a target of about 150 and a hard stop of 180, box idle-checked before the run.
Every rate is a look count, not a significance claim. Resolved since the iteration 3
list: LV5-01 (D-050) and LV5-03 (D-052) are fixed offline; the sanitizer bypass, the
`plan_execute` exception and LV6-01 are fixed (see the regression list above).

- **The `scripts/eval.py` baseline (95.3%, N=3 median, Run 006) was NOT re-measured
  against four iterations of prompt-content changes and is STALE. This is a release
  gate.** The prompt-content changes since that baseline are: the tag sanitizer
  (LS-01, restored in D-047), the bulk-prompt sanitising (CF-06), the `extracted_data`
  filter (LS-02, nested-dict security filtering), the plain-text stream prompt, and
  the `<rejected_corrections>` block and the extraction-failed line (each only on a turn
  with a rejection or a failure). Benign prompts are byte-identical (hash tests), but no
  gate available in this run observes prompt effects at eval scale (the fast gates mock
  the LLM). Re-run the eval suite before trusting the baseline.
- **The live audit found LV6-01 that the check booleans missed.** All three back-edge
  looks scored 5/5 while the raw Pass-2 prompt carried a wrong `<rejected_corrections>`
  block. Read raw `calls[]`, not `summary.checks`.
- **`EvaluatorOptimizerAgent` `success` after the refinement cap (LV5-02, concern 6,
  D-052).** `success=True` even though the evaluator never passed the output;
  `max_iterations_reached` in the final context is the only signal. It needs an owner
  decision on a public contract and was not changed in a final loop.
- **`<extracted_data>` is not scoped by `context_scope.read_keys` (D-054, final
  review concern 3).** `read_keys` scopes `<current_context>` and
  `<rejected_corrections>` only. `<extracted_data>` (the keys extracted this turn from
  the user's own message) is shown unscoped, so a key a state hides from
  `<current_context>` still reaches Pass 2 on the turn it is extracted. It is
  pre-existing since D-005 and was deliberately not changed in a final loop.
- **RB-04 declined (D-041).** A normalising CONTEXT_UPDATE handler on an extracted key
  still disables later LLM corrections of that key.
- **`restore_session` does not compare `fsm_id`** (concern 11): a session file can be
  restored onto a different FSM definition without error.
- **RB-08 fires on any transition into a filled state (concern 9)**, not only on back
  edges: +1 bulk call per return to a filled state (live: `confirm`), even when nothing
  changes. The name and cost are accepted for now.
- **`handler_only_keys` is opt-in.** LV2-04 stays the default for FSMs that do not
  declare it; only listed keys are covered and a classification-owned key is not (the
  validator now warns about that overlap). The bulk-output filter on a listed key was
  never exercised live (the model did not return the key in a bulk pass).
- **The grounding floor is 3 characters (D-049).** A real correction to a value of 1 or
  2 characters never produces the block, so the reply may still claim the change was
  made; the false-positive rate on real forms is unmeasured. The intended live
  true-positive of the instruction-only report (D-052) was not observed: the s14 value
  `US` is under the floor, so live s14 does not exercise it.
- **Not observed live:** the D-050 extraction-failed line (no bulk call errored while
  Pass 2 could still run), the D-046 unplannable-task branch (the live task was
  plannable), the RB-08 cost on real forms, non-Ollama providers.
- **plan_execute plans but does not execute (LV5-07) is unchanged:** the plan was
  extracted 4/4 live runs and the model narrated instead of calling the tool 4/4;
  `success` was never True, and the honest `completed with no execution evidence`
  WARNING was logged. Three of the four runs were truncated by the harness call cap.
- **LV6-02 / LV6-03 (low, info):** the bulk pass can store a model-invented key
  (`intent: jailbreak_attempt`); the low-confidence classification WARNING count depends
  on the classifier, so "exactly two WARNINGs" is not a property of the code.
- **The unchanged items of the iteration 3 list stay open:** LV3-01 (reply/data
  contradiction, measured 0/10 cumulative on s14, not significant), LV5-04, LV2-05,
  LV2-10/LV4-08, LV3-02/LV3-03, rewoo/maker_checker/debate at base, LV5-05, LV5-08,
  `tool_input` quality, and the accepted items of D-027 and D-052 (the `0.5` unscored
  confidence literal duplicated in `llm.py`, the 303-line `_execute_data_extraction`).

### Known limitations -- after iteration 3

Live evidence: `ollama_chat/qwen3.5:9b-q8_0` only, 1 to 3 looks per scenario at
temperature 0.3 to 0.5, about 282 calls against a 250-call budget; 15 of 32 records
ran at 10 to 20 times normal latency from unrelated load on the shared Ollama (the
scripts' own 280/300 s alarms fired inside in-flight calls, not provider timeouts).
No rate carries a significance claim.

- **`handler_only_keys` is opt-in (LV2-04 stays the default).** Without the list a
  key a transition reads can still be set by a steered value (live control arm:
  `is_admin` opened). With it, the per-field closure held 2/2 looks; the bulk-output
  filter was not exercised live (the model never returned `is_admin` in the bulk
  pass). Only listed keys are covered: an unlisted gate key stays writable and a
  classification-owned key is not covered. Say "closable per FSM for the listed
  keys", not "LV2-04 closed".
- **A normalising CONTEXT_UPDATE handler disables corrections on that key (D-041).**
  Provenance records the pre-handler value, so a same-timing handler edit of an
  extracted key blocks later LLM overwrites of it; the reply is told the change was
  not applied.
- **LV5-01, a failed back-edge re-extraction is silent to Pass 2 (FIXED in iteration 4,
  D-050; the text below is the iteration 3 state).** If the RB-08
  re-extraction call errors, `_bulk_extract_from_instructions` returns `{}` (a failed
  bulk is indistinguishable from "nothing to extract"), the store keeps the old value
  and the reply can still say it was updated (live: store `Jane Doe`, reply "I've
  updated your name to Janet Doe"). Live 2 of 3 looks landed the correction; the
  third failed on a script alarm and its re-run crashed. Same class as LV3-01 on a
  different path.
- **LV3-01, reply and data contradict: measured 0/8 after the rejected-corrections
  block (was 2/4).** Not significant at this n (Wilson upper bound about 32%); only 4
  of 8 replies say the value was not changed, the rest merely restate it.
- **LV5-03, `<rejected_corrections>` lists config-covered keys only (FIXED in
  iteration 4, D-052; the text below is the iteration 3 state).** An
  instruction-only key that already holds a value (`region` in the live fixture) is
  refused by skip-if-set and never reported, so the reply is not told (0/8 replies
  happened to claim it).
- **LV5-04, a bare "Yes" after the back edge does not flip `user_confirmation`** while
  the reply says confirmed: a stale set key is only correctable by the bulk path,
  which returns `{}` for a bare affirmation. Pre-existing shape of D-015.
- **LV2-05, low-confidence classification stall** is unchanged (three turns end in
  `triage`); it is now a visible WARNING (live: one line per stalled turn).
- **LV2-10 / LV4-08, history summary is never injected** into the prompt (the history
  cap is not measured against `scripts/eval.py`); CF-08 turn atomicity and LS-10 are
  deferred with it. **LV3-02 / LV3-03**: `<information_still_needed>` can list a
  filled key and `agent_trace` can be visible in per-field extraction prompts.
- **rewoo, maker_checker and debate fail at base too** (LV4-02/03/05): the extractor
  is fed the literal message "Continue." and the model reports it does not contain the
  plan. The typed-config recipe of D-024 does not transfer; a fix needs the plan
  produced in a call whose input is the task. Not run in the iteration-3 live block.
- **plan_execute plans but does not execute (LV5-07, carried forward from LV4-01).** The plan is extracted (6/6 live
  runs, non-empty `plan_steps`), but the model narrates "I've completed the search"
  and never calls the search tool (no tool output in any prompt, no tool-execution
  log), so `success` is False. Final `success` is UNMEASURED in this block: 6/6 runs
  ended on the 250 s agent timeout or the script alarm at 25 to 60 s per call.
- **LV5-02, `EvaluatorOptimizerAgent` `success=True` after the refinement cap even
  though the evaluator never passed the output.** `evaluator_optimizer.py:207-208`
  forces `evaluation_passed=True` at the cap (pre-existing design); since
  `generated_output` became an answer key (LV4-04) that reads as `success=True`.
  `success` means "produced a non-empty output"; `max_iterations_reached` in the
  context is the only signal. Live run 1: evaluator `passed False`, `refinement_count`
  2, `success True`. Needs an owner decision.
- **LV5-05, `evaluation_result` (a dict set by a handler) appears in the evaluator
  agent's Pass-2 `<current_context>`.** Handler-set, not extracted, so the no-dict-in-
  extraction rule is not violated; it is a dict-valued key in a prompt block.
- **LV5-08, literal backslash-n in extracted multi-line values.** Field
  extraction of a multi-line value can store a literal backslash-n (live n=1: the
  evaluator counted one line and the loop spent both refinements; not reproduced in the
  other two runs). The parse does not unescape.
- **`tool_input` quality (D-024).** The tool runs live (3/3, no crash), but `tool_input`
  is `{}` 3/3 and the tool receives the whole task sentence as `query`; parameter
  quality is not fixed. Reflexion is not measured live.
- **Accepted, not fixed (D-027).** `<=`/`>=` agree with numeric strings while `==` does
  not (review note 12); a classification-owned key above threshold blocks correction
  (the classifier is the owner, note 9); bulk values validate at a fixed confidence 1.0
  (note 8); a truncated or malformed brace envelope shows as text (D-030). Also
  deferred: object-typed `output_schema` fields, reasoning-mode mapping keys, GD-05
  code half (`find_dotenv(usecwd=True)`, default model), GD-16 (CLI output via the
  logger), GD-13..GD-25 beyond the doc items fixed in step 20.
- **Prompt-content changes are not measured by `scripts/eval.py`, and its 95.3%
  baseline is stale.** The sanitizer (LS-01), the secret-name filter (LS-06), the
  `extracted_data` filter (LS-02), the bulk-prompt sanitizer (CF-06), the
  `<rejected_corrections>` block (only for turns with a rejection), the plain-text
  stream prompt and the RB-06 tag pattern change prompt content; benign prompts are
  byte-identical (hash tests). Re-run the eval suite before trusting the baseline.
- **Not measured live.** RB-01 non-numeric confidence, RB-03 `<think>` prefix, RB-10
  fences, RB-05 restore-then-correct and RB-06 had no live exposure (0 of 247 raw calls
  carried `<think>`, a fence or a non-numeric confidence). The memo (RB-09) fired twice,
  both null-to-null, so a stale null masking a non-null resample is not observed, not
  excluded. Non-Ollama providers, non-`any`/`str` union types and the false-positive
  rate of the rejected-correction grounding test at scale are not measured.

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

[Unreleased]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NikolasMarkou/fsm_llm/releases/tag/v0.3.0
