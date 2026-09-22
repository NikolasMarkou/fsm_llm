# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[0.8.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NikolasMarkou/fsm_llm/releases/tag/v0.1.0
