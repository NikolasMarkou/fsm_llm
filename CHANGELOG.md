# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  (`"<a" * 10000` and `"<" + "a" * 100000` each under 0.5 s). A differential test pins
  the new pattern against the pre-D-029 pattern on every string of length up to 6.
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
    including a `<` followed by a space (`latency < threshold`). The exact residual:
    a `<` DIRECTLY followed by a letter (`x<y`), with 257 or more characters and no
    `>` after it, has its `<` escaped; and a `<` plus a space plus a name with no `/`
    (`< task`) and no `>` within 256 characters is kept raw. An unterminated closer
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

[0.5.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NikolasMarkou/fsm_llm/releases/tag/v0.1.0
