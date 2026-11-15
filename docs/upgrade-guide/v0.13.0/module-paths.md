# Module paths

Covers FSM-LLM 0.12.0 to 0.13.0. The flat `src/fsm_llm/` modules of 0.12.0 became layer packages (`fsm_llm.core`, `fsm_llm.engine`, `fsm_llm.llm`, `fsm_llm.context`, `fsm_llm.handlers`, `fsm_llm.security`, `fsm_llm.tools`, `fsm_llm.policy`) and the biggest modules were split into packages. Most of this is invisible to a caller of the public API: the names that `fsm_llm` and its subpackages (`fsm_llm.agents`, `.workflows`, `.reasoning`, `.monitor`, `.eval`, `.synthesis`) export from their package root moved nowhere (`from fsm_llm import API, FSMManager, Classifier` is unchanged; renames and removals of exported names are in the other guides). What breaks is a deep import of a module path that no longer exists, a deep import of a name that left its module, a patch that names the old module, and a log filter on the old loguru record name. A moved path has no alias and no shim, so the old import fails at once with an `ImportError`.

Each section below is one module of 0.12.0, named by its old dotted path, with where it went, what else changed for a user, and a before and after import. A module that became a package at the same dotted path keeps working for every public name; only its private names and patch targets moved. The complete machine-readable record, symbol by symbol, is `tests/fixtures/module_path_map.json`, and a test keeps this page in step with it: every old module of 0.12.0 that the map lists has a section here.

## Rewrite the imports with a script

Most of the old modules moved by a plain prefix rule, listed in the table of the script. The script rewrites them in every `.py` file under the directory you pass, in place. It touches only the dotted names in the table, so run it on a clean version control state and read the diff. The modules that were split name by name (`fsm_llm.utilities`, the `fsm_llm.agents.*` tool and policy modules, the monitor routes) need the tables in their sections below: after the script, run your test suite and fix what still fails with an `ImportError`.

**Sketch**

```python
import pathlib
import re
import sys

RENAMES = {
    "fsm_llm.agents.truncation": "fsm_llm.tools.truncation",
    "fsm_llm.api": "fsm_llm.engine.api",
    "fsm_llm.builders": "fsm_llm.engine.builders",
    "fsm_llm.classification": "fsm_llm.engine.classification",
    "fsm_llm.definitions": "fsm_llm.core.definitions",
    "fsm_llm.expressions": "fsm_llm.core.expressions",
    "fsm_llm.fsm": "fsm_llm.engine.fsm",
    "fsm_llm.ollama": "fsm_llm.llm.ollama",
    "fsm_llm.pipeline": "fsm_llm.engine.turn",
    "fsm_llm.prompts": "fsm_llm.context.layout",
    "fsm_llm.session": "fsm_llm.engine.session",
    "fsm_llm.transition_evaluator": "fsm_llm.core.transition_evaluator",
    "fsm_llm.validator": "fsm_llm.core.verify",
    "fsm_llm.visualizer": "fsm_llm.core.visualize",
}
pattern = re.compile(r"(?<![\w.])(" + "|".join(re.escape(old) for old in RENAMES) + r")\b")

for path in pathlib.Path(sys.argv[1]).rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    new = pattern.sub(lambda match: RENAMES[match.group(1)], text)
    if new != text:
        path.write_text(new, encoding="utf-8")
        print("rewrote", path)
```

Save it as `retarget.py` and run `python retarget.py src tests`. The names that moved by a prefix rule are also the ones whose public names the new package re-exports, so a line such as `from fsm_llm.definitions import FSMDefinition` becomes `from fsm_llm.core.definitions import FSMDefinition` and resolves. A few names the old module only imported, and every private name, still need a manual edit (see the section of the module).

## fsm_llm.agents.base

This module became a package at the same dotted path, split across 2 modules. Every one of its 8 public names still imports from `fsm_llm.agents.base`, so a caller of the public API changes nothing. The 7 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.agents.base.agent
    public: BaseAgent, accepts_tools
    private: _H
fsm_llm.agents.base.helpers
    public: artifact_text, caller_prompt_keys, flagged_tool_names,
        pattern_run_output_keys, reject_misplaced_kwargs, strip_caller_context
    private: _HITL_CTOR_KWARGS, _INSTRUCTION_SLOTS, _flagged_tool_policy,
        _output_response_format, _prompt_overflow_error, _with_instructions
```

**Before**

```python
from fsm_llm.agents.base import _flagged_tool_policy
```

**After**

```python
from fsm_llm.agents.base.helpers import _flagged_tool_policy
```

## fsm_llm.agents.constants

This module became a package at the same dotted path, split across 7 modules. Every one of its 53 public names still imports from `fsm_llm.agents.constants`, so a caller of the public API changes nothing. It has no private names in the table, but a patch aims at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.agents.constants.core
    public: ContextKeys, Defaults, ErrorMessages, LogMessages
fsm_llm.agents.constants.handler_names
    public: HandlerNames, HandlerPriorities
fsm_llm.agents.constants.keys
    public: CONFIG_OWNED_KWARGS, FRAMEWORK_ONLY_KEYS, MISPLACED_AGENT_KWARGS,
        REACT_THINK_FRESH_KEYS, RESULT_DROPPED_CONTEXT_KEYS, RUN_OUTPUT_KEYS, and 1
        more
fsm_llm.agents.constants.meta
    public: META_AGENT_PATTERN_INTENTS, META_ARTIFACT_TYPE_INTENTS,
        META_BUILD_CALL_FAILED, META_BUILD_NEGATIONS, META_BUILD_NEGATION_FILLERS,
        META_BUILD_OUTPUT_KEYS, and 14 more
fsm_llm.agents.constants.outcomes
    public: StopReason
fsm_llm.agents.constants.states
    public: ADaPTStates, AgentStates, DebateStates, EvalOptStates,
        MakerCheckerStates, NATIVE_FC_HANDLER_ONLY_KEYS, and 10 more
fsm_llm.tools.constants
    public: OBSERVATION_ENTRY, ToolObservationPrefix, ToolRunStatus
```

**After**

```python
from fsm_llm.agents.constants import ADaPTStates
from fsm_llm.agents.constants.states import ADaPTStates as defined

assert defined is ADaPTStates
```

## fsm_llm.agents.definitions

`fsm_llm.agents.definitions` stays (`AgentConfig`, `AgentResult` and the other agent models), but `ToolDefinition`, `ToolCall`, `ToolResult`, `ToolAnnotations` and `READ_ONLY_ANNOTATIONS` are defined in `fsm_llm.tools.models` and `ApprovalRequest` in `fsm_llm.policy.approval`. See [tools and approval policy](tools-and-policy.md#the-tool-models-moved-to-fsm_llmtools).

```text
fsm_llm.policy.approval
    public: ApprovalRequest
fsm_llm.tools.models
    public: READ_ONLY_ANNOTATIONS, ToolAnnotations, ToolCall, ToolDefinition,
        ToolResult
```

**Before**

```python
from fsm_llm.agents.definitions import ToolCall
```

**After**

```python
from fsm_llm.tools import ToolCall
```

## fsm_llm.agents.exceptions

`fsm_llm.agents.exceptions` stays, but `ToolExecutionError` and `ToolNotFoundError` are defined in `fsm_llm.tools.errors`, `ApprovalDeniedError` in `fsm_llm.policy.approval` and `AgentError` in `fsm_llm.core.definitions.errors` (only its `__module__` changes). `bounded_repr` and `require_int_limit` live in `fsm_llm.tools`. See [tools and approval policy](tools-and-policy.md#the-tool-errors-moved-to-fsm_llmtools-and-agenterror-to-core).

```text
fsm_llm.core.definitions.errors
    public: AgentError
fsm_llm.policy.approval
    public: ApprovalDeniedError
fsm_llm.tools.errors
    public: ToolExecutionError, ToolNotFoundError
fsm_llm.tools.limits
    public: bounded_repr, require_int_limit
```

**Before**

```python
from fsm_llm.agents.exceptions import ToolExecutionError
```

**After**

```python
from fsm_llm.tools import ToolExecutionError
```

## fsm_llm.agents.fsm_definitions

This module became a package at the same dotted path, split across 16 modules. Every one of its 16 public names still imports from `fsm_llm.agents.fsm_definitions`, so a caller of the public API changes nothing. The 21 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.agents.fsm_definitions._common
    private: _MAX_DESC, _by_id, _cond, _edge, _finalize_fsm, _flag_condition, and 3
        more
fsm_llm.agents.fsm_definitions._loop
    public: conclude_on_evidence_logic, loop_field_context_keys
    private: _LOOP_FIELD_CONTEXT_KEYS, _TERMINATE_ON_EVIDENCE, _act_state,
        _await_approval_state, _conclude_state, _field, and 4 more
fsm_llm.agents.fsm_definitions.adapt
    public: build_adapt_fsm
fsm_llm.agents.fsm_definitions.debate
    public: build_debate_fsm
fsm_llm.agents.fsm_definitions.evalopt
    public: build_evalopt_fsm
fsm_llm.agents.fsm_definitions.maker_checker
    public: build_maker_checker_fsm
fsm_llm.agents.fsm_definitions.meta_builder
    public: build_meta_builder_fsm
fsm_llm.agents.fsm_definitions.native_fc
    public: build_native_fc_fsm
    private: _reply_kind_condition
fsm_llm.agents.fsm_definitions.orchestrator
    public: build_orchestrator_fsm
fsm_llm.agents.fsm_definitions.parallel_react
    public: build_parallel_react_fsm
    private: _build_parallel_think_instructions
fsm_llm.agents.fsm_definitions.plan_execute
    public: build_plan_execute_fsm
fsm_llm.agents.fsm_definitions.prompt_chain
    public: build_prompt_chain_fsm
fsm_llm.agents.fsm_definitions.react
    public: build_react_fsm
fsm_llm.agents.fsm_definitions.reflexion
    public: build_reflexion_fsm
fsm_llm.agents.fsm_definitions.rewoo
    public: build_rewoo_fsm
fsm_llm.agents.fsm_definitions.self_consistency
    public: build_self_consistency_fsm
```

**Before**

```python
from fsm_llm.agents.fsm_definitions import _act_state
```

**After**

```python
from fsm_llm.agents.fsm_definitions._loop import _act_state
```

## fsm_llm.agents.handlers

This module became a package at the same dotted path, split across 7 modules. Every one of its 19 public names still imports from `fsm_llm.agents.handlers`, so a caller of the public API changes nothing. The 1 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.agents.handlers.approval
    public: approval_grant, call_label, call_ran, refusal_record
fsm_llm.agents.handlers.executor
    public: AgentHandlers
fsm_llm.agents.handlers.forced
    public: concluded_on_evidence, forced_stop_skip, is_forced_verdict,
        recorded_forced_reason
    private: _CONCLUDE_VERDICTS
fsm_llm.agents.handlers.freshness
    public: make_fresh_keys_handler, make_redraft_handlers
fsm_llm.agents.handlers.limiters
    public: ThinkTurnLimiter, make_iteration_limiter
fsm_llm.agents.handlers.recording
    public: next_step_number, record_tool_step, with_feedback
fsm_llm.agents.handlers.run_guards
    public: RUN_ENDING_ERRORS, RunEndingErrorHolder, run_evaluation_fn
```

**Before**

```python
from fsm_llm.agents.handlers import _CONCLUDE_VERDICTS
```

**After**

```python
from fsm_llm.agents.handlers.forced import _CONCLUDE_VERDICTS
```

## fsm_llm.agents.hitl

`fsm_llm.agents.hitl` no longer exists. `HumanInTheLoop`, `ApprovalPolicy`, `ApprovalCallback` and `EscalationCallback` are defined in `fsm_llm.policy`; `make_hitl_checker` is in `fsm_llm.agents.tools`. See [tools and approval policy](tools-and-policy.md#humanintheloop-moved-to-fsm_llmpolicy).

```text
fsm_llm.agents.tools
    public: make_hitl_checker
fsm_llm.policy.approval
    public: ApprovalCallback, ApprovalPolicy
fsm_llm.policy.hitl
    public: EscalationCallback, HumanInTheLoop
```

**Before**

```python
from fsm_llm.agents.hitl import HumanInTheLoop
```

**After**

```python
from fsm_llm.policy import HumanInTheLoop
```

## fsm_llm.agents.meta_builder

This module became a package at the same dotted path, split across 5 modules. Every one of its 1 public names still imports from `fsm_llm.agents.meta_builder`, so a caller of the public API changes nothing. The 26 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.agents.meta_builder.agent
    public: MetaBuilderAgent
fsm_llm.agents.meta_builder.hints
    private: _APOSTROPHES, _BUILD_PHRASE_PATTERN, _CLAUSE_BREAK,
        _GOVERNING_NEGATION, _SWITCH_PATTERN, _TRAILING_PUNCTUATION, and 4 more
fsm_llm.agents.meta_builder.names
    private: _TYPE_VALUES
fsm_llm.agents.meta_builder.session
    private: _BUILD_PROMPT_CORE, _QUOTES, _prompt_text, _with_build_prompt
fsm_llm.agents.meta_builder.specs
    private: _AgentSpec, _FSMSpec, _FSMStateSpec, _FSMTransitionSpec, _ReplySpec,
        _SPEC_MODELS, and 5 more
```

**Before**

```python
from fsm_llm.agents.meta_builder import _given
```

**After**

```python
from fsm_llm.agents.meta_builder.specs import _given
```

## fsm_llm.agents.meta_builders

This module became a package at the same dotted path, split across 4 modules. Every one of its 4 public names still imports from `fsm_llm.agents.meta_builders`, so a caller of the public API changes nothing. It has no private names in the table, but a patch aims at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.agents.meta_builders.agent
    public: AgentArtifactBuilder
fsm_llm.agents.meta_builders.base
    public: ArtifactBuilder
fsm_llm.agents.meta_builders.fsm
    public: FSMArtifactBuilder
fsm_llm.agents.meta_builders.workflow
    public: WorkflowArtifactBuilder
```

**After**

```python
from fsm_llm.agents.meta_builders import AgentArtifactBuilder
from fsm_llm.agents.meta_builders.agent import AgentArtifactBuilder as defined

assert defined is AgentArtifactBuilder
```

## fsm_llm.agents.tools

`fsm_llm.agents.tools` stays, but the tool registry left it: `ToolRegistry`, `tool`, `redact_secret_entries`, `tool_parameters_schema` and `schema_types` are defined in `fsm_llm.tools`. `register_agent`, `normalize_tool_input`, `refuse_execute_without_gated` and `tool_call_from_context` stay. See [tools and approval policy](tools-and-policy.md#toolregistry-and-the-tool-helpers-moved-to-fsm_llmtools).

```text
fsm_llm.tools.arguments
    public: redact_secret_entries
fsm_llm.tools.decorator
    public: tool
fsm_llm.tools.hints
    public: callable_name
fsm_llm.tools.registry
    public: ToolRegistry, registry_from
fsm_llm.tools.schema
    public: infer_schema_from_hints, schema_types, tool_parameters_schema
fsm_llm.tools.timeouts
    public: ToolTimedOut, call_with_timeout, check_timeout
```

**Before**

```python
from fsm_llm.agents.tools import ToolRegistry
```

**After**

```python
from fsm_llm.tools import ToolRegistry
```

## fsm_llm.agents.truncation

`fsm_llm.agents.truncation` moved whole to `fsm_llm.tools.truncation`; `smart_truncate` is exported by the package `fsm_llm.tools`. See [tools and approval policy](tools-and-policy.md#smart_truncate-moved-to-fsm_llmtools).

**Before**

```python
from fsm_llm.agents.truncation import smart_truncate
```

**After**

```python
from fsm_llm.tools.truncation import smart_truncate
```

## fsm_llm.api

The module `fsm_llm.api` is the package `fsm_llm.engine.api`. `API` is defined in `fsm_llm.engine.api.conversation` and the package re-exports it with five more names (`FSMStackFrame`, `ContextMergeStrategy`, `handle_conversation_errors`, `llm_settings_for`, `MAX_ENDED_CONVERSATIONS_CACHE`). `from fsm_llm import API` is unchanged.

A private helper comes from the module that defines it (`_unexpected_error` and `_read_or_ended` from `fsm_llm.engine.api.helpers`, `_turn_errors` from `fsm_llm.engine.api.conversation`, `_check_run_budgets` and `_run_rounds` from `fsm_llm.engine.api.run`). Names the old module only imported (`FSMManager`, `LiteLLMInterface`, `time`, `logger`) were never its API: aim a patch at the module that performs the lookup, for example `fsm_llm.engine.api.conversation.LiteLLMInterface` (was `fsm_llm.api.LiteLLMInterface`). The loguru record name reads `fsm_llm.engine.api.<module>` instead of `fsm_llm.api`.

**Before**

```python
from fsm_llm.api import API
```

**After**

```python
from fsm_llm.engine.api import API
```

## fsm_llm.builders

`fsm_llm.builders` moved whole to `fsm_llm.engine.builders`, with the private `refuse_named_options` that `fsm_llm.agents.builders` imports. `from fsm_llm import APIBuilder, FSMManagerBuilder` is unchanged. The loguru record name reads `fsm_llm.engine.builders`.

**Before**

```python
from fsm_llm.builders import APIBuilder
```

**After**

```python
from fsm_llm.engine.builders import APIBuilder
```

## fsm_llm.classification

`fsm_llm.classification` moved whole to `fsm_llm.engine.classification` (`Classifier`, `HierarchicalClassifier`, `IntentRouter`, `HandlerFn`). `from fsm_llm import Classifier, HierarchicalClassifier, IntentRouter, HandlerFn` is unchanged. Names the old module only imported (`Callable`, `time`, `logger`, `LiteLLMInterface`) were never its API: aim a patch at `fsm_llm.engine.classification.<name>`. The loguru record name reads `fsm_llm.engine.classification`.

**Before**

```python
from fsm_llm.classification import Classifier
```

**After**

```python
from fsm_llm.engine.classification import Classifier
```

## fsm_llm.context

`fsm_llm.context` is a package at the same dotted path (it was `context.py`): its code is in `fsm_llm.context.compactor` (`ContextCompactor`, `clean_context_keys`) and `fsm_llm.context.commit` (`commit_with_key_rollback`, `was_cut_off`, `record_cut_off`), and `clear_keys_delta` moved down from `fsm_llm.handlers` to `fsm_llm.context.commit` so the `context` layer imports no engine module. Every one of those names still imports from `fsm_llm.context`, and `from fsm_llm.handlers import clear_keys_delta` and `from fsm_llm import clear_keys_on_entry, ContextCompactor` are unchanged (`handlers` binds the moved function).

What breaks: names `fsm_llm.context` only imported are no longer its attributes (`MAX_CONTEXT_FILTER_DEPTH` comes from `fsm_llm.constants`, `logger` from `fsm_llm.logging`), and a patch of `fsm_llm.context.logger` must aim at the module that logs, `fsm_llm.context.compactor.logger`. `fsm_llm.handlers.clear_keys_delta` is the same function object as `fsm_llm.context.commit.clear_keys_delta`, so a patch of the old name changes only the `handlers` binding, not `ContextCompactor` or the agents.

**Before**

```python
from unittest.mock import patch

with patch("fsm_llm.context.logger"):
    ...
```

**After**

```python
from unittest.mock import patch

with patch("fsm_llm.context.compactor.logger"):
    ...
```

## fsm_llm.definitions

The module `fsm_llm.definitions` is the package `fsm_llm.core.definitions`. It re-exports every public name of the old module (74 names: the models, the exceptions, the constants and the helpers) with a static `__all__`, so the line to change is the module path alone. `from fsm_llm import State, FSMError, ...` is unchanged.

What else a user can see: `Model.__module__`, tracebacks and `repr` of a class show `fsm_llm.core.definitions.<module>` (for example `fsm_llm.core.definitions.state.State`); a pickle of a model made by an earlier version names the old path and no longer loads (the library pickles nothing: sessions and snapshots are JSON); the 13 private names of the old module (`_PromptRequest`, `_coerce_entity_values` and others) are attributes of the module that defines them, so a `patch("fsm_llm.definitions._x")` must name that module. `fsm_id`, `model_dump`, `model_json_schema` and every error message are unchanged.

**Before**

```python
from fsm_llm.definitions import FSMDefinition
```

**After**

```python
from fsm_llm.core.definitions import FSMDefinition
```

## fsm_llm.expressions

The module `fsm_llm.expressions` is the package `fsm_llm.core.expressions`, which re-exports its 13 public names (`evaluate_logic`, `is_missing`, `get_var`, `soft_equals`, `operations`, ...) with a static `__all__`. `from fsm_llm import evaluate_logic` is unchanged. `fsm_llm.core.expressions.operations` is the same dict object the evaluator reads, so adding an entry to it still takes effect.

The 22 private names of the old module (`_numeric_equal`, `_UNDEFINED`, `_data_operators` and others) are attributes of the module that defines them, and a patch such as `patch("fsm_llm.expressions.logger")` must name the module that performs the lookup, for example `fsm_llm.core.expressions.evaluator.logger`. `__module__` and tracebacks name `fsm_llm.core.expressions.<module>`.

**Before**

```python
from fsm_llm.expressions import evaluate_logic
```

**After**

```python
from fsm_llm.core.expressions import evaluate_logic
```

## fsm_llm.fsm

The module `fsm_llm.fsm` is the package `fsm_llm.engine.fsm`. `FSMManager` is defined in `fsm_llm.engine.fsm.manager` and the package re-exports it with `with_conversation_context`. `from fsm_llm import FSMManager` is unchanged.

Private helpers come from `fsm_llm.engine.fsm.helpers` (`_strip_internal_mapping`, `_unexpected_turn_failure`, `_conversation_entry`). Names the old module only imported (`FSMError`, `FSMDefinition`, `HandlerTiming`, `threading`, `logger`) were never its API: import them from where they are defined, and aim a patch at the module that performs the lookup, for example `fsm_llm.engine.fsm.snapshots.END_CONVERSATION_LOCK_TIMEOUT_SECONDS` (was `fsm_llm.fsm.END_CONVERSATION_LOCK_TIMEOUT_SECONDS`). The loguru record name reads `fsm_llm.engine.fsm.manager` (and `.turns`, `.snapshots`, `.helpers`).

**Before**

```python
from fsm_llm.fsm import FSMManager
```

**After**

```python
from fsm_llm.engine.fsm import FSMManager
```

## fsm_llm.handlers

This module became a package at the same dotted path, split across 7 modules. Every one of its 13 public names still imports from `fsm_llm.handlers`, so a caller of the public API changes nothing. The 1 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.handlers.base
    public: BaseHandler
fsm_llm.handlers.builder
    public: HandlerBuilder, create_handler
fsm_llm.handlers.errors
    public: HandlerExecutionError, HandlerSystemError
fsm_llm.handlers.helpers
    public: clear_keys_on_entry
fsm_llm.handlers.lambda_handler
    public: LambdaHandler
    private: _as_set
fsm_llm.handlers.protocol
    public: ConditionLambda, ExecutionLambda, FSMHandler, HandlerErrorMode,
        HandlerTiming
fsm_llm.handlers.system
    public: HandlerSystem
```

**Before**

```python
from fsm_llm.handlers import _as_set
```

**After**

```python
from fsm_llm.handlers.lambda_handler import _as_set
```

## fsm_llm.llm

`fsm_llm.llm` is a package at the same dotted path (it was `llm.py`). Every public name still imports from `fsm_llm.llm`, and `from fsm_llm import LiteLLMInterface` is unchanged, so a caller of the public API changes nothing. Three things break:

- the names `completion`, `embedding` and `get_supported_openai_params` are no longer attributes of `fsm_llm.llm`: a patch of `fsm_llm.llm.completion` (or `.get_supported_openai_params`) becomes `fsm_llm.llm.providers.litellm.completion` (or `.get_supported_openai_params`), and a patch of `fsm_llm.llm.embedding` becomes `fsm_llm.llm.providers.embedder.embedding`, the modules whose code looks them up (a patch of the old name raises `AttributeError`);
- private helpers no longer import from `fsm_llm.llm`: `_Metered`, `_meter_of` and `_UsageMeter` come from `fsm_llm.llm.usage`, `_finish_reason` and `_GENERIC_FALLBACK_MESSAGE` from `fsm_llm.llm.parsing`, `_completion_response` from `fsm_llm.llm.interface`, `_fill_empty_user_turns` from `fsm_llm.llm.providers.request` and `_embedding_vectors` from `fsm_llm.llm.providers.embedder`;
- litellm is imported in exactly one module, `fsm_llm.llm.providers.litellm` (importing `fsm_llm.llm` still sets `litellm.suppress_debug_info = True` once and still skips a `.env` file).

```text
fsm_llm.llm.interface
    public: DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_TIMEOUT_SECONDS,
        DEFAULT_MAX_TOKENS, LLMInterface, PLACEHOLDER_MODEL, check_tool_transcript,
        and 6 more
fsm_llm.llm.parsing
    public: cut_off_note
fsm_llm.llm.providers.embedder
    public: LiteLLMEmbedder
fsm_llm.llm.providers.litellm
    public: LiteLLMInterface
fsm_llm.llm.providers.request
fsm_llm.llm.usage
```

**Before**

```python
from unittest.mock import patch

with patch("fsm_llm.llm.completion"):
    ...
```

**After**

```python
from unittest.mock import patch

with patch("fsm_llm.llm.providers.litellm.completion"):
    ...
```

## fsm_llm.monitor.instance_manager

This module became a package at the same dotted path, split across 9 modules. Every one of its 13 public names still imports from `fsm_llm.monitor.instance_manager`, so a caller of the public API changes nothing. The 20 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.monitor.instance_manager.agent
    private: _AGENT_CLASSES, _TOOL_BASED_AGENTS, _agent_result_fields, _stub_fn,
        _stub_registry, _tools_used
fsm_llm.monitor.instance_manager.examples
    public: find_examples_dir, load_preset_json, read_json_file, validate_preset_id
fsm_llm.monitor.instance_manager.handlers
    public: monitor_handlers_on, register_monitor_handlers,
        unregister_monitor_handlers
    private: _HANDLER_STATES, _MonitorHandler, _build_context_capture_handlers,
        _build_monitor_handlers, _handlers_for
fsm_llm.monitor.instance_manager.lifecycle
    private: _pending_tasks
fsm_llm.monitor.instance_manager.managed
    public: ManagedAgent, ManagedFSM, ManagedInstance, ManagedWorkflow
    private: _ManagedBase, _new_instance_id
fsm_llm.monitor.instance_manager.manager
    public: InstanceManager
    private: _T
fsm_llm.monitor.instance_manager.snapshots
    public: snapshot_from_api
fsm_llm.monitor.instance_manager.workflow
    private: _status_str
fsm_llm.monitor.instance_manager.workflow_presets
    private: _TerminalStep, _WORKFLOW_PRESETS, _preset_branching, _preset_linear
```

**Before**

```python
from fsm_llm.monitor.instance_manager import _agent_result_fields
```

**After**

```python
from fsm_llm.monitor.instance_manager.agent import _agent_result_fields
```

## fsm_llm.monitor.server

`fsm_llm.monitor.server` is a package of route modules. The 49 route functions no longer import from it; `app`, `configure`, `get_manager`, `STATIC_DIR` and `TEMPLATE_DIR` do. See [monitor route functions](monitor-routes.md).

```text
fsm_llm.monitor.server.application
    public: STATIC_DIR, TEMPLATE_DIR, api_auth, app, configure, get_flows, and 5
        more
fsm_llm.monitor.server.routes_agent
    public: api_agent_cancel, api_agent_launch, api_agent_result, api_agent_status
fsm_llm.monitor.server.routes_builder
    public: api_builder_delete, api_builder_result, api_builder_send,
        api_builder_start
fsm_llm.monitor.server.routes_fsm
    public: api_fsm_conversations, api_fsm_converse, api_fsm_end_conversation,
        api_fsm_launch, api_fsm_start_conversation
fsm_llm.monitor.server.routes_monitoring
    public: api_activity, api_capabilities, api_config_get, api_config_set,
        api_conversation, api_conversations, and 11 more
fsm_llm.monitor.server.routes_visualize
    public: api_agent_visualize, api_fsm_load, api_fsm_visualize,
        api_fsm_visualize_preset, api_preset_fsm, api_presets, and 1 more
fsm_llm.monitor.server.routes_workflow
    public: api_workflow_advance, api_workflow_cancel, api_workflow_event,
        api_workflow_instances, api_workflow_launch, api_workflow_presets, and 1
        more
fsm_llm.monitor.server.support
fsm_llm.monitor.server.websocket
    public: websocket_endpoint
```

**Before**

```python
from fsm_llm.monitor.server import api_metrics
```

**After**

```python
from fsm_llm.monitor.server.routes_monitoring import api_metrics
```

## fsm_llm.ollama

`fsm_llm.ollama` moved whole to `fsm_llm.llm.ollama`. The four public helpers (`is_ollama_model`, `apply_ollama_params`, `prepare_ollama_messages`, `build_ollama_response_format`) are unchanged. `fsm_llm.llm.ollama` is not re-exported by `fsm_llm.llm`.

**Before**

```python
from fsm_llm.ollama import is_ollama_model
```

**After**

```python
from fsm_llm.llm.ollama import is_ollama_model
```

## fsm_llm.pipeline

The module `fsm_llm.pipeline` is the package `fsm_llm.engine.turn`. `MessagePipeline` is defined in `fsm_llm.engine.turn.message_pipeline` and the package re-exports it (it was never exported from `fsm_llm` itself).

Private helpers come from the module that defines them: `_TurnSnapshot`, `_TurnRecord`, `_NOT_CAPTURED`, `_is_silent` and `_response_temperature` from `fsm_llm.engine.turn.records`; the conversation Pass-1 helpers (`_BulkFailed`, `_coerce_bool` and the other `_coerce_*` functions) from `fsm_llm.engine.kinds.conversation.bulk` and `.fields`; the record helpers from `fsm_llm.engine.kinds.records`. Names the old module only imported (`Classifier`, `logger`, `copy`, `threading`) were never its API: aim a patch at the module that performs the lookup, for example `fsm_llm.engine.turn.transitions.Classifier` (was `fsm_llm.pipeline.Classifier`). The loguru record name reads `fsm_llm.engine.turn.<module>` instead of `fsm_llm.pipeline`.

**Before**

```python
from fsm_llm.pipeline import MessagePipeline
```

**After**

```python
from fsm_llm.engine.turn import MessagePipeline
```

## fsm_llm.prompts

The module `fsm_llm.prompts` is the package `fsm_llm.context.layout`, split by pass (`base`, `extraction`, `response`, `classification`). Every public name imports from the package, so the line to change is the module path alone. Two exceptions: the private helpers (`_cap_history_line`, `_history_messages`) come from `fsm_llm.context.layout.base`, and `ClassificationPromptConfig`, which the old module only imported, comes from `fsm_llm.core.definitions`. The prompt text is byte-identical.

**Before**

```python
from fsm_llm.prompts import BasePromptBuilder
```

**After**

```python
from fsm_llm.context.layout import BasePromptBuilder
```

## fsm_llm.reasoning.reasoning_modes

This module became a package at the same dotted path, split across 13 modules. Every one of its 12 public names still imports from `fsm_llm.reasoning.reasoning_modes`, so a caller of the public API changes nothing. The 4 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.reasoning.reasoning_modes._phrases
    private: _IMPORTANT_NO_QUESTIONS, _NO_QUESTIONS, _PRESENT_FINAL,
        _PRESENT_FINDINGS
fsm_llm.reasoning.reasoning_modes.abductive
    public: abductive_fsm
fsm_llm.reasoning.reasoning_modes.analogical
    public: analogical_fsm
fsm_llm.reasoning.reasoning_modes.analytical
    public: analytical_fsm
fsm_llm.reasoning.reasoning_modes.classifier
    public: classifier_fsm
fsm_llm.reasoning.reasoning_modes.creative
    public: creative_fsm
fsm_llm.reasoning.reasoning_modes.critical
    public: critical_fsm
fsm_llm.reasoning.reasoning_modes.deductive
    public: deductive_fsm
fsm_llm.reasoning.reasoning_modes.hybrid
    public: hybrid_fsm
fsm_llm.reasoning.reasoning_modes.inductive
    public: inductive_fsm
fsm_llm.reasoning.reasoning_modes.orchestrator
    public: orchestrator_fsm
fsm_llm.reasoning.reasoning_modes.registry
    public: ALL_REASONING_FSMS
fsm_llm.reasoning.reasoning_modes.simple_calculator
    public: simple_calculator_fsm
```

**Before**

```python
from fsm_llm.reasoning.reasoning_modes import _IMPORTANT_NO_QUESTIONS
```

**After**

```python
from fsm_llm.reasoning.reasoning_modes._phrases import _IMPORTANT_NO_QUESTIONS
```

## fsm_llm.security

This module became a package at the same dotted path, split across 6 modules. Every one of its 9 public names still imports from `fsm_llm.security`, so a caller of the public API changes nothing. The 61 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.security.api_keys
    public: api_key_matches, checked_api_key, presented_api_key
fsm_llm.security.names
    public: INTERNAL_KEY_PREFIXES, has_internal_prefix, is_forbidden_context_entry
    private: _ACRONYM_BOUNDARY, _AMBIGUOUS_CREDENTIAL_ABBREVIATIONS,
        _CAMEL_BOUNDARY, _CONTAINER_MAX_DEPTH, _CONTAINER_NODE_BUDGET,
        _CREDENTIAL_NAME_ANY_RE, and 18 more
fsm_llm.security.paths
    public: resolve_within
fsm_llm.security.patterns
    public: COMPILED_FORBIDDEN_CONTEXT_PATTERNS, FORBIDDEN_CONTEXT_PATTERNS
    private: _BEARER_TOKEN_QUALIFIERS, _CREDENTIAL_MATERIAL_HEADS_SHARED,
        _CRYPTO_AT_WORD, _CRYPTO_GAP, _CRYPTO_GAP_REACH, _CRYPTO_KEY_QUALIFIERS, and
        11 more
fsm_llm.security.shape_data
    private: _AUTH_SCHEME_WORDS, _CREDENTIAL_VALUE_CHARSET_RE,
        _CREDENTIAL_VALUE_PREFIXES, _IDENTIFIER_NOUN_VOCABULARY,
        _MIN_CREDENTIAL_VALUE_ENTROPY, _MIN_CREDENTIAL_VALUE_LENGTH, and 7 more
fsm_llm.security.shapes
    private: _PURE_HEX_RE, _colon_composite_tail, _generic_shape_is_credential,
        _is_path_shaped, _looks_like_credential_value, _normalise_credential_value,
        and 1 more
```

**Before**

```python
from fsm_llm.security import _colon_composite_tail
```

**After**

```python
from fsm_llm.security.shapes import _colon_composite_tail
```

## fsm_llm.session

`fsm_llm.session` moved whole to `fsm_llm.engine.session` (`SessionState`, `SessionStore`, `FileSessionStore`, `SESSION_ID_RE`). `from fsm_llm import FileSessionStore, SessionState, SessionStore` is unchanged. A patch aims at the module that performs the lookup, for example `fsm_llm.engine.session.os.replace` (was `fsm_llm.session.os.replace`). The loguru record name reads `fsm_llm.engine.session`.

**Before**

```python
from fsm_llm.session import FileSessionStore
```

**After**

```python
from fsm_llm.engine.session import FileSessionStore
```

## fsm_llm.transition_evaluator

`fsm_llm.transition_evaluator` moved whole to `fsm_llm.core.transition_evaluator` (the code is unchanged). `TransitionEvaluator` and `TransitionEvaluatorConfig` are still exported from `fsm_llm`.

**Before**

```python
from fsm_llm.transition_evaluator import TransitionEvaluator
```

**After**

```python
from fsm_llm.core.transition_evaluator import TransitionEvaluator
```

## fsm_llm.utilities

The module `fsm_llm.utilities` is split by layer and no longer exists: no alias, no shim. Each name moved to the module that defines it, so the codemod above cannot rewrite it with a prefix rule. The names imported from the package root (`from fsm_llm import extract_json_from_text, get_fsm_summary, load_fsm_definition, load_fsm_from_file`) are unchanged, and so is `fsm_llm.reasoning.utilities`, which is a different module. `fsm_llm.llm` and `fsm_llm.context` keep their dotted paths and are packages now (see their sections below).

The 13 private helpers moved with the code that uses them (for example `_cycle_reaching_containers` to `fsm_llm.context.filtering`, `_redact_mapping` to `fsm_llm.core.redaction` and `_match_brace_partners` to `fsm_llm.llm.json_extraction`). A mock patch aimed at a moved private name must aim at the new defining module (`fsm_llm.core.redaction.logger` for the redaction warning, `fsm_llm.llm.json_extraction.logger` for the extraction logs). `ContextFilterError` is defined in `fsm_llm.context.filtering` now; it is still an `FSMError` and a `ValueError`.

```text
fsm_llm.context.filtering
    public: ContextFilterError, filter_context_tree, redact_non_json_leaf,
        redacting_json_default
fsm_llm.core.coerce
    public: coerce_confidence, coerce_float, coerce_int, parse_bool_word
fsm_llm.core.loaders
    public: get_fsm_summary, load_fsm_definition, load_fsm_from_file,
        read_json_object
fsm_llm.core.redaction
    public: CYCLE_PLACEHOLDER, REDACTED, redact_context
fsm_llm.llm.json_extraction
    public: balanced_json_objects, extract_json_from_text
fsm_llm.llm.text
    public: remove_think_blocks, resolve_reasoning_trace, strip_think_and_fences
```

**Before**

```python
from fsm_llm.utilities import extract_json_from_text, filter_context_tree
```

**After**

```python
from fsm_llm.context.filtering import filter_context_tree
from fsm_llm.llm.json_extraction import extract_json_from_text
```

## fsm_llm.validator

The module `fsm_llm.validator` is the package `fsm_llm.core.verify`, which re-exports `FSMValidator`, `FSMValidationResult`, `validate_fsm_from_file`, `upgrade_fsm_file` and `UpgradeResult`. `from fsm_llm import FSMValidator, FSMValidationResult, validate_fsm_from_file` is unchanged.

The command line functions `main` and `main_cli` are not re-exported: import them from `fsm_llm.core.verify.cli`. The console script `fsm-llm-validate` now points at `fsm_llm.core.verify.cli:main_cli`: an editable install must run `pip install -e .` once to refresh the script, a wheel install gets it with the release. Private helpers move with their code (`_string_literals`, `_condition_keys`, `_RESULT_KINDS` to `fsm_llm.core.verify.structure`; `_no_duplicate_keys`, `_refuse` to `fsm_llm.core.verify.upgrade`), and a patch aims at the module that performs the lookup (`fsm_llm.core.verify.validator.FSMDefinition`, `fsm_llm.core.verify.cli.setup_cli_logging`).

**Before**

```python
from fsm_llm.validator import FSMValidator
```

**After**

```python
from fsm_llm.core.verify import FSMValidator
```

## fsm_llm.visualizer

The module `fsm_llm.visualizer` is the package `fsm_llm.core.visualize`. The public graph and diagram names import from the package: `build_fsm_graph`, `FSMGraph`, `FSMGraphEdge`, `FSMGraphNode`, `to_mermaid`, `to_dot`, `visualize_fsm_ascii`, `visualize_fsm_from_file`, `ASCII_STYLES`, `OUTPUT_FORMATS`. `from fsm_llm import build_fsm_graph, to_mermaid, to_dot` is unchanged.

The other names import from the module that defines them: the glyph tables and the ASCII geometry (`BOX_STYLES`, `ARROW_STYLES`, `ICONS`, `build_graph_representation`, `detect_loops` and the private row helpers) from `fsm_llm.core.visualize.ascii_layout`; the section and diagram builders (`build_fancy_header`, `generate_enhanced_ascii_diagram` and the others) from `fsm_llm.core.visualize.ascii_render`; the shape checks from `fsm_llm.core.visualize.shapes`; `main` and `main_cli` from `fsm_llm.core.visualize.cli`. The console script `fsm-llm-visualize` now points at `fsm_llm.core.visualize.cli:main_cli` (run `pip install -e .` once in a clone to refresh it), and `fsm-llm --mode visualize` is unchanged. The loguru record name of the CLI reads `fsm_llm.core.visualize.cli`.

**Before**

```python
from fsm_llm.visualizer import build_fsm_graph
```

**After**

```python
from fsm_llm.core.visualize import build_fsm_graph
```

## fsm_llm.workflows.engine

This module became a package at the same dotted path, split across 2 modules. Every one of its 3 public names still imports from `fsm_llm.workflows.engine`, so a caller of the public API changes nothing. The 3 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.workflows.engine.engine
    public: WorkflowEngine
fsm_llm.workflows.engine.helpers
    public: LifecycleHook, Timer
    private: _CancelRequest, _timeout_error, _timer_key
```

**Before**

```python
from fsm_llm.workflows.engine import _timeout_error
```

**After**

```python
from fsm_llm.workflows.engine.helpers import _timeout_error
```

## fsm_llm.workflows.steps

This module became a package at the same dotted path, split across 8 modules. Every one of its 12 public names still imports from `fsm_llm.workflows.steps`, so a caller of the public API changes nothing. The 5 private names (a leading underscore) and the attributes the old file only imported moved to the module that defines or uses them: import a private name from there, and aim a patch at the module that performs the lookup. `__module__`, tracebacks and loguru record names read the defining module.

```text
fsm_llm.workflows.steps.agent
    public: AgentStep
fsm_llm.workflows.steps.base
    public: WorkflowStep
fsm_llm.workflows.steps.calls
    public: APIStep, LLMStep
    private: _lookup_path
fsm_llm.workflows.steps.composite
    public: ParallelStep, RetryStep
fsm_llm.workflows.steps.conversation
    public: ConversationStep
fsm_llm.workflows.steps.helpers
    private: _STEP_EXECUTOR, _call_user_callable, _error_text, _is_async_callable
fsm_llm.workflows.steps.routing
    public: AutoStep, ConditionStep, SwitchStep
fsm_llm.workflows.steps.waits
    public: TimerStep, WaitEventStep
```

**Before**

```python
from fsm_llm.workflows.steps import _call_user_callable
```

**After**

```python
from fsm_llm.workflows.steps.helpers import _call_user_callable
```

## Removed and renamed modules that are not in the map

Two groups of modules of 0.12.0 stopped importing without a move recorded in the path map. The planner harness was removed together with its package (every module under `fsm_llm.harness`, see [removals](removals.md#the-planner-harness-is-removed)), and `fsm_llm.agents.skills` was renamed `fsm_llm.agents.python_tools` (see [agents renames](agents-renames.md)).

```text
fsm_llm.harness (and every module under it)  removed, no replacement
fsm_llm.agents.skills                        fsm_llm.agents.python_tools
```

**Before**

```python
from fsm_llm.agents.skills import SkillLoader
```

**Sketch**

```python
from fsm_llm.agents.python_tools import PythonToolLoader
```

## Modules added after 0.12.0

The modules below exist in the layout of this release but were never published at another path: they were added and then moved inside the unreleased development train of 0.13.0, so no released import of them stopped working and there is nothing to migrate. They are listed so the page accounts for every old key of the path map.

```text
fsm_llm.agent_skills  ->  fsm_llm.skills.catalog
fsm_llm.agent_skills_cli  ->  fsm_llm.skills.cli
fsm_llm.codes  ->  fsm_llm.core.codes
fsm_llm.executors  ->  fsm_llm.engine.kinds.base, fsm_llm.engine.kinds.completion, fsm_llm.engine.kinds.conversation.executor, fsm_llm.engine.kinds.registry
fsm_llm.flow_analysis  ->  fsm_llm.core.verify.flow
fsm_llm.okf  ->  fsm_llm.okf.bundle, fsm_llm.okf.checker, fsm_llm.okf.links, fsm_llm.okf.loader, fsm_llm.okf.models, fsm_llm.okf.parser, fsm_llm.okf.reading, fsm_llm.okf.writer, fsm_llm.text
fsm_llm.okf_bridge  ->  fsm_llm.engine.okf_bridge, fsm_llm.okf.memory_export
fsm_llm.okf_cli  ->  fsm_llm.okf.cli
fsm_llm.schema_registry  ->  fsm_llm.core.definitions.schema
fsm_llm.state_kinds  ->  fsm_llm.core.state_kinds
```

**Sketch**

```python
from fsm_llm.core.codes import Status
from fsm_llm.core.state_kinds import KINDS
```
