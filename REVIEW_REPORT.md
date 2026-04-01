# FSM-LLM Source Code Review Report

**Date**: 2026-03-31
**Scope**: All 5 packages in `src/` (~80 Python files)
**Focus**: Bugs, architecture, performance, security, dead code, type safety

---

## Executive Summary

| Package | Files | Issues | Critical | Medium | Low |
|---------|-------|--------|----------|--------|-----|
| `fsm_llm` (core) | 23 | 23 | 3 | 8 | 12 |
| `fsm_llm_reasoning` | 10 | 12 | 2 | 4 | 6 |
| `fsm_llm_workflows` | 9 | 16 | 3 | 5 | 8 |
| `fsm_llm_agents` | 42 | 42 | 6 | 15 | 21 |
| `fsm_llm_monitor` | 9 | 18 | 4 | 5 | 9 |
| **Total** | **93** | **111** | **18** | **37** | **56** |

The codebase is well-structured with good separation of concerns. The most impactful issues cluster around **concurrency bugs**, **duplicated patterns in agents**, and **silent error swallowing**.

---

## Critical Issues (Fix Immediately)

### 1. FSM Cache LRU Eviction Race — `fsm_llm/fsm.py:145-154`

If `self.fsm_loader(fsm_id)` raises, the cache loses the evicted entry without gaining the new one. Cache degrades silently.

```python
# Current: evicts BEFORE loading (risky)
if len(self.fsm_cache) >= self._max_fsm_cache_size:
    evicted_key, _ = self.fsm_cache.popitem(last=False)
self.fsm_cache[fsm_id] = self.fsm_loader(fsm_id)  # Can fail

# Fix: load first, then evict
fsm_def = self.fsm_loader(fsm_id)
if len(self.fsm_cache) >= self._max_fsm_cache_size:
    self.fsm_cache.popitem(last=False)
self.fsm_cache[fsm_id] = fsm_def
```

### 2. Per-Conversation Lock Race — `fsm_llm/fsm.py:452-480`

Between releasing `_lock` and acquiring `conv_lock`, `_cleanup_conversation_resources()` can run, making the lock reference stale. Concurrent `update_conversation_context()` + `end_conversation()` = orphaned data.

### 3. Uninitialized Variable in Force-Pop — `fsm_llm_reasoning/engine.py:428-432`

If `get_data()` succeeds but `pop_fsm()` fails, `force_popped` stays `False`. Later code re-reads context, potentially getting different data.

```python
# Fix: set flag before the operation that may fail
sub_final_context = self.orchestrator.get_data(conv_id)
force_popped = True  # Move here
self.orchestrator.pop_fsm(conv_id)
```

### 4. Workflow Orphaned on Step Exception — `fsm_llm_workflows/engine.py:350-352`

`_transition_to_state()` sets status to RUNNING before executing the step. If `_execute_workflow_step()` throws, the workflow is stuck in RUNNING with an already-changed `current_step_id`.

```python
# Fix: wrap in try-except
try:
    await self._execute_workflow_step(instance, _depth=_depth + 1)
except Exception as e:
    instance.update_status(WorkflowStatus.FAILED, error=str(e))
    raise
```

### 5. Off-by-One Errors Across Agents — Multiple files

| Agent | File:Line | Bug |
|-------|-----------|-----|
| Stall detection | `agents/handlers.py:87` | Triggers at 2, message says "3 consecutive" |
| Swarm handoffs | `agents/swarm.py:144` | `>` should be `>=`, allows 1 extra handoff |
| PlanExecute | `agents/plan_execute.py:284` | Terminates 1 iteration early vs ReactAgent |
| ADaPT | `agents/adapt.py:276` | Same inconsistency as PlanExecute |

### 6. Unsafe Float Check — `fsm_llm_agents/self_consistency.py:239-243`

`math.isnan()` on an `int` raises `TypeError` in some Python builds. The `isinstance` check runs first but uses `int | float` union, letting ints through to `isnan()`.

---

## Architecture Issues

### 7. God Classes

| Class | File | Lines | Responsibilities |
|-------|------|-------|-----------------|
| `API` | `fsm_llm/api.py` | 1200+ | FSM loading, stacking, conversations, handlers, sessions, context |
| `ReasoningEngine` | `reasoning/engine.py` | 500+ | Classification, execution, merging, validation, tracing |
| `InstanceManager` | `monitor/instance_manager.py` | 1400+ | FSM, workflow, agent lifecycle, events, metrics |
| `AgentHandlers` | `agents/handlers.py` | 292 | Tool execution, iteration limiting, stall detection |

**Recommendation**: Split `API` into `ConversationManager`, `FSMStackManager`, `SessionManager`. Split `AgentHandlers` into `ToolExecutor`, `IterationLimiter`, `StallDetector`.

### 8. Duplicated Patterns Across Agents

Five agent types reimplement iteration limiting with slight variations:
- `evaluator_optimizer.py:191-202`
- `maker_checker.py:197-208`
- `debate.py:190-201`
- `orchestrator.py:200-213`
- `adapt.py:269-283`

Similarly, answer extraction is duplicated in `base.py`, `evaluator_optimizer.py`, `prompt_chain.py`, and `adapt.py`.

**Fix**: Extract to `BaseAgent._make_standard_iteration_limiter()` and a configurable `AnswerExtractor`.

### 9. No Shared Base for Tool-Using Agents

`ReactAgent`, `ReflexionAgent`, `PlanExecuteAgent` all use tools with duplicated `_handlers` initialization. Should have a `ToolUsingAgent(BaseAgent)` abstract class.

### 10. Tight Coupling via Hard-Coded Context Keys — Reasoning Package

`merge_reasoning_results()` uses 9 hard-coded `if/elif` branches. Adding a 10th reasoning type requires changes in multiple files.

**Fix**: Registry/strategy map pattern:
```python
REASONING_RESULT_MAPPINGS = {
    ReasoningType.ANALYTICAL: {"key_insights": ContextKeys.KEY_INSIGHTS, ...},
}
```

---

## Performance Issues

### 11. Deep Copy on Every Handler Execution — `fsm_llm/pipeline.py:182`

`copy.deepcopy(instance.context.data)` is called per handler (8 timing points). For large contexts, this is O(n) per invocation. Use shallow copy for read-only handlers.

### 12. O(n^2) JSON Extraction — `fsm_llm/utilities.py:69-107`

For text with many `{` characters, the balanced-brace finder tries each starting position, each scanning the rest. Should use a single-pass state machine.

### 13. O(n^2) Terminal Check — `fsm_llm_monitor/instance_manager.py:766-773`

`has_conversation_ended()` called for ALL conversations every time one ends. Track ended conversations in a set instead.

### 14. JSON Serialization in Loop — `fsm_llm_reasoning/handlers.py:345-355`

`json.dumps(filtered)` called inside a `while` loop for context pruning. Each iteration re-serializes the entire dict. Estimate size reduction instead.

### 15. Preset Parsing on Every Request — `fsm_llm_monitor/server.py:747-781`

All FSM preset files are read and parsed on every `/presets` API call. Cache after startup.

---

## Security Issues

### 16. JsonLogic Depth Check Bypass — `fsm_llm/expressions.py:699-703`

Depth increments per argument, not per nesting level. A flat structure with 1000 arguments at depth 0 bypasses the DoS protection. Fix: increment only on dict nesting.

### 17. CORS Config Mutates FastAPI Internals — `fsm_llm_monitor/server.py:119-131`

Directly mutating `app.user_middleware` kwargs is undocumented and fragile. Rebuild the middleware stack instead.

### 18. Arbitrary File Read — `fsm_llm_monitor/bridge.py:132-142`

`load_fsm_from_file()` reads any path. Currently internal-only, but should be marked `_private` to prevent accidental API exposure.

---

## Dead Code & Bloat

### 19. Unused Legacy State Aliases — `fsm_llm_agents/constants.py:353-359`

`MetaBuilderStates.INTAKE`, `.REVIEW`, `.CLASSIFY` are never referenced anywhere.

### 20. Dead Parameter Recovery Code — `fsm_llm_agents/handlers.py:152-165`

`example_params` dict is built but never used. Log message claims "will provide format example" but doesn't.

### 21. Duplicate Context Keys — `fsm_llm_reasoning/handlers.py:209-216`

`SOLUTION_VALID` / `VALIDATION_RESULT` and `CONFIDENCE_LEVEL` / `SOLUTION_CONFIDENCE` both set to same values. Unclear which is canonical.

### 22. Missing `ConversationStep` in `__all__` — `fsm_llm_workflows/__init__.py`

Imported but not exported. Users can't `from fsm_llm_workflows import ConversationStep`.

---

## Silent Error Swallowing (Pattern)

This anti-pattern appears across multiple packages:

| Location | Impact |
|----------|--------|
| `reasoning/engine.py:222-223` | `except Exception: pass` hides cleanup failures |
| `monitor/bridge.py:140-142` | All load errors become `None` |
| `monitor/server.py:863-869` | Agent state errors return dummy data |
| `reasoning/handlers.py:373-466` | Unknown reasoning type returns empty dict silently |
| `workflows/engine.py:533-534` | Timer task errors logged but swallowed |

**Recommendation**: At minimum, log at warning level. For critical paths, re-raise or return error result objects.

---

## Type Safety Gaps

### 23. `dict[str, Any]` Overuse

26 occurrences in reasoning alone. Context dicts carry arbitrary keys with no compile-time safety. Consider typed context protocols for specific subsystems.

### 24. Untyped LLM Kwargs — `fsm_llm/llm.py:174-182`

`**kwargs` passed to LiteLLM are never validated. A typo like `temperatur=0.5` passes silently.

### 25. `Any` Instead of Union Types — `fsm_llm_monitor/instance_manager.py:154,190`

`self.engine: Any` and `self.result: Any` should be `WorkflowEngine | None` and `AgentResult | None`.

---

## Recommended Priority Order

### Sprint 1: Critical Bugs
1. Fix FSM cache eviction race (fsm.py)
2. Fix per-conversation lock race (fsm.py)
3. Fix workflow orphan on step exception (workflows/engine.py)
4. Fix off-by-one errors across agents (4 files)
5. Fix force_popped variable bug (reasoning/engine.py)
6. Fix unsafe float check (agents/self_consistency.py)

### Sprint 2: Performance & Security
7. Replace deepcopy with shallow copy in handler execution
8. Fix O(n^2) JSON extraction
9. Fix O(n^2) terminal check in monitor
10. Fix JsonLogic depth check bypass
11. Cache preset file parsing
12. Fix CORS internal mutation

### Sprint 3: Architecture
13. Extract iteration limiter from agents into shared base
14. Create ToolUsingAgent abstract class
15. Add reasoning type registry pattern
16. Split API god class (if feasible)

### Sprint 4: Cleanup
17. Remove dead code (legacy aliases, unused params)
18. Add ConversationStep to workflows __all__
19. Replace silent exception swallowing with logging
20. Consolidate duplicate context keys in reasoning

---

## Strengths Worth Preserving

- **Clean package boundaries**: Each sub-package has its own constants, exceptions, and definitions
- **Pydantic validation**: Models use `model_validator` effectively for complex rules
- **Lock discipline**: Thread safety is generally solid with consistent lock usage
- **Defensive session ID validation**: Regex + path resolution prevents traversal
- **Handler priority system**: Well-designed 8-timing-point hook architecture
- **Atomic file writes**: Session persistence uses tmp-then-rename pattern
