# Strands Feature Adaptation -- Phase 2

**Status**: Implemented
**Depends on**: Phase 1 (complete)

Phase 2 covers the remaining 8 features from the Strands adaptation report. These are larger in scope and build on the Phase 1 infrastructure (streaming, session persistence, invocation state, schema enforcement).

---

## 1. OpenTelemetry (OTEL) Observability (Strands Feature #1)

**Enhancement value**: Production readiness

**What FSM-LLM has today**: Custom `EventCollector` with bounded in-memory deques, 15+ event types (conversation lifecycle, state transitions, agent/workflow events), WebSocket streaming, and a web dashboard. No OTEL. No distributed tracing. No correlation IDs across multi-agent calls.

**What to build**: An OTEL exporter layer that wraps existing `EventCollector` events into OTEL spans and metrics using the adapter pattern (no modifications to the existing monitor).

**Key implementation details**:
- `OTELExporter` class in `fsm_llm_monitor/otel.py` wraps `EventCollector.record_event()` via the adapter pattern
- Conversation lifecycle events become parent spans; state transitions, processing, and errors become child spans
- Thread-safe span management via `_spans_lock`
- Pluggable backends: Jaeger, Datadog, Langfuse via OpenTelemetry SDK exporters
- New optional dependency: `opentelemetry-api`, `opentelemetry-sdk` (in `otel` extras)

**Not yet implemented** (future work):
- LLM call instrumentation (`LiteLLMInterface.generate_response()` / `extract_field()` spans)
- Tool execution instrumentation (`ToolRegistry.execute()` spans)
- Trace context propagation through `AgentResult.final_context`
- `trace_id` / `span_id` fields on `MonitorEvent`

**Scope**: ~270 LOC, 1 new file (`otel.py`)

---

## 2. Swarm Pattern -- Emergent Agent Coordination (Strands Feature #3)

**Enhancement value**: New multi-agent capability

**What FSM-LLM has today**: `OrchestratorAgent` (hierarchical delegation), `DebateAgent` (structured multi-agent), agents-as-tools via `ToolRegistry.register_agent()`. No emergent coordination where agents decide handoff targets dynamically.

**What to build**: A `SwarmAgent` pattern (#13) where each agent's conclude state includes a `next_agent` classification extraction. Agents hand off by returning the next agent ID + message + context. The swarm runner loops until an agent returns no `next_agent`.

**Key implementation details**:
- Shared state flows through `WorkingMemory` passed to all agents (leverages Phase 1 invocation state for hidden orchestration metadata)
- Each agent's FSM conclude state extracts: `next_agent` (agent ID or None), `handoff_message`, `handoff_context`
- `SwarmAgent.run()` loops: run agent -> extract handoff -> run next agent -> ... until no handoff
- Entry agent configurable, max handoff limit for safety
- Add `build_swarm_fsm()` to `fsm_definitions.py`

**Estimated scope**: ~400 LOC, 1-2 new files in `fsm_llm_agents/`

---

## 3. Graph-Based Agent Orchestration (Strands Feature #4)

**Enhancement value**: Deterministic multi-agent routing

**What FSM-LLM has today**: Workflows have DAG validation and state-based transitions, but they're step-typed (APICallStep, LLMProcessingStep, etc.), not agent-typed. No way to wire agents into a graph with conditional edges.

**What to build**: An `AgentGraph` that uses the existing workflow engine infrastructure but treats agents as nodes. Each node runs an agent, its `AgentResult.final_context` becomes the edge state, and conditional edge functions evaluate against that state.

**Key implementation details**:
- `AgentGraphBuilder` with `add_node(name, agent)`, `add_edge(source, target, condition=fn)`, `set_entry(name)`, `build()`
- Condition functions receive `AgentResult.final_context` and return bool
- Reuse workflow engine's DAG validation and cycle detection
- Support parallel branches (independent nodes execute concurrently)
- `AgentGraph.run(task)` returns combined `AgentResult`

**Estimated scope**: ~350 LOC, 1-2 new files in `fsm_llm_agents/`

---

## 4. MCP (Model Context Protocol) Tool Integration (Strands Feature #5)

**Enhancement value**: Ecosystem expansion

**What FSM-LLM has today**: No MCP support. All tools are native via `@tool` decorator and `ToolRegistry`.

**What to build**: An `MCPToolProvider` that connects to an MCP server and converts its tool definitions into `ToolDefinition` objects compatible with `ToolRegistry`.

**Key implementation details**:
- `MCPToolProvider` wraps an MCP client (stdio or HTTP transport)
- `registry.register_mcp(server_params)` loads all tools from an MCP server
- Each MCP tool becomes a `ToolDefinition` with a function that calls the MCP server
- The `@tool` decorator remains for custom tools -- MCP supplements, not replaces
- New optional dependency: `mcp` SDK (in `agents` extras)

**Estimated scope**: ~250 LOC, 1 new file in `fsm_llm_agents/`

---

## 5. Semantic Tool Retrieval (Strands Feature #6)

**Enhancement value**: Scalability

**What FSM-LLM has today**: `ToolRegistry.to_classification_schema()` converts tools to a `ClassificationSchema` for LLM-based intent classification. All registered tools are described to the model. Doesn't scale past ~20 tools.

**What to build**: A `SemanticToolRegistry` subclass that embeds tool descriptions at registration time and retrieves top-K tools per query using cosine similarity.

**Key implementation details**:
- Embed tool descriptions using a lightweight embedding model (e.g., sentence-transformers)
- On each query, compute similarity and return top-K tools
- Falls back to full registry for small tool sets (<20 tools)
- Integrates with existing `ToolRegistry` API -- drop-in replacement
- New optional dependency: embedding library (in `agents` extras)

**Estimated scope**: ~200 LOC, 1 new file in `fsm_llm_agents/`

---

## 6. Dependency-Based Workflow Parallelism (Strands Feature #7)

**Enhancement value**: Workflow ergonomics

**What FSM-LLM has today**: `ParallelStep` runs hardcoded parallel branches via `asyncio.gather`. No explicit dependency declarations. No automatic parallelism detection from dependency graph.

**What to build**: A `DependencyResolver` that computes execution waves from declared dependencies, and auto-parallelizes independent steps.

**Key implementation details**:
- `WorkflowDefinition.add_dependency(step_id, depends_on)` builds a dependency graph
- `DependencyResolver` computes execution waves (topological sort into parallel groups)
- The engine runs each wave's steps in parallel via existing `asyncio.gather`
- Backward-compatible -- existing workflows without dependencies behave identically
- Existing `ParallelStep` remains for explicit grouping; dependency-based parallelism is implicit

**Estimated scope**: ~200 LOC, 1 new file in `fsm_llm_workflows/`

---

## 7. Agent SOPs -- Standard Operating Procedures (Strands Feature #8)

**Enhancement value**: Reusability

**What FSM-LLM has today**: Agents have `persona` and pattern-specific prompts. The meta builder generates artifacts interactively. No reusable SOP/template system.

**What to build**: An `SOPRegistry` that loads YAML/JSON SOP definitions and configures agents from them.

**Key implementation details**:
- SOP definition format: task description, required tools, prompt template, output schema, agent pattern
- `Agent.from_sop("code-review", tools=[...])` configures an agent with the SOP's settings
- `SOPRegistry.register(path)` / `SOPRegistry.list()` / `SOPRegistry.get(name)`
- Ship 3-5 built-in SOPs for common patterns (code review, summarization, data extraction)
- SOPs are parameterized -- template variables filled at instantiation

**Estimated scope**: ~250 LOC, 1-2 new files in `fsm_llm_agents/`

---

## 8. A2A (Agent-to-Agent Protocol) for Remote Agents (Strands Feature #10)

**Enhancement value**: Distributed agents

**What FSM-LLM has today**: Agents communicate via synchronous method calls and shared `WorkingMemory`. `register_agent()` wraps agents as local tools. Monitor server has REST endpoints but for monitoring, not agent invocation.

**What to build**: An `AgentServer` that wraps any FSM-LLM agent as an HTTP endpoint, and a `RemoteAgentTool` that wraps a remote agent URL as a local tool.

**Key implementation details**:
- `AgentServer(agent, host, port)` exposes `/invoke` (full result) and `/stream` (SSE token stream, leveraging Phase 1 streaming)
- `RemoteAgentTool(url, name, description)` creates a `ToolDefinition` that calls the remote agent
- Uses existing `ToolRegistry.register_function()` pattern for integration
- Extends the existing FastAPI infrastructure from `fsm_llm_monitor`
- New optional dependency: `httpx` (for async HTTP client in `agents` extras)

**Estimated scope**: ~300 LOC, 1-2 new files in `fsm_llm_agents/`

---

## Implementation Priority

Recommended order based on value, dependencies, and Phase 1 synergies:

| Priority | Feature | Builds on Phase 1 | Estimated LOC |
|----------|---------|-------------------|---------------|
| 1 | MCP Tool Integration | -- | ~250 |
| 2 | Swarm Pattern | Invocation state | ~400 |
| 3 | Agent Graph | -- | ~350 |
| 4 | OTEL Observability | -- | ~300 |
| 5 | Dependency-Based Parallelism | -- | ~200 |
| 6 | Agent SOPs | -- | ~250 |
| 7 | Semantic Tool Retrieval | -- | ~200 |
| 8 | A2A Protocol | Streaming | ~300 |
| **Total** | | | **~2,250** |

---

## Features NOT Recommended

| Strands Feature | Why Skip |
|---|---|
| **Model-first philosophy** (no FSM) | FSM-LLM's 2-pass architecture IS the differentiator. Removing state machines removes the value proposition. |
| **Bidirectional audio streaming** | Niche, experimental even in Strands. Not aligned with FSM-LLM's text-first design. |
| **AWS-native deployment** (AgentCore, Lambda wrappers) | Vendor lock-in. FSM-LLM is provider-agnostic via LiteLLM. |
| **Minimal boilerplate philosophy** | FSM-LLM's explicit state definitions are a feature, not a bug -- they provide auditability and determinism that "3 lines of code" agents lack. |
