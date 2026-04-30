# Strands → FSM-LLM Feature Adaptation Report

## Executive Summary

After analyzing Strands Agents SDK against the FSM-LLM codebase (5 packages, 12 agent patterns, 11 workflow step types, custom monitoring), we identified **12 features** worth adapting. These are ordered by enhancement value, not implementation difficulty.

---

## 1. OpenTelemetry (OTEL) Observability

**Strands has**: Native OTEL spans for model inferences, tool invocations, agent handoffs, planner steps. Token usage, latency, and error rates as structured metrics. Pluggable backends (Jaeger, Datadog, X-Ray, Langfuse).

**FSM-LLM has**: Custom `EventCollector` with bounded in-memory deques, 15 event types, WebSocket streaming, and a web dashboard. No OTEL. No distributed tracing. No correlation IDs across multi-agent calls.

**Why adapt**: Our monitor already captures the right events (state transitions, tool calls, handler execution, agent lifecycle). But it's a closed system -- data can't flow to production observability stacks. OTEL would let users plug FSM-LLM into existing Grafana/Datadog/Jaeger infrastructure without replacing our dashboard.

**What to build**: An OTEL exporter layer that wraps existing `EventCollector` events into OTEL spans and metrics. Instrument `LiteLLMInterface.generate_response()` and `extract_field()` with spans capturing model params, token usage, and latency. Instrument `ToolRegistry.execute()` with tool invocation spans. Add trace context propagation through `AgentResult.final_context` for multi-agent correlation.

---

## 2. Token-Level Response Streaming

**Strands has**: `agent.stream()` iterator yielding chunks in real-time. Bidirectional audio streaming (experimental).

**FSM-LLM has**: Zero streaming. `LiteLLMInterface` uses synchronous `completion()`. All agents run to completion. The 2-pass architecture (extract → transition → respond) means Pass 1 must complete before Pass 2 starts.

**Why adapt**: Streaming is table-stakes for production chat applications. Users see nothing until both passes complete. For Pass 2 (response generation), there's no architectural reason to wait for the full response before yielding tokens. LiteLLM already supports streaming natively.

**What to build**: Add `stream=True` support to `LiteLLMInterface.generate_response()` for Pass 2 only (Pass 1 must complete fully for extraction). Expose `API.converse_stream()` that yields response tokens. For agents, expose `BaseAgent.run_stream()` that streams the final conclude-state response.

---

## 3. Swarm Pattern (Emergent Agent Coordination)

**Strands has**: `Swarm` class with named agents, agent-driven handoffs via structured response (`agentId`, `message`, `context`), shared working memory (`invocation_state`), and an `entry_agent`.

**FSM-LLM has**: `OrchestratorAgent` (hierarchical delegation), `DebateAgent` (structured multi-agent), agents-as-tools via `ToolRegistry.register_agent()`. No emergent/swarm coordination where agents decide handoff targets dynamically.

**Why adapt**: The orchestrator pattern requires a central coordinator that plans all delegation. Swarms are better for open-ended tasks where the optimal agent sequence isn't known upfront (e.g., research → write → edit pipelines). FSM-LLM's existing `WorkingMemory` buffers (core, scratch, environment, reasoning) are a natural fit for swarm shared state.

**What to build**: A `SwarmAgent` pattern (pattern #13) where each agent's conclude state includes a `next_agent` classification extraction. Agents hand off by returning the next agent ID + message + context. The swarm runner loops until an agent returns no `next_agent`. Shared state flows through `WorkingMemory` passed to all agents.

---

## 4. Graph-Based Agent Orchestration (Deterministic DAG with Conditional Edges)

**Strands has**: `GraphBuilder` with `add_node()`, `add_edge(condition=fn)`, `set_entry_node()`, `build()`. Supports both DAG and cyclic topologies. Conditional routing via Python functions on state.

**FSM-LLM has**: Workflows have DAG validation and state-based transitions, but they're step-typed (APICallStep, LLMProcessingStep, etc.), not agent-typed. No way to wire agents into a graph with conditional edges and have agent outputs drive routing decisions.

**Why adapt**: Our workflow engine already has the infrastructure (DAG validation, cycle detection, parallel execution, conditional routing). But it operates at the step level, not the agent level. A graph of agents with conditional routing would enable patterns like: analyzer → (critical_handler | normal_handler) → reporter, where routing depends on the analyzer's output.

**What to build**: An `AgentGraph` that uses the existing workflow engine infrastructure but treats agents as nodes. Each node runs an agent, its `AgentResult.final_context` becomes the edge state, and conditional edge functions evaluate against that state. This bridges our workflow engine and agent system.

---

## 5. MCP (Model Context Protocol) Tool Integration

**Strands has**: First-class MCP support via `MCPClient`. Loads tools from any MCP server (stdio or HTTP). Tools appear as native tools to the agent.

**FSM-LLM has**: No MCP support. All tools are native via `@tool` decorator and `ToolRegistry`.

**Why adapt**: MCP is becoming the universal tool protocol. Thousands of MCP servers exist for databases, APIs, file systems, cloud services. Supporting MCP would instantly expand FSM-LLM's tool ecosystem without writing custom integrations.

**What to build**: An `MCPToolProvider` that connects to an MCP server and converts its tool definitions into `ToolDefinition` objects compatible with `ToolRegistry`. Users would do: `registry.register_mcp(server_params)` and all MCP tools become available to agents. The `@tool` decorator remains for custom tools.

---

## 6. Semantic Tool Retrieval (Large Tool Sets)

**Strands has**: For 6,000+ tool registries, uses embedding-based semantic search to find relevant tools per query, only describing the top-K to the model.

**FSM-LLM has**: `ToolRegistry.to_classification_schema()` converts tools to a `ClassificationSchema` for LLM-based intent classification. `to_prompt_description()` generates text descriptions. No embedding-based retrieval. All registered tools are described to the model.

**Why adapt**: Our `to_classification_schema()` approach works well for <20 tools but doesn't scale. With large registries, the classification prompt becomes enormous. Semantic retrieval would let users register hundreds of tools while only presenting the relevant 5-10 per query.

**What to build**: A `SemanticToolRegistry` subclass that embeds tool descriptions at registration time and retrieves top-K tools per query using cosine similarity. Falls back to full registry for small tool sets. Integrates with existing `ToolRegistry` API.

---

## 7. Dependency-Based Workflow Parallelism

**Strands has**: Explicit `workflow.add_dependency("B", "A")` declarations. Independent branches auto-parallelize. Task graph with topological ordering.

**FSM-LLM has**: `ParallelStep` runs hardcoded parallel branches via `asyncio.gather`. State-based transitions (A→B, A→C). No explicit dependency declarations. No automatic parallelism detection from dependency graph.

**Why adapt**: Our `ParallelStep` requires developers to manually group parallel work. Strands' approach is more natural -- declare what depends on what, and the engine figures out what can run in parallel. Our workflow engine already has DAG validation and async execution; it just lacks the dependency resolver.

**What to build**: Add `WorkflowDefinition.add_dependency(step_id, depends_on)` that builds a dependency graph. A `DependencyResolver` computes execution waves (topological sort into parallel groups). The engine runs each wave's steps in parallel via existing `asyncio.gather`, waiting for all dependencies before starting the next wave.

---

## 8. Agent SOPs (Standard Operating Procedures)

**Strands has**: `strands-agents-sop` package with loadable natural-language workflow templates (`code-assist`, `codebase-summary`, `pdd`). SOPs become structured system prompts.

**FSM-LLM has**: Agents have `persona` and pattern-specific prompts. The meta builder can generate artifacts interactively. No reusable SOP/template system.

**Why adapt**: SOPs are essentially parameterized prompt templates that encode best practices for common tasks. FSM-LLM's meta builder already proves the value of guided artifact creation. SOPs would let users share and reuse proven agent configurations without the meta builder's interactive loop.

**What to build**: An `SOPRegistry` that loads YAML/JSON SOP definitions (task description, required tools, prompt template, output schema). `Agent.from_sop("code-review", tools=[...])` would configure an agent with the SOP's prompt, tools, and output expectations. Ship a few built-in SOPs for common patterns.

---

## 9. Session Persistence (Cross-Invocation State)

**Strands has**: `SessionManager(session_id="user-123")` for persisting conversation state across invocations. Community packages for Redis/Valkey backends.

**FSM-LLM has**: In-memory `Conversation` objects with `max_history_size`. FSM stacking with context merge. `WorkingMemory` buffers. No persistence across process restarts. `cleanup_stale_conversations()` implies sessions are ephemeral.

**Why adapt**: Any production deployment needs conversation persistence. Users currently lose all conversation history when the process restarts. The FSM stacking and context system are well-designed for in-memory use, but need a persistence layer.

**What to build**: A `SessionStore` interface with `save(conversation_id, state)` / `load(conversation_id)` methods. State includes: conversation history, current FSM state, context dict, working memory buffers, FSM stack. Implementations: `FileSessionStore` (JSON files), `RedisSessionStore`. Plug into `API` with `session_store=` parameter.

---

## 10. A2A (Agent-to-Agent Protocol) for Remote Agents

**Strands has**: `A2AServer` exposes agents as HTTP endpoints. `A2AClient` calls remote agents. Enables cross-framework, cross-process agent communication.

**FSM-LLM has**: Agents communicate via synchronous method calls and shared `WorkingMemory`. `register_agent()` wraps agents as local tools. Monitor server has REST endpoints but for monitoring, not agent invocation.

**Why adapt**: For microservice architectures and distributed systems, agents need to communicate across processes and machines. Our monitor's FastAPI server already demonstrates the HTTP serving pattern. A2A would let FSM-LLM agents call agents in other frameworks (or vice versa).

**What to build**: An `AgentServer` (FastAPI) that wraps any FSM-LLM agent as an HTTP endpoint (`/invoke`, `/stream`). An `RemoteAgentTool` that wraps a remote agent URL as a `ToolDefinition` for local agents. Uses existing `ToolRegistry.register_function()` pattern.

---

## 11. Invocation State (Hidden Multi-Agent Shared State)

**Strands has**: `invocation_state` dict passed to multi-agent patterns, shared across all agents but NOT exposed to the LLM. Used for metadata like `user_tier`, `region`, etc.

**FSM-LLM has**: `WorkingMemory` with named buffers, but all buffers can end up in LLM context. `initial_context` dict is passed to agents and visible to the LLM via extraction instructions. No clean separation between "agent infrastructure state" and "LLM-visible state".

**Why adapt**: We need a way to pass orchestration metadata (user permissions, retry counts, routing decisions, billing tier) through multi-agent pipelines without polluting LLM context. Our `WorkingMemory.scratch` buffer is close but gets included in prompts.

**What to build**: Add a `metadata` buffer to `WorkingMemory` that is explicitly excluded from prompt construction. Or simpler: add `AgentConfig.invocation_state: dict` that flows through `AgentResult.final_context` under a reserved `_invocation` prefix (already stripped by `clean_context_keys()`'s internal prefix filter).

---

## 12. Pydantic-Schema Output Enforcement at LLM Level

**Strands has**: `Agent(output_schema=ResearchReport)` where the result is a typed Pydantic instance. The schema is enforced at the model level (constrained decoding / JSON mode).

**FSM-LLM has**: `AgentConfig.output_schema` with 3-tier fallback validation (context keys → JSON parsing → scan observations). Works but is post-hoc -- the LLM generates free text, then FSM-LLM tries to extract structured data from it.

**Why adapt**: Our 3-tier fallback is robust but lossy. When the LLM doesn't produce JSON-like output, validation fails silently (returns None). LiteLLM supports `response_format` with JSON schema for many providers, which would guarantee valid JSON output.

**What to build**: When `output_schema` is set, pass `response_format={"type": "json_schema", "json_schema": schema.model_json_schema()}` to LiteLLM's `completion()` call for the conclude state's Pass 2. Keep the 3-tier fallback as a safety net for providers that don't support constrained output.

---

## Summary Matrix

| # | Feature | Enhancement Value | What FSM-LLM Has Today | Gap |
|---|---------|------------------|------------------------|-----|
| 1 | OTEL Observability | Production readiness | Custom EventCollector + dashboard | No industry-standard telemetry |
| 2 | Response Streaming | UX (chat apps) | None | Complete gap |
| 3 | Swarm Pattern | New multi-agent capability | Orchestrator (hierarchical only) | No emergent coordination |
| 4 | Agent Graph | Deterministic multi-agent routing | Workflow DAG (step-level, not agent-level) | No agent-as-node graphs |
| 5 | MCP Tools | Ecosystem expansion | Native @tool only | No external tool protocol |
| 6 | Semantic Tool Retrieval | Scalability | Classification-based selection | Doesn't scale past ~20 tools |
| 7 | Dependency-Based Parallelism | Workflow ergonomics | Manual ParallelStep | No auto-parallelization |
| 8 | Agent SOPs | Reusability | Meta builder (interactive) | No static template system |
| 9 | Session Persistence | Production readiness | In-memory only | No cross-restart state |
| 10 | A2A Protocol | Distributed agents | Local method calls only | No remote agent communication |
| 11 | Invocation State | Multi-agent metadata | WorkingMemory (LLM-visible) | No hidden infrastructure state |
| 12 | Schema-Enforced Output | Reliability | Post-hoc 3-tier fallback | No constrained decoding |

---

## Features NOT Recommended

| Strands Feature | Why Skip |
|---|---|
| **Model-first philosophy** (no FSM) | FSM-LLM's 2-pass architecture IS the differentiator. Removing state machines removes the value proposition. |
| **Bidirectional audio streaming** | Niche, experimental even in Strands. Not aligned with FSM-LLM's text-first design. |
| **AWS-native deployment** (AgentCore, Lambda wrappers) | Vendor lock-in. FSM-LLM is provider-agnostic via LiteLLM. |
| **Minimal boilerplate philosophy** | FSM-LLM's explicit state definitions are a feature, not a bug -- they provide auditability and determinism that "3 lines of code" agents lack. |
