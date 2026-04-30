# fsm_llm Threat Model

> **Status**: Living document. Last revision: 2026-04-29 (v0.3.0 / 0.5.x line).
>
> This document enumerates the trust boundaries of the fsm_llm framework
> and the threats we **do** and **do not** defend against. Threat modelling
> is a discipline, not a one-shot artefact — when you change a trust
> boundary (new transport, new persistence backend, new sandbox), update
> this file in the same PR.
>
> Structure mirrors the deepagents reference (`docs/deepagents.md` →
> `THREAT_MODEL.md`).

## Table of Contents

1. [Scope](#1-scope)
2. [Assumptions](#2-assumptions)
3. [Components](#3-components)
4. [Trust Boundaries](#4-trust-boundaries)
5. [Data Flows](#5-data-flows)
6. [Threats](#6-threats)
7. [Out of Scope](#7-out-of-scope)
8. [Investigated and Dismissed](#8-investigated-and-dismissed)
9. [Revision History](#9-revision-history)

---

## 1. Scope

This document covers the **fsm_llm Python package** as installed from PyPI
or built from `src/fsm_llm/`, plus the sibling `fsm_llm_monitor` package.
(The `fsm_llm_reasoning`, `fsm_llm_workflows`, and `fsm_llm_agents`
sibling shim packages were deleted at 0.7.0; their canonical homes are
`fsm_llm.stdlib.{reasoning,workflows,agents}` inside the main package.)

**In scope**:

- The unified `Program` facade, `Executor`, `Planner`, `Oracle`, the
  λ-AST and its compiler from FSM JSON.
- `LiteLLMInterface` / `LiteLLMOracle` and the litellm-based provider
  bridge.
- Host-side handler execution via `HandlerSystem`.
- Persistence via `FileSessionStore`.
- Prompt assembly and the secret/PII filtering done in
  `dialog/prompts.py`.
- The optional `fsm_llm_monitor` web dashboard.

**Out of scope** (covered by upstream / your deployment):

- The security of LLM providers themselves (OpenAI, Anthropic, Ollama,
  etc.). We trust litellm's connections to them; we do not audit their
  servers.
- The Python interpreter, the OS, or your container/orchestrator.
- Network transport layer security beyond what `httpx` / `litellm`
  configure on your behalf.
- Any third-party tools, MCP servers, or A2A peers you wire in.

The scope is intentionally narrow: fsm_llm is a **library**, not a
service. You — the integrator — own the trust boundary at process
ingress.

---

## 2. Assumptions

We make the following assumptions explicitly. If any is false **in your
deployment**, the corresponding mitigations may not hold.

1. **Trusted operator code.** Code that constructs `Program` instances,
   registers handlers, defines FSM JSON, or authors λ-DSL programs is
   written by the integrator and is itself trusted. A malicious operator
   can trivially bypass every mitigation in this document — we do not
   defend against the framework's own caller.

2. **Untrusted user input.** End-user messages passed via
   `Program.invoke(message=...)` and any data extracted by classifiers
   are untrusted. They may contain prompt injection, control characters,
   or attempts to mutate context state.

3. **Semi-trusted LLM output.** LLM responses are treated as **partially
   untrusted**: structured outputs are validated against Pydantic
   schemas (`LiteLLMOracle._invoke_structured`,
   `src/fsm_llm/runtime/oracle.py:449`); free-text responses are NOT
   sanitised by the framework before being returned to your application.

4. **Filesystem write access where requested.** `FileSessionStore` is
   given a directory by the operator. We assume that directory is on a
   filesystem the operator controls, with permissions appropriate to the
   data sensitivity.

5. **Network egress is permitted to configured providers.** litellm
   reaches out to provider HTTPS endpoints. We assume your deployment
   either allows that egress (`LiteLLMInterface._make_llm_call`,
   `src/fsm_llm/runtime/_litellm.py:391`) or blocks it at firewall level
   — we do not enforce egress policy.

6. **Single-tenant process by default.** `_DEDUPE_REGISTRY` in
   `src/fsm_llm/_api/deprecation.py` and the module-level handler /
   profile registries are process-local. Multi-tenant deployments
   sharing one Python process across security domains are **not
   supported**.

---

## 3. Components

The framework decomposes into the following load-bearing components.
Each is a node in the threat-flow diagram in §5.

| Component | File / Module | Role |
|-----------|---------------|------|
| **Program facade** | `src/fsm_llm/program.py` | Public entrypoint; mode-invariant adapter over FSM dialog and λ-term execution. |
| **API / FSMManager** | `src/fsm_llm/dialog/api.py`, `src/fsm_llm/dialog/manager.py` | Stateful dialog loop. Holds session keys and per-conversation context. |
| **Compile pipeline** | `src/fsm_llm/dialog/compile_fsm.py` | Lowers FSM JSON to a typed λ-term at construction. Validates schema. |
| **Executor** | `src/fsm_llm/runtime/executor.py` | β-reduces a `Term`. Calls the `Oracle` for `Leaf` nodes. Depth-bounded. |
| **Planner** | `src/fsm_llm/runtime/planner.py` | Computes `predicted_calls` (Theorem-2). |
| **Oracle / LiteLLMOracle** | `src/fsm_llm/runtime/oracle.py` | Contract for "ask the model." Concrete implementation calls litellm. |
| **LLMInterface / LiteLLMInterface** | `src/fsm_llm/runtime/_litellm.py` | Provider-side wrapper. Owns model name, kwargs, and the actual `litellm.completion(...)` call (`:391`). |
| **Prompt builders** | `src/fsm_llm/dialog/prompts.py` | Assemble system prompts. Apply `_escape_format_braces` (`:73`) and the secret/PII filter loop (`:323`). |
| **Constants** | `src/fsm_llm/constants.py` | `INTERNAL_KEY_PREFIXES` (`:13`), `FORBIDDEN_CONTEXT_PATTERNS` (`:110`), `COMPILED_FORBIDDEN_CONTEXT_PATTERNS` (`:121`). |
| **HandlerSystem** | `src/fsm_llm/handlers.py:246`, `:291` | Executes operator-supplied Python callables at 8 timing points. Trust boundary into operator code. |
| **FileSessionStore** | `src/fsm_llm/dialog/session.py:111` | Atomic temp-write + `os.replace` (`:144`). |
| **Deprecation machinery** | `src/fsm_llm/_api/deprecation.py` | Process-local dedupe registry. |
| **Monitor (optional)** | `src/fsm_llm_monitor/` | FastAPI dashboard. Bound to a local interface by the operator. |
| **MCP / A2A (optional)** | `src/fsm_llm/stdlib/agents/mcp.py`, `src/fsm_llm/stdlib/agents/a2a.py` | External tool / peer transports. |

---

## 4. Trust Boundaries

We identify the following boundaries. Each is a place where data
crosses from one trust zone into another and where validation /
sanitisation is required.

### TB-1: Operator code ↔ Framework

- **Direction**: bidirectional.
- **Crossing**: `Program.from_fsm` / `from_term` / `from_factory`,
  handler registration, FSM JSON loading, λ-DSL term construction.
- **Trust gradient**: even.
- **Defenses**: schema validation on FSM JSON
  (`compile_fsm`), Pydantic frozen models for the AST, layered
  `__all__` partitioning to discourage importing private names.

### TB-2: End-user input ↔ Framework

- **Direction**: untrusted → framework.
- **Crossing**: `Program.invoke(message=...)`, classifier outputs being
  written into context.
- **Defenses**:
  - Internal key prefix filter (`INTERNAL_KEY_PREFIXES`,
    `src/fsm_llm/constants.py:13`) prevents user data from polluting
    framework-reserved keys.
  - Forbidden-pattern regex filter (`FORBIDDEN_CONTEXT_PATTERNS`,
    `:110`; compiled at `:121`) drops candidate context keys whose names
    match credential / secret patterns.
  - `_escape_format_braces` (`src/fsm_llm/dialog/prompts.py:73`)
    prevents `{...}` injection from breaking format strings during
    prompt assembly.

### TB-3: Framework ↔ LLM provider (network)

- **Direction**: bidirectional, network.
- **Crossing**: `LiteLLMInterface._make_llm_call`
  (`src/fsm_llm/runtime/_litellm.py:391`),
  `LiteLLMOracle._invoke_structured`
  (`src/fsm_llm/runtime/oracle.py:449`).
- **Defenses**:
  - Structured-output path validates against a Pydantic schema before
    returning to caller.
  - Free-text path is **NOT** sanitised; downstream consumers must treat
    LLM output as semi-trusted.
  - litellm handles TLS and provider authentication; we do not pin
    certificates ourselves.
- **Residual risk**: a compromised provider (or MITM with valid TLS)
  can return arbitrary content. Pydantic schema validation rejects
  structurally invalid output but does not catch semantic adversarial
  content.

### TB-4: Framework ↔ Filesystem

- **Direction**: bidirectional.
- **Crossing**: `FileSessionStore`
  (`src/fsm_llm/dialog/session.py:111`), session writes via
  `os.replace` (`:144`).
- **Defenses**:
  - Atomic temp-then-rename prevents partial writes from corrupting
    session state.
  - Operator chooses the directory; framework does not perform
    `os.path.expanduser` or trust environment-variable paths.
- **Residual risk**: the framework does not enforce path-traversal
  protection on `conversation_id`. **The operator must treat
  `conversation_id` as a trusted identifier** (e.g., a server-issued
  UUID, not an end-user-controlled string). See §6 T-04.

### TB-5: Framework ↔ Operator-supplied handlers

- **Direction**: framework → operator code.
- **Crossing**: `HandlerSystem.execute_handlers`
  (`src/fsm_llm/handlers.py:291`).
- **Trust gradient**: handlers run in-process with full Python
  privileges — they are **operator code**, see TB-1.
- **Defenses**:
  - Handler errors raise `HandlerExecutionError` (`:218`); they do not
    silently corrupt state.
  - The two AST-side timings (`PRE_PROCESSING`, `POST_PROCESSING`) flow
    through `compose` and inherit Theorem-2 cost accounting; the other
    six remain host-side per `docs/lambda_fsm_merge.md` §8.
- **Residual risk**: handlers can do anything Python can do. They are
  not sandboxed.

### TB-6: Framework ↔ MCP servers and A2A peers (optional extras)

- **Direction**: bidirectional, network.
- **Crossing**: `fsm_llm[mcp]` and `fsm_llm[a2a]` extras (not installed
  by default).
- **Defenses**: tool-output validation done by the agent layer (Pydantic
  schemas on results); transport-level security is the responsibility
  of the MCP / A2A library you choose.

### TB-7: Framework ↔ Monitor dashboard (optional extra)

- **Direction**: bidirectional, network (HTTP).
- **Crossing**: `fsm_llm[monitor]` — FastAPI app served by uvicorn.
- **Defenses**: none enabled by default. The dashboard binds to a
  loopback interface unless the operator configures otherwise.
- **Residual risk**: if the operator binds the dashboard to a public
  interface without authentication, every conversation, plan, and leaf
  call is observable. **The dashboard has no auth layer.** Treat it as
  a debugging aid in trusted environments only.

---

## 5. Data Flows

```text
                +-----------------+
                |  operator code  |  TB-1
                +--------+--------+
                         |
                         v
   end-user msg   +------+------+   handler callbacks    +-----------+
   (untrusted) -->|             |<---------------------->|  handlers |
        TB-2     |  Program /  |        TB-5             | (op code) |
                 |  API +      |                         +-----------+
                 |  FSMManager |
                 |             |--- session read/write -->+----------+
                 |             |        TB-4              | filesys  |
                 +------+------+                          +----------+
                        | compile_fsm
                        v
                 +-------------+      LeafCall
                 | Term (AST)  |---->-----+
                 +-------------+          |
                                          v
                                    +-----------+   HTTPS    +----------+
                                    |  Oracle / |<---------->| provider |
                                    | LiteLLM   |   TB-3     | (LLM)    |
                                    +-----------+            +----------+
```

Optional flows (extras):

```text
   +---------+    HTTPS    +-------------+
   | Program |<----------->| MCP server  |   TB-6
   +---------+             +-------------+

   +---------+    HTTP     +-------------+
   | runtime |<----------->| Monitor UI  |   TB-7
   +---------+             +-------------+
```

---

## 6. Threats

The threats below are numbered T-NN and reference the boundary they
cross.

### T-01 — Prompt injection via end-user message (TB-2)

**Vector**: a user crafts a message that re-instructs the model to
ignore its system prompt, leak context keys, or call a tool out of
turn.

**Defenses**:

- We do not strip prompt-injection keywords from user content (doing so
  reliably is an open research problem). Instead:
  - Structured-output Leaves use Pydantic schemas — semantic deviations
    by the model surface as `OracleError` rather than silent state
    pollution.
  - `INTERNAL_KEY_PREFIXES` (`src/fsm_llm/constants.py:13`) prevents
    extracted user data from overwriting `_`/`system_`/`internal_`/`__`
    keys.

**Residual risk**: free-text Leaf outputs and conversational responses
that downstream code consumes verbatim. Treat all conversational LLM
output as untrusted text — never `eval`, `exec`, or shell-interpolate
it.

### T-02 — Secrets exfiltrated through context keys (TB-2)

**Vector**: an integrator writes secrets into the context dict by
mistake; a future extraction step renders them into a prompt.

**Defenses**:

- `FORBIDDEN_CONTEXT_PATTERNS` (`src/fsm_llm/constants.py:110`,
  compiled at `:121`) is a regex blocklist for keys matching common
  secret patterns (`password`, `secret`, `token`, etc.). The compiled
  patterns are consulted in the prompt builder loop at
  `src/fsm_llm/dialog/prompts.py:323`.

**Residual risk**: blocklists are not exhaustive. Operators must not
write secrets into context, period. The filter is defense in depth, not
the primary defense.

### T-03 — Format-string injection during prompt assembly (TB-2)

**Vector**: a user message containing `{some_internal_key}` interpolates
context state into the rendered prompt.

**Defenses**: `_escape_format_braces`
(`src/fsm_llm/dialog/prompts.py:73`) doubles all literal braces in
user-supplied content before format-string rendering. Applied at every
`.format(...)` site in the builders (`:582`, `:943`, `:1469`, `:1500`,
`:1541`).

### T-04 — Path traversal via `conversation_id` (TB-4)

**Vector**: an end-user-controlled `conversation_id` containing `..`
escapes the session directory.

**Defenses**: **NONE in framework.** `FileSessionStore` builds the path
by concatenating its directory and the `conversation_id` (see
`src/fsm_llm/dialog/session.py:111`).

**Mitigation (operator's responsibility)**: never pass an
end-user-controlled `conversation_id` directly to `Program.invoke`.
Generate it server-side (UUID4 is sufficient) and map untrusted user
identifiers to the trusted ID via your own table.

### T-05 — Concurrent-write corruption of session files (TB-4)

**Vector**: two processes writing the same conversation simultaneously.

**Defenses**: atomic temp-then-rename via `os.replace`
(`src/fsm_llm/dialog/session.py:144`). Last-write-wins; no torn writes.

**Residual risk**: lost updates if two processes mutate concurrently.
For multi-process deployments, use an external session store (Redis,
SQL) — `FileSessionStore` is intended for single-process or
single-worker setups.

### T-06 — Unbounded LLM cost (TB-3)

**Vector**: a runaway recursion or deeply nested `Fix` blows past the
expected token budget.

**Defenses**: Theorem-2 — `Executor.run(...).oracle_calls` is statically
predicted by the planner for τ·k^d-aligned input. Inputs that diverge
from alignment surface as oracle-count drift, observable via
`Result.oracle_calls`. Operators can set hard ceilings via
`max_oracle_calls` on the Executor.

### T-07 — Provider response injection (TB-3)

**Vector**: a compromised provider (or upstream proxy) returns
adversarial structured output that conforms to the declared schema but
contains malicious payloads (URLs, code snippets, etc.).

**Defenses**: Pydantic schema validation (`_invoke_structured` at
`src/fsm_llm/runtime/oracle.py:449`) catches structurally invalid
output. **Semantic content is not validated.**

**Mitigation**: as with T-01, treat LLM output as semi-trusted.

### T-08 — Handler privilege escalation (TB-5)

**Vector**: a handler shipped by a third-party plugin reads or mutates
state it shouldn't.

**Defenses**: NONE — handlers run with full Python privileges. The
trust gradient is even (TB-5 is operator code). If you load handlers
from third-party plugins, you have inherited their security posture.

### T-09 — Information disclosure via Monitor (TB-7)

**Vector**: monitor dashboard bound to a public network interface
exposes conversation history, plans, and LLM responses to anyone who
can reach the port.

**Defenses**: **NONE in framework.** The dashboard has no
authentication.

**Mitigation**: do not expose the monitor port. Bind it to loopback or
a wireguard interface. If you need remote access, terminate auth at a
reverse proxy.

### T-10 — Dependency confusion / supply chain (operator deployment)

**Vector**: a malicious package named `fsm-llm` (with a hyphen) or
typo-squatting one of the historical sibling-shim names is published to
a private index.

**Defenses**: NONE in framework. The remaining sibling package after
the 0.7.0 cleanup is `fsm_llm_monitor`; the canonical project is
`fsm-llm` on PyPI under the GitHub project `NikolasMarkou/fsm_llm`.
The `fsm_llm_{reasoning,workflows,agents}` namespaces were retired
in 0.7.0 — third parties publishing those names today carry zero
sanctioned content.

**Mitigation**: pin versions in `constraints.txt`; use a single index
or carefully ordered indexes.

### T-11 — Stale deprecation cache hiding security-relevant warnings

**Vector**: `_DEDUPE_REGISTRY` in
`src/fsm_llm/_api/deprecation.py` suppresses the same
`(name, since, removal)` triple after the first emission. A long-running
process never re-sees the warning.

**Defenses**: dedupe is bounded to a process; restarting clears it.
`reset_deprecation_dedupe()` is exposed for tests but also as a deliberate
operator escape hatch.

**Residual risk**: low — `DeprecationWarning` is intentionally
quiet-after-once. Genuinely security-critical signals belong in
exceptions, not warnings.

---

## 7. Out of Scope

We **do not** defend against:

- Sandboxing operator handlers, λ-term hooks, or stdlib factories.
  Handlers are operator code; if you need sandboxing, run the whole
  process in a container with seccomp / gVisor / similar.
- Denial of service from end-user input. `max_oracle_calls`,
  `max_depth`, and Theorem-2 give you handles, but rate-limiting is
  the operator's job.
- Cryptographic protections on session storage. `FileSessionStore`
  writes plaintext JSON; if you need encryption at rest, use full-disk
  encryption or a different store.
- Side-channel leakage between simultaneous `Program.invoke` calls in
  the same process. Executors are independent, but LLM provider
  caching, prompt-cache key collisions, and CPU-side caches are out of
  scope.
- Multi-tenancy. The handler / profile / dedupe registries are
  process-global. Run one tenant per process.

---

## 8. Investigated and Dismissed

The following ideas were considered and consciously not implemented.
Listed here so future contributors do not re-litigate them silently.

### ID-01: Sandbox handler execution

**Considered**: running operator handlers in a `multiprocessing.Process`
or via `restrictedpython`.

**Dismissed because**: handlers commonly need to read application
state, call external services, and update databases. A useful sandbox
would have to expose so much of the parent process's state that it
provides little real isolation. **If you need a sandbox, the right
boundary is the OS / container, not Python.**

### ID-02: Path-traversal protection in `FileSessionStore`

**Considered**: rejecting `conversation_id` containing `/`, `..`, or
non-printable characters.

**Dismissed because**: this is necessary but not sufficient — what
counts as a safe identifier depends on the operator's namespacing
conventions. Documenting "treat `conversation_id` as a trusted
identifier" (T-04) is more honest than a half-measure that lulls
operators into writing user-controlled IDs. **A future revision MAY
add an opt-in `safe_conversation_ids=True` flag** that does enforce a
tight whitelist.

### ID-03: Strip prompt-injection keywords from user input

**Considered**: a regex filter that drops phrases like "ignore previous
instructions," "you are now …," etc.

**Dismissed because**: trivially bypassable; gives a false sense of
security. The framework's actual defense against prompt injection is
**structured-output schemas** (which constrain what state user input
can mutate), not input filtering.

### ID-04: Built-in auth for the Monitor dashboard

**Considered**: shipping a basic-auth or OAuth proxy in
`fsm_llm_monitor`.

**Dismissed because**: auth is the reverse-proxy's job. Shipping our
own basic-auth would be either insecure (timing attacks, weak password
storage) or a maintenance burden. Operators already have
nginx/Caddy/Traefik.

### ID-05: TLS certificate pinning for litellm

**Considered**: forcing certificate pins on provider connections.

**Dismissed because**: providers rotate certificates routinely; pinning
breaks more often than it helps. Operators who want pinning configure
it at the OS or container level.

### ID-06: Encrypt `FileSessionStore` payloads at rest

**Considered**: a transparent encryption layer over the JSON payloads.

**Dismissed because**: key management is hard and operator-specific.
The right answer is full-disk encryption or a separate
`EncryptedSessionStore` implementation that operators write themselves
against the `SessionStore` interface.

---

## 9. Revision History

| Date       | Version  | Author          | Change |
|------------|----------|-----------------|--------|
| 2026-04-29 | 0.3.0    | initial         | Initial threat model. Covers `Program`, `Executor`, `Oracle`, `HandlerSystem`, `FileSessionStore`, `Monitor`. Documents 11 threats, 6 trust boundaries, 6 dismissed proposals. |

---

## How to update this document

When you change any of the following, update §3, §4, and §6 in the
**same PR**:

- A new component is added (new oracle, new transport, new persistence
  backend).
- A trust boundary moves (e.g., handlers gain sandboxing, a session
  store gains encryption).
- A new threat is identified (regression-test it; add a row to §6).
- A previously-dismissed proposal is implemented (move it from §8 to
  §6 with the mitigation it now provides).

For threats that affect Theorem-2 cost accounting, also update
`docs/lambda_fsm_merge.md` §8.
