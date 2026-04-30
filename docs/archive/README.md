# Archived Documentation

Historical documents preserved for context. None of these reflect the
current API or architecture; they are kept verbatim as a record of the
decision history.

## Index

| File | Original purpose | Archived in |
|---|---|---|
| [`lambda_integration.md`](lambda_integration.md) | v2.0 audit-and-plan from pre-M1, before the λ-calculus runtime shipped. Superseded by `docs/lambda.md` (the architectural thesis) and `docs/lambda_fsm_merge.md` (the canonical merge contract). | 0.5.0 |
| [`strands_features.md`](strands_features.md) | Aspirational feature analysis adapting 12 patterns from the Strands SDK (OTEL, streaming, swarm, graph, MCP, semantic tools, dependency parallelism, SOPs). Phase 1/2 implementation logs below. None of the 12 features shipped through 0.7.0. | 0.7.0 |
| [`strands_features_phase_1.md`](strands_features_phase_1.md) | Phase 1 implementation notes (3 features: OTEL exporter, token streaming, swarm pattern). Deferred. | 0.7.0 |
| [`strands_features_phase_2.md`](strands_features_phase_2.md) | Phase 2 implementation notes (3 features: graph orchestration, MCP integration, semantic tools). Deferred. | 0.7.0 |

## Current canonical docs

For the live architecture, API, and roadmap, see:

- [`../README.md`](../../README.md) — public-facing overview.
- [`../lambda_fsm_merge.md`](../lambda_fsm_merge.md) — the merge contract.
- [`../lambda.md`](../lambda.md) — the architectural thesis.
- [`../api_reference.md`](../api_reference.md) — the API surface.
- [`../architecture.md`](../architecture.md) — system design.
- [`../../CHANGELOG.md`](../../CHANGELOG.md) — release notes.
