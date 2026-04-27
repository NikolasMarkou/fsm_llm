# fsm_llm_reasoning — Back-Compat Shim

> This package is a `sys.modules` shim. The canonical home is **`fsm_llm.stdlib.reasoning`**.

Per `docs/lambda.md` §11 (M3 reorganisation), the reasoning subsystem moved into `fsm_llm/stdlib/reasoning/`. The top-level `fsm_llm_reasoning` package is preserved as a silent back-compat shim — every legacy import keeps working, with the same module identity.

## Quick Start

```python
# Either of these works — identical objects.
from fsm_llm_reasoning import ReasoningEngine

# (Preferred for new code)
from fsm_llm.stdlib.reasoning import ReasoningEngine

engine = ReasoningEngine(model="openai/gpt-4o-mini")
solution, trace = engine.solve_problem("What is the probability of rolling two sixes?")
print(solution)
```

For λ-term factory usage (M3 slice 2 — `analytical_term`, `deductive_term`, etc.), see the canonical README.

## Installation

```bash
pip install fsm-llm[reasoning]   # Same install path
```

## Where Everything Lives Now

| What | New canonical path |
|------|--------------------|
| `ReasoningEngine` | `fsm_llm.stdlib.reasoning.engine` |
| `ReasoningType` enum, `ContextKeys` | `fsm_llm.stdlib.reasoning.constants` |
| Pydantic models (`ReasoningStep`, `ReasoningTrace`, …) | `fsm_llm.stdlib.reasoning.definitions` |
| Handlers (validate, trace, prune) | `fsm_llm.stdlib.reasoning.handlers` |
| Strategy FSMs (`ALL_REASONING_FSMS`) | `fsm_llm.stdlib.reasoning.reasoning_modes` |
| Exceptions | `fsm_llm.stdlib.reasoning.exceptions` |
| Utilities (`get_available_reasoning_types`) | `fsm_llm.stdlib.reasoning.utilities` |
| **λ-term factories** (M3 slice 2) | `fsm_llm.stdlib.reasoning.lam_factories` — 11 factories: `analytical_term`, `deductive_term`, `inductive_term`, `abductive_term`, `analogical_term`, `creative_term`, `critical_term`, `hybrid_term`, `calculator_term`, `classifier_term`, `solve_term` |

## Documentation

See **`src/fsm_llm/stdlib/reasoning/`** for:
- `README.md` — public-facing walkthrough
- `CLAUDE.md` — file map, factories, exceptions, testing

For the architectural thesis behind the move, read `docs/lambda.md` §11 (sub-package reorganisation table) and §13 (M3 milestone status).

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE).
