"""Context-key security filters for the FSM-LLM framework.

Holds the internal-key prefix test (`has_internal_prefix`) and the
secret-looking context-entry filter (`is_forbidden_context_entry` and its
name, value-shape and vocabulary layers). These are the single decision
points every context filter imports; never re-inline either check.

Also holds the shared path-confinement check (`resolve_within`) and the
API-key compare of the two HTTP servers (`presented_api_key`,
`api_key_matches`, `checked_api_key`).

This package imports only the standard library, never `fsm_llm.constants` or any
other `fsm_llm` module. `fsm_llm.constants` re-exports five public names from here
(`FORBIDDEN_CONTEXT_PATTERNS`, `COMPILED_FORBIDDEN_CONTEXT_PATTERNS`,
`INTERNAL_KEY_PREFIXES`, `has_internal_prefix`, `is_forbidden_context_entry`).
The `__init__` re-exports the public names only (P1); a private helper is imported from
the module that defines it:

- `names`: `INTERNAL_KEY_PREFIXES`, `has_internal_prefix`, `is_forbidden_context_entry`
  and the name-scan data and value judges the decision calls.
- `paths`: `resolve_within`.
- `api_keys`: `checked_api_key`, `presented_api_key`, `api_key_matches`.
- `patterns`: layer 1 name regexes (`FORBIDDEN_CONTEXT_PATTERNS`).
- `shape_data`, `shapes`: layer 2 value-shape tables and detectors.
"""

from .api_keys import api_key_matches, checked_api_key, presented_api_key
from .names import (
    INTERNAL_KEY_PREFIXES,
    has_internal_prefix,
    is_forbidden_context_entry,
)
from .paths import resolve_within
from .patterns import COMPILED_FORBIDDEN_CONTEXT_PATTERNS, FORBIDDEN_CONTEXT_PATTERNS

__all__ = [
    "COMPILED_FORBIDDEN_CONTEXT_PATTERNS",
    "FORBIDDEN_CONTEXT_PATTERNS",
    "INTERNAL_KEY_PREFIXES",
    "api_key_matches",
    "checked_api_key",
    "has_internal_prefix",
    "is_forbidden_context_entry",
    "presented_api_key",
    "resolve_within",
]
