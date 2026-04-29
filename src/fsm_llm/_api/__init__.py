"""Private API package for fsm_llm.

The leading underscore signals "internal — not part of the public surface."
Names under :mod:`fsm_llm._api` are NOT exported via ``fsm_llm.__all__`` and
have no compatibility guarantees across minor releases.

Currently houses:

* :mod:`fsm_llm._api.deprecation` — shared ``warn_deprecated`` formatter and
  per-target dedupe registry, used by R13/I5 deprecation surfaces and tests.

Convention mirrors the deepagents reference: ship deprecation MACHINERY
ahead of the warnings themselves so that v0.6.0 release work only needs to
flip call-sites, not introduce infrastructure.
"""

__all__: list[str] = []
