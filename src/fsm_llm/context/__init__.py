"""
Context management for FSM-LLM (layer ``context``).

The package is the one import surface of the layer (``from fsm_llm.context import
ContextCompactor``); each name is defined in the module named in ``module_path_map.json`` and
re-exported here, so a patch or a private-name import aims at the defining module:

- ``compactor``: ``ContextCompactor`` and ``clean_context_keys`` (the old ``fsm_llm.context``);
- ``commit``: ``clear_keys_delta`` (moved down from ``fsm_llm.handlers``, step 74 of plan
  d454fd62, so this layer imports no engine module), ``commit_with_key_rollback``,
  ``was_cut_off`` and ``record_cut_off``;
- ``filtering``: the depth-bounded context-tree filter and the redaction walk (not re-exported
  here: import ``fsm_llm.context.filtering``);
- ``layout``: the prompt builders (the old ``fsm_llm.prompts``).
"""

from .commit import (
    clear_keys_delta,
    commit_with_key_rollback,
    record_cut_off,
    was_cut_off,
)
from .compactor import ContextCompactor, clean_context_keys

__all__ = [
    "ContextCompactor",
    "clean_context_keys",
    "clear_keys_delta",
    "commit_with_key_rollback",
    "record_cut_off",
    "was_cut_off",
]
