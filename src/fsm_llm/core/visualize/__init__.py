"""FSM diagrams: the graph model (``FSMGraph``), Mermaid and DOT export, the ASCII renderer.

Layer ``core`` (plan d454fd62, step 76): the old ``fsm_llm.visualizer`` lives here. This
package re-exports the public names of the old module (the one import surface, ``from
fsm_llm.core.visualize import build_fsm_graph``); a private helper or an ASCII building
block is imported from its defining module, and the console script entry points ``main``
and ``main_cli`` live in ``cli``.

Modules: ``shapes`` (the malformed-input checks and small readers shared by every
renderer), ``ascii_layout`` (glyph tables, row fitting, state ordering, state boxes),
``ascii_render`` (``visualize_fsm_ascii`` and its three layouts), ``graph`` (``FSMGraph``,
``to_mermaid``, ``to_dot``, ``OUTPUT_FORMATS``) and ``cli`` (``visualize_fsm_from_file``,
``main``, ``main_cli``).
"""

from .ascii_layout import ASCII_STYLES
from .ascii_render import visualize_fsm_ascii
from .cli import visualize_fsm_from_file
from .graph import (
    OUTPUT_FORMATS,
    FSMGraph,
    FSMGraphEdge,
    FSMGraphNode,
    build_fsm_graph,
    to_dot,
    to_mermaid,
)

__all__ = [
    "ASCII_STYLES",
    "OUTPUT_FORMATS",
    "FSMGraph",
    "FSMGraphEdge",
    "FSMGraphNode",
    "build_fsm_graph",
    "to_dot",
    "to_mermaid",
    "visualize_fsm_ascii",
    "visualize_fsm_from_file",
]
