"""FSM definition verification: the validator, the upgrade rewrite and the flow analyses.

Layer ``core`` (plan d454fd62, step 75): the old ``fsm_llm.validator`` and
``fsm_llm.flow_analysis`` live here. This package re-exports the public names of the
validator (the one import surface, ``from fsm_llm.core.verify import FSMValidator``); a
private helper is imported from its defining module, and the console script entry points
``main`` and ``main_cli`` live in ``cli``. The flow analyses are the module ``flow``
(``strongly_connected_components``, ``analyze_data_flow``, ``analyze_liveness``,
``declared_available``, ``FlowFinding``), imported by name: ``fsm_llm.core.verify.flow``.

Modules: ``result`` (``FSMValidationResult``), ``validator`` (``FSMValidator`` and
``validate_fsm_from_file``), ``structure`` (``StructureMixin``: the structure, key and kind
checks), ``graph`` (``GraphMixin``: complexity, cycles, paths), ``upgrade``
(``upgrade_fsm_file``, ``UpgradeResult``), ``cli`` (``main``, ``main_cli``) and ``flow``.
"""

from .result import FSMValidationResult
from .upgrade import UpgradeResult, upgrade_fsm_file
from .validator import FSMValidator, validate_fsm_from_file

__all__ = [
    "FSMValidationResult",
    "FSMValidator",
    "UpgradeResult",
    "upgrade_fsm_file",
    "validate_fsm_from_file",
]
