"""``FSMValidator``: the validation run (schema, structure, graph) and ``validate_fsm_from_file``.

Layer ``core`` (plan d454fd62, step 75): the shell of the old ``fsm_llm.validator``; the
structure checks are ``StructureMixin`` and the graph analysis ``GraphMixin``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.loaders import read_json_object

from .graph import GraphMixin
from .result import FSMValidationResult
from .structure import StructureMixin


class FSMValidator(StructureMixin, GraphMixin):
    """
    Validator for FSM definitions with detailed analysis capabilities.

    This class performs comprehensive validation and analysis of FSM definitions,
    identifying issues, potential problems, and providing structural insights.

    Attributes:
        fsm_data (dict[str, Any]): The FSM definition to validate
        fsm_name (str): Name of the FSM
        states (dict[str, Any]): States defined in the FSM
        initial_state (str): The FSM's initial state
        result (FSMValidationResult): Container for validation results
    """

    def __init__(self, fsm_data: dict[str, Any]):
        """
        Initialize the validator with FSM data.

        Args:
            fsm_data: The FSM definition as a dictionary
        """
        self.fsm_data = fsm_data
        self.fsm_name = fsm_data.get("name", "Unnamed FSM")
        self.states = fsm_data.get("states", {})
        self.initial_state = fsm_data.get("initial_state", "")
        self.result = FSMValidationResult(self.fsm_name)
        self._reachable_states_cache: set[str] | None = None
        self._definition: FSMDefinition | None = None

    def validate(self) -> FSMValidationResult:
        """
        Perform complete validation of the FSM.

        The validation process includes multiple stages:
        1. Basic structure validation
        2. Terminal state validation
        3. Required context keys validation
        4. State complexity analysis
        5. Cycle detection
        6. Path analysis

        Returns:
            FSMValidationResult object containing validation results
        """
        # Stage 0: Pydantic schema validation (catches type errors, missing fields)
        self._validate_pydantic_schema()

        # DECISION plan-2026-09-21T203800-8a03483a/D-017 (B12): every later
        # stage reads the RAW dicts with `.get()` / `in`, so a type-invalid
        # shape (a state that is a string, `transitions: "x"`, a dict target)
        # used to crash `validate()` instead of returning a result. Do NOT run
        # them on such input and do NOT add an error here: stage 0 has already
        # reported what the loader rejects, and an extra error for a shape the
        # loader accepts would make this validator stricter than the loader.
        if not self._structure_is_well_typed():
            return self.result

        # Stage 1: Basic structure validation
        self._validate_fsm_structure()

        # If basic structure is invalid, don't continue with more detailed checks
        if not self.result.is_valid:
            return self.result

        # Stage 2-3: Enhanced validation
        self._validate_terminal_states()
        self._validate_required_context_keys()
        self._validate_handler_only_keys()
        self._validate_kind_states()
        self._validate_unknown_keys()
        self._validate_schema_versions()
        self._validate_transition_classification_keys()

        # Stage 4-6: Analysis (won't affect validity but provides insights)
        self._analyze_state_complexity()
        self._detect_cycles()
        self._analyze_paths()

        return self.result

    def _validate_pydantic_schema(self):
        """Validate FSM data against the Pydantic FSMDefinition model.

        Anything that makes `FSMDefinition(**data)` -- and therefore
        `API.from_file` -- hard-fail on a KNOWN error class is reported as an
        ERROR: framework-authored semantic rules, absent required fields,
        container and string length bounds, `pattern=` mismatches, numeric
        bounds, `Literal` mismatches, and type/coercion failures.

        Only pydantic error classes that are NOT on that enumerated allow-list
        stay WARNINGS. That fail-safe branch is deliberate and load-bearing: an
        unknown/future pydantic type string must never make this validator
        stricter than the loader.
        """
        try:
            self._definition = FSMDefinition(**self.fsm_data)
        except ValidationError as e:
            # DECISION plan-2026-07-19T075908-70b6bdec/D-006 [STALE]: this ALLOW-list now
            # promotes `missing` too, closing the OPEN FOLLOW-UP that
            # plan-2026-07-18T051819-80b0bd4d/D-008 left here and that
            # plan-2026-07-18T162030-a02151fe/D-018 deferred for lack of authority.
            # Maintainer direction, verbatim:
            #
            #     "api from_file is the truth always, make sure fsm-llm-validate
            #      aligns with it"
            #
            # So `API.from_file` is the reference implementation and this validator
            # is a diagnostic that must AGREE with it. `missing`-class failures are
            # exactly the class where they used to disagree: no entry path that
            # builds an FSMDefinition from a dict (core/loaders.py, engine/api/,
            # agents/meta_builders/) backfills `description`/`id`, so `fsm-llm-validate`
            # was reporting is_valid=True for files API.from_file hard-crashes on.
            #
            # Do NOT "tidy" `missing` back out — that reopens the lie, and the
            # inverted test_agreement_property_across_fixtures will fail.
            #
            # DECISION plan-2026-07-20T040150-876e7164/D-001 [STALE]: the ALLOW-list below
            # was widened from 5 names to the full MEASURED loader-raising class.
            # A mechanical sweep of every `Field(pattern=/min_length=/max_length=/
            # ge=/le=)` and every model_validator reachable from `FSMDefinition`
            # measured 40 of 51 constraint violations as LIVE FALSE GREENS at the
            # previous 5-name list: `fsm-llm-validate` exited 0 on files
            # `API.from_file` hard-refuses. This supersedes D-013's "string bounds
            # are deliberately NOT promoted" carve-out and D-029's disclosed-but-
            # unfixed `string_pattern_mismatch` gap, both of which are now DELETED
            # rather than left standing as falsified comments.
            #
            # It also supersedes the "type-coercion leniency is untouched" framing
            # that carved out `int_parsing`. That framing was falsified by
            # measurement, not by opinion: `API.from_file` hard-refuses
            # `priority: "not-an-int"` exactly as it refuses `id: "café"`. Under
            # "api from_file is the truth always" the carve-out was a false green,
            # so `int_parsing` is promoted and the test that asserted the old
            # behavior (`test_simplified_dict_leniency_preserved`) was INVERTED.
            #
            # Do NOT blanket-promote by deleting the membership test and erroring
            # on every `e.errors()` entry. That is a real and tempting simplification
            # — every error seen at THIS site is by construction one the loader also
            # raised — but it destroys the `else: add_warning` fail-safe below, which
            # is the HARD invariant (plans/SYSTEM.md): an unknown/future pydantic type
            # string must keep falling through to WARNING so a new pydantic release
            # can never silently make this validator STRICTER than the loader.
            # Do NOT delete the `else` branch, and do NOT invert this into a
            # deny-list ("promote everything except known-lenient types") for the
            # same reason.
            #
            # The list is NOT the control. `test_constraint_sweep_validator_agrees_
            # with_loader` in tests/test_fsm_llm/test_validator_unit.py enumerates
            # the constraints mechanically from `core/definitions/` and asserts
            # agreement as an IFF, so a future constraint that introduces an
            # unlisted type name fails loudly there instead of rotting silently.
            for error in e.errors():
                loc = " → ".join(str(x) for x in error["loc"])
                if error["type"] in (
                    # Framework-authored semantic rules (model_validator /
                    # field_validator raising ValueError or asserting).
                    "value_error",
                    "assertion_error",
                    # Absent required field.
                    "missing",
                    # Container (list/dict) length bounds — D-013.
                    "too_short",
                    "too_long",
                    # String length bounds and `pattern=` — the D-029 gap.
                    "string_too_short",
                    "string_too_long",
                    "string_pattern_mismatch",
                    # Numeric bounds: every `Field(ge=/le=)`.
                    "greater_than_equal",
                    "less_than_equal",
                    # `Literal[...]` enum mismatch.
                    "literal_error",
                    # Type/coercion failures. Each name below was probed against
                    # `FSMDefinition(**data)` and confirmed reachable; none is a
                    # speculative entry.
                    "int_parsing",
                    "int_parsing_size",
                    "int_type",
                    "int_from_float",
                    "float_parsing",
                    "float_type",
                    "finite_number",
                    "bool_parsing",
                    "bool_type",
                    "string_type",
                    "string_unicode",
                    "dict_type",
                    "list_type",
                    # Typed sub-models with `extra="forbid"` (ContextScope,
                    # D-015): a non-dict value and an unknown key.
                    "model_type",
                    "extra_forbidden",
                ):
                    self.result.add_error(f"Schema: {loc}: {error['msg']}")
                else:
                    self.result.add_warning(f"Schema: {loc}: {error['msg']}")
        except Exception as e:
            # DECISION plan-2026-09-21T203800-8a03483a/D-017 (B12): the loader
            # (`FSMDefinition(**data)` in `API` and `load_fsm_definition`)
            # raises this same exception, so it is an ERROR. Do NOT downgrade
            # it to a warning: that reported is_valid=True for files
            # `API.from_file` refuses. The `else: add_warning` fail-safe above
            # is about unknown pydantic error TYPES inside a ValidationError,
            # not about construction failing outright.
            self.result.add_error(
                f"Schema validation failed: {type(e).__name__}: {e!s}"
            )


def validate_fsm_from_file(json_file: str) -> FSMValidationResult:
    """
    Validate an FSM definition from a JSON file.

    This function serves as the main entry point for validating FSM definitions
    stored in JSON files.

    Args:
        json_file: Path to the JSON file containing the FSM definition

    Returns:
        FSMValidationResult object containing validation results

    Raises:
        No exceptions - all errors are captured in the result object
    """
    result = FSMValidationResult(json_file)
    try:
        fsm_data = read_json_object(json_file)
    except FileNotFoundError:
        result.add_error(f"File '{json_file}' not found")
    except json.JSONDecodeError:
        result.add_error(f"'{json_file}' is not a valid JSON file")
    except UnicodeDecodeError:
        result.add_error(f"'{json_file}' is not valid UTF-8 text")
    except ValueError as e:
        result.add_error(str(e))  # the loader's sentence: not a JSON object
    except Exception as e:
        result.add_error(f"Error validating FSM: {e!s}")
    else:
        try:
            return FSMValidator(fsm_data).validate()
        except Exception as e:
            result.add_error(f"Error validating FSM: {e!s}")
    return result
