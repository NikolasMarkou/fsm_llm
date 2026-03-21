"""
Unit tests for fsm_llm.validator module.

Tests cover:
- FSMValidationResult: error/warning/info accumulation, is_valid property
- FSMValidator.validate: valid FSMs, orphaned states, unreachable terminals, missing initial
- Cycle detection: simple cycles, no cycles, self-loops
- Complex FSM with multiple issues
"""


from fsm_llm.validator import FSMValidationResult, FSMValidator

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _valid_fsm_data():
    """A simple valid FSM: start -> middle -> end."""
    return {
        "name": "valid_fsm",
        "initial_state": "start",
        "states": {
            "start": {
                "description": "Start",
                "purpose": "Begin",
                "transitions": [
                    {"target_state": "middle", "description": "Go to middle"}
                ],
            },
            "middle": {
                "description": "Middle",
                "purpose": "Process",
                "transitions": [
                    {"target_state": "end", "description": "Go to end"}
                ],
            },
            "end": {
                "description": "End",
                "purpose": "Finish",
                "transitions": [],
            },
        },
    }


# ==================================================================
# FSMValidationResult
# ==================================================================

class TestFSMValidationResult:

    def test_initially_valid(self):
        result = FSMValidationResult("test")
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.info == []

    def test_add_error_invalidates(self):
        result = FSMValidationResult("test")
        result.add_error("something broke")
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        result = FSMValidationResult("test")
        result.add_warning("watch out")
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_add_info(self):
        result = FSMValidationResult("test")
        result.add_info("FYI")
        assert result.is_valid is True
        assert len(result.info) == 1

    def test_as_dict(self):
        result = FSMValidationResult("my_fsm")
        result.add_error("err1")
        result.add_warning("warn1")
        d = result.as_dict()
        assert d["fsm_name"] == "my_fsm"
        assert d["is_valid"] is False
        assert "err1" in d["errors"]
        assert "warn1" in d["warnings"]

    def test_str_representation(self):
        result = FSMValidationResult("demo")
        result.add_error("bad thing")
        text = str(result)
        assert "demo" in text
        assert "INVALID" in text
        assert "bad thing" in text


# ==================================================================
# FSMValidator — valid FSM
# ==================================================================

class TestValidatorValidFSM:

    def test_valid_linear_fsm(self):
        result = FSMValidator(_valid_fsm_data()).validate()
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_valid_single_state_terminal(self):
        """An FSM where the initial state is also the terminal state."""
        data = {
            "name": "one_state",
            "initial_state": "only",
            "states": {
                "only": {
                    "description": "Only state",
                    "purpose": "Everything",
                    "transitions": [],
                }
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is True


# ==================================================================
# FSMValidator — missing / bad initial state
# ==================================================================

class TestValidatorInitialState:

    def test_missing_initial_state(self):
        data = _valid_fsm_data()
        data["initial_state"] = ""
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("initial state" in e.lower() for e in result.errors)

    def test_initial_state_not_in_states(self):
        data = _valid_fsm_data()
        data["initial_state"] = "nonexistent"
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("nonexistent" in e for e in result.errors)


# ==================================================================
# FSMValidator — orphaned states
# ==================================================================

class TestValidatorOrphanedStates:

    def test_orphaned_state_detected(self):
        data = _valid_fsm_data()
        data["states"]["orphan"] = {
            "description": "Orphan",
            "purpose": "Unreachable",
            "transitions": [],
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("orphan" in e.lower() for e in result.errors)


# ==================================================================
# FSMValidator — unreachable terminal
# ==================================================================

class TestValidatorUnreachableTerminal:

    def test_no_terminal_state(self):
        """Every state has transitions, so there is no terminal state."""
        data = {
            "name": "no_terminal",
            "initial_state": "a",
            "states": {
                "a": {
                    "description": "A",
                    "purpose": "A",
                    "transitions": [
                        {"target_state": "b", "description": "to b"}
                    ],
                },
                "b": {
                    "description": "B",
                    "purpose": "B",
                    "transitions": [
                        {"target_state": "a", "description": "to a"}
                    ],
                },
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("terminal" in e.lower() for e in result.errors)


# ==================================================================
# FSMValidator — cycle detection
# ==================================================================

class TestValidatorCycleDetection:

    def test_no_cycle_linear(self):
        """Linear FSM should report no cycles."""
        result = FSMValidator(_valid_fsm_data()).validate()
        has_cycle_info = any("cycle" in i.lower() for i in result.info)
        has_no_cycle_info = any("no cycle" in i.lower() for i in result.info)
        # Either no mention of cycles, or explicit "no cycles" message
        assert has_no_cycle_info or not has_cycle_info

    def test_simple_cycle_detected(self):
        data = {
            "name": "cyclic",
            "initial_state": "a",
            "states": {
                "a": {
                    "description": "A",
                    "purpose": "A",
                    "transitions": [
                        {"target_state": "b", "description": "to b"},
                        {"target_state": "done", "description": "to done"},
                    ],
                },
                "b": {
                    "description": "B",
                    "purpose": "B",
                    "transitions": [
                        {"target_state": "a", "description": "back to a"}
                    ],
                },
                "done": {
                    "description": "Done",
                    "purpose": "End",
                    "transitions": [],
                },
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is True
        # The cycle a->b->a should be detected in info or warnings
        all_messages = result.info + result.warnings
        assert any("cycle" in m.lower() for m in all_messages)

    def test_self_loop_detected(self):
        data = {
            "name": "self_loop",
            "initial_state": "a",
            "states": {
                "a": {
                    "description": "A",
                    "purpose": "A",
                    "transitions": [
                        {"target_state": "a", "description": "stay in a"},
                        {"target_state": "end", "description": "to end"},
                    ],
                },
                "end": {
                    "description": "End",
                    "purpose": "End",
                    "transitions": [],
                },
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is True
        all_messages = result.info + result.warnings
        assert any("cycle" in m.lower() for m in all_messages)


# ==================================================================
# Complex FSM with multiple issues
# ==================================================================

class TestValidatorComplexFSM:

    def test_multiple_issues(self):
        data = {
            "name": "complex_issues",
            "initial_state": "start",
            "states": {
                "start": {
                    "description": "Start",
                    "purpose": "Begin",
                    "transitions": [
                        {"target_state": "middle", "description": "go"}
                    ],
                },
                "middle": {
                    "description": "Middle",
                    "purpose": "Work",
                    "transitions": [
                        {"target_state": "end", "description": "finish"}
                    ],
                },
                "end": {
                    "description": "End",
                    "purpose": "Done",
                    "transitions": [],
                },
                "orphan": {
                    "description": "Orphan",
                    "purpose": "Nowhere",
                    "transitions": [],
                },
            },
        }
        result = FSMValidator(data).validate()
        # Orphan should trigger an error
        assert result.is_valid is False
        assert any("orphan" in e.lower() for e in result.errors)

    def test_transition_to_nonexistent_state(self):
        data = {
            "name": "bad_target",
            "initial_state": "start",
            "states": {
                "start": {
                    "description": "Start",
                    "purpose": "Begin",
                    "transitions": [
                        {"target_state": "ghost", "description": "to ghost"}
                    ],
                },
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("ghost" in e for e in result.errors)
