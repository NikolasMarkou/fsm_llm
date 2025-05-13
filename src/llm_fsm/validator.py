"""
FSM Validator: A utility for validating and analyzing Finite State Machine definitions.

This module provides comprehensive validation capabilities for FSM definitions, including:
- Basic structure validation (states, transitions, initial state)
- Terminal state analysis (existence and reachability)
- Cycle detection and analysis
- Path analysis from initial to terminal states
- State complexity assessment

It can be used both programmatically and via command-line interface.
"""

import os
import json
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, deque

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from .logging import logger
from .definitions import FSMDefinition

# --------------------------------------------------------------

class FSMValidationResult:
    """
    Container for FSM validation results.

    This class stores and categorizes validation findings into errors, warnings,
    and informational messages for comprehensive FSM analysis.

    Attributes:
        fsm_name (str): Name of the validated FSM
        is_valid (bool): Overall validity status (True if no errors)
        errors (List[str]): List of errors that make the FSM invalid
        warnings (List[str]): List of warnings about potential issues
        info (List[str]): List of informational insights about the FSM
    """

    def __init__(self, fsm_name: str):
        """
        Initialize a validation result container.

        Args:
            fsm_name: Name of the FSM being validated
        """
        self.fsm_name = fsm_name
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.info = []

    def add_error(self, message: str):
        """
        Add an error message and mark the FSM as invalid.

        Errors represent critical issues that prevent the FSM from functioning correctly.

        Args:
            message: The error message to add
        """
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"Validation error in {self.fsm_name}: {message}")

    def add_warning(self, message: str):
        """
        Add a warning message.

        Warnings represent potential issues that don't make the FSM invalid
        but might cause problems or indicate design flaws.

        Args:
            message: The warning message to add
        """
        self.warnings.append(message)
        logger.warning(f"Validation warning in {self.fsm_name}: {message}")

    def add_info(self, message: str):
        """
        Add an informational message.

        Info messages provide insights about the FSM structure and characteristics.

        Args:
            message: The informational message to add
        """
        self.info.append(message)
        logger.info(message)

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.

        Returns:
            Dict containing validation results in a structured format
        """
        return {
            "fsm_name": self.fsm_name,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info
        }

    def __str__(self) -> str:
        """
        Convert the result to a human-readable string for display.

        Returns:
            Formatted string representation of validation results
        """
        lines = [f"=== Validation Results for {self.fsm_name} ==="]

        # Overall status with emoji for visibility
        if self.is_valid:
            lines.append("✅ FSM is VALID")
        else:
            lines.append("❌ FSM is INVALID")

        # Group and format different message types
        if self.errors:
            lines.append("\n🔴 ERRORS:")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")

        if self.warnings:
            lines.append("\n🟠 WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")

        if self.info:
            lines.append("\n🔵 INFO:")
            for i, info in enumerate(self.info, 1):
                lines.append(f"  {i}. {info}")

        return "\n".join(lines)


class FSMValidator:
    """
    Validator for FSM definitions with detailed analysis capabilities.

    This class performs comprehensive validation and analysis of FSM definitions,
    identifying issues, potential problems, and providing structural insights.

    Attributes:
        fsm_data (Dict[str, Any]): The FSM definition to validate
        fsm_name (str): Name of the FSM
        states (Dict[str, Any]): States defined in the FSM
        initial_state (str): The FSM's initial state
        result (FSMValidationResult): Container for validation results
    """

    def __init__(self, fsm_data: Dict[str, Any]):
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
        # Stage 1: Basic structure validation
        self._validate_fsm_structure()

        # If basic structure is invalid, don't continue with more detailed checks
        if not self.result.is_valid:
            return self.result

        # Stage 2-3: Enhanced validation
        self._validate_terminal_states()
        self._validate_required_context_keys()

        # Stage 4-6: Analysis (won't affect validity but provides insights)
        self._analyze_state_complexity()
        self._detect_cycles()
        self._analyze_paths()

        return self.result

    def _validate_fsm_structure(self):
        """
        Validate the basic structure of the FSM.

        Checks:
        - Initial state exists and is defined
        - States dictionary exists
        - All transition target states exist
        - No orphaned states (unreachable from initial state)
        """
        # Check if initial state exists
        if not self.initial_state:
            self.result.add_error("No initial state defined")
        elif self.initial_state not in self.states:
            self.result.add_error(f"Initial state '{self.initial_state}' not found in states")

        # Check if states dictionary exists
        if not self.states:
            self.result.add_error("No states defined in the FSM")
            return  # Can't continue validation without states

        # Check all target states exist
        for state_id, state in self.states.items():
            transitions = state.get("transitions", [])
            for transition in transitions:
                target = transition.get("target_state", "")
                if not target:
                    self.result.add_error(f"Missing target_state in transition from '{state_id}'")
                elif target not in self.states:
                    self.result.add_error(f"Transition from '{state_id}' to non-existent state '{target}'")

        # Check for orphaned states (not reachable from initial state)
        reachable_states = self._get_reachable_states()
        orphaned_states = set(self.states.keys()) - reachable_states
        if orphaned_states:
            states_str = ", ".join(orphaned_states)
            self.result.add_error(f"Orphaned states detected: {states_str}")

    def _validate_terminal_states(self):
        """
        Validate terminal states (states with no outgoing transitions).

        Checks:
        - At least one terminal state exists
        - At least one terminal state is reachable from the initial state
        """
        terminal_states = self._get_terminal_states()

        # Check if at least one terminal state exists
        if not terminal_states:
            self.result.add_error("No terminal states found. At least one state must have no outgoing transitions.")
            return

        self.result.add_info(f"Terminal states: {', '.join(terminal_states)}")

        # Check if at least one terminal state is reachable
        reachable_states = self._get_reachable_states()
        reachable_terminal_states = terminal_states.intersection(reachable_states)

        if not reachable_terminal_states:
            self.result.add_error("No terminal states are reachable from the initial state.")
        else:
            self.result.add_info(f"Reachable terminal states: {', '.join(reachable_terminal_states)}")

    def _validate_required_context_keys(self):
        """
        Validate that required context keys are properly handled.

        Checks:
        - States with required_context_keys have transitions with conditions that use those keys
        - Warns if required keys aren't used in transition conditions
        """
        for state_id, state in self.states.items():
            required_keys = state.get("required_context_keys", [])

            if not required_keys:
                continue  # No required keys to validate

            # Check if there are transitions that require these keys
            transitions = state.get("transitions", [])
            has_conditional_transition = False

            # Look for any transition that actually checks these required keys
            for transition in transitions:
                conditions = transition.get("conditions", [])
                for condition in conditions:
                    if condition.get("requires_context_keys", []):
                        has_conditional_transition = True
                        break

            # Warn if we have required keys but no transitions that check them
            if required_keys and not has_conditional_transition:
                self.result.add_warning(
                    f"State '{state_id}' has required_context_keys {required_keys} "
                    "but no transitions with conditions requiring these keys"
                )

    def _analyze_state_complexity(self):
        """
        Analyze state complexity (number of transitions, conditions, etc.).

        Identifies states with many outgoing transitions that might be overly complex.
        """
        complex_states = []

        for state_id, state in self.states.items():
            transitions = state.get("transitions", [])

            # Consider states with more than 3 transitions as potentially complex
            # This threshold could be configurable in a future version
            if len(transitions) > 3:
                complex_states.append((state_id, len(transitions)))

        if complex_states:
            states_str = ", ".join([f"'{s}' ({t} transitions)" for s, t in complex_states])
            self.result.add_info(f"Complex states with many transitions: {states_str}")

    def _detect_cycles(self):
        """
        Detect cycles in the FSM that don't lead to terminal states.

        Cycles are sequences of states that form a loop. This method:
        1. Identifies all cycles in the FSM
        2. Determines which cycles have no escape path to a terminal state
        3. Reports both normal and problematic cycles
        """
        # First find all terminal states
        terminal_states = self._get_terminal_states()

        # Then find all cycles using DFS
        cycles = self._find_cycles()

        if not cycles:
            self.result.add_info("No cycles detected in the FSM")
            return

        # Check if cycles can escape to terminal states
        problematic_cycles = []
        for cycle in cycles:
            can_escape = False
            for state in cycle:
                state_obj = self.states.get(state, {})
                transitions = state_obj.get("transitions", [])

                # Check if any transition leads outside the cycle
                for transition in transitions:
                    target = transition.get("target_state", "")
                    if target not in cycle:
                        can_escape = True
                        break

            # Mark cycles with no escape path as problematic
            if not can_escape:
                problematic_cycles.append(cycle)

        # Report findings
        for cycle in cycles:
            cycle_str = " → ".join(cycle)
            if cycle in problematic_cycles:
                self.result.add_warning(f"Problematic cycle with no escape: {cycle_str}")
            else:
                self.result.add_info(f"Cycle detected: {cycle_str}")

    def _analyze_paths(self):
        """
        Analyze possible paths through the FSM to terminal states.

        Identifies the shortest path from initial state to each terminal state,
        helping understand the flow structure of the FSM.
        """
        terminal_states = self._get_terminal_states()

        if not terminal_states:
            return  # This would be caught by the terminal state validation

        # Find shortest paths to each terminal state using BFS
        paths = {}
        for terminal in terminal_states:
            shortest_path = self._find_shortest_path(self.initial_state, terminal)
            if shortest_path:
                paths[terminal] = shortest_path
                path_str = " → ".join(shortest_path)
                self.result.add_info(f"Shortest path to terminal state '{terminal}': {path_str}")
            else:
                # This case would be caught by terminal state reachability check
                pass

    def _get_reachable_states(self) -> Set[str]:
        """
        Get all states reachable from the initial state.

        Uses a breadth-first search approach to find all states that can be
        reached by following transitions from the initial state.

        Returns:
            Set of state IDs that are reachable from the initial state
        """
        reachable = {self.initial_state}

        # Keep expanding the set of reachable states until no new states are found
        change_made = True
        while change_made:
            change_made = False
            for state_id, state in self.states.items():
                if state_id in reachable:
                    transitions = state.get("transitions", [])
                    for transition in transitions:
                        target = transition.get("target_state", "")
                        if target and target not in reachable:
                            reachable.add(target)
                            change_made = True

        return reachable

    def _get_terminal_states(self) -> Set[str]:
        """
        Get all terminal states (states with no outgoing transitions).

        Terminal states are essential for FSMs to have proper end points.

        Returns:
            Set of state IDs that have no outgoing transitions
        """
        return {
            state_id for state_id, state in self.states.items()
            if not state.get("transitions", [])
        }

    def _find_cycles(self) -> List[List[str]]:
        """
        Find all simple cycles in the FSM using DFS.

        A cycle is a path that starts and ends at the same state,
        passing through at least one other state.

        Returns:
            List of cycles, where each cycle is a list of state IDs
        """
        def dfs(node, path, cycles):
            """
            Depth-first search helper function to find cycles.

            Args:
                node: Current state being explored
                path: Current path from initial state to current node
                cycles: List to collect discovered cycles
            """
            if node in path:
                # Found a cycle - extract the portion of the path that forms the cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                cycles.append(cycle)
                return

            # Continue DFS exploration
            path.append(node)
            state = self.states.get(node, {})
            transitions = state.get("transitions", [])

            for transition in transitions:
                target = transition.get("target_state", "")
                if target:
                    # Create a copy of the path for each branch to avoid cross-contamination
                    dfs(target, path.copy(), cycles)

        # Start DFS from the initial state
        cycles = []
        dfs(self.initial_state, [], cycles)

        # Remove duplicates (cycles can be detected multiple times from different entry points)
        unique_cycles = []
        cycle_sets = []

        for cycle in cycles:
            # Use frozenset to create a hashable representation of the cycle
            cycle_set = frozenset(cycle)
            if cycle_set not in cycle_sets:
                cycle_sets.append(cycle_set)
                unique_cycles.append(cycle)

        return unique_cycles

    def _find_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        Find the shortest path from start to end using BFS.

        This method is used to find the most direct route through the FSM
        from initial state to terminal states.

        Args:
            start: Starting state ID
            end: Target state ID

        Returns:
            List of state IDs representing the shortest path, or None if no path exists
        """
        if start == end:
            return [start]  # Already at the target

        # Use breadth-first search to find shortest path
        queue = deque([(start, [start])])  # (current_state, path_so_far)
        visited = {start}

        while queue:
            node, path = queue.popleft()
            state = self.states.get(node, {})
            transitions = state.get("transitions", [])

            for transition in transitions:
                target = transition.get("target_state", "")
                if not target:
                    continue

                # Check if we've reached the destination
                if target == end:
                    return path + [target]  # Return the complete path

                # Otherwise continue the search if we haven't visited this state yet
                if target not in visited:
                    visited.add(target)
                    queue.append((target, path + [target]))

        return None  # No path found


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
    try:
        # Load and parse the JSON file
        with open(json_file, 'r') as f:
            fsm_data = json.load(f)

        # Create a validator and run the validation
        validator = FSMValidator(fsm_data)
        return validator.validate()

    except FileNotFoundError:
        # Handle case when file doesn't exist
        result = FSMValidationResult(json_file)
        result.add_error(f"File '{json_file}' not found")
        return result

    except json.JSONDecodeError:
        # Handle case when file isn't valid JSON
        result = FSMValidationResult(json_file)
        result.add_error(f"'{json_file}' is not a valid JSON file")
        return result

    except Exception as e:
        # Catch-all for other exceptions
        result = FSMValidationResult(json_file)
        result.add_error(f"Error validating FSM: {str(e)}")
        return result


def main_cli():
    """
    Entry point for the command-line interface.

    Parses command-line arguments and runs the validator on the specified FSM file.
    Returns appropriate exit code for integration with shell scripts.

    Returns:
        0 if validation passed, 1 if validation failed or an error occurred
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate LLM-FSM definitions")
    parser.add_argument("--fsm", "-f", type=str, required=True, help="Path to FSM definition JSON file")
    parser.add_argument("--json", "-j", action="store_true", help="Output results in JSON format")
    parser.add_argument("--output", "-o", help="Output file (default: print to console)")

    args = parser.parse_args()

    # Run validation
    validation_result = validate_fsm_from_file(args.fsm)

    # Format output based on user preference
    if args.json:
        output = json.dumps(validation_result.as_dict(), indent=2)
    else:
        output = str(validation_result)

    # Output to file or console
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        logger.info(f"Validation results saved to {args.output}")
    else:
        print(output)

    # Return exit code based on validation result
    return 0 if validation_result.is_valid else 1


if __name__ == "__main__":
    import sys

    sys.exit(main_cli())