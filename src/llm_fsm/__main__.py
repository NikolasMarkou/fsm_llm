#!/usr/bin/env python3
"""
Command-line entry-point for the **LLM-FSM** framework.

This module exposes a *single* public function – :pyfunc:`main_cli` – which
implements a thin wrapper around the three high-level workflows offered by the
package:

* **run** – interactively execute an FSM-driven conversation
* **validate** – perform static analysis on an FSM definition
* **visualize** – produce an ASCII diagram of an FSM

All heavy-lifting is delegated to the corresponding ``main`` functions in
:pyfile:`runner.py`, :pyfile:`validator.py`, and :pyfile:`visualizer.py`.  That
keeps this file small and prevents cyclic imports while still allowing users to
invoke the framework with the canonical

------------------------------------------------------------------------------
CLI synopsis
------------

.. code-block:: text

   usage: python -m llm_fsm [-h]
                            [--mode {run,validate,visualize}]
                            --fsm FILE
                            [--style {full,compact,minimal}]
                            [--history-size N] [--message-length L]
                            [--json] [--output FILE] [--version]

Positional / required arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--fsm, -f``
    Path to a *JSON* file containing a Finite-State-Machine definition
    (see :pyfile:`definitions.py` for the concrete schema).

Optional switches
^^^^^^^^^^^^^^^^^

``--mode`` *(default: ``run``)*
    • ``run`` – chat with the FSM using the model and environment settings
    • ``validate`` – lint the FSM and emit a pass/fail report
    • ``visualize`` – render an ASCII diagram to stdout (or ``--output``)

``--style, -s`` *(visualize only)*
    Rendering style: ``full`` (rich, multi-section), ``compact``, or ``minimal``.

``--history-size, -n``
    Circular buffer length for conversation memory
    (default: :pydata:`llm_fsm.constants.DEFAULT_MAX_HISTORY_SIZE`).

``--message-length, -l``
    Soft limit on characters retained per message before truncation
    (default: :pydata:`llm_fsm.constants.DEFAULT_MAX_MESSAGE_LENGTH`).

``--json, -j``
    Emit *machine-readable* JSON instead of human-friendly text (where supported).

``--output, -o``
    Write results to a file instead of *stdout*.

``--version, -v``
    Print the package version and exit.
"""

import sys
import argparse

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .constants import (
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH
)

from .__version__ import __version__

# --------------------------------------------------------------


def main_cli():
    from .runner import main as main_runner
    from .validator import main as main_validator
    from .visualizer import main as main_visualizer
    """Entry point for the CLI."""
    parser = (
        argparse.ArgumentParser(
            description=f"Run / Evaluate / Visualize an FSM-based conversations v{__version__}"
        )
    )
    parser.add_argument(
        "--mode",
        choices=["run", "validate", "visualize"],
        default="run",
        help="Mode in which to run the program: 'run', 'validate', or 'visualize' (default: 'run')"
    )
    parser.add_argument(
        "--fsm", "-f",
        type=str,
        required=True,
        help="Path to FSM definition JSON file")
    parser.add_argument(
        "--style", "-s",
        default="full",
        choices=["full", "compact", "minimal"],
        help="Visualization style (default: full)")

    parser.add_argument(
        "--history-size", "-n",
        type=int,
        default=DEFAULT_MAX_HISTORY_SIZE,
        help=f"Maximum number of conversation exchanges to include in history (default: {DEFAULT_MAX_HISTORY_SIZE})"
    )
    parser.add_argument(
        "--message-length", "-l",
        type=int,
        default=DEFAULT_MAX_MESSAGE_LENGTH,
        help=f"Maximum length of messages in characters (default: {DEFAULT_MAX_MESSAGE_LENGTH})"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: print to console)"
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Output version information"
    )
    args = parser.parse_args()

    if args.version:
        print(f"llm_fsm v{__version__}")
        return 0

    # Run with the provided parameters
    if args.mode == "run":
        return main_runner(
            fsm_path=args.fsm,
            max_history_size=args.history_size,
            max_message_length=args.message_length)
    elif args.mode == "validate":
        return main_validator(
            fsm_path=args.fsm
        )
    elif args.mode == "visualize":
        return main_visualizer(
            fsm_path=args.fsm,
            style=args.style
        )
    else:
        raise ValueError(f"Invalid mode {args.mode}")

# --------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main_cli())

# --------------------------------------------------------------
