#!/usr/bin/env python3
"""
Parallel evaluation runner for FSM-LLM examples (compatibility shim).

The runner now lives in ``fsm_llm.eval`` and is installed as ``fsm-llm-eval``;
this script forwards its arguments to ``fsm-llm-eval examples`` from the
repository root, so every old invocation keeps working. Relative paths
(``--output-dir``, ``--examples-dir``, ``--config``) are read from the
repository root.

Usage:
    # Run with defaults (auto-detect examples, 4 workers)
    .venv/bin/python scripts/eval.py

    # Custom model and parallelism
    .venv/bin/python scripts/eval.py --model ollama_chat/qwen3.5:4b --workers 6

    # Filter by category
    .venv/bin/python scripts/eval.py --category agents

    # Custom timeout and output directory
    .venv/bin/python scripts/eval.py --timeout 180 --output-dir evaluation/run_012

    # List discovered examples without running
    .venv/bin/python scripts/eval.py --list

    # Same thing, installed entry point
    fsm-llm-eval examples --list
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    from fsm_llm.eval.__main__ import run

    os.chdir(ROOT)
    sys.exit(run(["examples", *sys.argv[1:]]))
