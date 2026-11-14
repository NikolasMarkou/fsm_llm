"""``visualize_fsm_from_file`` and the ``fsm-llm-visualize`` console script (``main``, ``main_cli``).

Layer ``core`` (plan d454fd62, step 76): split out of the old ``fsm_llm.visualizer``.
"""

from __future__ import annotations

import json

from fsm_llm.constants import CLI_EXIT_FAILURE, CLI_EXIT_OK
from fsm_llm.core.loaders import read_json_object
from fsm_llm.logging import enable_library_logging, logger, setup_cli_logging

from .ascii_layout import ASCII_STYLES
from .ascii_render import visualize_fsm_ascii
from .graph import _GRAPH_EXPORTERS, OUTPUT_FORMATS, build_fsm_graph


def visualize_fsm_from_file(
    json_file: str, style: str = "full", *, output_format: str = "ascii"
) -> str:
    """
    Visualize an FSM definition from a JSON file.

    Args:
        json_file: Path to the JSON file containing the FSM definition
        style: ASCII style, one of ``ASCII_STYLES`` (ASCII only)
        output_format: one of ``OUTPUT_FORMATS``: "ascii", "mermaid" or "dot"

    Returns:
        The diagram text, or a line starting with ``Error:`` on any failure
        (missing file, invalid JSON, invalid definition, unknown format).
    """
    try:
        fsm_data = read_json_object(json_file)

        if output_format == "ascii":
            return visualize_fsm_ascii(fsm_data, style)
        if output_format not in _GRAPH_EXPORTERS:
            raise ValueError(
                f"unknown output format {output_format!r} "
                f"(choose from {', '.join(OUTPUT_FORMATS)})"
            )
        return _GRAPH_EXPORTERS[output_format](build_fsm_graph(fsm_data))
    except FileNotFoundError:
        return f"Error: File '{json_file}' not found."
    except json.JSONDecodeError:
        return f"Error: '{json_file}' is not a valid JSON file."
    except Exception as e:
        return f"Error: {e}"


def main(fsm_path, style: str = "full", *, output_format: str = "ascii"):
    # DECISION plan-2026-07-19T191147-4b664252/D-014 [STALE]
    # NOT redundant — do not delete. See the twin anchor in `core/verify/cli.py:main`.
    # `logging.py`'s import-time `logger.disable("fsm_llm")` is deliberate library
    # hygiene; this function is the APPLICATION (the `fsm-llm-visualize` console
    # script and `fsm-llm --mode visualize`) and must opt back in, or the diagram
    # it exists to print never reaches the user. See decisions.md D-014. The
    # enable goes through the one `enable_library_logging`.
    enable_library_logging()
    # The default diagram is box drawing, which a redirected Windows stdout
    # (cp1252) cannot encode: write it as \uXXXX through the shared helper
    # instead of ending in a UnicodeEncodeError. Imported here, not at module
    # level: `cli_support` pulls in `okf`, and `import fsm_llm` loads neither.
    from fsm_llm.cli_support import escape_unencodable_output

    escape_unencodable_output()

    diagram = visualize_fsm_from_file(fsm_path, style, output_format=output_format)

    if diagram.startswith("Error:"):
        logger.error(diagram)
        return CLI_EXIT_FAILURE

    # DECISION plan-2026-09-21T203800-8a03483a/D-037
    # The diagram is the primary output and goes to stdout (pipeable);
    # errors stay on the logger (stderr). Do NOT move it back to `logger`.
    print(diagram)
    return CLI_EXIT_OK


def main_cli():
    """CLI entry point for fsm-llm-visualize."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Visualize an FSM definition as an ASCII, Mermaid or DOT diagram"
    )
    parser.add_argument(
        "--fsm", "-f", required=True, help="Path to FSM definition JSON file"
    )
    parser.add_argument(
        "--style",
        "-s",
        default="full",
        choices=ASCII_STYLES,
        help="ASCII visualization style (default: full)",
    )
    parser.add_argument(
        "--format",
        default="ascii",
        choices=OUTPUT_FORMATS,
        dest="output_format",
        help="Output format (default: ascii); --style applies to ascii only",
    )
    args = parser.parse_args()
    setup_cli_logging("INFO")
    sys.exit(
        main(fsm_path=args.fsm, style=args.style, output_format=args.output_format)
    )
