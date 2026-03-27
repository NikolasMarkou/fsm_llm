from __future__ import annotations

"""
CLI entry point for the meta-agent.

Usage:
    python -m fsm_llm_meta
    fsm-llm-meta --model gpt-4o-mini --output my_fsm.json
"""

import argparse
import sys

from fsm_llm_agents.constants import MetaDefaults
from fsm_llm_agents.definitions import MetaBuilderConfig
from fsm_llm_agents.exceptions import MetaBuilderError
from fsm_llm_agents.meta_builder import MetaBuilderAgent
from fsm_llm_agents.meta_output import format_summary, save_artifact


def main_cli() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fsm-llm-meta",
        description="Interactively build FSMs, Workflows, and Agents",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"LLM model to use (default: {MetaDefaults.MODEL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for the artifact JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"LLM temperature (default: {MetaDefaults.TEMPERATURE})",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=f"Maximum conversation turns (default: {MetaDefaults.MAX_TURNS})",
    )

    args = parser.parse_args()

    # Build config
    config_kwargs: dict = {}
    if args.model:
        config_kwargs["model"] = args.model
    if args.temperature is not None:
        config_kwargs["temperature"] = args.temperature
    if args.max_turns is not None:
        config_kwargs["max_turns"] = args.max_turns

    config = MetaBuilderConfig(**config_kwargs)
    agent = MetaBuilderAgent(config=config)

    try:
        result = agent.run_interactive()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except MetaBuilderError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print(format_summary(result))
    print("=" * 60)

    if result.is_valid and result.artifact_json:
        if args.output:
            path = save_artifact(result.artifact, args.output)
            print(f"\nArtifact saved to: {path}")
        else:
            print("\nGenerated artifact JSON:")
            print(result.artifact_json)
    elif not result.is_valid:
        print("\nArtifact has validation errors. Fix them and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
