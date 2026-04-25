"""Backward-compat CLI entry point.

Delegates to fsm_llm.stdlib.agents.__main__.main() so that
`python -m fsm_llm_agents [--info|--version]` keeps working unchanged.
"""

import sys

from fsm_llm.stdlib.agents.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
