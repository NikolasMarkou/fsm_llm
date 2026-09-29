"""
Constants for the fsm_llm.eval package.

Frozen data only: run-directory naming, timestamp formats and fallback values
shared by the record writers. Runner defaults and the examples tables join
this module with the modules that read them.
"""

from __future__ import annotations

#: Default root directory under which each run gets its own directory.
DEFAULT_OUTPUT_ROOT = "evaluation"

#: Minute-resolution run-directory timestamp (local time), the historical
#: ``scripts/eval.py`` layout: ``<stamp>_<git-short-hash>_<model-slug>``.
RUN_DIR_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"

#: Highest ``_N`` suffix tried when run directories with the same name exist.
MAX_RUN_DIR_SUFFIX = 1000

#: UTC timestamp format written into rows and results files.
UTC_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Recorded commit when ``git`` is missing or the directory is not a repo.
GIT_UNKNOWN = "unknown"
