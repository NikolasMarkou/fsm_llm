"""
Constants for the fsm_llm.eval package.

Frozen data only: run-directory naming, timestamp formats and fallback values
shared by the record writers, CLI exit codes, runner defaults, the 0-4 score
labels, and the repository examples tables (stdin scripts and timeouts).
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

# ---------------------------------------------------------------------------
# CLI exit codes (0, 1 and 130 are core's ``CLI_EXIT_*``; a usage error exits 1)
# ---------------------------------------------------------------------------

#: The run finished but scored below the ``--fail-under`` threshold.
EXIT_BELOW_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Runner defaults (overridable by a config file or CLI flags)
# ---------------------------------------------------------------------------

#: Parallel example subprocesses.
DEFAULT_WORKERS = 4
#: Seconds per example when neither timeout table names it.
DEFAULT_TIMEOUT = 120
#: Directory scanned for ``<category>/<example>/run.py`` scripts.
DEFAULT_EXAMPLES_DIR = "examples"
#: Script names discovered as examples; ``run_<x>.py`` becomes ``<dir>_<x>``.
EXAMPLE_SCRIPT_NAMES = ("run.py", "run_manual.py")
#: Trials per conversation case (the repository's N=3 practice).
DEFAULT_TRIALS = 3

# ---------------------------------------------------------------------------
# Conversation cases (``fsm-llm-eval run``)
# ---------------------------------------------------------------------------

#: Recorded in a conversation run's ``results.json`` and ``summary.md``.
CASES_EVALUATOR_NAME = "fsm-llm-eval run (conversation cases)"
#: Dataset files with this suffix hold one case per line; others hold JSON.
JSONL_SUFFIX = ".jsonl"
#: Per-trial rows, appended and flushed as each trial finishes.
ROWS_FILENAME = "rows.jsonl"

# ---------------------------------------------------------------------------
# Example scoring (heuristic 0-4 rubric, EVALUATE.md section 3)
# ---------------------------------------------------------------------------

#: Label per score, indexed by the score itself.
SCORE_LABELS = ("CRASH", "BROKEN", "PARTIAL", "MOSTLY", "PASS")
#: Highest heuristic score; health is ``sum(scores) / (MAX_SCORE * n)``.
MAX_SCORE = 4
#: Recorded in ``results.json`` and the scorecard header.
EVALUATOR_NAME = "fsm-llm-eval examples (automated)"

# ---------------------------------------------------------------------------
# Repository example tables, moved verbatim from scripts/eval.py (0facf56).
# A config file can add or override entries (merged over these).
# ---------------------------------------------------------------------------

# Timeout overrides per category (seconds)
CATEGORY_TIMEOUTS: dict[str, int] = {
    "agents": 180,
    "reasoning": 300,
    "workflows": 180,
    "advanced": 180,
    "meta": 120,
}

# Per-example timeout overrides (for known slow examples).
# An example whose agent runs on the default AgentConfig.timeout_seconds
# (300 s) gets 360 here, budget + 60: at 300 the agent's own timeout and the
# harness timeout fired together (an Error: Agent timed out line or a
# score-1 kill, by chance); at 360 the agent's timeout always fires first.
# tests/test_fsm_llm_eval/test_timeout_table.py pins this against the
# examples' source.
EXAMPLE_TIMEOUTS: dict[str, int] = {
    "advanced/e_commerce": 300,
    "reasoning/math_tutor": 300,
    "agents/reflexion": 240,
    "agents/debate": 360,
    "agents/evaluator_optimizer": 360,
    "agents/orchestrator": 360,
    "agents/adapt": 240,
    "agents/memory_agent": 360,
    "agents/concurrent_react": 240,
    "agents/agent_as_tool": 360,
    "agents/structured_output": 240,
    "agents/react_hitl_combined": 240,
    "agents/skill_loader": 240,
    "agents/orchestrator_specialist": 360,
    "agents/hierarchical_orchestrator": 360,
    "agents/pipeline_review": 360,
    "agents/adapt_with_memory": 360,
    "agents/react_structured_pipeline": 360,
    "agents/multi_debate_panel": 360,
    "agents/reflexion_code_gen": 240,
    "agents/agent_memory_chain": 360,
    "agents/debate_with_tools": 240,
    "agents/eval_opt_structured": 360,
    "agents/legal_document_review": 360,
    "agents/investment_portfolio": 360,
    "agents/security_audit": 360,
    "agents/medical_literature": 360,
    "agents/architecture_review": 360,
    "agents/supply_chain_optimizer": 360,
    "agents/regulatory_compliance": 360,
    "intermediate/adaptive_quiz": 300,
    "meta/meta_review_loop": 240,
    "meta/meta_from_spec": 240,
    "meta/build_fsm": 240,
    "meta/build_agent": 240,
    "meta/build_workflow": 240,
    "workflows/workflow_agent_loop": 360,
    "workflows/loan_processing": 300,
    "workflows/release_management": 360,
    "workflows/customer_onboarding": 360,
}

# Stdin inputs for interactive examples (those with input() calls or a
# run_interactive() session). tests/test_fsm_llm_eval/test_example_inputs.py pins
# every script against its example's source.
# Each value is a newline-separated string piped to stdin.
EXAMPLE_INPUTS: dict[str, str] = {
    # basic
    "basic/simple_greeting": "Hello there!\nMy name is Alex\nquit\n",
    "basic/form_filling": "My name is John Smith\njohn@example.com\n30\nSoftware engineer\nyes\nquit\n",
    "basic/story_time": "That's a silly choice, straw is way too weak!\nOh no, the wolf is coming!\nSticks are barely any better, the wolf will blow them down too\nThe wolf huffed and puffed!\nI predict the wolf cannot blow down the brick house\nThey should put a pot of boiling water in the fireplace\nGreat story, I loved it!\nquit\n",
    # intermediate
    "intermediate/book_recommendation": "I like science fiction\nSomething like Dune\nquit\n",
    "intermediate/product_recommendation": "I need a laptop for programming\nAround 1500 dollars\nquit\n",
    "intermediate/adaptive_quiz": "My name is Sam\nParis\n42\nH2O\nThat was fun!\nquit\n",
    # advanced
    "advanced/yoga_instructions": "I want to do some yoga\nI'm a beginner\nquit\n",
    "advanced/support_pipeline": "Hi, I'm Sarah and my order hasn't arrived\nOrder number 12345\nyes that fixed it\nno thanks\nquit\nquit\nquit\n",
    # classification
    "classification/smart_helpdesk": "My internet is down\nI've already tried restarting the router\nquit\n",
    "classification/classified_transitions_manual": "I want to buy something\nA new laptop\nquit\n",
    # agents (interactive)
    "agents/react_search": "What is the population of Tokyo?\nquit\n",
    "agents/hitl_approval": "Send an email to bob@example.com saying hello\ny\nquit\n",
    # The task must call a gated tool (publish_report, archive_data) for the example
    # to ask for approval; the "y" answers that one prompt. A task that calls no
    # gated tool would turn the "y" into a second task.
    "agents/react_hitl_combined": "Research recent AI papers and publish a summary report\ny\nquit\n",
    "agents/classified_dispatch": "Search for information about climate change\nquit\n",
    "agents/classified_tools": "Calculate the sum of 15, 23, and 100\nquit\n",
    "agents/full_pipeline": "Look up the latest news about AI\nquit\n",
    "agents/hierarchical_tools": "What is 15 times 23?\nquit\n",
    "agents/reasoning_stacking": "What is the square root of 144?\nquit\n",
    # reasoning
    "reasoning/math_tutor": "What is 15 + 27?\nquit\n",
    # meta: run_interactive reads until EOF and sends every line to the builder, so
    # no "quit" line (it would cost one more model call)
    "meta/build_fsm": "Build a simple greeting bot\nyes\n",
    "meta/build_workflow": "Build a workflow for order processing with validation payment and fulfillment\nyes\n",
    "meta/build_agent": "Build a research agent that can search the web and summarize using ReAct pattern\nyes\n",
}
