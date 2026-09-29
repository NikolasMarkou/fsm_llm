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
# CLI exit codes
# ---------------------------------------------------------------------------

#: Success, including a low score when no ``--fail-under`` threshold is set.
EXIT_OK = 0
#: Usage error, bad config or dataset, unwritable output, or nothing matched.
EXIT_ERROR = 1
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

# Per-example timeout overrides (for known slow examples)
EXAMPLE_TIMEOUTS: dict[str, int] = {
    "advanced/e_commerce": 300,
    "reasoning/math_tutor": 300,
    "agents/reflexion": 240,
    "agents/debate": 300,
    "agents/evaluator_optimizer": 300,
    "agents/orchestrator": 300,
    "agents/adapt": 240,
    "agents/memory_agent": 300,
    "agents/concurrent_react": 240,
    "agents/agent_as_tool": 300,
    "agents/structured_output": 240,
    "agents/react_hitl_combined": 240,
    "agents/skill_loader": 240,
    "agents/orchestrator_specialist": 300,
    "agents/hierarchical_orchestrator": 300,
    "agents/pipeline_review": 300,
    "agents/adapt_with_memory": 300,
    "agents/react_structured_pipeline": 300,
    "agents/multi_debate_panel": 300,
    "agents/reflexion_code_gen": 240,
    "agents/agent_memory_chain": 300,
    "agents/debate_with_tools": 240,
    "agents/eval_opt_structured": 300,
    "agents/legal_document_review": 300,
    "agents/investment_portfolio": 300,
    "agents/security_audit": 300,
    "agents/medical_literature": 300,
    "agents/architecture_review": 300,
    "agents/supply_chain_optimizer": 300,
    "agents/regulatory_compliance": 300,
    "intermediate/adaptive_quiz": 300,
    "meta/meta_review_loop": 240,
    "meta/meta_from_spec": 240,
    "meta/build_fsm": 240,
    "meta/build_agent": 240,
    "meta/build_workflow": 240,
    "workflows/workflow_agent_loop": 300,
    "workflows/loan_processing": 300,
    "workflows/release_management": 300,
    "workflows/customer_onboarding": 300,
}

# Stdin inputs for interactive examples (those with input() calls).
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
    "classification/intent_routing": "I want to cancel my subscription\nI need help with billing\nquit\n",
    "classification/smart_helpdesk": "My internet is down\nI've already tried restarting the router\nquit\n",
    "classification/classified_transitions": "I want to buy something\nA new laptop\nquit\n",
    "classification/classified_transitions_manual": "I want to buy something\nA new laptop\nquit\n",
    # agents (interactive)
    "agents/react_search": "What is the population of Tokyo?\nquit\n",
    "agents/hitl_approval": "Send an email to bob@example.com saying hello\ny\nquit\n",
    "agents/react_hitl_combined": "Search for weather in Paris\ny\nquit\nquit\n",
    "agents/classified_dispatch": "Search for information about climate change\nquit\n",
    "agents/classified_tools": "Calculate the sum of 15, 23, and 100\nquit\n",
    "agents/full_pipeline": "Look up the latest news about AI\nquit\n",
    "agents/hierarchical_tools": "What is 15 times 23?\nquit\n",
    "agents/reasoning_stacking": "What is the square root of 144?\nquit\n",
    "agents/reasoning_tool": "Solve: if x + 5 = 12, what is x?\nquit\n",
    "agents/tool_decorator": "What is 5 plus 3?\nquit\n",
    "agents/skill_loader": "What time is it?\nquit\n",
    # reasoning
    "reasoning/math_tutor": "What is 15 + 27?\nquit\n",
    # meta
    "meta/build_fsm": "Build a simple greeting bot\nyes\nquit\n",
    "meta/build_workflow": "Build a workflow for order processing with validation payment and fulfillment\nyes\nquit\n",
    "meta/build_agent": "Build a research agent that can search the web and summarize using ReAct pattern\nyes\nquit\n",
}
