"""
Prompt builders for agent tool awareness and observation formatting.

Trust boundary: builders that take constructor text (maker/checker
instructions, debate personas) interpolate it unsanitized into static FSM
definition text, the same class of content as an FSM's ``persona``. That text
must be developer-authored; no builder here takes live end-user content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .constants import ContextKeys, Defaults, EvalOptStates, MakerCheckerStates
from .tools import ToolRegistry, schema_types, tool_parameters_schema

if TYPE_CHECKING:
    from .semantic_tools import SemanticToolRegistry


# Generated loop values (debate rounds, drafts, chain step outputs,
# reflections, plan step results) are
# written, not found: the field prompt frames every value as an extraction,
# and live qwen3.5:4b returned null for text nothing in the prompt holds. Every generated text field opens with this sentence.
_COMPOSE = (
    "This value does not exist yet: do not look for it in the messages or the "
    "context, compose it yourself now and never return null. "
)


def _build_tool_example(tool_name: str, params: dict) -> str:
    """Build a compact JSON example for a specific tool call."""
    import json

    example_values: dict[str, object] = {}
    for pname, pschema in params.items():
        types = [t for t in schema_types(pschema) if t != "null"]
        ptype = types[0] if types else "string"
        if ptype == "number" or ptype == "integer":
            example_values[pname] = 0
        elif ptype == "boolean":
            example_values[pname] = True
        else:
            example_values[pname] = f"<{pname}>"

    return json.dumps({"tool_name": tool_name, "tool_input": example_values})


def _get_tool_list(
    registry: ToolRegistry,
    task_description: str | None = None,
) -> tuple[str, list]:
    """Get tool prompt text and tool list, with optional semantic filtering.

    When the registry is a SemanticToolRegistry and a task_description is
    provided, returns only the most relevant tools for the task.
    """
    if task_description and hasattr(registry, "retrieve"):
        # The `hasattr(registry, "retrieve")` guard means registry is a
        # SemanticToolRegistry here, whose `to_prompt_description` accepts `query`
        # and which defines `retrieve` (base ToolRegistry has neither). mypy types
        # the param as base ToolRegistry; cast follows the runtime narrowing.
        # Type gap from duck-typing, not a bug — see D-007. Annotation-only.
        semantic = cast("SemanticToolRegistry", registry)
        tool_text = semantic.to_prompt_description(query=task_description)
        tools = semantic.retrieve(task_description)
    else:
        tool_text = registry.to_prompt_description()
        tools = registry.list_tools()
    return tool_text, tools


def build_think_extraction_instructions(
    registry: ToolRegistry,
    include_observations: bool = True,
    task_description: str | None = None,
) -> str:
    """Build extraction instructions for the think state."""
    tool_list, tools = _get_tool_list(registry, task_description)

    if include_observations:
        intro = (
            "Analyze the task and all previous observations to decide your next action."
        )
    else:
        intro = "Analyze the task to decide your next action."

    parts = [
        intro,
        "",
        tool_list,
        "",
        "Extract the following as JSON:",
        '- "tool_name": name of the tool to use (must be one of the available tools), or "none" if no tool is needed',
        '- "tool_input": a JSON object with the parameters for the tool (MUST include all required parameters)',
        '- "reasoning": your step-by-step reasoning for choosing this action',
        '- "should_terminate": true if you have enough information to answer the task, false otherwise',
    ]

    # Generate per-tool examples from registry (uses filtered list if semantic)
    tools_with_params = []
    for tool in tools:
        props = tool_parameters_schema(tool).get("properties", {})
        if props:
            tools_with_params.append((tool.name, props))

    if tools_with_params:
        parts.append("")
        parts.append("Examples for each tool (you MUST provide tool_input parameters):")
        for tool_name, params in tools_with_params:
            parts.append(_build_tool_example(tool_name, params))
    else:
        parts.append("")
        parts.append("Example — using a tool:")
        parts.append(
            '{"tool_name": "search", "tool_input": {"query": "example query"}, '
            '"reasoning": "I need to find this information", "should_terminate": false}'
        )

    parts.extend(
        [
            "",
            "Example — terminating with enough information:",
            '{"tool_name": "none", "tool_input": {}, '
            '"reasoning": "I have all the information needed", "should_terminate": true}',
        ]
    )

    parts.extend(
        [
            "",
            "RULES:",
            "1. You MUST select a tool on the first iteration. "
            "Do not set should_terminate=true without calling at least one tool first.",
            "2. Only set should_terminate=true AFTER you have tool results that fully answer the task.",
            "3. Always provide the required parameters for the selected tool.",
        ]
    )

    if include_observations:
        parts.extend(
            [
                "",
                "IMPORTANT: Review all previous observations carefully before deciding.",
                "If previous tool calls have provided sufficient information, set should_terminate to true.",
                "Do NOT repeat the same tool call with the same parameters.",
            ]
        )

    return "\n".join(parts)


# Appended to the conclude instructions of approval-gated FSMs (D-034 of plan
# 07ad3f8c). It names no prompt, signal or request to go on (D-031).
_REFUSED_ACTIONS_SENTENCE = (
    f" If the context has a '{ContextKeys.REFUSED_ACTIONS}' list, every action "
    "in it was refused by a human approver and was NOT performed: state in "
    "the answer that it was not performed because approval was refused, and "
    "never say or imply that a refused action was done or will be done."
)


def build_conclude_response_instructions(*, refused_actions: bool = False) -> str:
    """Build response instructions for the conclude state.

    ``refused_actions`` (approval-gated FSMs only) adds one sentence: actions
    listed under the ``refused_actions`` context key were not performed and
    the answer must say so (D-034 of plan 07ad3f8c). Without it the text is
    unchanged.
    """
    # DECISION plan_2026-05-30_5598b755/D-003 [STALE]
    # Re-anchor the original task: the reply is written with no user message,
    # and small models otherwise answer the turn instead of the task. The real
    # task is available in the context as `task`.
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-031
    # Do NOT name the turn mechanics here (a prompt, a message to ignore, a
    # request to go on): live, qwen3.5:4b answered that wording instead of
    # the task on a forced stop. Say what the reply is (the run's last
    # output, from the observations) and what to do when the evidence is
    # thin. See decisions.md D-031.
    text = (
        "Write the final answer to the ORIGINAL task (the 'task' value in the "
        "context) clearly and completely. Base it on the tool observations and "
        "on facts given in the task, and cite the observations that support it. "
        "This reply is the last output of the run: no further tool will run, "
        "so do not describe work in progress or planned next steps. If the "
        "observations do not hold enough evidence, say plainly what could not "
        "be determined and give the best answer the evidence supports."
    )
    if refused_actions:
        text += _REFUSED_ACTIONS_SENTENCE
    return text


def build_think_terminate_instructions() -> str:
    """Per-field instructions for a think turn's ``should_terminate`` (bool).

    Permissive on purpose (plan 06a5ec0a D-034/D-035): the conclude edge's
    evidence guard (D-008 of plan c1d5bfbc) already stops a turn-1 True with
    no tool run, so the wording must not also forbid it.
    """
    return (
        "true if you have enough information to answer the task: set it to "
        "true when the observations already answer the task or when no "
        "further tool call is needed; false only when another tool call is "
        "still required to answer it."
    )


# ---------------------------------------------------------------------------
# Reflexion prompts
# ---------------------------------------------------------------------------


def build_evaluate_field_instructions() -> dict[str, str]:
    """Per-field instructions for the Reflexion ``evaluate`` state.

    Returns ``{field_name: instructions}`` for ``evaluation_passed`` (bool),
    ``evaluation_score`` (float) and ``evaluation_feedback`` (str).
    """
    criteria = (
        "Judge whether the observations so far answer the task correctly and "
        "completely: every part of the question addressed, the evidence "
        "reliable and consistent."
    )
    return {
        "evaluation_passed": (
            f"{criteria} true if they are sufficient, false otherwise."
        ),
        "evaluation_score": (
            f"{criteria} A number from 0.0 to 1.0 rating the answer quality."
        ),
        "evaluation_feedback": (
            f"{_COMPOSE}{criteria} One or two sentences on what is good or missing."
        ),
    }


def build_reflect_field_instructions() -> dict[str, str]:
    """Per-field instructions for the Reflexion ``reflect`` state.

    Returns ``{field_name: instructions}`` for ``reflection`` and ``lessons``
    (both str). The prompt shows the evaluation feedback and the episodic
    memory, so a new episode's reflection differs from earlier ones.
    """
    return {
        "reflection": (
            f"{_COMPOSE}The last evaluation found the answer insufficient (see "
            "evaluation_feedback). Critique what went wrong in this attempt "
            "and what to try differently. Do not repeat a reflection already "
            "in episodic_memory."
        ),
        "lessons": (
            f"{_COMPOSE}One short lesson to remember for the next attempt, based on "
            "evaluation_feedback and not already listed in episodic_memory."
        ),
    }


# ---------------------------------------------------------------------------
# Plan-and-Execute prompts
# ---------------------------------------------------------------------------


def _tool_section(registry: ToolRegistry | None, task_description: str | None) -> str:
    """The registry's tool list as a prompt section, or "" with no tools."""
    if registry is None or len(registry) == 0:
        return ""
    tool_text, _ = _get_tool_list(registry, task_description)
    return "\n\n" + tool_text


def build_plan_steps_instructions(
    registry: ToolRegistry | None = None,
    task_description: str | None = None,
    *,
    replan: bool = False,
) -> str:
    """Per-field instructions for the typed ``plan_steps`` list.

    Used by the ``plan`` state and, with ``replan=True``, by ``replan``,
    whose prompt also shows ``step_results`` and ``previous_plan_steps``.
    """
    limit = Defaults.MAX_PLAN_STEPS
    if replan:
        ask = (
            "A step of the previous plan (previous_plan_steps) failed; its "
            "tool result is in step_results. Write a revised list of the "
            "remaining step descriptions that avoids that failure"
        )
    else:
        ask = "Break the task into a list of concrete, actionable step descriptions"
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-055
    # Without these two rules qwen3.5:4b added plan steps that need no tool
    # ("Combine", "Compare", "Synthesize", "Draft") and steps that confirm or
    # wait, each costing extra execute turns (compare task 30 calls vs 22 at
    # d4b1626). Do NOT add "one tool call per step" (it split every item into
    # its own step) or "plan exactly the task's listed steps" (the planner then
    # copied the task and dropped work), and do NOT name a loop, a signal or a
    # message to go on (D-031). See decisions.md D-055.
    no_tool_rule = ""
    if registry is not None and len(registry) > 0:
        no_tool_rule = (
            " Add no step that needs no tool (such as combining, comparing, "
            "synthesizing, drafting or reviewing results already gathered) "
            "unless the task asks for one: the final answer is written from "
            "the step results after the last step."
        )
    return (
        f"{ask}: a JSON list of strings, in order, at most {limit} steps. Each "
        f"step is self-contained and produces a clear result.{no_tool_rule} "
        "Add no step that confirms, waits for or asks for anything."
        f"{_tool_section(registry, task_description)}"
    )


def build_execute_step_instructions(
    registry: ToolRegistry | None = None,
    task_description: str | None = None,
    *,
    step_result: bool = False,
) -> str:
    """Per-field instructions for the ``execute_step`` state's fields.

    Shared by ``step_result`` and the typed tool selection (``tool_name``
    names a listed tool or "none"; ``tool_input`` is its parameter object).
    ``step_result=True`` opens with :data:`_COMPOSE` (a generated value; a
    tool observation still wins over it when a tool ran).
    """
    return (
        f"{_COMPOSE if step_result else ''}"
        "Carry out the current plan step (current_step_description), using "
        "the results of earlier steps (step_results). With a tool, "
        'tool_name is the tool to call (or "none" if no tool is needed) and '
        "tool_input its parameters; step_result is what the step produces."
        f"{_tool_section(registry, task_description)}"
    )


def build_synthesize_response_instructions() -> str:
    """Build response instructions for the synthesize state."""
    return (
        "Present a clear, complete answer that integrates results from all "
        "plan steps. Reference specific step results as evidence."
    )


# ---------------------------------------------------------------------------
# REWOO prompts
# ---------------------------------------------------------------------------


def build_rewoo_plan_extraction_instructions(
    registry: ToolRegistry,
    task_description: str | None = None,
) -> str:
    """Build extraction instructions for the REWOO plan_all state."""
    tool_list, _ = _get_tool_list(registry, task_description)

    return "\n".join(
        [
            "Analyze the task and create a COMPLETE plan of all tool calls needed "
            "to solve it. You must plan everything upfront in a single pass.",
            "",
            tool_list,
            "",
            "Each plan step can reference the output of a previous step using "
            "#E1, #E2, etc. (where the number matches the plan_id).",
            "",
            "Extract the following as JSON:",
            '- "plan_blueprint": a JSON list of plan step objects, each with:',
            '  - "plan_id": integer step number starting from 1 (e.g. 1, 2, 3)',
            '  - "description": what this step accomplishes',
            '  - "tool_name": which tool to call',
            '  - "tool_input": parameters for the tool (may contain #E1, #E2 references)',
            '- "reasoning": your overall reasoning for this plan',
            "",
            "Example plan_blueprint:",
            "[",
            '  {"plan_id": 1, "description": "Search for X", '
            '"tool_name": "search", "tool_input": {"query": "X"}},',
            '  {"plan_id": 2, "description": "Search for Y using result of step 1", '
            '"tool_name": "search", "tool_input": {"query": "Y context: #E1"}}',
            "]",
        ]
    )


def build_rewoo_plan_field_instructions(
    registry: ToolRegistry,
    task_description: str | None = None,
) -> str:
    """Instructions for the REWOO ``plan_blueprint`` typed field.

    The plan is generated, so the text opens with :data:`_COMPOSE`, then the
    tool list, reference rules and example of
    :func:`build_rewoo_plan_extraction_instructions` (fix 21.1 of plan
    06a5ec0a).
    """
    return _COMPOSE + build_rewoo_plan_extraction_instructions(
        registry, task_description=task_description
    )


def build_rewoo_solve_response_instructions() -> str:
    """Build response instructions for the REWOO solve state."""
    return (
        "Present your final answer clearly, referencing the evidence gathered "
        "from the planned tool executions."
    )


# ---------------------------------------------------------------------------
# Evaluator-Optimizer prompts
# ---------------------------------------------------------------------------


def build_evalopt_field_instructions() -> dict[str, str]:
    """Per-field instructions for EvalOpt's typed ``generated_output``.

    Returns ``{state: instructions}`` for ``generate`` and ``refine``. The
    refine prompt shows ``previous_output`` and ``refinement_feedback``.
    """
    return {
        EvalOptStates.GENERATE: (
            f"{_COMPOSE}Your complete output for the task: the full "
            "deliverable itself, exactly as it should be delivered, with no "
            "commentary before or after it."
        ),
        EvalOptStates.REFINE: (
            f"{_COMPOSE}Your complete refined output for the task: rewrite "
            "previous_output so that it fixes every point of "
            "refinement_feedback. Give the full deliverable, not only the "
            "changes, with no commentary before or after it."
        ),
    }


def build_evalopt_output_response_instructions() -> str:
    """Build response instructions for the EvalOpt output state."""
    return "Present the final, evaluated output as your answer."


# ---------------------------------------------------------------------------
# Maker-Checker prompts
# ---------------------------------------------------------------------------


def build_maker_checker_field_instructions(
    maker_instructions: str,
    checker_instructions: str,
) -> dict[str, str]:
    """Per-field instructions for the Maker-Checker typed fields.

    Returns ``{name: instructions}`` for ``make`` and ``revise`` (both the
    ``draft_output`` any field of that state), ``checker_feedback`` (str),
    ``quality_score`` (float) and ``checker_passed`` (bool). The revise
    prompt shows ``previous_draft`` and ``checker_feedback``; the check
    prompts show ``draft_output``.
    """
    draft = (
        "Give the full deliverable itself, with no commentary before or after it. "
        f"Maker instructions: {maker_instructions}"
    )
    criteria = f"Evaluation criteria: {checker_instructions}"
    return {
        MakerCheckerStates.MAKE: f"{_COMPOSE}Your complete draft for the task. {draft}",
        MakerCheckerStates.REVISE: (
            f"{_COMPOSE}Your complete revised draft for the task: rewrite "
            "previous_draft so that it fixes every point of checker_feedback "
            f"(the whole draft, not only the changes). {draft}"
        ),
        ContextKeys.CHECKER_FEEDBACK: (
            f"{_COMPOSE}Your review of the draft_output value as the checker: "
            "what is good and each specific issue to fix. " + criteria
        ),
        "quality_score": (
            "Your rating of the draft_output value against the criteria, a "
            "number from 0.0 (unusable) to 1.0 (meets every criterion). " + criteria
        ),
        ContextKeys.CHECKER_PASSED: (
            "true if the draft_output value meets the criteria, false if it "
            "needs revision. " + criteria
        ),
    }


def build_maker_checker_output_response_instructions() -> str:
    """Build response instructions for the Maker-Checker output state."""
    return "Present the final, reviewed output as your answer."


# ---------------------------------------------------------------------------
# Orchestrator-Workers prompts
# ---------------------------------------------------------------------------


def build_orchestrate_response_instructions() -> str:
    """Build response instructions for the orchestrate state."""
    return (
        "Explain how you are decomposing the task into subtasks and "
        "your delegation strategy."
    )


def build_delegate_response_instructions() -> str:
    """Build response instructions for the delegate state."""
    return (
        "Summarize which subtasks were delegated and report the status "
        "of each worker execution."
    )


def build_collect_response_instructions() -> str:
    """Build response instructions for the collect state."""
    return (
        "Summarize the worker results and explain whether all needed "
        "information has been gathered."
    )


def build_orchestrator_field_instructions() -> dict[str, str]:
    """Per-field instructions for the orchestrator's typed fields.

    Returns ``{field_name: instructions}`` for ``subtasks`` (orchestrate) and
    ``all_collected`` (collect). Both prompts show the task and
    ``worker_results`` only (fix 21.1 of plan 06a5ec0a). ``subtasks`` is
    generated, so it opens with :data:`_COMPOSE`; the ``all_collected``
    wording is permissive, like the debate judge's (D-035): a false sends
    the run back to orchestrate for another delegation round.
    """
    return {
        ContextKeys.SUBTASKS: (
            f"{_COMPOSE}Your decomposition of the task: a JSON list of subtask "
            "description strings, each specific, self-contained and actionable "
            "for one worker. If worker_results holds results from earlier "
            "rounds, list only the work those results do not cover yet."
        ),
        ContextKeys.ALL_COLLECTED: (
            "Weigh the worker_results value against the task. true when those "
            "results are enough to write a useful final answer, even if some "
            "detail is thin; false only when a clearly required part of the "
            "task has no result at all."
        ),
    }


def build_orchestrator_synthesize_response_instructions() -> str:
    """Build response instructions for the orchestrator synthesize state."""
    return (
        "Present a clear, complete answer that integrates results from all "
        "workers. Reference specific worker results as evidence."
    )


# ---------------------------------------------------------------------------
# ADaPT prompts
# ---------------------------------------------------------------------------


def build_attempt_response_instructions() -> str:
    """Build response instructions for the ADaPT attempt state."""
    return (
        "Present your attempt at solving the task. Be thorough but note "
        "any areas of uncertainty."
    )


def build_assess_response_instructions() -> str:
    """Build response instructions for the ADaPT assess state."""
    return (
        "Explain your assessment of the attempt quality. "
        "If the attempt failed, explain what went wrong."
    )


def build_decompose_response_instructions() -> str:
    """Build response instructions for the ADaPT decompose state."""
    return (
        "Explain how you are breaking the task into subtasks and "
        "why this decomposition should help solve the problem."
    )


def build_adapt_field_instructions(
    registry: ToolRegistry | None = None,
    task_description: str | None = None,
) -> dict[str, str]:
    """Per-field instructions for the ADaPT states' typed fields.

    Returns ``{field_name: instructions}`` for ``attempt_result`` (attempt),
    ``attempt_succeeded`` (assess), ``subtasks`` and ``operator`` (decompose);
    fix 21.1 of plan 06a5ec0a. The generated values open with
    :data:`_COMPOSE`; the ``attempt_succeeded`` wording is permissive (a false
    costs a whole decomposition). ``registry``/``task_description`` add the
    tool list to the ``attempt_result`` instructions.
    """
    tool_section = ""
    if registry is not None and len(registry) > 0:
        tool_text, _ = _get_tool_list(registry, task_description)
        tool_section = "\n\n" + tool_text
    return {
        ContextKeys.ATTEMPT_RESULT: (
            f"{_COMPOSE}Your direct attempt at the task: a complete answer "
            "written from your own knowledge." + tool_section
        ),
        ContextKeys.ATTEMPT_SUCCEEDED: (
            "Weigh the attempt_result value against the task. true when it "
            "answers the task usefully and completely enough to hand over, "
            "even if it could be polished; false only when it is missing, "
            "off-task, or leaves a clearly required part of the task "
            "unanswered."
        ),
        ContextKeys.SUBTASKS: (
            f"{_COMPOSE}The attempt_result value was judged insufficient. Your "
            "decomposition of the task: a JSON list of short subtask "
            "description strings, each simpler than the task and solvable on "
            "its own, together covering what the attempt missed."
        ),
        ContextKeys.OPERATOR: (
            "How the subtasks of your decomposition combine. 'AND' when every "
            "subtask is needed to answer the task (their results are combined); "
            "'OR' when any single subtask would answer the task on its own (the "
            "first one that succeeds is enough). Answer with exactly AND or OR."
        ),
    }


def build_combine_response_instructions() -> str:
    """Build response instructions for the ADaPT combine state."""
    return (
        "Present your final answer clearly and completely, integrating "
        "all available results and evidence."
    )


# ---------------------------------------------------------------------------
# Prompt Chain prompts
# ---------------------------------------------------------------------------


def build_chain_step_field_instructions(
    index: int,
    name: str,
    response_instructions: str,
    extraction_instructions: str,
) -> str:
    """Per-field instructions for step ``index`` (0-based) of a prompt chain.

    ``chain_step_result`` (any) is the step's output. The ChainStep's
    ``response_instructions`` say what the step does and its
    ``extraction_instructions`` what the output must hold; the prompt shows
    ``chain_results`` (the earlier steps' outputs).
    """
    return (
        f"{_COMPOSE}The complete output of pipeline step {index + 1} "
        f"('{name}') as plain text (not a JSON object), building on the "
        f"earlier steps' outputs in chain_results. Step: {response_instructions} "
        f"The output must cover: {extraction_instructions}"
    )


def build_chain_output_response_instructions() -> str:
    """Build response instructions for the chain output (terminal) state."""
    return (
        "Present the final output of the pipeline. Integrate and summarize "
        "the results from all preceding steps."
    )


# ---------------------------------------------------------------------------
# Self-Consistency prompts
# ---------------------------------------------------------------------------


def build_generate_response_instructions() -> str:
    """Build response instructions for the self-consistency generate state.

    The closing ``Answer:`` line is what ``self_consistency._majority_vote``
    counts, so samples that agree in different prose still agree.
    """
    return (
        "Reply in two parts: first your reasoning in one or two sentences, "
        "then a last line that is exactly 'Answer: <your answer>', stating "
        "only the answer. Always include that last line."
    )


# ---------------------------------------------------------------------------
# Debate prompts
# ---------------------------------------------------------------------------


def _persona_line(persona: str) -> str:
    """Return the ``Role:`` sentence for a debate persona, or ``""`` when unset."""
    return f" Role: {persona}" if persona else ""


def build_debate_field_instructions(
    proposer_persona: str = "",
    critic_persona: str = "",
    judge_persona: str = "",
    max_rounds: int = 3,
) -> dict[str, str]:
    """Per-field instructions for the debate states' typed fields.

    Returns ``{field_name: instructions}`` for ``proposition`` (propose),
    ``critique`` (critique), ``counter_argument`` (counter), ``judge_verdict``
    and ``consensus_reached`` (judge). Each names the context values its
    prompt shows, so a round is argued against this round's text. The ``consensus_reached`` wording is permissive
    (plan 06a5ec0a D-035): the judge handler still caps the rounds. The four
    text fields open with :data:`_COMPOSE`: live, qwen3.5:4b answered them
    null when asked only to extract.
    """
    return {
        ContextKeys.PROPOSITION: (
            f"{_COMPOSE}Your proposition: a clear position on the task with its "
            "supporting arguments. If debate_rounds holds earlier rounds, "
            "improve the latest proposition using its critique and "
            f"counter-argument.{_persona_line(proposer_persona)}"
        ),
        ContextKeys.CRITIQUE: (
            f"{_COMPOSE}Your critique of the proposition value: its weaknesses, logical "
            "gaps, missing evidence and counterexamples."
            f"{_persona_line(critic_persona)}"
        ),
        ContextKeys.COUNTER_ARGUMENT: (
            f"{_COMPOSE}Your counter-arguments to the critique value: answer each of its "
            "points, concede the valid ones and strengthen the proposition."
            f"{_persona_line(proposer_persona)}"
        ),
        ContextKeys.JUDGE_VERDICT: (
            f"{_COMPOSE}Your verdict on this round: weigh the proposition, critique and "
            "counter_argument values and name the strongest points of each."
            f"{_persona_line(judge_persona)}"
        ),
        ContextKeys.CONSENSUS_REACHED: (
            "Weigh the proposition, critique and counter_argument values. "
            "true when the proposition, as defended in the counter-argument, "
            "is a satisfactory answer to the task; false only when another "
            f"round (at most {max_rounds} in total; current_round is this "
            "round's number) would clearly improve it. debate_rounds holds "
            "the earlier rounds: true also when this round no longer changes "
            "the position materially."
            f"{_persona_line(judge_persona)}"
        ),
    }


def build_debate_conclude_response_instructions() -> str:
    """Build response instructions for the debate conclude state."""
    return (
        "Give the final, definitive answer to the task from the debate: state "
        "the answer first, then the key arguments from debate_rounds that "
        "shaped it."
    )
