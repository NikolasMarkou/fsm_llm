"""Meta-builder FSM (``build_meta_builder_fsm``)."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents import meta_prompts
from fsm_llm.agents.constants import (
    META_ARTIFACT_TYPE_INTENTS,
    META_HANDLER_ONLY_KEYS,
    META_UNKNOWN_ARTIFACT_TYPE,
    META_UNKNOWN_ARTIFACT_TYPE_DESCRIPTION,
    MetaBuilderStates,
    MetaBuildOutcome,
    MetaContextKeys,
    MetaDefaults,
)

from ._common import _by_id, _edge, _finalize_fsm, _flag_condition, _state


def build_meta_builder_fsm() -> dict[str, Any]:
    """Build the meta-builder FSM: classify, collect, build, build_failed, done.

    Contract: returns a v4.1 FSM definition dict that loads as
    ``FSMDefinition`` (no arguments; the artifact type, requirements and
    build request live in context, never in the definition). States
    (``MetaBuilderStates``):

    - ``classify`` (initial, silent): ``classification_extractions`` writes
      ``artifact_type`` (fsm, workflow, agent; fallback ``unknown``, which
      the driver's exit handler resolves to a type), reading
      ``latest_request`` as its context so a message-free step classifies
      too. ``-> build`` p10 when ``build_requested`` is True, else
      ``-> collect`` p900.
    - ``collect`` (speaking, no extraction): the reply asks for details and
      ends with the build prompt; it sees only ``artifact_type``,
      ``requirements`` and ``validation_errors``. ``-> classify`` p10 on
      ``type_switch``, ``-> build`` p20 on ``build_requested`` (a message
      that does both is reclassified first, then ``classify -> build``),
      otherwise BLOCKED (stay).
    - ``build`` (silent completion state): one structured completion over
      ``_build_messages`` with the response format in
      ``_build_response_format``, result in ``build_reply``. ``-> done`` p10
      when ``build_outcome == "valid"``, else ``-> build_failed`` p900.
    - ``build_failed`` (silent): ``-> classify`` p10 on ``type_switch``,
      ``-> build`` p20 on ``build_requested``, ``-> collect`` p900.
    - ``done`` (terminal, silent).

    The driver writes ``requirements``, ``latest_request``,
    ``build_requested``, ``type_switch`` and ``keyword_type`` before each
    turn; handlers write the build request on ``build`` entry and
    ``build_outcome``, ``artifact``, ``validation_errors`` and
    ``review_presentation`` from ``build_reply``. All of those are
    ``handler_only_keys`` (``META_HANDLER_ONLY_KEYS``). Never raises.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-011: five states with the
    # artifact type in context, and the build as a core completion state.
    # Do NOT split this into per-artifact-type states (three copies of the
    # same transitions), do NOT make the build an LLM call inside a handler
    # (a second call path beside Pass 1, outside rollback and the meter), and
    # do NOT replace the driver-computed gate keys (build_requested,
    # type_switch) with a per-turn classifier: JsonLogic has no phrase
    # predicate and a classifier per turn adds the calls A-ISSUE-011 removed.
    # The gate keys and the build verdict stay handler-only so the model
    # cannot forge them. See decisions.md D-011.
    states_ = MetaBuilderStates
    keys = MetaContextKeys
    build_requested = _flag_condition(
        keys.BUILD_REQUESTED, "The user asked to build the artifact now"
    )
    type_switch = _flag_condition(
        keys.TYPE_SWITCH, "The user asked for a different artifact type"
    )
    states = _by_id(
        _state(
            states_.CLASSIFY,
            "Decide which kind of artifact the user wants",
            "Classify the request as an FSM, a workflow or an agent",
            [
                _edge(
                    states_.BUILD,
                    "The user asked to build right away",
                    10,
                    build_requested,
                ),
                _edge(states_.COLLECT, "Collect the artifact's requirements", 900),
            ],
            extraction_instructions="",
            # DECISION plan-2026-10-01T093600-944e2692/D-035: the fallback is
            # `unknown`, not an artifact type. Do NOT make `fsm` the fallback
            # again: core stores the fallback at any confidence, so a vague
            # edit ("change the tool name") classified `fsm` at 0.1 flipped an
            # agent session to an FSM, and `fsm` meant both "an FSM" and "no
            # idea". The classify exit handler resolves `unknown` (and a
            # discarded low-confidence intent) to the previous type, else the
            # keyword type. See decisions.md D-035.
            classification_extractions=[
                {
                    "field_name": keys.ARTIFACT_TYPE,
                    "intents": [
                        {"name": name, "description": description}
                        for name, description in (
                            *META_ARTIFACT_TYPE_INTENTS,
                            (
                                META_UNKNOWN_ARTIFACT_TYPE,
                                META_UNKNOWN_ARTIFACT_TYPE_DESCRIPTION,
                            ),
                        )
                    ],
                    "fallback_intent": META_UNKNOWN_ARTIFACT_TYPE,
                    "confidence_threshold": MetaDefaults.TYPE_CONFIDENCE_THRESHOLD,
                    "context_keys": [keys.LATEST_REQUEST],
                }
            ],
        ),
        _state(
            states_.COLLECT,
            "Gather the artifact's requirements from the user",
            (
                "Acknowledge each detail, ask about what is unclear and "
                "invite the user to say 'build it'"
            ),
            [
                _edge(
                    states_.CLASSIFY,
                    "The user switched to another artifact type",
                    10,
                    type_switch,
                ),
                _edge(
                    states_.BUILD,
                    "The user asked to build the artifact",
                    20,
                    build_requested,
                ),
            ],
            response_instructions=meta_prompts.build_meta_collect_response_instructions(),
            extraction_instructions="",
            context_scope={
                "read_keys": [
                    keys.ARTIFACT_TYPE,
                    keys.REQUIREMENTS,
                    keys.VALIDATION_ERRORS,
                ]
            },
        ),
        _state(
            states_.BUILD,
            "Generate the artifact spec in one structured call",
            (
                "Ask the model for the whole artifact spec as JSON matching "
                "the per-type schema"
            ),
            [
                _edge(
                    states_.DONE,
                    "The assembled artifact validated",
                    10,
                    {
                        "description": "The build produced a valid artifact",
                        "requires_context_keys": [keys.BUILD_OUTCOME],
                        "logic": {
                            "===": [{"var": keys.BUILD_OUTCOME}, MetaBuildOutcome.VALID]
                        },
                    },
                ),
                _edge(
                    states_.BUILD_FAILED,
                    "The build did not produce a valid artifact",
                    900,
                ),
            ],
            completion={
                "response_format_key": keys.BUILD_RESPONSE_FORMAT,
                "messages_key": keys.BUILD_MESSAGES,
                "result_key": keys.BUILD_REPLY,
            },
        ),
        _state(
            states_.BUILD_FAILED,
            "The last build failed validation",
            (
                "Route the user's next message: build again, switch the "
                "artifact type, or collect more details"
            ),
            [
                _edge(
                    states_.CLASSIFY,
                    "The user switched to another artifact type",
                    10,
                    type_switch,
                ),
                _edge(
                    states_.BUILD, "The user asked to build again", 20, build_requested
                ),
                _edge(
                    states_.COLLECT, "Collect more details before the next build", 900
                ),
            ],
            extraction_instructions="",
        ),
        _state(
            states_.DONE,
            "The artifact is built and valid",
            "End the session with the valid artifact",
        ),
    )
    fsm = _finalize_fsm(
        "meta_builder",
        "",
        (
            "Meta-builder: classify the artifact type, collect requirements, "
            "build the artifact with one structured completion"
        ),
        states_.CLASSIFY,
        (
            "A concise assistant that helps users design FSM-LLM artifacts: "
            "FSMs, workflows and agents"
        ),
        states,
    )
    fsm["handler_only_keys"] = list(META_HANDLER_ONLY_KEYS)
    return fsm
