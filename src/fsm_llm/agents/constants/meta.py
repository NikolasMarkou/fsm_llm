"""Meta-builder constants: defaults, states, context keys, build outcomes, driver words and messages."""

from __future__ import annotations

from .core import Defaults


class MetaDefaults:
    """Default configuration values for the meta-builder."""

    TEMPERATURE = 0.7
    MAX_TOKENS = 4096
    MAX_TURNS = 50

    # Builder defaults
    DEFAULT_PRIORITY = 100
    SUMMARY_TRUNCATE_WIDTH = 80
    SUMMARY_LINE_WIDTH = 40  # the description beside a name in a standard summary

    # Artifact-type classification (the ``classify`` state of the meta FSM)
    TYPE_CONFIDENCE_THRESHOLD = 0.4

    # Agent builder defaults (for the agent artifact being built)
    AGENT_MODEL = "gpt-4o-mini"
    AGENT_MAX_ITERATIONS = Defaults.MAX_ITERATIONS
    AGENT_TIMEOUT_SECONDS = Defaults.TIMEOUT_SECONDS
    AGENT_TEMPERATURE = Defaults.TEMPERATURE
    AGENT_MAX_TOKENS = Defaults.MAX_TOKENS


class MetaBuilderStates:
    """States in the meta-builder FSM (``build_meta_builder_fsm``)."""

    CLASSIFY = "classify"
    COLLECT = "collect"
    BUILD = "build"
    BUILD_FAILED = "build_failed"
    DONE = "done"


class MetaContextKeys:
    """Context keys of the meta-builder FSM.

    ``ARTIFACT_TYPE`` is the one key the model writes (the ``classify``
    state's classification). ``BUILD_REPLY`` is the ``build`` state's
    completion result. ``BUILD_MESSAGES`` and ``BUILD_RESPONSE_FORMAT`` are
    the build request's transcript and response format (internal: no prompt,
    extraction or ``get_data`` reaches them). Every other key is written only
    by the driver or its handlers (``META_HANDLER_ONLY_KEYS``).
    """

    ARTIFACT_TYPE = "artifact_type"
    # The type before a reclassification (``classify`` entry stash, internal).
    PREVIOUS_ARTIFACT_TYPE = "_previous_artifact_type"
    # Driver hints, recomputed from each user message before the turn.
    REQUIREMENTS = "requirements"
    LATEST_REQUEST = "latest_request"
    BUILD_REQUESTED = "build_requested"
    TYPE_SWITCH = "type_switch"
    KEYWORD_TYPE = "keyword_type"
    # Build state: request, raw reply and the handler-assembled outcome.
    BUILD_MESSAGES = "_build_messages"
    BUILD_RESPONSE_FORMAT = "_build_response_format"
    BUILD_REPLY = "build_reply"
    BUILD_OUTCOME = "build_outcome"
    ARTIFACT = "artifact"
    VALIDATION_ERRORS = "validation_errors"
    REVIEW_PRESENTATION = "review_presentation"
    # The assembled builder's progress record and summary (monitor view).
    BUILD_PROGRESS = "build_progress"
    BUILD_SUMMARY = "build_summary"


class MetaBuildOutcome:
    """Values of ``MetaContextKeys.BUILD_OUTCOME``.

    ``INVALID``: the reply was assembled and failed validation. ``MALFORMED``:
    the reply is not an artifact of the requested schema (a JSON-schema echo,
    or a field of the wrong type); nothing was assembled from it.
    """

    VALID = "valid"
    INVALID = "invalid"
    MALFORMED = "malformed"


# ``kind`` of the ``build_reply`` record ``MetaBuilderAgent.send`` writes when
# the build call itself failed (provider outage): a ``build`` state whose
# result key is set makes no call, so the next step routes to ``build_failed``.
# Never a ``CompletionResponse`` kind (``calls``, ``final``, ``malformed``).
META_BUILD_CALL_FAILED = "call_failed"


# The build prompt that closes every ``collect`` reply: the last sentence of
# the collect instructions (``meta_prompts``) and the line the driver appends
# when the model dropped it (``MetaBuilderAgent``, D-024). It carries a phrase
# the driver's build trigger accepts ("build it").
META_BUILD_PROMPT = "Say 'build it' when you're ready."


# Keyword hints the meta-builder driver computes from each user message
# (``MetaBuilderAgent._turn_hints``). Every phrase matches on word boundaries.
# A message that changes the artifact type routes back to ``classify``.
META_SWITCH_WORDS: tuple[str, ...] = (
    "instead",
    "actually",
    "change",
    "switch",
    "no,",
    "not a",
)


# Whole messages that ask for the build.
META_BUILD_TRIGGERS: frozenset[str] = frozenset(
    {
        "build it",
        "build",
        "go",
        "build now",
        "create it",
        "make it",
        "generate",
        "done",
        "finish",
        "approve",
        "yes",
        "ok",
        "lgtm",
        "ship it",
        "do it",
    }
)


# Phrases that ask for the build anywhere in a message, unless a negation
# governs them: it stands right before the phrase, at most with filler words
# between ("don't build it yet", "do not ever build it"). "why not" is no
# negation ("why not build it" asks for the build).
META_BUILD_PHRASES: tuple[str, ...] = (
    "build it",
    "rebuild it",
    "create it",
    "generate it",
)


META_BUILD_NEGATIONS: tuple[str, ...] = (
    "don't",
    "dont",
    "do not",
    "not",
    "never",
    "won't",
    "can't",
    "cannot",
)


# Words that may stand between a negation and the build phrase it governs.
META_BUILD_NEGATION_FILLERS: tuple[str, ...] = (
    "ever",
    "yet",
    "just",
    "actually",
    "really",
    "even",
)


# The ``classify`` state's fallback intent: the message names no artifact
# type. Never an ``ArtifactType`` value, so a fallback or an unrecognised
# reply is told apart from a classified ``fsm`` (D-035).
META_UNKNOWN_ARTIFACT_TYPE = "unknown"


META_UNKNOWN_ARTIFACT_TYPE_DESCRIPTION = (
    "The message does not say which kind of artifact it is: a detail, an "
    "edit, a question or a vague idea"
)


class MetaHandlerNames:
    """Names of the handlers ``MetaBuilderAgent`` registers on its API."""

    CLASSIFY_ENTRY = "meta_classify_entry"
    CLASSIFY_EXIT = "meta_classify_exit"
    BUILD_ENTRY = "meta_build_entry"
    BUILD_REPLY = "meta_build_reply"


# Keys of the meta-builder FSM that only the driver and its handlers write:
# listed as core ``handler_only_keys``, so no extraction channel can plant a
# build trigger, a type switch or a build verdict. The two internal-prefixed
# build keys need no entry (core never extracts an internal key, see the
# FRAMEWORK_ONLY_KEYS anchor); ``ARTIFACT_TYPE`` is a classification field,
# which the list does not cover (core's validator warns on it).
META_HANDLER_ONLY_KEYS: tuple[str, ...] = (
    MetaContextKeys.REQUIREMENTS,
    MetaContextKeys.LATEST_REQUEST,
    MetaContextKeys.BUILD_REQUESTED,
    MetaContextKeys.TYPE_SWITCH,
    MetaContextKeys.KEYWORD_TYPE,
    MetaContextKeys.BUILD_OUTCOME,
    MetaContextKeys.ARTIFACT,
    MetaContextKeys.VALIDATION_ERRORS,
    MetaContextKeys.REVIEW_PRESENTATION,
    MetaContextKeys.BUILD_REPLY,
    MetaContextKeys.BUILD_PROGRESS,
    MetaContextKeys.BUILD_SUMMARY,
)


# Keys a build writes. Cleared on ``build`` entry (a retry starts from
# nothing) and on ``classify`` entry (they belong to the old type).
META_BUILD_OUTPUT_KEYS: tuple[str, ...] = (
    MetaContextKeys.BUILD_REPLY,
    MetaContextKeys.BUILD_OUTCOME,
    MetaContextKeys.ARTIFACT,
    MetaContextKeys.VALIDATION_ERRORS,
    MetaContextKeys.REVIEW_PRESENTATION,
    MetaContextKeys.BUILD_PROGRESS,
    MetaContextKeys.BUILD_SUMMARY,
)


# Artifact types the meta-builder classifies, with the intent descriptions the
# classifier sees (names are ``ArtifactType`` values; fallback ``fsm``).
META_ARTIFACT_TYPE_INTENTS: tuple[tuple[str, str], ...] = (
    (
        "fsm",
        "A finite state machine, chatbot, dialogue system, "
        "conversational flow, survey, quiz, FAQ bot, help desk, "
        "interview, or onboarding flow",
    ),
    (
        "workflow",
        "A multi-step workflow, data pipeline, automation, "
        "ETL process, batch job, sequential process, or "
        "async task orchestration",
    ),
    (
        "agent",
        "An AI agent that uses tools, a ReAct agent, "
        "plan-and-execute agent, research agent, browsing "
        "agent, or any agentic pattern with tool use",
    ),
)


# Agent patterns an agent artifact may use, with their descriptions (names are
# ``AgentArtifactBuilder.VALID_AGENT_TYPES``; ``react`` is the default).
META_AGENT_PATTERN_INTENTS: tuple[tuple[str, str], ...] = (
    (
        "react",
        "A ReAct agent: think-act-observe loop, tool-using "
        "agent, search agent, general-purpose agent. "
        "This is the default and most common pattern.",
    ),
    (
        "plan_execute",
        "A plan-and-execute agent: first creates a plan "
        "then executes steps sequentially, with replanning",
    ),
    (
        "reflexion",
        "A reflexion agent: attempts a task, reflects on "
        "failures, retries with improved approach",
    ),
    (
        "rewoo",
        "A REWOO agent: plans all tool calls upfront "
        "then executes them sequentially without interleaving",
    ),
    (
        "evaluator_optimizer",
        "An evaluator-optimizer agent: generates output, "
        "evaluates quality, optimizes iteratively",
    ),
    (
        "maker_checker",
        "A maker-checker agent: one agent drafts, "
        "another reviews and approves or sends back",
    ),
    (
        "debate",
        "A debate agent: multiple perspectives argue, "
        "a judge synthesizes the best answer",
    ),
    (
        "orchestrator",
        "An orchestrator agent: delegates subtasks to "
        "specialized worker agents and synthesizes results",
    ),
    (
        "adapt",
        "An ADaPT agent: estimates task complexity, "
        "decomposes if too complex, adapts strategy",
    ),
    (
        "prompt_chain",
        "A prompt chain agent: sequential prompts with quality gates between each step",
    ),
    (
        "self_consistency",
        "A self-consistency agent: generates multiple samples and uses majority voting",
    ),
)


class MetaLogMessages:
    """Standard log message templates for meta-builder."""

    META_STARTED = "Meta-agent started with model={model}"
    BUILD_STARTED = "Build phase started for {artifact_type}"


class MetaErrorMessages:
    """Standard error messages for meta-builder."""

    CONVERSATION_NOT_STARTED = "Conversation has not been started"
    CONVERSATION_ALREADY_STARTED = "Conversation has already been started"
