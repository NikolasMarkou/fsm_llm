"""The exception hierarchy of the core: ``FSMError`` and its subclasses."""

from __future__ import annotations

from typing import Any, ClassVar, Literal


class FSMError(Exception):
    """Base exception for FSM-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class ConversationBusyError(FSMError):
    """A turn held the conversation's lock past
    ``END_CONVERSATION_LOCK_TIMEOUT_SECONDS``, so ending it was refused.
    Nothing was changed; retry after the turn completes."""

    def __init__(self, message: str, conversation_id: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.conversation_id = conversation_id


class RunBudgetExceededError(FSMError):
    """A bounded run (``API.run_until_terminal`` or its stream form) spent a
    budget before the conversation ended.

    Raised before the step that would exceed the budget, so every step already
    run is kept (nothing is rolled back) and the conversation stays usable.
    A state that stays BLOCKED on every step ends here too. The error does not
    carry the ``AdvanceResult`` of the completed steps: read their replies from
    ``API.get_conversation_history`` and the position from
    ``API.get_current_state`` / ``API.get_data``.

    Attributes:
        budget: ``"steps"`` (``max_steps``) or ``"seconds"`` (``max_seconds``).
        limit: the value of the budget that was spent.
        steps_done: steps the run completed before it stopped.
    """

    def __init__(
        self, budget: Literal["steps", "seconds"], limit: float, steps_done: int
    ):
        super().__init__(
            f"Run stopped before the conversation ended: {budget} budget of "
            f"{limit} spent after {steps_done} step(s)",
            details={"budget": budget, "limit": limit, "steps_done": steps_done},
        )
        self.budget = budget
        self.limit = limit
        self.steps_done = steps_done

    def __reduce__(self):
        # Pickling replays ``cls(*self.args)``; ``args`` holds the formatted
        # message, not the three arguments. Rebuild from the real ones.
        return (self.__class__, (self.budget, self.limit, self.steps_done))


class BuildError(FSMError, ValueError):
    """A builder's ``build()`` refused its accumulated configuration.

    Interface contract: ``BuildError(message, errors=None, **kwargs)``.
    ``errors`` is the list of problem strings (defaults to ``[message]``);
    ``kwargs`` (e.g. ``details``) go to ``FSMError``. A domain or constructor
    error that caused the failure is chained as ``__cause__``.
    """

    # DECISION plan-2026-10-02T052921-89b03f61/D-003
    # Do NOT make this a separate error family per subpackage, and do NOT drop
    # the ValueError base: existing ``pytest.raises(ValueError, match=...)`` and
    # ``except ValueError`` on builders must keep passing. One class, one
    # ``except BuildError`` for every builder. See decisions.md D-003.
    def __init__(self, message: str, errors: list[str] | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.errors = list(errors) if errors is not None else [message]


class ModelNotConfiguredError(FSMError, ValueError):
    """No LLM model was named anywhere: not by argument, not by ``LLM_MODEL``.

    Interface contract: ``ModelNotConfiguredError(message=None, details=None)``.
    ``message`` defaults to ``ModelNotConfiguredError.DEFAULT_MESSAGE`` (one
    line, names the three ways to set a model, names no concrete model), so
    every raiser shares one text. Raised by ``llm.resolve_model`` and so by every constructor that
    builds a real interface (``API``, ``AgentConfig``, ...); an injected
    ``llm_interface`` needs no model and never raises it. It is a
    ``ValueError`` too, so a REST layer that maps ``ValueError`` to a 400 shows
    the message as is.
    """

    # DECISION plan-2026-11-12T073150-d454fd62/D-005
    # Do NOT replace this error with a built-in default model: a fresh install
    # must fail with a clear message, never download or call a model nobody
    # chose. Do NOT drop the ValueError base (the monitor maps it to a 400).
    DEFAULT_MESSAGE: ClassVar[str] = (
        'No LLM model is set. Pass model="provider/name", or set the LLM_MODEL '
        "environment variable, or call fsm_llm.load_env() (CLI: --env-file PATH) "
        "to read LLM_MODEL from a .env file."
    )

    def __init__(
        self, message: str | None = None, details: dict[str, Any] | None = None
    ):
        super().__init__(self.DEFAULT_MESSAGE if message is None else message, details)


class OKFError(FSMError):
    """An Open Knowledge Format operation was refused (``fsm_llm.okf``).

    Raised for a missing or unreadable bundle root and for a write that would
    leave the target directory or replace a file unasked. Bundle content never
    raises it: a malformed concept becomes an ``OKFIssue`` instead.
    """


class SkillError(FSMError):
    """An Agent Skills operation was refused (``fsm_llm.skills``).

    Raised for caller misuse: a missing or non-directory skills root, no path
    given, an unknown skill name, or a body or resource read that is refused
    (a path that leaves the skill folder, a binary or oversize file). Skill
    content never raises it: a malformed skill becomes a ``SkillIssue`` instead.
    """


class AgentError(FSMError):
    """Base exception for all agent-related errors."""


class FSMDefinitionNotFoundError(FSMError, ValueError):
    """No FSM definition is resolvable for a non-path id.

    Raised by ``core.loaders.load_fsm_definition`` (and so by the ``API`` loader
    fallback) for an id that is neither a file path nor cached: there is no
    FSM registry. Also a ``ValueError`` so existing handlers keep catching it.
    """

    def __init__(self, fsm_id: str):
        super().__init__(
            f"Unknown FSM ID '{fsm_id}': no FSM registry: only file paths are "
            f"loadable; id '{fsm_id}' is not cached",
            details={"fsm_id": fsm_id},
        )
        self.fsm_id = fsm_id

    def __reduce__(self):
        # Pickling replays ``cls(*self.args)``; ``args`` holds the formatted
        # message, not ``fsm_id``. Rebuild from the real argument.
        return (self.__class__, (self.fsm_id,), self.__dict__.copy())


class StateNotFoundError(FSMError):
    """Exception for non-existent state references."""

    def __init__(self, message: str, state_id: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.state_id = state_id


class InvalidTransitionError(FSMError):
    """Exception for invalid state transitions."""

    def __init__(
        self,
        message: str,
        source_state: str | None = None,
        target_state: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.source_state = source_state
        self.target_state = target_state


class LLMResponseError(FSMError):
    """Exception for LLM response processing errors."""

    pass


class TransitionEvaluationError(FSMError):
    """Exception for transition evaluation errors."""

    def __init__(self, message: str, state_id: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.state_id = state_id


class ClassificationError(FSMError):
    """Base exception for classification operations."""

    pass


class ClassificationResponseError(ClassificationError):
    """Raised when the LLM returns an unparseable classification."""

    pass
