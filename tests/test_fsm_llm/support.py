"""Helpers shared by several test files of this suite. Not collected (no test_ prefix)."""

from __future__ import annotations

import json
import threading
from contextlib import ExitStack, contextmanager
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

from fsm_llm import API
from fsm_llm.classification import Classifier
from fsm_llm.core.definitions import (
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    CompletionRequest,
    CompletionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    IntentDefinition,
    ResponseGenerationResponse,
    State,
    Transition,
    TransitionCondition,
)
from fsm_llm.handlers import HandlerTiming
from fsm_llm.llm import LiteLLMInterface, LLMInterface


def _mock_llm() -> MagicMock:
    """A spec'd mock LLM: field extraction returns nothing, Pass 2 says "ok"."""
    llm = MagicMock(spec=LLMInterface)
    llm.model = "gpt-4"

    def _extract_field(request):
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None,
            confidence=0.0,
            reasoning="mock",
            is_valid=False,
        )

    llm.extract_field.side_effect = _extract_field
    llm.generate_response.return_value = MagicMock(
        message="ok", message_type="response", reasoning="mock"
    )
    llm.generate_response_stream.side_effect = lambda request: iter(["o", "k"])
    # The classifier sends through the conversation's own interface (D-006 of
    # plan 944e2692): forward `complete` to a real LiteLLMInterface for the
    # same model, so a patched `fsm_llm.llm.providers.litellm.completion` still receives the
    # classifier's provider request.
    llm.complete.side_effect = LiteLLMInterface(model="gpt-4").complete
    return llm


def _classification(intent: str, confidence: float) -> ClassificationResult:
    return ClassificationResult(
        reasoning=f"picked {intent}", intent=intent, confidence=confidence
    )


def _ambiguous_then_blocked_fsm(
    transition_classification: dict[str, Any] | None = None,
) -> FSMDefinition:
    """``start`` has two unconditioned, equal-priority transitions (AMBIGUOUS
    every turn). ``a`` has one transition gated on a key nobody sets, so a turn
    in ``a`` is BLOCKED and never reaches the classifier.
    """
    never = TransitionCondition(
        description="never set",
        logic={"==": [{"var": "never_set_flag"}, True]},
    )
    states = {
        "start": State(
            id="start",
            description="Start",
            purpose="Pick a branch",
            response_instructions="Respond",
            transitions=[
                Transition(target_state="a", description="Go to a"),
                Transition(target_state="b", description="Go to b"),
            ],
            transition_classification=transition_classification,
        ),
        "a": State(
            id="a",
            description="Branch a",
            purpose="Wait",
            response_instructions="Respond",
            transitions=[
                Transition(target_state="end", description="Done", conditions=[never])
            ],
        ),
        "b": State(id="b", description="Branch b", purpose="End", transitions=[]),
        "end": State(id="end", description="End", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="audit_fsm",
        description="Audit FSM",
        initial_state="start",
        states=states,
    )


def _api(fsm_def: FSMDefinition) -> tuple[API, str, MagicMock]:
    llm = _mock_llm()
    api = API.from_definition(fsm_def, llm_interface=llm)
    conv_id, _ = api.start_conversation()
    return api, conv_id, llm


_A4_TRANSITION_KEY = "transition_classification"


def _raw_data(api: API, conv_id: str) -> dict[str, Any]:
    """The unfiltered instance context (``get_data`` strips internal keys)."""
    return api.fsm_manager.instances[conv_id].context.data


def _a2_state(transitions: list[Transition]) -> State:
    return State(
        id="start",
        description="start",
        purpose="route",
        response_instructions="Respond",
        transitions=transitions,
    )


class _ClassifierCapture:
    """Patch the classifier's litellm boundary and record every ``messages``
    list it is called with; replies with a fixed intent and confidence."""

    def __init__(self, intent: str, confidence: float = 0.9):
        self.intent, self.confidence = intent, confidence
        self.calls: list[list[dict[str, str]]] = []
        self._patches = [
            patch("fsm_llm.llm.providers.litellm.completion", side_effect=self._reply),
            patch(
                "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                return_value=[],
            ),
        ]

    def _reply(self, **kwargs):
        self.calls.append(kwargs["messages"])
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(
            {
                "reasoning": "r",
                "intent": self.intent,
                "confidence": self.confidence,
                "entities": {},
            }
        )
        return resp

    def system(self, index: int = -1) -> str:
        return self.calls[index][0]["content"]

    def __enter__(self) -> _ClassifierCapture:
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc) -> None:
        for p in reversed(self._patches):
            p.stop()


def _a3_extraction_fsm(context_keys: list[str] | None = None) -> FSMDefinition:
    """``triage`` owns a classification field ``intent`` (browse is the
    fallback, so a "browse" reply keeps the conversation in ``triage``)."""
    states = {
        "triage": State(
            id="triage",
            description="Triage",
            purpose="Route the shopper to checkout or browsing",
            response_instructions="Respond",
            classification_extractions=[
                ClassificationExtractionConfig(
                    field_name="intent",
                    intents=[
                        IntentDefinition(name="buy", description="wants to buy"),
                        IntentDefinition(name="browse", description="just looking"),
                    ],
                    fallback_intent="browse",
                    confidence_threshold=0.5,
                    context_keys=context_keys,
                )
            ],
            transitions=[
                Transition(
                    target_state="done",
                    description="Buy",
                    conditions=[
                        TransitionCondition(
                            description="buy",
                            logic={"==": [{"var": "intent"}, "buy"]},
                        )
                    ],
                )
            ],
        ),
        "done": State(id="done", description="Done", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="a3_fsm", description="A3 FSM", initial_state="triage", states=states
    )


_A3_CONTEXT = {
    "topic": "running shoes",
    "other": "hidden-value-xyz",
    "password": "hunter2-secret",
    "profile": {"api_key": "sk-live-123456", "tier": "gold"},
}


def _a3_api(fsm_def: FSMDefinition) -> tuple[API, str]:
    api = API.from_definition(fsm_def, llm_interface=_mock_llm())
    conv_id, _ = api.start_conversation(initial_context=dict(_A3_CONTEXT))
    return api, conv_id


def _a6_last_prompt(mock_method: MagicMock) -> str:
    request = mock_method.call_args_list[-1].args[0]
    return request.system_prompt


def _a7_fsm(read_keys: list[str] | None = None) -> FSMDefinition:
    """``chat`` loops (Pass 2 every turn); its edge to ``vip`` is gated on
    ``tier == "gold"``, a key only WorkingMemory sets in these tests."""
    gold = TransitionCondition(
        description="tier is gold",
        requires_context_keys=["tier"],
        logic={"==": [{"var": "tier"}, "gold"]},
    )
    states = {
        "chat": State(
            id="chat",
            description="Chat",
            purpose="Talk",
            response_instructions="Respond",
            context_scope={"read_keys": read_keys} if read_keys else None,
            transitions=[
                Transition(target_state="vip", description="Gold", conditions=[gold])
            ],
        ),
        "vip": State(
            id="vip",
            description="VIP",
            purpose="Serve a VIP",
            response_instructions="Respond",
            transitions=[],
        ),
    }
    return FSMDefinition(
        name="a7_fsm", description="A7 FSM", initial_state="chat", states=states
    )


def _d3_aliased(levels: int) -> dict[str, Any]:
    """3-way aliasing per level: 3**levels paths through ``levels + 1`` dicts."""
    cur: dict[str, Any] = {"leaf": 1}
    for _ in range(levels):
        cur = {"a": cur, "b": cur, "c": cur}
    return cur


def _d3_walkers() -> dict[str, Any]:
    """The three context filters, each as ``data -> filtered``."""
    from fsm_llm.context import clean_context_keys
    from fsm_llm.context.layout import BasePromptBuilder
    from fsm_llm.engine.fsm.helpers import _strip_internal_mapping

    builder = BasePromptBuilder()
    return {
        "clean_context_keys": lambda d: clean_context_keys(d, "conv-d3"),
        "get_data": _strip_internal_mapping,
        "prompt": builder._filter_context_for_security,
    }


#: Wall-clock kill switch for a worker that must finish: about 70x the plain CPU
#: cost of the slowest d3 walk (0.85 s). Only a hang or an exponential walk
#: reaches it; the BUDGET a test asserts is CPU seconds (``_D3Run.cpu_seconds``).
_D3_HANG_S = 60.0


class _D3Run(NamedTuple):
    """Outcome of one ``fn(data)`` run in a daemon thread.

    ``finished``: the worker ended inside the wall kill switch. ``result`` /
    ``error``: what ``fn`` returned or raised (``None`` when unfinished or not
    applicable). ``cpu_seconds``: CPU time of that thread only (``inf`` when it
    did not finish): CPU time does not inflate while a loaded host schedules
    another process, so it is the right measure of a CPU-bound walk.
    """

    finished: bool
    result: Any
    error: BaseException | None
    cpu_seconds: float


def _d3_run_in_thread(fn: Any, data: Any, hang_seconds: float = _D3_HANG_S) -> _D3Run:
    """Run ``fn(data)`` in a daemon thread, join for at most *hang_seconds* of
    wall time (a kill switch, never the budget) and report the thread's CPU time.

    Interface contract: never raises for what ``fn`` raises (any ``BaseException``
    lands in ``error``); a worker still running at the join is left behind as a
    daemon thread and reported ``finished=False``.
    """
    import threading

    from tests.conftest import cpu_timer

    box: dict[str, Any] = {}

    def _target() -> None:
        with cpu_timer(thread=True) as timer:
            try:
                box["result"] = fn(data)
            except BaseException as exc:
                box["error"] = exc
        box["cpu"] = timer.elapsed

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(hang_seconds)
    finished = not worker.is_alive()
    return _D3Run(
        finished,
        box.get("result"),
        box.get("error"),
        box.get("cpu", float("inf")) if finished else float("inf"),
    )


def _d3_run_bounded(
    fn: Any, data: Any, cpu_seconds: float, hang_seconds: float = _D3_HANG_S
) -> tuple[bool, Any]:
    """Run ``fn(data)`` in a daemon thread; ``(finished, result)``.

    ``finished`` means the worker ended AND used at most *cpu_seconds* of CPU
    time (callers multiply the budget by ``timing_factor``); the wall clock only
    guards against a hang (*hang_seconds*).
    """
    run = _d3_run_in_thread(fn, data, hang_seconds)
    return (run.finished and run.cpu_seconds <= cpu_seconds, run.result)


def _c3_fsm() -> FSMDefinition:
    """``start`` -> ``mid`` deterministically; ``mid`` is BLOCKED forever."""
    never = TransitionCondition(
        description="never set",
        logic={"==": [{"var": "never_set_flag"}, True]},
    )
    states = {
        "start": State(
            id="start",
            description="Start",
            purpose="Begin",
            response_instructions="Respond",
            transitions=[Transition(target_state="mid", description="Go on")],
        ),
        "mid": State(
            id="mid",
            description="Mid",
            purpose="Wait",
            response_instructions="Respond",
            transitions=[
                Transition(target_state="end", description="Done", conditions=[never])
            ],
        ),
        "end": State(id="end", description="End", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="c3_fsm", description="C3 FSM", initial_state="start", states=states
    )


@contextmanager
def _c_log_capture():
    """Enable the library logger; collect every record as ``LEVEL|message``."""
    from loguru import logger

    messages: list[str] = []
    logger.enable("fsm_llm")
    hid = logger.add(
        lambda m: messages.append(f"{m.record['level'].name}|{m.record['message']}"),
        level="DEBUG",
    )
    try:
        yield messages
    finally:
        logger.remove(hid)
        logger.disable("fsm_llm")


_JOIN_TIMEOUT_S = 10.0


def _blocked_fsm() -> FSMDefinition:
    """``start`` waits on a key nobody sets, so every turn is BLOCKED."""
    never = TransitionCondition(
        description="never set",
        logic={"==": [{"var": "never_set_flag"}, True]},
    )
    states = {
        "start": State(
            id="start",
            description="Start",
            purpose="Wait",
            response_instructions="Respond",
            transitions=[
                Transition(target_state="end", description="Done", conditions=[never])
            ],
        ),
        "end": State(id="end", description="End", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="audit_2026_09_22",
        description="Blocked FSM for audit regression tests",
        initial_state="start",
        states=states,
    )


def _a2026_09_22_api() -> API:
    return API.from_definition(_blocked_fsm(), llm_interface=_mock_llm())


def _join_all(threads: list[threading.Thread]) -> None:
    for t in threads:
        t.join(timeout=_JOIN_TIMEOUT_S)
    assert not any(t.is_alive() for t in threads), "a thread hung"


OLLAMA_MODEL = "ollama_chat/qwen3.5:9b-q8_0"


def _fake_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _color_fsm() -> dict:
    return {
        "name": "ColorBot",
        "description": "typed field seam",
        "version": "4.1",
        "initial_state": "profile",
        "persona": "Concise.",
        "states": {
            "profile": {
                "id": "profile",
                "description": "collect color",
                "purpose": "Learn favorite color",
                "field_extractions": [
                    {
                        "field_name": "favorite_color",
                        "field_type": "str",
                        "extraction_instructions": "Favorite color, one word.",
                        "required": True,
                    }
                ],
                "response_instructions": "Reply in one short sentence.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "got color",
                        "conditions": [
                            {
                                "description": "has color",
                                "requires_context_keys": ["favorite_color"],
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "end",
                "response_instructions": "Say goodbye.",
                "transitions": [],
            },
        },
    }


def _stream_chunk(text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    chunk.choices[0].delta.reasoning_content = None
    return chunk


def _greeter_fsm(terminal: bool = False) -> dict:
    """chat -> other. ``terminal=True`` gates the edge on ``go`` (the harness
    sets it), so Pass 2 runs in the terminal state ``other``; otherwise the
    edge never fires and Pass 2 runs in the non-terminal state ``chat``."""
    gate = "go" if terminal else "never_set"
    transitions = [
        {
            "target_state": "other",
            "description": "gated",
            "conditions": [{"description": "gated", "requires_context_keys": [gate]}],
        }
    ]
    fsm = {
        "name": "Greeter",
        "description": "stream prompt seam",
        "version": "4.1",
        "initial_state": "chat",
        "persona": "Concise.",
        "states": {
            "chat": {
                "id": "chat",
                "description": "chat",
                "purpose": "Chat briefly",
                "response_instructions": "Answer in one short sentence.",
                "transitions": transitions,
            },
            "other": {
                "id": "other",
                "description": "other",
                "purpose": "other",
                "response_instructions": "Say bye.",
                "transitions": [],
            },
        },
    }
    return fsm


def _correction_fsm() -> dict:
    """One collecting state: `favorite_color` has a config (required key),
    `nickname` is named only in extraction_instructions. No transitions."""
    return {
        "name": "CorrectionBot",
        "description": "bulk correction seam",
        "version": "4.1",
        "initial_state": "profile",
        "persona": "Concise.",
        "states": {
            "profile": {
                "id": "profile",
                "description": "collect color",
                "purpose": "Learn favorite color",
                "required_context_keys": ["favorite_color"],
                "extraction_instructions": (
                    "Extract favorite_color (one word) and nickname."
                ),
                "response_instructions": "Reply in one short sentence.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "sentinel, never true",
                        "priority": 10,
                        "conditions": [
                            {
                                "description": "sentinel",
                                "requires_context_keys": ["favorite_color"],
                                "logic": {"==": [{"var": "favorite_color"}, "__x__"]},
                            }
                        ],
                    },
                    {
                        "target_state": "profile",
                        "description": "keep chatting",
                        "priority": 100,
                    },
                ],
            },
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "end",
                "response_instructions": "Say goodbye.",
                "transitions": [],
            },
        },
    }


def _classified_fsm(explicit_field_extraction: bool = False, topic: bool = False):
    """`intent` is owned by classification_extractions and gates the only
    transition. ``topic`` adds an ordinary required key (vacuity guard)."""
    state: dict = {
        "id": "triage",
        "description": "classify intent",
        "purpose": "route",
        "response_instructions": "Reply briefly.",
        "classification_extractions": [
            {
                "field_name": "intent",
                "intents": [
                    {"name": "buy", "description": "wants to purchase"},
                    {"name": "browse", "description": "just looking"},
                ],
                "fallback_intent": "browse",
                "confidence_threshold": 0.7,
                "required": False,
            }
        ],
        "transitions": [
            {
                "target_state": "shop",
                "description": "buying",
                "conditions": [
                    {
                        "description": "intent is buy",
                        "requires_context_keys": ["intent"],
                        "logic": {"==": [{"var": "intent"}, "buy"]},
                    }
                ],
            }
        ],
    }
    if topic:
        state["required_context_keys"] = ["topic"]
    if explicit_field_extraction:
        state["field_extractions"] = [
            {
                "field_name": "intent",
                "field_type": "str",
                "extraction_instructions": "Explicit intent extraction.",
                "required": False,
            }
        ]
    return {
        "name": "ClassifiedBot",
        "description": "classification ownership seam",
        "version": "4.1",
        "initial_state": "triage",
        "persona": "Concise.",
        "states": {
            "triage": state,
            "shop": {
                "id": "shop",
                "description": "shop",
                "purpose": "shop",
                "response_instructions": "Help shop.",
                "transitions": [],
            },
        },
    }


def _aiter1_ambiguous_fsm(self_loop: bool = False) -> dict:
    """`hub` has two unconditional, equal-priority exits (AMBIGUOUS), or a single
    declared unconditional self-loop (DETERMINISTIC, target == current state)."""

    def _target(name: str, description: str) -> dict:
        return {"target_state": name, "description": description, "priority": 100}

    if self_loop:
        gated = _target("billing", "billing, gated on a key that is never set")
        gated["conditions"] = [
            {
                "description": "go set",
                "requires_context_keys": ["go"],
                "logic": {"==": [{"var": "go"}, True]},
            }
        ]
        transitions = [_target("hub", "declared self-loop"), gated]
    else:
        transitions = [
            _target("billing", "user asks about billing"),
            _target("support", "user needs tech support"),
        ]
    states = {
        "hub": {
            "id": "hub",
            "description": "hub",
            "purpose": "route",
            "response_instructions": "Reply briefly.",
            "transitions": transitions,
        },
    }
    for name in ("billing",) if self_loop else ("billing", "support"):
        states[name] = {
            "id": name,
            "description": name,
            "purpose": name,
            "response_instructions": f"Handle {name}.",
            "transitions": [],
        }
    return {
        "name": "HubBot",
        "description": "ambiguous transition seam",
        "version": "4.1",
        "initial_state": "hub",
        "persona": "Concise.",
        "states": states,
    }


def _is_classifier_call(kwargs: dict) -> bool:
    """True for a provider request the ``Classifier`` issued.

    Every core call goes through ``fsm_llm.llm.providers.litellm.completion``; a classifier
    request is the one carrying the ``intent_classification`` schema (sent
    when the model supports ``response_format``).
    """
    fmt = kwargs.get("response_format") or {}
    return fmt.get("json_schema", {}).get("name") == "intent_classification"


class _ConnectionHarness:
    """`API.converse` with the REAL Classifier over a scripted
    ``fsm_llm.llm.providers.litellm.completion`` (the one binding every core call uses);
    records the kwargs of every classifier call, fakes Pass 1 / Pass 2."""

    def __init__(self, fsm: dict, intent: str, **api_kwargs):
        self.fsm = fsm
        self.intent = intent
        self.api_kwargs = api_kwargs
        self.classifier_calls: list[dict] = []

    def _classifier_completion(self, **kwargs):
        self.classifier_calls.append(kwargs)
        return _fake_response(
            json.dumps({"reasoning": "r", "intent": self.intent, "confidence": 0.95})
        )

    def _llm_completion(self, **kwargs):
        if _is_classifier_call(kwargs):
            return self._classifier_completion(**kwargs)
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def run(self):
        with (
            patch(
                "fsm_llm.llm.providers.litellm.completion",
                side_effect=self._llm_completion,
            ),
            patch(
                "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            api = API.from_definition(self.fsm, **self.api_kwargs)
            cid, _ = api.start_conversation()
            api.converse("I would like to buy a phone", cid)
            return api.get_data(cid), api.get_current_state(cid)


def _classifier() -> Classifier:
    schema = ClassificationSchema(
        intents=[
            IntentDefinition(name="buy", description="wants to buy"),
            IntentDefinition(name="browse", description="just looking"),
        ],
        fallback_intent="browse",
    )
    return Classifier(schema, model="gpt-4o")


class _Prov:
    """Scripted ``API`` session: per-field replies by field name, bulk reply dict.

    ``field`` maps a field name to the value its per-field extractor returns
    (absent name means "no value"); ``bulk`` is the bulk ``extracted_data``.
    """

    def __init__(self, fsm: dict, store=None):
        self.fsm = fsm
        self.store = store
        self.field: dict = {}
        self.bulk: dict = {}
        self.context_updates: list[list[str]] = []
        self._stack = ExitStack()

    def _completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if '"extracted_data"' in system:
            return _fake_response(
                json.dumps({"extracted_data": self.bulk, "confidence": 0.9})
            )
        if kwargs.get("response_format") is not None:
            for name, value in self.field.items():
                if f"Extract the field '{name}'" in system:
                    return _fake_response(
                        json.dumps(
                            {"field_name": name, "value": value, "confidence": 0.9}
                        )
                    )
            return _fake_response(
                json.dumps({"field_name": "x", "value": None, "confidence": 0.0})
            )
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def __enter__(self):
        self._stack.enter_context(
            patch(
                "fsm_llm.llm.providers.litellm.completion", side_effect=self._completion
            )
        )
        self._stack.enter_context(
            patch(
                "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                return_value=["response_format"],
            )
        )
        self.api = self.make_api()
        self.cid, _ = self.api.start_conversation()
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def make_api(self) -> API:
        api = API.from_definition(
            self.fsm, model="gpt-4o", api_key="test", session_store=self.store
        )
        api.register_handler(
            api.create_handler("ctx_spy")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .do(lambda ctx: self.context_updates.append(sorted(ctx.keys())) or {})
        )
        return api

    def turn(self, field: dict | None = None, bulk: dict | None = None) -> dict:
        self.field, self.bulk = field or {}, bulk or {}
        self.api.converse("hello", self.cid)
        return dict(self.api.get_data(self.cid))


def _typed_fsm() -> dict:
    """ra01: int and bool configs; a gate fires on ``years_old == 25`` (int)."""
    return {
        "name": "Typed",
        "description": "typed drift seam",
        "version": "4.1",
        "initial_state": "s",
        "persona": "x",
        "states": {
            "s": {
                "id": "s",
                "description": "d",
                "purpose": "collect",
                "extraction_instructions": "Extract years_old and is_subscribed.",
                "field_extractions": [
                    {
                        "field_name": "years_old",
                        "field_type": "int",
                        "extraction_instructions": "age",
                        "required": True,
                    },
                    {
                        "field_name": "is_subscribed",
                        "field_type": "bool",
                        "extraction_instructions": "sub",
                        "required": True,
                    },
                ],
                "response_instructions": "Reply.",
                "transitions": [
                    {
                        "target_state": "adult",
                        "description": "years_old is 25",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "a",
                                "requires_context_keys": ["years_old"],
                                "logic": {"==": [{"var": "years_old"}, 25]},
                            }
                        ],
                    },
                ],
            },
            "adult": {
                "id": "adult",
                "description": "d",
                "purpose": "p",
                "response_instructions": "ok",
                "transitions": [],
            },
        },
    }


def _gate_fsm() -> dict:
    """repro5: a transition gated on ``is_verified == True``; the bulk pass has
    instructions but no per-field config beyond the auto-minted gate key."""
    return {
        "name": "Gate",
        "description": "gate flip seam",
        "version": "4.1",
        "initial_state": "gate",
        "persona": "x",
        "states": {
            "gate": {
                "id": "gate",
                "description": "d",
                "purpose": "verify",
                "extraction_instructions": "Extract the user's stated details.",
                "response_instructions": "Reply.",
                "transitions": [
                    {
                        "target_state": "admin",
                        "description": "verified users only",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "verified",
                                "requires_context_keys": ["is_verified"],
                                "logic": {"==": [{"var": "is_verified"}, True]},
                            }
                        ],
                    }
                ],
            },
            "admin": {
                "id": "admin",
                "description": "d",
                "purpose": "admin",
                "response_instructions": "Admin area.",
                "transitions": [],
            },
        },
    }


def _reentry_fsm() -> dict:
    """repro1: one loopable state so a turn does Pass 2 only.

    The two edges tie at priority 100 so every turn is AMBIGUOUS and the mocked
    classifier fails soft to "stay". Under A2 (plan-2026-09-21T203800-8a03483a
    D-003) the old 1-vs-100 spacing would be DETERMINISTIC to ``done``.
    """
    return {
        "name": "Reentry",
        "description": "d",
        "initial_state": "chat",
        "persona": "p",
        "states": {
            "chat": {
                "id": "chat",
                "description": "d",
                "purpose": "p",
                "response_instructions": "Reply.",
                "transitions": [
                    {"target_state": "chat", "description": "stay", "priority": 100},
                    {"target_state": "done", "description": "bye", "priority": 100},
                ],
            },
            "done": {
                "id": "done",
                "description": "d",
                "purpose": "p",
                "response_instructions": "Bye.",
                "transitions": [],
            },
        },
    }


class _Provider:
    """Scripted provider with an outage switch and a call counter.

    ``fail_pass2`` fails only the non-extraction, non-stream generation call, so
    a Pass 1 extraction can commit before Pass 2 dies; ``fail_stream`` fails the
    streaming call. ``down`` fails everything.
    """

    def __init__(self):
        self.calls = 0
        self.down = False
        self.fail_pass2 = False
        self.fail_stream = False
        self.name_value = "Bob"

    def completion(self, **kwargs):
        self.calls += 1
        if self.down:
            raise RuntimeError("provider down")
        if kwargs.get("stream"):
            if self.fail_stream:
                raise RuntimeError("stream provider down")
            return iter(_stream_chunks(["Hel", "lo"]))
        system = kwargs["messages"][0]["content"]
        if kwargs.get("response_format") is not None:
            return _fake_response(
                json.dumps(
                    {"field_name": "name", "value": self.name_value, "confidence": 0.9}
                )
            )
        if self.fail_pass2 and "Extract" not in system[:300]:
            raise RuntimeError("provider down")
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))


def _stream_chunks(parts):
    from unittest.mock import MagicMock

    out = []
    for p in parts:
        c = MagicMock()
        c.choices = [MagicMock()]
        c.choices[0].delta.content = p
        c.choices[0].delta.reasoning_content = None
        c.choices[0].finish_reason = None
        out.append(c)
    return out


class _Session:
    def __init__(self, fsm: dict):
        self.fsm = fsm
        self.provider = _Provider()
        self._stack = ExitStack()

    def __enter__(self):
        self._stack.enter_context(
            patch(
                "fsm_llm.llm.providers.litellm.completion",
                side_effect=self.provider.completion,
            )
        )
        self._stack.enter_context(
            patch(
                "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                return_value=["response_format"],
            )
        )
        self.api = API.from_definition(self.fsm, model="gpt-4o", api_key="test")
        self.cid, _ = self.api.start_conversation()
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def on(self, timing, fn, name="h"):
        self.api.register_handler(self.api.create_handler(name).at(timing).do(fn))


class _PromptSpy:
    """Runs one ``API.converse`` on the loopable reentry FSM and records every
    system prompt the (patched) provider received."""

    def __init__(self, message: str, initial_context: dict | None = None):
        self.system_prompts: list[str] = []

        def completion(**kwargs):
            self.system_prompts.append(kwargs["messages"][0]["content"])
            return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "fsm_llm.llm.providers.litellm.completion", side_effect=completion
                )
            )
            stack.enter_context(
                patch(
                    "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                    return_value=["response_format"],
                )
            )
            api = API.from_definition(_reentry_fsm(), model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation(initial_context)
            self.system_prompts.clear()
            api.converse(message, cid)
        self.prompt = self.system_prompts[-1]


def _declared_secret_fsm() -> dict:
    """`password` is a DECLARED per-field extraction; `favorite_color` is benign."""
    fsm = _correction_fsm()
    fsm["states"]["profile"]["field_extractions"] = [
        {
            "field_name": "favorite_color",
            "field_type": "str",
            "extraction_instructions": "the color",
            "required": True,
        },
        {
            "field_name": "password",
            "field_type": "str",
            "extraction_instructions": "the password",
            "required": False,
        },
    ]
    return fsm


class _BulkProv(_Prov):
    """``_Prov`` that also records every bulk-extraction system prompt."""

    def __init__(self, fsm: dict, initial_context: dict | None = None):
        super().__init__(fsm)
        self.bulk_prompts: list[str] = []
        self.initial_context = initial_context

    def _completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if '"extracted_data"' in system:
            self.bulk_prompts.append(system)
        return super()._completion(**kwargs)

    def __enter__(self):
        super().__enter__()
        if self.initial_context:
            self.cid, _ = self.api.start_conversation(self.initial_context)
        return self

    def say(self, message: str, field: dict | None = None, bulk: dict | None = None):
        self.field, self.bulk = field or {}, bulk or {}
        self.api.converse(message, self.cid)
        return dict(self.api.get_data(self.cid))


class _ClassBulkProv(_BulkProv):
    """``_BulkProv`` with the REAL Classifier over a scripted
    ``fsm_llm.llm.providers.litellm.completion`` (fixed intent and confidence)."""

    def __init__(
        self,
        fsm: dict,
        intent: str,
        confidence: float,
        initial_context: dict | None = None,
    ):
        super().__init__(fsm, initial_context)
        self.intent, self.confidence = intent, confidence

    def _classifier_completion(self, **kwargs):
        return _fake_response(
            json.dumps(
                {
                    "reasoning": "r",
                    "intent": self.intent,
                    "confidence": self.confidence,
                    "entities": {},
                }
            )
        )

    def _completion(self, **kwargs):
        if _is_classifier_call(kwargs):
            return self._classifier_completion(**kwargs)
        return super()._completion(**kwargs)


def _classified_bulk_fsm() -> dict:
    """ra02: `intent` is classification-owned (threshold 0.7) and gates the only
    transition; the state also has `extraction_instructions` (bulk pass runs)."""
    fsm = _classified_fsm()
    fsm["states"]["triage"]["extraction_instructions"] = (
        "Extract the intent the user expresses and any topic."
    )
    return fsm


def _extract_field(reply: str, field_type: str = "str"):
    req = FieldExtractionRequest(
        system_prompt="extract the field",
        user_message="My name is Bob.",
        field_name="name",
        field_type=field_type,  # type: ignore[arg-type]
    )
    with (
        patch("fsm_llm.llm.providers.litellm.completion") as mock_comp,
        patch(
            "fsm_llm.llm.providers.litellm.get_supported_openai_params",
            return_value=["response_format"],
        ),
    ):
        mock_comp.return_value = _fake_response(reply)
        return LiteLLMInterface(model="gpt-4o", api_key="k").extract_field(req)


class _PassTwoSpy(_Prov):
    """``_Prov`` that also records the response-generation system prompt and
    answers a streaming call, so both Pass-2 call sites can be inspected."""

    def __init__(self, fsm: dict, store=None):
        super().__init__(fsm, store)
        self.pass2: list[str] = []

    def _completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "<response_generation>" in system:
            self.pass2.append(system)
            if kwargs.get("stream"):
                return iter([_stream_chunk("ok")])
        return super()._completion(**kwargs)

    def say(self, message: str, bulk: dict, stream: bool = False) -> str:
        self.field, self.bulk = {}, bulk
        self.pass2.clear()
        if stream:
            list(self.api.converse_stream(message, self.cid))
        else:
            self.api.converse(message, self.cid)
        assert self.pass2, "no response-generation call captured"
        return self.pass2[-1]


def _back_edge_fsm(name_extra_required: list[str] | None = None) -> dict:
    """port of repros/back_edge.py: `name` -> `email` -> back to `name`.

    ``name_extra_required`` adds required keys to `name` that the user never
    supplies (the still-null case the call budget is stated for)."""

    def _cond(key: str, logic: dict) -> dict:
        return {"description": key, "requires_context_keys": [key], "logic": logic}

    return {
        "name": "Form",
        "description": "back edge",
        "version": "4.1",
        "initial_state": "name",
        "persona": "Concise.",
        "states": {
            "name": {
                "id": "name",
                "description": "collect name",
                "purpose": "collect name",
                "extraction_instructions": "Extract the user's full name.",
                "required_context_keys": ["full_name", *(name_extra_required or [])],
                "response_instructions": "Ask for name.",
                "transitions": [
                    {
                        "target_state": "email",
                        "description": "have name",
                        "priority": 100,
                        "conditions": [
                            _cond("full_name", {"has_context": "full_name"})
                        ],
                    }
                ],
            },
            "email": {
                "id": "email",
                "description": "collect email",
                "purpose": "collect email",
                "extraction_instructions": (
                    "Extract the email and whether the user wants to change "
                    "their name (wants_name_change)."
                ),
                "required_context_keys": ["email"],
                "response_instructions": "Ask for email.",
                "transitions": [
                    {
                        "target_state": "name",
                        "description": "user wants to go back",
                        "priority": 10,
                        "conditions": [
                            _cond(
                                "wants_name_change",
                                {"==": [{"var": "wants_name_change"}, True]},
                            )
                        ],
                    },
                    {
                        "target_state": "done",
                        "description": "have email",
                        "priority": 100,
                        "conditions": [_cond("email", {"has_context": "email"})],
                    },
                ],
            },
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "end",
                "response_instructions": "Done.",
                "transitions": [],
            },
        },
    }


class _EdgeProv(_Prov):
    """``_Prov`` that records the kind of every Pass-1 provider call per turn."""

    def __init__(self, fsm: dict, store=None):
        super().__init__(fsm, store)
        self.kinds: list[str] = []

    def _completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if '"extracted_data"' in system:
            self.kinds.append("bulk")
        elif kwargs.get("response_format") is not None:
            self.kinds.append("field")
        return super()._completion(**kwargs)

    def say(self, message: str, field: dict, bulk: dict) -> list[str]:
        """One turn; returns the Pass-1 call kinds it made."""
        self.field, self.bulk = field, bulk
        self.kinds.clear()
        self.api.converse(message, self.cid)
        return list(self.kinds)

    def reach_email(self, name: str = "Alice Smith") -> None:
        kinds = self.say(f"I'm {name}", {"full_name": name}, {})
        assert self.api.get_current_state(self.cid) == "email"
        # a forward hop into a state with nothing pre-filled costs no bulk call
        # beyond the one the source state's own extraction makes
        assert kinds.count("bulk") == 1


_BACK_TURN = "wait, change my name, it's Bob Jones"


_BACK_FIELD = {"wants_name_change": True}


_BACK_BULK = {"full_name": "Bob Jones", "wants_name_change": True}


_TWO_STATE = {
    "name": "b",
    "description": "d",
    "initial_state": "s",
    "persona": "p",
    "states": {
        "s": {
            "id": "s",
            "description": "d",
            "purpose": "p",
            "response_instructions": "hi",
            "transitions": [
                {"target_state": "e", "description": "go", "priority": 100}
            ],
        },
        "e": {
            "id": "e",
            "description": "d",
            "purpose": "p",
            "response_instructions": "bye",
        },
    },
}


def _fsm() -> dict[str, Any]:
    return {
        "name": "t",
        "description": "d",
        "initial_state": "a",
        "states": {
            "a": {
                "id": "a",
                "description": "a",
                "purpose": "p",
                "transitions": [{"target_state": "b", "description": "go"}],
            },
            "b": {"id": "b", "description": "b", "purpose": "p"},
        },
    }


_INTENTS = ClassificationSchema(
    intents=[
        IntentDefinition(name="buy", description="wants to buy"),
        IntentDefinition(name="browse", description="just looking"),
    ],
    fallback_intent="browse",
)


class _Custom(LLMInterface):
    """A user's own interface: Pass 1/2 answer, ``complete`` runs ``bug``."""

    model = "custom/m"

    def __init__(self, bug: Any) -> None:
        self.bug = bug

    def generate_response(self, request: Any) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning=""
        )

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        return FieldExtractionResponse(
            field_name=request.field_name, value=None, confidence=0.0, is_valid=False
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        return self.bug(request)


def _tie_fsm() -> dict[str, Any]:
    return {
        "name": "Tie",
        "description": "Two transitions at one priority: the classifier decides",
        "initial_state": "route",
        "states": {
            "route": _state(
                "route",
                response_instructions="Respond",
                transitions=[
                    {"target_state": "a", "description": "A"},
                    {"target_state": "b", "description": "B"},
                ],
            ),
            "a": _state("a"),
            "b": _state("b"),
        },
    }


def _classifying_fsm() -> dict[str, Any]:
    return {
        "name": "Classify",
        "description": "One classification field",
        "initial_state": "triage",
        "states": {
            "triage": _state(
                "triage",
                response_instructions="Respond",
                classification_extractions=[
                    {
                        "field_name": "intent",
                        "intents": [
                            {"name": "buy", "description": "wants to buy"},
                            {"name": "browse", "description": "just looking"},
                        ],
                        "fallback_intent": "browse",
                    }
                ],
                transitions=[
                    {
                        "target_state": "done",
                        "description": "Classified",
                        "conditions": [
                            {
                                "description": "intent known",
                                "requires_context_keys": ["intent"],
                                "logic": {"==": [{"var": "intent"}, "buy"]},
                            }
                        ],
                    }
                ],
            ),
            "done": _state("done"),
        },
    }


def _chain(error: BaseException) -> list[type[BaseException]]:
    kinds: list[type[BaseException]] = []
    current: BaseException | None = error
    while current is not None:
        kinds.append(type(current))
        current = current.__cause__ or current.__context__
    return kinds


def _state(state_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": state_id,
        "description": f"The {state_id} state",
        "purpose": f"Do the {state_id} work",
        "response_instructions": "",
        **extra,
    }
