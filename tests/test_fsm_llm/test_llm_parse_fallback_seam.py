"""Robustness tests for the LLM response-parsing degradation ladder (T4 / D-016).

The documented ladder is: **structured JSON -> embedded-JSON fallback -> raw text**.
The contract is that a ``_parse_*`` method NEVER raises for a well-formed but
oversized / oddly-typed model response — it degrades down the ladder instead.

These tests drive the REAL parse methods
(``LiteLLMInterface._parse_response_generation_response`` and
``._parse_field_extraction_response``); nothing here re-implements the ladder.
The only thing faked is the litellm response envelope
(``response.choices[0].message.content``), which is the actual seam the parsers
read from.
"""

from __future__ import annotations

import json

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationResponse,
)
from fsm_llm.llm import (
    _GENERIC_FALLBACK_MESSAGE,
    _RESPONSE_MESSAGE_MAX_LEN,
    LiteLLMInterface,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    """Minimal stand-in for a litellm completion envelope."""

    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


@pytest.fixture
def llm() -> LiteLLMInterface:
    return LiteLLMInterface(model="test", api_key="test")


@pytest.fixture
def field_request() -> FieldExtractionRequest:
    return FieldExtractionRequest(
        system_prompt="extract the field",
        user_message="some user text",
        field_name="destination",
        field_type="str",
    )


class TestCapDriftGuard:
    """The duplicated 5000 literal cannot silently diverge from the model."""

    def test_constant_matches_the_declared_model_max_length(self):
        metadata = ResponseGenerationResponse.model_fields["message"].metadata
        declared = [
            m.max_length for m in metadata if getattr(m, "max_length", None) is not None
        ]
        assert declared, "ResponseGenerationResponse.message lost its max_length"
        assert _RESPONSE_MESSAGE_MAX_LEN == declared[0], (
            f"llm._RESPONSE_MESSAGE_MAX_LEN ({_RESPONSE_MESSAGE_MAX_LEN}) has drifted "
            f"from ResponseGenerationResponse.message max_length ({declared[0]}); "
            "update both (definitions.py and llm.py)"
        )


class TestResponseGenerationLadderNeverRaises:
    """SC-7: no ValidationError escapes ``_parse_response_generation_response``."""

    def test_oversized_message_via_embedded_json_fallback(self, llm):
        """The reported defect: >5000-char message in the fallback branch."""
        oversized = "A" * 6000
        # Prose prefix => not `_looks_like_json`, so the PRIMARY rung is skipped
        # and the embedded-JSON fallback is the branch under test.
        content = "Here is my answer: " + json.dumps({"message": oversized})

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert isinstance(result, ResponseGenerationResponse)
        assert len(result.message) <= _RESPONSE_MESSAGE_MAX_LEN
        assert result.message  # usable, not empty

    def test_oversized_raw_text_at_the_terminal_rung(self, llm):
        """The real trap: the bottom rung builds the SAME capped model."""
        result = llm._parse_response_generation_response(_FakeResponse("B" * 12000))

        assert isinstance(result, ResponseGenerationResponse)
        assert len(result.message) == _RESPONSE_MESSAGE_MAX_LEN
        assert result.message == "B" * _RESPONSE_MESSAGE_MAX_LEN

    def test_oversized_message_via_primary_structured_rung(self, llm):
        """Pure JSON >5000: primary rung fails, ladder still lands on its feet."""
        content = json.dumps({"message": "C" * 9000})

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert isinstance(result, ResponseGenerationResponse)
        assert len(result.message) <= _RESPONSE_MESSAGE_MAX_LEN

    def test_oversized_reasoning_does_not_raise(self, llm):
        """`reasoning` carries the same 5000 cap as `message`."""
        content = "Answer: " + json.dumps({"message": "hi", "reasoning": "R" * 7000})

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert isinstance(result, ResponseGenerationResponse)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "   ",
            "```",
            "```json\n```",
            "```json\n{\n```",
            '{"message": "he said \\"hi {nested} there\\" and left"}',
            'prose {"message": "brace } inside \\"a {string}\\" value"} tail',
            "{not json at all",
            "[1, 2, 3]",
            '{"message": null}',
            '{"message": 42}',
            '{"message_type": "response"}',
            "null",
        ],
        ids=[
            "empty",
            "whitespace",
            "bare-fence",
            "empty-json-fence",
            "truncated-fence",
            "escaped-quotes-and-braces",
            "nested-braces-in-string-value",
            "malformed-object",
            "top-level-array",
            "null-message",
            "int-message",
            "message-key-absent",
            "literal-null",
        ],
    )
    def test_degenerate_inputs_return_rather_than_raise(self, llm, content):
        result = llm._parse_response_generation_response(_FakeResponse(content))
        assert isinstance(result, ResponseGenerationResponse)

    def test_non_string_content_types_return_rather_than_raise(self, llm):
        for content in (None, 123, ["a", "b"], {"other": "key"}):
            result = llm._parse_response_generation_response(_FakeResponse(content))
            assert isinstance(result, ResponseGenerationResponse)


class TestFieldExtractionLadderNeverRaises:
    """SC-7: no ValidationError/TypeError escapes ``_parse_field_extraction_response``."""

    def test_oversized_reasoning_via_embedded_json_fallback(self, llm, field_request):
        content = "Result: " + json.dumps(
            {"value": "Paris", "reasoning": "R" * 7000, "confidence": 0.9}
        )

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert isinstance(result, FieldExtractionResponse)

    def test_non_numeric_confidence_in_fallback_does_not_raise(
        self, llm, field_request
    ):
        """`float({...})` raises TypeError, which is NOT a ValueError."""
        content = "Result: " + json.dumps({"value": "Paris", "confidence": {"a": 1}})

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert isinstance(result, FieldExtractionResponse)

    def test_non_numeric_confidence_in_primary_rung_does_not_raise(
        self, llm, field_request
    ):
        """Sibling found by the step-4 sweep: the PRIMARY rung escaped too."""
        content = json.dumps({"value": "Paris", "confidence": {"a": 1}})

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert isinstance(result, FieldExtractionResponse)

    def test_null_confidence_in_primary_rung_does_not_raise(self, llm, field_request):
        content = json.dumps({"value": "Paris", "confidence": None})

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert isinstance(result, FieldExtractionResponse)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "   ",
            "```",
            "```json\n```",
            '{"value": "he said \\"hi {nested} there\\""}',
            'prose {"value": "brace } in \\"a {string}\\""} tail',
            "{not json at all",
            "[1, 2, 3]",
            '{"value": null}',
            '{"confidence": "not-a-number"}',
            "null",
        ],
        ids=[
            "empty",
            "whitespace",
            "bare-fence",
            "empty-json-fence",
            "escaped-quotes-and-braces",
            "nested-braces-in-string-value",
            "malformed-object",
            "top-level-array",
            "null-value",
            "non-numeric-confidence-string",
            "literal-null",
        ],
    )
    def test_degenerate_inputs_return_rather_than_raise(
        self, llm, field_request, content
    ):
        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )
        assert isinstance(result, FieldExtractionResponse)

    def test_terminal_rung_reports_failure_rather_than_raising(
        self, llm, field_request
    ):
        """The bottom rung is the last line of defence: it returns, flagged."""
        result = llm._parse_field_extraction_response(
            _FakeResponse(None), field_request
        )

        assert isinstance(result, FieldExtractionResponse)
        assert result.is_valid is False
        assert result.value is None
        assert result.validation_error


class TestTerminalRungNeverEmitsARawEnvelope:
    """SC-14 (D-020): the user must never READ the serialized payload.

    ``TestResponseGenerationLadderNeverRaises`` above asserts only that the
    ladder RETURNS.  A raw-JSON dump satisfies that while being strictly worse
    than the exception it replaced, which is exactly how the D-016 regression
    shipped.  These tests assert the *user-visible string*.
    """

    # Each case pairs a payload with the human text that MUST survive it.
    # ``None`` means "no human text is recoverable, so a generic apology is the
    # only acceptable answer".
    @pytest.mark.parametrize(
        ("content", "expected_human_text"),
        [
            # The F1 regression, verbatim: a perfectly good `message` beside a
            # `reasoning` that breaches the SAME max_length=5000.
            (
                "Here: "
                + json.dumps(
                    {"message": "Your booking is confirmed.", "reasoning": "R" * 9000}
                ),
                "Your booking is confirmed.",
            ),
            # Same trapdoor with no prose prefix (primary rung, not the fallback).
            (
                json.dumps(
                    {"message": "Your booking is confirmed.", "reasoning": "R" * 9000}
                ),
                "Your booking is confirmed.",
            ),
            # A sibling capped/typed field: `message_type` must be a str.
            (
                json.dumps(
                    {"message": "Your booking is confirmed.", "message_type": {"a": 1}}
                ),
                "Your booking is confirmed.",
            ),
            # Non-str `reasoning`.
            (
                json.dumps({"message": "All set.", "reasoning": ["a", "b"]}),
                "All set.",
            ),
            # Nothing human to recover -> generic apology, never the envelope.
            ('{"message_type": "response"}', None),
            ('{"message": null}', None),
            ('{"message": 42}', None),
            ("[1, 2, 3]", None),
        ],
        ids=[
            "oversized-reasoning-via-fallback",
            "oversized-reasoning-via-primary",
            "dict-message-type",
            "list-reasoning",
            "message-key-absent",
            "null-message",
            "int-message",
            "top-level-array",
        ],
    )
    def test_user_visible_message_is_never_a_serialized_envelope(
        self, llm, content, expected_human_text
    ):
        result = llm._parse_response_generation_response(_FakeResponse(content))
        returned = result.message

        # 1. Not a raw JSON blob by the obvious shape check...
        assert not returned.strip().startswith("{"), (
            f"terminal rung emitted a raw JSON envelope: {returned[:120]!r}"
        )
        # 2. ...nor by the subtler one: a prose-prefixed envelope does not start
        #    with '{' but is still plumbing leaking to the end user.
        assert '"message":' not in returned, (
            f"terminal rung leaked serialized payload keys: {returned[:120]!r}"
        )
        # 3. The intended human text survives when there IS one.
        if expected_human_text is None:
            assert returned == _GENERIC_FALLBACK_MESSAGE
        else:
            assert returned == expected_human_text

    def test_oversized_message_is_truncated_not_enveloped(self, llm):
        """A >5000 `message` degrades to truncated TEXT, not to the payload."""
        result = llm._parse_response_generation_response(
            _FakeResponse(json.dumps({"message": "C" * 9000}))
        )

        assert result.message == "C" * _RESPONSE_MESSAGE_MAX_LEN
        assert '"message":' not in result.message

    def test_oversized_reasoning_is_truncated_and_message_preserved(self, llm):
        """The `reasoning` cap must not cost us a valid `message` (D-020)."""
        content = json.dumps({"message": "Done.", "reasoning": "R" * 9000})

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert result.message == "Done."
        assert result.reasoning is not None
        assert len(result.reasoning) == _RESPONSE_MESSAGE_MAX_LEN

    def test_field_extraction_oversized_reasoning_keeps_the_value(
        self, llm, field_request
    ):
        """Same trapdoor on FieldExtractionResponse.reasoning (max_length=5000)."""
        content = json.dumps(
            {"value": "Paris", "reasoning": "R" * 9000, "confidence": 0.9}
        )

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert result.value == "Paris", "a good value was discarded by a capped sibling"
        assert result.reasoning is not None
        assert len(result.reasoning) == _RESPONSE_MESSAGE_MAX_LEN


class TestEndToEndUserVisibleString:
    """W5: at least one assertion driven through the REAL response path.

    Every other test in this file calls a private ``_parse_*`` method directly.
    That is precisely why the D-016 raw-JSON regression went unnoticed: 40 tests
    asserted "did not raise" and none asserted what the end user reads.  These
    drive ``API.converse`` / ``API.start_conversation`` end-to-end, with only
    ``LiteLLMInterface._make_llm_call`` (the litellm network boundary) faked.
    """

    @staticmethod
    def _fsm_dict() -> dict:
        return {
            "name": "parse_seam_e2e",
            "description": "FSM for end-to-end parse-ladder assertions",
            "initial_state": "start",
            "persona": "Test bot",
            "states": {
                "start": {
                    "id": "start",
                    "description": "Start state",
                    "purpose": "Begin conversation",
                    "response_instructions": "Say something",
                    "transitions": [
                        {
                            "target_state": "end",
                            "description": "Move to end",
                            "conditions": [
                                {"description": "Never fires", "logic": {"==": [1, 2]}}
                            ],
                        }
                    ],
                },
                "end": {
                    "id": "end",
                    "description": "End state",
                    "purpose": "Finish",
                    "response_instructions": "Say goodbye",
                    "transitions": [],
                },
            },
        }

    def _api_returning(self, payload: str) -> tuple[API, str]:
        llm = LiteLLMInterface(model="test", api_key="test")
        # Fake ONLY the network boundary; the real parse ladder still runs.
        llm._make_llm_call = lambda *a, **kw: _FakeResponse(payload)  # type: ignore[method-assign]
        api = API(fsm_definition=self._fsm_dict(), llm_interface=llm)
        conv_id, _ = api.start_conversation()
        return api, conv_id

    def test_converse_never_returns_a_raw_json_envelope(self):
        """The F1 regression as the END USER experiences it."""
        payload = json.dumps(
            {"message": "Your booking is confirmed.", "reasoning": "R" * 9000}
        )
        api, conv_id = self._api_returning(payload)

        reply = api.converse("book me a room", conv_id)

        assert not reply.strip().startswith("{"), f"user was shown raw JSON: {reply!r}"
        assert '"reasoning":' not in reply
        assert reply == "Your booking is confirmed."

        # And the same string is what gets persisted into history, so the next
        # turn's prompt is not poisoned with plumbing either.
        assert {"system": "Your booking is confirmed."} in api.get_conversation_history(
            conv_id
        )

    def test_converse_falls_back_to_prose_when_nothing_is_recoverable(self):
        api, conv_id = self._api_returning('{"message_type": "response"}')

        reply = api.converse("hello", conv_id)

        assert reply == _GENERIC_FALLBACK_MESSAGE

    # ---------------------------------------------------------------
    # C2: the field-extraction half of the ladder, end to end.
    #
    # Everything above drives ``_parse_response_generation_response``.
    # ``_parse_field_extraction_response`` had NO end-to-end anchor at all: its
    # failures land in the conversation CONTEXT rather than in the reply string,
    # so a rung-level unit test cannot show what the next turn's prompt (or a
    # transition condition reading that key) actually sees.
    # ---------------------------------------------------------------

    @staticmethod
    def _extracting_fsm_dict() -> dict:
        """Same shape as ``_fsm_dict`` plus an explicit ``field_extractions``
        entry, which the pipeline turns into an ``extract_field`` call.

        ``field_type`` must be declared ``str`` explicitly: auto-generating the
        config from ``required_context_keys`` yields ``field_type='any'``, and
        the D-016 envelope guard is scoped to ``str`` on purpose (for
        ``dict``/``list``/``any`` a JSON payload is the legitimately expected
        value). A test built on the auto-generated config silently misses the
        rung entirely — this was caught by the guard-removal probe, not by
        reading.
        """
        fsm = TestEndToEndUserVisibleString._fsm_dict()
        fsm["name"] = "extract_seam_e2e"
        fsm["states"]["start"]["extraction_instructions"] = "Extract the destination"
        fsm["states"]["start"]["field_extractions"] = [
            {
                "field_name": "destination",
                "field_type": "str",
                "extraction_instructions": "The city the user wants to travel to",
            }
        ]
        return fsm

    def _api_extracting(self, field_payload: str) -> tuple[API, str]:
        """As ``_api_returning``, but only the FIELD-EXTRACTION call gets the
        payload under test; every other call gets a clean envelope, so an
        assertion failure can only be about the field-extraction rung."""
        clean = json.dumps({"message": "Noted.", "reasoning": "ok"})
        llm = LiteLLMInterface(model="test", api_key="test")
        llm._make_llm_call = (  # type: ignore[method-assign]
            lambda messages, call_type, *a, **kw: _FakeResponse(
                field_payload if call_type == "field_extraction" else clean
            )
        )
        api = API(fsm_definition=self._extracting_fsm_dict(), llm_interface=llm)
        conv_id, _ = api.start_conversation()
        return api, conv_id

    def test_well_formed_extraction_lands_in_the_conversation_context(self):
        """Positive control: without this, the two tests below could pass because
        extraction never ran at all."""
        api, conv_id = self._api_extracting(
            json.dumps(
                {
                    "field_name": "destination",
                    "value": "Paris",
                    "confidence": 0.95,
                    "reasoning": "user said Paris",
                }
            )
        )

        api.converse("I want to go to Paris", conv_id)

        assert api.get_data(conv_id).get("destination") == "Paris"

    def test_free_text_containing_json_lands_in_context_instead_of_vanishing(
        self,
    ):
        """The D-022 twin rung as the CONTEXT experiences it.

        This method previously asserted the OPPOSITE (that such a payload is
        nulled), pinning the D-016 guard that R1 reverted. Retargeted rather
        than deleted, because the end-to-end plumbing is exactly what shows why
        the revert matters: a nulled value means the key never reaches context,
        so a state naming it in ``required_context_keys`` never satisfies and
        the turn silently loops re-asking. Storing the text is the lesser evil.
        """
        # Deliberately carries NO ``value`` key: that makes the embedded-JSON
        # rung above decline it, so the payload actually reaches the
        # unstructured-coercion rung this test is about. With a ``value`` key
        # the earlier rung recovers the value and the test proves nothing —
        # verified by probe.
        payload = 'The API said {"error": "not_found"} when I tried to save.'
        api, conv_id = self._api_extracting(payload)

        api.converse("it will not save", conv_id)

        stored = api.get_data(conv_id).get("destination")
        assert stored == payload, (
            f"free-text field value did not survive to context: {stored!r}"
        )

    def test_an_unrecoverable_extraction_degrades_instead_of_raising(self):
        """The ladder's contract at the public seam: the turn still completes."""
        api, conv_id = self._api_extracting("")

        reply = api.converse("hello", conv_id)

        assert isinstance(reply, str) and reply
        assert api.get_data(conv_id).get("destination") is None


class TestNormalResponsesUnchanged:
    """Non-regression: short/well-formed responses behave exactly as before."""

    def test_structured_response_still_parsed_from_the_primary_rung(self, llm):
        content = json.dumps(
            {"message": "Hello there", "message_type": "response", "reasoning": "why"}
        )

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert result.message == "Hello there"
        assert result.message_type == "response"
        assert result.reasoning == "why"

    def test_plain_text_response_still_used_verbatim(self, llm):
        result = llm._parse_response_generation_response(_FakeResponse("Just text."))

        assert result.message == "Just text."
        assert result.message_type == "response"

    def test_embedded_json_fallback_still_wins_over_raw_text(self, llm):
        content = '<think>musing</think> {"message": "Extracted!"}'

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert result.message == "Extracted!"

    def test_field_extraction_structured_response_unchanged(self, llm, field_request):
        content = json.dumps(
            {"value": "Paris", "confidence": 0.8, "reasoning": "stated explicitly"}
        )

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert result.value == "Paris"
        assert result.confidence == pytest.approx(0.8)
        assert result.reasoning == "stated explicitly"

    def test_field_extraction_fallback_still_extracts(self, llm, field_request):
        content = "Sure! " + json.dumps({"value": "Paris", "confidence": 0.7})

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert result.value == "Paris"


# Deliberate reuse: the end-to-end FSM dict + `_make_llm_call` fake below are the
# ones `TestEndToEndUserVisibleString` already owns. Copying a second 40-line FSM
# dict into this file would be a second source of truth for the same fixture.
_e2e = TestEndToEndUserVisibleString()

# A schema-MISMATCHED envelope: carries neither `message` nor `reasoning`, which
# is the only way the terminal raw-text rung is reached with the envelope still
# intact (`extract_json_from_text` recovers those two keys even from malformed
# JSON). See findings/a-class-decisions.md § A5.
_MISMATCHED = '{"note": "Your booking is confirmed.", "status": "ok"}'

_LEAK_SHAPES = {
    "prose-prefixed": f"Here you go: {_MISMATCHED}",
    "prose-suffixed": f"{_MISMATCHED} Let me know if anything else is needed!",
    "markdown-fenced": f"```json\n{_MISMATCHED}\n```",
}

# Prose a real assistant emits that CONTAINS a valid, non-empty JSON object.
# This list is the whole point of D-022. Every entry is a reply that a user is
# entitled to read verbatim, and every entry was DESTROYED (replaced by
# `_GENERIC_FALLBACK_MESSAGE`) by the D-016 `is_envelope` widening, which fired
# on "any parseable non-empty JSON object appears anywhere in the text".
#
# Note the first entry against `_MISMATCHED` above: they are the same string
# shape. That is the structural reason no text-shape discriminator can work
# here, and the reason the widening was reverted rather than narrowed.
_PROSE_CONTAINING_VALID_JSON = [
    'The server returned {"error": "not_found"} which means the record does not exist.',
    "Sure! To create a user, POST to /api/users with a body like "
    '{"name": "Alice", "role": "admin"}.',
    'In Python you would write d = {"key": "value"} and then access d["key"].',
    'Here is the JSON you asked me to write: ```json\n{"order_id": 123}\n```',
]

# The four adversarial negatives from findings/a-class-decisions.md § A5. Kept,
# but they are NOT sufficient on their own: not one of them contains a valid
# non-empty JSON object (`{1, 2, 3}` is a set literal, `config = {}` is empty,
# `{a: 1}` has an unquoted key, the fourth has no braces), so they were green
# throughout the D-016 regression. `_PROSE_CONTAINING_VALID_JSON` above is the
# list that actually probes the boundary. Do not delete either list.
_LEGITIMATE_PROSE = [
    "The variance is minimal: {1, 2, 3} are all valid outcomes here.",
    "You can define it like this: config = {} and then populate it later.",
    "If x = {a: 1}, then increment a by one to get the next state.",
    "Your booking is confirmed for tomorrow at 3pm. See you then!",
]


class TestLegitimateProseIsNeverEatenByTheEnvelopeGuard:
    """D-022: the terminal rung must not destroy replies that contain JSON.

    The guard here is `_looks_like_json`, which requires the text to BOTH start
    and end with a brace. Any widening of it that keys on "is there JSON in
    here" over-fires on ordinary assistant prose, and an over-firing guard is
    strictly worse than the leak it closes: it silently replaces correct replies
    with a generic apology, where the leak merely shows something obviously
    wrong.

    These tests are the regression pin. If a future change makes them fail,
    that change is destroying real user-visible answers -- read D-022 before
    touching them.
    """

    @pytest.mark.parametrize("prose", _PROSE_CONTAINING_VALID_JSON)
    def test_prose_containing_valid_json_survives_byte_for_byte(self, prose):
        api, conv_id = _e2e._api_returning(prose)

        reply = api.converse("how do I create a user?", conv_id)

        assert reply == prose, (
            f"envelope guard over-fired and ate a legitimate reply: {reply!r}"
        )

    @pytest.mark.parametrize("prose", _LEGITIMATE_PROSE)
    def test_legitimate_prose_survives_byte_for_byte(self, prose):
        api, conv_id = _e2e._api_returning(prose)

        reply = api.converse("hello", conv_id)

        assert reply == prose, (
            f"envelope guard over-fired and ate a legitimate reply: {reply!r}"
        )

    @pytest.mark.parametrize("shape", sorted(_LEAK_SHAPES))
    def test_prose_wrapped_envelope_still_reaches_the_user(self, shape):
        """ACCEPTED RESIDUAL, pinned so it stays a decision and not a surprise.

        A schema-mismatched envelope with prose on either side, or inside a
        markdown fence, DOES still reach the user verbatim. This is not the
        desired end state -- it is the price of not destroying the four replies
        pinned above, which are indistinguishable from it by text shape.

        Closing this needs a different signal (schema provenance, or a
        response-format flag), not a better regex. If someone lands that signal,
        this test is the one to rewrite; assert the new behavior here rather
        than deleting it.
        """
        api, conv_id = _e2e._api_returning(_LEAK_SHAPES[shape])

        reply = api.converse("book me a room", conv_id)

        assert reply == _LEAK_SHAPES[shape], (
            f"{shape} behavior changed; if deliberate, see D-022: {reply!r}"
        )


class TestFieldExtractionRungAcceptsProseContainingJson:
    """The twin rung, same trade-off, worse blast radius.

    ``_parse_field_extraction_response``'s ``str`` branch returns the raw text.
    D-016 nulled it whenever the text contained JSON; that is reverted, because
    a ``None`` field value never lands in context, so a state naming the key in
    ``required_context_keys`` never satisfies and the conversation silently
    loops re-asking. A free-text ``complaint`` / ``error_message`` field -- the
    shape of the repo's own ``tech_support_intake`` example -- is the canonical
    victim.
    """

    @staticmethod
    def _extract(llm, payload: str):
        llm._make_llm_call = lambda *a, **kw: _FakeResponse(payload)  # type: ignore[method-assign]
        return llm.extract_field(
            FieldExtractionRequest(
                system_prompt="extract the field",
                user_message="some user text",
                field_name="complaint",
                field_type="str",
            )
        )

    # The markdown-fenced entry is excluded on purpose. This rung strips code
    # fences up front (`_parse_field_extraction_response`, the two `re.sub`
    # calls), so a fenced value never survives byte-for-byte here -- and that is
    # PRE-EXISTING behavior, older than D-016 and unrelated to it. Asserting
    # byte-equality on it would pin the fence stripper, not the envelope guard.
    @pytest.mark.parametrize("prose", _PROSE_CONTAINING_VALID_JSON[:3])
    def test_prose_containing_valid_json_is_kept_as_the_field_value(self, prose, llm):
        result = self._extract(llm, prose)

        assert result.value == prose, (
            f"a legitimate free-text field value was nulled: {result.value!r}"
        )

    @pytest.mark.parametrize("prose", _LEGITIMATE_PROSE)
    def test_legitimate_prose_still_extracts_as_a_string_value(self, prose, llm):
        result = self._extract(llm, prose)

        assert result.value == prose, (
            f"a legitimate free-text field value was nulled: {result.value!r}"
        )


# ``reasoning`` text in which the model drafts an answer, reconsiders, and then
# emits its real answer.  Both recovery helpers are reachable on this shape --
# which one you hit depends only on whether the provider split the trace into a
# separate ``thinking`` field, and they return DIFFERENT objects.  See D-023.
_DRAFT_THEN_FINAL = (
    "Let me think about this.\n"
    '{"message": "DRAFT - ignore this", "reasoning": "still thinking"}\n'
    "Hmm, that is not right. Let me redo it.\n"
    '{"message": "FINAL ANSWER", "reasoning": "done"}\n'
)


class _FakeThinkingMessage:
    """A message whose answer arrived in the provider's ``thinking`` field."""

    def __init__(self, thinking: str):
        self.content = ""
        self.thinking = thinking


class TestMultiJsonTieBreakDivergence:
    """D-023: the two embedded-JSON recovery helpers DISAGREE on multi-JSON text.

    ``utilities.extract_json_from_text`` Strategy 3 returns the FIRST parsed
    object; ``LiteLLMInterface._extract_content_from_thinking`` prefers the LAST
    (and so does Strategy 4, which makes Strategy 3 the outlier inside its own
    module). Which helper you hit depends only on whether the provider split the
    reasoning trace into a separate ``thinking`` field.

    This divergence is REAL and still OPEN. It is pinned here rather than fixed,
    because flipping Strategy 3 to last-wins was tried (D-021) and reverted: it
    left the fenced case untouched (Strategy 2 runs first and is also
    first-match) while breaking the answer-then-example shape below. These tests
    document the status quo -- if a future change resolves the divergence, they
    are the tests to REWRITE deliberately, not to delete.
    """

    def test_the_two_recovery_helpers_still_disagree(self):
        from fsm_llm.utilities import extract_json_from_text

        via_utilities = extract_json_from_text(_DRAFT_THEN_FINAL)
        via_thinking = LiteLLMInterface._extract_content_from_thinking(
            _FakeThinkingMessage(_DRAFT_THEN_FINAL)
        )

        assert via_utilities is not None
        assert via_thinking is not None
        assert via_utilities["message"] == "DRAFT - ignore this", (
            "Strategy 3 is first-wins; see D-023 before changing this"
        )
        assert json.loads(via_thinking)["message"] == "FINAL ANSWER", (
            "the thinking helper is last-wins; see D-023 before changing this"
        )

    def test_answer_followed_by_a_schema_example_returns_the_ANSWER(self):
        """The case last-wins broke, and the reason it was reverted.

        A classifier that restates its schema after answering is a realistic
        shape. Under last-wins this returned ``{"intent": "<name>"}`` -- an
        unknown intent that ``classification.py`` silently degrades to
        ``fallback_intent``.
        """
        from fsm_llm.utilities import extract_json_from_text

        text = (
            'The intent is {"intent": "buy", "confidence": 0.9}. '
            'For reference the schema is {"intent": "<name>", "confidence": 0.0}.'
        )

        assert extract_json_from_text(text) == {"intent": "buy", "confidence": 0.9}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Single object: unchanged.
            (
                'The result is {"selected_transition": "greeting"} here.',
                {"selected_transition": "greeting"},
            ),
            # Nested object: the OUTER object wins.  Genuinely valuable
            # regression coverage -- a naive "accumulate and return the last"
            # rewrite returns the INNER object here, because the inner braces
            # are later start positions inside an already-parsed span.
            (
                'Some text {"outer": {"inner": "v"}, "x": 1} trailing',
                {"outer": {"inner": "v"}, "x": 1},
            ),
            # Strategy 4 only (no parseable braces at all): unchanged.
            (
                'blah "selected_transition": "next_state" blah',
                {"selected_transition": "next_state"},
            ),
        ],
        ids=["single-object", "nested-object", "strategy-4-only"],
    )
    def test_existing_shapes_return_exactly_what_they_return_today(
        self, text, expected
    ):
        from fsm_llm.utilities import extract_json_from_text

        assert extract_json_from_text(text) == expected
