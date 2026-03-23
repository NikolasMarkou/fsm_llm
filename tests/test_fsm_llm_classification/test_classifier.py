"""Tests for the Classifier and HierarchicalClassifier (mocked LLM)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm_classification import (
    ClassificationResponseError,
    ClassificationSchema,
    Classifier,
    HierarchicalClassifier,
    HierarchicalSchema,
    IntentDefinition,
)

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------


def _schema():
    return ClassificationSchema(
        intents=[
            IntentDefinition(name="order_status", description="About orders"),
            IntentDefinition(name="product_info", description="About products"),
            IntentDefinition(name="general_support", description="Fallback"),
        ],
        fallback_intent="general_support",
        confidence_threshold=0.6,
    )


def _mock_completion(content: dict):
    """Build a mock litellm.completion response."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(content)
    return mock_resp


# --------------------------------------------------------------
# Classifier
# --------------------------------------------------------------


class TestClassifier:
    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_classify_single(self, mock_comp, mock_params):
        mock_comp.return_value = _mock_completion(
            {
                "reasoning": "User asks about order",
                "intent": "order_status",
                "confidence": 0.95,
                "entities": {"order_id": "12345"},
            }
        )

        clf = Classifier(_schema(), model="test-model")
        result = clf.classify("Where is my order #12345?")

        assert result.intent == "order_status"
        assert result.confidence == 0.95
        assert result.entities == {"order_id": "12345"}

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_unknown_intent_falls_back(self, mock_comp, mock_params):
        mock_comp.return_value = _mock_completion(
            {
                "reasoning": "test",
                "intent": "hallucinated_intent",
                "confidence": 0.8,
                "entities": {},
            }
        )

        clf = Classifier(_schema(), model="test-model")
        result = clf.classify("something weird")

        assert result.intent == "general_support"

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_low_confidence_detection(self, mock_comp, mock_params):
        mock_comp.return_value = _mock_completion(
            {
                "reasoning": "unclear",
                "intent": "order_status",
                "confidence": 0.4,
                "entities": {},
            }
        )

        clf = Classifier(_schema(), model="test-model")
        result = clf.classify("hmm")

        assert clf.is_low_confidence(result)

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_classify_multi(self, mock_comp, mock_params):
        mock_comp.return_value = _mock_completion(
            {
                "reasoning": "compound query",
                "intents": [
                    {
                        "intent": "order_status",
                        "confidence": 0.9,
                        "entities": {"order_id": "1"},
                    },
                    {"intent": "product_info", "confidence": 0.7, "entities": {}},
                ],
            }
        )

        clf = Classifier(_schema(), model="test-model")
        result = clf.classify_multi("Where is order 1 and what about product X?")

        assert len(result.intents) == 2
        assert result.primary.intent == "order_status"

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Classifier(_schema(), model="")

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_thinking_model_fallback(self, mock_comp, mock_params):
        """When content is empty but thinking field has JSON, extract from it."""
        expected = {
            "reasoning": "order inquiry",
            "intent": "order_status",
            "confidence": 0.9,
            "entities": {},
        }
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = None
        mock_resp.choices[0].message.thinking = (
            "Let me classify this...\n" + json.dumps(expected)
        )
        mock_comp.return_value = mock_resp

        clf = Classifier(_schema(), model="test-model")
        result = clf.classify("Where is my order?")

        assert result.intent == "order_status"
        assert result.confidence == 0.9

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_empty_content_no_thinking_raises(self, mock_comp, mock_params):
        """Empty content with no thinking field raises ClassificationResponseError."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = None
        mock_resp.choices[0].message.thinking = None
        mock_comp.return_value = mock_resp

        clf = Classifier(_schema(), model="test-model")
        with pytest.raises(ClassificationResponseError, match="empty content"):
            clf.classify("test")

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_dict_content_passthrough(self, mock_comp, mock_params):
        """When provider returns a dict directly, pass it through."""
        expected = {
            "reasoning": "direct dict",
            "intent": "product_info",
            "confidence": 0.8,
            "entities": {},
        }
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = expected
        mock_comp.return_value = mock_resp

        clf = Classifier(_schema(), model="test-model")
        result = clf.classify("Tell me about product X")

        assert result.intent == "product_info"

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_multi_intent_uses_multi_prompt(self, mock_comp, mock_params):
        """classify_multi() should use the multi-intent system prompt."""
        mock_comp.return_value = _mock_completion(
            {
                "reasoning": "compound",
                "intents": [
                    {"intent": "order_status", "confidence": 0.9, "entities": {}},
                ],
            }
        )

        clf = Classifier(_schema(), model="test-model")
        clf.classify_multi("Where is my order and tell me about product X?")

        call_args = mock_comp.call_args
        system_msg = call_args[1]["messages"][0]["content"]
        assert "one or more" in system_msg

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_single_intent_uses_single_prompt(self, mock_comp, mock_params):
        """classify() should use the single-intent system prompt."""
        mock_comp.return_value = _mock_completion(
            {
                "reasoning": "order",
                "intent": "order_status",
                "confidence": 0.9,
                "entities": {},
            }
        )

        clf = Classifier(_schema(), model="test-model")
        clf.classify("Where is my order?")

        call_args = mock_comp.call_args
        system_msg = call_args[1]["messages"][0]["content"]
        assert "exactly one" in system_msg

    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_json_in_code_block(self, mock_comp, mock_params):
        """Content with JSON inside markdown code block should parse."""
        payload = {
            "reasoning": "order check",
            "intent": "order_status",
            "confidence": 0.85,
            "entities": {},
        }
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = (
            "```json\n" + json.dumps(payload) + "\n```"
        )
        mock_comp.return_value = mock_resp

        clf = Classifier(_schema(), model="test-model")
        result = clf.classify("Where is my order?")

        assert result.intent == "order_status"
        assert result.confidence == 0.85


# --------------------------------------------------------------
# HierarchicalClassifier
# --------------------------------------------------------------


class TestHierarchicalClassifier:
    @patch(
        "fsm_llm_classification.classifier.get_supported_openai_params", return_value=[]
    )
    @patch("fsm_llm_classification.classifier.completion")
    def test_two_stage(self, mock_comp, mock_params):
        # Stage 1 returns domain, stage 2 returns intent
        mock_comp.side_effect = [
            _mock_completion(
                {
                    "reasoning": "billing domain",
                    "intent": "billing",
                    "confidence": 0.9,
                    "entities": {},
                }
            ),
            _mock_completion(
                {
                    "reasoning": "refund",
                    "intent": "refund_request",
                    "confidence": 0.85,
                    "entities": {"amount": "50"},
                }
            ),
        ]

        domain_schema = ClassificationSchema(
            intents=[
                IntentDefinition(name="billing", description="Billing queries"),
                IntentDefinition(name="other", description="Other"),
            ],
            fallback_intent="other",
        )
        billing_schema = ClassificationSchema(
            intents=[
                IntentDefinition(name="refund_request", description="Refund"),
                IntentDefinition(name="invoice_query", description="Invoice"),
                IntentDefinition(name="billing_other", description="Other billing"),
            ],
            fallback_intent="billing_other",
        )
        h_schema = HierarchicalSchema(
            domain_schema=domain_schema,
            intent_schemas={"billing": billing_schema},
        )

        clf = HierarchicalClassifier(h_schema, model="test-model")
        result = clf.classify("I want a refund for $50")

        assert result.domain.intent == "billing"
        assert result.intent.intent == "refund_request"
        assert result.intent.entities == {"amount": "50"}
