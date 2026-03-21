"""Tests for classification prompt building."""


from fsm_llm_classification import (
    ClassificationPromptConfig,
    ClassificationSchema,
    IntentDefinition,
    build_json_schema,
    build_system_prompt,
)


def _schema():
    return ClassificationSchema(
        intents=[
            IntentDefinition(name="order_status", description="About orders"),
            IntentDefinition(name="general_support", description="Fallback"),
        ],
        fallback_intent="general_support",
    )


class TestBuildJsonSchema:
    def test_single_intent_schema(self):
        s = build_json_schema(_schema())
        assert s["type"] == "object"
        assert "intent" in s["properties"]
        assert s["properties"]["intent"]["enum"] == ["order_status", "general_support"]
        assert s["additionalProperties"] is False

    def test_reasoning_field_comes_first(self):
        s = build_json_schema(_schema(), include_reasoning=True)
        keys = list(s["properties"].keys())
        assert keys[0] == "reasoning"
        assert keys[1] == "intent"

    def test_no_reasoning(self):
        s = build_json_schema(_schema(), include_reasoning=False)
        assert "reasoning" not in s["properties"]

    def test_multi_intent_schema(self):
        s = build_json_schema(_schema(), multi_intent=True, max_intents=3)
        assert "intents" in s["properties"]
        items = s["properties"]["intents"]["items"]
        assert items["properties"]["intent"]["enum"] == ["order_status", "general_support"]
        assert s["properties"]["intents"]["maxItems"] == 3

    def test_no_entities(self):
        s = build_json_schema(_schema(), include_entities=False)
        assert "entities" not in s["properties"]


class TestBuildSystemPrompt:
    def test_includes_intent_descriptions(self):
        prompt = build_system_prompt(_schema())
        assert "order_status: About orders" in prompt
        assert "general_support: Fallback" in prompt

    def test_includes_json_schema(self):
        prompt = build_system_prompt(_schema())
        assert '"enum"' in prompt

    def test_multi_intent_mode(self):
        config = ClassificationPromptConfig(multi_intent=True)
        prompt = build_system_prompt(_schema(), config=config)
        assert "multiple intents" in prompt.lower()

    def test_fallback_instruction(self):
        prompt = build_system_prompt(_schema())
        assert "general_support" in prompt
