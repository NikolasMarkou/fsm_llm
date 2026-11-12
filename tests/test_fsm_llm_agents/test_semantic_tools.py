"""SemanticToolRegistry: retrieval, ``top_k`` bounds, embedding-failure log level."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fsm_llm.agents.semantic_tools import SemanticToolRegistry


class TestSemanticToolRetrieveTopK:
    """F-AI-18: ``top_k=0`` meant the default and ``-1`` all but one."""

    @staticmethod
    def _registry() -> SemanticToolRegistry:
        registry = SemanticToolRegistry(
            embed_fn=lambda texts: [[float(len(t)), 1.0] for t in texts]
        )
        for i in range(12):
            registry.register_function(
                lambda: "x", name=f"tool_{i}", description="d" * (i + 1)
            )
        return registry

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_top_k_is_refused(self, bad: int):
        with pytest.raises(ValueError, match="top_k"):
            self._registry().retrieve("query", top_k=bad)

    def test_a_valid_top_k_still_limits(self):
        assert len(self._registry().retrieve("query", top_k=2)) == 2


class TestSemanticToolsLogLevel:
    """Verify embedding failures logged at warning level."""

    def test_embed_failure_logs_warning(self):
        """Failed embedding logs warning, not debug."""
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        with patch.object(
            SemanticToolRegistry,
            "_get_embedding",
            side_effect=RuntimeError("no model"),
        ):
            with patch("fsm_llm.agents.semantic_tools.logger") as mock_logger:
                registry = SemanticToolRegistry(auto_embed=True)
                registry.register_function(
                    lambda x: x, name="test-tool", description="Test"
                )
                warning_calls = [
                    c
                    for c in mock_logger.warning.call_args_list
                    if "test-tool" in str(c)
                ]
                assert len(warning_calls) > 0


class TestSemanticToolRegistry:
    """Tests for SemanticToolRegistry."""

    def test_inherits_tool_registry(self):
        """SemanticToolRegistry is a ToolRegistry subclass."""
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry
        from fsm_llm.agents.tools import ToolRegistry

        assert issubclass(SemanticToolRegistry, ToolRegistry)

    def test_register_without_embedding(self):
        """Tools register even when embedding fails."""
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        registry = SemanticToolRegistry(auto_embed=False)
        registry.register_function(lambda x: x, name="test", description="Test tool")
        assert "test" in registry
        assert registry.embedded_tool_count == 0

    def test_fallback_for_small_registry(self):
        """Small registries return all tools without embedding."""
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        registry = SemanticToolRegistry(auto_embed=False)
        for i in range(5):
            registry.register_function(
                lambda x: x, name=f"tool-{i}", description=f"Tool {i}"
            )
        result = registry.retrieve("query")
        assert len(result) == 5  # All tools returned

    def test_cosine_similarity(self):
        """Cosine similarity computation is correct."""
        from fsm_llm.agents.semantic_tools import cosine_similarity

        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
        assert cosine_similarity([0, 0], [1, 1]) == pytest.approx(0.0)

    @patch("fsm_llm.agents.semantic_tools.SemanticToolRegistry._get_embedding")
    def test_retrieve_with_mock_embeddings(self, mock_embed):
        """Semantic retrieval returns top-K tools by similarity."""
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        # Set up mock embeddings
        embeddings = {
            "search": [1.0, 0.0, 0.0],
            "calculate": [0.0, 1.0, 0.0],
            "translate": [0.0, 0.0, 1.0],
        }
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            for name, emb in embeddings.items():
                if name in text.lower():
                    return emb
            # Query embedding - assume it's about search
            return [0.9, 0.1, 0.0]

        mock_embed.side_effect = side_effect

        registry = SemanticToolRegistry(auto_embed=True)
        registry.FALLBACK_THRESHOLD = 0  # Force embedding path

        for name in ["search", "calculate", "translate"]:
            registry.register_function(
                lambda x: x, name=name, description=f"{name.title()} tool"
            )

        # Register enough tools to bypass fallback
        result = registry.retrieve("find information", top_k=2)
        assert len(result) <= 2
        assert result[0].name == "search"  # Most similar

    def test_prompt_description_without_query(self):
        """to_prompt_description without query returns all tools."""
        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        registry = SemanticToolRegistry(auto_embed=False)
        registry.register_function(lambda: None, name="tool-a", description="Tool A")
        desc = registry.to_prompt_description()
        assert "tool-a" in desc
