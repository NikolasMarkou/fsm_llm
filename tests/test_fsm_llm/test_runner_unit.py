from __future__ import annotations

"""Unit tests for runner.py — the CLI runner module."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestRunnerUsesAPI:
    """Verify runner uses API class instead of FSMManager."""

    def test_runner_imports_api_not_fsm_manager(self):
        """runner.py should import API, not FSMManager."""
        import inspect

        from fsm_llm import runner
        source = inspect.getsource(runner)
        assert "API" in source
        # Should not import FSMManager (mentions in comments/docstrings are OK)
        import_lines = [ln for ln in source.split("\n") if ln.strip().startswith(("from", "import"))]
        for line in import_lines:
            assert "FSMManager" not in line, f"runner.py imports FSMManager: {line}"

    def test_runner_main_requires_llm_model_env(self):
        """runner.main() should raise if LLM_MODEL env var is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with pytest.raises(RuntimeError, match="Missing required environment variable"):
                    from fsm_llm.runner import main
                    main(fsm_path=None, max_history_size=5, max_message_length=1000)

    def test_runner_main_creates_api_instance(self):
        """runner.main() should call API.from_file()."""
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.return_value = True
        mock_api.get_data.return_value = {}

        env = {
            "LLM_MODEL": "test-model",
            "LLM_TEMPERATURE": "0.5",
            "LLM_MAX_TOKENS": "100",
        }

        with patch.dict(os.environ, env, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with patch("fsm_llm.runner.API.from_file", return_value=mock_api) as mock_from_file:
                    with patch("fsm_llm.runner.setup_file_logging"):
                        from fsm_llm.runner import main
                        result = main(
                            fsm_path="/tmp/test.json",
                            max_history_size=5,
                            max_message_length=1000
                        )

                        mock_from_file.assert_called_once_with(
                            "/tmp/test.json",
                            model="test-model",
                            temperature=0.5,
                            max_tokens=100,
                            max_history_size=5,
                            max_message_length=1000
                        )
                        mock_api.start_conversation.assert_called_once()
                        mock_api.end_conversation.assert_called_once_with("conv-1")
                        assert result == 0

    def test_runner_uses_converse_not_process_message(self):
        """runner should use fsm.converse(), not fsm_manager.process_message()."""
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.side_effect = [False, True]
        mock_api.converse.return_value = "Response"
        mock_api.get_data.return_value = {}

        env = {"LLM_MODEL": "test-model"}

        with patch.dict(os.environ, env, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with patch("fsm_llm.runner.API.from_file", return_value=mock_api):
                    with patch("fsm_llm.runner.setup_file_logging"):
                        with patch("builtins.input", return_value="test input"):
                            from fsm_llm.runner import main
                            result = main("/tmp/t.json", 5, 1000)

                            mock_api.converse.assert_called_once_with(
                                user_message="test input",
                                conversation_id="conv-1"
                            )
                            assert result == 0

    def test_runner_handles_exit_command(self):
        """runner should exit cleanly on 'exit' command."""
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.return_value = False
        mock_api.get_data.return_value = {}

        env = {"LLM_MODEL": "test-model"}

        with patch.dict(os.environ, env, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with patch("fsm_llm.runner.API.from_file", return_value=mock_api):
                    with patch("fsm_llm.runner.setup_file_logging"):
                        with patch("builtins.input", return_value="exit"):
                            from fsm_llm.runner import main
                            result = main("/tmp/t.json", 5, 1000)
                            assert result == 0
                            # converse should NOT have been called
                            mock_api.converse.assert_not_called()
