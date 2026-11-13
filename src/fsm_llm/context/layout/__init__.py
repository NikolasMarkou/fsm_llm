"""
Prompt Builders FSM-LLM Architecture.

This module provides specialized prompt builders that support:
1. Data extraction prompts (focused on understanding and extracting information)
2. Transition decision prompts (focused on choosing between transition options)
3. Response generation prompts (focused on generating user-facing messages)

The architecture prevents FSM structure leakage while maintaining rich prompting
capabilities and ensuring responses are generated with full context of the final state.

Key Features:
- Information isolation prevents FSM structure leakage
- Rich XML-like prompt structure with comprehensive sections
- Configurable token budget and history management
- Comprehensive text sanitization and security measures
- Few-shot examples and detailed guidelines
- CDATA protection for JSON data

The package is the one import surface of the prompt layout (``from fsm_llm.context.layout
import BasePromptBuilder``); each name is defined in the module named in
``module_path_map.json`` (``base``, ``extraction``, ``response``, ``classification``) and
re-exported here, so a patch or a private-name import aims at the defining module. It is the old
``fsm_llm.prompts`` (plan d454fd62, step 74): the prompt-sanitizer allow-list and tag pattern
(``_SAFE_TAGS``, ``_TAG_PATTERN``) stay together on ``BasePromptBuilder`` in ``base``.
"""

from .base import (
    BasePromptBuilder,
    BasePromptConfig,
    HistoryManagementStrategy,
)
from .classification import (
    build_classification_context_block,
    build_classification_json_schema,
    build_classification_system_prompt,
    sanitize_text_for_prompt,
)
from .extraction import (
    DataExtractionPromptBuilder,
    DataExtractionPromptConfig,
    FieldExtractionPromptBuilder,
    FieldExtractionPromptConfig,
)
from .response import ResponseGenerationPromptBuilder, ResponsePromptConfig

__all__ = [
    "BasePromptBuilder",
    "BasePromptConfig",
    "DataExtractionPromptBuilder",
    "DataExtractionPromptConfig",
    "FieldExtractionPromptBuilder",
    "FieldExtractionPromptConfig",
    "HistoryManagementStrategy",
    "ResponseGenerationPromptBuilder",
    "ResponsePromptConfig",
    "build_classification_context_block",
    "build_classification_json_schema",
    "build_classification_system_prompt",
    "sanitize_text_for_prompt",
]
