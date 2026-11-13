"""
LLM Interface Module for FSM-Driven Conversational AI.

This module provides the core interface and implementation for Large Language Model (LLM)
communication within the fsm-llm library's 2-pass architecture. It defines how
FSM-driven applications interact with various LLM providers while maintaining clear
separation of concerns between data extraction and response generation (transitions are
decided by rules; the LLM is asked only to break a tie, through the classifier).

Architecture Overview
---------------------
The module implements a 2-pass architecture that separates LLM operations into distinct phases:

1. **Data Extraction Pass**: Extract and understand information from user input without
   generating any user-facing content. This prevents premature response generation and
   ensures all necessary data is captured before state transitions.

2. **Response Generation Pass**: Generate appropriate user-facing messages based on the
   final state context after all data extraction and transition evaluation is complete.

Key Components
--------------
LLMInterface : abc.ABC
    Abstract base class defining the contract for LLM communication with support for
    the 2-pass architecture. All LLM implementations must inherit from this interface.

LiteLLMInterface : LLMInterface
    Concrete implementation using LiteLLM for multi-provider support. Handles OpenAI,
    Anthropic, and other popular LLM providers through a unified interface.

Core Methods
------------
generate_response(request: ResponseGenerationRequest) -> ResponseGenerationResponse
    Generate user-facing messages based on final state context and extracted data.

generate_response_stream(request: ResponseGenerationRequest) -> Iterator[str]
    The same Pass 2 reply, streamed as text fragments.

extract_field(request: FieldExtractionRequest) -> FieldExtractionResponse
    Extract a single specific field from user input with custom instructions.

extract_bulk_data(request: BulkExtractionRequest) -> DataExtractionResponse
    Extract several fields from one user input in a single request.

complete(request: CompletionRequest) -> CompletionResponse
    The one request primitive: tool calling, plain and structured completion.

Integration with FSM System
---------------------------
This module integrates with the broader fsm-llm system:

- **FSMManager** (`engine/fsm/`) orchestrates the overall FSM execution and calls these
  methods at appropriate points in the conversation flow.
- **PromptBuilder** (`context/layout/`) constructs the specialized prompts for each pass
  based on current FSM state, context, and conversation history.
- **API** (`api.py`) provides the high-level interface that developers use, internally
  coordinating between FSM management and LLM communication.

Layout (plan d454fd62, step 73)
-------------------------------
This package is the old ``llm.py`` split by concern. ``__init__`` re-exports the public
names with a static ``__all__`` (P1) so ``from fsm_llm.llm import LiteLLMInterface`` keeps
working; a private helper is imported from its defining module.

- ``interface``: ``LLMInterface``, ``tool_exchange``, ``check_tool_transcript``,
  ``resolve_model``, ``PLACEHOLDER_MODEL``, the connection defaults.
- ``usage``: the per-instance provider-call meter and ``_Metered``.
- ``parsing``: reply parsing helpers and ``ParseMixin``.
- ``providers.litellm``: ``LiteLLMInterface`` and the ONE ``import litellm`` of the package
  with its environment guard and banner setting; ``providers.request``:
  ``RequestMixin``, the one provider request builder; ``providers.embedder``:
  ``LiteLLMEmbedder``.
- ``ollama``, ``text``, ``json_extraction``: Ollama parameter helpers and reply text helpers.

``completion``, ``embedding`` and ``get_supported_openai_params`` are NOT names of this
package: a test replaces them on the module whose code looks them up
(``fsm_llm.llm.providers.litellm`` and ``fsm_llm.llm.providers.embedder``).
"""

from fsm_llm.core.definitions import (
    CompletionRequest,
    CompletionResponse,
    LLMResponseError,
    LLMUsage,
    ModelNotConfiguredError,
    ModelToolCall,
)

from .interface import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS,
    PLACEHOLDER_MODEL,
    LLMInterface,
    check_tool_transcript,
    decode_tool_arguments,
    implements_complete,
    interface_model,
    is_malformed_tool_call_error,
    resolve_model,
    tool_exchange,
)
from .parsing import cut_off_note
from .providers.embedder import LiteLLMEmbedder
from .providers.litellm import LiteLLMInterface

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LLM_TIMEOUT_SECONDS",
    "DEFAULT_MAX_TOKENS",
    "PLACEHOLDER_MODEL",
    "CompletionRequest",
    "CompletionResponse",
    "LLMInterface",
    "LLMResponseError",
    "LLMUsage",
    "LiteLLMEmbedder",
    "LiteLLMInterface",
    "ModelNotConfiguredError",
    "ModelToolCall",
    "check_tool_transcript",
    "cut_off_note",
    "decode_tool_arguments",
    "implements_complete",
    "interface_model",
    "is_malformed_tool_call_error",
    "resolve_model",
    "tool_exchange",
]
