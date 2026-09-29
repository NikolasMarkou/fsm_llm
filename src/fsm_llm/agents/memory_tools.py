"""
Memory management tools for FSM-LLM agents.

Provides tool functions that allow agents to explicitly manage their
working memory: remember facts, recall by query, forget keys, and
list stored memories. Implements the "memory management is an action"
principle from CoALA and Letta/MemGPT research.

Usage::

    from fsm_llm.memory import WorkingMemory
    from fsm_llm.agents.memory_tools import create_memory_tools

    memory = WorkingMemory()
    tools = create_memory_tools(memory)

    # Register with a ToolRegistry
    registry = ToolRegistry()
    for tool_def in tools:
        registry.register(tool_def)
"""

from __future__ import annotations

from typing import Annotated

from fsm_llm.memory import BUFFER_CORE, WorkingMemory

from .definitions import ToolDefinition
from .tools import _infer_schema_from_hints, redact_secret_entries


def create_memory_tools(
    memory: WorkingMemory,
) -> list[ToolDefinition]:
    """Create memory management tools bound to a WorkingMemory instance.

    Returns a list of ``ToolDefinition`` objects ready to register
    with a ``ToolRegistry``.

    Args:
        memory: The WorkingMemory instance these tools will operate on.

    Returns:
        List of 4 tool definitions: remember, recall, forget, list_memories.

    Hidden buffers (``memory.hidden_buffers``, e.g. ``metadata``) are never
    listed, forgotten or written, and secret-looking values are shown as
    ``<redacted>`` (``redact_secret_entries``, D-016).
    """

    def visible_buffers() -> list[str]:
        return [b for b in memory.list_buffers() if b not in memory.hidden_buffers]

    def remember(
        key: Annotated[str, "The key/name to store the value under"],
        value: Annotated[str, "The value to remember"],
        buffer: Annotated[
            str,
            "Which memory buffer to store in (core, scratch, environment, reasoning)",
        ] = BUFFER_CORE,
    ) -> str:
        """Store a key-value pair in working memory.

        Use this to remember important facts, intermediate results,
        or user preferences for later use.
        """
        if buffer in memory.hidden_buffers:
            raise ValueError(f"Buffer '{buffer}' is hidden and cannot be written.")
        memory.set(buffer, key, value)
        return f"Remembered '{key}' in {buffer} buffer."

    def recall(
        query: Annotated[str, "Search query to find matching memories"],
    ) -> str:
        """Search working memory for entries matching a query.

        Searches across all memory buffers by key name and value content.
        Returns matching entries with their buffer location.
        """
        results = memory.search(query, limit=10)
        if not results:
            return f"No memories found matching '{query}'."

        lines = [f"Found {len(results)} matching memories:"]
        for buffer_name, key, raw in results:
            value = redact_secret_entries({key: raw})[key]
            value_preview = str(value)[:200]
            if len(str(value)) > 200:
                value_preview += "..."
            lines.append(f"  [{buffer_name}] {key} = {value_preview}")
        return "\n".join(lines)

    def forget(
        key: Annotated[str, "The key/name to remove from memory"],
    ) -> str:
        """Remove a key from working memory.

        Searches all buffers and removes the first match found.
        """
        for buffer_name in visible_buffers():
            if memory.delete(buffer_name, key):
                return f"Forgot '{key}' from {buffer_name} buffer."
        return f"Key '{key}' not found in any memory buffer."

    def list_memories(
        buffer: Annotated[
            str,
            "Which buffer to list (core, scratch, environment, reasoning, or 'all')",
        ] = "all",
    ) -> str:
        """List all keys stored in working memory.

        Shows key names and value previews organized by buffer.
        """
        if buffer != "all":
            if not memory.has_buffer(buffer) or buffer in memory.hidden_buffers:
                return f"Buffer '{buffer}' does not exist."
            data = redact_secret_entries(memory.get_buffer(buffer))
            if not data:
                return f"Buffer '{buffer}' is empty."
            lines = [f"Buffer '{buffer}' ({len(data)} entries):"]
            for key, value in data.items():
                preview = str(value)[:100]
                if len(str(value)) > 100:
                    preview += "..."
                lines.append(f"  {key} = {preview}")
            return "\n".join(lines)

        # List all buffers
        all_buffers = visible_buffers()
        if not all_buffers:
            return "No memory buffers exist."

        shown = {
            name: redact_secret_entries(memory.get_buffer(name)) for name in all_buffers
        }
        total = sum(len(data) for data in shown.values())
        if total == 0:
            return "All memory buffers are empty."

        lines = [f"Working memory ({total} total entries):"]
        for buf_name, data in shown.items():
            lines.append(f"\n  [{buf_name}] ({len(data)} entries):")
            for key, value in data.items():
                preview = str(value)[:100]
                if len(str(value)) > 100:
                    preview += "..."
                lines.append(f"    {key} = {preview}")
        return "\n".join(lines)

    # Build ToolDefinition objects with auto-inferred schemas
    tools = []
    for fn in (remember, recall, forget, list_memories):
        schema = _infer_schema_from_hints(fn)
        tool_def = ToolDefinition(
            name=fn.__name__,
            description=(fn.__doc__ or "").strip().split("\n")[0],
            parameter_schema=schema,
            execute_fn=fn,
        )
        tools.append(tool_def)

    return tools
