"""Handler registration for ``API`` (``HandlerRegistrationMixin``): ``register_handler``,
``register_handlers`` and ``create_handler``.

Layer ``engine`` (plan 5a6a1f80, step 3): moved out of ``conversation.py`` to give the shell room;
the method bodies are unchanged. Each one only forwards to ``self.handler_system``.
"""

from __future__ import annotations

from typing import Any, cast

from fsm_llm.handlers import (
    BaseHandler,
    FSMHandler,
    HandlerBuilder,
    HandlerTiming,
    create_handler,
)

from .surface import _ApiSurface


class HandlerRegistrationMixin(_ApiSurface):
    """Register handlers on the one ``HandlerSystem`` an ``API`` owns."""

    # ==========================================
    # HANDLER MANAGEMENT METHODS
    # ==========================================

    def register_handler(self, handler: FSMHandler) -> None:
        """Register handler with the system."""
        self.handler_system.register_handler(handler)

    def register_handlers(self, handlers: list[FSMHandler]) -> None:
        """Register multiple handlers."""
        for handler in handlers:
            self.register_handler(handler)

    def create_handler(
        self,
        name: str = "CustomHandler",
        timing: HandlerTiming | None = None,
        action: Any | None = None,
    ) -> HandlerBuilder:
        """Create new handler using fluent builder.

        When *timing* and *action* are both provided the handler is built
        and registered automatically, providing a convenient shorthand::

            fsm.create_handler(
                name="on_start",
                timing=HandlerTiming.START_CONVERSATION,
                action=lambda ctx: print("started"),
            )
        """
        # mypy: `do()` returns a built BaseHandler (not the builder), so the local
        # is annotated as the union. register_handler receives a BaseHandler at
        # runtime (only reachable after `.do()` ran); cast narrows for the Protocol.
        # The declared `-> HandlerBuilder` return is pre-existing and unchanged;
        # cast preserves the exact runtime object returned. Annotation-only.
        builder: HandlerBuilder | BaseHandler = create_handler(name)
        if timing is not None:
            builder = cast(HandlerBuilder, builder).at(timing)
        if action is not None:
            builder = cast(HandlerBuilder, builder).do(action)
        if timing is not None and action is not None:
            self.register_handler(cast(FSMHandler, builder))
        return cast(HandlerBuilder, builder)
