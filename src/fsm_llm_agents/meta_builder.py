from __future__ import annotations

"""
MetaBuilderAgent — agentic artifact builder with classification-based tool routing.

Uses a classify-then-extract loop to build artifacts incrementally:
1. Classify which tool to call next (reliable on small models)
2. Extract that tool's parameters via targeted prompts
3. Execute the tool
4. Loop until validate() passes or max iterations reached

This approach works with small models (4B+) because each LLM call
is focused: either pick from a list OR extract one value.
"""

import json
from typing import Any, ClassVar

import litellm

from fsm_llm.logging import logger

from .constants import MetaErrorMessages, MetaLogMessages
from .definitions import (
    AgentResult,
    ArtifactType,
    MetaBuilderConfig,
    MetaBuilderResult,
)
from .exceptions import MetaBuilderError
from .meta_builders import (
    AgentBuilder,
    ArtifactBuilder,
    FSMBuilder,
    MonitorBuilder,
    WorkflowBuilder,
)
from .meta_output import format_artifact_json
from .meta_prompts import build_review_presentation
from .meta_tools import create_builder_tools
from .tools import ToolRegistry


class MetaBuilderAgent:
    """Agentic meta-builder using classification-based tool routing.

    Each iteration:
    1. LLM classifies which tool to call next (from a numbered list)
    2. LLM extracts that tool's parameters (one focused prompt)
    3. Tool is executed on the builder
    4. Loop until validate() passes

    Works reliably with small models (4B+) because each call is focused.

    Usage (single-shot)::

        agent = MetaBuilderAgent()
        result = agent.run("Build a customer support chatbot with 3 states")
        print(result.artifact_json)

    Usage (turn-by-turn)::

        agent = MetaBuilderAgent()
        response = agent.start()
        while not agent.is_complete():
            response = agent.send(input("> "))
        result = agent.get_result()
    """

    @classmethod
    def _build_type_aliases(cls) -> dict[str, str]:
        """Return type alias map sorted longest-first for correct matching."""
        raw: dict[str, str] = {
            "finite state machine": "fsm",
            "state machine": "fsm",
            "state_machine": "fsm",
            "conversational": "fsm",
            "conversation": "fsm",
            "help desk": "fsm",
            "onboarding": "fsm",
            "interview": "fsm",
            "chat bot": "fsm",
            "helpdesk": "fsm",
            "chatbot": "fsm",
            "dialogue": "fsm",
            "dialog": "fsm",
            "survey": "fsm",
            "quiz": "fsm",
            "faq": "fsm",
            "bot": "fsm",
            "data pipeline": "workflow",
            "automation": "workflow",
            "pipeline": "workflow",
            "sequence": "workflow",
            "process": "workflow",
            "steps": "workflow",
            "batch": "workflow",
            "flow": "workflow",
            "etl": "workflow",
            "agentic": "agent",
            "research": "agent",
            "navigate": "agent",
            "browse": "agent",
            "search": "agent",
            "react": "agent",
            "tools": "agent",
            "tool": "agent",
            "monitoring dashboard": "monitor",
            "monitor dashboard": "monitor",
            "web dashboard": "monitor",
            "dashboard": "monitor",
            "monitoring": "monitor",
            "telemetry": "monitor",
            "metrics": "monitor",
            "observability": "monitor",
        }
        return dict(sorted(raw.items(), key=lambda kv: -len(kv[0])))

    _JUST_BUILD_PHRASES: ClassVar[frozenset[str]] = frozenset(
        {
            "just build it",
            "just build",
            "whatever",
            "anything",
            "surprise me",
            "random",
            "just do it",
            "just make it",
            "build something",
            "make something",
        }
    )

    def __init__(
        self,
        config: MetaBuilderConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        if config is None:
            config = MetaBuilderConfig()
        self.meta_config = config
        self._api_kwargs = api_kwargs

        self._artifact_type: ArtifactType | None = None
        self._builder: ArtifactBuilder | None = None
        self._result: MetaBuilderResult | None = None

        self._started = False
        self._complete = False
        self._messages: list[str] = []
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Single-shot API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the meta-builder in single-shot mode."""
        logger.info(MetaLogMessages.META_STARTED.format(model=self.meta_config.model))

        artifact_type = self._detect_type(task)
        self._artifact_type = artifact_type
        builder = self._create_builder(artifact_type)
        self._builder = builder
        tools = create_builder_tools(builder, artifact_type)

        logger.info(
            MetaLogMessages.BUILD_STARTED.format(artifact_type=artifact_type.value)
        )

        self._run_build_loop(task, tools, artifact_type)

        self._build_result()
        self._complete = True
        return self._result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Classification-based build loop
    # ------------------------------------------------------------------

    def _run_build_loop(
        self,
        task: str,
        tools: ToolRegistry,
        artifact_type: ArtifactType,
    ) -> None:
        """Run the classify → extract → execute loop."""
        tool_list = tools.list_tools()
        tool_names = [t.name for t in tool_list]
        tool_map = {t.name: t for t in tool_list}
        max_iter = self.meta_config.max_iterations
        observations: list[str] = []

        for iteration in range(1, max_iter + 1):
            logger.debug(f"Build iteration {iteration}/{max_iter}")

            # Step 1: Classify which tool to call next
            tool_name = self._classify_next_tool(
                task, artifact_type, tool_names, observations
            )

            if tool_name == "done" or tool_name not in tool_map:
                logger.info(
                    f"Build loop ended at iteration {iteration}: tool={tool_name}"
                )
                break

            tool_def = tool_map[tool_name]
            props = tool_def.parameter_schema.get("properties", {})
            required = tool_def.parameter_schema.get("required", [])

            # Step 2: Extract parameters for the selected tool
            if props:
                params = self._extract_tool_params(
                    task, tool_name, props, required, observations
                )
            else:
                params = {}

            # Step 3: Execute the tool
            from .definitions import ToolCall

            result = tools.execute(ToolCall(tool_name=tool_name, parameters=params))

            obs = f"[{iteration}] {tool_name}({params}) → {result.summary[:200]}"
            observations.append(obs)
            logger.info(
                f"Tool '{tool_name}' {'OK' if result.success else 'FAILED'}: "
                f"{result.summary[:100]}"
            )

            # Auto-stop if validate returns clean
            if (
                tool_name == "validate"
                and result.success
                and "no errors" in result.summary.lower()
            ):
                logger.info("Validation passed — stopping build loop")
                break

    def _classify_next_tool(
        self,
        task: str,
        artifact_type: ArtifactType,
        tool_names: list[str],
        observations: list[str],
    ) -> str:
        """Ask the LLM to pick the next tool from a numbered list."""
        # Build numbered tool list
        tool_list_str = "\n".join(
            f"  {i + 1}. {name}" for i, name in enumerate(tool_names)
        )
        tool_list_str += f"\n  {len(tool_names) + 1}. done (finished building)"

        obs_str = "\n".join(observations[-8:]) if observations else "(none yet)"

        prompt = (
            f"You are building a {artifact_type.value.upper()} artifact.\n"
            f"User request: {task}\n\n"
            f"Tools already called:\n{obs_str}\n\n"
            f"Available tools:\n{tool_list_str}\n\n"
            f"Which tool should be called NEXT? "
            f"Reply with ONLY the tool name (e.g. 'set_overview' or 'done')."
        )

        response = self._llm_call(prompt)
        choice = response.strip().lower().strip("'\"`.").split("\n")[0].strip()

        # Try to match to a tool name
        for name in tool_names:
            if name in choice:
                return name
        if "done" in choice or "finish" in choice or "stop" in choice:
            return "done"

        # Try numbered response
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(tool_names):
                return tool_names[idx]
            if idx == len(tool_names):
                return "done"
        except ValueError:
            pass

        # Default: first tool not yet called, or done
        called = {
            o.split("]")[1].split("(")[0].strip() for o in observations if "]" in o
        }
        for name in tool_names:
            if name not in called:
                return name
        return "done"

    def _extract_tool_params(
        self,
        task: str,
        tool_name: str,
        properties: dict[str, Any],
        required: list[str],
        observations: list[str],
    ) -> dict[str, Any]:
        """Extract parameters for a specific tool via structured LLM call."""
        obs_str = "\n".join(observations[-5:]) if observations else "(none)"

        prompt = (
            f"You are building an artifact. User request: {task}\n\n"
            f"Previous actions:\n{obs_str}\n\n"
            f"Now calling tool: {tool_name}\n"
            f"Provide the parameter values as JSON."
        )

        # Build JSON schema for response_format enforcement
        response_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        response = self._llm_call(prompt, response_schema=response_schema)
        return self._parse_json_params(response, properties)

    @staticmethod
    def _format_example_json(keys: list[str]) -> str:
        """Format an example JSON object for the prompt."""
        pairs = ", ".join(f'"{k}": "..."' for k in keys)
        return "{" + pairs + "}"

    def _parse_json_params(
        self, response: str, properties: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse JSON parameters from LLM response."""
        text = response.strip()

        # Try direct JSON parse
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return self._coerce_params(parsed, properties)
            except json.JSONDecodeError:
                pass

        # Try to find JSON in response
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return self._coerce_params(parsed, properties)
            except json.JSONDecodeError:
                pass

        # Fallback: empty params
        logger.warning(f"Could not parse params from: {text[:200]}")
        return {}

    @staticmethod
    def _coerce_params(
        params: dict[str, Any], properties: dict[str, Any]
    ) -> dict[str, Any]:
        """Coerce parameter types to match schema."""
        result: dict[str, Any] = {}
        for key, value in params.items():
            if key not in properties:
                continue
            expected = properties[key].get("type", "string")
            if expected == "integer" and isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    pass
            elif expected == "number" and isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
            result[key] = value
        return result

    def _llm_call(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Make a single LLM call with thinking disabled.

        Args:
            prompt: The user prompt.
            response_schema: Optional JSON schema to enforce structured output
                via ``response_format``. The schema is also included in the
                prompt so the model sees it as context.

        Returns:
            Response text, or empty string on failure.
        """
        from fsm_llm.ollama import is_ollama_model

        model = self.meta_config.model
        reserved = {"model", "messages", "temperature", "max_tokens"}
        safe_kwargs = {k: v for k, v in self._api_kwargs.items() if k not in reserved}

        # Prepend /nothink for Qwen3 models on Ollama
        nothink_prefix = "/nothink\n" if is_ollama_model(model) else ""
        full_prompt = f"{nothink_prefix}{prompt}"

        # If schema provided, append it to prompt so model sees it as context
        if response_schema:
            schema_str = json.dumps(response_schema, indent=2)
            full_prompt += f"\n\nRespond in JSON matching this schema:\n{schema_str}"

        call_params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0,
            "max_tokens": self.meta_config.max_tokens,
            **safe_kwargs,
        }

        # Disable thinking via reasoning_effort=none
        if is_ollama_model(model):
            call_params["reasoning_effort"] = "none"

        # Apply response_format for structured output
        if response_schema:
            call_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "tool_params",
                    "schema": response_schema,
                },
            }

        try:
            response = litellm.completion(**call_params)
            content = response.choices[0].message.content
            if not content and hasattr(response.choices[0].message, "thinking"):
                thinking = response.choices[0].message.thinking or ""
                if thinking:
                    for line in reversed(thinking.strip().split("\n")):
                        line = line.strip()
                        if line and not line.startswith("<"):
                            return line
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Turn-by-turn API (for monitor server + interactive)
    # ------------------------------------------------------------------

    def start(self, initial_message: str = "") -> str:
        """Initialize a builder session."""
        if self._started:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_ALREADY_STARTED)
        self._started = True

        if initial_message:
            self._messages.append(initial_message)
            artifact_type = self._detect_type(initial_message)
            self._artifact_type = artifact_type
            return (
                f"I'll build a {artifact_type.value.upper()} based on your "
                f"description. Tell me more details, "
                f"or say 'build it' when you're ready."
            )

        return (
            "Welcome! I can help you build:\n"
            "  1. An FSM for stateful conversations\n"
            "  2. A Workflow for multi-step processes\n"
            "  3. An Agent for tool-using AI\n"
            "  4. A Monitor dashboard for metrics and alerts\n\n"
            "Describe what you'd like to create."
        )

    def send(self, message: str) -> str:
        """Send a message in a turn-by-turn session."""
        if not self._started:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_NOT_STARTED)
        if self._complete:
            raise MetaBuilderError("Session has already completed")

        self._turn_count += 1
        if self._turn_count > self.meta_config.max_turns:
            raise MetaBuilderError(
                f"Maximum turns ({self.meta_config.max_turns}) exceeded"
            )

        self._messages.append(message)
        normalized = message.strip().lower()

        if self._is_build_trigger(normalized):
            return self._execute_build()

        if self._artifact_type is None:
            self._artifact_type = self._detect_type(message)

        if self._artifact_type:
            return (
                f"Got it — adding to your {self._artifact_type.value.upper()} "
                f"spec. Say 'build it' when ready, or keep adding details."
            )

        return (
            "I'm not sure what type of artifact you want. "
            "Could you mention: FSM, Workflow, Agent, or Monitor?"
        )

    def is_complete(self) -> bool:
        return self._complete

    def get_result(self) -> MetaBuilderResult:
        if not self._complete:
            raise MetaBuilderError(
                "Build is not complete. Say 'build it' to trigger the build."
            )
        if self._result is None:
            self._build_result()
        return self._result  # type: ignore[return-value]

    def get_internal_state(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "phase": "complete" if self._complete else "collecting",
            "turn_count": self._turn_count,
            "is_complete": self._complete,
            "started": self._started,
            "message_count": len(self._messages),
        }
        if self._artifact_type is not None:
            result["artifact_type"] = self._artifact_type.value

        builder = self._builder
        if builder is not None:
            progress = builder.get_progress()
            validation_errors = builder.validate_complete()
            result["builder_progress"] = {
                "percentage": progress.percentage,
                "completed": progress.completed,
                "total_required": progress.total_required,
                "missing": builder.get_missing_fields(),
                "warnings": progress.warnings,
            }
            result["builder_summary"] = builder.get_summary(detail_level="standard")
            result["artifact_preview"] = builder.to_dict()
            result["validation_errors"] = validation_errors
            result["is_valid"] = len(validation_errors) == 0
        else:
            result["builder_progress"] = None
            result["builder_summary"] = None
            result["artifact_preview"] = None
            result["validation_errors"] = []
            result["is_valid"] = False

        return result

    def run_interactive(self) -> MetaBuilderResult:
        response = self.start()
        print(f"\n{response}\n")

        while not self.is_complete():
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended by user.")
                break
            if not user_input.strip():
                continue
            response = self.send(user_input)
            print(f"\n{response}\n")

        if self.is_complete():
            return self.get_result()
        self._build_result()
        return self._result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Build execution (for turn-by-turn)
    # ------------------------------------------------------------------

    def _execute_build(self) -> str:
        combined_task = "\n".join(self._messages)
        try:
            self.run(combined_task)
        except Exception as e:
            logger.error(f"Build execution failed: {e}")
            self._build_result()

        self._complete = True

        if self._result and self._result.is_valid and self._builder:
            presentation = build_review_presentation(
                self._builder, self._artifact_type or ArtifactType.FSM
            )
            return (
                f"Build complete!\n\n{presentation}\n\n"
                f"The artifact JSON has been generated."
            )

        errors = self._result.validation_errors if self._result else ["Build failed"]
        return "Build completed with issues:\n" + "\n".join(f"  - {e}" for e in errors)

    @staticmethod
    def _is_build_trigger(normalized: str) -> bool:
        triggers = {
            "build it",
            "build",
            "go",
            "build now",
            "create it",
            "make it",
            "generate",
            "done",
            "finish",
            "approve",
            "yes",
            "ok",
            "lgtm",
            "ship it",
            "do it",
        }
        return normalized in triggers or any(
            t in normalized for t in ("build it", "create it", "generate it")
        )

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    def _detect_type(self, text: str) -> ArtifactType:
        normalized = text.strip().lower()
        if normalized in self._JUST_BUILD_PHRASES:
            return ArtifactType.FSM
        aliases = self._build_type_aliases()
        for alias, type_str in aliases.items():
            if alias in normalized:
                try:
                    return ArtifactType(type_str)
                except ValueError:
                    pass
        return ArtifactType.FSM

    # ------------------------------------------------------------------
    # Builder creation + result
    # ------------------------------------------------------------------

    def _create_builder(
        self, artifact_type: ArtifactType
    ) -> FSMBuilder | WorkflowBuilder | AgentBuilder | MonitorBuilder:
        if artifact_type == ArtifactType.FSM:
            return FSMBuilder()
        if artifact_type == ArtifactType.WORKFLOW:
            return WorkflowBuilder()
        if artifact_type == ArtifactType.AGENT:
            return AgentBuilder()
        if artifact_type == ArtifactType.MONITOR:
            return MonitorBuilder()
        raise MetaBuilderError(f"Unknown artifact type: {artifact_type}")

    def _build_result(self) -> None:
        artifact_type = self._artifact_type or ArtifactType.FSM
        builder = self._builder

        if builder is None:
            self._result = MetaBuilderResult(
                answer="Build was not completed",
                success=False,
                artifact_type=artifact_type,
                artifact={},
                artifact_json="{}",
                is_valid=False,
                validation_errors=["Builder was not initialized"],
                conversation_turns=self._turn_count,
                final_context={},
            )
            return

        errors = builder.validate_complete()
        artifact = builder.to_dict()
        artifact_json = format_artifact_json(artifact)

        self._result = MetaBuilderResult(
            answer=artifact_json,
            success=len(errors) == 0,
            artifact_type=artifact_type,
            artifact=artifact,
            artifact_json=artifact_json,
            is_valid=len(errors) == 0,
            validation_errors=errors,
            conversation_turns=self._turn_count,
            final_context={
                "artifact_json": artifact,
                "artifact_type": artifact_type.value,
                "is_valid": len(errors) == 0,
                "validation_errors": errors,
            },
        )
