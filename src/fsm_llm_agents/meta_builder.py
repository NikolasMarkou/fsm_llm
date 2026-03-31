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

from fsm_llm.classification import Classifier
from fsm_llm.definitions import ClassificationSchema, IntentDefinition
from fsm_llm.logging import logger

from .constants import MetaErrorMessages, MetaLogMessages
from .definitions import (
    ArtifactType,
    MetaBuilderConfig,
    MetaBuilderResult,
)
from .exceptions import MetaBuilderError
from .meta_builders import (
    AgentBuilder,
    ArtifactBuilder,
    FSMBuilder,
    WorkflowBuilder,
)
from .meta_output import format_artifact_json
from .meta_prompts import build_review_presentation


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

        # Lazy-initialized classifiers (built on first use)
        self._type_classifier: Classifier | None = None
        self._agent_type_classifier: Classifier | None = None

    # ------------------------------------------------------------------
    # Single-shot API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> MetaBuilderResult:
        """Run the meta-builder in single-shot mode."""
        logger.info(MetaLogMessages.META_STARTED.format(model=self.meta_config.model))

        artifact_type = self._detect_type(task)
        self._artifact_type = artifact_type
        builder = self._create_builder(artifact_type)
        self._builder = builder

        # Pre-set agent type from task description when detectable
        if artifact_type == ArtifactType.AGENT:
            self._preseed_agent_type(task, builder)

        logger.info(
            MetaLogMessages.BUILD_STARTED.format(artifact_type=artifact_type.value)
        )

        self._run_deterministic_pipeline(task, artifact_type, builder)

        self._build_result()
        self._complete = True
        return self._result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Deterministic build pipeline
    # ------------------------------------------------------------------

    # JSON schemas for single-call artifact extraction
    _FSM_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "persona": {"type": "string"},
            "states": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "state_id": {"type": "string"},
                        "description": {"type": "string"},
                        "purpose": {"type": "string"},
                    },
                    "required": ["state_id", "description", "purpose"],
                },
            },
            "transitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_state": {"type": "string"},
                        "target_state": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["from_state", "target_state", "description"],
                },
            },
        },
        "required": ["name", "description", "states"],
    }

    _WORKFLOW_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "workflow_id": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string"},
                        "step_type": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["step_id", "step_type", "name"],
                },
            },
        },
        "required": ["name", "description", "steps"],
    }

    _AGENT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                },
            },
        },
        "required": ["name", "description", "tools"],
    }

    _ARTIFACT_SCHEMAS: ClassVar[dict[str, dict]] = {
        "fsm": _FSM_SCHEMA,
        "workflow": _WORKFLOW_SCHEMA,
        "agent": _AGENT_SCHEMA,
    }

    def _run_deterministic_pipeline(
        self,
        task: str,
        artifact_type: ArtifactType,
        builder: ArtifactBuilder,
    ) -> None:
        """Extract the complete artifact spec in one LLM call, then build deterministically."""
        schema = self._ARTIFACT_SCHEMAS.get(artifact_type.value)
        if schema is None:
            logger.error(f"No extraction schema for {artifact_type.value}")
            return

        type_label = artifact_type.value.upper()
        prompt = (
            f"Design a {type_label} artifact based on the following requirement.\n\n"
            f"Requirement: {task}\n\n"
            f"Produce the complete specification as JSON."
        )

        response = self._llm_call(prompt, response_schema=schema)
        spec = self._parse_extraction_response(response)
        if not spec:
            logger.warning(
                "Extraction returned empty spec — builder will be incomplete"
            )
            return

        logger.debug(f"Extracted spec keys: {list(spec.keys())}")

        # Dispatch to type-specific assembly
        if artifact_type == ArtifactType.FSM:
            self._assemble_fsm(spec, builder)
        elif artifact_type == ArtifactType.WORKFLOW:
            self._assemble_workflow(spec, builder)
        elif artifact_type == ArtifactType.AGENT:
            self._assemble_agent(spec, builder)

    def _assemble_fsm(self, spec: dict[str, Any], builder: ArtifactBuilder) -> None:
        """Deterministic FSM assembly from extracted spec."""
        builder.set_overview(
            name=spec.get("name", "Unnamed FSM"),
            description=spec.get("description", ""),
            persona=spec.get("persona"),
        )

        states = spec.get("states", [])
        for i, state in enumerate(states):
            if not isinstance(state, dict):
                continue
            sid = state.get("state_id", f"state_{i}")
            try:
                builder.add_state(
                    state_id=sid,
                    description=state.get("description", sid),
                    purpose=state.get("purpose", sid),
                    extraction_instructions=state.get("extraction_instructions"),
                    response_instructions=state.get("response_instructions"),
                )
            except Exception as e:
                logger.warning(f"Failed to add state '{sid}': {e}")

        # Set initial state to first
        if states:
            first_id = states[0].get("state_id", "state_0")
            try:
                builder.set_initial_state(first_id)
            except Exception:
                pass

        # Add transitions
        for trans in spec.get("transitions", []):
            if not isinstance(trans, dict):
                continue
            try:
                builder.add_transition(
                    from_state=trans.get("from_state", ""),
                    target_state=trans.get("target_state", ""),
                    description=trans.get("description", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to add transition: {e}")

    def _assemble_workflow(
        self, spec: dict[str, Any], builder: ArtifactBuilder
    ) -> None:
        """Deterministic workflow assembly from extracted spec."""
        builder.set_overview(
            workflow_id=spec.get("workflow_id", "wf_default"),
            name=spec.get("name", "Unnamed Workflow"),
            description=spec.get("description", ""),
        )

        steps = spec.get("steps", [])
        step_ids: list[str] = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            sid = step.get("step_id", f"step_{i}")
            step_ids.append(sid)
            try:
                builder.add_step(
                    step_id=sid,
                    step_type=step.get("step_type", "auto_transition"),
                    name=step.get("name", sid),
                    description=step.get("description", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to add step '{sid}': {e}")

        # Sequential transitions
        for i in range(len(step_ids) - 1):
            try:
                builder.set_step_transition(step_ids[i], step_ids[i + 1])
            except Exception as e:
                logger.warning(f"Failed to set transition: {e}")

        # Set initial step
        if step_ids:
            try:
                builder.set_initial_step(step_ids[0])
            except Exception:
                pass

    def _assemble_agent(self, spec: dict[str, Any], builder: ArtifactBuilder) -> None:
        """Deterministic agent assembly from extracted spec."""
        builder.set_overview(
            name=spec.get("name", "Unnamed Agent"),
            description=spec.get("description", ""),
        )

        # Agent type already pre-seeded by classifier — don't overwrite

        for tool_spec in spec.get("tools", []):
            if not isinstance(tool_spec, dict):
                continue
            try:
                builder.add_tool(
                    name=tool_spec.get("name", "unnamed_tool"),
                    description=tool_spec.get("description", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to add tool: {e}")

    @staticmethod
    def _parse_extraction_response(response: str) -> dict[str, Any]:
        """Parse JSON from the extraction LLM response."""
        text = response.strip()
        if not text:
            return {}

        # Direct parse
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Find JSON in response
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse extraction response: {text[:200]}")
        return {}

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
            "  3. An Agent for tool-using AI\n\n"
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

        if self._artifact_type is None:
            return (
                "I'm not sure what type of artifact you want. "
                "Could you mention: FSM, Workflow, Agent, or Monitor?"
            )

        msg_count = len(self._messages)
        type_label = self._artifact_type.value.upper()
        article = "an" if type_label[0] in "AEIOU" else "a"
        if msg_count <= 1:
            return (
                f"Got it — I'll build {article} {type_label}. "
                f"Describe what it should do, or say 'build it' when ready."
            )
        return (
            f"Noted — added to your {type_label} spec "
            f"({msg_count} messages collected). "
            f"Say 'build it' when ready, or keep adding details."
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
    # Type detection (LLM classification)
    # ------------------------------------------------------------------

    def _get_type_classifier(self) -> Classifier:
        """Lazily build a Classifier for artifact type detection."""
        if self._type_classifier is None:
            schema = ClassificationSchema(
                intents=[
                    IntentDefinition(
                        name="fsm",
                        description=(
                            "A finite state machine, chatbot, dialogue system, "
                            "conversational flow, survey, quiz, FAQ bot, help desk, "
                            "interview, or onboarding flow"
                        ),
                    ),
                    IntentDefinition(
                        name="workflow",
                        description=(
                            "A multi-step workflow, data pipeline, automation, "
                            "ETL process, batch job, sequential process, or "
                            "async task orchestration"
                        ),
                    ),
                    IntentDefinition(
                        name="agent",
                        description=(
                            "An AI agent that uses tools, a ReAct agent, "
                            "plan-and-execute agent, research agent, browsing "
                            "agent, or any agentic pattern with tool use"
                        ),
                    ),
                ],
                fallback_intent="fsm",
                confidence_threshold=0.4,
            )
            self._type_classifier = Classifier(
                schema,
                model=self.meta_config.model,
                **self._api_kwargs,
            )
        return self._type_classifier

    def _get_agent_type_classifier(self) -> Classifier:
        """Lazily build a Classifier for agent pattern detection."""
        if self._agent_type_classifier is None:
            schema = ClassificationSchema(
                intents=[
                    IntentDefinition(
                        name="react",
                        description=(
                            "A ReAct agent: think-act-observe loop, tool-using "
                            "agent, search agent, general-purpose agent. "
                            "This is the default and most common pattern."
                        ),
                    ),
                    IntentDefinition(
                        name="plan_execute",
                        description=(
                            "A plan-and-execute agent: first creates a plan "
                            "then executes steps sequentially, with replanning"
                        ),
                    ),
                    IntentDefinition(
                        name="reflexion",
                        description=(
                            "A reflexion agent: attempts a task, reflects on "
                            "failures, retries with improved approach"
                        ),
                    ),
                    IntentDefinition(
                        name="rewoo",
                        description=(
                            "A REWOO agent: plans all tool calls upfront "
                            "then executes them sequentially without interleaving"
                        ),
                    ),
                    IntentDefinition(
                        name="evaluator_optimizer",
                        description=(
                            "An evaluator-optimizer agent: generates output, "
                            "evaluates quality, optimizes iteratively"
                        ),
                    ),
                    IntentDefinition(
                        name="maker_checker",
                        description=(
                            "A maker-checker agent: one agent drafts, "
                            "another reviews and approves or sends back"
                        ),
                    ),
                    IntentDefinition(
                        name="debate",
                        description=(
                            "A debate agent: multiple perspectives argue, "
                            "a judge synthesizes the best answer"
                        ),
                    ),
                    IntentDefinition(
                        name="orchestrator",
                        description=(
                            "An orchestrator agent: delegates subtasks to "
                            "specialized worker agents and synthesizes results"
                        ),
                    ),
                    IntentDefinition(
                        name="adapt",
                        description=(
                            "An ADaPT agent: estimates task complexity, "
                            "decomposes if too complex, adapts strategy"
                        ),
                    ),
                    IntentDefinition(
                        name="prompt_chain",
                        description=(
                            "A prompt chain agent: sequential prompts with "
                            "quality gates between each step"
                        ),
                    ),
                    IntentDefinition(
                        name="self_consistency",
                        description=(
                            "A self-consistency agent: generates multiple "
                            "samples and uses majority voting"
                        ),
                    ),
                ],
                fallback_intent="react",
                confidence_threshold=0.3,
            )
            self._agent_type_classifier = Classifier(
                schema,
                model=self.meta_config.model,
                **self._api_kwargs,
            )
        return self._agent_type_classifier

    def _detect_type(self, text: str) -> ArtifactType:
        """Classify the artifact type from user text using LLM classification."""
        normalized = text.strip().lower()
        if normalized in self._JUST_BUILD_PHRASES:
            return ArtifactType.FSM

        try:
            classifier = self._get_type_classifier()
            result = classifier.classify(text)
            logger.debug(
                f"Type classification: intent={result.intent}, "
                f"confidence={result.confidence:.2f}"
            )
            return ArtifactType(result.intent)
        except Exception as e:
            logger.warning(f"Type classification failed, using fallback: {e}")
            return self._detect_type_fallback(text)

    @classmethod
    def _detect_type_fallback(cls, text: str) -> ArtifactType:
        """Keyword-based fallback when LLM classification is unavailable."""
        aliases = cls._build_type_aliases()
        normalized = text.strip().lower()
        for alias, type_str in aliases.items():
            if alias in normalized:
                try:
                    return ArtifactType(type_str)
                except ValueError:
                    pass
        return ArtifactType.FSM

    def _preseed_agent_type(self, task: str, builder: ArtifactBuilder) -> None:
        """Classify agent pattern type from task and pre-set it on the builder."""
        if not isinstance(builder, AgentBuilder):
            return

        try:
            classifier = self._get_agent_type_classifier()
            result = classifier.classify(task)
            logger.debug(
                f"Agent type classification: intent={result.intent}, "
                f"confidence={result.confidence:.2f}"
            )
            builder.set_agent_type(result.intent)
        except Exception as e:
            logger.warning(f"Agent type classification failed: {e}")
            # Keyword fallback
            normalized = task.strip().lower()
            for agent_type in AgentBuilder.VALID_AGENT_TYPES:
                pattern = agent_type.replace("_", " ")
                if agent_type in normalized or pattern in normalized:
                    try:
                        builder.set_agent_type(agent_type)
                    except Exception:
                        pass
                    return

    # ------------------------------------------------------------------
    # Builder creation + result
    # ------------------------------------------------------------------

    def _create_builder(
        self, artifact_type: ArtifactType
    ) -> FSMBuilder | WorkflowBuilder | AgentBuilder:
        if artifact_type == ArtifactType.FSM:
            return FSMBuilder()
        if artifact_type == ArtifactType.WORKFLOW:
            return WorkflowBuilder()
        if artifact_type == ArtifactType.AGENT:
            return AgentBuilder()
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
