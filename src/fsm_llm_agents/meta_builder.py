from __future__ import annotations

"""
MetaBuilderAgent — hybrid architecture: LLM collects, software builds.

Uses a 4-state extraction-driven FSM:
  CLASSIFY → COLLECT → CONFIRM ↔ (self-loop on revise) → OUTPUT

The LLM collects data piece by piece via ``field_extractions`` and
``classification_extractions``. Software assembles the artifact
deterministically using builder APIs on CONFIRM entry.

Supports two interfaces:
  - ``run(task)`` for single-shot programmatic use (like other agents)
  - ``start()/send()/is_complete()/get_result()`` for turn-by-turn
    interactive use (CLI, monitor endpoints)
"""

import json
import time
from typing import Any, ClassVar

import litellm

from fsm_llm import API
from fsm_llm.logging import logger
from fsm_llm.utilities import extract_json_from_text

from .base import BaseAgent
from .constants import (
    MetaBuilderStates,
    MetaDefaults,
    MetaErrorMessages,
    MetaLogMessages,
)
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
from .meta_fsm import build_meta_builder_fsm
from .meta_output import format_artifact_json
from .meta_prompts import build_review_presentation


class MetaBuilderAgent(BaseAgent):
    """Meta-agent that builds FSMs, Workflows, Agents, and Monitors.

    Hybrid architecture: the LLM collects data (type, name, components)
    via the FSM extraction pipeline, then software assembles the artifact
    deterministically using builder APIs.

    Usage (turn-by-turn)::

        agent = MetaBuilderAgent()
        response = agent.start()
        print(response)

        while not agent.is_complete():
            user_input = input("> ")
            response = agent.send(user_input)
            print(response)

        result = agent.get_result()
        print(result.artifact_json)

    Usage (single-shot)::

        agent = MetaBuilderAgent()
        result = agent.run("Build a customer support chatbot with 3 states")
        print(result.artifact_json)
    """

    # Phrases that mean "stop asking, just build with defaults"
    _JUST_BUILD_PHRASES: frozenset[str] = frozenset(
        {
            "just build it",
            "just build",
            "just fill it",
            "fill it up",
            "fill it in",
            "dont care",
            "don't care",
            "whatever",
            "anything",
            "surprise me",
            "random",
            "just do it",
            "just make it",
            "just create it",
            "build something",
            "make something",
            "i dont care",
            "i don't care",
            "doesnt matter",
            "doesn't matter",
        }
    )

    # Sorted longest-first so multi-word phrases match before single words.
    _TYPE_ALIASES: ClassVar[dict[str, str]] = {}

    @classmethod
    def _build_type_aliases(cls) -> dict[str, str]:
        """Return type alias map sorted longest-first for correct matching."""
        raw = {
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

    def __init__(
        self,
        config: MetaBuilderConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        if config is None:
            config = MetaBuilderConfig()
        super().__init__(config=config, **api_kwargs)
        self.meta_config: MetaBuilderConfig = config

        # FSM state
        self._api: API | None = None
        self._conv_id: str | None = None
        self._started: bool = False

        # Builder state (shared with handlers via closures)
        self._artifact_type: ArtifactType | None = None
        self._builder: ArtifactBuilder | None = None
        self._requirements: dict[str, Any] = {}
        self._build_errors: list[str] = []
        self._result: MetaBuilderResult | None = None
        self._turn_count: int = 0

    # ------------------------------------------------------------------
    # BaseAgent abstract method implementations
    # ------------------------------------------------------------------

    def _register_handlers(self, api: API) -> None:
        """Register POST_TRANSITION handlers for assembly, revision, output."""
        # Handler: entry to CONFIRM (assemble artifact or apply revision)
        api.register_handler(
            api.create_handler("MetaAssembleOrRevise")
            .with_priority(50)
            .on_state_entry(MetaBuilderStates.CONFIRM)
            .do(self._confirm_entry_handler)
        )

        # Handler: OUTPUT entry (finalize result)
        api.register_handler(
            api.create_handler("MetaFinalizeOutput")
            .with_priority(50)
            .on_state_entry(MetaBuilderStates.OUTPUT)
            .do(self._finalize_handler)
        )

    _MAX_RUN_TURNS: int = 10

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the meta-builder in single-shot mode.

        Drives the multi-state FSM internally:
        CLASSIFY (extract type/name) → COLLECT (extract components) →
        CONFIRM (assemble + auto-approve) → OUTPUT
        """
        self.start(initial_message=task)

        if self.is_complete():
            return self.get_result()

        # Drive the FSM through remaining states automatically
        for _ in range(self._MAX_RUN_TURNS):
            if self.is_complete():
                return self.get_result()

            current = self._get_current_state()
            if current == MetaBuilderStates.CONFIRM:
                self.send("approve")
            elif current == MetaBuilderStates.OUTPUT:
                break
            else:
                # CLASSIFY or COLLECT — resend the task for extraction
                self.send(task)

        if self.is_complete():
            return self.get_result()

        # Build partial result from whatever state we reached
        self._build_result()
        return self._result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Turn-by-turn interface
    # ------------------------------------------------------------------

    def start(self, initial_message: str = "") -> str:
        """Initialize the conversation. Returns the first agent response."""
        if self._started:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_ALREADY_STARTED)

        self._started = True
        logger.info(MetaLogMessages.META_STARTED.format(model=self.meta_config.model))

        # Build the FSM and create the API
        fsm_def = build_meta_builder_fsm()
        self._api = API.from_definition(
            fsm_def,
            model=self.meta_config.model,
            temperature=self.meta_config.temperature,
            max_tokens=self.meta_config.max_tokens,
            **{
                k: v
                for k, v in self._api_kwargs.items()
                if k not in {"model", "temperature", "max_tokens"}
            },
        )
        self._register_handlers(self._api)

        # Start the conversation
        self._conv_id, welcome = self._api.start_conversation()

        if initial_message:
            # Pre-resolve artifact type deterministically
            self._pre_resolve_type(initial_message)
            response = self._api.converse(initial_message, self._conv_id)
            self._turn_count += 1
            return response

        return welcome

    def send(self, message: str) -> str:
        """Send a user message and get the agent's response."""
        if not self._started:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_NOT_STARTED)
        if self.is_complete():
            raise MetaBuilderError("Conversation has already completed")

        self._turn_count += 1
        if self._turn_count > self.meta_config.max_turns:
            raise MetaBuilderError(
                f"Maximum turns ({self.meta_config.max_turns}) exceeded"
            )

        # Pre-resolve artifact type for CLASSIFY state
        if self._get_current_state() == MetaBuilderStates.CLASSIFY:
            self._pre_resolve_type(message)

        response = self._api.converse(message, self._conv_id)  # type: ignore[union-attr]
        return response

    def is_complete(self) -> bool:
        """Whether the artifact has been fully built and approved."""
        if self._api is None or self._conv_id is None:
            return False
        return self._api.has_conversation_ended(self._conv_id)

    def get_result(self) -> MetaBuilderResult:
        """Get the final build result."""
        if not self.is_complete():
            raise MetaBuilderError(
                "Build is not complete. Continue the conversation until the "
                "user approves the artifact."
            )
        if self._result is None:
            self._build_result()
        return self._result  # type: ignore[return-value]

    def get_internal_state(self) -> dict[str, Any]:
        """Get the current internal state for debugging/monitoring."""
        result: dict[str, Any] = {
            "phase": self._get_current_state(),
            "turn_count": self._turn_count,
            "is_complete": self.is_complete(),
            "started": self._started,
        }

        if self._artifact_type is not None:
            result["artifact_type"] = self._artifact_type.value

        if self._requirements:
            result["requirements"] = self._requirements

        result["build_errors"] = list(self._build_errors)

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
        """Run in interactive mode, reading from stdin."""
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

        # Build partial result if interrupted
        self._build_result()
        return self._result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # POST_TRANSITION handlers
    # ------------------------------------------------------------------

    def _confirm_entry_handler(self, context: dict[str, Any]) -> dict[str, Any]:
        """Handler: fires on CONFIRM entry. Assembles or revises artifact.

        On first entry (from COLLECT): creates builder, assembles artifact
        from extracted context data, stores summary for display.

        On subsequent entries (CONFIRM self-loop): applies revision.
        """
        if self._builder is None:
            return self._assemble_artifact(context)
        return self._do_revision(context)

    def _finalize_handler(self, context: dict[str, Any]) -> dict[str, Any]:
        """Handler: fires on OUTPUT entry. Finalizes the result."""
        self._build_result()
        artifact_json = self._result.artifact_json if self._result else "{}"
        return {"artifact_json": artifact_json}

    # ------------------------------------------------------------------
    # Assembly: deterministic build from extracted context
    # ------------------------------------------------------------------

    def _assemble_artifact(self, context: dict[str, Any]) -> dict[str, Any]:
        """Assemble artifact deterministically from extracted context data.

        Reads extracted fields (artifact_type, artifact_name,
        artifact_description, component_names, component_flow) and
        populates the builder using direct API calls. No monolithic
        LLM JSON generation.
        """
        # Resolve artifact type
        raw_type = context.get("artifact_type", "fsm")
        artifact_type = self._resolve_artifact_type(raw_type)
        if artifact_type is None:
            artifact_type = ArtifactType.FSM
        self._artifact_type = artifact_type

        # Gather extracted requirements
        self._requirements = {
            "artifact_name": context.get("artifact_name"),
            "artifact_description": context.get("artifact_description"),
            "artifact_persona": context.get("artifact_persona"),
        }
        user_request = context.get("user_request", "")
        if not self._requirements.get("artifact_description") and user_request:
            self._requirements["artifact_description"] = str(user_request)[:300]
        if not self._requirements.get("artifact_name"):
            desc = self._requirements.get("artifact_description", "")
            if desc:
                self._requirements["artifact_name"] = self._generate_name(str(desc))
            else:
                self._requirements["artifact_name"] = (
                    f"Sample_{artifact_type.value.upper()}"
                )

        # Create builder and set overview
        self._builder = self._create_builder(artifact_type)
        name = str(self._requirements.get("artifact_name") or "Untitled")
        desc = str(self._requirements.get("artifact_description") or "")
        persona = str(self._requirements.get("artifact_persona") or "")

        logger.info(
            MetaLogMessages.BUILD_STARTED.format(artifact_type=artifact_type.value)
        )

        # Get component names from extraction
        component_names = context.get("component_names") or []
        if isinstance(component_names, str):
            # Fallback: LLM returned comma-separated string instead of list
            component_names = [
                c.strip() for c in component_names.split(",") if c.strip()
            ]
        if not isinstance(component_names, list):
            component_names = []
        # Clean: deduplicate, filter empty
        seen: set[str] = set()
        clean_names: list[str] = []
        for cn in component_names:
            cn_str = str(cn).strip().lower().replace(" ", "_")
            if cn_str and cn_str not in seen:
                seen.add(cn_str)
                clean_names.append(cn_str)
        component_names = clean_names

        component_flow = str(context.get("component_flow", ""))

        # Assemble based on artifact type
        if isinstance(self._builder, FSMBuilder):
            self._assemble_fsm(self._builder, name, desc, persona, component_names)
        elif isinstance(self._builder, WorkflowBuilder):
            self._assemble_workflow(self._builder, name, desc, component_names)
        elif isinstance(self._builder, AgentBuilder):
            self._assemble_agent(
                self._builder, name, desc, component_names, component_flow
            )
        elif isinstance(self._builder, MonitorBuilder):
            self._assemble_monitor(self._builder, name, desc, component_names)

        # Generate transitions/connections via targeted LLM calls
        if isinstance(self._builder, FSMBuilder) and len(component_names) > 1:
            self._generate_fsm_transitions(self._builder, component_names, desc)

        # Present result
        presentation = build_review_presentation(self._builder, artifact_type)
        return {
            "builder_summary": presentation,
            "validation_status": (
                "passed" if not self._builder.validate_complete() else "has errors"
            ),
        }

    def _assemble_fsm(
        self,
        builder: FSMBuilder,
        name: str,
        desc: str,
        persona: str,
        components: list[str],
    ) -> None:
        """Populate FSM builder from extracted component names."""
        builder.set_overview(name=name, description=desc, persona=persona)

        if not components:
            components = ["start"]

        for cname in components:
            builder.add_state(
                state_id=cname,
                description=f"{cname.replace('_', ' ').title()} state",
                purpose=f"Handle the {cname.replace('_', ' ')} phase",
                extraction_instructions=f"Extract relevant data for {cname}",
                response_instructions=f"Respond appropriately for the {cname} state",
            )

        builder.set_initial_state(components[0])

    def _assemble_workflow(
        self,
        builder: WorkflowBuilder,
        name: str,
        desc: str,
        components: list[str],
    ) -> None:
        """Populate workflow builder from extracted step names."""
        wf_id = name.lower().replace(" ", "_")[:40]
        builder.set_overview(workflow_id=wf_id, name=name, description=desc)

        if not components:
            components = ["step_1"]

        for sname in components:
            builder.add_step(
                step_id=sname,
                step_type="auto_transition",
                name=sname.replace("_", " ").title(),
                description=f"Process the {sname.replace('_', ' ')} step",
            )

        # Sequential transitions (correct ordering by construction)
        for i in range(len(components) - 1):
            builder.set_step_transition(components[i], components[i + 1])

        builder.set_initial_step(components[0])

    def _assemble_agent(
        self,
        builder: AgentBuilder,
        name: str,
        desc: str,
        components: list[str],
        flow: str,
    ) -> None:
        """Populate agent builder from extracted tool names."""
        builder.set_overview(name=name, description=desc)
        builder.set_agent_type("react")

        if not components:
            components = ["search"]

        for tname in components:
            builder.add_tool(
                name=tname,
                description=f"{tname.replace('_', ' ').title()} tool",
            )

    def _assemble_monitor(
        self,
        builder: MonitorBuilder,
        name: str,
        desc: str,
        components: list[str],
    ) -> None:
        """Populate monitor builder from extracted panel names."""
        builder.set_overview(name=name, description=desc)

        if not components:
            components = ["status"]

        for pname in components:
            builder.add_panel(
                panel_id=pname,
                title=pname.replace("_", " ").title(),
                panel_type="metric",
                metric=pname,
                description=f"Monitor {pname.replace('_', ' ')}",
            )

    def _generate_fsm_transitions(
        self,
        builder: FSMBuilder,
        components: list[str],
        description: str,
    ) -> None:
        """Generate FSM transitions using a targeted LLM call.

        Instead of generating the full artifact as JSON, we ask one
        focused question: given these states, what transitions should exist?
        """
        states_str = ", ".join(components)
        prompt = (
            f"Given an FSM called '{builder.name}' ({description}) "
            f"with these states: {states_str}\n\n"
            f"List the transitions as JSON: "
            f'[{{"from": "state_a", "to": "state_b", '
            f'"description": "when to transition"}}]\n'
            f"Include only the most important transitions. "
            f"The last state should be terminal (no outgoing transitions). "
            f"Respond with ONLY the JSON array."
        )

        result = self._call_llm_json(
            system_prompt="You output JSON arrays of transitions. Nothing else.",
            user_message=prompt,
            temperature=MetaDefaults.BUILD_TEMPERATURE,
            parse_array=True,
        )

        transitions_applied = False
        if isinstance(result, list):
            for t in result:
                if not isinstance(t, dict):
                    continue
                src = str(t.get("from", t.get("source", "")))
                tgt = str(t.get("to", t.get("target", "")))
                t_desc = str(t.get("description", ""))
                if src in builder.states and tgt in builder.states:
                    try:
                        builder.add_transition(src, tgt, t_desc)
                        transitions_applied = True
                    except Exception:
                        pass

        # Fallback: sequential transitions if LLM failed
        if not transitions_applied and len(components) > 1:
            for i in range(len(components) - 1):
                builder.add_transition(
                    components[i],
                    components[i + 1],
                    f"Proceed to {components[i + 1]}",
                )

    # ------------------------------------------------------------------
    # Revision logic (uses targeted LLM call)
    # ------------------------------------------------------------------

    def _do_revision(self, context: dict[str, Any]) -> dict[str, Any]:
        """Apply a revision to the existing artifact."""
        if self._builder is None or self._artifact_type is None:
            logger.warning("Revision requested but no builder/type exists")
            return {}

        revision_request = context.get(
            "revision_request", context.get("user_request", "")
        )
        if not revision_request:
            return {
                "builder_summary": build_review_presentation(
                    self._builder, self._artifact_type
                )
            }

        logger.info(
            MetaLogMessages.REVISION_STARTED.format(revision=str(revision_request)[:80])
        )

        # Use a targeted LLM call for revision spec
        current_spec = json.dumps(self._builder.to_dict(), indent=2)
        prompt = (
            f"Current artifact:\n{current_spec}\n\n"
            f"User wants these changes: {revision_request}\n\n"
            f"Output the COMPLETE updated JSON. Respond with ONLY JSON."
        )

        spec = self._call_llm_json(
            system_prompt=(
                "You update artifact specs. Output ONLY valid JSON. "
                "Apply the requested changes to the current spec."
            ),
            user_message=prompt,
            temperature=MetaDefaults.BUILD_TEMPERATURE,
        )

        if spec and isinstance(spec, dict):
            new_builder = self._create_builder(self._artifact_type)
            old_builder = self._builder
            self._builder = new_builder
            try:
                self._apply_spec_to_builder(spec)
            except Exception as e:
                logger.error(f"Revision apply failed, restoring previous: {e}")
                self._builder = old_builder
        else:
            logger.warning("Revision returned empty spec, keeping current artifact")

        presentation = build_review_presentation(self._builder, self._artifact_type)
        return {
            "builder_summary": presentation,
            "validation_status": (
                "passed" if not self._builder.validate_complete() else "has errors"
            ),
        }

    # ------------------------------------------------------------------
    # Pre-resolution (deterministic, before LLM classification)
    # ------------------------------------------------------------------

    def _pre_resolve_type(self, message: str) -> None:
        """Pre-resolve artifact type deterministically from message text."""
        if self._api is None or self._conv_id is None:
            return

        normalized = message.strip().lower()
        if self._is_just_build_request(normalized):
            self._api.update_context(
                self._conv_id,
                {
                    "artifact_type": "fsm",
                    "artifact_description": "A sample FSM artifact",
                    "artifact_name": "Sample_FSM",
                    "component_names": ["greeting", "help", "goodbye"],
                },
            )
            return

        aliases = self._build_type_aliases()
        for alias, type_str in aliases.items():
            if alias in normalized:
                self._api.update_context(self._conv_id, {"artifact_type": type_str})
                return

    @staticmethod
    def _is_just_build_request(normalized: str) -> bool:
        """Check if the user wants us to just build with defaults."""
        phrases = MetaBuilderAgent._JUST_BUILD_PHRASES
        if normalized in phrases:
            return True
        for phrase in phrases:
            if len(phrase) > 4 and phrase in normalized:
                return True
        return False

    # ------------------------------------------------------------------
    # Spec application (used by revision only)
    # ------------------------------------------------------------------

    def _apply_spec_to_builder(self, spec: dict[str, Any]) -> None:
        """Apply a JSON spec to the builder (used for revisions)."""
        builder = self._builder
        if builder is None:
            return

        if isinstance(builder, FSMBuilder):
            self._apply_fsm_spec(builder, spec)
        elif isinstance(builder, WorkflowBuilder):
            self._apply_workflow_spec(builder, spec)
        elif isinstance(builder, AgentBuilder):
            self._apply_agent_spec(builder, spec)
        elif isinstance(builder, MonitorBuilder):
            self._apply_monitor_spec(builder, spec)

    def _apply_fsm_spec(self, builder: FSMBuilder, spec: dict[str, Any]) -> None:
        """Apply an FSM spec to the builder."""
        name = spec.get("name", self._requirements.get("artifact_name", ""))
        desc = spec.get(
            "description", self._requirements.get("artifact_description", "")
        )
        persona = spec.get("persona", self._requirements.get("artifact_persona", ""))
        builder.set_overview(
            name=str(name), description=str(desc), persona=str(persona)
        )

        states_raw = spec.get("states", [])
        state_items: list[dict[str, Any]] = []
        if isinstance(states_raw, list):
            state_items = [s for s in states_raw if isinstance(s, dict)]
        elif isinstance(states_raw, dict):
            for sid, sdata in states_raw.items():
                if isinstance(sdata, dict):
                    entry = dict(sdata)
                    if "id" not in entry:
                        entry["id"] = sid
                    state_items.append(entry)

        for s in state_items:
            state_id = str(s.get("id", s.get("state_id", "")))
            if not state_id:
                continue
            try:
                builder.add_state(
                    state_id=state_id,
                    description=str(s.get("description", "")),
                    purpose=str(s.get("purpose", s.get("description", ""))),
                    extraction_instructions=str(s.get("extraction_instructions", "")),
                    response_instructions=str(s.get("response_instructions", "")),
                )
            except Exception as e:
                logger.warning(f"Failed to add state '{state_id}': {e}")

        initial = spec.get("initial_state")
        if initial and str(initial) in builder.states:
            builder.set_initial_state(str(initial))

        # Transitions — handle both top-level and per-state
        transitions: list[dict[str, Any]] = []
        for t in spec.get("transitions", []):
            if isinstance(t, dict):
                transitions.append(t)
        for s in state_items:
            state_id = str(s.get("id", s.get("state_id", "")))
            for t in s.get("transitions", []):
                if not isinstance(t, dict):
                    continue
                target = (
                    t.get("target_state")
                    or t.get("target")
                    or t.get("to")
                    or t.get("to_state")
                )
                if target:
                    transitions.append(
                        {
                            "source": state_id,
                            "target": str(target),
                            "description": t.get("description", ""),
                        }
                    )

        for t in transitions:
            src = str(t.get("source", t.get("from", t.get("from_state", ""))))
            tgt = str(t.get("target", t.get("to", t.get("target_state", ""))))
            if src and tgt and src in builder.states and tgt in builder.states:
                try:
                    builder.add_transition(src, tgt, str(t.get("description", "")))
                except Exception as e:
                    logger.warning(f"Failed to add transition: {e}")

    def _apply_workflow_spec(
        self, builder: WorkflowBuilder, spec: dict[str, Any]
    ) -> None:
        """Apply a workflow spec to the builder."""
        wf_id = spec.get("workflow_id") or "workflow_1"
        name = spec.get("name", self._requirements.get("artifact_name", ""))
        desc = spec.get(
            "description", self._requirements.get("artifact_description", "")
        )
        builder.set_overview(
            workflow_id=str(wf_id), name=str(name), description=str(desc)
        )

        # Two-pass: add all steps first, THEN set transitions
        steps = spec.get("steps", [])
        step_transitions: list[tuple[str, str]] = []
        if isinstance(steps, list):
            for s in steps:
                if not isinstance(s, dict):
                    continue
                step_id = str(s.get("id", s.get("step_id", "")))
                if not step_id:
                    continue
                builder.add_step(
                    step_id=step_id,
                    step_type=str(s.get("step_type", "auto_transition")),
                    name=str(s.get("name", step_id)),
                    description=str(s.get("description", "")),
                )
                next_step = s.get("next_step")
                if next_step:
                    step_transitions.append((step_id, str(next_step)))

            for from_step, to_step in step_transitions:
                if from_step in builder.steps and to_step in builder.steps:
                    try:
                        builder.set_step_transition(from_step, to_step)
                    except Exception as e:
                        logger.warning(f"Failed to set step transition: {e}")

        initial = spec.get("initial_step_id")
        if initial and str(initial) in builder.steps:
            builder.set_initial_step(str(initial))

    def _apply_agent_spec(self, builder: AgentBuilder, spec: dict[str, Any]) -> None:
        """Apply an agent spec to the builder."""
        name = spec.get("name", self._requirements.get("artifact_name", ""))
        desc = spec.get(
            "description", self._requirements.get("artifact_description", "")
        )
        builder.set_overview(name=str(name), description=str(desc))
        builder.set_agent_type(str(spec.get("agent_type", "react")))

        for t in spec.get("tools", []):
            if isinstance(t, dict) and t.get("name"):
                builder.add_tool(
                    name=str(t["name"]),
                    description=str(t.get("description", "")),
                )

        config = spec.get("config")
        if isinstance(config, dict):
            builder.set_config(**config)

    def _apply_monitor_spec(
        self, builder: MonitorBuilder, spec: dict[str, Any]
    ) -> None:
        """Apply a monitor spec to the builder."""
        name = spec.get("name", self._requirements.get("artifact_name", ""))
        desc = spec.get(
            "description", self._requirements.get("artifact_description", "")
        )
        builder.set_overview(name=str(name), description=str(desc))

        for p in spec.get("panels", []):
            if isinstance(p, dict):
                pid = str(p.get("id", p.get("panel_id", "")))
                if pid:
                    builder.add_panel(
                        panel_id=pid,
                        title=str(p.get("title", pid)),
                        panel_type=str(p.get("panel_type", "metric")),
                        metric=str(p.get("metric", "")),
                        description=str(p.get("description", "")),
                    )

        for a in spec.get("alerts", []):
            if isinstance(a, dict):
                aid = str(a.get("id", a.get("alert_id", "")))
                if aid:
                    builder.add_alert(
                        alert_id=aid,
                        metric=str(a.get("metric", "")),
                        condition=str(a.get("condition", ">")),
                        threshold=float(a.get("threshold", 0)),
                        description=str(a.get("description", "")),
                    )

        config = spec.get("config")
        if isinstance(config, dict):
            builder.set_config(**config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_state(self) -> str:
        """Get the current FSM state."""
        if self._api is None or self._conv_id is None:
            return MetaBuilderStates.CLASSIFY
        return self._api.get_current_state(self._conv_id)

    def _resolve_artifact_type(self, raw: Any) -> ArtifactType | None:
        """Resolve a raw artifact type string to an ArtifactType enum."""
        if raw is None:
            return self._artifact_type
        if isinstance(raw, ArtifactType):
            return raw
        if not isinstance(raw, str):
            return None

        normalized = raw.strip().lower()
        aliases = self._build_type_aliases()
        normalized = aliases.get(normalized, normalized)
        try:
            return ArtifactType(normalized)
        except ValueError:
            return None

    _MAX_LLM_RETRIES: int = 2

    def _call_llm_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float | None = None,
        parse_array: bool = False,
    ) -> dict[str, Any] | list[Any]:
        """Call the LLM and parse as JSON. Used for targeted calls only.

        Returns empty dict/list on failure (never raises).
        """
        model = self.meta_config.model
        temp = temperature if temperature is not None else self.meta_config.temperature
        reserved = {"model", "messages", "temperature", "max_tokens"}
        safe_kwargs = {k: v for k, v in self._api_kwargs.items() if k not in reserved}
        empty: dict[str, Any] | list[Any] = [] if parse_array else {}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        last_error = ""
        for attempt in range(1, self._MAX_LLM_RETRIES + 1):
            try:
                t0 = time.time()
                retry_temp = temp + (0.1 * (attempt - 1))
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=min(retry_temp, 2.0),
                    max_tokens=self.meta_config.max_tokens,
                    **safe_kwargs,
                )
                dt = time.time() - t0
                logger.debug(
                    f"Meta-agent LLM call completed in {dt:.2f}s "
                    f"(attempt {attempt}/{self._MAX_LLM_RETRIES})"
                )

                content = response.choices[0].message.content
                if not content:
                    last_error = "LLM returned empty content"
                    logger.warning(f"{last_error} (attempt {attempt})")
                    continue

                text = content.strip()

                # Try direct JSON parse
                start_char = "[" if parse_array else "{"
                if text.startswith(start_char):
                    try:
                        parsed = json.loads(text)
                        expected = list if parse_array else dict
                        if isinstance(parsed, expected):
                            return parsed
                    except json.JSONDecodeError:
                        pass

                # Try extraction from text
                data = extract_json_from_text(text)
                if parse_array and isinstance(data, list):
                    return data
                if not parse_array and isinstance(data, dict):
                    return data

                last_error = f"Could not parse LLM response as JSON: {text[:200]}"
                logger.warning(f"{last_error} (attempt {attempt})")

            except Exception as e:
                last_error = str(e)
                logger.error(f"Meta-agent LLM call failed: {e} (attempt {attempt})")

        logger.error(f"All {self._MAX_LLM_RETRIES} LLM attempts failed: {last_error}")
        return empty

    def _create_builder(
        self, artifact_type: ArtifactType
    ) -> FSMBuilder | WorkflowBuilder | AgentBuilder | MonitorBuilder:
        """Create the appropriate builder for the artifact type."""
        if artifact_type == ArtifactType.FSM:
            return FSMBuilder()
        if artifact_type == ArtifactType.WORKFLOW:
            return WorkflowBuilder()
        if artifact_type == ArtifactType.AGENT:
            return AgentBuilder()
        if artifact_type == ArtifactType.MONITOR:
            return MonitorBuilder()
        raise MetaBuilderError(f"Unknown artifact type: {artifact_type}")

    @staticmethod
    def _generate_name(description: str) -> str:
        """Generate a short artifact name from a description."""
        stop = {
            "a",
            "an",
            "the",
            "for",
            "and",
            "or",
            "to",
            "is",
            "that",
            "it",
            "of",
        }
        words = [w for w in description.split() if w.lower() not in stop]
        name_words = words[:3] if words else ["Untitled"]
        return "_".join(w.capitalize() for w in name_words)

    def _build_result(self) -> None:
        """Build the MetaBuilderResult from current builder state."""
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
