"""Presets replace the SOP dicts (plan a555dcfe step 22, D-033; roadmap E4.5).

A preset is a ``pattern`` document whose body ``extends`` another pattern: ``resolve_extends`` makes
the flat document, ``load_executable(bases=)`` checks it, ``contract.input_templates`` render the
caller's arguments into an input value (``render_inputs``), the three SOPs of ``agents/sop.py`` are
now files served by id, and the SOP module is gone (clean break). The old ``task_template`` texts
below are the ones of the deleted ``BUILTIN_SOPS``.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any, ClassVar

import pytest

from fsm_llm.agents import AgentConfig, DebateAgent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import (
    Contract,
    ExecutableError,
    PatternEnvelope,
    parse_executable,
)
from fsm_llm.core.definitions.executable_instance import render_inputs
from fsm_llm.core.definitions.executable_preset import (
    MAX_EXTENDS_DEPTH,
    PresetError,
    resolve_extends,
)
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.namespaces import namespace_resolver
from fsm_llm.testing import ScriptedLLM
from fsm_llm.tools import ToolRegistry
from tests import pattern_runs as runs

#: ``task_template`` and ``config_overrides`` of the deleted ``BUILTIN_SOPS`` (``str.format`` text).
OLD_SOPS: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "code-review": (
        "Review the following {language} code for bugs, style issues, "
        "and potential improvements. Provide specific, actionable feedback.\n\n"
        "```{language}\n{code}\n```",
        {"language": "python", "code": "def f(x):\n    return x  # {braces}\n"},
        {"temperature": 0.3, "max_iterations": 5},
    ),
    "summarize": (
        "Summarize the following text into {num_points} key points. "
        "Be concise and focus on the most important information.\n\n"
        "{text}",
        {"num_points": 3, "text": "Cats sleep. Dogs bark."},
        {"temperature": 0.3, "max_iterations": 3},
    ),
    "data-extraction": (
        "Extract the following fields from the text: {fields}\n\n"
        "Text:\n{text}\n\n"
        "Return the extracted data as a structured response.",
        {"fields": "name, age", "text": "Ann is 31."},
        {"temperature": 0.1, "max_iterations": 3},
    ),
}


def _registry() -> ExecutableRegistry:
    registry = ExecutableRegistry()
    registry.load_sources()
    return registry


def _preset(doc_id: str = "p", extends: str = "react", **body: Any) -> dict[str, Any]:
    return {
        "schema": "fsm-llm/executable",
        "kind": "pattern",
        "id": doc_id,
        "body": {"extends": extends, **body},
    }


def _added(**properties: Any) -> dict[str, Any]:
    """``options_schema`` of declared option names to types (``temperature=\"number\"``)."""
    return {
        "options_schema": {
            "type": "object",
            "properties": {n: {"type": t} for n, t in properties.items()},
        }
    }


def _load(document: dict[str, Any], **kwargs: Any):
    kwargs.setdefault("bases", lambda name: pf._as_written(name))
    return load_executable(document, resolver=namespace_resolver(), **kwargs)


def _chain(length: int, *, cycle: bool = False) -> dict[str, PatternEnvelope]:
    """``p0`` extends ``p1`` ... the last extends ``react`` (or ``p0`` for a cycle)."""
    docs = {}
    for index in range(length):
        last = index == length - 1
        base = "p0" if cycle and last else "react" if last else f"p{index + 1}"
        docs[f"p{index}"] = parse_executable(_preset(f"p{index}", base))
    docs["react"] = parse_executable(pf.load_document("react"))
    return docs  # type: ignore[return-value]


def _ids(registry: ExecutableRegistry) -> set[str]:
    return {e.id for e in registry.list(kind="pattern")}


class TestTheRenderer:
    @pytest.mark.parametrize("name", OLD_SOPS)
    def test_a_built_in_preset_renders_the_task_text_of_its_sop(self, name):
        template, arguments, _ = OLD_SOPS[name]
        contract = pf.load_envelope(name).contract

        rendered = render_inputs(contract, arguments)

        assert rendered == {"task": template.format(**arguments)}

    def test_an_input_given_directly_is_used_as_it_is(self):
        contract = pf.load_envelope("summarize").contract

        assert render_inputs(contract, {"task": "mine"}) == {"task": "mine"}
        assert render_inputs(contract, {"task": "mine", "text": "t"}) == {
            "task": "mine"
        }

    def test_no_template_returns_the_inputs_object_itself(self):
        inputs = {"task": "x"}

        assert render_inputs(pf.load_envelope("react").contract, inputs) is inputs

    def test_a_value_is_one_pass_and_never_rescanned(self):
        contract = pf.load_envelope("summarize").contract

        rendered = render_inputs(contract, {"num_points": 2, "text": "{{num_points}}"})

        assert rendered["task"].endswith("\n\n{{num_points}}")

    @pytest.mark.parametrize("key", ["_approval_granted", "system_x", "__y", "other"])
    def test_an_unknown_or_internal_prefixed_argument_is_refused(self, key):
        contract = pf.load_envelope("summarize").contract

        with pytest.raises(ValueError, match="neither inputs nor placeholders") as info:
            render_inputs(contract, {"num_points": 1, "text": "t", key: True})

        assert key in str(info.value)

    def test_a_missing_or_non_scalar_argument_is_named_never_its_value(self):
        contract = pf.load_envelope("summarize").contract

        with pytest.raises(ValueError, match=r"\['num_points'\]"):
            render_inputs(contract, {"text": "t"})
        with pytest.raises(ValueError, match=r"\['text'\]") as info:
            render_inputs(contract, {"num_points": 1, "text": {"secret": "s3cr3t"}})
        assert "s3cr3t" not in str(info.value)

    def test_the_optional_inputs_of_an_executable_pass_through(self):
        contract = Contract(inputs=["task"], input_templates={"task": "Do {{x}}"})

        out = render_inputs(
            contract,
            {"x": "it", "initial_context": {"k": 1}},
            optional=["initial_context"],
        )

        assert out == {"initial_context": {"k": 1}, "task": "Do it"}

    def test_the_arguments_are_consumed_and_reach_no_input(self):
        contract = pf.load_envelope("data-extraction").contract

        assert set(render_inputs(contract, OLD_SOPS["data-extraction"][1])) == {"task"}


def _context(model: Any, **callables: Any) -> RunContext:
    from fsm_llm.core.names import NameRegistry

    named: NameRegistry[Any] = NameRegistry("callables")
    for key, value in {"tools": runs._tools(), **callables}.items():
        named.register(key, value)
    return RunContext(llm_interface=model, callables=named)


@pytest.fixture
def tasks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The task each attempt of a ``PatternAgent`` ran on (the real loop still runs)."""
    seen: list[str] = []
    original = PatternAgent._run_loop

    def spy(self, task, initial_context, handlers):
        seen.append(task)
        return original(self, task, initial_context, handlers)

    monkeypatch.setattr(PatternAgent, "_run_loop", spy)
    return seen


class TestTheFsmBodyRunPathRendersToo:
    """The renderer is called by both run bodies (``FsmInstance`` and ``CodeBackedInstance``)."""

    DOCUMENT: ClassVar[dict[str, Any]] = {
        "schema": "fsm-llm/executable",
        "kind": "fsm",
        "id": "greet",
        "contract": {
            "inputs": ["task"],
            "outputs": ["task"],
            "input_templates": {"task": "Say {{word}} to {{name}}"},
        },
        "body": {
            "fsm": {
                "name": "g",
                "description": "d",
                "initial_state": "s",
                "states": {"s": {"id": "s", "description": "d", "purpose": "p"}},
            }
        },
    }

    def test_arguments_render_into_the_input_value_of_an_fsm_body(self):
        registry = ExecutableRegistry()
        registry.add_document(self.DOCUMENT)

        result = registry.get("greet").run(
            {"word": "hi", "name": "Ann"},
            context=RunContext(llm_interface=ScriptedLLM(default_response="ok")),
        )

        assert result.status is Status.OK and result.outputs == {
            "task": "Say hi to Ann"
        }

    def test_an_internal_prefixed_argument_is_refused_on_that_path_too(self):
        registry = ExecutableRegistry()
        registry.add_document(self.DOCUMENT)

        with pytest.raises(ValueError, match="neither inputs nor placeholders"):
            registry.get("greet").run(
                {"word": "hi", "name": "Ann", "_approval_granted": True},
                context=RunContext(llm_interface=ScriptedLLM(default_response="ok")),
            )


class TestBuiltInPresetsRunById:
    def test_the_three_presets_are_served_by_a_source_and_the_class_table_is_unchanged(
        self,
    ):
        registry = _registry()

        assert set(OLD_SOPS) <= _ids(registry)
        for name in OLD_SOPS:
            assert registry.get(name).kind == "pattern"

    @pytest.mark.parametrize("name", OLD_SOPS)
    def test_each_preset_runs_by_id_on_the_task_its_sop_rendered(self, name, tasks):
        template, arguments, _ = OLD_SOPS[name]

        result = (
            _registry().get(name).run(arguments, context=_context(runs.ScriptedModel()))
        )

        assert result.status is Status.OK
        assert tasks == [template.format(**arguments)]
        assert result.detail["pattern"] == name

    def test_a_task_given_directly_runs_as_it_is(self, tasks):
        _registry().get("summarize").run(
            {"task": "just this"}, context=_context(runs.ScriptedModel())
        )

        assert tasks == ["just this"]

    @pytest.mark.parametrize("key", ["_approval_granted", "bogus"])
    def test_an_argument_outside_the_contract_is_refused_before_any_model_call(
        self, key
    ):
        model = runs.ScriptedModel()

        with pytest.raises(ValueError, match="neither inputs nor placeholders"):
            _registry().get("summarize").run(
                {"num_points": 1, "text": "t", key: "x"}, context=_context(model)
            )

        assert model.log == []

    def test_the_preset_agent_takes_the_config_options_of_its_file(self):
        agent = PatternAgent(
            pf.load_envelope("code-review"),
            runs._tools(),
            llm_interface=runs.ScriptedModel(),
        )

        assert (agent.config.temperature, agent.config.max_iterations) == (0.3, 5)

    def test_the_hosts_own_config_field_beats_the_file_default_and_a_given_option_beats_both(
        self,
    ):
        envelope = pf.load_envelope("code-review")
        host = AgentConfig(model="m", temperature=0.9)

        kept = PatternAgent(envelope, runs._tools(), host)
        given = PatternAgent(
            envelope, runs._tools(), host, options={"temperature": 0.5}
        )

        assert (kept.config.temperature, kept.config.max_iterations) == (0.9, 5)
        assert given.config.temperature == 0.5

    def test_a_class_that_stands_behind_a_file_keeps_its_own_options(self):
        agent = DebateAgent(llm_interface=runs.ScriptedModel())

        assert agent.config.max_iterations == AgentConfig(model="m").max_iterations


class TestAFileThatDeclaresAnOptionNamedModel:
    @pytest.mark.parametrize(
        "body",
        [
            {**_added(model="string"), "options": {"model": "x/y"}},
            {**_added(temperature="number"), "options": {"model": "x/y"}},
        ],
    )
    def test_a_preset_naming_model_is_refused(self, body):
        envelope, report = _load(_preset(**body))

        assert envelope is None and not report.is_valid
        assert any(
            "one model resolver" in e or "undeclared" in e for e in report.errors
        )

    def test_a_plain_pattern_file_naming_model_is_refused_too(self):
        document = pf.load_document("react")
        document["id"] = "user.react"
        document["body"]["options_schema"]["properties"]["model"] = {"type": "string"}
        document["body"]["options"]["model"] = "x/y"

        envelope, report = _load(document)

        assert envelope is None and any(
            "one model resolver" in e for e in report.errors
        )

    def test_a_given_model_option_is_never_applied_to_the_config(self):
        agent = PatternAgent(
            pf.load_envelope("code-review"),
            runs._tools(),
            llm_interface=runs.ScriptedModel(),
        )

        assert agent.config.model == "mock-model"


class TestExtends:
    def test_the_flat_document_is_the_base_with_the_preset_applied(self):
        document = _preset(
            "mine",
            **_added(temperature="number"),
            options={"temperature": 0.2, "classification": True},
            run={"hooks": [{"hook": "agents.verify_retry", "at": "after_run"}]},
        )
        document.update(
            name="Mine",
            contract={"inputs": ["task", "extra"]},
            requires={"handlers": ["agents.verify_retry"]},
        )

        envelope, report = _load(document)

        assert report.is_valid, report.errors
        react = pf.load_envelope("react")
        assert (
            envelope.id == "mine"
            and envelope.name == "Mine"
            and envelope.lifecycle.value == "draft"
        )
        assert (
            envelope.body.fsm == react.body.fsm
            and envelope.body.variants == react.body.variants
        )
        assert envelope.contract.inputs == [
            "task",
            "extra",
        ]  # written: replaces the base's
        assert envelope.requires.handlers == [
            *react.requires.handlers,
            "agents.verify_retry",
        ]
        assert envelope.requires.executables == [
            "react"
        ]  # `extends` is an implied requirement
        assert envelope.requires.tools == "some" and envelope.requires.callables == [
            "tools"
        ]
        assert (
            envelope.body.options["classification"] is True
        )  # a base option, overridden
        assert envelope.body.options["temperature"] == 0.2
        assert [h.hook for h in envelope.body.run.hooks] == ["agents.verify_retry"]
        assert envelope.body.extends is None

    def test_the_flat_envelope_runs_as_any_pattern_and_its_identity_is_not_the_bases(
        self,
    ):
        envelope, _ = _load(_preset("mine"))

        assert envelope.body.run.agent_type == "react"
        agent = PatternAgent(
            envelope, runs._tools(), llm_interface=runs.ScriptedModel()
        )
        assert agent.envelope is envelope
        assert (
            _registry().add_document(_preset("mine")).template_identity
            != _registry().get("react").template_identity
        )

    def test_an_unflattened_preset_cannot_be_instantiated(self):
        from fsm_llm.core.definitions.executable import InstantiationError
        from fsm_llm.core.definitions.executable_instance import instantiate_envelope

        with pytest.raises(InstantiationError, match="resolved when it is loaded"):
            instantiate_envelope(parse_executable(_preset()))

    def test_a_preset_with_no_bases_is_refused_not_guessed(self):
        envelope, report = load_executable(_preset(), resolver=namespace_resolver())

        assert envelope is None and any("bases=" in e for e in report.errors)

    def test_an_unknown_base_is_named(self):
        envelope, report = _load(_preset("p", "nowhere"))

        assert envelope is None and any("'nowhere'" in e for e in report.errors)

    def test_a_base_that_is_code_backed_has_no_template_and_is_refused_by_name(self):
        with pytest.raises(Exception, match="swarm"):
            _registry().add_document(_preset("p", "swarm"))

    def test_a_base_that_is_not_a_pattern_is_refused_by_name(self):
        registry = ExecutableRegistry()
        registry.add_document(TestTheFsmBodyRunPathRendersToo.DOCUMENT)

        with pytest.raises(ExecutableError, match="no pattern with a template"):
            registry.add_document(_preset("p", "greet"))

    def test_a_registry_resolves_a_user_preset_over_a_built_in_preset(self):
        registry = _registry()

        mine = registry.add_document(
            _preset("mine", "code-review", options={"max_iterations": 9})
        )

        assert (
            mine.envelope.body.options["temperature"] == 0.3
        )  # inherited from code-review
        assert mine.envelope.body.options["max_iterations"] == 9
        assert (
            mine.envelope.contract.input_templates
            == registry.get("code-review").envelope.contract.input_templates
        )

    @pytest.mark.parametrize("length", range(1, MAX_EXTENDS_DEPTH + 1))
    def test_a_chain_up_to_the_depth_limit_resolves(self, length):
        docs = _chain(length)

        document = resolve_extends(docs["p0"], docs.get)

        assert document["id"] == "p0" and document["body"]["fsm"] is not None

    def test_a_chain_one_link_over_the_limit_is_refused_by_name(self):
        docs = _chain(MAX_EXTENDS_DEPTH + 1)

        with pytest.raises(
            PresetError, match=rf"deeper than {MAX_EXTENDS_DEPTH} links.*'p0'"
        ):
            resolve_extends(docs["p0"], docs.get)

    @pytest.mark.parametrize("length", [1, 2, 4])
    def test_a_cycle_is_refused_by_name(self, length):
        docs = _chain(length, cycle=True)

        with pytest.raises(PresetError, match=r"extends cycle: 'p0'.*-> 'p0'"):
            resolve_extends(docs["p0"], docs.get)

    def test_a_preset_extending_itself_is_a_cycle(self):
        document = parse_executable(_preset("p", "p"))

        with pytest.raises(PresetError, match="extends cycle: 'p' -> 'p'"):
            resolve_extends(document, {"p": document}.get)

    def test_the_limits_come_through_load_executable_as_one_schema_error(self):
        docs = _chain(MAX_EXTENDS_DEPTH + 1)

        envelope, report = _load(_preset("top", "p0"), bases=docs.get)

        assert envelope is None
        assert any("deeper than" in e and "[schema]" in e for e in report.errors)

    def test_a_preset_may_not_redeclare_a_base_option(self):
        envelope, report = _load(_preset(**_added(classification="boolean")))

        assert envelope is None and any("redeclares" in e for e in report.errors)

    @pytest.mark.parametrize(
        "body",
        [
            {"fsm": {"states": {}}},
            {"handlers": [{"handler": "agents.tool_executor", "on": "entry"}]},
            {"variants": {"v": []}},
            {"run": {"agent_type": "x"}},
        ],
    )
    def test_a_preset_carries_no_template_and_no_handlers(self, body):
        with pytest.raises(
            Exception,
            match=r"exactly one of fsm, generator and extends|only options and hooks",
        ):
            parse_executable(_preset(**body))

    def test_a_body_names_one_of_fsm_generator_extends(self):
        document = pf.load_document("react")
        document["body"]["extends"] = "react"

        with pytest.raises(
            Exception, match="exactly one of fsm, generator and extends"
        ):
            parse_executable(document)

    def test_a_preset_option_default_is_checked_against_the_merged_schema(self):
        envelope, report = _load(_preset(options={"classification": "yes"}))

        assert envelope is None and not report.is_valid


class TestRequiredToolsAndInstructionText:
    def test_requires_tools_names_must_exist_in_the_registry(self):
        document = _preset("needs")
        document["requires"] = {"tools": ["search"]}
        envelope, report = _load(document)
        assert report.is_valid, report.errors
        other = ToolRegistry()
        other.register_function(lambda params: "r", name="calc", description="calc")

        with pytest.raises(AgentError, match=r"requires the tools \['search'\]"):
            PatternAgent(envelope, other, llm_interface=runs.ScriptedModel())
        assert PatternAgent(envelope, runs._tools(), llm_interface=runs.ScriptedModel())

    def test_a_template_placeholder_in_instruction_text_is_refused_by_v4(self):
        base = pf.load_document("react")
        base["id"] = "leaky"
        base["body"]["fsm"]["states"]["think"]["purpose"] += " about {{code}}"
        document = _preset("review", "leaky")
        document["contract"] = {
            "inputs": ["task"],
            "input_templates": {"task": "Review {{code}}"},
        }

        envelope, report = _load(document, bases={"leaky": parse_executable(base)}.get)

        assert envelope is None
        assert any(
            "{{code}}" in e and "no caller input renders" in e for e in report.errors
        )

    def test_a_template_placeholder_named_like_a_variable_or_option_is_refused(self):
        for name in ("think_instructions", "classification"):
            document = _preset()
            document["contract"] = {
                "inputs": ["task"],
                "input_templates": {"task": f"{{{{{name}}}}}"},
            }

            envelope, report = _load(document)

            assert envelope is None and any("named like" in e for e in report.errors), (
                name
            )

    def test_a_template_of_an_undeclared_input_is_refused(self):
        document = _preset()
        document["contract"] = {"inputs": ["task"], "input_templates": {"other": "x"}}

        envelope, report = _load(document)

        assert envelope is None and any("undeclared inputs" in e for e in report.errors)


class TestModelOriginPresets:
    ALLOW: ClassVar[dict[str, list[str]]] = {
        "executables": ["react"],
        "tools": [],
        "callables": ["tools"],
    }

    def test_a_model_may_extend_an_allow_listed_base_and_set_a_base_option(self):
        document = _preset("m", options={"classification": True})

        envelope, report = _load(document, origin="model", allow=self.ALLOW)

        assert report.is_valid, report.errors
        assert envelope.body.options["classification"] is True

    def test_a_base_off_the_list_is_refused(self):
        envelope, report = _load(
            _preset("m"), origin="model", allow={"executables": []}
        )

        assert envelope is None
        assert any(
            "extends names 'react'" in e and "allow-list" in e for e in report.errors
        )

    @pytest.mark.parametrize(
        ("body", "needle"),
        [
            (_added(temperature="number"), "options_schema"),
            (
                {
                    "run": {
                        "hooks": [{"hook": "agents.verify_retry", "at": "after_run"}]
                    }
                },
                "body.run.hooks",
            ),
        ],
    )
    def test_a_model_preset_may_not_declare_options_or_bind_hooks(self, body, needle):
        envelope, report = _load(_preset("m", **body), origin="model", allow=self.ALLOW)

        assert envelope is None and any(needle in e for e in report.errors)

    def test_a_model_preset_may_not_require_handlers(self):
        document = _preset("m")
        document["requires"] = {"handlers": ["agents.approval_gate"]}

        envelope, report = _load(document, origin="model", allow=self.ALLOW)

        assert envelope is None and any("requires.handlers" in e for e in report.errors)

    def test_the_base_is_not_judged_as_the_models_own_document(self):
        """The base binds handlers and a template; a preset of it must not be refused for those."""
        envelope, report = _load(_preset("m"), origin="model", allow=self.ALLOW)

        assert report.is_valid and envelope.body.handlers


class TestCleanBreak:
    @pytest.mark.parametrize(
        "name", ["SOPDefinition", "SOPRegistry", "load_builtin_sops"]
    )
    def test_the_sop_names_are_gone_from_the_package(self, name):
        import fsm_llm.agents as agents

        assert not hasattr(agents, name) and name not in agents.__all__
        with pytest.raises(ImportError):
            exec(f"from fsm_llm.agents import {name}")

    def test_the_module_is_gone_and_no_alias_stands_in_for_it(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("fsm_llm.agents.sop")

    def test_the_shipped_presets_hold_no_template_copy(self):
        for name in ("verified_react", "auto_memory", *OLD_SOPS):
            body = pf.load_document(name)["body"]
            assert body["extends"] == "react" and not {
                "fsm",
                "handlers",
                "variants",
            } & set(body)

    def test_a_preset_file_is_served_by_the_registry_like_any_pattern_file(self):
        registry = _registry()
        document = copy.deepcopy(_preset("viaregistry"))

        registry.add_document(document)

        assert "viaregistry" in _ids(registry)
        assert isinstance(registry.get("viaregistry").envelope, PatternEnvelope)
