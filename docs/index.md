# FSM-LLM

A language model call starts from nothing. A multi-turn conversation that must collect data,
follow a set flow and decide the same way every time needs structure around the model.
**FSM-LLM** is a Python framework for chatbots and agents that follow a fixed structure: you
describe the conversation as a finite state machine in JSON, and the language model does the
language work inside each state.

It is one package, `fsm_llm`, with a small core and seven subpackages on top: reasoning
strategies, a workflow engine, agent patterns, a monitoring dashboard, an evaluation harness,
runtime FSM synthesis and public test doubles. Every subpackage is built from the same parts as
the core (stacked FSMs, message-free steps, one LLM layer), so there is one way to run a model.

## Where to start

- **Run a first conversation:** the [Quick Start](quickstart.md) installs the package, sets the
  model and walks through a small FSM.
- **Understand the idea:** [Concepts](concepts.md) says where the library fits, how one message
  travels through the two passes, and what each subpackage adds; the [Examples](examples.md)
  page lists the runnable examples in the repository.
- **Design your own machine:** the [FSM Design Guide](fsm_design.md) covers states, transitions,
  priorities, context and classification; [Handler Development](handlers.md) covers code that
  runs at the eight points of a turn.
- **See how it fits together:** [Architecture](architecture.md) draws the layers, the two-pass
  turn and the message-free step.
- **Look something up:** the [API Reference](api_reference.md) lists the public classes,
  functions and command-line tools.
- **Compile a procedure at run time:** [Runtime FSM synthesis](synthesis.md) explains how a model
  writes an FSM, the gates that judge it and how a host conversation runs it.
- **Upgrade from 0.12.0:** every breaking change of 0.13.0 has an upgrade guide with the code before and after, linked from the [Changelog](changelog.md). Start with the [upgrade guides overview](upgrade-guide/index.md): it lists the guides in the order to migrate (distribution and Python, default model, module paths, renames, removals).
- **Test code built on the library:** [Testing](testing.md) documents the public test doubles,
  the scripted user and the assertions in `fsm_llm.testing`.

## Why a decision was taken

Code that guards a non-obvious choice carries a `DECISION` comment that says what not to do and
why. The [Decisions](decisions/index.md) page is the generated index of all of them: every
decision, its rationale, and where in the code it applies.

## Plans

The [2026 roadmap](2026_roadmap.md) and the [agents roadmap](agents_roadmap.md) are the plans and
audit records the project works from. They are kept as written.

## Project

* [Contributing](contributing.md): the development setup, the checks, and the rules for a change.
* [Security](security.md): how to report a vulnerability and how the dependencies are checked.
* [Evaluate](evaluate.md): the method for scoring the examples against a model, and the run log.
* [Changelog](changelog.md): the release history and the known open items.

## Source

The code, the examples and the issue tracker are on
[GitHub](https://github.com/NikolasMarkou/fsm_llm). The library is released under the Apache
License 2.0.
