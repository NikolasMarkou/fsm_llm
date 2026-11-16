# Upgrade guides

Each release that breaks a caller gets a folder here, `v<version>/`, with one page per theme. A page explains, for every `**BREAKING:**` entry of the [Changelog](../changelog.md), what changed, why, and the code before and after. The guides are not a tour of the current API: read the folder for the release you upgrade to.

## Upgrading to 0.13.0

Version 0.13.0 changes the following, in the order a migration meets them:

- the distribution is published as `llm-fsm` and the minimum Python is 3.11, so `pip` needs the new name before it installs anything;
- the package has no default model and reads no `.env` file at import, so every entry point needs a model from you;
- the flat modules of 0.12.0 became layer packages, so deep imports and patch targets move;
- public names are renamed so that one concept has one spelling in core, workflows, agents, monitor and reasoning;
- unused names, parameters and a whole package (the planner harness) are removed.

Every change is a clean break: no alias, no shim and no deprecation period. The old spelling fails at once with an `ImportError`, a `TypeError` or an `AttributeError`, except where a page says that a call still runs and changes meaning (the pages on the definition version, on lone dict tool parameters and on constructor limits are those). A search of your code for the old name finds every place to edit.

## How to read a guide

Each page opens with `Covers FSM-LLM 0.12.0 to 0.13.0.`, then has one `##` section per changelog entry, in the order of the entries. A section starts with the change in words and shows a **Before** block (the old code, which no longer works) and an **After** block (the new code). Every **After** block is checked by the test suite against the real package and most of them are executed, so they run as written; a block labelled **Sketch** needs your own model, tool or file and is only parsed and import-checked. In the changelog, the link at the end of an entry opens its section.

## The guides for 0.13.0

In the order to migrate:

1. [Distribution rename](v0.13.0/distribution-rename.md): install `llm-fsm`, uninstall the old name, and the licence note.
2. [Python and dependency floors](v0.13.0/python-floor.md): Python 3.11, and the dependency ranges with upper bounds, inside the tested major.
3. [Default model and dotenv files](v0.13.0/default-model-and-env.md): `ModelNotConfiguredError`, `load_env`, `--env-file`, the removed default model constants and the new default embedder.
4. [Module paths](v0.13.0/module-paths.md): where every module of 0.12.0 went, a script that rewrites the imports, and the patch targets that changed.
5. [Tools and approval policy](v0.13.0/tools-and-policy.md): `fsm_llm.tools` and `fsm_llm.policy`, out of `fsm_llm.agents`.
6. [Monitor route functions](v0.13.0/monitor-routes.md): the 49 route functions now in `fsm_llm.monitor.server` submodules.
7. [Core renames](v0.13.0/core-renames.md): `Classifier(llm_interface=)`, the `path` parameter, the visualizer text helpers, `ContextFilterError` and `HandlerSystem(handler_error_mode=)`.
8. [Definition version](v0.13.0/definition-version.md): the default definition version is `4.2`, and what it does to `fsm_id`.
9. [Workflows vocabulary](v0.13.0/workflows-vocabulary.md): time limits, step budget, `_step_id` fields and the renamed step classes.
10. [Agents renames and behaviour](v0.13.0/agents-renames.md): the per-tool `timeout_seconds`, `conversation_id`, the Python-tool API, lone dict tool parameters and constructor limits.
11. [Monitor and reasoning](v0.13.0/monitor-and-reasoning.md): `MonitorError` is an `FSMError`, `InstanceManager` close and `end_instance`, `ReasoningError` and the reasoning step budgets.
12. [Removals and small API changes](v0.13.0/removals.md): unused names, dead parameters, the changed behaviour of two workflow steps and the planner harness.

If you only call the public API of `fsm_llm` and keep a model in `LLM_MODEL`, the first three guides and the removals page are usually all that applies. If you patch or deep-import internals, read the module paths page first: its script handles most of the work.
