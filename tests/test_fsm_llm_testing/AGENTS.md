# test_fsm_llm_testing

Path: `tests/test_fsm_llm_testing`
Purpose: Offline pytest suite for `fsm_llm.testing` (the public test doubles, scripted user and assertions). Today it pins the layering of the wired skeleton; each public name brings its own tests.

## Scope

Covers `fsm_llm.testing` as a leaf of the package: imported by name, loaded by no library module, importing no `pytest`, `litellm` or other subpackage. Not covered here: the repository's own use of the doubles (that stays in the suites that use them).

## Architecture

| Piece | Where | What it does |
| --- | --- | --- |
| `_run_probe(code)` | `test_layering.py` | Runs a snippet under `python -I` with `OLLAMA_API_BASE`, `OLLAMA_HOST` on a closed port and an empty `PATH` |
| `_imported_modules(tree)` | `test_layering.py` | `(module, level, line)` of every import, resolving `from fsm_llm import x` and `from . import x` to the module they bind |
| `_PKG_ROOT` | `test_layering.py` | `Path(fsm_llm.__file__).parent`, so the scans survive a layout move |

## Key files

| File | Covers |
| --- | --- |
| `test_layering.py` | `TestImportProbes`, `TestSourceLayering`, `TestSkeleton` |
| `__init__.py` | Empty package marker |

## Invariants and constraints

- A probe that fails to import `fsm_llm.testing` must fail the test, never skip it.
- The library scan asserts it saw files (`api.py`, `pipeline.py`) so an empty derivation cannot pass vacuously.
- No test reaches a model, a network or a clock.
