# The distribution is llm-fsm

Covers FSM-LLM 0.12.0 to 0.13.0. The package is published on PyPI as `llm-fsm`. Only the distribution name changed: the import package is still `fsm_llm`, every console script keeps its name, and the extras (`agents`, `workflows`, `reasoning`, `monitor`, `eval`, `synthesis`, `testing`, `mcp`, `a2a`, `otel`, `all`) keep theirs. Code that imports `fsm_llm` needs no edit. What changes is the name you give to `pip`, `requirements.txt`, `pyproject.toml` and lock files.

## The distribution is published as llm-fsm

`[project] name` in `pyproject.toml` is `llm-fsm` (it was `fsm-llm`), so the wheel is `llm_fsm-<version>-py3-none-any.whl`, the sdist `llm_fsm-<version>.tar.gz`, and the extras are requested as `pip install "llm-fsm[monitor]"` (the `all` extra refers to `llm-fsm[...]`). Console scripts (`fsm-llm`, `fsm-llm-validate`, `fsm-llm-visualize`, `fsm-llm-monitor`, `fsm-llm-meta`, `fsm-llm-eval`, `fsm-llm-okf`, `fsm-llm-skills`) and the OKF product string `fsm-llm/<version>` written into bundles are unchanged.

`fsm-llm` on PyPI is a different project. Never run `pip install fsm-llm`: it installs that other project.

A note on the licence: version 0.1.1 of `llm-fsm` on PyPI (2025-05-19) carries the GPLv3 licence classifier; this project is Apache-2.0, so 0.13.0 is licensed differently from that earlier upload under the same name. Nothing was yanked or deleted.

An environment that installed the old name from a clone needs a clean reinstall, because the editable install of the new name does not replace the old one, and a leftover `src/fsm_llm.egg-info` keeps reporting the old distribution to `importlib.metadata` until it is deleted (`make clean` removes it):

1. uninstall the old distribution: `pip uninstall fsm-llm`;
2. delete the leftover metadata directory `src/fsm_llm.egg-info` in your clone;
3. install again: `pip install -e .` (a clone) or `pip install llm-fsm` (PyPI);
4. check which distribution owns the import: `python -c "import importlib.metadata as m; print(m.version('llm-fsm'))"`.

The error messages that tell you to install an extra now name the new distribution (`pip install "llm-fsm[extra]"`). The "From a clone" hints of the old name are gone. Also fixed: the wheel build no longer warns that `fsm_llm.monitor.static` is "absent from the `packages` configuration" (`include-package-data = false`; the wheel and sdist file lists are unchanged).

**Before**

```bash
pip install fsm-llm                # not this project, never run it again
pip install "fsm-llm[monitor]"
```

```text
# requirements.txt
fsm-llm[agents]>=0.12
```

**After**

```bash
pip uninstall fsm-llm
rm -rf src/fsm_llm.egg-info        # clone owners only
pip install llm-fsm
pip install "llm-fsm[monitor]"
```

```text
# requirements.txt
llm-fsm[agents]>=0.13
```

**After**

```python
import fsm_llm  # the import package did not change

assert fsm_llm.__name__ == "fsm_llm"
assert fsm_llm.API.__module__.startswith("fsm_llm.")
```
