# Python and dependency floors

Covers FSM-LLM 0.12.0 to 0.13.0. Two floors rise in this release: the Python version (3.11, was 3.10) and the dependencies (every declared range has an upper bound, and `litellm` and the `mcp` extra raise their lower bound). Neither changes a line of your code. They decide whether `pip` can install the package into your environment, so read this page before you upgrade a production environment.

## Python 3.11 is the minimum

The minimum Python version is 3.11: `requires-python = ">=3.11"` (it was `>=3.10`). The Python 3.10 classifier, the tox `py310` environment and the 3.10 CI leg are removed, and the build backend requires `setuptools>=77`. The package is also tested on 3.12 and declared for 3.13, 3.14 and 3.15 (see the CHANGELOG entry for the evidence). `pip` on Python 3.10 refuses the 0.13.0 release and keeps resolving 0.12.0.

What to do: run the library on 3.11 or later. If you build the package from a clone, use a 3.11+ interpreter for the virtual environment too.

**Before**

```bash
python3.10 -m venv .venv && .venv/bin/pip install -e .
```

**After**

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e .
```

**After**

```python
import sys

assert sys.version_info >= (3, 11), "FSM-LLM 0.13.0 needs Python 3.11 or later"
```

## Dependency ranges have upper bounds and two floors rise

Every declared dependency range now has an upper bound that contains its exact pin in `constraints.txt`, and two floors rise:

- `litellm>=1.109.0,<2.0` (was `>=1.83.0,<2.0`);
- the `mcp` extra `mcp>=2.2,<3` (was `>=1.0.0`, which admitted the 1.x line the code no longer supports).

The other ranges keep their floors and gain an upper bound at the next major: `pydantic<3`, `loguru<1`, `python-dotenv<2`, `tenacity<10`, `pyyaml<7`, `fastapi<1`, `uvicorn<1`, `jinja2<4`, `opentelemetry-api` and `-sdk<2`, `httpx<1`, and the dev tools. `constraints.txt` pins every direct dependency and the build backend exactly (it held eight names), chosen with `uv pip compile` on Python 3.11 and 3.15 for linux, macos and windows. `tests/test_packaging.py::TestDeclaredRangesMatchPins` fails on an unbounded range, a missing pin or a pin outside its range.

An environment that holds litellm below 1.109 or mcp 1.x must upgrade. An environment that pins a dependency past an upper bound (for example pydantic 3 or loguru 1) cannot install this release until the bound is lifted by a later one. From a clone, `-c constraints.txt` installs the exact pins the project tests against.

**Before**

```text
litellm>=1.83.0,<2.0
mcp>=1.0.0
```

**After**

```text
litellm>=1.109.0,<2.0
mcp>=2.2,<3
```

```bash
pip install -c constraints.txt -e ".[all]"   # from a clone
```

**After**

```python
from importlib.metadata import PackageNotFoundError, version


def at_least(name: str, floor: tuple[int, ...]) -> bool | None:
    try:
        parts = tuple(int(x) for x in version(name).split(".")[: len(floor)])
    except PackageNotFoundError:
        return None  # not installed: an optional extra
    return parts >= floor


assert at_least("litellm", (1, 109)) is not False
assert at_least("mcp", (2, 2)) is not False
```
