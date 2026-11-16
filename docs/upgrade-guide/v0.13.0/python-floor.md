# Python and dependency floors

Covers FSM-LLM 0.12.0 to 0.13.0. Two floors rise in this release: the Python version (3.11, was 3.10) and the dependencies (every declared range has an upper bound inside the tested major, and `litellm`, the `mcp` extra and `tenacity` raise their lower bound). Neither changes a line of your code. They decide whether `pip` can install the package into your environment, so read this page before you upgrade a production environment.

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

The other ranges keep their floors and gain an upper bound at the next major (the floors that still reached into an older major are narrowed in the next section): `pydantic<3`, `loguru<1`, `python-dotenv<2`, `tenacity<10`, `pyyaml<7`, `fastapi<1`, `uvicorn<1`, `jinja2<4`, `opentelemetry-api` and `-sdk<2`, `httpx<1`, and the dev tools. `constraints.txt` pins every direct dependency and the build backend exactly (it held eight names), chosen with `uv pip compile` on Python 3.11 and 3.15 for linux, macos and windows. `tests/test_packaging.py::TestDeclaredRangesMatchPins` fails on an unbounded range, a missing pin or a pin outside its range.

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

## Dependency ranges admit only the tested major

No declared range admits a major version other than the one the project tests against. The first version component of every floor now equals that of the exact pin in `constraints.txt`, and the ceiling is the next major; `0.x` packages (the major of a `0.x` pin is 0) stay bounded by `<1`. Four floors rise, one of them in a runtime dependency, and the build backend gains an upper bound:

- `tenacity>=9.0,<10` (was `>=8.0,<10`; a core dependency that litellm's retry path imports, tested on 9.2.1 only);
- `pytest>=9.0,<10` (was `>=7.4.0,<10`), `pytest-asyncio>=1.0,<2` (was `>=0.21.0,<2`) and `pytest-cov>=7.0,<8` (was `>=4.0.0,<8`), all in the `dev` extra;
- `[build-system] requires = ["setuptools>=77,<85"]` (was `["setuptools>=77"]`, so building the sdist could pick a setuptools newer than the one tested). It is the one named exception to the first-component rule: setuptools has no stable major semantics, so its floor is the first release that reads the PEP 639 license expression (77) and its ceiling is the release after the pinned 84. The `dev` extra already carried the same range.

An environment that pins `tenacity` below 9 cannot install this release until it lifts that pin (a plain `pip install --upgrade llm-fsm` upgrades `tenacity` by itself). The three dev tools matter only to a contributor who installs the `dev` extra with older tools. `tests/test_packaging.py::TestNoRangeAdmitsAnUntestedMajor` fails on any range that admits another major.

**Before**

```text
tenacity>=8.0,<10
pytest>=7.4.0,<10
pytest-asyncio>=0.21.0,<2
pytest-cov>=4.0.0,<8
setuptools>=77
```

**After**

```text
tenacity>=9.0,<10
pytest>=9.0,<10
pytest-asyncio>=1.0,<2
pytest-cov>=7.0,<8
setuptools>=77,<85
```

```bash
pip install --upgrade "tenacity>=9.0,<10"   # an environment that held tenacity 8
```

**After**

```python
from importlib.metadata import PackageNotFoundError, version


def major_of(name: str) -> int | None:
    try:
        return int(version(name).split(".")[0])
    except PackageNotFoundError:
        return None  # not installed


assert major_of("tenacity") in (None, 9)
```
