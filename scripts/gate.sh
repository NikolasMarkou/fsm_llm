#!/usr/bin/env bash
# The lean gates of the repository (plan ace7a8c8, D-037). One script, run from any directory.
#
#   scripts/gate.sh step [--base REV] [--tests PATH...]
#       The step gate: about a minute, one pytest process. Run it before a commit. The touched
#       files are the working tree against REV (default HEAD), untracked files included.
#   scripts/gate.sh push
#       The push gate (the pre-push hook): the step checks over the diff against origin/main,
#       then everything linted, then one pytest process over the suites of the layers.
#   scripts/gate.sh watch
#       After a push: wait for the CI run(s) of HEAD, then compare the remote head with HEAD.
#   scripts/gate.sh venv PYTEST_ARG...
#       Run the named pytest paths on the Python 3.12, 3.14 and 3.15 venvs, for a cancel,
#       stream-close or schema-bytes step only. Needs GATE_VENV_DIR holding v312, v314, v315.
#
# Environment: FSM_LLM_VENV (default <repo>/.venv) is the virtualenv that holds the tools and the
# dependencies; the package is always imported from THIS checkout (PYTHONPATH=<repo>/src, and the
# run stops when fsm_llm.__file__ is elsewhere), so a git-archive copy of the tree is gateable
# with the shared venv. The tree must be a git work tree. GATE_BASE overrides the base of the
# push gate (default origin/main, as last fetched: the script never fetches).
#
# The wide run is CI's job (7 legs). No model, no network: LLM_MODEL is unset and the model server
# variables point at a closed port.
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 99
VENV=${FSM_LLM_VENV:-$ROOT/.venv}
PY=$VENV/bin/python

die() { echo "gate: $*" >&2; exit 2; }
[ -x "$PY" ] || die "no python at $PY (set FSM_LLM_VENV to a virtualenv with the dev extras)"
git rev-parse --git-dir >/dev/null 2>&1 || die "$ROOT is not a git work tree"

unset LLM_MODEL
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1 OLLAMA_API_BASE=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9

# The step set of the plan: the files that guard the invariants every step must keep.
FIXED_TESTS=(
  tests/test_fsm_llm_regression/test_module_size_guard.py
  tests/test_fsm_llm_regression/test_string_imports.py
  tests/test_fsm_llm_regression/test_import_laziness.py
  tests/test_fsm_llm/test_fsm_lock_order.py
  tests/test_fsm_llm/test_aio.py
  tests/test_fsm_llm/test_kind_equality.py
  tests/test_fsm_llm/test_definition_identity.py
  tests/test_fsm_llm/test_executables_corpus.py
  tests/test_fsm_llm/test_schema_registry.py
  tests/test_fsm_llm/test_state_kinds.py
  tests/test_fsm_llm/test_codes.py
)
MARKERS="not slow and not real_llm and not integration"
FAILS=0

check() {  # check NAME CMD...: run, say RED and count a failure, never stop
  local name=$1; shift
  echo "== $name"
  "$@" || { echo "RED: $name"; FAILS=$((FAILS + 1)); }
}

assert_checkout() {
  python -c "import sys, fsm_llm; p = fsm_llm.__file__; assert p.startswith(sys.argv[1]), p" "$ROOT/src/" \
    || die "fsm_llm is not imported from $ROOT/src"
}

TOUCHED=()
collect_touched() {  # TOUCHED := files changed against $1 (deleted ones left out) plus untracked ones
  mapfile -t TOUCHED < <({ git diff --name-only --diff-filter=d "$1" --; git ls-files --others --exclude-standard; } | sort -u)
}

dash_count() { LC_ALL=C grep -o $'\xe2\x80\x94' 2>/dev/null | wc -l; }

# A touched path may not gain an em dash (U+2014). Exempt, as in the ratchet: the plan record, the
# frozen baselines and the append-only logs. Compared with the base, so inherited ones do not count.
em_dash_gain() {
  local base=$1 f new old bad=0
  for f in "${TOUCHED[@]}"; do
    case $f in plans/*|scripts/bench_data/*|examples/*|docs/2026_roadmap.md|EVALUATE.md) continue ;; esac
    [ -f "$f" ] || continue
    new=$(dash_count < "$f")
    [ "$new" -gt 0 ] || continue
    old=$(git show "$base:$f" 2>/dev/null | dash_count)
    if [ "$new" -gt "$old" ]; then echo "em dash: $f has $new, the base has $old"; bad=1; fi
  done
  return $bad
}

py_ruff() {  # ruff check and format --check on the touched python files
  local files=() f
  for f in "${TOUCHED[@]}"; do case $f in *.py) [ -f "$f" ] && files+=("$f") ;; esac; done
  [ "${#files[@]}" -gt 0 ] || { echo "(no python file touched)"; return 0; }
  ruff check "${files[@]}" && ruff format --check "${files[@]}"
}

any_src_py() { local f; for f in "${TOUCHED[@]}"; do case $f in src/*.py) return 0 ;; esac; done; return 1; }

# The two generated fixtures come from their writers; the diff is for the eye of the person
# running the gate (an unexpected change is theirs to read before the commit).
regenerate_fixtures() {
  "$PY" -m tests.test_fsm_llm_regression.test_module_path_map --write \
    && "$PY" scripts/api_snapshot.py --write tests/fixtures/api_snapshot.txt \
    && git --no-pager diff --stat "$1" -- tests/fixtures/module_path_map.json tests/fixtures/api_snapshot.txt
}

# The test arguments of a step: the fixed set, the touched test files and the suite directories
# mapped from the touched src prefix. A touch of engine/, core/, llm/ or aio.py maps to nothing:
# the step names its test files with --tests (never all of tests/test_fsm_llm/ on a step).
build_step_args() {
  ARGS=("${FIXED_TESTS[@]}")
  local f d
  for f in "${TOUCHED[@]}"; do
    case $f in
      src/fsm_llm/agents/*) ARGS+=(tests/test_fsm_llm_agents tests/test_fsm_llm_meta) ;;
      src/fsm_llm/reasoning/*) ARGS+=(tests/test_fsm_llm_reasoning) ;;
      src/fsm_llm/workflows/*) ARGS+=(tests/test_fsm_llm_workflows) ;;
      src/fsm_llm/synthesis/*) ARGS+=(tests/test_fsm_llm_synthesis) ;;
      src/fsm_llm/monitor/*) ARGS+=(tests/test_fsm_llm_monitor) ;;
      src/fsm_llm/eval/*) ARGS+=(tests/test_fsm_llm_eval) ;;
      src/fsm_llm/testing/*) ARGS+=(tests/test_fsm_llm_testing) ;;
      tests/fixtures/*) ;;
      tests/*/test_*.py|tests/test_*.py) [ -f "$f" ] && ARGS+=("$f") ;;
      tests/test_fsm_llm/*.py) echo "note: $f is a helper of tests/test_fsm_llm/; name the files that use it with --tests" ;;
      tests/*/*.py) d=${f#tests/}; ARGS+=("tests/${d%%/*}") ;;
    esac
  done
}

run_pytest() { nice "$PY" -m pytest -q -p no:cacheprovider -m "$MARKERS" "$@"; }

# The cheap checks shared by both gates. $1 is the base the touched files are read against.
common_checks() {
  local base=$1
  collect_touched "$base"
  echo "touched against $base: ${#TOUCHED[@]} file(s)"
  check "ruff on touched python" py_ruff
  check "no new em dash" em_dash_gain "$base"
  check "whitespace (git diff --check)" git --no-pager diff --check "$base" --
  check "import contracts" lint-imports
  check "decision anchors resolve" "$PY" scripts/decision_index.py --check
  if [ "${ALWAYS_MYPY:-0}" = 1 ] || any_src_py; then check "mypy" "$PY" -m mypy -p fsm_llm; fi
  check "generated fixtures" regenerate_fixtures "$base"
}

finish() {  # finish LABEL: summary line and exit status
  if [ "$FAILS" -gt 0 ]; then echo "GATE RED ($1): $FAILS check(s) failed"; exit 1; fi
  echo "GATE GREEN ($1)"
}

mode_step() {
  local base=HEAD
  local extra=()
  while [ $# -gt 0 ]; do
    case $1 in
      --base) [ $# -ge 2 ] || die "--base needs a revision"; base=$2; shift 2 ;;
      --tests) shift; extra=("$@"); break ;;
      *) die "step: unknown argument $1" ;;
    esac
  done
  assert_checkout
  common_checks "$base"
  # a red cheap check skips the pytest run: fix those first
  if [ "$FAILS" -eq 0 ]; then
    build_step_args
    check "pytest (one process)" run_pytest "${ARGS[@]}" "${extra[@]}"
  fi
  finish step
}

mode_push() {
  local base=${GATE_BASE:-origin/main}
  git rev-parse --verify --quiet "$base" >/dev/null || die "no ref $base (set GATE_BASE)"
  assert_checkout
  ALWAYS_MYPY=1 common_checks "$base"
  check "ruff (everything)" ruff check src/ tests/
  check "ruff format (everything)" ruff format --check src/ tests/
  check "changelog form" "$PY" scripts/changelog_check.py check
  check "site-packages .pth audit" "$PY" scripts/audit_pth.py
  if [ "$FAILS" -eq 0 ]; then
    build_step_args
    ARGS+=(tests/test_fsm_llm_agents tests/test_fsm_llm_synthesis tests/test_fsm_llm_reasoning
      tests/test_fsm_llm_monitor tests/test_fsm_llm_eval tests/test_fsm_llm_testing
      tests/test_fsm_llm_workflows tests/test_fsm_llm_meta tests/test_fsm_llm_regression
      tests/test_examples tests/test_packaging.py)
    local f whole=0
    for f in "${TOUCHED[@]}"; do
      case $f in src/fsm_llm/engine/*|src/fsm_llm/llm/*|src/fsm_llm/core/*|src/fsm_llm/runtime/*|src/fsm_llm/aio.py) whole=1 ;; esac
    done
    [ "$whole" -eq 1 ] && ARGS+=(tests/test_fsm_llm)
    check "pytest (one process)" run_pytest "${ARGS[@]}"
  fi
  finish push
}

mode_watch() {
  command -v gh >/dev/null || die "gh is not installed"
  local head ids id i
  head=$(git rev-parse HEAD)
  for i in $(seq 1 30); do
    ids=$(gh run list --commit "$head" --json databaseId --jq '.[].databaseId')
    [ -n "$ids" ] && break
    sleep 4
  done
  [ -n "${ids:-}" ] || die "no CI run found for $head"
  for id in $ids; do
    check "CI run $id" gh run watch "$id" --exit-status
  done
  check "remote main equals HEAD" test "$(git ls-remote origin main | cut -f1)" = "$head"
  finish watch
}

mode_venv() {
  [ $# -gt 0 ] || die "venv: name the pytest paths to run"
  [ -n "${GATE_VENV_DIR:-}" ] || die "venv: set GATE_VENV_DIR to a directory holding v312, v314 and v315"
  local v py
  for v in 312 314 315; do
    py=$GATE_VENV_DIR/v$v/bin/python
    [ -x "$py" ] || die "no python at $py"
    echo "== Python $v"
    "$py" -c "import sys, fsm_llm; p = fsm_llm.__file__; assert p.startswith(sys.argv[1]), p" "$ROOT/src/" \
      || die "fsm_llm is not imported from $ROOT/src on v$v"
    check "pytest on v$v" nice "$py" -m pytest -q -p no:cacheprovider "$@"
  done
  finish venv
}

mode=${1:-}
[ $# -gt 0 ] && shift
case $mode in
  step) mode_step "$@" ;;
  push) mode_push ;;
  watch) mode_watch ;;
  venv) mode_venv "$@" ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac
