#!/usr/bin/env bash
# The lean gates of the repository (plan ace7a8c8, D-037). One script, run from any directory.
#
#   scripts/gate.sh step [--base REV] [--tests PATH...]
#       The step gate: about a minute, one pytest process. Run it before a commit. The touched
#       files are the working tree against REV (default HEAD), untracked files included.
#   scripts/gate.sh push
#       The push gate (the pre-push hook): the step checks over the diff against origin/main,
#       then everything linted, then one pytest process over the suites of the layers and the
#       named guard files of census section D (always, whatever the diff touches).
#   scripts/gate.sh watch
#       After a push: wait for the CI run(s) of HEAD, then compare the remote head with HEAD.
#   scripts/gate.sh map PATH...
#       Print the pytest arguments a step would take for the named touched paths (no run): the
#       check of the touched-file map.
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
# The guards of the invariants (census section D of plan ace7a8c8). The push gate runs every row
# whatever the diff touches (a security/ or context/ edit triggers no whole-directory run); a
# missing file is RED, so a deleted guard cannot drop a row unseen. One row per line: LABEL: FILES.
A=tests/test_fsm_llm_agents
F=tests/test_fsm_llm
R=tests/test_fsm_llm_regression
GUARD_ROWS=(
  "D1 HITL grant: $A/test_hitl_security.py $A/test_trust_boundary.py $A/test_hitl.py $A/test_tool_timeouts.py $A/test_policy_layer.py $A/test_tool_registry_names.py $A/test_executables.py $A/test_pattern_preview.py tests/test_fsm_llm_monitor/test_agent_launch.py $A/test_pattern_shapes.py $A/test_pattern_handler_order.py $A/test_pattern_agent.py $A/test_pattern_reflexion.py $A/test_pattern_prompt_chain.py $A/test_pattern_native_fc.py $A/test_pattern_orchestrator.py $A/test_pattern_rewoo.py $A/test_pattern_plan_execute.py $A/test_pattern_maker_checker.py $A/test_pattern_debate.py $A/test_pattern_evaluator_optimizer.py $A/test_hooks.py $F/test_call_settle.py $F/test_call_executor.py $F/test_call_delegated.py $F/test_call_delegated_settle.py $F/test_call_delegated_real.py $F/test_call_delegated_review.py tests/test_fsm_llm_reasoning/test_orchestrator_call.py tests/test_fsm_llm_synthesis/test_executor.py $A/test_handler_library.py $A/test_pattern_user_dir.py $A/test_presets.py $F/test_pattern_verify.py $F/test_workflow_verify.py tests/test_fsm_llm_workflows/test_workflow_executable.py $F/test_workflow_probe.py $F/test_resolve_target_guard.py $F/test_event_log_tools.py $F/test_event_log_approvals.py $F/test_event_log_parity.py $F/test_message_source.py $F/test_delegated_workflow.py"
  "D2 secret filter and internal prefix: $F/test_context_unit.py $F/fixtures/context_key_corpus.py $F/fixtures/census_key_corpus.py $F/fixtures/holdout_key_corpus.py $A/test_secret_hygiene.py $R/test_bugs.py tests/test_fsm_llm_reasoning/test_strategy_contract.py $F/test_event_log_extraction.py $F/test_event_log_handlers.py"
  "D3 handler-only keys, forged seed: $F/test_advance.py $F/test_completion_state.py $F/test_synthesize_field.py $F/test_event_log_restore_trust.py $F/test_internal_key_seeding_log.py $F/test_call_executor.py $F/test_call_delegated.py $F/test_call_delegated_settle.py $F/test_call_delegated_crash.py $F/test_call_delegated_review.py $F/test_call_settle.py $F/test_call_state.py $F/test_call_flow.py $F/test_kind_call.py tests/test_fsm_llm_reasoning/test_orchestrator_call.py tests/test_fsm_llm_synthesis/test_pushed_child_result.py $F/test_resolve_target_guard.py $F/test_event_log_tools.py $F/test_event_log_approvals.py $F/test_event_log_extraction.py $F/test_delegated_workflow.py"
  "D4 litellm, layering, laziness: $F/test_simulation_layering.py $F/test_simulation_stats.py $F/test_llm_similarity.py $R/test_import_contract.py $R/test_string_imports.py $R/test_import_laziness.py $F/test_llm_provider_banner.py $F/test_embedder.py $F/test_no_implicit_env.py $F/test_failure_codes.py tests/test_fsm_llm_workflows/test_workflow_loader.py tests/test_fsm_llm_workflows/test_serialize_golden.py tests/test_fsm_llm_monitor/test_workflow_preset_golden.py $F/test_target_resolver.py $F/test_delegated_resolution.py $F/test_required_executor.py $A/test_pattern_files.py tests/test_fsm_llm_reasoning/test_selection.py tests/test_fsm_llm_reasoning/test_selection_intents.py"
  "D5 request bytes: $F/test_event_log_greeting.py $F/test_kind_equality.py $A/test_native_fc_golden.py $A/test_pattern_request_bytes.py $F/test_llm_request_path.py $F/test_llm_complete.py tests/test_fsm_llm_reasoning/test_classifier_intents.py $F/test_purpose.py $F/test_event_log_parity.py $F/test_message_source.py $F/test_simulation_scenarios.py $F/test_run_context_adoption.py"
  "D6 untrusted text, paths: $F/test_pattern_generator.py $F/test_okf.py $F/test_agent_skills.py $A/test_okf.py tests/test_fsm_llm_synthesis/test_policy.py tests/test_fsm_llm_synthesis/test_gates.py tests/test_fsm_llm_synthesis/test_dryrun.py tests/test_fsm_llm_synthesis/test_synthesizer.py $F/test_executable_load.py $F/test_executable_models.py $F/test_executable_registry.py tests/test_fsm_llm_synthesis/test_synthesizer_async.py tests/test_fsm_llm_reasoning/test_cli_registry.py tests/test_fsm_llm_synthesis/test_executor.py tests/test_fsm_llm_synthesis/test_driver_rows.py tests/test_fsm_llm_synthesis/test_executor_bounds.py tests/test_fsm_llm_synthesis/test_pushed_child_result.py $F/test_pattern_body.py $F/test_workflow_body.py $F/test_workflow_verify.py tests/test_fsm_llm_workflows/test_workflow_executable.py $F/test_workflow_probe.py $F/test_pattern_verify.py $A/test_presets.py $A/test_pattern_user_dir.py tests/test_fsm_llm_meta/test_meta_executables.py $F/test_simulation_scenarios.py tests/test_fsm_llm_synthesis/test_diagnose_signals.py tests/test_fsm_llm_monitor/test_agent_types.py tests/test_fsm_llm_monitor/test_agent_draw.py tests/test_fsm_llm_monitor/test_agent_selects.py tests/test_fsm_llm_monitor/test_workflow_presets.py"
  "D7 one turn, cancel, lock order, budgets: $F/test_simulation_driver.py $F/test_event_log_greeting.py $F/test_event_log_lanes.py $F/test_event_log_hops.py $F/test_turn_gate.py $F/test_async_cancel.py $F/test_fsm_lock_order.py $F/test_turn_guard_deterministic.py $F/test_run_until_terminal.py $F/test_turn_budgets.py $F/test_executable_adapters.py tests/test_fsm_llm_workflows/test_native_steps.py tests/test_fsm_llm_workflows/test_executable_step.py tests/test_fsm_llm_workflows/test_executable_context.py tests/test_fsm_llm_workflows/test_workflow_executable.py $F/test_call_settle.py $F/test_call_delegated.py $F/test_call_delegated_settle.py $F/test_call_delegated_real.py $F/test_call_delegated_review.py $F/test_apush_fsm.py $F/test_run_host.py tests/test_fsm_llm_reasoning/test_orchestrator_call.py tests/test_fsm_llm_reasoning/test_solve_stream.py tests/test_fsm_llm_synthesis/test_executor.py tests/test_fsm_llm_synthesis/test_executor_bounds.py tests/test_fsm_llm_synthesis/test_pushed_child_result.py $A/test_async_agents.py $F/test_event_log_parity.py $F/test_simulation_simulator.py $F/test_delegated_workflow.py $F/test_run_context_adoption.py"
  "D8 event log: tests/test_fsm_llm_workflows/test_executable_context.py tests/test_fsm_llm_workflows/test_workflow_executable.py $F/test_event_log_lanes.py $F/test_event_log_census.py $F/test_event_log_hops.py $F/test_event_log_greeting.py $F/test_event_log_tools.py $F/test_event_log_approvals.py $F/test_event_log_extraction.py $F/test_event_log_handlers.py $F/test_event_log_producers.py $F/test_event_log_acceptance.py $F/test_event_log_engine.py $F/test_event_log_crash.py $F/test_event_log_stack_crash.py $F/test_event_log_jsonl.py $F/test_event_log_privacy.py $F/test_event_log_retention.py $F/test_purpose.py $F/test_failure_codes.py $F/test_call_log.py $F/test_call_delegated_settle.py $F/test_call_delegated_crash.py $F/test_call_delegated_review.py $F/test_event_log_parity.py $F/test_message_source.py $F/test_event_log_signals.py tests/test_fsm_llm_synthesis/test_trace_signals.py tests/test_fsm_llm_synthesis/test_diagnose_signals.py $F/test_delegated_workflow.py $F/test_run_context_adoption.py"
  "D9 server auth: tests/test_fsm_llm_monitor/test_server_security.py tests/test_fsm_llm_monitor/test_agent_types.py tests/test_fsm_llm_monitor/test_agent_launch.py $F/test_api_key_compare_sites.py $A/test_remote.py"
  "D10 supply chain: tests/test_packaging.py $R/test_supply_chain.py $R/test_distribution_name.py"
  "D11 no silent model, distinct roles: tests/test_fsm_llm_workflows/test_executable_context.py tests/test_fsm_llm_workflows/test_workflow_executable.py $F/test_model_not_configured.py $R/test_no_model_paths.py tests/test_fsm_llm_monitor/test_agent_launch.py $F/test_simulation_roles.py tests/test_fsm_llm_synthesis/test_roles.py tests/test_fsm_llm_monitor/test_workflow_presets.py $F/test_run_context_adoption.py"
  "D12 examples bind: tests/test_examples/test_example_calls_bind.py tests/test_examples/test_example_import_paths.py $A/test_agent_preview_golden.py"
)
unset A F R
MARKERS="not slow and not real_llm and not integration"
FAILS=0
mode=${1:-}

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

# A touched path may not gain an em dash (U+2014). Exempt, as in the ratchet: the plan record and
# the append-only logs. Compared with the base, so inherited ones do not count.
em_dash_gain() {
  local base=$1 f new old bad=0
  for f in "${TOUCHED[@]}"; do
    case $f in plans/*|docs/2026_roadmap.md|EVALUATE.md) continue ;; esac
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

# `fsm-llm` on PyPI is another project: no tracked text says `pip install fsm-llm` (with or without
# an extra). The history of CHANGELOG.md, the plan record, the bench data and the test that pins the
# refusal may name it.
no_wrong_install_hint() {
  local hits
  hits=$(git grep -nIE "pip install[ \"'-]*(--[a-z-]+ )*[\"']?fsm-llm([^-[:alnum:]_]|$)" -- . \
    ':!plans' ':!CHANGELOG.md' ':!scripts/gate.sh' \
    ':!tests/test_fsm_llm_regression/test_distribution_name.py')
  [ -z "$hits" ] || { echo "$hits"; echo "say pip install llm-fsm (the PyPI project fsm-llm is another one)"; return 1; }
}

any_src_py() { local f; for f in "${TOUCHED[@]}"; do case $f in src/*.py) return 0 ;; esac; done; return 1; }

# The two generated fixtures come from their writers. Step mode writes and prints the diff for the
# eye of the person running the gate (an unexpected change is theirs to read before the commit).
# Push mode never keeps a rewrite: the committed files are judged as they are, so a stale fixture
# (a module with no decision, a public name not in the snapshot) is RED here, not in CI.
FIXTURES=(tests/fixtures/module_path_map.json tests/fixtures/api_snapshot.txt)

write_fixtures() {
  "$PY" -m tests.test_fsm_llm_regression.test_module_path_map --write \
    && "$PY" scripts/api_snapshot.py --write tests/fixtures/api_snapshot.txt
}

regenerate_fixtures() {  # regenerate_fixtures BASE
  write_fixtures || return 1
  git --no-pager diff --stat "$1" -- "${FIXTURES[@]}"
}

verify_fixtures() {  # the writers change nothing, and HEAD holds what the tree holds
  local keep f bad=0
  keep=$(mktemp -d) || return 1
  for f in "${FIXTURES[@]}"; do mkdir -p "$keep/$(dirname "$f")"; cp "$f" "$keep/$f"; done
  write_fixtures || bad=1
  for f in "${FIXTURES[@]}"; do
    if ! cmp -s "$keep/$f" "$f"; then
      echo "stale fixture: $f differs from its writer's output (run scripts/gate.sh step, read the diff, commit it)"
      diff -u "$keep/$f" "$f" | head -20
      cp "$keep/$f" "$f"; bad=1
    fi
  done
  rm -rf "$keep"
  git --no-pager diff --quiet HEAD -- "${FIXTURES[@]}" \
    || { echo "a fixture has an uncommitted change: commit it before the push"; bad=1; }
  return $bad
}

# The test files of every GUARD_ROWS row, one per line. A missing file is reported and counted.
guard_files() {
  local row f
  for row in "${GUARD_ROWS[@]}"; do
    for f in ${row#*: }; do
      if [ ! -f "$f" ]; then echo "gate: guard file missing: $f ($row)" >&2; continue; fi
      case ${f##*/} in test_*.py) echo "$f" ;; esac
    done
  done
}

# The test files that use the helper $1 (a tests/ python file that is not a test file): the ones
# that import it, found with git grep. A conftest maps to the guard files under its directory (all
# of them for tests/conftest.py); a helper with no importer maps to all the guard files.
helper_users() {
  local stem=${1##*/} users dir=${1%/*}
  stem=${stem%.py}
  if [ "$stem" = conftest ]; then
    if [ "$dir" = tests ]; then guard_files; else guard_files | grep "^$dir/"; fi
    return 0
  fi
  users=$(git grep -l -E "^[[:space:]]*(from|import)[[:space:]].*\<$stem\>" -- 'tests/*test_*.py' 2>/dev/null)
  if [ -n "$users" ]; then printf '%s\n' "$users"; else guard_files; fi
}

# The test arguments of a step: the fixed set, the touched test files, the test files that use a
# touched helper and the suite directories mapped from the touched src prefix. A touch of engine/,
# core/, llm/ or aio.py maps to nothing: the step names its test files with --tests (never all of
# tests/test_fsm_llm/ on a step).
build_step_args() {
  ARGS=("${FIXED_TESTS[@]}")
  local f u
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
      tests/*.py)
        # `*` also matches `/` in a case pattern, so decide on the base name
        case ${f##*/} in
          test_*.py) [ -f "$f" ] && ARGS+=("$f") ;;
          *) while IFS= read -r u; do [ -n "$u" ] && ARGS+=("$u"); done < <(helper_users "$f")
             echo "note: $f is a test helper: ran the test files that use it" >&2 ;;
        esac ;;
    esac
  done
}

# One pytest process over the named paths, each once, a root-level file (tests/test_packaging.py)
# last: a root-level file placed between paths of tests/test_fsm_llm/ makes pytest 9 lose that
# folder's conftest for the files after it ("fixture 'log' not found", measured on the push set).
run_pytest() {
  local seen=" " a args=() tail_args=()
  for a in "$@"; do
    case $seen in *" $a "*) continue ;; esac
    seen="$seen$a "
    case $a in tests/*/*) args+=("$a") ;; tests/*.py) tail_args+=("$a") ;; *) args+=("$a") ;; esac
  done
  nice "$PY" -m pytest -q -p no:cacheprovider -m "$MARKERS" "${args[@]}" "${tail_args[@]}"
}

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
  check "no pip install fsm-llm hint" no_wrong_install_hint
  if [ "$mode" = push ]; then
    check "generated fixtures match their writers" verify_fixtures
  else
    check "generated fixtures" regenerate_fixtures "$base"
  fi
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
    ARGS+=($(guard_files))
    echo "guard rows run whatever the diff touches:"
    local row
    for row in "${GUARD_ROWS[@]}"; do echo "  ${row%%:*}"; done
    local f whole=0 missing
    missing=$(guard_files 2>&1 >/dev/null)
    [ -z "$missing" ] || { echo "$missing"; check "named guard files exist" false; }
    for f in "${TOUCHED[@]}"; do
      case $f in src/fsm_llm/engine/*|src/fsm_llm/llm/*|src/fsm_llm/core/*|src/fsm_llm/runtime/*|src/fsm_llm/aio.py) whole=1 ;; esac
    done
    [ "$whole" -eq 1 ] && ARGS+=(tests/test_fsm_llm)
    [ "$FAILS" -eq 0 ] && check "pytest (one process)" run_pytest "${ARGS[@]}"
  fi
  finish push
}

mode_map() {  # map PATH...: the pytest arguments of a step for these touched paths, no run
  TOUCHED=("$@")
  build_step_args
  printf '%s\n' "${ARGS[@]}"
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

[ $# -gt 0 ] && shift
case $mode in
  step) mode_step "$@" ;;
  push) mode_push ;;
  map) mode_map "$@" ;;
  watch) mode_watch ;;
  venv) mode_venv "$@" ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac
