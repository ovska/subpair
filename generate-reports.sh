#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBPAIR_EXE="${SUBPAIR_EXE:-${REPO_ROOT}/.venv/bin/subpair}"
SUBPAIR_CACHE_DIR="${SUBPAIR_CACHE_DIR:-${REPO_ROOT}/.subpair-cache}"

if [[ ! -x "${SUBPAIR_EXE}" ]]; then
  printf 'subpair executable not found at %s\n' "${SUBPAIR_EXE}" >&2
  printf 'Create the repository virtualenv or set SUBPAIR_EXE explicitly.\n' >&2
  exit 1
fi

if [[ ! -f "${SUBPAIR_CACHE_DIR}/manifest.json" ]]; then
  printf 'subpair cache not found at %s\n' "${SUBPAIR_CACHE_DIR}" >&2
  printf 'Run subpair fetch first or set SUBPAIR_CACHE_DIR explicitly.\n' >&2
  exit 1
fi

SEARCH_ARGS=(
  --cache "${SUBPAIR_CACHE_DIR}"
  --band 35 150
  --delay-range -10 10 0.05
  --gain-range -6 6 0.1
  --ppo 64
  --eq-range 35 120
  --eq-range-slope 48
  --max-boost 6
  --max-cut 18
  --eq-bands 16
  --score-low-end-weight 0.5
  --score-dip-weight 1
  --top 8
  --low-shelf on
  --modal on
)

generate_report() {
  local label="$1"
  local report_title="$2"
  shift 2
  local results_path="${SUBPAIR_CACHE_DIR}/search-results-${label}.json"
  local report_path="${REPO_ROOT}/subpair-report-${label}.html"

  printf '\nGenerating %s search results...\n' "${label}"
  "${SUBPAIR_EXE}" search \
    "${SEARCH_ARGS[@]}" \
    --eq-target dsp \
    "$@" \
    --results "${results_path}"

  printf 'Building %s report...\n' "${label}"
  "${SUBPAIR_EXE}" report \
    --cache "${SUBPAIR_CACHE_DIR}" \
    --results "${results_path}" \
    --output "${report_path}" \
    --report-title "${report_title}" \
    --top 5 \
    --limit 15 \
    --room 345x274x248
}

generate_report dsp "Subpair DSP"
generate_report no-eq "Subpair No EQ" --eq-bands 0

printf '\nReports generated:\n'
printf '  %s\n' "${REPO_ROOT}/subpair-report-dsp.html"
printf '  %s\n' "${REPO_ROOT}/subpair-report-no-eq.html"
