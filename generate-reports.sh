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
  --band 20 250
  --delay-range -10 10 0.1
  --gain-range -6 6 0.1
  --ppo 64
  --eq-range 30 120
  --eq-range-slope 48
  --max-boost 6
  --max-cut 18
  --eq-bands 16
  --tie-tolerance-db 2
  --top 8
  --low-shelf on
)

generate_report() {
  local target="$1"
  local results_path="${SUBPAIR_CACHE_DIR}/search-results-${target}.json"
  local report_path="${REPO_ROOT}/subpair-report-${target}.html"

  printf '\nGenerating %s search results...\n' "${target}"
  "${SUBPAIR_EXE}" search \
    "${SEARCH_ARGS[@]}" \
    --eq-target "${target}" \
    --results "${results_path}"

  printf 'Building %s report...\n' "${target}"
  "${SUBPAIR_EXE}" report \
    --cache "${SUBPAIR_CACHE_DIR}" \
    --results "${results_path}" \
    --output "${report_path}" \
    --top 5 \
    --limit 15
}

# generate_report flat
generate_report dsp

printf '\nReports generated:\n'
# printf '  %s\n' "${REPO_ROOT}/subpair-report-flat.html"
printf '  %s\n' "${REPO_ROOT}/subpair-report-dsp.html"
