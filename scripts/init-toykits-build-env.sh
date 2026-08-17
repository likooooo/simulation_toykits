#!/usr/bin/env bash
# Toykits runtime env (source before pytest/streamlit). Build compile env is internal to build_toykits.py.
# Usage: source scripts/init-toykits-build-env.sh

_MANAGE_SHELL_OPTS=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]] && [[ -z "${BASH_SOURCE[2]:-}" ]]; then
  _MANAGE_SHELL_OPTS=1
fi
if (( _MANAGE_SHELL_OPTS )); then
  _INIT_SAVED_OPTS="$(set +o)"
fi

set -euo pipefail

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ARTIFACTS="${SIMULATION_ARTIFACTS_DIR:-${_REPO}/.simulation_toolkits}"

export SIMULATION_ARTIFACTS_DIR="${_ARTIFACTS}"
export SIMULATION_DATABASE_DIR="${SIMULATION_ARTIFACTS_DIR}/assets"
export GENERATE_GOLDEN_TOOLS_DIR="${GENERATE_GOLDEN_TOOLS_DIR:-${HOME}/repos/simulation_baseline_tools}"
export PYTHONPATH="${_REPO}:${SIMULATION_ARTIFACTS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${SIMULATION_ARTIFACTS_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if (( _MANAGE_SHELL_OPTS )); then
  eval "${_INIT_SAVED_OPTS}"
fi
