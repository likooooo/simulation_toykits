#!/usr/bin/env bash
# Toykits runtime env (source before pytest/streamlit).
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
_ARTIFACTS="${_REPO}/.simulation_core"
# shellcheck source=../simulation_core/scripts/init-simulation-build-env.sh
source "${_REPO}/simulation_core/scripts/init-simulation-build-env.sh" "${_ARTIFACTS}" "${_REPO}"
export SIMULATION_DATABASE_DIR="${SIMULATION_ARTIFACTS_DIR}/assets/database"

if (( _MANAGE_SHELL_OPTS )); then
  eval "${_INIT_SAVED_OPTS}"
fi
