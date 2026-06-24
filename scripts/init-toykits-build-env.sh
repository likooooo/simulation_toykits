#!/usr/bin/env bash
# Toykits runtime env (source before pytest/streamlit).
# Usage: source scripts/init-toykits-build-env.sh

_IS_SOURCED=0
[[ "${BASH_SOURCE[0]}" != "${0}" ]] && _IS_SOURCED=1
if (( _IS_SOURCED )); then
  _INIT_SAVED_OPTS="$(set +o)"
fi

set -euo pipefail

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ARTIFACTS="${_REPO}/.simulation_core"
# shellcheck source=../simulation_core/scripts/init-simulation-build-env.sh
source "${_REPO}/simulation_core/scripts/init-simulation-build-env.sh" "${_ARTIFACTS}"

export SIMULATION_DATABASE_DIR="${_REPO}/simulation_core/assets/database"
export SIMULATION_TMM_ASSETS_DIR="${_REPO}/simulation_core/assets/ipynb/simulation/TMM"
export PYTHONPATH="${_REPO}:${SIMULATION_ARTIFACTS_DIR}"
export LD_LIBRARY_PATH="${SIMULATION_ARTIFACTS_DIR}:${LD_LIBRARY_PATH:-}"

if (( _IS_SOURCED )); then
  eval "${_INIT_SAVED_OPTS}"
fi
