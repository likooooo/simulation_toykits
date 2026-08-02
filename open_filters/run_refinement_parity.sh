#!/usr/bin/env bash
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TK_ROOT="$(cd "${PKG_ROOT}/.." && pwd)"
CC_DIR="${PKG_ROOT}/crosscheck"

export SIMULATION_ARTIFACTS_DIR="${SIMULATION_ARTIFACTS_DIR:-${TK_ROOT}/.simulation_toolkits}"
export SIMULATION_DATABASE_DIR="${SIMULATION_DATABASE_DIR:-${TK_ROOT}/simulation_core/assets/database}"
export PYTHONPATH="${PKG_ROOT}:${CC_DIR}:${TK_ROOT}/simulation_core/simulation_plugins:${SIMULATION_ARTIFACTS_DIR}:${PYTHONPATH:-}"

cd "${PKG_ROOT}"

if [[ ! -f "${SIMULATION_DATABASE_DIR}/of/materials/TiO2.yml" ]]; then
  echo "Exporting of materials..."
  python3 "${TK_ROOT}/simulation_core/assets/database/of/update_current_database.py"
fi

python3 -m unittest -v \
  refinement.tests.test_lm_smoke \
  refinement.tests.test_refinement_parity \
  "$@"

python3 "${PKG_ROOT}/refinement/plot_parity_reports.py" --out-dir "${PKG_ROOT}/refinement/output"
