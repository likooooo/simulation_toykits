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

TESTS=(
  "${CC_DIR}/test_material_alignment.py"
  "${CC_DIR}/test_thickness_derivatives.py"
)

if [[ "${1:-}" == "--maps" ]]; then
  shift || true
  python3 "${CC_DIR}/plot_nk_derivative_maps.py" \
    --out "${CC_DIR}/output/nk_derivative_R_layer0_wl550.png"
  TESTS+=("${CC_DIR}/test_nk_derivative_maps.py")
fi

python3 -m unittest -v "${TESTS[@]}" "$@"
