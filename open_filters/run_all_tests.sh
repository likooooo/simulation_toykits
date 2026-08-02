#!/usr/bin/env bash
# One-shot: run all OpenFilters cross-check / parity tests, generate report PNGs,
# write output/index.html, and open it in the browser.
set -uo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TK_ROOT="$(cd "${PKG_ROOT}/.." && pwd)"
CC_DIR="${PKG_ROOT}/crosscheck"
OUT_DIR="${PKG_ROOT}/output"

export SIMULATION_ARTIFACTS_DIR="${SIMULATION_ARTIFACTS_DIR:-${TK_ROOT}/.simulation_toolkits}"
export SIMULATION_DATABASE_DIR="${SIMULATION_DATABASE_DIR:-${TK_ROOT}/simulation_core/assets/database}"
export PYTHONPATH="${PKG_ROOT}:${CC_DIR}:${TK_ROOT}/simulation_core/simulation_plugins:${SIMULATION_ARTIFACTS_DIR}:${PYTHONPATH:-}"

cd "${PKG_ROOT}"

if [[ ! -f "${SIMULATION_DATABASE_DIR}/of/materials/TiO2.yml" ]]; then
  echo "=== exporting of materials ==="
  python3 "${TK_ROOT}/simulation_core/assets/database/of/update_current_database.py"
fi

echo ""
echo "=== [1/3] cross-check tests ==="
set +e
python3 -m unittest -v \
  "${CC_DIR}/test_material_alignment.py" \
  "${CC_DIR}/test_thickness_derivatives.py"
CROSS_RC=$?
set -e

echo ""
echo "=== [2/3] refinement / LM parity tests ==="
set +e
python3 -m unittest -v \
  refinement.tests.test_lm_smoke \
  refinement.tests.test_refinement_parity
PARITY_RC=$?
set -e

echo ""
echo "=== [3/3] generating all report figures ==="
python3 "${PKG_ROOT}/plot_all_reports.py" --out-dir "${OUT_DIR}" --open --skip-nk-map
PLOT_RC=$?

echo ""
echo "=== summary ==="
echo "  cross-check exit code : ${CROSS_RC}  (expected failures on 6-layer derivatives are OK)"
echo "  parity tests exit code: ${PARITY_RC}"
echo "  plot reports exit code: ${PLOT_RC}"
echo "  report gallery        : ${OUT_DIR}/index.html"
echo ""

# List all PNGs
find "${OUT_DIR}" -maxdepth 1 -name '*.png' | sort | while read -r f; do
  echo "  $(basename "$f")"
done

if [[ ${PLOT_RC} -ne 0 ]]; then
  exit "${PLOT_RC}"
fi
if [[ ${PARITY_RC} -ne 0 ]]; then
  exit "${PARITY_RC}"
fi
if [[ ${CROSS_RC} -ne 0 ]]; then
  exit "${CROSS_RC}"
fi
exit 0
