# open_filters

OpenFilters ↔ simulation TMM cross-check and LM refinement for `simulation_toykits`.

## Layout

```
open_filters/
├── paths.py
├── moremath/                # vendored Levenberg–Marquardt + QR
├── refinement/              # thickness LM refinement (abeles + simulation backends)
├── run_crosscheck.sh        # derivative cross-check tests
├── run_refinement_parity.sh # LM parity: abeles vs simulation adjoint
├── run_all_tests.sh         # one-shot tests
├── plot_all_reports.py      # aggregate plots/reports
└── crosscheck/
```

Material YAML: `simulation_core/assets/database/of/materials/`

## Quick start

```bash
# Export materials (once)
python simulation_core/assets/database/of/update_current_database.py

# Derivative cross-check (requires .simulation_toolkits/simulation.so)
./open_filters/run_crosscheck.sh

# LM refinement parity (abeles vs simulation, same LM)
./open_filters/run_refinement_parity.sh
# PNG output: open_filters/refinement/output/parity_*.png

# Or run everything:
./open_filters/run_all_tests.sh
```

See [crosscheck/README.md](crosscheck/README.md).
