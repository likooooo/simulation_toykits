# OpenFilters ↔ simulation TMM thickness derivative cross-check

Compare **OpenFilters abeles** analytic thickness derivatives (`dR/dd`, `dT/dd`) with
**simulation** TMM adjoint (`thickness_reflectance_adjoint` / `thickness_transmittance_adjoint`) on identical film stacks.

## Prerequisites

1. Built runtime: `simulation_toykits/.simulation_toolkits/simulation.so`
2. OpenFilters source under `$GENERATE_GOLDEN_TOOLS_DIR/OpenFilters` (default `~/repos/simulation_baseline_tools/OpenFilters`)
3. Exported materials: `simulation_core/assets/database/of/materials/*.yml`

```bash
python simulation_core/assets/database/of/update_current_database.py
# or: python simulation_core/assets/database/update_all.py --only openfilters
```

## Environment

| Variable | Default |
|----------|---------|
| `SIMULATION_ARTIFACTS_DIR` | `simulation_toykits/.simulation_toolkits` |
| `SIMULATION_DATABASE_DIR` | `simulation_core/assets/database` |
| `GENERATE_GOLDEN_TOOLS_DIR` | `~/repos/simulation_baseline_tools` (OpenFilters at `$GENERATE_GOLDEN_TOOLS_DIR/OpenFilters`) |

## Run

```bash
cd simulation_toykits/open_filters
./run_crosscheck.sh          # Phase 1–2
./run_crosscheck.sh --maps   # + n–k 三子图 + Phase 3 测试
```

## Conventions

- **Layer order:** abeles substrate→medium; simulation incident→exit (`openfilters_derivatives.py` reverses indices)
- **nk sign:** simulation `n+ik` → abeles `n-ik` via `simulation_nk_to_abeles`
- **Units:** `dR/d(depth_um) = dR/d(thickness_nm) × 1000`
