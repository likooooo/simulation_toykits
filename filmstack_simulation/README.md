# filmstack_simulation

Streamlit module for filmstack bar charts, 2D R/T/Psi/Delta maps, and 1D slices.

Copy this directory into another Streamlit project; provide `simulation.so` and a materials dictionary (or inline `n k` in formulas).

## Public API

```python
from filmstack_simulation.page import PageContext, render_page

render_page(
    context=PageContext(
        get_materials_db=your_get_materials_callable,
        sim_wl_from=0.38,
        sim_wl_to=0.78,
    ),
    materials_db=your_materials_dict,
)
```

`materials_db` maps names to `simulation.database_material` objects.

This package does **not** import host modules (`pages/`, `core/`). Runtime loads artifacts via `SIMULATION_ARTIFACTS_DIR` (`simulation.so` + `py_core_plugins/` + `simulation_plugins/`). Wire workspace/materials from the app layer.

## Module layout

| File | Role |
|------|------|
| `page.py` | `PageContext` + `render_page()` — presets + three-section UI |
| `page_widgets.py` | Shared widgets and session helpers (ranges, preset formula, polarization) |
| `materials.py` | `DEFAULT_MATERIAL_PATH_KEYS` + recommended sim wl constants |
| `presets.py` | Seven TMM-aligned presets + formula builder |
| `simulation.py` | TMM primitives, 2D maps, 1D slices |
| `sweep.py` | C++ batch wavelength/angle sweeps |
| `plots.py` | Fixed-width matplotlib display |
| `design_tokens.css` | Design tokens (sync with `ui/design_tokens.css`) |
| `filmstack_optimization/` | Freehand optimization UI; solver in `filmstack_optimization_utils` |

Visualization primitives: `simulation_plugins/filmstack_visualizer.py`.

## Freehand optimization

Host page: `pages/filmstack_toolkits/freehand optimization.py` → `filmstack_optimization.local_search.page.render_page`.

Solver: `simulation_plugins/filmstack_optimization_utils.py` (import after `import simulation`).

## Deploy after editing simulation_core

Streamlit reads `.simulation_core/py_core_plugins/` and `simulation_plugins/`, not the source tree. Re-run deploy or restart Streamlit after plugin edits.

```bash
python scripts/deploy.py local
```

## Host integration (simulation_toykits)

**Current wiring**

- Simulation Database page (`pages/filmstack_toolkits/simulation database.py`) → `render_page()` → `ensure_workspace_initialized(sim_db)` (AM1.5G spectrum only; materials added manually).
- Filmstack page (`pages/filmstack_toolkits/filmstack simulation.py`) → `common.render_filmstack_host()` → reads `get_workspace_materials()` from session.

**Optional pre-load** — pass recommended material paths before Filmstack presets that need workspace entries:

```python
from simulation_database.database_ui import prepare_simulation_database, ensure_simulation_database_initialized
from simulation_database.workspace import ensure_workspace_initialized
from filmstack_simulation.materials import DEFAULT_MATERIAL_PATH_KEYS

prepare_simulation_database()
sim_db = ensure_simulation_database_initialized()
ensure_workspace_initialized(sim_db, material_path_keys=DEFAULT_MATERIAL_PATH_KEYS)
```

**Filmstack page glue** (simplified; see `common.render_filmstack_host`):

```python
from simulation_database.workspace import ensure_sim_workspace_ui, get_workspace_materials
from filmstack_simulation.page import PageContext, render_page

ui = ensure_sim_workspace_ui()
render_page(
    context=PageContext(
        get_materials_db=get_workspace_materials,
        sim_wl_from=ui.sim_wl_from,
        sim_wl_to=ui.sim_wl_to,
    ),
    materials_db=get_workspace_materials(),
)
```

## Formula syntax

See [docs/filmstack_formula_usage.md](../docs/filmstack_formula_usage.md).
