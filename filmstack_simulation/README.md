# filmstack_simulation

Streamlit module for filmstack bar charts, 2D R/T/Psi/Delta maps, and 1D slices.

Copy this directory into another Streamlit project; provide `simulation.so`, a materials dictionary (or inline `n k` in formulas), and a host `design_tokens.css` path.

## Public API

```python
from pathlib import Path

from filmstack_simulation.page import PageContext, render_page

render_page(
    context=PageContext(
        get_materials_db=your_get_materials_callable,
        preset_catalog=your_preset_catalog,
        sim_wl_from=0.38,
        sim_wl_to=0.78,
        tokens_path=Path("path/to/design_tokens.css"),
    ),
    materials_db=your_materials_dict,
)
```

`materials_db` maps names to `simulation.database_material` objects.

`tokens_path` is required. The package reads it once per rerun and injects tokens into the page shell and the bundled `panel_section_head` component iframe. When copying into another project, provide your own `design_tokens.css` (see `ui/design_tokens.css` in simulation_toykits for reference).

This package does **not** import host modules (`pages/`, `core/`). Runtime loads artifacts via `SIMULATION_ARTIFACTS_DIR` (`simulation.so` + `py_core_plugins/` + `simulation_plugins/`). Wire workspace/materials and UI assets from the app layer.

## Module layout

| File | Role |
|------|------|
| `page.py` | `PageContext` + `render_page()` — presets + three-section UI |
| `page_widgets.py` | Shared widgets and session helpers (ranges, preset formula, polarization) |
| `presets.py` | Seven TMM-aligned presets + formula builder |
| `simulation.py` | TMM primitives, 2D maps, 1D slices |
| `sweep.py` | C++ batch wavelength/angle sweeps |
| `plots.py` | Fixed-width matplotlib display |
| `filmstack_optimization/` | Freehand optimization UI; solver in `filmstack_optimization_utils` |

Visualization primitives: `simulation_plugins/filmstack_visualizer.py`.

## Freehand optimization

Host page: `pages/filmstack_toolkits/freehand optimization.py` → `filmstack_optimization.local_search.page.render_page`.

Solver: `simulation_plugins/filmstack_optimization_utils.py` (import after `import simulation`).

## Deploy after editing simulation_core

Streamlit reads `.simulation_core/py_core_plugins/` and `simulation_plugins/`, not the source tree. Re-run deploy or restart Streamlit after plugin edits.

```bash
python scripts/build_toykits.py local
```

## Host integration (simulation_toykits)

**Current wiring**

- **simulation_toykits**：`app.py` 启动时预加载 `DEFAULT_MATERIAL_PATH_KEYS` + [`DEFAULT_SPECTRUM_PATH`](../toykits_config.py)（AM1.5G）；Simulation Database 页传入 `tokens_path=HOST_DESIGN_TOKENS_PATH`。
- **Standalone 宿主**：Simulation Database 页 `render_page(tokens_path=...)` 后工作区初始为空；Filmstack 预设所需材料须手动加入，或在下方示例中传入 `material_path_keys` / `spectrum_path_keys`。
- Filmstack page → `common.render_filmstack_host()` → `get_workspace_materials()` + `PageContext.tokens_path`。

**Optional pre-load** — pass recommended material and spectrum paths before Filmstack presets that need workspace entries:

```python
import simulation_database_parser as sdp
from simulation_database.workspace import ensure_workspace_initialized
from toykits_config import DEFAULT_MATERIAL_PATH_KEYS, DEFAULT_SPECTRUM_PATH

sim_db = sdp.get_simulation_database(init=True)
ensure_workspace_initialized(
    sim_db,
    material_path_keys=DEFAULT_MATERIAL_PATH_KEYS,
    spectrum_path_keys=[DEFAULT_SPECTRUM_PATH],
)
```

**Filmstack page glue** (simplified; see `common.render_filmstack_host`):

```python
from common import HOST_DESIGN_TOKENS_PATH
from simulation_database.workspace import ensure_sim_workspace_ui, get_workspace_materials
from filmstack_simulation.page import PageContext, render_page

ui = ensure_sim_workspace_ui()
render_page(
    context=PageContext(
        get_materials_db=get_workspace_materials,
        preset_catalog=your_preset_catalog,
        sim_wl_from=ui.sim_wl_from,
        sim_wl_to=ui.sim_wl_to,
        tokens_path=HOST_DESIGN_TOKENS_PATH,
    ),
    materials_db=get_workspace_materials(),
)
```

## Formula syntax

See [docs/filmstack_formula_usage.md](../docs/filmstack_formula_usage.md).
