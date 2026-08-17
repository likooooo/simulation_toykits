# filmstack_simulation

Streamlit module for filmstack bar charts, 2D R/T/Psi/Delta maps, and 1D slices.

Copy this directory into another Streamlit project; provide `simulation.so`, a materials dictionary (or inline `n k` in formulas), and a host `design_tokens.css` path. Nested `filmstack_optimization/` must be copied together (it imports `filmstack_simulation.*`).

## Public API

```python
from pathlib import Path

# Eager-load plugins from SIMULATION_ARTIFACTS_DIR before business imports.
import simulation  # noqa: F401

from filmstack_simulation.page import PageContext, render_page

render_page(
    context=PageContext(
        get_materials_db=your_get_materials_callable,
        preset_catalog=your_preset_catalog,
        template_by_id=your_template_map,
        recommended_wl_from=0.38,
        recommended_wl_to=0.78,
        initial_preset_id=your_default_preset_id,
        initial_formula=your_initial_formula,
        tokens_path=Path("path/to/design_tokens.css"),
    ),
    materials_db=your_materials_dict,
)
```

`materials_db` maps names to `simulation.material_s` objects.

`tokens_path` is required. The package reads it once per rerun and injects tokens into the page shell and the bundled `panel_section_head` component iframe. When copying into another project, provide your own `design_tokens.css` (see `ui/design_tokens.css` in simulation_toykits for reference).

This package does **not** import host modules (`pages/`, `core/`, `ui/`). Native artifacts and pruned plugins come from `SIMULATION_ARTIFACTS_DIR` (`.simulation_toolkits/` in this repo). After `import simulation`, C++ adds `simulation_plugins/` to `sys.path` so `import filmstack_visualizer` works; putting only the artifact root on `PYTHONPATH` is not enough for plugin modules. Wire workspace/materials and UI assets from the app layer.

## Module layout

| File | Role |
|------|------|
| `page.py` | `render_page()` — 构建指令、参数区、结构图、n/k 曲线、2D 谱图与切片 |
| `page_shell.py` | `PageContext` + bootstrap / session helpers |
| `page_widgets.py` | Shared widgets and session helpers (ranges, preset formula, polarization) |
| `page_styles.py` | Page CSS helpers |
| `help_texts.py` | Formula / UI help strings |
| `template_types.py` | Template-related types |
| `presets.py` | Preset catalog types; formulas live in host `filmstack_templates.json` |
| `simulation.py` | TMM primitives, 2D maps, 1D slices |
| `sweep.py` | C++ batch wavelength/angle sweeps |
| `plots.py` | Fixed-width matplotlib display |
| `filmstack_optimization/` | Freehand optimization UI; solver in `filmstack_optimization_utils` |

Visualization primitives: `simulation_plugins/filmstack_visualizer.py`.

## Freehand optimization

Host page: `pages/filmstack_toolkits/freehand optimization.py` → `filmstack_optimization.local_search.page.render_page`.

Solver: `simulation_plugins/filmstack_optimization_utils.py` (import after `import simulation`). After editing the solver plugin, re-deploy artifacts (`python scripts/build_toykits.py` or `--toolkits`); `local` alone does not sync plugins.

## Deploy after editing simulation_core

Streamlit reads `.simulation_toolkits/py_core_plugins/` and `simulation_plugins/`, not the source tree. Re-sync plugins after edits, then start Streamlit if needed:

```bash
python scripts/build_toykits.py              # or: --toolkits
# optional UI only (does not collect/sync plugins):
python scripts/build_toykits.py local
```

## Host integration (simulation_toykits)

**Current wiring**

- **simulation_toykits**：`app.py` 启动时预加载 `common.get_default_material_path_keys()` + [`DEFAULT_SPECTRUM_PATH`](../toykits_config.py)（AM1.5G）；Simulation Database 页须传入 `tokens_path` 与 `material_path_keys` / `required_material_names` / `spectrum_path_keys`（任一为 `None` 会 `ValueError`）。空工作区请显式传空 list / 空 `frozenset`。
- **Standalone 宿主**：同样三参数必填；依赖 `import simulation` 与 artifact 内已 prune 的插件（见 [scripts/simulation_toykits_deploy.md](../scripts/simulation_toykits_deploy.md) §3.1）。
- Filmstack page → `common.render_filmstack_host()` → `get_workspace_materials()` + `PageContext.tokens_path`。

**Optional pre-load** — pass recommended material and spectrum paths before Filmstack presets that need workspace entries:

```python
import simulation  # noqa: F401
import simulation_database_parser as sdp
from simulation_database.workspace import ensure_workspace_initialized
from common import get_default_material_path_keys, HOST_DESIGN_TOKENS_PATH
from toykits_config import DEFAULT_SPECTRUM_PATH

sim_db = sdp.get_simulation_database(init=True)
ensure_workspace_initialized(
    sim_db,
    material_path_keys=get_default_material_path_keys(),
    spectrum_path_keys=[DEFAULT_SPECTRUM_PATH],
)
```

**Filmstack page glue** (simplified; see `common.render_filmstack_host`):

```python
from common import (
    HOST_DESIGN_TOKENS_PATH,
    build_filmstack_preset_catalog,
    get_filmstack_template_by_id,
)
from simulation_database.workspace import get_workspace_materials
from filmstack_simulation.page import PageContext, render_page
from toykits_config import resolve_filmstack_initial_defaults

catalog = build_filmstack_preset_catalog()
template_map = get_filmstack_template_by_id()
initial = resolve_filmstack_initial_defaults(
    catalog.valid_preset_ids,
    template_by_id=template_map,
)
render_page(
    context=PageContext(
        get_materials_db=get_workspace_materials,
        preset_catalog=catalog,
        template_by_id=template_map,
        recommended_wl_from=initial.wl_from_um,
        recommended_wl_to=initial.wl_to_um,
        initial_preset_id=initial.preset_id,
        initial_formula=initial.formula,
        tokens_path=HOST_DESIGN_TOKENS_PATH,
    ),
    materials_db=get_workspace_materials(),
)
```

## Formula syntax

See [docs/filmstack_formula_usage.md](../docs/filmstack_formula_usage.md).
