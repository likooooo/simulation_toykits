# ui/

Shared design tokens for Streamlit pages in this monorepo.

## `design_tokens.css`

Single source of truth for CSS custom properties (colors, spacing, typography). Host pages read this file once per rerun and pass the contents as `tokens_css` into:

- Page shell styles (`inject_*_styles`)
- Custom component iframes (`simulation_db_panel`, `panel_section_head`)

Portable packages (`simulation_database`, `filmstack_simulation`) do **not** ship a copy. The host app provides `tokens_path: Path` pointing at its own `design_tokens.css`.

## Host wiring

```python
from pathlib import Path

from common import HOST_DESIGN_TOKENS_PATH  # simulation_toykits only

# simulation_database
from simulation_database.page import render_page
render_page(tokens_path=HOST_DESIGN_TOKENS_PATH)

# filmstack_simulation
from filmstack_simulation.page import PageContext, render_page
render_page(
    context=PageContext(..., tokens_path=HOST_DESIGN_TOKENS_PATH),
    materials_db=...,
)
```

When copying a package into another project, place `design_tokens.css` in the host app and pass its path through the package API.
