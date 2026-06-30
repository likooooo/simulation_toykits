from common import (
    HOST_DESIGN_TOKENS_PATH,
    get_default_material_path_keys,
    get_required_default_material_names,
)
from toykits_config import DEFAULT_SPECTRUM_PATH
from simulation_database.page import render_page

render_page(
    tokens_path=HOST_DESIGN_TOKENS_PATH,
    material_path_keys=get_default_material_path_keys(),
    required_material_names=get_required_default_material_names(),
    spectrum_path_keys=[DEFAULT_SPECTRUM_PATH],
)
