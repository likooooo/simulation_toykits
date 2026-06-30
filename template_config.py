"""Load filmstack templates from JSON.

Template definitions live in ``filmstack_templates.json`` (repo root). Each entry:

- ``id`` / ``label``: preset identity; FreeSnell examples use ``fs_`` label prefix.
- ``stack``: ``{type: formula, formula: ...}`` — numeric filmstack build instruction (v1 syntax).
- ``sim``: UI defaults for preset switch (non-null → template value; null → page default).
  Shared with ``build_freesnell_compare_ui.run_toykits`` for angle/wavelength/polarization.
- ``material_path_keys`` / ``required_material_names``: workspace preload for this template.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from filmstack_simulation.presets import FilmstackPreset, PresetCatalog
from filmstack_simulation.template_types import FilmstackSimParams, FilmstackTemplate

TEMPLATES_JSON_PATH = Path(__file__).resolve().parent / "filmstack_templates.json"


def polarization_from_quantity(quantity: str) -> str:
    """Map compare ``quantity`` to UI polarization (TE / TM / UNPOLARIZED)."""
    if quantity in ("R_p", "T_p"):
        return "TM"
    if quantity in ("R_s", "T_s"):
        return "TE"
    return "UNPOLARIZED"


def quantity_from_rt(
    r_s: np.ndarray,
    t_s: np.ndarray,
    r_p: np.ndarray,
    t_p: np.ndarray,
    quantity: str,
) -> np.ndarray:
    """Map polarized R/T arrays to a scalar quantity (compare / baseline alignment)."""
    if quantity == "T":
        return 0.5 * (t_s + t_p)
    if quantity == "R":
        return 0.5 * (r_s + r_p)
    if quantity == "T_s":
        return t_s
    if quantity == "T_p":
        return t_p
    if quantity == "R_s":
        return r_s
    if quantity == "R_p":
        return r_p
    if quantity == "neg_log_T":
        t = 0.5 * (t_s + t_p)
        return -np.log(np.maximum(t, 1e-30))
    raise ValueError(f"unknown quantity: {quantity}")


def target_ang_deg_from_sim(raw: Mapping[str, Any], *, default: float = 0.0) -> float:
    sim = raw.get("sim") or {}
    value = sim.get("target_ang_deg")
    return float(value) if value is not None else default


def target_wl_um_from_sim(raw: Mapping[str, Any], *, default: float = 0.55) -> float:
    sim = raw.get("sim") or {}
    value = sim.get("target_wl_um")
    return float(value) if value is not None else default


def validate_fs_sim_for_compare(raw: Mapping[str, Any], *, x_axis: str) -> None:
    """Ensure template JSON sim has fields required by compare / UI reproduction."""
    preset_id = str(raw.get("id", ""))
    sim = raw.get("sim") or {}
    if sim.get("polarization") is None:
        raise ValueError(f"{preset_id}: sim.polarization is required for fs_* templates")
    if x_axis == "angle_deg":
        if sim.get("target_wl_um") is None:
            raise ValueError(f"{preset_id}: sim.target_wl_um required for angle sweep")
        if sim.get("ang_from_deg") is None or sim.get("ang_to_deg") is None:
            raise ValueError(f"{preset_id}: sim.ang_from_deg/to required for angle sweep")
        return
    if sim.get("target_ang_deg") is None:
        raise ValueError(f"{preset_id}: sim.target_ang_deg required for wavelength sweep")


def _parse_sim(raw: Mapping[str, Any] | None) -> FilmstackSimParams:
    if not raw:
        return FilmstackSimParams()
    return FilmstackSimParams(
        wl_from_um=raw.get("wl_from_um"),
        wl_to_um=raw.get("wl_to_um"),
        ang_from_deg=raw.get("ang_from_deg"),
        ang_to_deg=raw.get("ang_to_deg"),
        target_wl_um=raw.get("target_wl_um"),
        target_ang_deg=raw.get("target_ang_deg"),
        polarization=raw.get("polarization"),
    )


def _preset_from_stack(raw: Mapping[str, Any], *, preset_id: str) -> FilmstackPreset:
    stack_type = raw.get("type", "formula")
    if stack_type != "formula":
        raise ValueError(f"stack.type must be 'formula' (preset {preset_id}), got {stack_type!r}")
    formula = str(raw.get("formula") or "").strip()
    if not formula:
        raise ValueError(f"stack.formula is empty (preset {preset_id})")
    return FilmstackPreset(id=preset_id, label="", formula=formula)


def _template_from_json(raw: Mapping[str, Any]) -> FilmstackTemplate:
    preset_id = str(raw["id"])
    preset = _preset_from_stack(raw["stack"], preset_id=preset_id)
    preset = FilmstackPreset(
        id=preset.id,
        label=str(raw.get("label") or preset_id),
        formula=preset.formula,
    )
    paths = tuple(tuple(p) for p in raw.get("material_path_keys") or [])
    required = frozenset(str(x) for x in raw.get("required_material_names") or [])
    return FilmstackTemplate(
        preset=preset,
        sim=_parse_sim(raw.get("sim")),
        material_path_keys=paths,
        required_material_names=required,
        notes=str(raw.get("notes") or ""),
        incoherent=bool(raw.get("incoherent")),
    )


@lru_cache(maxsize=1)
def load_templates_json(path: Path | None = None) -> Mapping[str, Any]:
    json_path = path or TEMPLATES_JSON_PATH
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_all_templates(path: Path | None = None) -> tuple[FilmstackTemplate, ...]:
    doc = load_templates_json(path)
    return tuple(_template_from_json(item) for item in doc["templates"])


def is_incoherent_template(
    template_id: str,
    *,
    template_json: Mapping[str, Any] | None = None,
    label: str = "",
    invoke: str = "",
) -> bool:
    """True when compare/UI should treat the template as incoherent (non-TMM layer stack)."""
    if template_json and template_json.get("incoherent"):
        return True
    return (
        template_id.endswith("_inc")
        or template_id.endswith("_ni")
        or "非相干" in label
        or "-inc" in invoke
    )


@lru_cache(maxsize=1)
def incoherent_template_ids() -> frozenset[str]:
    doc = load_templates_json()
    ids: set[str] = set()
    for raw in doc["templates"]:
        tid = str(raw["id"])
        if is_incoherent_template(
            tid,
            template_json=raw,
            label=str(raw.get("label") or ""),
        ):
            ids.add(tid)
    return frozenset(ids)


INCOHERENT_TEMPLATE_IDS = incoherent_template_ids()


def default_preset_id(path: Path | None = None) -> str:
    return str(load_templates_json(path).get("default_preset_id") or "ar_qw_si")


def template_by_id(templates: Iterable[FilmstackTemplate]) -> Dict[str, FilmstackTemplate]:
    return {t.preset.id: t for t in templates}


def build_preset_catalog(
    templates: Iterable[FilmstackTemplate],
    *,
    default_preset_id: str,
) -> PresetCatalog:
    presets = tuple(t.preset for t in templates)
    return PresetCatalog(presets=presets, default_preset_id=default_preset_id)


def aggregate_material_path_keys(templates: Iterable[FilmstackTemplate]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for tpl in templates:
        for path in tpl.material_path_keys:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                out.append(list(path))
    return out


def aggregate_required_material_names(templates: Iterable[FilmstackTemplate]) -> frozenset[str]:
    names: set[str] = set()
    for tpl in templates:
        names.update(tpl.required_material_names)
    return frozenset(names)
