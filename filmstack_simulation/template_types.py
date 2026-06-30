"""Filmstack template types (configs live in filmstack_templates.json)."""

from __future__ import annotations

from dataclasses import dataclass, field

from filmstack_simulation.presets import FilmstackPreset


@dataclass(frozen=True)
class FilmstackSimParams:
    """Filmstack UI inputs; None on preset switch → page default value."""

    wl_from_um: float | None = None
    wl_to_um: float | None = None
    ang_from_deg: float | None = None
    ang_to_deg: float | None = None
    target_wl_um: float | None = None
    target_ang_deg: float | None = None
    polarization: str | None = None


@dataclass(frozen=True)
class FilmstackTemplate:
    preset: FilmstackPreset
    sim: FilmstackSimParams = field(default_factory=FilmstackSimParams)
    material_path_keys: tuple[tuple[str, ...], ...] = ()
    required_material_names: frozenset[str] = frozenset()
    notes: str = ""
    incoherent: bool = False


@dataclass(frozen=True)
class FilmstackUIApplySpec:
    """Session keys to refresh when applying a template."""

    wl_from_key: str
    wl_to_key: str
    polarization_key: str
    ang_from_key: str | None = None
    ang_to_key: str | None = None
    target_wl_key: str | None = None
    target_ang_key: str | None = None
    fixed_angle_key: str | None = None


@dataclass(frozen=True)
class FilmstackUIDefaults:
    """Page-level widget defaults when template.sim field is null."""

    wl_from: float
    wl_to: float
    polarization: str
    ang_from: float = 0.0
    ang_to: float = 60.0
    target_wl: float = 0.55
    target_ang: float = 0.0
    fixed_angle: float = 0.0
    formula: str = ""
