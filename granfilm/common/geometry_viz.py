"""Schematic geometry diagrams for GranFilm compare UI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from granfilm.oblate_prolate.case import SpheroidCase
from granfilm.sphere_island.case import GranFilmCase

_PIPELINE = Literal["sphere", "spheroid"]

_COLOR_VAC = "#e8f4fc"
_COLOR_ISLAND = "#f0c040"
_COLOR_SUB = "#a8a8a8"
_COLOR_COAT = "#90c090"
_COLOR_OUTLINE = "#333333"


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _draw_layer_bar(
    ax: plt.Axes,
    layers: list[tuple[str, float, str]],
    *,
    title: str,
) -> None:
    """Horizontal layer stack: (label, thickness_nm, color)."""
    z = 0.0
    margin = 8.0
    for label, thick, color in layers:
        if thick <= 0:
            continue
        ax.barh(0, thick, left=z, height=0.55, color=color, edgecolor=_COLOR_OUTLINE, linewidth=0.8)
        ax.text(z + thick / 2, 0, f"{label}\n{thick:g} nm", ha="center", va="center", fontsize=7)
        z += thick
    ax.barh(0, margin, left=z, height=0.55, color=_COLOR_SUB, edgecolor=_COLOR_OUTLINE, linewidth=0.8)
    ax.text(z + margin / 2, 0, "substrate\n∞", ha="center", va="center", fontsize=7)
    ax.set_xlim(-margin * 0.3, z + margin * 1.2)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("depth (nm)")
    ax.set_title(title, fontsize=9)
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)


def _truncated_sphere_profile(R: float, tr: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Side-view profile of truncated sphere cap on z=0 plane (x >= 0 half)."""
    z_c = -tr * R
    theta = np.linspace(0, np.pi, n)
    x = R * np.sin(theta)
    z = z_c + R * np.cos(theta)
    mask = z >= -1e-9
    return x[mask], z[mask]


def _draw_truncated_sphere(
    ax: plt.Axes,
    *,
    R: float,
    tr: float,
    island: str,
    substrate: str,
    network: str,
    title: str,
    below: bool = False,
) -> None:
    x_cap, z_cap = _truncated_sphere_profile(R, tr)
    sub_w = max(R * 2.8, 20.0)
    sub_h = R * 0.35
    ax.add_patch(
        mpatches.Rectangle(
            (-sub_w / 2, -sub_h),
            sub_w,
            sub_h,
            facecolor=_COLOR_SUB,
            edgecolor=_COLOR_OUTLINE,
            linewidth=0.8,
        )
    )
    ax.fill_between(x_cap, z_cap, 0.0, color=_COLOR_ISLAND, alpha=0.9)
    ax.plot(x_cap, z_cap, color=_COLOR_OUTLINE, lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.text(0, -sub_h * 0.55, substrate, ha="center", va="center", fontsize=7, color="white")
    note = f"R={R:g} nm, tr={tr:g}"
    if below:
        note += ", below substrate"
    ax.text(
        0.02,
        0.98,
        f"{island} | {network}\n{note}",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )
    ymax = max(float(np.max(z_cap)) if z_cap.size else R, R * 0.3)
    ax.set_xlim(-sub_w / 2, sub_w / 2)
    ax.set_ylim(-sub_h * 1.2, ymax * 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(title, fontsize=9)


def _spheroid_semi_axes(case: SpheroidCase) -> tuple[float, float, str]:
    if case.R_par > case.R_per:
        return case.R_par, case.R_per, "oblate"
    if case.R_par < case.R_per:
        return case.R_par, case.R_per, "prolate"
    return case.R_par, case.R_per, "sphere"


def _draw_spheroid_cross_section(
    ax: plt.Axes,
    *,
    a_horiz: float,
    b_vert: float,
    island: str,
    substrate: str,
    island_type: str,
    title: str,
    coating: str | None = None,
    coat_thick: float = 0.0,
) -> None:
    sub_w = max(a_horiz * 2.8, 20.0)
    sub_h = b_vert * 0.4
    ax.add_patch(
        mpatches.Rectangle(
            (-sub_w / 2, -sub_h),
            sub_w,
            sub_h,
            facecolor=_COLOR_SUB,
            edgecolor=_COLOR_OUTLINE,
            linewidth=0.8,
        )
    )
    t = np.linspace(0, np.pi, 200)
    x = a_horiz * np.sin(t)
    z = b_vert * np.cos(t)
    mask = z >= -1e-9
    x, z = x[mask], z[mask]

    if coating and coat_thick > 0:
        scale = 1.0 + coat_thick / max(a_horiz, b_vert, 1e-6)
        x_o = a_horiz * scale * np.sin(t)
        z_o = b_vert * scale * np.cos(t)
        m = z_o >= -1e-9
        ax.fill_between(x_o[m], z_o[m], 0.0, color=_COLOR_COAT, alpha=0.5)
        ax.plot(x_o[m], z_o[m], color=_COLOR_COAT, lw=1.0, ls="--")

    ax.fill_between(x, z, 0.0, color=_COLOR_ISLAND, alpha=0.9)
    ax.plot(x, z, color=_COLOR_OUTLINE, lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.text(0, -sub_h * 0.55, substrate, ha="center", va="center", fontsize=7, color="white")
    coat_line = f", coat {coating} {coat_thick:g} nm" if coating and coat_thick > 0 else ""
    ax.text(
        0.02,
        0.98,
        f"{island} | {island_type}\nR∥={a_horiz:g} R⊥={b_vert:g} nm{coat_line}",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )
    ymax = max(float(np.max(z)) if z.size else b_vert, b_vert * 0.3)
    ax.set_xlim(-sub_w / 2, sub_w / 2)
    ax.set_ylim(-sub_h * 1.2, ymax * 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(title, fontsize=9)


def _plot_sphere_geometry(case: GranFilmCase, out: Path) -> None:
    geom = case.geometry.strip().lower()
    fig, ax = plt.subplots(figsize=(7, 2.2 if geom in {"film", "2film"} else 3.0))

    if geom == "film":
        _draw_layer_bar(
            ax,
            [
                ("vacuum", 8.0, _COLOR_VAC),
                (case.island, case.film_thickness1, _COLOR_ISLAND),
            ],
            title=f"film: {case.island} on {case.substrate}",
        )
    elif geom == "2film":
        _draw_layer_bar(
            ax,
            [
                ("vacuum", 8.0, _COLOR_VAC),
                (case.island, case.film_thickness1, _COLOR_ISLAND),
                (case.coating or "film2", case.film_thickness2, _COLOR_COAT),
            ],
            title=f"2film: {case.island}/{case.coating} on {case.substrate}",
        )
    else:
        below = case.tr < case.MPpos
        _draw_truncated_sphere(
            ax,
            R=case.R,
            tr=case.tr,
            island=case.island,
            substrate=case.substrate,
            network=case.network,
            title=f"{geom}: truncated sphere on substrate",
            below=below,
        )
    _save(fig, out)


def _plot_spheroid_geometry(case: SpheroidCase, out: Path) -> None:
    geom = case.geometry.strip().lower()
    a_h, b_v, kind = _spheroid_semi_axes(case)
    fig, ax = plt.subplots(figsize=(7, 3.0))
    coat = case.coating if geom == "coated" else None
    coat_t = case.thickness if geom == "coated" else 0.0
    _draw_spheroid_cross_section(
        ax,
        a_horiz=a_h,
        b_vert=b_v,
        island=case.island,
        substrate=case.substrate,
        island_type=kind if geom != "yamaguchi" else "yamaguchi",
        title=f"{geom}: rotational spheroid",
        coating=coat,
        coat_thick=coat_t,
    )
    _save(fig, out)


def plot_geometry_schematic(
    case: GranFilmCase | SpheroidCase,
    *,
    pipeline: _PIPELINE,
    out: Path,
) -> None:
    """Draw geometry schematic for a GranFilm case (sphere or spheroid pipeline)."""
    if pipeline == "sphere":
        if not isinstance(case, GranFilmCase):
            raise TypeError(f"sphere pipeline requires GranFilmCase, got {type(case)}")
        _plot_sphere_geometry(case, out)
    else:
        if not isinstance(case, SpheroidCase):
            raise TypeError(f"spheroid pipeline requires SpheroidCase, got {type(case)}")
        _plot_spheroid_geometry(case, out)
