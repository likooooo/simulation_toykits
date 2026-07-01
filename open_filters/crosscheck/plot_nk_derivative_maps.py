"""Phase 3: n-k 2D derivative maps (OpenFilters / Simulation / Diff)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bootstrap_simulation import bootstrap_toykits_session
from openfilters_derivatives import POL_TE, openfilters_thickness_rt_derivatives_constant_layer
from paths import crosscheck_dir
from simulation_derivatives import Polarization, simulation_thickness_rt_derivatives_constant_layer
from stack_spec import DEFAULT_BRAGG_STACK, load_default_materials_db


def compute_nk_derivative_maps(
    *,
    layer_index: int = 0,
    wl_nm: float = 550.0,
    angle_deg: float = 0.0,
    pol: str = Polarization.TE,
    n_grid: np.ndarray | None = None,
    k_grid: np.ndarray | None = None,
    quantity: str = "R",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bootstrap_toykits_session()
    materials_db = load_default_materials_db()
    spec = DEFAULT_BRAGG_STACK
    if n_grid is None:
        n_grid = np.linspace(1.8, 2.4, 40)
    if k_grid is None:
        k_grid = np.linspace(0.0, 0.05, 40)

    of_pol = POL_TE if pol == Polarization.TE else 0.0
    of_map = np.zeros((len(k_grid), len(n_grid)), dtype=float)
    sim_map = np.zeros_like(of_map)

    for ik, k in enumerate(k_grid):
        for jn, n in enumerate(n_grid):
            nk = complex(n, k)
            dr, dt = openfilters_thickness_rt_derivatives_constant_layer(
                spec, materials_db, layer_index, nk, [wl_nm], angle_deg, of_pol
            )
            sr, st = simulation_thickness_rt_derivatives_constant_layer(
                spec, materials_db, layer_index, nk, wl_nm, angle_deg, pol
            )
            if quantity.upper() == "T":
                of_map[ik, jn] = dt * 1000.0  # nm -> um
                sim_map[ik, jn] = st
            else:
                of_map[ik, jn] = dr * 1000.0
                sim_map[ik, jn] = sr

    diff_map = sim_map - of_map
    return n_grid, k_grid, of_map, sim_map, diff_map


def plot_nk_derivative_maps(
    out_path: Path,
    *,
    layer_index: int = 0,
    wl_nm: float = 550.0,
    angle_deg: float = 0.0,
    pol: str = Polarization.TE,
    quantity: str = "R",
    grid_size: int = 40,
) -> Path:
    n_grid = np.linspace(1.8, 2.4, grid_size)
    k_grid = np.linspace(0.0, 0.05, grid_size)
    n_grid, k_grid, of_map, sim_map, diff_map = compute_nk_derivative_maps(
        layer_index=layer_index,
        wl_nm=wl_nm,
        angle_deg=angle_deg,
        pol=pol,
        n_grid=n_grid,
        k_grid=k_grid,
        quantity=quantity,
    )

    extent = [n_grid[0], n_grid[-1], k_grid[0], k_grid[-1]]
    vmax = max(np.max(np.abs(of_map)), np.max(np.abs(sim_map)), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    titles = ["OpenFilters", "Simulation", "Diff (Sim - OF)"]
    data = [of_map, sim_map, diff_map]
    for ax, title, arr in zip(axes, titles, data):
        if title.startswith("Diff"):
            lim = np.max(np.abs(arr))
            im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("k")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"d{quantity}/dd layer {layer_index} @ {wl_nm} nm, {angle_deg}°, {pol}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot n-k derivative maps.")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--wl-nm", type=float, default=550.0)
    parser.add_argument("--angle-deg", type=float, default=0.0)
    parser.add_argument("--pol", default=Polarization.TE)
    parser.add_argument("--quantity", choices=["R", "T"], default="R")
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument(
        "--out",
        type=Path,
        default=crosscheck_dir() / "output" / "nk_derivative_map.png",
    )
    args = parser.parse_args()
    plot_nk_derivative_maps(
        args.out,
        layer_index=args.layer,
        wl_nm=args.wl_nm,
        angle_deg=args.angle_deg,
        pol=args.pol,
        quantity=args.quantity,
        grid_size=args.grid,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
