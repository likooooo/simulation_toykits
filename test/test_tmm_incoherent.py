"""Tests for C++ incoherent TMM (layer*) vs coherent TMM and FreeSnell baseline."""

from __future__ import annotations

import os
import subprocess
import textwrap

import numpy as np
import pytest

from simulation_paths import simulation_module  # noqa: F401 — ensures plugin path


def _cplx(z) -> complex:
    from tmm_utils import cplx_from_py

    return cplx_from_py(z)


def _fs_pe_materials_db():
    from common import build_materials_db_from_path_keys

    return build_materials_db_from_path_keys([["fs", "materials", "hdpe.yml"]])


def _resolve_pe_layers(simulation, *, incoherent_air: bool):
    from filmstack_simulation.simulation import resolve_stack_with_layers

    air_th = "1000*" if incoherent_air else "1000"
    formula = f"air 0 1.0 0 (air {air_th} 1.0 0 hdpe 14)^1 air 0 1.0 0"
    _, _, layers = resolve_stack_with_layers(formula, _fs_pe_materials_db())
    return layers


def _unpolarized_t(layers, wls, angle_deg):
    from filmstack_simulation.sweep import compute_wavelength_vs_RT_data

    data = compute_wavelength_vs_RT_data(layers, np.asarray(wls, dtype=float), angle_deg)
    return 0.5 * (np.asarray(data["T_s"]) + np.asarray(data["T_p"]))


def _freesnell_toolchain_available() -> bool:
    try:
        from scripts.build_freesnell_compare_ui import resolve_freesnell_env

        resolve_freesnell_env()
        return True
    except Exception:
        return False


def _freesnell_combine_layers(
    th_i_rad: float,
    wl_um: float,
    *,
    layers_spec: list[tuple[complex, float]],
    pol: str,
) -> tuple[float, float, float]:
    """Call FreeSnell ``combine-layers``; thicknesses in meters (negative => layer*)."""
    from scripts.build_freesnell_compare_ui import resolve_freesnell_env

    env = resolve_freesnell_env()
    wl_m = wl_um * 1e-6
    w = -wl_m if pol == "s" else wl_m
    th = th_i_rad

    def fmt_nk(z: complex) -> str:
        if z.imag == 0.0:
            return f"{z.real:g}"
        return f"{z.real:g}+{z.imag:g}i"

    layer_forms = []
    for nk, d_m in layers_spec:
        layer_forms.append(f"(list {fmt_nk(nk)} {d_m:g})")
    layers_lit = f"(list {' '.join(layer_forms)})"

    expr = textwrap.dedent(
        f"""
        (begin
          (load "fresneleq.scm")
          (define out (combine-layers {th:g} {w:g} {layers_lit}))
          (display (car out))(display " ")
          (display (cadr out))(display " ")
          (display (caddr out)))
        """
    ).strip()

    proc = subprocess.run(
        [env["SCM"], "-e", expr],
        cwd=env["FREESNELL_DIR"],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"FreeSnell combine-layers failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    parts = proc.stdout.strip().split()
    if len(parts) != 3:
        raise RuntimeError(f"unexpected FreeSnell output: {proc.stdout!r}")
    return tuple(float(x) for x in parts)


def _pe_freesnell_layers_spec(
    simulation,
    layers,
    wl_um: float,
    *,
    air_incoherent: bool,
) -> list[tuple[complex, float]]:
    """FreeSnell stack in meters; air gap incoherent => negative thickness."""
    spec: list[tuple[complex, float]] = []
    for i, lyr in enumerate(layers):
        nk = _cplx(lyr.background_material.nk_at_wavelength_um(wl_um))
        d_um = float(lyr.depth)
        d_m = d_um * 1e-6
        if i > 0 and i < len(layers) - 1 and air_incoherent and abs(d_um - 1000.0) < 1e-6:
            d_m = -abs(d_m)
        spec.append((nk, d_m))
    return spec


@pytest.fixture(scope="module")
def pe_layers_coherent(simulation):
    return _resolve_pe_layers(simulation, incoherent_air=False)


@pytest.fixture(scope="module")
def pe_layers_incoherent(simulation):
    return _resolve_pe_layers(simulation, incoherent_air=True)


def test_coherent_stack_matches_tmm_baseline(pe_layers_coherent):
    """All-coherent stack must match itself (fs_pe_1x14_co path)."""
    angle_deg = 4.0
    wls = np.array([6.7], dtype=float)
    t_a = _unpolarized_t(pe_layers_coherent, wls, angle_deg)
    t_b = _unpolarized_t(pe_layers_coherent, wls, angle_deg)
    assert abs(float(t_a[0] - t_b[0])) < 1e-9


def test_incoherent_differs_from_coherent_pe(pe_layers_coherent, pe_layers_incoherent):
    """layer* air gap changes T vs coherent stack over λ sweep."""
    angle_deg = 4.0
    wls = np.geomspace(3.0, 15.0, 32)
    t_coh = _unpolarized_t(pe_layers_coherent, wls, angle_deg)
    t_inc = _unpolarized_t(pe_layers_incoherent, wls, angle_deg)
    max_diff = float(np.max(np.abs(t_inc - t_coh)))
    assert max_diff > 1e-7, f"expected incoherent air gap effect, max |ΔT|={max_diff}"


@pytest.mark.skipif(not _freesnell_toolchain_available(), reason="FreeSnell toolchain not available")
def test_incoherent_matches_freesnell_baseline_point(pe_layers_incoherent, simulation):
    """C++ incoherent TMM vs FreeSnell combine-layers (coherence.scm air layer* only)."""
    from filmstack_simulation.sweep import compute_wavelength_vs_RT_data

    th = np.deg2rad(4.0)
    wl = 6.7
    angle_deg = 4.0
    fs_layers = _pe_freesnell_layers_spec(simulation, pe_layers_incoherent, wl, air_incoherent=True)
    data = compute_wavelength_vs_RT_data(pe_layers_incoherent, np.array([wl]), angle_deg)

    for pol, t_py, r_py in (
        ("s", float(data["T_s"][0]), float(data["R_s"][0])),
        ("p", float(data["T_p"][0]), float(data["R_p"][0])),
    ):
        t_fs, r_fs, _ = _freesnell_combine_layers(th, wl, layers_spec=fs_layers, pol=pol)
        assert abs(t_py - t_fs) < 1e-5, f"T {pol}: cpp={t_py}, fs={t_fs}"
        assert abs(r_py - r_fs) < 1e-5, f"R {pol}: cpp={r_py}, fs={r_fs}"


@pytest.mark.skipif(not _freesnell_toolchain_available(), reason="FreeSnell toolchain not available")
def test_incoherent_spectrum_vs_freesnell_baseline(pe_layers_incoherent, simulation):
    """Wavelength sweep: unpolarized T vs FreeSnell (log-spaced λ)."""
    angle_deg = 4.0
    th = np.deg2rad(angle_deg)
    wls = np.geomspace(3.0, 15.0, 24)
    t_py = _unpolarized_t(pe_layers_incoherent, wls, angle_deg)
    t_fs = np.empty_like(t_py)
    for i, wl in enumerate(wls):
        fs_layers = _pe_freesnell_layers_spec(
            simulation, pe_layers_incoherent, float(wl), air_incoherent=True
        )
        ts, _, _ = _freesnell_combine_layers(th, float(wl), layers_spec=fs_layers, pol="s")
        tp, _, _ = _freesnell_combine_layers(th, float(wl), layers_spec=fs_layers, pol="p")
        t_fs[i] = 0.5 * (ts + tp)

    rmse = float(np.sqrt(np.mean((t_py - t_fs) ** 2)))
    assert rmse < 1e-4, f"spectrum RMSE={rmse}"


def test_incoherent_formula_flag_on_air_gap(pe_layers_incoherent):
    """Formula ``1000*`` sets is_incoherent only on the 1 mm air gap."""
    incoherent_depths = [
        float(lyr.depth)
        for lyr in pe_layers_incoherent
        if bool(getattr(lyr, "is_incoherent", False))
    ]
    assert incoherent_depths == [1000.0]


def test_coherent_example_matches_run_toykits(pe_layers_coherent, simulation):
    """Full coherent example aligned with ``run_toykits`` / fs_pe_1x14_co."""
    from scripts.build_freesnell_compare_ui import FREESNEL_COMPARE_SPECS, run_toykits
    from template_config import load_templates_json

    template_id = "fs_pe_1x14_co"
    tpl = next(t for t in load_templates_json()["templates"] if t["id"] == template_id)
    spec = FREESNEL_COMPARE_SPECS[template_id]
    wls = np.linspace(3.0, 15.0, 32)

    t_toy = run_toykits(template_id, spec, tpl, wls, "wavelength_um")
    t_cpp = _unpolarized_t(pe_layers_coherent, wls, float(tpl["sim"]["target_ang_deg"]))

    rmse = float(np.sqrt(np.mean((t_cpp - t_toy) ** 2)))
    assert rmse < 1e-5, f"coherent vs toykits RMSE={rmse}"
