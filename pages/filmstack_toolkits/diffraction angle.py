"""Diffraction angle calculator: calls ``test_diffraction`` CLI (next to simulation.so)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from common import ensure_fresnel_session_state, get_nk_at_wavelength, pyplot_fixed_width, get_available_materials


def _diffraction_artifacts_dir() -> Path:
    return Path(os.environ["SIMULATION_ARTIFACTS_DIR"]).resolve()


def _diffraction_binary_path() -> Path:
    return _diffraction_artifacts_dir() / "test_diffraction"


def _prepend_ld_library_path(env: dict, directory: str) -> None:
    if sys.platform == "win32":
        return
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = directory + (":" + prev if prev else "")


def _parse_diffraction_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("empty stdout from test_diffraction")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in stdout: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _pair_order_angle(branch: dict[str, Any]) -> list[tuple[int, float]]:
    angles = [float(a) for a in branch["angles"]]
    mn = int(branch["min"])
    mx = int(branch["max"])
    orders = branch.get("orders")
    if orders is not None and len(orders) == len(angles):
        pairs = list(zip((int(o) for o in orders), angles))
        pairs.sort(key=lambda t: t[0])
        return pairs

    expected = mx - mn + 1
    if len(angles) == expected:
        return list(zip(range(mn, mx + 1), angles))

    raise ValueError(
        f"Cannot pair orders: len(angles)={len(angles)}, len(orders)="
        f"{len(orders) if orders is not None else 'n/a'}, min={mn}, max={mx} "
        f"(consecutive fallback needs {expected} entries). "
        'Rebuild test_diffraction so JSON includes matching "orders" array.'
    )


def _run_diffraction(
    L_um: float,
    wl_um: float,
    n_from: float,
    n_to: float,
    degree: float,
    *,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    exe = _diffraction_binary_path()
    if not exe.is_file():
        raise FileNotFoundError(
            f"test_diffraction not found at {exe}. "
            "Build the simulation test target and copy the binary next to simulation.so."
        )

    artifacts = str(_diffraction_artifacts_dir())
    env = os.environ.copy()
    _prepend_ld_library_path(env, artifacts)

    proc = subprocess.run(
        [str(exe), str(L_um), str(wl_um), str(n_from), str(n_to), str(degree)],
        cwd=artifacts,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or "(no stderr)"
        out = (proc.stdout or "").strip()[:500]
        raise RuntimeError(
            f"test_diffraction failed (code {proc.returncode}): {err}\nstdout: {out}"
        )

    return _parse_diffraction_stdout(proc.stdout or "")


def _diffraction_ray_label(prefix: str, m: int) -> str:
    if m == 0:
        return f"{prefix}0"
    return f"{prefix}{m:+d}"


def _build_polar_diffraction_figure(
    incident_deg: float,
    transmitted: list[tuple[int, float]],
    reflected: list[tuple[int, float]],
):
    def _tick_label_deg(deg: float) -> str:
        x = float(deg)
        if x > 0:
            return f"+{x:.6g}°"
        if x < 0:
            return f"{x:.6g}°"
        return "0°"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    _circle_lw = 0.95
    ph = np.linspace(0, 2 * np.pi, 480)
    ax.plot(np.cos(ph), np.sin(ph), color="black", lw=_circle_lw)
    ax.plot([0.0, 0.0], [-1.0, 1.0], color="black", lw=_circle_lw, solid_capstyle="round")

    label_r = 1.14
    tick_inner, tick_outer = 0.97, 1.03
    tick_degs = list(range(-80, 81, 20))

    for psi in tick_degs:
        rad = np.deg2rad(float(psi))
        ux, uy = np.cos(rad), np.sin(rad)
        ax.plot(
            [tick_inner * ux, tick_outer * ux],
            [tick_inner * uy, tick_outer * uy],
            color="black",
            lw=0.65,
        )
        if abs(psi) < 89:
            lx, ly = label_r * ux, label_r * uy
        elif psi > 0:
            lx, ly = 0.14, label_r * uy
        else:
            lx, ly = -0.14, label_r * uy
        ax.text(lx, ly, _tick_label_deg(float(psi)), ha="center", va="center", fontsize=8)

    for psi in tick_degs:
        th = np.deg2rad(float(psi + 180.0))
        ux, uy = np.cos(th), np.sin(th)
        on_shared_pole = abs(ux) < 1e-6 and abs(abs(uy) - 1.0) < 1e-6
        if not on_shared_pole:
            ax.plot(
                [tick_inner * ux, tick_outer * ux],
                [tick_inner * uy, tick_outer * uy],
                color="black",
                lw=0.65,
            )
        if abs(psi) < 89:
            lx, ly = label_r * ux, label_r * uy
        elif uy > 0:
            lx, ly = -0.14, label_r * uy
        else:
            lx, ly = 0.14, label_r * uy
        ax.text(lx, ly, _tick_label_deg(float(psi)), ha="center", va="center", fontsize=8)

    ann_r = 1.18

    for m, a_deg in transmitted:
        rad = np.deg2rad(float(a_deg))
        x1, y1 = np.cos(rad), np.sin(rad)
        ax.plot([0, x1], [0, y1], color="#1f77b4", lw=1.25, solid_capstyle="round")
        ax.text(
            ann_r * x1,
            ann_r * y1,
            _diffraction_ray_label("T", m),
            color="#1f77b4",
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold",
        )

    for m, a_deg in reflected:
        rad = np.deg2rad(float(a_deg))
        x1, y1 = -np.cos(rad), -np.sin(rad)
        ax.plot([0, x1], [0, y1], color="#d62728", lw=1.25, solid_capstyle="round")
        ax.text(
            ann_r * x1,
            ann_r * y1,
            _diffraction_ray_label("R", m),
            color="#d62728",
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ir = np.deg2rad(float(incident_deg))
    ix, iy = -np.cos(ir), -np.sin(ir)
    ax.plot([0, ix], [0, iy], color="#2ca02c", lw=2.2, solid_capstyle="round")
    ax.text(
        ann_r * ix,
        ann_r * iy,
        "Incident",
        color="#2ca02c",
        fontsize=10,
        ha="center",
        va="center",
        fontweight="bold",
    )

    ax.plot([], [], color="#2ca02c", lw=2.2, label="Incident")
    ax.plot([], [], color="#d62728", lw=1.25, label="Reflected")
    ax.plot([], [], color="#1f77b4", lw=1.25, label="Transmitted")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    ax.set_title("Diffraction diagram")
    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.38, 1.38)
    ax.axis("off")
    return fig


ensure_fresnel_session_state()

st.set_page_config(page_title="Diffraction angle (grating orders)", layout="wide")
st.header("衍射角计算器")

exe_path = _diffraction_binary_path()
if not exe_path.is_file():
    st.error(
        f"未找到 `test_diffraction` 可执行文件：{exe_path}\n"
        "请编译 simulation 的 `test_diffraction` 目标，并将其复制到 `.simulation_core/`（与 `simulation.so` 同目录）。"
    )
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    mats = get_available_materials()
    m1 = st.selectbox(
        "入射材料",
        options=mats,
        index=mats.index("air") if "air" in mats else 0,
        key="diffraction_mat1",
        help="在仿真数据库中添加更多材料数据",
    )
    m2 = st.selectbox(
        "出射材料",
        options=mats,
        index=min(1, len(mats) - 1),
        key="diffraction_mat2",
    )
with col_b:
    L_um = st.number_input(
        "光栅周期 Grating period (µm)",
        min_value=1e-9,
        value=float(st.session_state.get("diffraction_L_um", 5.0)),
        format="%.6f",
    )
    st.session_state["diffraction_L_um"] = L_um
    wl_um = st.number_input(
        "波长 Wavelength (µm)",
        min_value=1e-9,
        value=float(st.session_state.get("wavelength", 0.532)),
        format="%.3f",
    )
    incident_deg = st.slider(
        "入射角 Incident angle (°)",
        min_value=-89.0,
        max_value=89.0,
        value=float(st.session_state.get("diffraction_incident_deg", 0.0)),
        step=0.1,
        key="diffraction_incident_deg",
    )

nk1 = get_nk_at_wavelength(m1, wl_um)
nk2 = get_nk_at_wavelength(m2, wl_um)
n_from = float(np.real(nk1))
n_to = float(np.real(nk2))

with col_a:
    max_display_order = st.number_input(
        "Maximum Shown Order",
        value=int(st.session_state.get("diffraction_max_display_order", 3)),
        min_value=0,
        step=1,
        help="一个朴实无华的低通滤波器, 例如设为 3 时，透射与反射均只显示 m ∈ [−3, +3] 且计算结果中存在的级次。",
    )
st.session_state["diffraction_max_display_order"] = max_display_order

st.markdown(
    f"**n (@ {wl_um} µm)** from `{m1}`(n={n_from:.3f})"
    f" to `{m2}`(n={n_to:.6f})"
)

errors = []
if L_um <= 0:
    errors.append("光栅周期必须 > 0")
if wl_um <= 0:
    errors.append("波长必须 > 0")
if not (-90 < incident_deg < 90):
    errors.append("入射角须在 (-90°, 90°) 内")
if n_from <= 0 or n_to <= 0:
    errors.append("两侧材料在该波长下的折射率实部必须 > 0（请检查材料库与波长范围）")

for e in errors:
    st.error(e)

data = None
if not errors:
    try:
        with st.spinner("运行 test_diffraction…"):
            data = _run_diffraction(L_um, wl_um, n_from, n_to, incident_deg)
    except Exception as e:
        st.error(f"运行失败: {e}")

if data is not None:
    tj = data["transmitted"]
    rj = data["reflected"]

    t_pairs = _pair_order_angle(tj)
    r_pairs = _pair_order_angle(rj)

    n_show = int(max_display_order)
    t_show = [(m, a) for m, a in t_pairs if abs(m) <= n_show]
    r_show = [(m, a) for m, a in r_pairs if abs(m) <= n_show]

    st.divider()

    tc1, tc2 = st.columns([1, 4])
    with tc1:
        st.subheader("结果")
        st.markdown(
            f"- **入射角**: {incident_deg:g}°\n"
            f"- **透射** 级次范围: **{tj['min']} ~ {tj['max']}**；"
            f"显示 **|m| ≤ {n_show}**（共 {len(t_show)} 条）\n"
            f"- **反射** 级次范围: **{rj['min']} ~ {rj['max']}**；"
            f"显示 **|m| ≤ {n_show}**（共 {len(r_show)} 条）"
        )

        def format_order_row(pairs):
            parts = []
            for m, ang in pairs:
                sign = "+" if m > 0 else ""
                parts.append(f"**{sign}{m}**: {ang:.6g}°")
            return "  \n".join(parts) if parts else "（无）"

        st.markdown("###### Transmitted Orders")
        st.markdown(format_order_row(t_show))
        st.markdown("###### Reflected Orders")
        st.markdown(format_order_row(r_show))
    with tc2:
        fig = _build_polar_diffraction_figure(incident_deg, t_show, r_show)
        pyplot_fixed_width(fig, width=700)
        plt.close(fig)
