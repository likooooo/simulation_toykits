"""Run the `test_diffraction` CLI next to simulation.so and parse JSON output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core import simulation_loader


def diffraction_artifacts_dir() -> Path:
    return Path(simulation_loader._artifacts_dir()).resolve()


def diffraction_binary_path() -> Path:
    return diffraction_artifacts_dir() / "test_diffraction"


def _prepend_ld_library_path(env: dict, directory: str) -> None:
    if sys.platform == "win32":
        return
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = directory + (":" + prev if prev else "")


def parse_diffraction_stdout(stdout: str) -> Dict[str, Any]:
    """Parse JSON object from CLI stdout (tolerates stray lines by taking outermost braces)."""
    text = stdout.strip()
    if not text:
        raise ValueError("empty stdout from test_diffraction")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in stdout: {text[:200]!r}")
    blob = text[start : end + 1]
    return json.loads(blob)


def pair_order_angle(branch: Dict[str, Any]) -> List[Tuple[int, float]]:
    """
    Pair diffraction order m with angle (deg).

    Prefer ``orders`` from JSON when present and same length as ``angles`` (handles
    non-consecutive or duplicate orders). Else if ``len(angles) == max - min + 1``,
    use ``m = min + i`` for each index (consecutive orders only).
    """
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
        "Rebuild test_diffraction so JSON includes matching \"orders\" array."
    )


def run_diffraction(
    L_um: float,
    wl_um: float,
    n_from: float,
    n_to: float,
    degree: float,
    *,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """
    Invoke: test_diffraction L wl nFrom nTo degree

    argv order matches simulation/tests/test_diffraction.cpp main().
    """
    exe = diffraction_binary_path()
    if not exe.is_file():
        raise FileNotFoundError(
            f"test_diffraction not found at {exe}. "
            "Build the simulation test target and copy the binary next to simulation.so."
        )

    artifacts = str(diffraction_artifacts_dir())
    env = os.environ.copy()
    _prepend_ld_library_path(env, artifacts)

    args = [
        str(exe),
        str(L_um),
        str(wl_um),
        str(n_from),
        str(n_to),
        str(degree),
    ]
    proc = subprocess.run(
        args,
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

    return parse_diffraction_stdout(proc.stdout or "")


def _diffraction_ray_label(prefix: str, m: int) -> str:
    if m == 0:
        return f"{prefix}0"
    return f"{prefix}{m:+d}"


def build_polar_diffraction_figure(
    incident_deg: float,
    transmitted: List[Tuple[int, float]],
    reflected: List[Tuple[int, float]],
):
    """
    Unit circle in Cartesian coordinates.

    - Left semicircle: incident + reflected. User angle 0° at (-1, 0); moving top→bottom
      along the left arc runs −90° … +90° (odd symmetry vs the right side).
    - Right semicircle: transmitted. User angle 0° at (1, 0); top→bottom runs +90° … −90°.
    - Tick marks every 20° with numeric labels; single unit circle (no concentric rings).
    - Endpoints of transmitted/reflected rays are annotated with diffraction order.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def _tick_label_deg(deg: float) -> str:
        """仅用于本图圆周刻度：正数带前导 +。"""
        x = float(deg)
        if x > 0:
            return f"+{x:.6g}°"
        if x < 0:
            return f"{x:.6g}°"
        return "0°"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    # --- Unit circle + vertical interface (x=0), same stroke as circumference ---
    _circle_lw = 0.95
    ph = np.linspace(0, 2 * np.pi, 480)
    ax.plot(np.cos(ph), np.sin(ph), color="black", lw=_circle_lw)
    ax.plot([0.0, 0.0], [-1.0, 1.0], color="black", lw=_circle_lw, solid_capstyle="round")

    label_r = 1.14
    tick_inner, tick_outer = 0.97, 1.03
    tick_degs = list(range(-80, 81, 20))

    # Right semicircle: user ψ at (cos ψ, sin ψ); 0° = (1,0); top (0,1) = +90°
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

    # Left semicircle: user ψ at θ = ψ + 180°; ψ=0 → (-1,0); top (0,1) → ψ = −90°
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

    # Transmitted (right): angle a → (cos a, sin a)
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

    # Reflected (left): (−cos a, −sin a) so tilt sign matches physical convention vs transmitted
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

    # Incident (left): same sign convention as reflected; i → (−cos i, −sin i)
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
