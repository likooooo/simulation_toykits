#!/usr/bin/env python3
"""Generate all OpenFilters cross-check / parity report figures and an HTML index."""

from __future__ import annotations

import argparse
import base64
import html
import sys
import webbrowser
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

_CROSSCHECK = Path(__file__).resolve().parent / "crosscheck"
if str(_CROSSCHECK) not in sys.path:
    sys.path.insert(0, str(_CROSSCHECK))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bootstrap_simulation import bootstrap_toykits_session, simulation_available  # noqa: E402
from compare import compare_thickness_derivatives  # noqa: E402
from openfilters_derivatives import (  # noqa: E402
    POL_TE,
    openfilters_rt_spectrum,
    openfilters_thickness_rt_derivatives,
)
from simulation_derivatives import Polarization, simulation_thickness_rt_derivatives  # noqa: E402
from stack_spec import StackSpec, load_default_materials_db  # noqa: E402

from refinement.fixtures import (  # noqa: E402
    PARITY_COMPLEX_STACK,
    PARITY_DA_TOL_NM,
    PARITY_MAX_ITER,
    PARITY_REFINEMENT_STACK,
    PARITY_SPECTRUM_WLS_NM,
)
from refinement.parity import build_parity_scenarios, make_r_target_problem, run_parity  # noqa: E402
from refinement.plot_parity import plot_parity_report, plot_parity_summary  # noqa: E402
from refinement.problem import RefinementProblem  # noqa: E402
from refinement.target_spec import RefinementTargetSpec  # noqa: E402


@dataclass
class ReportItem:
    section: str
    title: str
    path: Path
    status: str
    detail: str


def _plot_derivative_layers(
    spec: StackSpec,
    materials_db: dict,
    out_path: Path,
    *,
    title: str,
) -> ReportItem:
    result = compare_thickness_derivatives(
        spec, materials_db, [550.0], 0.0, Polarization.TE, rtol=5e-3
    )
    dR_of, dT_of = openfilters_thickness_rt_derivatives(
        spec, materials_db, [550.0], 0.0, POL_TE
    )
    dR_sim, dT_sim = simulation_thickness_rt_derivatives(
        spec, materials_db, [550.0], 0.0, Polarization.TE
    )

    n = len(spec.film_tokens)
    layers = list(range(n))
    dR_of_nm = [float(dR_of[j, 0]) for j in range(n)]
    dR_sim_nm = [float(dR_sim[1 + j, 0]) / 1000.0 for j in range(n)]
    dT_of_nm = [float(dT_of[j, 0]) for j in range(n)]
    dT_sim_nm = [float(dT_sim[1 + j, 0]) / 1000.0 for j in range(n)]
    rel_r = [
        abs(a - b) / max(abs(a), 1e-12) for a, b in zip(dR_of_nm, dR_sim_nm)
    ]
    rel_t = [
        abs(a - b) / max(abs(a), 1e-12) for a, b in zip(dT_of_nm, dT_sim_nm)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    status = "PASS" if result.ok_dR and result.ok_dT else "FAIL"
    fig.suptitle(
        f"{title}  [{status}]  forward_ok={result.ok_rt}  "
        f"max_dR_rel={result.max_dR_err:.2e}  max_dT_rel={result.max_dT_err:.2e}",
        fontsize=11,
    )

    x = np.arange(n)
    w = 0.35
    labels = [f"L{j}" for j in layers]

    ax = axes[0, 0]
    ax.bar(x - w / 2, dR_of_nm, w, label="abeles", color="C0")
    ax.bar(x + w / 2, dR_sim_nm, w, label="simulation adjoint", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("dR/dd (1/nm)")
    ax.set_title("Reflectance thickness derivative @ 550 nm")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x - w / 2, dT_of_nm, w, label="abeles", color="C0")
    ax.bar(x + w / 2, dT_sim_nm, w, label="simulation adjoint", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("dT/dd (1/nm)")
    ax.set_title("Transmittance thickness derivative @ 550 nm")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(x, rel_r, color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|abeles − sim| / |abeles|")
    ax.set_title("dR relative error")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(x, rel_t, color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|abeles − sim| / |abeles|")
    ax.set_title("dT relative error")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    detail = (
        f"forward R/T max err {result.max_rt_err:.2e}; "
        f"dR max rel {result.max_dR_err:.2e}; dT max rel {result.max_dT_err:.2e}"
    )
    return ReportItem("derivatives", title, out_path, status, detail)


def _build_complex_parity_scenarios(materials_db) -> list[tuple[str, RefinementProblem]]:
    spec = deepcopy(PARITY_COMPLEX_STACK)
    r0, _ = openfilters_rt_spectrum(spec, materials_db, [550.0], 0.0, POL_TE)
    p1 = make_r_target_problem(
        deepcopy(spec),
        materials_db,
        target_r=max(0.05, min(0.95, float(r0[0]) * 0.9)),
        sigma=0.01,
        polarization="TE",
    )
    wls = list(PARITY_SPECTRUM_WLS_NM)
    r_vals, _ = openfilters_rt_spectrum(spec, materials_db, wls, 0.0, POL_TE)
    p2 = RefinementProblem(
        deepcopy(spec),
        materials_db,
        [
            RefinementTargetSpec(
                kind="R_spectrum",
                wavelengths_nm=wls,
                values=[float(v * 0.92) for v in r_vals],
                tolerances=[0.01] * len(wls),
                polarization="TE",
            )
        ],
    )
    return [("C6_P1_single_R", p1), ("C6_P2_R_spectrum", p2)]


def _run_parity_plots(
    scenarios: list[tuple[str, RefinementProblem]],
    out_dir: Path,
    *,
    prefix: str,
    section: str,
    max_iter: int = PARITY_MAX_ITER,
) -> list[ReportItem]:
    items: list[ReportItem] = []
    summary_rows: list = []
    for label, problem in scenarios:
        report = run_parity(
            problem,
            max_iter=max_iter,
            da_tol_nm=PARITY_DA_TOL_NM,
            chi2_rtol=1e-3,
            label=label,
        )
        png = plot_parity_report(
            report, problem, out_dir / f"{prefix}{label}.png"
        )
        status = "PASS" if report.ok else "FAIL"
        items.append(
            ReportItem(section, label, png, status, report.message)
        )
        summary_rows.append((label, report, problem))

    if summary_rows:
        summary_path = plot_parity_summary(
            summary_rows, out_dir / f"{prefix}summary.png"
        )
        worst = max(summary_rows, key=lambda row: row[1].max_da_nm)
        items.append(
            ReportItem(
                section,
                f"{prefix}summary",
                summary_path,
                "PASS" if all(r[1].ok for r in summary_rows) else "FAIL",
                worst[1].message,
            )
        )
    return items


def _png_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _write_index(
    out_dir: Path,
    items: list[ReportItem],
    *,
    parity_max_iter: int = PARITY_MAX_ITER,
) -> Path:
    sections: dict[str, list[ReportItem]] = {}
    for item in items:
        sections.setdefault(item.section, []).append(item)

    section_titles = {
        "derivatives": "Thickness derivative cross-check",
        "parity_2layer": f"LM parity — 2-layer stack ({parity_max_iter} iter)",
        "parity_6layer": f"LM parity — 6-layer stack ({parity_max_iter} iter, expected divergence)",
    }

    body_parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>OpenFilters test reports</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;}",
        "h1{border-bottom:2px solid #333;padding-bottom:.5rem;}",
        "h2{margin-top:2rem;color:#444;}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0;}",
        ".pass{color:#0a0;font-weight:bold;}.fail{color:#c00;font-weight:bold;}",
        "img{max-width:100%;height:auto;border:1px solid #eee;margin-top:.5rem;}",
        ".detail{color:#666;font-size:.9rem;margin:.25rem 0;}",
        "</style></head><body>",
        "<h1>OpenFilters cross-check &amp; parity reports</h1>",
        f"<p>Generated in <code>{html.escape(str(out_dir))}</code></p>",
    ]

    for section, title in section_titles.items():
        group = sections.get(section, [])
        if not group:
            continue
        body_parts.append(f"<h2>{html.escape(title)}</h2>")
        for item in group:
            cls = "pass" if item.status == "PASS" else "fail"
            img_src = _png_data_uri(item.path) if item.path.is_file() else ""
            body_parts.append("<div class='card'>")
            body_parts.append(
                f"<strong>{html.escape(item.title)}</strong> "
                f"<span class='{cls}'>[{item.status}]</span>"
            )
            body_parts.append(f"<div class='detail'>{html.escape(item.detail)}</div>")
            if img_src:
                body_parts.append(
                    f"<img src='{img_src}' alt='{html.escape(item.title)}'>"
                )
            body_parts.append("</div>")

    body_parts.append("</body></html>")
    index_path = out_dir / "index.html"
    index_path.write_text("\n".join(body_parts), encoding="utf-8")
    return index_path


def _try_open(path: Path) -> None:
    uri = path.resolve().as_uri()
    print(f"Open in browser: {uri}")

    import os
    import subprocess

    if os.environ.get("WSL_DISTRO_NAME"):
        try:
            win_path = subprocess.check_output(
                ["wslpath", "-w", str(path.resolve())], text=True
            ).strip()
            subprocess.run(["explorer.exe", win_path], check=False)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    try:
        webbrowser.open(uri)
    except Exception as exc:  # pragma: no cover
        print(f"(could not auto-open browser: {exc})")
        print(f"  Manual: open {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate all OpenFilters test report figures and HTML index."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory for PNGs and index.html",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open index.html in the default browser after generation",
    )
    parser.add_argument(
        "--skip-nk-map",
        action="store_true",
        help="Skip n-k derivative map (slow)",
    )
    parser.add_argument(
        "--parity-max-iter",
        type=int,
        default=PARITY_MAX_ITER,
        help=f"LM parity outer-loop iterations (default: {PARITY_MAX_ITER})",
    )
    args = parser.parse_args()

    if not simulation_available():
        print(
            "error: simulation runtime missing (.simulation_toolkits/simulation.so)",
            file=sys.stderr,
        )
        return 1

    bootstrap_toykits_session()
    materials_db = load_default_materials_db()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[ReportItem] = []

    items.append(
        _plot_derivative_layers(
            PARITY_REFINEMENT_STACK,
            materials_db,
            out_dir / "derivatives_2layer.png",
            title="2-layer parity stack",
        )
    )
    items.append(
        _plot_derivative_layers(
            PARITY_COMPLEX_STACK,
            materials_db,
            out_dir / "derivatives_6layer.png",
            title="6-layer complex stack",
        )
    )

    items.extend(
        _run_parity_plots(
            build_parity_scenarios(materials_db),
            out_dir,
            prefix="parity_2L_",
            section="parity_2layer",
            max_iter=args.parity_max_iter,
        )
    )
    items.extend(
        _run_parity_plots(
            _build_complex_parity_scenarios(materials_db),
            out_dir,
            prefix="parity_6L_",
            section="parity_6layer",
            max_iter=args.parity_max_iter,
        )
    )

    if not args.skip_nk_map:
        try:
            from plot_nk_derivative_maps import plot_nk_derivative_maps  # noqa: WPS433

            nk_path = out_dir / "nk_derivative_map.png"
            plot_nk_derivative_maps(nk_path, grid_size=30)
            items.append(
                ReportItem(
                    "derivatives",
                    "n-k derivative map (layer 0, R @ 550 nm)",
                    nk_path,
                    "INFO",
                    "2D scan: abeles vs simulation vs diff",
                )
            )
        except Exception as exc:
            print(f"warning: skipped n-k map: {exc}", file=sys.stderr)

    index_path = _write_index(out_dir, items, parity_max_iter=args.parity_max_iter)
    print(f"\nwrote {index_path}")
    for item in items:
        print(f"  [{item.status:4s}] {item.path}")

    if args.open:
        _try_open(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
