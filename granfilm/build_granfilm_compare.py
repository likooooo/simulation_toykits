#!/usr/bin/env python3
"""Build portable GranFilm Fortran baseline vs Python comparison HTML.

Usage (after ``source scripts/init-toykits-build-env.sh``)::

    python granfilm/build_granfilm_compare.py
    python granfilm/build_granfilm_compare.py --output /tmp/gf_ui

Writes ``gf_baseline_vs_python.html`` to the output directory.
Intermediate artifacts go under ``/tmp/gf_compare_build_*``.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict
from zoneinfo import ZoneInfo

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from granfilm.paths import PACKAGE_DIR, REPO_ROOT  # noqa: E402

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output" / "compare_ui"
COMPARE_HTML_NAME = "gf_baseline_vs_python.html"
MAX_ABS_WARN_THRESHOLD = 1e-5

from granfilm.common.baseline import (  # noqa: E402
    default_2film_baseline_path,
    default_aspnes_baseline_path,
    default_baseline_path,
    default_below_baseline_path,
    default_coated_baseline_path,
    default_film_baseline_path,
    default_hexagonal_baseline_path,
    default_invariants_baseline_path,
    default_prolate_baseline_path,
    default_quadrupole_baseline_path,
    default_spheroid_baseline_path,
    default_square_baseline_path,
    default_thin_cap_baseline_path,
    default_yamaguchi_baseline_path,
    granfilm_dir,
    load_baseline,
)
from granfilm.common.compare_viz import compare_metrics, plot_case_summary  # noqa: E402
from granfilm.common.constants import DEFAULT_GRANFILM_DIR  # noqa: E402
from granfilm.oblate_prolate.case import (  # noqa: E402
    default_coated_case,
    default_prolate_case,
    default_spheroid_case,
    default_yamaguchi_case,
)
from granfilm.oblate_prolate.pipeline import run_granfilm_spheroid  # noqa: E402
from granfilm.paths import COMPARE_DIR, OBLATE_COMPARE_DIR, SPHERE_COMPARE_DIR  # noqa: E402
from granfilm.sphere_island.case import (  # noqa: E402
    default_2film_case,
    default_aspnes_case,
    default_film_case,
    default_hexagonal_case,
    default_invariants_case,
    default_island_below_case,
    default_quadrupole_case,
    default_sphere_case,
    default_square_case,
    default_thin_cap_case,
)
from granfilm.sphere_island.pipeline import run_granfilm_sphere  # noqa: E402

PipelineKind = Literal["sphere", "spheroid"]


class CompareSpec(TypedDict):
    id: str
    label: str
    pipeline: PipelineKind
    subdir: str
    case_fn: Callable[[], object]
    baseline_fn: Callable[[], Path]
    out_root: Path


GRANFILM_COMPARE_SPECS: list[CompareSpec] = [
    {
        "id": "gf_sphere_island",
        "label": "截顶球岛层 Ag/MgO",
        "pipeline": "sphere",
        "subdir": "island",
        "case_fn": default_sphere_case,
        "baseline_fn": default_baseline_path,
        "out_root": SPHERE_COMPARE_DIR,
    },
    {
        "id": "gf_sphere_island_below",
        "label": "岛中心在衬底下方",
        "pipeline": "sphere",
        "subdir": "island_below",
        "case_fn": default_island_below_case,
        "baseline_fn": default_below_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_sphere_film",
        "label": "连续薄膜",
        "pipeline": "sphere",
        "subdir": "film",
        "case_fn": default_film_case,
        "baseline_fn": default_film_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_sphere_2film",
        "label": "双层薄膜",
        "pipeline": "sphere",
        "subdir": "2film",
        "case_fn": default_2film_case,
        "baseline_fn": default_2film_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_sphere_thin_cap",
        "label": "极扁帽层",
        "pipeline": "sphere",
        "subdir": "thin_cap",
        "case_fn": default_thin_cap_case,
        "baseline_fn": default_thin_cap_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_sphere_quadrupole",
        "label": "四极相互作用",
        "pipeline": "sphere",
        "subdir": "quadrupole",
        "case_fn": default_quadrupole_case,
        "baseline_fn": default_quadrupole_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_sphere_square",
        "label": "方格晶格",
        "pipeline": "sphere",
        "subdir": "square",
        "case_fn": default_square_case,
        "baseline_fn": default_square_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_sphere_hexagonal",
        "label": "六方晶格",
        "pipeline": "sphere",
        "subdir": "hexagonal",
        "case_fn": default_hexagonal_case,
        "baseline_fn": default_hexagonal_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_sphere_invariants",
        "label": "Invariants Fresnel",
        "pipeline": "sphere",
        "subdir": "invariants",
        "case_fn": default_invariants_case,
        "baseline_fn": default_invariants_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_sphere_aspnes",
        "label": "Aspnes Fresnel",
        "pipeline": "sphere",
        "subdir": "aspnes",
        "case_fn": default_aspnes_case,
        "baseline_fn": default_aspnes_baseline_path,
        "out_root": COMPARE_DIR,
    },
    {
        "id": "gf_spheroid_oblate",
        "label": "扁椭球岛",
        "pipeline": "spheroid",
        "subdir": "oblate",
        "case_fn": default_spheroid_case,
        "baseline_fn": default_spheroid_baseline_path,
        "out_root": OBLATE_COMPARE_DIR,
    },
    {
        "id": "gf_spheroid_prolate",
        "label": "长椭球岛",
        "pipeline": "spheroid",
        "subdir": "prolate",
        "case_fn": default_prolate_case,
        "baseline_fn": default_prolate_baseline_path,
        "out_root": OBLATE_COMPARE_DIR,
    },
    {
        "id": "gf_spheroid_yamaguchi",
        "label": "Yamaguchi 偶极",
        "pipeline": "spheroid",
        "subdir": "yamaguchi",
        "case_fn": default_yamaguchi_case,
        "baseline_fn": default_yamaguchi_baseline_path,
        "out_root": OBLATE_COMPARE_DIR,
    },
    {
        "id": "gf_spheroid_coated",
        "label": "涂层椭球",
        "pipeline": "spheroid",
        "subdir": "coated",
        "case_fn": default_coated_case,
        "baseline_fn": default_coated_baseline_path,
        "out_root": OBLATE_COMPARE_DIR,
    },
]


def resolve_granfilm_env(*, granfilm_root: Path | None = None) -> Path:
    """Validate GranFilm root under GENERATE_GOLDEN_TOOLS_DIR and return it."""
    root = Path(granfilm_root or DEFAULT_GRANFILM_DIR).resolve()
    testing = root / "testing"
    issues: list[str] = []
    if not root.is_dir():
        issues.append(f"GranFilm 根目录不存在: {root}")
    elif not testing.is_dir():
        issues.append(f"缺少 testing/: {testing}")
    if issues:
        raise RuntimeError(
            "GranFilm 基准目录不可用（期望 $GENERATE_GOLDEN_TOOLS_DIR/GranFilm-v1.0）。\n"
            + "\n".join(f"  - {x}" for x in issues)
        )
    return root


def _prepend_sys_path(path: Path) -> None:
    s = str(path.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _check_toykits_runtime_issues(*, try_import: bool = True) -> list[str]:
    issues: list[str] = []
    artifacts_str = os.environ.get("SIMULATION_ARTIFACTS_DIR", "").strip()
    if not artifacts_str:
        issues.append(
            "环境变量 SIMULATION_ARTIFACTS_DIR 未设置"
            "（请先 source scripts/init-toykits-build-env.sh）"
        )
        return issues

    artifacts = Path(artifacts_str).resolve()
    _prepend_sys_path(REPO_ROOT)
    _prepend_sys_path(artifacts)

    so = artifacts / "simulation.so"
    if not so.is_file():
        issues.append(f"缺少 runtime artifact: {so}")

    db_bin = artifacts / "assets" / "database.bin"
    if not db_bin.is_file():
        issues.append(f"缺少预编译材料库: {db_bin}")

    if not os.environ.get("SIMULATION_DATABASE_DIR", "").strip():
        issues.append("环境变量 SIMULATION_DATABASE_DIR 未设置")

    if try_import and not issues:
        try:
            import simulation  # noqa: F401
        except ImportError as exc:
            issues.append(f"import simulation 失败: {exc}")

    return issues


def prepare_toykits_runtime() -> str | None:
    issues = _check_toykits_runtime_issues(try_import=True)
    if issues:
        return "toykits 运行环境不可用：" + "；".join(issues)
    artifacts = Path(os.environ["SIMULATION_ARTIFACTS_DIR"]).resolve()
    os.chdir(artifacts)
    return None


def build_materials_db() -> dict[str, Any]:
    import simulation_database_parser as sdp

    from granfilm.common.materials import build_granfilm_materials_db

    return build_granfilm_materials_db(sim_db=sdp.get_simulation_database(init=True))


def make_build_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="gf_compare_build_", dir="/tmp"))


def clean_output_dir(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / COMPARE_HTML_NAME
    if html_path.is_file():
        html_path.unlink()


def describe_case_failure(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return str(exc)
    return str(exc)


def process_case(
    spec: CompareSpec,
    materials_db: dict[str, Any],
    build_case_dir: Path,
    *,
    write_legacy_viz: bool = False,
) -> dict[str, Any]:
    """Run one compare case; return row dict for HTML table."""
    case_id = spec["id"]
    label = spec["label"]
    row: dict[str, Any] = {
        "id": case_id,
        "label": label,
        "pipeline": spec["pipeline"],
        "subdir": spec["subdir"],
        "status": "ok",
    }
    build_case_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline_path = spec["baseline_fn"]()
        if not baseline_path.is_file():
            raise FileNotFoundError(f"GranFilm baseline not found: {baseline_path}")
        baseline = load_baseline(baseline_path)
        case = spec["case_fn"]()

        legacy_dir = spec["out_root"] / spec["subdir"] if write_legacy_viz else None
        viz_dir = legacy_dir or build_case_dir

        if spec["pipeline"] == "sphere":
            result = run_granfilm_sphere(
                case,
                materials_db,
                viz_dir=viz_dir if write_legacy_viz else None,
                write_viz=write_legacy_viz,
                baseline=baseline if write_legacy_viz else None,
            )
            energy = result.energy
            python_dr = result.dr
        else:
            result = run_granfilm_spheroid(
                case,
                materials_db,
                viz_dir=viz_dir if write_legacy_viz else None,
                write_viz=write_legacy_viz,
                baseline=baseline if write_legacy_viz else None,
            )
            energy = result.energy
            python_dr = result.dr

        compare_png = build_case_dir / "compare.png"
        metrics_path = build_case_dir / "metrics.json"
        panel_metrics = plot_case_summary(
            case,
            pipeline=spec["pipeline"],
            energy=energy,
            python_dr=python_dr,
            baseline=baseline,
            out=compare_png,
            metrics_path=metrics_path,
        )

        py_on_base = np.interp(baseline.energy_ev, energy, python_dr)
        stats = compare_metrics(py_on_base, baseline.value)
        row.update(stats)
        row["max_abs_err"] = panel_metrics.get("max_abs_err", stats["max_abs"])
        row["note"] = panel_metrics.get("note", "")
        if panel_metrics.get("chi2_reliance") is not None:
            row["chi2_reliance"] = panel_metrics["chi2_reliance"]

        if stats["max_abs"] >= MAX_ABS_WARN_THRESHOLD:
            row["expected_mismatch"] = True
            row["comment"] = str(panel_metrics.get("note", ""))
        else:
            row["expected_mismatch"] = False

    except Exception as exc:
        row["status"] = "failed"
        row["error"] = describe_case_failure(exc)
        (build_case_dir / "error.txt").write_text(row["error"], encoding="utf-8")

    return row


def process_all_cases(
    materials_db: dict[str, Any],
    build_dir: Path,
    *,
    write_legacy_viz: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in GRANFILM_COMPARE_SPECS:
        case_dir = build_dir / "cases" / spec["id"]
        rows.append(
            process_case(
                spec,
                materials_db,
                case_dir,
                write_legacy_viz=write_legacy_viz,
            )
        )
    return rows


def png_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def write_static_site(
    rows: list[dict[str, Any]],
    output: Path,
    build_dir: Path,
    *,
    built_at: str,
) -> None:
    def fmt(v: Any) -> str:
        if v is None or v == "":
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    table_rows: list[str] = []
    for row in rows:
        case_id = row["id"]
        img_path = build_dir / "cases" / case_id / "compare.png"
        has_img = img_path.is_file()
        img_src = png_data_uri(img_path) if has_img else ""
        status = row.get("status", "")
        if status != "ok":
            cls = "failed"
        elif row.get("expected_mismatch"):
            cls = "warn"
        else:
            cls = "ok"
        label = html.escape(str(row.get("label", "")))
        if status != "ok":
            comment = html.escape(str(row.get("error", "")))
        elif row.get("expected_mismatch") or row.get("comment"):
            comment = html.escape(str(row.get("comment", row.get("note", ""))))
        else:
            comment = ""
        if has_img:
            img_cell = (
                f'<a href="{img_src}" target="_blank" rel="noopener" class="thumb-link">'
                f'<img src="{img_src}" alt="" class="thumb"></a>'
            )
        else:
            img_cell = ""
        table_rows.append(
            f'<tr class="{cls}">'
            f"<td>{label}</td>"
            f"<td>{fmt(row.get('rmse'))}</td>"
            f"<td>{fmt(row.get('max_abs'))}</td>"
            f"<td>{fmt(row.get('corr'))}</td>"
            f"<td>{img_cell}</td>"
            f"<td>{comment}</td></tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>GranFilm baseline vs Python</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1rem 2rem;color:#222}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:6px 8px;text-align:left;font-size:13px;vertical-align:top}}
tr:nth-child(even){{background:#f9f9f9}}
tr.warn{{background:#fff8e1}}
tr.failed{{background:#fdecea}}
.thumb{{max-width:220px;max-height:140px;object-fit:contain;border:1px solid #ccc;display:block}}
.meta{{color:#555;font-size:14px;margin-bottom:0.75rem}}
td:nth-child(6){{max-width:320px;font-size:12px;line-height:1.4}}
</style></head><body>
<h1>GranFilm baseline vs Python</h1>
<p class="meta">Built {built_at}，共 {len(rows)} 项基准测试。基准来源：{html.escape(str(granfilm_dir()))}</p>
<table><thead><tr>
<th>算例</th><th>rmse</th><th>max_abs</th><th>corr</th><th>image</th><th>备注</th>
</tr></thead><tbody>
{"".join(table_rows)}
</tbody></table>
<script>
document.addEventListener("click", function (e) {{
  if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
  var a = e.target.closest("a.thumb-link");
  if (!a) return;
  e.preventDefault();
  var img = a.querySelector("img");
  if (!img || !img.src) return;
  var src = img.src;
  var comma = src.indexOf(",");
  if (comma < 0) return;
  var meta = src.slice(0, comma);
  var payload = src.slice(comma + 1);
  var mime = "image/png";
  var m = /^data:([^;]+)/.exec(meta);
  if (m) mime = m[1];
  var bin = atob(payload);
  var bytes = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  var url = URL.createObjectURL(new Blob([bytes], {{ type: mime }}));
  var w = window.open(url, "_blank", "noopener");
  if (w) w.opener = null;
}});
</script>
</body></html>
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / COMPARE_HTML_NAME).write_text(html_doc, encoding="utf-8")


def build_granfilm_compare_ui(
    output: Path | None = None,
    *,
    granfilm_root: Path | None = None,
    write_legacy_viz: bool = False,
) -> int:
    """Build GranFilm baseline vs Python comparison HTML. Returns 0 on success."""
    env_err = prepare_toykits_runtime()
    if env_err:
        print(f">>> 错误: {env_err}", file=sys.stderr)
        return 1

    try:
        resolve_granfilm_env(granfilm_root=granfilm_root)
    except RuntimeError as exc:
        print(f">>> 错误: {exc}", file=sys.stderr)
        return 1

    out = (output or DEFAULT_OUTPUT_DIR).resolve()
    clean_output_dir(out)
    build_dir = make_build_dir()
    print(f">>> 中间产物目录: {build_dir}", flush=True)

    rows: list[dict[str, Any]] = []
    try:
        materials_db = build_materials_db()
        for spec in GRANFILM_COMPARE_SPECS:
            print(f"[{spec['id']}] ...", flush=True)
            case_dir = build_dir / "cases" / spec["id"]
            row = process_case(
                spec,
                materials_db,
                case_dir,
                write_legacy_viz=write_legacy_viz,
            )
            rows.append(row)
            if row["status"] != "ok":
                print(f"  -> failed: {row.get('error', '')}", flush=True)
            else:
                print(
                    f"  -> ok max_abs={row.get('max_abs', float('nan')):.3e}",
                    flush=True,
                )

        built_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
        write_static_site(rows, out, build_dir, built_at=built_at)

        summary_path = out / "summary.json"
        summary = {
            spec["id"]: rows[i].get("max_abs_err", rows[i].get("max_abs"))
            for i, spec in enumerate(GRANFILM_COMPARE_SPECS)
            if rows[i].get("status") == "ok"
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    html_path = out / COMPARE_HTML_NAME
    failed = [r for r in rows if r.get("status") != "ok"]
    print(f"Done: {len(rows)} cases, {len(failed)} failed. HTML: {html_path}")
    if failed:
        print("Failed:", ", ".join(r["id"] for r in failed), file=sys.stderr)
        for r in failed:
            print(f"  {r['id']}: {r.get('error', '')}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for {COMPARE_HTML_NAME}",
    )
    parser.add_argument(
        "--write-legacy-viz",
        action="store_true",
        help="Also write step figures under granfilm/output/compare/",
    )
    args = parser.parse_args(argv)
    return build_granfilm_compare_ui(
        args.output,
        granfilm_root=None,
        write_legacy_viz=args.write_legacy_viz,
    )


if __name__ == "__main__":
    raise SystemExit(main())
