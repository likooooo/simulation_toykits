#!/usr/bin/env python3
"""Build portable fs baseline vs toykits comparison HTML from filmstack_templates.json.

Usage (after ``source scripts/init-toykits-build-env.sh``)::

    python scripts/build_freesnell_compare_ui.py
    python scripts/build_freesnell_compare_ui.py --output /tmp/fs_ui

Writes ``fs_baseline_vs_toykits.html`` to the output directory.
Intermediate artifacts go under ``/tmp/fs_compare_build_*`` (see ``make_build_dir()``).

Baseline export strips FreeSnell ``(smooth …)``; toykits traces are compared raw.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from template_config import (
    is_incoherent_template,
    load_templates_json,
    quantity_from_rt,
    target_ang_deg_from_sim,
    target_wl_um_from_sim,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / ".simulation_toolkits" / "assets" / "fs_compare"
FS_MATERIALS_DIR = REPO_ROOT / "simulation_core" / "assets" / "database" / "fs" / "materials"
EV_TO_UM = 1.23984193

def _toolchain_path(env_key: str, default: str) -> Path:
    return Path(os.environ.get(env_key, default))


DEFAULT_FREESNELL_DIR = _toolchain_path("FREESNELL_DIR", "/home/like/repos/freesnell-build/FreeSnell")
DEFAULT_SCM = _toolchain_path("SCM", "/home/like/repos/freesnell-install/bin/scm")
DEFAULT_SLIB = _toolchain_path("SLIB", "/home/like/repos/freesnell-install/lib/slib/")


def resolve_freesnell_env() -> dict[str, str]:
    """Resolve FreeSnell runtime env: os.environ overrides, else hardcoded defaults."""
    freesnell_dir = Path(
        os.environ.get("FREESNELL_DIR", "").strip() or str(DEFAULT_FREESNELL_DIR)
    ).resolve()
    scm = Path(os.environ.get("SCM", "").strip() or str(DEFAULT_SCM)).resolve()
    slib_raw = os.environ.get("SLIB", "").strip() or str(DEFAULT_SLIB)
    slib = Path(slib_raw).resolve()
    slib_str = str(slib)
    if not slib_str.endswith("/"):
        slib_str += "/"
    nk_rwb = os.environ.get("NK_RWB", "").strip() or str(freesnell_dir / "nk.rwb")
    scheme_lib = os.environ.get("SCHEME_LIBRARY_PATH", "").strip() or slib_str
    nk_db = os.environ.get("NK_DATABASE_PATH", "").strip() or nk_rwb

    issues: list[str] = []
    if not freesnell_dir.is_dir():
        issues.append(f"FREESNELL_DIR 不存在: {freesnell_dir}")
    if not Path(nk_rwb).is_file():
        issues.append(f"NK_RWB 不存在: {nk_rwb}")
    if not scm.is_file() or not os.access(scm, os.X_OK):
        issues.append(f"SCM 不可执行: {scm}")
    slib_path = Path(slib_str)
    if not slib_path.is_dir():
        issues.append(f"SLIB 不存在: {slib_path}")
    elif not (slib_path / "require.scm").is_file() and not (slib_path / "require").is_file():
        issues.append(f"SLIB 缺少 require.scm: {slib_path}")

    if issues:
        raise RuntimeError(
            "FreeSnell toolchain 不可用，--bench 失败。\n"
            + "\n".join(f"  - {item}" for item in issues)
            + "\n请 export FREESNELL_DIR/SCM/SLIB 或将工具链放到默认路径"
            f"（FREESNELL_DIR={DEFAULT_FREESNELL_DIR}，"
            f"SCM={DEFAULT_SCM}，SLIB={DEFAULT_SLIB}）。"
        )

    return {
        "FREESNELL_DIR": str(freesnell_dir),
        "SCM": str(scm),
        "SLIB": slib_str,
        "SCHEME_LIBRARY_PATH": scheme_lib,
        "NK_RWB": nk_rwb,
        "NK_DATABASE_PATH": nk_db,
    }

class CompareSpec(TypedDict, total=False):
    scm_file: str
    invoke: str
    x_axis: str
    quantity: str
    png_dims: tuple[int, int]
    granular_ir: bool
    quantity_col: int
    multi_angle_baseline: bool
    baseline_incident_angles_deg: list[float]


FREESNEL_COMPARE_SPECS: dict[str, CompareSpec] = {'fs_ag_mgo_3nm': {'granular_ir': True,
                   'invoke': 'export/AgMgO-Rp',
                   'png_dims': [400, 333],
                   'quantity': 'R_p',
                   'quantity_col': 1,
                   'scm_file': 'granular.scm',
                   'x_axis': 'eV'},
 'fs_ag_mgo_ang_film': {'granular_ir': True,
                        'invoke': 'export/AgMgO-ang-film',
                        'png_dims': [335, 320],
                        'quantity': 'R_p',
                        'quantity_col': 1,
                        'scm_file': 'granular.scm',
                        'x_axis': 'angle_deg'},
 'fs_ag_mgo_ang_mgo': {'granular_ir': True,
                       'invoke': 'export/AgMgO-ang-mgo',
                       'png_dims': [335, 320],
                       'quantity': 'R',
                       'quantity_col': 1,
                       'scm_file': 'granular.scm',
                       'x_axis': 'angle_deg'},
 'fs_ag_mgo_p': {'granular_ir': True,
                 'invoke': 'export/AgMgO-p-Tp',
                 'png_dims': [440, 285],
                 'quantity': 'T_p',
                 'quantity_col': 1,
                 'scm_file': 'granular.scm',
                 'x_axis': 'eV'},
 'fs_ag_mgo_s': {'granular_ir': True,
                 'invoke': 'export/AgMgO-s-Rs',
                 'png_dims': [440, 285],
                 'quantity': 'R_s',
                 'quantity_col': 1,
                 'scm_file': 'granular.scm',
                 'x_axis': 'eV'},
 'fs_al_mirror': {'granular_ir': False,
                  'invoke': 'al-mirror',
                  'png_dims': [390, 200],
                  'quantity': 'R',
                  'quantity_col': 1,
                  'scm_file': 'metallic.scm',
                  'x_axis': 'wavelength_um'},
 'fs_au_mirror': {'granular_ir': False,
                  'invoke': 'bare-au',
                  'png_dims': [390, 200],
                  'quantity': 'R',
                  'quantity_col': 1,
                  'scm_file': 'metallic.scm',
                  'x_axis': 'wavelength_um'},
 'fs_bb_hr_50deg': {'granular_ir': False,
                    'invoke': 'reflector-50',
                    'png_dims': [512, 256],
                    'quantity': 'R_p',
                    'quantity_col': 1,
                    'scm_file': 'dielectric.scm',
                    'x_axis': 'wavelength_um'},
 'fs_cloud_h2o': {'granular_ir': True,
                  'invoke': 'export/cloud-h2o',
                  'png_dims': [350, 250],
                  'quantity': 'T',
                  'quantity_col': 1,
                  'scm_file': 'coherence.scm',
                  'x_axis': 'wavenumber_cm'},
 'fs_cloud_ice': {'granular_ir': True,
                  'invoke': 'export/cloud-ice',
                  'png_dims': [350, 250],
                  'quantity': 'T',
                  'quantity_col': 1,
                  'scm_file': 'coherence.scm',
                  'x_axis': 'wavenumber_cm'},
 'fs_cold_mirror_jk': {'granular_ir': False,
                       'invoke': 'export/cold-mirror-jk',
                       'png_dims': [512, 256],
                       'quantity': 'R',
                       'quantity_col': 1,
                       'scm_file': 'dielectric.scm',
                       'x_axis': 'wavelength_um'},
 'fs_cold_mirror_tc1': {'granular_ir': False,
                        'invoke': 'export/cold-mirror1',
                        'png_dims': [512, 256],
                        'quantity': 'R',
                        'quantity_col': 1,
                        'scm_file': 'dielectric.scm',
                        'x_axis': 'wavelength_um'},
 'fs_cold_mirror_tc2': {'granular_ir': False,
                        'invoke': 'export/cold-mirror2',
                        'png_dims': [512, 256],
                        'quantity': 'R',
                        'quantity_col': 1,
                        'scm_file': 'dielectric.scm',
                        'x_axis': 'wavelength_um'},
 'fs_dual_bp': {'granular_ir': False,
                'invoke': 'dual-bp',
                'png_dims': [512, 256],
                'quantity': 'T',
                'quantity_col': 1,
                'scm_file': 'metallic.scm',
                'x_axis': 'wavelength_um'},
 'fs_enhanced_al_1hl': {'granular_ir': False,
                        'invoke': 'export/enhanced-al-1hl',
                        'png_dims': [435, 155],
                        'quantity': 'R',
                        'quantity_col': 1,
                        'scm_file': 'metallic.scm',
                        'x_axis': 'wavelength_um'},
 'fs_enhanced_al_2hl': {'granular_ir': False,
                        'invoke': 'export/enhanced-al-2hl',
                        'png_dims': [435, 155],
                        'quantity': 'R',
                        'quantity_col': 1,
                        'scm_file': 'metallic.scm',
                        'x_axis': 'wavelength_um'},
 'fs_enhanced_al_3hl': {'granular_ir': False,
                        'invoke': 'export/enhanced-al-3hl',
                        'png_dims': [435, 155],
                        'quantity': 'R',
                        'quantity_col': 1,
                        'scm_file': 'metallic.scm',
                        'x_axis': 'wavelength_um'},
 'fs_enhanced_al_bare': {'granular_ir': False,
                         'invoke': 'export/enhanced-al-bare',
                         'png_dims': [435, 155],
                         'quantity': 'R',
                         'quantity_col': 1,
                         'scm_file': 'metallic.scm',
                         'x_axis': 'wavelength_um'},
 'fs_fig_2_2_a': {'granular_ir': False,
                  'invoke': 'export/fig-2-2-a',
                  'png_dims': [512, 256],
                  'quantity': 'R',
                  'quantity_col': 1,
                  'scm_file': 'dielectric.scm',
                  'x_axis': 'wavelength_um'},
 'fs_fig_2_2_b': {'granular_ir': False,
                  'invoke': 'export/fig-2-2-b',
                  'png_dims': [512, 256],
                  'quantity': 'R',
                  'quantity_col': 1,
                  'scm_file': 'dielectric.scm',
                  'x_axis': 'wavelength_um'},
 'fs_fig_2_5': {'granular_ir': False,
                'invoke': 'fig.2.5',
                'png_dims': [512, 256],
                'quantity': 'R',
                'quantity_col': 1,
                'scm_file': 'dielectric.scm',
                'x_axis': 'wavelength_um'},
 'fs_film_flat_50um': {'granular_ir': False,
                       'invoke': 'film-flat',
                       'png_dims': [600, 184],
                       'quantity': 'T',
                       'quantity_col': 1,
                       'scm_file': 'coherence.scm',
                       'x_axis': 'wavelength_um'},
 'fs_hdpe_12_7': {'granular_ir': False,
                  'invoke': 'export/hdpe-12.7-inc',
                  'png_dims': [428, 168],
                  'quantity': 'T',
                  'quantity_col': 1,
                  'scm_file': 'polyethylene.scm',
                  'x_axis': 'wavelength_um'},
 'fs_hdpe_130': {'granular_ir': False,
                 'invoke': 'export/hdpe-130-inc',
                 'png_dims': [325, 317],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'polyethylene.scm',
                 'x_axis': 'wavelength_um'},
 'fs_hdpe_14': {'granular_ir': False,
                'invoke': 'hdpe-14',
                'png_dims': [342, 310],
                'quantity': 'T',
                'quantity_col': 1,
                'scm_file': 'polyethylene.scm',
                'x_axis': 'wavelength_um'},
 'fs_hdpe_33': {'granular_ir': False,
                'invoke': 'export/hdpe-33-raw',
                'png_dims': [300, 200],
                'quantity': 'T',
                'quantity_col': 1,
                'scm_file': 'polyethylene.scm',
                'x_axis': 'wavelength_um'},
 'fs_hdpe_33ni': {'granular_ir': False,
                  'invoke': 'export/hdpe-33ni-inc',
                  'png_dims': [300, 200],
                  'quantity': 'T',
                  'quantity_col': 1,
                  'scm_file': 'polyethylene.scm',
                  'x_axis': 'wavelength_um'},
 'fs_hdpe_50': {'granular_ir': False,
                'invoke': 'export/hdpe-50-raw',
                'png_dims': [800, 245],
                'quantity': 'T',
                'quantity_col': 1,
                'scm_file': 'polyethylene.scm',
                'x_axis': 'wavelength_um'},
 'fs_hdpe_age_100': {'granular_ir': False,
                     'invoke': 'hdpe-age-1',
                     'png_dims': [625, 395],
                     'quantity': 'T',
                     'quantity_col': 1,
                     'scm_file': 'polyethylene.scm',
                     'x_axis': 'wavenumber_cm'},
 'fs_hdpe_card_14': {'granular_ir': False,
                     'invoke': 'export/hdpe-card-inc',
                     'png_dims': [640, 442],
                     'quantity': 'T',
                     'quantity_col': 1,
                     'scm_file': 'polyethylene.scm',
                     'x_axis': 'wavenumber_cm'},
 'fs_hdpe_card_abs_18': {'granular_ir': False,
                         'invoke': 'export/hdpe-card-abs-inc',
                         'png_dims': [365, 215],
                         'quantity': 'neg_log_T',
                         'quantity_col': 1,
                         'scm_file': 'polyethylene.scm',
                         'x_axis': 'wavelength_um'},
 'fs_hdpe_const_50': {'granular_ir': False,
                      'invoke': 'hdpe-const',
                      'png_dims': [600, 184],
                      'quantity': 'T',
                      'quantity_col': 1,
                      'scm_file': 'polyethylene.scm',
                      'x_axis': 'wavelength_um'},
 'fs_hdpe_ftir_32': {'granular_ir': False,
                     'invoke': 'hdpe-ftir',
                     'png_dims': [425, 265],
                     'quantity': 'T',
                     'quantity_col': 1,
                     'scm_file': 'polyethylene.scm',
                     'x_axis': 'wavenumber_cm'},
 'fs_hdpe_l100': {'granular_ir': False,
                  'invoke': 'export/hdpe-L100-co-raw',
                  'png_dims': [300, 200],
                  'quantity': 'T',
                  'quantity_col': 1,
                  'scm_file': 'polyethylene.scm',
                  'x_axis': 'wavelength_um'},
 'fs_hdpe_l100_inc': {'granular_ir': False,
                      'invoke': 'export/hdpe-L100-inc',
                      'png_dims': [300, 200],
                      'quantity': 'T',
                      'quantity_col': 1,
                      'scm_file': 'polyethylene.scm',
                      'x_axis': 'wavelength_um'},
 'fs_hdpe_l33': {'granular_ir': False,
                 'invoke': 'export/hdpe-L33-co-raw',
                 'png_dims': [300, 200],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'polyethylene.scm',
                 'x_axis': 'wavelength_um'},
 'fs_hdpe_pas_32': {'granular_ir': False,
                    'invoke': 'export/hdpe-pas-inc',
                    'png_dims': [425, 265],
                    'quantity': 'T',
                    'quantity_col': 1,
                    'scm_file': 'polyethylene.scm',
                    'x_axis': 'wavenumber_cm'},
 'fs_hdpe_pe_25_4': {'granular_ir': False,
                     'invoke': 'hdpe-pe',
                     'png_dims': [745, 255],
                     'quantity': 'T',
                     'quantity_col': 1,
                     'scm_file': 'polyethylene.scm',
                     'x_axis': 'wavelength_um'},
 'fs_hdpe_polyeth_35': {'granular_ir': False,
                        'invoke': 'export/polyeth-35-co',
                        'png_dims': [490, 360],
                        'quantity': 'T',
                        'quantity_col': 1,
                        'scm_file': 'polyethylene.scm',
                        'x_axis': 'wavenumber_cm'},
 'fs_hdpe_polyeth_35_inc': {'granular_ir': False,
                            'invoke': 'export/polyeth-35-inc',
                            'png_dims': [490, 360],
                            'quantity': 'T',
                            'quantity_col': 1,
                            'scm_file': 'polyethylene.scm',
                            'x_axis': 'wavenumber_cm'},
 'fs_hl7_mirror': {'granular_ir': False,
                   'invoke': 'HL7',
                   'png_dims': [512, 256],
                   'quantity': 'R',
                   'quantity_col': 1,
                   'scm_file': 'dielectric.scm',
                   'x_axis': 'wavelength_um'},
 'fs_hot_mirror_2': {'granular_ir': False,
                     'invoke': 'export/hot-mirror-2-T',
                     'png_dims': [420, 365],
                     'quantity': 'T',
                     'quantity_col': 1,
                     'scm_file': 'metallic.scm',
                     'x_axis': 'wavelength_um'},
 'fs_immersed_polarizer': {'granular_ir': False,
                           'invoke': 'export/immersed-Tp',
                           'png_dims': [512, 256],
                           'quantity': 'T_p',
                           'quantity_col': 1,
                           'scm_file': 'dielectric.scm',
                           'x_axis': 'angle_deg'},
 'fs_ir_ar_growing': {'granular_ir': False,
                      'invoke': 'anti',
                      'png_dims': [512, 256],
                      'quantity': 'R',
                      'quantity_col': 1,
                      'scm_file': 'dielectric.scm',
                      'x_axis': 'wavelength_um'},
 'fs_metal_bp': {'granular_ir': False,
                 'invoke': 'metal-bp',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'},
 'fs_nobles_ag': {'granular_ir': True,
                  'invoke': 'export/nobles-ag',
                  'png_dims': [270, 270],
                  'quantity': 'R_p',
                  'quantity_col': 1,
                  'scm_file': 'granular.scm',
                  'x_axis': 'eV'},
 'fs_nobles_al': {'granular_ir': True,
                  'invoke': 'export/nobles-al',
                  'png_dims': [270, 270],
                  'quantity': 'R_p',
                  'quantity_col': 1,
                  'scm_file': 'granular.scm',
                  'x_axis': 'eV'},
 'fs_nobles_au': {'granular_ir': True,
                  'invoke': 'export/nobles-au',
                  'png_dims': [270, 270],
                  'quantity': 'R_p',
                  'quantity_col': 1,
                  'scm_file': 'granular.scm',
                  'x_axis': 'eV'},
 'fs_nobles_cu': {'granular_ir': True,
                  'invoke': 'export/nobles-cu',
                  'png_dims': [270, 270],
                  'quantity': 'R_p',
                  'quantity_col': 1,
                  'scm_file': 'granular.scm',
                  'x_axis': 'eV'},
 'fs_omni_mirror': {'baseline_incident_angles_deg': [10, 20, 30, 40, 50, 60, 70, 80, 89],
                    'granular_ir': False,
                    'invoke': 'omni-mirror',
                    'multi_angle_baseline': True,
                    'png_dims': [512, 256],
                    'quantity': 'R',
                    'scm_file': 'dielectric.scm',
                    'x_axis': 'wavelength_um'},
 'fs_pe_1x14_co': {'granular_ir': False,
                   'invoke': 'export/pe-1x14-co',
                   'png_dims': [384, 256],
                   'quantity': 'T',
                   'quantity_col': 1,
                   'scm_file': 'coherence.scm',
                   'x_axis': 'wavelength_um'},
 'fs_pe_1x14_inc': {'granular_ir': False,
                    'invoke': 'export/pe-1x14-inc',
                    'png_dims': [384, 256],
                    'quantity': 'T',
                    'quantity_col': 1,
                    'scm_file': 'coherence.scm',
                    'x_axis': 'wavelength_um'},
 'fs_pe_2x14_co': {'granular_ir': False,
                   'invoke': 'export/pe-2x14-co',
                   'png_dims': [384, 256],
                   'quantity': 'T',
                   'quantity_col': 1,
                   'scm_file': 'coherence.scm',
                   'x_axis': 'wavelength_um'},
 'fs_pe_2x14_inc': {'granular_ir': False,
                    'invoke': 'export/pe-2x14-inc',
                    'png_dims': [384, 256],
                    'quantity': 'T',
                    'quantity_col': 1,
                    'scm_file': 'coherence.scm',
                    'x_axis': 'wavelength_um'},
 'fs_pe_33': {'granular_ir': False,
              'invoke': 'export/pe-33-inc',
              'png_dims': [390, 260],
              'quantity': 'T',
              'quantity_col': 1,
              'scm_file': 'polyethylene.scm',
              'x_axis': 'eV'},
 'fs_pe_3x14_co': {'granular_ir': False,
                   'invoke': 'export/pe-3x14-co',
                   'png_dims': [384, 256],
                   'quantity': 'T',
                   'quantity_col': 1,
                   'scm_file': 'coherence.scm',
                   'x_axis': 'wavelength_um'},
 'fs_pe_3x14_inc': {'granular_ir': False,
                    'invoke': 'export/pe-3x14-inc',
                    'png_dims': [384, 256],
                    'quantity': 'T',
                    'quantity_col': 1,
                    'scm_file': 'coherence.scm',
                    'x_axis': 'wavelength_um'},
 'fs_polymer_ag': {'granular_ir': True,
                   'invoke': 'polymer-ag',
                   'png_dims': [512, 256],
                   'quantity': 'R',
                   'quantity_col': 1,
                   'scm_file': 'granular.scm',
                   'x_axis': 'wavelength_um'},
 'fs_prism_polarizer': {'granular_ir': False,
                        'invoke': 'export/prism-Ts',
                        'png_dims': [512, 256],
                        'quantity': 'T_s',
                        'quantity_col': 1,
                        'scm_file': 'dielectric.scm',
                        'x_axis': 'wavelength_um'},
 'fs_protected_al_bare': {'granular_ir': False,
                          'invoke': 'export/protected-al-bare',
                          'png_dims': [435, 155],
                          'quantity': 'R',
                          'quantity_col': 1,
                          'scm_file': 'metallic.scm',
                          'x_axis': 'wavelength_um'},
 'fs_protected_al_n2': {'granular_ir': False,
                        'invoke': 'export/protected-al-n2',
                        'png_dims': [435, 155],
                        'quantity': 'R',
                        'quantity_col': 1,
                        'scm_file': 'metallic.scm',
                        'x_axis': 'wavelength_um'},
 'fs_protected_al_sio': {'granular_ir': False,
                         'invoke': 'export/protected-al-sio',
                         'png_dims': [435, 155],
                         'quantity': 'R',
                         'quantity_col': 1,
                         'scm_file': 'metallic.scm',
                         'x_axis': 'wavelength_um'},
 'fs_rdad': {'granular_ir': False,
             'invoke': 'r-d-a-d',
             'png_dims': [410, 250],
             'quantity': 'R',
             'quantity_col': 1,
             'scm_file': 'metallic.scm',
             'x_axis': 'wavelength_um'},
 'fs_ruby_glass_14um': {'granular_ir': True,
                        'invoke': 'export/ruby-14um',
                        'png_dims': [512, 256],
                        'quantity': 'T',
                        'quantity_col': 1,
                        'scm_file': 'granular.scm',
                        'x_axis': 'wavelength_um'},
 'fs_ruby_glass_2um': {'granular_ir': True,
                       'invoke': 'export/ruby-2um',
                       'png_dims': [512, 256],
                       'quantity': 'T',
                       'quantity_col': 1,
                       'scm_file': 'granular.scm',
                       'x_axis': 'wavelength_um'},
 'fs_ruby_glass_8um': {'granular_ir': True,
                       'invoke': 'export/ruby-8um',
                       'png_dims': [512, 256],
                       'quantity': 'T',
                       'quantity_col': 1,
                       'scm_file': 'granular.scm',
                       'x_axis': 'wavelength_um'},
 'fs_sio2_al': {'granular_ir': False,
                'invoke': 'Si2O3-Al',
                'png_dims': [395, 205],
                'quantity': 'R',
                'quantity_col': 1,
                'scm_file': 'metallic.scm',
                'x_axis': 'wavelength_um'},
 'fs_transitions_co': {'granular_ir': True,
                       'invoke': 'export/transitions-co',
                       'png_dims': [270, 270],
                       'quantity': 'R_p',
                       'quantity_col': 1,
                       'scm_file': 'granular.scm',
                       'x_axis': 'eV'},
 'fs_transitions_pd': {'granular_ir': True,
                       'invoke': 'export/transitions-pd',
                       'png_dims': [270, 270],
                       'quantity': 'R_p',
                       'quantity_col': 1,
                       'scm_file': 'granular.scm',
                       'x_axis': 'eV'},
 'fs_transitions_pt': {'granular_ir': True,
                       'invoke': 'export/transitions-pt',
                       'png_dims': [270, 270],
                       'quantity': 'R_p',
                       'quantity_col': 1,
                       'scm_file': 'granular.scm',
                       'x_axis': 'eV'},
 'fs_transitions_ti': {'granular_ir': True,
                       'invoke': 'export/transitions-ti',
                       'png_dims': [270, 270],
                       'quantity': 'R_p',
                       'quantity_col': 1,
                       'scm_file': 'granular.scm',
                       'x_axis': 'eV'},
 'fs_wide_bp': {'granular_ir': False,
                'invoke': 'wide-bp',
                'png_dims': [512, 256],
                'quantity': 'T',
                'quantity_col': 1,
                'scm_file': 'dielectric.scm',
                'x_axis': 'wavelength_um'},
 'fs_zns_ot_1': {'granular_ir': False,
                 'invoke': 'export/zns-ot-1',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'},
 'fs_zns_ot_2': {'granular_ir': False,
                 'invoke': 'export/zns-ot-2',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'},
 'fs_zns_ot_3': {'granular_ir': False,
                 'invoke': 'export/zns-ot-3',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'},
 'fs_zns_ot_4': {'granular_ir': False,
                 'invoke': 'export/zns-ot-4',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'},
 'fs_zns_ot_5': {'granular_ir': False,
                 'invoke': 'export/zns-ot-5',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'},
 'fs_zns_ot_6': {'granular_ir': False,
                 'invoke': 'export/zns-ot-6',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'},
 'fs_zns_ot_7': {'granular_ir': False,
                 'invoke': 'export/zns-ot-7',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'},
 'fs_zns_ot_8': {'granular_ir': False,
                 'invoke': 'export/zns-ot-8',
                 'png_dims': [512, 256],
                 'quantity': 'T',
                 'quantity_col': 1,
                 'scm_file': 'metallic.scm',
                 'x_axis': 'wavelength_um'}}

FREESNEL_CUSTOM_EXPORT_SCM = r""";;; --- custom single-stack exporters ---

(define (export-fig-2-2-a)
  (define nH 2.30) (define nL 1.45) (define nS 1.52)
  (plot-response (title "fig-2-2-a" "fig-2-2-a")
    (samples 300) (incident 0 'R) (wavelengths 400e-9 900e-9)
    (optical-stack (nominal (/ 500e-9 4)) (substrate 1)
      (layer nL 1.099) (layer nH 0.375) (layer nL 0.106) (layer nH 1.977)
      (layer nL 0.34) (layer nH 0.318) (substrate nS))))

(define (export-fig-2-2-b)
  (define nH 2.30) (define nL 1.45) (define nS 1.52)
  (plot-response (title "fig-2-2-b" "fig-2-2-b")
    (samples 300) (incident 0 'R) (wavelengths 400e-9 900e-9)
    (optical-stack (nominal (/ 500e-9 4)) (substrate 1)
      (layer nL 1.216) (layer nH 0.710) (layer nL 0.188) (layer nH 1.104)
      (layer nL 0.391) (layer nH 0.34) (substrate nS))))

(define (export-cold-mirror1)
  (define glass 1.52)
  (define SiO2 1.46)
  (define TiO2 2.40)
  (plot-response (title "Cold Mirrors 1" "cold-mirror1")
    (samples 300) (range 0 1) (wavelengths 380e-9 1000e-9) (incident 0 'R)
    (optical-stack
      (layer TiO2 34.40e-9) (layer SiO2 158.61e-9) (layer TiO2 53.24e-9)
      (layer SiO2 128.87e-9) (layer TiO2 67.77e-9) (layer SiO2 102.19e-9)
      (layer TiO2 79.13e-9) (layer SiO2 97.19e-9) (layer TiO2 53.07e-9)
      (layer SiO2 162.54e-9) (layer TiO2 36.16e-9) (layer SiO2 66.56e-9)
      (layer TiO2 46.48e-9) (layer SiO2 87.18e-9) (layer TiO2 48.92e-9)
      (layer SiO2 82.52e-9) (layer TiO2 42.56e-9) (layer SiO2 73.13e-9)
      (layer TiO2 43.97e-9) (layer SiO2 102.57e-9) (layer TiO2 44.21e-9)
      (substrate glass))))

(define (export-cold-mirror2)
  (define glass 1.52)
  (define SiO2 1.46)
  (define TiO2 2.40)
  (plot-response (title "Cold Mirrors 2" "cold-mirror2")
    (samples 300) (range 0 1) (wavelengths 380e-9 2000e-9) (incident 0 'R)
    (optical-stack
      (layer TiO2 19.03e-9) (layer SiO2 191.13e-9) (layer TiO2 51.30e-9)
      (layer SiO2 137.38e-9) (layer TiO2 61.86e-9) (layer SiO2 99.64e-9)
      (layer TiO2 81.48e-9) (layer SiO2 102.08e-9) (layer TiO2 53.15e-9)
      (layer SiO2 162.37e-9) (layer TiO2 36.85e-9) (layer SiO2 70.41e-9)
      (layer TiO2 39.55e-9) (layer SiO2 88.77e-9) (layer TiO2 51.97e-9)
      (layer SiO2 86.41e-9) (layer TiO2 51.05e-9) (layer SiO2 66.36e-9)
      (layer TiO2 28.39e-9) (layer SiO2 124.04e-9) (layer TiO2 33.76e-9)
      (substrate glass))))

(define (export-cold-mirror-jk)
  (define glass 1.52)
  (define SiO2 1.46)
  (define TiO2 2.40)
  (define nom 550e-9)
  (plot-response (title "Cold Mirror JK" "cold-mirror-jk")
    (samples 300) (range 0 1) (wavelengths 300e-9 1100e-9) (incident 0 'R)
    (optical-stack (substrate 1)
      (nominal nom)
      (layer SiO2 188.36e-9)
      (nominal (* 0.8 nom))
      (repeat 5
        (layer TiO2 22.92e-9)
        (layer SiO2 75.34e-9)
        (layer TiO2 22.92e-9))
      (nominal (* 1.0 nom))
      (repeat 5
        (layer TiO2 28.65e-9)
        (layer SiO2 94.18e-9)
        (layer TiO2 28.65e-9))
      (nominal (* 1.2 nom))
      (repeat 5
        (layer TiO2 34.38e-9)
        (layer SiO2 113.01e-9)
        (layer TiO2 34.38e-9))
      (substrate glass))))

(define (export-protected-al-sio)
  (plot-response (title "Protected Al SiO" "protected-al-sio")
    (samples 300) (incident 0 'R) (wavelengths .3e-6 .9e-6)
    (optical-stack (layer 1.45 (/ 1.75 4) .6e-6) (substrate AL))))

(define (export-protected-al-n2)
  (plot-response (title "Protected Al n=2" "protected-al-n2")
    (samples 300) (incident 0 'R) (wavelengths .3e-6 .9e-6)
    (optical-stack (layer 2 (/ 1.65 4) .6e-6) (substrate AL))))

(define (export-protected-al-bare)
  (plot-response (title "Protected Al bare" "protected-al-bare")
    (samples 300) (incident 0 'R) (wavelengths .3e-6 .9e-6)
    (optical-stack (substrate AL))))

(define (export-enhanced-al-bare)
  (define H 2.40) (define L 1.46)
  (plot-response (title "Enhanced Al bare" "enhanced-al-bare")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'R)
    (optical-stack (substrate AL))))

(define (export-enhanced-al-1hl)
  (define H 2.40) (define L 1.46)
  (plot-response (title "Enhanced Al 1HL" "enhanced-al-1hl")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'R)
    (optical-stack (nominal 550e-9) (layer H 1/4) (layer L 1/4) (substrate AL))))

(define (export-enhanced-al-2hl)
  (define H 2.40) (define L 1.46)
  (plot-response (title "Enhanced Al 2HL" "enhanced-al-2hl")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'R)
    (optical-stack (nominal 550e-9)
      (layer H 1/4) (layer L 1/4) (layer H 1/4) (layer L 1/4) (substrate AL))))

(define (export-enhanced-al-3hl)
  (define H 2.40) (define L 1.46)
  (plot-response (title "Enhanced Al 3HL" "enhanced-al-3hl")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'R)
    (optical-stack (nominal 550e-9)
      (layer H 1/4) (layer L 1/4) (layer H 1/4) (layer L 1/4)
      (layer H 1/4) (layer L 1/4) (substrate AL))))

(define (export-hot-mirror-2-T)
  (plot-response (title "Hot Mirror T" "hot-2")
    (samples 1200) (range 0 1) (incident 0 'T) (logarithmic 0.3e-6 2e-6)
    (optical-stack (layer TiO2 18e-9) (layer Ag 18e-9) (layer TiO2 18e-9) (layer COR7059 1e-3))))

(define (export-immersed-Tp)
  (define SiO2 1.45) (define TiO2 2.35) (define BK7 1.5164)
  (plot-response (title "Immersed Tp" "immersed-Tp")
    (samples 50) (range .9 1) (wavelength .643e-6 'T_p) (angles 51 71)
    (optical-stack (substrate BK7)
      (layer TiO2 83.62e-9) (layer SiO2 74.91e-9) (layer TiO2 94.86e-9)
      (layer SiO2 119.09e-9) (layer TiO2 90.21e-9) (layer SiO2 129.48e-9)
      (layer TiO2 81.87e-9) (layer SiO2 123.04e-9) (layer TiO2 84.03e-9)
      (layer SiO2 138.23e-9) (layer TiO2 81.34e-9) (layer SiO2 131.15e-9)
      (layer TiO2 83.85e-9) (layer SiO2 152.21e-9) (layer TiO2 80.48e-9)
      (layer SiO2 126.78e-9) (layer TiO2 59.95e-9) (substrate BK7))))

(define (export-prism-Ts)
  (define H 2.35) (define L 1.45)
  (plot-response (title "Prism Ts" "prism-Ts")
    (samples 50) (range .99 1) (wavelengths .4e-6 .7e-6) (incident 70 'T_s)
    (optical-stack (nominal .55e-6) (substrate 1.85)
      (layer H 11.16e-9) (layer L 48.87e-9) (layer H 35.94e-9) (layer L 68.58e-9)
      (layer H 39.50e-9) (layer L 79.77e-9) (layer H 46.20e-9) (layer L 96.61e-9)
      (layer H 49.71e-9) (layer L 102.74e-9) (layer H 50.49e-9) (layer L 102.74e-9)
      (layer H 49.71e-9) (layer L 96.61e-9) (layer H 46.20e-9) (layer L 79.77e-9)
      (layer H 39.50e-9) (layer L 68.58e-9) (layer H 35.94e-9) (layer L 48.87e-9)
      (layer H 11.16e-9) (substrate 1.85))))

(define (export-zns-ot-1)
  (define wv 5461e-10)
  (plot-response (title "ZnS OT 1/8" "ZnS-ot-1")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'T)
    (optical-stack (nominal wv) (layer ZnS (/ 1 8)) (substrate 1.55))))
(define (export-zns-ot-2)
  (define wv 5461e-10)
  (plot-response (title "ZnS OT 2/8" "ZnS-ot-2")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'T)
    (optical-stack (nominal wv) (layer ZnS (/ 2 8)) (substrate 1.55))))
(define (export-zns-ot-3)
  (define wv 5461e-10)
  (plot-response (title "ZnS OT 3/8" "ZnS-ot-3")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'T)
    (optical-stack (nominal wv) (layer ZnS (/ 3 8)) (substrate 1.55))))
(define (export-zns-ot-4)
  (define wv 5461e-10)
  (plot-response (title "ZnS OT 4/8" "ZnS-ot-4")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'T)
    (optical-stack (nominal wv) (layer ZnS (/ 4 8)) (substrate 1.55))))
(define (export-zns-ot-5)
  (define wv 5461e-10)
  (plot-response (title "ZnS OT 5/8" "ZnS-ot-5")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'T)
    (optical-stack (nominal wv) (layer ZnS (/ 5 8)) (substrate 1.55))))
(define (export-zns-ot-6)
  (define wv 5461e-10)
  (plot-response (title "ZnS OT 6/8" "ZnS-ot-6")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'T)
    (optical-stack (nominal wv) (layer ZnS (/ 6 8)) (substrate 1.55))))
(define (export-zns-ot-7)
  (define wv 5461e-10)
  (plot-response (title "ZnS OT 7/8" "ZnS-ot-7")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'T)
    (optical-stack (nominal wv) (layer ZnS (/ 7 8)) (substrate 1.55))))
(define (export-zns-ot-8)
  (define wv 5461e-10)
  (plot-response (title "ZnS OT 8/8" "ZnS-ot-8")
    (samples 300) (wavelengths .3e-6 .9e-6) (incident 0 'T)
    (optical-stack (nominal wv) (layer ZnS (/ 8 8)) (substrate 1.55))))
(define (export-ruby-2um)
  (define grg (granular-IR au 2e-6 1.5))
  (plot-response (title "gold-ruby-glass 2um" "ruby-2um")
    (samples 300) (range 0 1) (wavelengths 380e-9 780e-9) (incident 0 'T)
    (optical-stack (layer* grg 4e-3))))
(define (export-ruby-8um)
  (define grg (granular-IR au 08e-6 1.5))
  (plot-response (title "gold-ruby-glass 8um" "ruby-8um")
    (samples 300) (range 0 1) (wavelengths 380e-9 780e-9) (incident 0 'T)
    (optical-stack (layer* grg 4e-3))))
(define (export-ruby-14um)
  (define grg (granular-IR au 14e-6 1.5))
  (plot-response (title "gold-ruby-glass 14um" "ruby-14um")
    (samples 300) (range 0 1) (wavelengths 380e-9 780e-9) (incident 0 'T)
    (optical-stack (layer* grg 4e-3))))
(define (export-AgMgO-Rp)
  (define metal (granular-IR Ag 0.67 1))
  (plot-response (title "AgMgO Rp" "AgMgO-Rp")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))

(define (export-AgMgO-p-Tp)
  (define metal (granular-IR Ag 0.67 1))
  (plot-response (title "AgMgO Tp" "AgMgO-p-Tp")
    (samples 300) (eVs 1.5 5) (incident 45 'T_p)
    (optical-stack (layer metal 3e-9) (layer MgO 1e-9))))

(define (export-AgMgO-s-Rs)
  (define metal (granular-IR Ag 0.67 1))
  (plot-response (title "AgMgO Rs" "AgMgO-s-Rs")
    (samples 300) (eVs 1.5 5) (incident 45 'R_s)
    (optical-stack (layer metal 3e-9) (substrate MgO))))

(define (export-AgMgO-ang-mgo)
  (plot-response (title "MgO substrate" "MgO-ang")
    (samples 300) (range 0 1) (angles 0 90) (wavelength (eV<->L 2.5) 'R)
    (optical-stack (substrate MgO))))

(define (export-AgMgO-ang-film)
  (define metal (granular-IR Ag 0.67 1))
  (plot-response (title "AgMgO ang film" "AgMgO-ang-film")
    (samples 300) (range 0 1) (angles 0 90) (wavelength (eV<->L 2.5) 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))

(define (export-nobles-ag)
  (define metal (granular-IR Ag 0.67 1))
  (plot-response (title "nobles Ag" "nobles-ag")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))
(define (export-nobles-au)
  (define metal (granular-IR Au 0.67 1))
  (plot-response (title "nobles Au" "nobles-au")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))
(define (export-nobles-cu)
  (define metal (granular-IR Cu 0.67 1))
  (plot-response (title "nobles Cu" "nobles-cu")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))
(define (export-nobles-al)
  (define metal (granular-IR Al 0.67 1))
  (plot-response (title "nobles Al" "nobles-al")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))
(define (export-transitions-co)
  (define metal (granular-IR Co 0.67 1))
  (plot-response (title "transitions Co" "transitions-co")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))
(define (export-transitions-pt)
  (define metal (granular-IR Pt 0.67 1))
  (plot-response (title "transitions Pt" "transitions-pt")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))
(define (export-transitions-pd)
  (define metal (granular-IR Pd 0.67 1))
  (plot-response (title "transitions Pd" "transitions-pd")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))
(define (export-transitions-ti)
  (define metal (granular-IR Ti 0.67 1))
  (plot-response (title "transitions Ti" "transitions-ti")
    (samples 300) (eVs 1.5 5) (incident 45 'R_p)
    (optical-stack (layer metal 3e-9) (substrate MgO))))
(define (export-pe-1x14-co)
  (plot-response (title "1x14 PE co" "PE-1x14-co")
    (samples 200) (range 0 1) (logarithmic 3e-6 15e-6) (incident 4 'T)
    (optical-stack (nominal 6.7e-6) (substrate 1)
      (repeat 1 (layer 1.0 1e-3) (layer HDPE 14e-6))
      (substrate 1))))
(define (export-pe-1x14-inc)
  (plot-response (title "1x14 PE inc" "PE-1x14-inc")
    (samples 200) (range 0 1) (logarithmic 3e-6 15e-6) (incident 4 'T)
    (optical-stack (nominal 6.7e-6) (substrate 1)
      (repeat 1 (layer* 1.0 1e-3) (layer* HDPE 14e-6))
      (substrate 1))))
(define (export-pe-2x14-co)
  (plot-response (title "2x14 PE co" "PE-2x14-co")
    (samples 1200) (range 0 1) (logarithmic 3e-6 15e-6) (incident 4 'T)
    (optical-stack (nominal 6.7e-6) (substrate 1)
      (repeat 2 (layer 1.0 1e-3) (layer HDPE 14e-6))
      (substrate 1))))
(define (export-pe-2x14-inc)
  (plot-response (title "2x14 PE inc" "PE-2x14-inc")
    (samples 200) (range 0 1) (logarithmic 3e-6 15e-6) (incident 4 'T)
    (optical-stack (nominal 6.7e-6) (substrate 1)
      (repeat 2 (layer* 1.0 1e-3) (layer* HDPE 14e-6))
      (substrate 1))))
(define (export-pe-3x14-co)
  (plot-response (title "3x14 PE co" "PE-3x14-co")
    (samples 1200) (range 0 1) (logarithmic 3e-6 15e-6) (incident 4 'T)
    (optical-stack (nominal 6.7e-6) (substrate 1)
      (repeat 3 (layer 1.0 1e-3) (layer HDPE 14e-6))
      (substrate 1))))
(define (export-pe-3x14-inc)
  (plot-response (title "3x14 PE inc" "PE-3x14-inc")
    (samples 200) (range 0 1) (logarithmic 3e-6 15e-6) (incident 4 'T)
    (optical-stack (nominal 6.7e-6) (substrate 1)
      (repeat 3 (layer* 1.0 1e-3) (layer* HDPE 14e-6))
      (substrate 1))))
(define (export-cloud-h2o)
  (define cld01 (granular-IR h2o 1e-6 1))
  (plot-response (title "Cloud h2o" "cloud-h2o")
    (samples 300) (range 0 0.01) (wavenumbers (/ 0.01 4.5e-6) (/ 0.01 40e-6))
    (incident 0 'T) (optical-stack (layer* cld01 250))))

(define (export-cloud-ice)
  (define ice01 (granular-IR ice 1e-6 1))
  (plot-response (title "Cloud ice" "cloud-ice")
    (samples 300) (range 0 0.06) (wavenumbers (/ 0.01 4.5e-6) (/ 0.01 40e-6))
    (incident 0 'T) (optical-stack (layer* ice01 250))))

(define (export-hdpe-L100-co-raw)
  (plot-response (title "hdpe L100 co raw" "hdpe-L100-co-raw")
    (samples 600) (wavelengths 2e-6 20e-6) (incident 0 'T)
    (optical-stack (nominal 11e-6) (substrate 1) (layer HDPE-BE 100e-6) (substrate 1))))

(define (export-hdpe-L100-inc)
  (plot-response (title "hdpe L100 inc" "hdpe-L100-inc")
    (samples 600) (wavelengths 2e-6 20e-6) (incident 0 'T)
    (optical-stack (nominal 11e-6) (substrate 1) (layer* HDPE-BE 100e-6) (substrate 1))))

(define (export-hdpe-L33-co-raw)
  (plot-response (title "hdpe L33 co raw" "hdpe-L33-co-raw")
    (samples 600) (wavelengths 2e-6 20e-6) (incident 0 'T)
    (optical-stack (nominal 11e-6) (substrate 1) (layer HDPE-BE 33e-6) (substrate 1))))

(define (export-hdpe-12.7-inc)
  (plot-response (title "hdpe 12.7 inc" "hdpe-12.7-inc")
    (samples 400) (wavelengths 2e-6 15e-6) (incident 0 'T)
    (optical-stack (nominal 11e-6) (substrate 1) (layer* HDPE 12.7e-6) (substrate 1))))

(define (export-hdpe-33ni-inc)
  (plot-response (title "hdpe 33ni inc" "hdpe-33ni-inc")
    (samples 1200) (wavelengths 2e-6 20e-6) (incident 0 'T)
    (optical-stack (nominal 11e-6) (substrate 1) (layer* hdpe 33e-6) (substrate 1))))

(define (export-hdpe-130-inc)
  (plot-response (title "hdpe 130 inc" "hdpe-130-inc")
    (samples 1000) (wavelengths 2.5e-6 14.9e-6) (incident 0 'T)
    (optical-stack (nominal 11e-6) (substrate 1) (layer* HDPE 130e-6) (substrate 1))))

(define (export-hdpe-card-inc)
  (plot-response (title "hdpe card inc" "hdpe-card-inc")
    (samples 2000) (wavenumbers 4000 400) (incident 0 'T)
    (optical-stack (nominal 11e-6) (substrate 1) (layer* HDPE 14e-6) (substrate 1))))

(define (export-hdpe-card-abs-inc)
  (plot-response (title "hdpe card abs inc" "hdpe-card-abs-inc")
    (samples 600) (logarithmic (/ .01 5500) (/ .01 550)) (incident 0 '(- (ln T)))
    (optical-stack (nominal 11e-6) (substrate 1) (layer* HDPE 18e-6) (substrate 1))))

(define (export-hdpe-pas-inc)
  (plot-response (title "hdpe pas inc" "hdpe-pas-inc")
    (samples 1000) (wavenumbers 4000 450) (incident 0 '(- 1 A (/ R 2)))
    (optical-stack (nominal 11e-6) (substrate 1) (layer* HDPE 32e-6) (substrate 1))))

(define (export-polyeth-35-co)
  (plot-response (title "polyeth 35 co" "polyeth-35-co")
    (samples 1000) (wavenumbers 4000 400) (incident 0 'T)
    (optical-stack (substrate 1) (layer HDPE 35e-6) (substrate 1))))

(define (export-polyeth-35-inc)
  (plot-response (title "polyeth 35 inc" "polyeth-35-inc")
    (samples 1000) (wavenumbers 4000 400) (incident 0 'T)
    (optical-stack (substrate 1) (layer* HDPE 35e-6) (substrate 1))))

(define (export-hdpe-33-raw)
  (plot-response (title "hdpe 33 raw" "hdpe-33-raw")
    (samples 1200) (wavelengths 2e-6 20e-6 0e-6 20e-6) (incident 0 'T 'R)
    (range 0 1)
    (optical-stack (nominal 11e-6) (substrate 1) (layer hdpe 33e-6) (substrate 1))))

(define (export-hdpe-50-raw)
  (plot-response (title "hdpe 50 raw" "hdpe-50-raw")
    (samples 1200) (wavelengths 0.5e-6 15e-6) (incident 0 'T)
    (range 0 1)
    (optical-stack (nominal 11e-6) (substrate 1) (layer HDPE 50e-6) (substrate 1))))

(define (export-pe-33-inc)
  (plot-response (title "pe 33 inc" "pe-33-inc")
    (samples 300) (eVs 0.5 8.45) (incident 0 'T)
    (optical-stack (layer* PE 33e-6))))

"""


SCM_HEADER = r"""(require 'FreeSnell)

(define *export-outdir* #f)
(define *export-png-dims* '(512 256))
(define orig-title title)
(define orig-plot-response plot-response)

(define (title label filebase)
  (if *export-outdir*
      (list 'title label (string-append *export-outdir* "/baseline"))
      (orig-title label filebase)))

(define (strip-smooth-args args)
  (cond ((null? args) '())
        ((and (pair? (car args)) (eq? (caar args) 'smooth))
         (strip-smooth-args (cdr args)))
        (else (cons (car args) (strip-smooth-args (cdr args))))))

(define (plot-response . args)
  (if *export-outdir*
      (apply orig-plot-response
             (append (strip-smooth-args args)
                     (list (output-data (string-append *export-outdir* "/baseline.dat"))
                           (output-format 'png (car *export-png-dims*) (cadr *export-png-dims*)))))
      (apply orig-plot-response args)))

(define (export-with invoke-name scm-file png-dims)
  (set! *export-png-dims* png-dims)
  (load scm-file)
  (eval-string (string-append "(" invoke-name ")")))

(define (export-one tid outdir)
  (system (string-append "mkdir -p \"" outdir "\""))
  (set! *export-outdir* outdir)
  (dynamic-wind
      (lambda () #t)
      (lambda ()
        (cond
__DISPATCH__
          (else (error "unknown template id" tid))))
      (lambda () (set! *export-outdir* #f))))

"""


@dataclass(frozen=True)
class FreeSnellConfig:
    freesnell_dir: Path
    scm: Path
    slib: Path
    nk_rwb: Path


def validate_specs_against_json(specs: dict[str, CompareSpec], json_doc: dict[str, Any]) -> None:
    fs_ids = {t["id"] for t in json_doc["templates"] if str(t["id"]).startswith("fs_")}
    spec_ids = set(specs.keys())
    if spec_ids != fs_ids:
        missing = fs_ids - spec_ids
        extra = spec_ids - fs_ids
        raise SystemExit(f"spec/json mismatch: missing={sorted(missing)!r} extra={sorted(extra)!r}")
    for tid, spec in specs.items():
        for key in ("scm_file", "invoke", "x_axis", "quantity", "png_dims"):
            if key not in spec:
                raise SystemExit(f"{tid}: missing required spec field {key!r}")


def template_ids_in_json_order(
    json_doc: dict[str, Any],
    spec_ids: set[str],
) -> list[str]:
    return [t["id"] for t in json_doc["templates"] if t["id"] in spec_ids]


def _dispatch_clause(tid: str, spec: CompareSpec) -> str:
    w, h = spec.get("png_dims", (512, 256))
    scm_file = spec["scm_file"]
    invoke = spec["invoke"]
    if invoke.startswith("export/"):
        fn = invoke.split("/", 1)[1]
        return (
            f'          ((string=? tid "{tid}") '
            f"(set! *export-png-dims* (list {w} {h})) "
            f'(load "{scm_file}") (export-{fn}))'
        )
    return (
        f'          ((string=? tid "{tid}") '
        f'(export-with "{invoke}" "{scm_file}" (list {w} {h})))'
    )


def build_export_scm(specs: dict[str, CompareSpec] | None = None) -> str:
    specs = specs or FREESNEL_COMPARE_SPECS
    clauses = "\n".join(_dispatch_clause(tid, spec) for tid, spec in sorted(specs.items()))
    return SCM_HEADER.replace("__DISPATCH__", clauses) + FREESNEL_CUSTOM_EXPORT_SCM


def write_export_scm(build_dir: Path, specs: dict[str, CompareSpec] | None = None) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    path = build_dir / "export.scm"
    path.write_text(build_export_scm(specs), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template_ids": sorted((specs or FREESNEL_COMPARE_SPECS).keys()),
    }
    (build_dir / "export.scm.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def clean_output_dir(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    html = output / "fs_baseline_vs_toykits.html"
    if html.is_file():
        html.unlink()


def make_build_dir() -> Path:
    """Create a unique intermediate build directory under /tmp."""
    return Path(tempfile.mkdtemp(prefix="fs_compare_build_", dir="/tmp"))


def compare_metrics(ours: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    a = np.asarray(ours, dtype=float)
    b = np.asarray(baseline, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"rmse": float("nan"), "mae": float("nan"), "max_abs": float("nan"), "corr": float("nan")}
    d = a[mask] - b[mask]
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    max_abs = float(np.max(np.abs(d)))
    if a[mask].size < 2:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return {"rmse": rmse, "mae": mae, "max_abs": max_abs, "corr": corr}


@dataclass(frozen=True)
class CompareSeries:
    """Aligned baseline vs toykits on FreeSnell native x grid (所见即所比)."""

    x: np.ndarray
    baseline_y: np.ndarray
    toykits_y: np.ndarray


@dataclass(frozen=True)
class UiWavelengthSeries:
    """Optional UI wavelength panel: same N samples as CompareSeries, x relabeled to µm."""

    wl_um: np.ndarray
    toykits_y: np.ndarray
    baseline_y: np.ndarray


def assert_series_aligned(x: np.ndarray, baseline_y: np.ndarray, toykits_y: np.ndarray) -> None:
    n = len(x)
    if len(baseline_y) != n or len(toykits_y) != n:
        raise ValueError(
            f"compare series length mismatch: x={n}, baseline={len(baseline_y)}, toykits={len(toykits_y)}"
        )


def _baseline_y_multi_angle(
    arr: np.ndarray,
    angle_deg: float,
    incident_angles_deg: list[float] | None = None,
) -> np.ndarray:
    """Interpolate baseline.dat angle columns (col0 = wl; col1.. = traces per incident)."""
    angles = incident_angles_deg or [float(i) * 10.0 for i in range(9)]
    a = float(angle_deg)
    if a <= angles[0]:
        return arr[:, 1]
    if a >= angles[-1]:
        return arr[:, len(angles)]
    for i in range(len(angles) - 1):
        lo, hi = angles[i], angles[i + 1]
        if lo <= a <= hi:
            if hi == lo:
                return arr[:, i + 1]
            frac = (a - lo) / (hi - lo)
            return (1.0 - frac) * arr[:, i + 1] + frac * arr[:, i + 2]
    return arr[:, 1]


def parse_baseline_dat(
    path: Path,
    *,
    x_axis: str,
    quantity_col: int,
    spec: CompareSpec | None = None,
    template_json: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse FreeSnell ``baseline.dat``; x is native axis (µm, eV, cm⁻¹, or deg)."""
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < quantity_col + 1:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    arr = np.asarray(rows, dtype=float)
    x_raw = arr[:, 0]
    if spec and spec.get("multi_angle_baseline"):
        angle_deg = target_ang_deg_from_sim(template_json or {})
        y = _baseline_y_multi_angle(
            arr,
            angle_deg,
            spec.get("baseline_incident_angles_deg"),
        )
    else:
        y = arr[:, quantity_col]
    return x_native_from_raw(x_raw, x_axis), y


def x_native_from_raw(x_raw: np.ndarray, x_axis: str) -> np.ndarray:
    """Map ``baseline.dat`` column 0 to native comparison/plot x."""
    x = np.asarray(x_raw, dtype=float)
    if x_axis in ("wavelength_um", "log_wavelength_um"):
        return x * 1e6
    return x


def x_display_to_wl_um(x: np.ndarray, x_axis: str) -> np.ndarray:
    if x_axis in ("wavelength_um", "log_wavelength_um"):
        return x
    if x_axis == "eV":
        return EV_TO_UM / np.maximum(x, 1e-12)
    if x_axis == "wavenumber_cm":
        return 1e4 / np.maximum(x, 1e-12)
    raise ValueError(f"x_axis {x_axis} is not a wavelength sweep")


def needs_ui_wavelength_panel(x_axis: str) -> bool:
    return x_axis in ("eV", "wavenumber_cm")


def native_x_label(spec: CompareSpec) -> str:
    labels = {
        "wavelength_um": "Wavelength (µm)",
        "log_wavelength_um": "Wavelength (µm)",
        "eV": "Photon energy (eV)",
        "wavenumber_cm": "Wavenumber (cm⁻¹)",
        "angle_deg": "Angle (deg)",
    }
    return labels.get(str(spec.get("x_axis", "")), str(spec.get("x_axis", "")))


def build_compare_series(
    template_id: str,
    spec: CompareSpec,
    template_json: dict[str, Any],
    x_native: np.ndarray,
    baseline_y: np.ndarray,
) -> CompareSeries:
    x_axis = str(spec["x_axis"])
    toykits_y = run_toykits(template_id, spec, template_json, x_native, x_axis)
    assert_series_aligned(x_native, baseline_y, toykits_y)
    return CompareSeries(x=x_native, baseline_y=baseline_y, toykits_y=toykits_y)


def build_ui_wavelength_series(
    template_id: str,
    spec: CompareSpec,
    template_json: dict[str, Any],
    series: CompareSeries,
) -> UiWavelengthSeries:
    del template_id, template_json
    x_axis = str(spec["x_axis"])
    wl_um = x_display_to_wl_um(series.x, x_axis)
    assert_series_aligned(wl_um, series.baseline_y, series.toykits_y)
    return UiWavelengthSeries(wl_um=wl_um, toykits_y=series.toykits_y, baseline_y=series.baseline_y)


def resolve_quantity_col(
    spec: CompareSpec,
    template_json: dict[str, Any] | None = None,
) -> int:
    if "quantity_col" in spec:
        return int(spec["quantity_col"])
    if spec.get("multi_angle_baseline"):
        # baseline.dat: col 0 = wl; cols 1..9 = R at 0°, 10°, …, 80°
        angle_deg = target_ang_deg_from_sim(template_json or {})
        col = int(angle_deg / 10.0 + 0.5) + 1
        return max(1, min(9, col))
    return 1


def run_baseline(template_id: str, build_template_dir: Path, export_scm: Path, fs_cfg: FreeSnellConfig) -> None:
    build_template_dir.mkdir(parents=True, exist_ok=True)
    expr = f'(export-one "{template_id}" "{build_template_dir.resolve().as_posix()}")'
    env = os.environ.copy()
    slib = str(fs_cfg.slib)
    if not slib.endswith(os.sep):
        slib += os.sep
    env["SCHEME_LIBRARY_PATH"] = slib
    env["NK_DATABASE_PATH"] = str(fs_cfg.nk_rwb)
    cmd = [str(fs_cfg.scm), "-l", str(export_scm), "-e", expr]
    log_path = build_template_dir / "baseline_scm.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(fs_cfg.freesnell_dir),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=180,
        )
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"FreeSnell export failed for {template_id} (exit {proc.returncode}):\n{tail}")
    if not (build_template_dir / "baseline.dat").is_file():
        raise FileNotFoundError(f"baseline.dat not created for {template_id}")


def build_materials_db(template_json: dict[str, Any]) -> dict[str, Any]:
    import simulation  # noqa: F401
    from common import build_materials_db_from_path_keys

    return build_materials_db_from_path_keys(template_json.get("material_path_keys") or [])


_TOYKITS_LAYERS_CACHE: dict[tuple[str, str], Any] = {}


def _toykits_layers_cache_key(template_id: str, template_json: dict[str, Any]) -> tuple[str, str]:
    paths = template_json.get("material_path_keys") or []
    return template_id, json.dumps(paths, sort_keys=True)


def resolve_toykits_layers(template_id: str, template_json: dict[str, Any]) -> Any:
    """Resolve TMM layers once per template/material-path set (avoids repeated chdir)."""
    key = _toykits_layers_cache_key(template_id, template_json)
    cached = _TOYKITS_LAYERS_CACHE.get(key)
    if cached is not None:
        return cached

    import simulation  # noqa: F401
    artifacts = Path(os.environ["SIMULATION_ARTIFACTS_DIR"])
    prev_cwd = Path.cwd()
    os.chdir(artifacts)
    try:
        from filmstack_simulation.simulation import resolve_stack_with_layers

        formula = str((template_json.get("stack") or {}).get("formula") or "").strip()
        if not formula:
            raise ValueError(f"stack.formula is empty (preset {template_id})")
        materials_db = build_materials_db(template_json)
        _materials, _thicknesses, layers = resolve_stack_with_layers(formula, materials_db)
        _TOYKITS_LAYERS_CACHE[key] = layers
        return layers
    finally:
        os.chdir(prev_cwd)


def run_toykits(
    template_id: str,
    spec: CompareSpec,
    template_json: dict[str, Any],
    x_display: np.ndarray,
    x_axis: str,
) -> np.ndarray:
    from filmstack_simulation.sweep import angle_rt_polarized, compute_wavelength_vs_RT_data

    layers = resolve_toykits_layers(template_id, template_json)
    quantity = spec["quantity"]
    if x_axis == "angle_deg":
        fixed_wl = target_wl_um_from_sim(template_json)
        r_s, t_s, r_p, t_p = angle_rt_polarized(layers, fixed_wl, np.deg2rad(x_display))
        return quantity_from_rt(r_s, t_s, r_p, t_p, quantity)
    angle_deg = target_ang_deg_from_sim(template_json)
    wls = x_display_to_wl_um(x_display, x_axis)
    data = compute_wavelength_vs_RT_data(layers, wls, angle_deg)
    return quantity_from_rt(data["R_s"], data["T_s"], data["R_p"], data["T_p"], quantity)


_CJK_FONT_CANDIDATES = (
    "AR PL UMing CN",
    "AR PL SungtiL GB",
    "AR PL KaitiM GB",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Source Han Sans SC",
    "SimHei",
    "Microsoft YaHei",
    "Droid Sans Fallback",
)

_MATPLOTLIB_FONT_FAMILY: str | None = None
_MATPLOTLIB_CONFIGURED = False


def _pick_cjk_font_family() -> str | None:
    from matplotlib import font_manager

    if _MATPLOTLIB_FONT_FAMILY:
        return _MATPLOTLIB_FONT_FAMILY
    names = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in names:
            return candidate
    for f in font_manager.fontManager.ttflist:
        if any(token in f.name for token in ("UMing", "Sungti")):
            return f.name
    return None


def _configure_matplotlib() -> None:
    global _MATPLOTLIB_CONFIGURED
    if _MATPLOTLIB_CONFIGURED:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    _MATPLOTLIB_CONFIGURED = True


def _suptitle_fontproperties():
    from matplotlib import font_manager

    family = _pick_cjk_font_family()
    if family is None:
        return None
    return font_manager.FontProperties(family=family)


def plot_compare_png(
    label: str,
    spec: CompareSpec,
    out_path: Path,
    baseline_png: Path,
    series: CompareSeries,
    ui_wl: UiWavelengthSeries | None,
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    if re.search(r"[\u4e00-\u9fff]", label) and _pick_cjk_font_family() is None:
        print(
            f"warning: no CJK font for compare.png title; install fonts-noto-cjk "
            f"or pass --font-family: {label!r}",
            file=sys.stderr,
        )

    quantity = str(spec.get("quantity", "T"))
    x_label = native_x_label(spec)
    diff = series.toykits_y - series.baseline_y
    n_cols = 4 if ui_wl is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    if baseline_png.is_file():
        axes[0].imshow(plt.imread(baseline_png))
        axes[0].set_title("fs baseline (PNG)")
        axes[0].axis("off")
    else:
        axes[0].plot(series.x, series.baseline_y, "C0-", lw=1.2)
        axes[0].set_title("fs baseline")
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel(quantity)
        axes[0].grid(True, alpha=0.3)

    axes[1].plot(series.x, series.toykits_y, "C1-", lw=1.2)
    axes[1].set_title("toykits")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(quantity)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(series.x, diff, "C2-", lw=1.2)
    axes[2].axhline(0.0, color="k", lw=0.6)
    axes[2].set_title("toykits − baseline")
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(f"Δ{quantity}")
    axes[2].grid(True, alpha=0.3)

    if ui_wl is not None:
        axes[3].plot(ui_wl.wl_um, ui_wl.toykits_y, "C1-", lw=1.2)
        axes[3].set_title("toykits (UI wl)")
        axes[3].set_xlabel("Wavelength (µm)")
        axes[3].set_ylabel(quantity)
        axes[3].grid(True, alpha=0.3)

    suptitle_kw: dict[str, Any] = {"fontsize": 11}
    title_fp = _suptitle_fontproperties()
    if title_fp is not None:
        suptitle_kw["fontproperties"] = title_fp
    fig.suptitle(label, **suptitle_kw)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


RMSE_WARN_THRESHOLD = 1e-4
GRANULAR_NOTE = "掺杂模型"
COMMENT_INCOHERENT = "暂不支持非相干模型"

# FreeSnell export/cloud-* and export/ruby-* use layer* (opticompute.scm).

_N_AT_WL_CACHE: dict[tuple[str, float], float] = {}


def _n_at_wl_um(material: str, wl_um: float) -> float | None:
    key = (material, wl_um)
    if key in _N_AT_WL_CACHE:
        return _N_AT_WL_CACHE[key]
    path = FS_MATERIALS_DIR / f"{material}.yml"
    if not path.is_file():
        return None
    points: list[tuple[float, float]] = []
    in_data = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"\s*data:\s*\|", line):
            in_data = True
            continue
        if not in_data:
            continue
        if line and not line[:1].isspace():
            break
        parts = line.split()
        if len(parts) >= 2:
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    if wl_um <= points[0][0]:
        n = points[0][1]
    elif wl_um >= points[-1][0]:
        n = points[-1][1]
    else:
        n = points[-1][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= wl_um <= x1:
                t = (wl_um - x0) / (x1 - x0)
                n = y0 + t * (y1 - y0)
                break
    _N_AT_WL_CACHE[key] = n
    return n


def _dominant_optical_thickness(template_json: dict[str, Any]) -> float | None:
    """Optical thickness n·d/λ at sim.wl_from_um (dominant layer in stack formula)."""
    sim = template_json.get("sim") or {}
    wl_min = float(sim.get("wl_from_um", 0))
    if wl_min <= 0:
        return None
    formula = str((template_json.get("stack") or {}).get("formula", ""))
    if not formula:
        return None

    if re.search(r"\bair 1000\b", formula):
        return 1000.0 / wl_min

    best: float | None = None
    for m in re.finditer(r"\bfilm\s+([\d.]+)\s+([\d.]+)", formula):
        d_um, n = float(m.group(1)), float(m.group(2))
        ot = n * d_um / wl_min
        best = ot if best is None else max(best, ot)

    material_map = {
        "hdpe-be": "hdpe-be",
        "hdpe": "hdpe",
        "cor7059": "cor7059",
    }
    for m in re.finditer(r"\b(hdpe-be|hdpe|cor7059)\s+([\d.]+)\b", formula):
        mat_key, d_um = m.group(1), float(m.group(2))
        n = _n_at_wl_um(material_map[mat_key], wl_min)
        if n is None:
            continue
        ot = n * d_um / wl_min
        best = ot if best is None else max(best, ot)
    return best


def _format_optical_thickness(ot: float) -> str:
    if ot >= 100:
        return f"{ot:.0f}"
    if ot >= 10:
        text = f"{ot:.1f}"
        return text.rstrip("0").rstrip(".")
    return f"{ot:.2g}"


def _optical_thickness_comment(template_json: dict[str, Any]) -> str:
    ot = _dominant_optical_thickness(template_json)
    if ot is None:
        return ""
    return f"光学厚度过大({_format_optical_thickness(ot)})，放大了 nk 插值误差"


def _join_comment(*parts: str) -> str:
    items = [p.strip() for p in parts if p and p.strip()]
    return "；".join(items)


def _specific_mismatch_comment(
    template_id: str,
    spec: CompareSpec,
    label: str,
    template_json: dict[str, Any],
) -> str:
    if is_incoherent_template(
        template_id,
        template_json=template_json,
        label=label,
        invoke=str(spec.get("invoke", "")),
    ):
        return COMMENT_INCOHERENT
    return _optical_thickness_comment(template_json)


def resolve_comment(
    template_id: str,
    spec: CompareSpec,
    label: str,
    template_json: dict[str, Any],
    *,
    over_threshold: bool = False,
) -> str:
    specific = (
        _specific_mismatch_comment(template_id, spec, label, template_json)
        if over_threshold
        else ""
    )
    if spec.get("granular_ir"):
        return _join_comment(GRANULAR_NOTE, specific)
    return specific


def finalize_row_comment(
    row: dict[str, Any],
    spec: CompareSpec,
    label: str,
    template_id: str,
    template_json: dict[str, Any],
) -> None:
    """Set expected_mismatch/comment from RMSE after comparison."""
    rmse = row.get("rmse")
    if row.get("status") != "ok" or not isinstance(rmse, (int, float)) or not np.isfinite(rmse):
        return
    over_threshold = rmse >= RMSE_WARN_THRESHOLD
    row["expected_mismatch"] = over_threshold
    if over_threshold or spec.get("granular_ir"):
        row["comment"] = resolve_comment(
            template_id, spec, label, template_json, over_threshold=over_threshold
        )
    else:
        row.pop("comment", None)


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

    table_rows = []
    for row in rows:
        tid = row["id"]
        img_path = build_dir / "templates" / tid / "compare.png"
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
        elif row.get("expected_mismatch") or row.get("granular_ir"):
            comment = html.escape(str(row.get("comment", "")))
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
<title>基准测试 (Filmstack Simulation)</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1rem 2rem;color:#222}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:6px 8px;text-align:left;font-size:13px;vertical-align:top}}
tr:nth-child(even){{background:#f9f9f9}}
tr.warn{{background:#fff8e1}}
tr.failed{{background:#fdecea}}
.thumb{{max-width:180px;max-height:100px;object-fit:contain;border:1px solid #ccc;display:block}}
.meta{{color:#555;font-size:14px;margin-bottom:0.75rem}}
td:nth-child(6){{max-width:320px;font-size:12px;line-height:1.4}}
</style></head><body>
<h1>基准测试 (Filmstack Simulation)</h1>
<p class="meta">Built {built_at}， 共 {len(rows)} 项基准测试。</p>
<table><thead><tr>
<th>膜系配置</th><th>rmse</th><th>max_abs</th><th>corr</th><th>image</th><th>备注</th>
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
    (output / "fs_baseline_vs_toykits.html").write_text(html_doc, encoding="utf-8")


def process_template(
    template_id: str,
    spec: CompareSpec,
    template_json: dict[str, Any],
    build_dir: Path,
    export_scm: Path,
    fs_cfg: FreeSnellConfig,
) -> dict[str, Any]:
    build_template_dir = build_dir / "templates" / template_id
    compare_png = build_template_dir / "compare.png"
    label = str(template_json.get("label") or template_id)
    row: dict[str, Any] = {
        "id": template_id,
        "label": label,
        "granular_ir": bool(spec.get("granular_ir", False)),
        "status": "ok",
    }
    try:
        run_baseline(template_id, build_template_dir, export_scm, fs_cfg)
        dat_path = build_template_dir / "baseline.dat"
        if not dat_path.is_file():
            raise FileNotFoundError("missing baseline.dat")
        x_axis = spec["x_axis"]
        x_native, baseline_y = parse_baseline_dat(
            dat_path,
            x_axis=x_axis,
            quantity_col=resolve_quantity_col(spec, template_json),
            spec=spec,
            template_json=template_json,
        )
        series = build_compare_series(template_id, spec, template_json, x_native, baseline_y)
        metrics = compare_metrics(series.toykits_y, series.baseline_y)
        ui_wl = (
            build_ui_wavelength_series(template_id, spec, template_json, series)
            if needs_ui_wavelength_panel(x_axis)
            else None
        )
        np.savetxt(
            build_template_dir / "toykits.csv",
            np.column_stack([series.x, series.toykits_y]),
            delimiter=",",
            header="x,value",
            comments="",
        )
        if ui_wl is not None:
            np.savetxt(
                build_template_dir / "toykits_ui.csv",
                np.column_stack([ui_wl.wl_um, ui_wl.toykits_y]),
                delimiter=",",
                header="wavelength_um,value",
                comments="",
            )
        row.update(metrics)
        finalize_row_comment(row, spec, label, template_id, template_json)
        plot_compare_png(
            label,
            spec,
            compare_png,
            build_template_dir / "baseline.png",
            series,
            ui_wl,
        )
        (build_template_dir / "metrics.json").write_text(
            json.dumps(row, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = describe_template_failure(exc)
        build_template_dir.mkdir(parents=True, exist_ok=True)
        (build_template_dir / "error.txt").write_text(row["error"], encoding="utf-8")
    return row


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
        issues.append(f"缺少预编译材料库: {db_bin}（请先运行 build_toykits collect）")

    if not os.environ.get("SIMULATION_DATABASE_DIR", "").strip():
        issues.append("环境变量 SIMULATION_DATABASE_DIR 未设置")

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(artifacts) not in [p for p in ld.split(":") if p]:
        issues.append(
            f"LD_LIBRARY_PATH 未包含 {artifacts}"
            "（simulation.so 依赖的 .so 可能无法加载）"
        )

    if try_import and not issues:
        try:
            import simulation  # noqa: F401
        except ImportError as exc:
            issues.append(f"import simulation 失败: {exc}")

    return issues


def prepare_toykits_runtime() -> str | None:
    """Validate toykits artifact env, sync sys.path. Return error message or None."""
    issues = _check_toykits_runtime_issues(try_import=True)
    if issues:
        return "toykits 运行环境不可用：" + "；".join(issues)
    return None


def describe_template_failure(exc: BaseException) -> str:
    if isinstance(exc, ModuleNotFoundError):
        name = getattr(exc, "name", None)
        if name == "simulation":
            issues = _check_toykits_runtime_issues(try_import=False)
            if issues:
                return "toykits 运行环境不可用：" + "；".join(issues)
            return (
                "toykits 运行环境不可用：无法 import simulation"
                f"（{exc}；请先 source scripts/init-toykits-build-env.sh）"
            )
    msg = str(exc)
    if isinstance(exc, FileNotFoundError):
        if "baseline.dat" in msg:
            return f"FreeSnell baseline 未生成：{msg}"
    if isinstance(exc, RuntimeError) and "FreeSnell export failed" in msg:
        return msg
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"FreeSnell 导出超时（>{exc.timeout}s）"
    return msg


def diagnose_freesnell_config(
    *,
    freesnell_dir: Path = DEFAULT_FREESNELL_DIR,
    scm: Path = DEFAULT_SCM,
    slib: Path = DEFAULT_SLIB,
    nk_rwb: Path | None = None,
) -> tuple[FreeSnellConfig | None, str | None]:
    """Return (config, error_message). error_message is set when toolchain is unavailable."""
    resolved_nk = nk_rwb or (freesnell_dir / "nk.rwb")
    cfg = FreeSnellConfig(
        freesnell_dir=freesnell_dir,
        scm=scm,
        slib=slib,
        nk_rwb=resolved_nk,
    )
    missing: list[str] = []
    if not cfg.scm.is_file():
        missing.append(f"scm 可执行文件不存在: {cfg.scm}")
    if not cfg.nk_rwb.is_file():
        missing.append(f"nk.rwb 不存在: {cfg.nk_rwb}")
    if not cfg.freesnell_dir.is_dir():
        missing.append(f"FreeSnell 源码目录不存在: {cfg.freesnell_dir}")
    if not cfg.slib.is_dir():
        missing.append(f"Scheme 库目录不存在: {cfg.slib}")
    if missing:
        return None, "FreeSnell 工具链不可用：" + "；".join(missing)
    return cfg, None


def build_freesnell_compare_ui(
    output: Path | None = None,
    *,
    freesnell_dir: Path = DEFAULT_FREESNELL_DIR,
    scm: Path = DEFAULT_SCM,
    slib: Path = DEFAULT_SLIB,
    nk_rwb: Path | None = None,
    font_family: str | None = None,
) -> int:
    """Build fs baseline vs toykits comparison HTML.

    Returns:
        0 — all templates ok
        1 — toolchain/runtime unavailable or one or more template comparisons failed
    """
    global _MATPLOTLIB_FONT_FAMILY
    _MATPLOTLIB_FONT_FAMILY = font_family

    env_err = prepare_toykits_runtime()
    if env_err:
        print(f">>> 错误: {env_err}", file=sys.stderr)
        return 1

    fs_cfg, fs_err = diagnose_freesnell_config(
        freesnell_dir=freesnell_dir,
        scm=scm,
        slib=slib,
        nk_rwb=nk_rwb,
    )
    if fs_cfg is None:
        print(f">>> 错误: {fs_err}", file=sys.stderr)
        return 1

    json_doc = load_templates_json()
    validate_specs_against_json(FREESNEL_COMPARE_SPECS, json_doc)
    json_by_id = {t["id"]: t for t in json_doc["templates"]}
    template_ids = template_ids_in_json_order(json_doc, set(FREESNEL_COMPARE_SPECS.keys()))

    out = (output or DEFAULT_OUTPUT_DIR).resolve()
    clean_output_dir(out)
    build_dir = make_build_dir()
    print(f">>> 中间产物目录: {build_dir}", flush=True)
    try:
        export_scm = write_export_scm(build_dir)

        rows: list[dict[str, Any]] = []
        for tid in template_ids:
            if tid not in json_by_id:
                raise RuntimeError(f"template {tid} missing from filmstack_templates.json")
            print(f"[{tid}] ...", flush=True)
            row = process_template(
                tid,
                FREESNEL_COMPARE_SPECS[tid],
                json_by_id[tid],
                build_dir,
                export_scm,
                fs_cfg,
            )
            rows.append(row)
            if row["status"] != "ok":
                print(f"  -> failed: {row.get('error', '')}", flush=True)
            else:
                print(f"  -> {row['status']}", flush=True)

        built_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
        write_static_site(
            rows,
            out,
            build_dir,
            built_at=built_at,
        )
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    html_path = out / "fs_baseline_vs_toykits.html"
    failed_rows = [r for r in rows if r["status"] != "ok"]
    failed = len(failed_rows)
    print(f"Done: {len(rows)} templates, {failed} failed. HTML: {html_path}")
    if failed_rows:
        print("Failed templates:", ", ".join(r["id"] for r in failed_rows), file=sys.stderr)
        by_error: dict[str, list[str]] = {}
        for r in failed_rows:
            by_error.setdefault(str(r.get("error", "")), []).append(r["id"])
        for err, ids in by_error.items():
            print(f"  原因 ({len(ids)}): {err}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for fs_baseline_vs_toykits.html",
    )
    parser.add_argument("--freesnell-dir", type=Path, default=DEFAULT_FREESNELL_DIR)
    parser.add_argument("--scm", type=Path, default=DEFAULT_SCM)
    parser.add_argument("--slib", type=Path, default=DEFAULT_SLIB)
    parser.add_argument("--nk-rwb", type=Path, default=None)
    parser.add_argument(
        "--font-family",
        default=None,
        help="matplotlib font family for compare.png titles (e.g. 'Droid Sans Fallback')",
    )
    args = parser.parse_args(argv)

    code = build_freesnell_compare_ui(
        args.output,
        freesnell_dir=args.freesnell_dir,
        scm=args.scm,
        slib=args.slib,
        nk_rwb=args.nk_rwb,
        font_family=args.font_family,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
