#!/usr/bin/env python3
"""Build GranFilm Sphere (optional) and export baseline DR to npz."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from granfilm.common.baseline import (
    GEOMETRY_BASELINE_PATHS,
    granfilm_dir,
    parse_sphere_test_dat,
    run_sphere_with_inc,
    save_baseline_npz,
)
from granfilm.paths import BASELINE_NPZ
from granfilm.sphere_island.case import inc_path_for_geometry

_GEOMETRY_NPZ = {
    "film": BASELINE_NPZ.parent / "baseline_film_dr.npz",
    "2film": BASELINE_NPZ.parent / "baseline_2film_dr.npz",
    "thin_cap": BASELINE_NPZ.parent / "baseline_thin_cap_dr.npz",
    "quadrupole": BASELINE_NPZ.parent / "baseline_quadrupole_dr.npz",
    "square": BASELINE_NPZ.parent / "baseline_square_dr.npz",
    "hexagonal": BASELINE_NPZ.parent / "baseline_hexagonal_dr.npz",
    "island_below": BASELINE_NPZ.parent / "baseline_island_below_dr.npz",
    "invariants": BASELINE_NPZ.parent / "baseline_invariants_dr.npz",
    "aspnes": BASELINE_NPZ.parent / "baseline_aspnes_dr.npz",
}


def generate_geometry_baseline(
    geometry: str,
    root: Path,
    *,
    build: bool = False,
) -> Path:
    """Run Fortran with inc template; golden .dat lives in GranFilm testing/."""
    if build:
        build_sphere(root)
    inc = inc_path_for_geometry(geometry)
    golden = GEOMETRY_BASELINE_PATHS[geometry]
    dat_path = run_sphere_with_inc(inc, granfilm_root=root)
    golden.write_text(dat_path.read_text(encoding="utf-8"), encoding="utf-8")
    return golden


def _patch_fortran_format_strings(sphere_src: Path) -> None:
    """Replace non-standard '(A\\)' no-newline formats for gfortran."""
    for name in ("initialize_mod.f90", "sphere.f90", "potential_mod.f90"):
        path = sphere_src / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        patched = text.replace("(A\\)", "(A)").replace("(a\\)", "(a)").replace("(a,f6.2,a\\)", "(a,f6.2,a)")
        if patched != text:
            path.write_text(patched, encoding="utf-8")


def _patch_os_h(sphere_src: Path) -> None:
    os_h = sphere_src / "os.h"
    text = os_h.read_text(encoding="utf-8")
    unix = (
        "character(len=1) :: directory_separator='/'    ! Unix path separator\n"
        "character(len=5) :: pgplot_device='/null'        ! headless stub\n"
    )
    if "directory_separator='/'" not in text or "directory_separator='\\'" in text:
        text = text.replace(
            "!character(len=1) :: directory_separator='/'    ! The Path separator ",
            "character(len=1) :: directory_separator='/'    ! The Path separator ",
        )
        text = text.replace(
            "!character(len=5) :: pgplot_device='/xwin'      ! Pgplot standard graphical device",
            "character(len=5) :: pgplot_device='/null'      ! headless stub",
        )
        text = text.replace(
            "character(len=1) :: directory_separator='\\'    ! For MS Windows",
            "!character(len=1) :: directory_separator='\\'    ! For MS Windows",
        )
        text = text.replace(
            "character(len=5) :: pgplot_device='/w9'        ! Pgplot standard graphical device",
            "!character(len=5) :: pgplot_device='/w9'        ! Pgplot standard graphical device",
        )
        if "directory_separator='/'" not in text:
            text = text.replace(
                "character(len=1) :: directory_separator='\\'    ! For MS Windows",
                unix,
            )
        os_h.write_text(text, encoding="utf-8")


def _write_stub_graphics(sphere_src: Path) -> Path:
    stub = sphere_src / "graphics_stub.f90"
    stub.write_text(
        """
module Graphics
  contains
    subroutine plot(xr, yr1, yr2, yr3, xlabel, ylabel, title, flag)
      real, intent(in) :: xr(:), yr1(:)
      real, intent(in), optional :: yr2(:), yr3(:)
      character(len=*), optional :: xlabel, ylabel, title
      logical, optional :: flag
    end subroutine plot
    subroutine contour(z)
      real, intent(in) :: z(:,:)
    end subroutine contour
end module Graphics
""",
        encoding="utf-8",
    )
    return stub


def _write_stub_linsolver(sphere_src: Path) -> Path:
    stub = sphere_src / "linsolver_stub.f90"
    stub.write_text(
        """
module linsolver_mod
  use global_definitions, only: wp, wpc
contains
  subroutine linsolver(a, b, eps)
    complex(wpc), intent(inout) :: a(:,:), b(:)
    real(wp), intent(in) :: eps
    complex(wpc), allocatable :: aa(:,:), bb(:)
    integer, allocatable :: pivot(:)
    integer :: m, n, info
    m = size(a, 1)
    n = size(a, 2)
    allocate(aa(m,n), bb(n), pivot(n))
    aa = a
    bb = b
    call zgetrf(m, n, aa, m, pivot, info)
    call zgetrs('n', n, 1, aa, m, pivot, bb, n, info)
    b = bb
    deallocate(aa, bb, pivot)
  end subroutine linsolver
end module linsolver_mod
""",
        encoding="utf-8",
    )
    return stub


def _write_stub_quadpack(sphere_src: Path) -> Path:
    stub = sphere_src / "quadpack_stub.f90"
    stub.write_text(
        """
subroutine dqag(f, a, b, epsabs, epsrel, key, result, abserr, neval, ier, limit, lenw, last, iwork, work)
  use global_definitions, only: wp
  use legendre, only: gausslegendre
  interface
    function f(x) result(y)
      import wp
      real(wp), intent(in) :: x
      real(wp) :: y
    end function f
  end interface
  real(wp), intent(in) :: a, b, epsabs, epsrel
  integer, intent(in) :: key, limit, lenw
  real(wp), intent(out) :: result, abserr
  integer, intent(out) :: neval, ier, last
  integer, intent(out) :: iwork(limit)
  real(wp), intent(out) :: work(lenw)
  integer, parameter :: nint = 250
  real(wp) :: x(nint), w(nint)
  integer :: i
  call gausslegendre(a, b, x, w)
  result = 0._wp
  do i = 1, nint
    result = result + w(i) * f(x(i))
  end do
  abserr = 0._wp
  neval = nint
  ier = 0
  last = nint
end subroutine dqag

subroutine dqagi(f, bound, inf, epsabs, epsrel, result, abserr, neval, ier, limit, lenw, last, iwork, work)
  use global_definitions, only: wp
  use legendre, only: gausslegendre
  interface
    function f(x) result(y)
      import wp
      real(wp), intent(in) :: x
      real(wp) :: y
    end function f
  end interface
  real(wp), intent(in) :: bound, epsabs, epsrel
  integer, intent(in) :: inf, limit, lenw
  real(wp), intent(out) :: result, abserr
  integer, intent(out) :: neval, ier, last
  integer, intent(out) :: iwork(limit)
  real(wp), intent(out) :: work(lenw)
  integer, parameter :: nint = 250
  real(wp) :: x(nint), w(nint)
  integer :: i
  if (inf == 1) then
    call gausslegendre(bound, bound + 100._wp, x, w)
  else
    call gausslegendre(bound - 100._wp, bound, x, w)
  end if
  result = 0._wp
  do i = 1, nint
    result = result + w(i) * f(x(i))
  end do
  abserr = 0._wp
  neval = nint
  ier = 0
  last = nint
end subroutine dqagi
""",
        encoding="utf-8",
    )
    return stub


def _write_stub_potential(sphere_src: Path) -> Path:
    stub = sphere_src / "potential_stub.f90"
    stub.write_text(
        """
module Potential_mod
contains
  subroutine potential(bd, ienergy, int, int_tr1, zeta, param)
    use global_definitions
    complex(wpc), intent(in) :: bd(:,:)
    integer, intent(in) :: ienergy
    type(integrals), intent(in) :: int(:,:,:), int_tr1(:,:,:)
    real(wp), intent(in) :: zeta(:,:,:)
    type(parameters), intent(in) :: param
  end subroutine potential
end module Potential_mod
""",
        encoding="utf-8",
    )
    return stub


def _write_stub_fmzm(sphere_src: Path) -> Path:
    stub = sphere_src / "fmzm_stub.f90"
    stub.write_text(
        """
module FMZM
contains
  subroutine ZM_SET(digits)
    integer, intent(in) :: digits
  end subroutine ZM_SET
end module FMZM
""",
        encoding="utf-8",
    )
    return stub


def _write_stub_windows_menu(sphere_src: Path) -> Path:
    stub = sphere_src / "windows_menu_stub.f90"
    stub.write_text(
        """
module Windows_menu_mod
contains
  subroutine Windows_menu
  end subroutine Windows_menu
end module Windows_menu_mod
""",
        encoding="utf-8",
    )
    return stub


def build_sphere(granfilm_root: Path) -> Path:
    sphere_src = granfilm_root / "src" / "Sphere"
    _patch_os_h(sphere_src)
    _patch_fortran_format_strings(sphere_src)
    _write_stub_graphics(sphere_src)
    _write_stub_linsolver(sphere_src)
    _write_stub_quadpack(sphere_src)
    _write_stub_potential(sphere_src)
    _write_stub_fmzm(sphere_src)
    _write_stub_windows_menu(sphere_src)

    srcs = [
        "global_def.f90",
        "legendre.f90",
        "graphics_stub.f90",
        "fmzm_stub.f90",
        "windows_menu_stub.f90",
        "initialize_mod.f90",
        "integral_mod.f90",
        "interaction_mod.f90",
        "linsolver_stub.f90",
        "matrix_system_mod.f90",
        "optics_mod.f90",
        "boundary_mod.f90",
        "quadpack_stub.f90",
        "potential_stub.f90",
        "write_mod.f90",
        "sphere.f90",
    ]
    objs = [s.replace(".f90", ".o") for s in srcs]
    exe = sphere_src / "Sphere"
    f90 = "gfortran"
    flags = ["-O2", "-fallow-argument-mismatch", "-std=legacy"]
    lapack_so = Path("/usr/lib/x86_64-linux-gnu/liblapack.so.3")
    libs = [str(lapack_so), "/usr/lib/x86_64-linux-gnu/libblas.so.3", "-lgfortran"]

    for s in srcs:
        o = s.replace(".f90", ".o")
        subprocess.run(
            [f90, *flags, "-c", s, "-o", o],
            cwd=str(sphere_src),
            check=True,
        )
    subprocess.run([f90, *flags, "-o", str(exe), *objs, *libs], cwd=str(sphere_src), check=True)
    return exe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Compile GranFilm Sphere with gfortran")
    parser.add_argument(
        "--geometry",
        choices=("island", "island_below", "film", "2film", "thin_cap", "quadrupole", "square", "hexagonal", "invariants", "aspnes"),
        default="island",
        help="Which Sphere.inc variant to run (default: island / SphereTest.dat)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output npz path (default depends on --geometry)",
    )
    parser.add_argument(
        "--granfilm-dir",
        type=Path,
        default=None,
        help=f"GranFilm root (default: GRANFILM_DIR or {granfilm_dir()})",
    )
    args = parser.parse_args()
    root = args.granfilm_dir or granfilm_dir()
    output = args.output or _GEOMETRY_NPZ.get(args.geometry, BASELINE_NPZ)

    if args.geometry in {
        "island",
        "island_below",
        "film",
        "2film",
        "thin_cap",
        "quadrupole",
        "square",
        "hexagonal",
        "invariants",
        "aspnes",
    }:
        if args.build:
            try:
                build_sphere(root)
                print(f"Built {root / 'src' / 'Sphere' / 'Sphere'}")
            except subprocess.CalledProcessError as exc:
                print(f"GranFilm build failed: {exc}", file=sys.stderr)
                return 1
        try:
            golden = generate_geometry_baseline(args.geometry, root, build=False)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Baseline generation failed: {exc}", file=sys.stderr)
            return 1
        spec = parse_sphere_test_dat(golden)
        save_baseline_npz(spec, output)
        print(f"Saved {golden} and npz -> {output}")
        return 0

    raise RuntimeError(f"unsupported geometry: {args.geometry!r}")


if __name__ == "__main__":
    raise SystemExit(main())
