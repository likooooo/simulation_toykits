#!/usr/bin/env python3
"""Build GranFilm Spheroid (optional) and export oblate/prolate baseline DR."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from granfilm.common.baseline import (
    default_coated_baseline_path,
    default_prolate_baseline_path,
    default_spheroid_baseline_path,
    default_yamaguchi_baseline_path,
    granfilm_dir,
    parse_spheroid_test_dat,
    run_spheroid_with_inc,
    save_baseline_npz,
)
from granfilm.oblate_prolate.case import (
    coated_inc_path,
    prolate_inc_path,
    spheroid_inc_path,
    yamaguchi_inc_path,
)
from granfilm.paths import SPHEROID_BASELINE_NPZ

_VARIANTS = {
    "oblate": {
        "inc": lambda _root: spheroid_inc_path(),
        "golden": default_spheroid_baseline_path,
    },
    "prolate": {
        "inc": lambda _root: prolate_inc_path(),
        "golden": default_prolate_baseline_path,
    },
    "yamaguchi": {
        "inc": lambda _root: yamaguchi_inc_path(),
        "golden": default_yamaguchi_baseline_path,
    },
    "coated": {
        "inc": lambda _root: coated_inc_path(),
        "golden": default_coated_baseline_path,
    },
}


def _patch_fortran_format_strings(spheroid_src: Path) -> None:
    for name in ("initialize_mod.f90", "spheroid.f90", "potential_mod.f90"):
        path = spheroid_src / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        patched = text.replace("(A\\)", "(A)").replace("(a\\)", "(a)").replace("(a,f6.2,a\\)", "(a,f6.2,a)")
        if patched != text:
            path.write_text(patched, encoding="utf-8")


def _patch_os_h(spheroid_src: Path) -> None:
    os_h = spheroid_src / "os.h"
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


def _write_stub_graphics(spheroid_src: Path) -> Path:
    stub = spheroid_src / "graphics_stub.f90"
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


def _write_stub_linsolver(spheroid_src: Path) -> Path:
    stub = spheroid_src / "linsolver_stub.f90"
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


def _write_stub_quadpack(spheroid_src: Path) -> Path:
    stub = spheroid_src / "quadpack_stub.f90"
    stub.write_text(
        """
subroutine dqag(f, a, b, epsabs, epsrel, key, result, abserr, neval, ier, limit, lenw, last, iwork, work)
  use global_definitions, only: wp
  use legendre_mod, only: gausslegendre
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
  use legendre_mod, only: gausslegendre
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


def _write_stub_potential(spheroid_src: Path) -> Path:
    stub = spheroid_src / "potential_stub.f90"
    stub.write_text(
        """
module Potential_mod
contains
  subroutine potential(bd, int, xz, ienergy, param)
    use global_definitions
    complex(wpc), intent(in) :: bd(:,:)
    type(integrals), intent(in) :: int(:,:,:)
    type(XZ_type), intent(in) :: xz(:,:)
    integer, intent(in) :: ienergy
    type(parameters), intent(in) :: param
  end subroutine potential
end module Potential_mod
""",
        encoding="utf-8",
    )
    return stub


def _write_stub_fmzm(spheroid_src: Path) -> Path:
    stub = spheroid_src / "fmzm_stub.f90"
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


def _write_stub_windows_menu(spheroid_src: Path) -> Path:
    stub = spheroid_src / "windows_menu_stub.f90"
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


def _patch_legendre_no_fmzm(spheroid_src: Path) -> None:
    path = spheroid_src / "legendre.f90"
    text = path.read_text(encoding="utf-8")
    old = """    Subroutine  Gauleg_dp(x1,x2,x,w)
      
      Use FMZM
      Implicit None
      Real(wp)      ::  x1,x2,x(:),w(:)
      Type(FM)      ::	EPS
      Integer       ::  i,j,m,n
      ! Use multiple precision for intermediate steps
      Type(FM)      ::	p1,p2,p3,pp,xl,xm,z,z1
      Type(FM)      ::	val1,val2
      
      n	  = size(x)
      ! Values to keep the precision
      EPS     = TO_FM('3e-20')
      val1    = TO_FM(1)/TO_FM(2)
      val2    = TO_FM(1)/TO_FM(4)
      m=(n+1)/2
      xm=(x2+x1)/2
      xl=(x2-x1)/2
      Do i=1,m
         z=cos(pi*(i-val2)/(n+val1))
1        continue
         p1=1
         p2=0
         Do j=1,n
            p3=p2
            p2=p1
            p1=((2*j-1)*z*p2-(j-1)*p3)/j
         Enddo
         pp=n*(z*p1-p2)/(z*z-1)
         z1=z
         z=z1-p1/pp
         If(abs(z-z1).gt.EPS) goto 1
         x(i)        =	TO_DP(xm-xl*z)
         x(n+1-i)    =	TO_DP(xm+xl*z)
         w(i)        =	TO_DP(2*xl/((1-z*z)*pp*pp))
         w(n+1-i)    =	w(i)
      Enddo
      Return
      
    End Subroutine Gauleg_dp"""
    new = """    Subroutine  Gauleg_dp(x1,x2,x,w)
      Implicit None
      Real(wp)      ::  x1,x2,x(:),w(:)
      Call Gauleg_sp(x1,x2,x,w)
      Return
    End Subroutine Gauleg_dp"""
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")


def build_spheroid(granfilm_root: Path) -> Path:
    spheroid_src = granfilm_root / "src" / "Spheroid"
    _patch_os_h(spheroid_src)
    _patch_fortran_format_strings(spheroid_src)
    _patch_legendre_no_fmzm(spheroid_src)
    _write_stub_graphics(spheroid_src)
    _write_stub_linsolver(spheroid_src)
    _write_stub_quadpack(spheroid_src)
    _write_stub_potential(spheroid_src)
    _write_stub_fmzm(spheroid_src)
    _write_stub_windows_menu(spheroid_src)

    srcs = [
        "global_def.f90",
        "legendre.f90",
        "graphics_stub.f90",
        "fmzm_stub.f90",
        "windows_menu_stub.f90",
        "quadpack_stub.f90",
        "initialize_mod.f90",
        "oblate_int_gauleg.f90",
        "oblate_int_quadpack.f90",
        "prolate_int_gauleg.f90",
        "prolate_int_quadpack.f90",
        "integral_mod.f90",
        "interaction_mod.f90",
        "linsolver_stub.f90",
        "oblate_mod.f90",
        "prolate_mod.f90",
        "optics_mod.f90",
        "potential_stub.f90",
        "write_mod.f90",
        "yamaguchi.f90",
        "spheroid.f90",
    ]

    objs = [s.replace(".f90", ".o") for s in srcs]
    exe = spheroid_src / "Spheroid"
    f90 = "gfortran"
    flags = ["-O2", "-fallow-argument-mismatch", "-std=legacy"]
    lapack_so = Path("/usr/lib/x86_64-linux-gnu/liblapack.so.3")
    libs = [str(lapack_so), "/usr/lib/x86_64-linux-gnu/libblas.so.3", "-lgfortran"]

    for s in srcs:
        o = s.replace(".f90", ".o")
        subprocess.run([f90, *flags, "-c", s, "-o", o], cwd=str(spheroid_src), check=True)
    subprocess.run([f90, *flags, "-o", str(exe), *objs, *libs], cwd=str(spheroid_src), check=True)
    return exe


def generate_variant_baseline(variant: str, root: Path, *, build: bool = False) -> Path:
    if build:
        build_spheroid(root)
    cfg = _VARIANTS[variant]
    inc = cfg["inc"](root)
    golden = cfg["golden"]()
    dat_path = run_spheroid_with_inc(inc, granfilm_root=root)
    golden.parent.mkdir(parents=True, exist_ok=True)
    golden.write_text(dat_path.read_text(encoding="utf-8"), encoding="utf-8")
    return golden


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Compile GranFilm Spheroid with gfortran")
    parser.add_argument(
        "--variant",
        choices=tuple(_VARIANTS),
        default="oblate",
        help="Which Spheroid.inc variant to run (default: oblate / SpheroidTest.dat)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output npz path (default: baseline_spheroid_dr.npz for oblate)",
    )
    parser.add_argument(
        "--granfilm-dir",
        type=Path,
        default=None,
        help=f"GranFilm root (default: $GENERATE_GOLDEN_TOOLS_DIR/GranFilm-v1.0 = {granfilm_dir()})",
    )
    args = parser.parse_args()
    root = args.granfilm_dir or granfilm_dir()
    output = args.output or SPHEROID_BASELINE_NPZ

    if args.build:
        try:
            build_spheroid(root)
            print(f"Built {root / 'src' / 'Spheroid' / 'Spheroid'}")
        except subprocess.CalledProcessError as exc:
            print(f"GranFilm Spheroid build failed: {exc}", file=sys.stderr)
            return 1

    try:
        golden = generate_variant_baseline(args.variant, root, build=False)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Baseline generation failed: {exc}", file=sys.stderr)
        return 1

    spec = parse_spheroid_test_dat(golden)
    save_baseline_npz(spec, output)
    print(f"Saved {golden} and npz -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
