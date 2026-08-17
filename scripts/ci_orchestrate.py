#!/usr/bin/env python3
"""两阶段 CI 编排：Phase Build（全量编译）→ Phase Test（全量测试）。

详细输出写入 .ci/logs/<timestamp>/；控制台仅显示阶段与步骤 summary。

用法:
  python scripts/ci_orchestrate.py build [--dry-run]
  python scripts/ci_orchestrate.py test [--only TARGET] [--dry-run]
  python scripts/ci_orchestrate.py all [--dry-run]

TARGET（test --only）:
  infra | simulation_core | toykits | tmm
  infra-release | infra-asan | sim-release | sim-asan
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ci_orchestrate_lib import OrchestrateSession, bash_step, log

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_ROOT = REPO_ROOT / "simulation_core"
INFRA_ROOT = SIM_ROOT / "3rdparty" / "infrastructure"

TOYKITS_VENV = REPO_ROOT / ".venv"
INFRA_VENV = INFRA_ROOT / ".venv"

INIT_INF = INFRA_ROOT / "scripts" / "init-inf-build-env.sh"
INIT_SIM = SIM_ROOT / "scripts" / "init-simulation-build-env.sh"
INIT_TOYKITS = REPO_ROOT / "scripts" / "init-toykits-build-env.sh"

BUILD_INF = INFRA_ROOT / "build_inf.py"
BUILD_SIM = SIM_ROOT / "scripts" / "build_simulation.py"
BUILD_TOYKITS = REPO_ROOT / "scripts" / "build_toykits.py"
CHECK_IMPORT = REPO_ROOT / "scripts" / "check_import_contract.py"
TMM_DIR = SIM_ROOT / "assets" / "ipynb" / "simulation" / "TMM"

INFRA_BUILD = INFRA_ROOT / "build"
INFRA_BUILD_ASAN = INFRA_ROOT / "build-asan"
SIM_BUILD = SIM_ROOT / "build"
SIM_BUILD_ASAN = SIM_ROOT / "build-asan"

_INFRA_VENV_HINT = (
    "cd simulation_core/3rdparty/infrastructure && "
    "python3 -m venv .venv && pip install -r requirements.txt"
)
_TOYKITS_VENV_HINT = (
    "python3 -m venv .venv && pip install -r requirements.txt "
    "(build 路径另需 requirements-build.txt)"
)


@dataclass(frozen=True)
class ShellStep:
    name: str
    cwd: Path
    body: str


def require_venv(venv: Path, *, hint: str) -> str:
    activate = venv / "bin" / "activate"
    if not activate.is_file():
        raise FileNotFoundError(f"缺少 venv: {venv}\n提示: {hint}")
    return f"source {shlex.quote(str(activate))}"


def golden_tools_export() -> str:
    """对照工具树：simulation_baseline_tools（含 simulation_golden_data submodule）。"""
    default = Path.home() / "repos" / "simulation_baseline_tools"
    return (
        f'export GENERATE_GOLDEN_TOOLS_DIR="${{GENERATE_GOLDEN_TOOLS_DIR:-{default}}}"'
    )


def infra_env_block() -> str:
    """infrastructure README: init-inf-build-env.sh → infrastructure/.venv."""
    return "\n".join(
        [
            golden_tools_export(),
            f"source {shlex.quote(str(INIT_INF))}",
            require_venv(INFRA_VENV, hint=_INFRA_VENV_HINT),
        ]
    )


def sim_env_block(artifacts: str) -> str:
    """simulation_core README: init-simulation-build-env.sh → infrastructure/.venv."""
    return "\n".join(
        [
            golden_tools_export(),
            f"source {shlex.quote(str(INIT_SIM))} {shlex.quote(artifacts)}",
            require_venv(INFRA_VENV, hint=_INFRA_VENV_HINT),
        ]
    )


def toykits_runtime_env_block() -> str:
    """toykits README / .github/workflows/ci.yml: .venv → init-toykits-build-env.sh."""
    return "\n".join(
        [
            require_venv(TOYKITS_VENV, hint=_TOYKITS_VENV_HINT),
            f"source {shlex.quote(str(INIT_TOYKITS))}",
        ]
    )


def toykits_build_env_block() -> str:
    """toykits README compile path: .venv only (build_toykits.py 内部 capture simulation init)."""
    return require_venv(TOYKITS_VENV, hint=_TOYKITS_VENV_HINT)


def tmm_env_block() -> str:
    """TMM 须 simulation_core YAML 材料库与 build 产物；仅在 TMM 子 shell 内 source。"""
    return sim_env_block("build")


def infra_build_steps() -> list[ShellStep]:
    prefix = infra_env_block()
    return [
        ShellStep(
            "infra Release",
            INFRA_ROOT,
            f"{prefix}\npython3 {shlex.quote(str(BUILD_INF))} -B build --build-type Release --mkl on --cuda on",
        ),
        ShellStep(
            "infra ASAN",
            INFRA_ROOT,
            f"{prefix}\npython3 {shlex.quote(str(BUILD_INF))} -B build-asan --build-type Debug --mkl on --cuda on",
        ),
    ]


def simulation_build_steps() -> list[ShellStep]:
    # Align with simulation_core/.github/workflows/build-ci-full.yml (--mkl on).
    return [
        ShellStep(
            "simulation_core Release",
            SIM_ROOT,
            (
                f"{sim_env_block('build')}\n"
                f"python3 {shlex.quote(str(BUILD_SIM))} -B build --build-type Release --mkl on"
            ),
        ),
        ShellStep(
            "simulation_core ASAN",
            SIM_ROOT,
            (
                f"{sim_env_block('build-asan')}\n"
                f"python3 {shlex.quote(str(BUILD_SIM))} -B build-asan --build-type Debug --mkl on"
            ),
        ),
    ]


def toykits_deploy_step() -> ShellStep:
    return ShellStep(
        "toykits deploy",
        REPO_ROOT,
        (
            f"{toykits_build_env_block()}\n"
            f"python3 {shlex.quote(str(BUILD_TOYKITS))}\n"
            "test -f .simulation_toolkits/assets/database.bin\n"
            "test ! -d .simulation_toolkits/assets/database\n"
            "test -f .simulation_toolkits/assets/fs_compare/fs_baseline_vs_toykits.html\n"
            "test -f .simulation_toolkits/simulation.so\n"
            "test -d .simulation_toolkits/simulation_plugins"
        ),
    )


def run_build_phase(*, session: OrchestrateSession) -> None:
    log("=== Phase Build ===")
    for step in (
        *infra_build_steps(),
        *simulation_build_steps(),
        toykits_deploy_step(),
    ):
        bash_step(step, session)
    log("=== Phase Build 完成 ===")


@dataclass(frozen=True)
class TestTarget:
    name: str
    aliases: tuple[str, ...]
    step: ShellStep


def test_targets() -> dict[str, TestTarget]:
    targets = [
        TestTarget(
            "infra-release",
            ("infra-release",),
            ShellStep(
                "infra ctest Release",
                INFRA_ROOT,
                f"{infra_env_block()}\ncd build\nctest -LE test_cuda --output-on-failure",
            ),
        ),
        TestTarget(
            "infra-asan",
            ("infra-asan",),
            ShellStep(
                "infra ctest ASAN",
                INFRA_ROOT,
                f"{infra_env_block()}\ncd build-asan\nctest -L test_asan --output-on-failure",
            ),
        ),
        TestTarget(
            "sim-release",
            ("sim-release",),
            ShellStep(
                "simulation_core ctest Release",
                SIM_ROOT,
                f"{sim_env_block('build')}\ncd build\nctest --output-on-failure -L test",
            ),
        ),
        TestTarget(
            "sim-asan",
            ("sim-asan",),
            ShellStep(
                "simulation_core ctest ASAN",
                SIM_ROOT,
                (
                    f"{sim_env_block('build-asan')}\n"
                    'export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:report_objects=0:malloc_context_size=0:fast_unwind_on_malloc=1}"\n'
                    f'export LSAN_OPTIONS="${{LSAN_OPTIONS:-suppressions={SIM_ROOT / "lsan.supp"}}}"\n'
                    "cd build-asan\n"
                    "ctest --output-on-failure -L test_asan"
                ),
            ),
        ),
        TestTarget(
            "toykits",
            ("toykits",),
            ShellStep(
                "toykits pytest",
                REPO_ROOT,
                (
                    f"{toykits_runtime_env_block()}\n"
                    'test -f "${SIMULATION_ARTIFACTS_DIR}/assets/database.bin"\n'
                    'python3 -c "import simulation; import filmstack_visualizer; print(\'OK\')"\n'
                    f"python3 {shlex.quote(str(CHECK_IMPORT))}\n"
                    "python3 -m pytest test/ -v --tb=short"
                ),
            ),
        ),
        TestTarget(
            "tmm",
            ("tmm",),
            ShellStep(
                "TMM scripts",
                REPO_ROOT,
                (
                    f"{tmm_env_block()}\n"
                    f"bash {shlex.quote(str(TMM_DIR / 'run_tmm.sh'))} test"
                ),
            ),
        ),
    ]
    return {target.name: target for target in targets}


def _resolve_only(names: Iterable[str] | None) -> list[str]:
    if not names:
        return [
            "infra-release",
            "infra-asan",
            "sim-release",
            "sim-asan",
            "toykits",
            "tmm",
        ]
    expanded: list[str] = []
    for name in names:
        if name == "infra":
            expanded.extend(["infra-release", "infra-asan"])
        elif name in ("simulation_core", "sim"):
            expanded.extend(["sim-release", "sim-asan"])
        else:
            expanded.append(name)
    return expanded


def run_test_phase(
    only: Iterable[str] | None, session: OrchestrateSession
) -> None:
    log("=== Phase Test ===")
    catalog = test_targets()
    selected = _resolve_only(only)
    unknown = [name for name in selected if name not in catalog]
    if unknown:
        known = ", ".join(sorted(catalog))
        raise SystemExit(
            f"未知 test target: {', '.join(unknown)}（可选: {known}）"
        )
    for name in selected:
        target = catalog[name]
        if name in ("infra", "simulation_core"):
            continue
        bash_step(target.step, session)
    log("=== Phase Test 完成 ===")


def validate_paths() -> list[str]:
    issues: list[str] = []
    required = [
        INIT_INF,
        INIT_SIM,
        INIT_TOYKITS,
        BUILD_INF,
        BUILD_SIM,
        BUILD_TOYKITS,
        CHECK_IMPORT,
        TMM_DIR,
    ]
    for path in required:
        if not path.exists():
            issues.append(f"缺少路径: {path}")
    for venv, hint in (
        (INFRA_VENV, _INFRA_VENV_HINT),
        (TOYKITS_VENV, _TOYKITS_VENV_HINT),
    ):
        if not (venv / "bin" / "activate").is_file():
            issues.append(f"缺少 venv: {venv}\n提示: {hint}")
    return issues


def cmd_build(args: argparse.Namespace) -> int:
    issues = validate_paths()
    if issues:
        for item in issues:
            log(item, err=True)
        return 1
    session = OrchestrateSession.start(dry_run=args.dry_run)
    run_build_phase(session=session)
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    issues = validate_paths()
    if issues:
        for item in issues:
            log(item, err=True)
        return 1
    session = OrchestrateSession.start(dry_run=args.dry_run)
    run_test_phase(only=getattr(args, "only", None), session=session)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    issues = validate_paths()
    if issues:
        for item in issues:
            log(item, err=True)
        return 1
    session = OrchestrateSession.start(dry_run=args.dry_run)
    run_build_phase(session=session)
    run_test_phase(only=None, session=session)
    return 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印步骤，不执行",
    )

    parser = argparse.ArgumentParser(description="两阶段 CI 编排（build 与 test 解耦）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印步骤，不执行",
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    build_parser = sub.add_parser(
        "build",
        parents=[common],
        help="Phase Build：infra → simulation_core → toykits deploy",
    )
    build_parser.set_defaults(func=cmd_build)

    test_parser = sub.add_parser(
        "test",
        parents=[common],
        help="Phase Test：ctest / pytest / TMM",
    )
    test_parser.add_argument(
        "--only",
        action="append",
        choices=[
            "infra",
            "infra-release",
            "infra-asan",
            "simulation_core",
            "sim",
            "sim-release",
            "sim-asan",
            "toykits",
            "tmm",
        ],
        help="仅运行指定测试目标（可重复）；默认运行全部",
    )
    test_parser.set_defaults(func=cmd_test)

    all_parser = sub.add_parser(
        "all",
        parents=[common],
        help="依次执行 build 与 test",
    )
    all_parser.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
