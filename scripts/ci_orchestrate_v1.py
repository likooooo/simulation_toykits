#!/usr/bin/env python3
"""交错 CI 编排：每个阶段先编译再测试，再进入下一阶段。

详细输出写入 .ci/logs/<timestamp>/；控制台仅显示阶段与步骤 summary。

用法:
  python scripts/ci_orchestrate_v1.py all [--only STAGE] [--dry-run]
  python scripts/ci_orchestrate_v1.py build [--dry-run]
  python scripts/ci_orchestrate_v1.py test [--only TARGET] [--dry-run]

STAGE（all --only）:
  infra-release | infra-asan | sim-release | sim-asan | toykits
  infra | simulation_core | sim | toykits（组合别名）

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


def require_venv(venv: Path, *, hint: str) -> str:
    activate = venv / "bin" / "activate"
    if not activate.is_file():
        raise FileNotFoundError(f"缺少 venv: {venv}\n提示: {hint}")
    return f"source {shlex.quote(str(activate))}"


def infra_env_block() -> str:
    """infrastructure README: init-inf-build-env.sh → infrastructure/.venv."""
    return "\n".join(
        [
            f"source {shlex.quote(str(INIT_INF))}",
            require_venv(INFRA_VENV, hint=_INFRA_VENV_HINT),
        ]
    )


def sim_env_block(artifacts: str) -> str:
    """simulation_core README: init-simulation-build-env.sh → infrastructure/.venv."""
    return "\n".join(
        [
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


@dataclass(frozen=True)
class ShellStep:
    name: str
    cwd: Path
    body: str


def infra_release_build_step() -> ShellStep:
    prefix = infra_env_block()
    return ShellStep(
        "infra Release",
        INFRA_ROOT,
        f"{prefix}\npython3 {shlex.quote(str(BUILD_INF))} -B build --build-type Release --mkl on --cuda on",
    )


def infra_asan_build_step() -> ShellStep:
    prefix = infra_env_block()
    return ShellStep(
        "infra ASAN",
        INFRA_ROOT,
        f"{prefix}\npython3 {shlex.quote(str(BUILD_INF))} -B build-asan --build-type Debug --mkl on --cuda on",
    )


def sim_release_build_step() -> ShellStep:
    return ShellStep(
        "simulation_core Release",
        SIM_ROOT,
        (
            f"{sim_env_block('build')}\n"
            f"python3 {shlex.quote(str(BUILD_SIM))} -B build --build-type Release"
        ),
    )


def sim_asan_build_step() -> ShellStep:
    return ShellStep(
        "simulation_core ASAN",
        SIM_ROOT,
        (
            f"{sim_env_block('build-asan')}\n"
            f"python3 {shlex.quote(str(BUILD_SIM))} -B build-asan --build-type Debug"
        ),
    )


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


@dataclass(frozen=True)
class PipelineStage:
    """单个阶段：先执行 build_steps，再执行 test_target_names 对应测试。"""

    name: str
    build_steps: tuple[ShellStep, ...]
    test_target_names: tuple[str, ...]


def pipeline_stages() -> list[PipelineStage]:
    return [
        PipelineStage(
            "infra-release",
            (infra_release_build_step(),),
            ("infra-release",),
        ),
        PipelineStage(
            "infra-asan",
            (infra_asan_build_step(),),
            ("infra-asan",),
        ),
        PipelineStage(
            "sim-release",
            (sim_release_build_step(),),
            ("sim-release",),
        ),
        PipelineStage(
            "sim-asan",
            (sim_asan_build_step(),),
            ("sim-asan",),
        ),
        PipelineStage(
            "toykits",
            (toykits_deploy_step(),),
            ("toykits", "tmm"),
        ),
    ]


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


def _resolve_stages(names: Iterable[str] | None) -> list[str]:
    if not names:
        return [stage.name for stage in pipeline_stages()]
    expanded: list[str] = []
    for name in names:
        if name == "infra":
            expanded.extend(["infra-release", "infra-asan"])
        elif name in ("simulation_core", "sim"):
            expanded.extend(["sim-release", "sim-asan"])
        else:
            expanded.append(name)
    return expanded


def run_test_targets(names: Iterable[str], session: OrchestrateSession) -> None:
    catalog = test_targets()
    selected = _resolve_only(names)
    unknown = [name for name in selected if name not in catalog]
    if unknown:
        known = ", ".join(sorted(catalog))
        raise SystemExit(
            f"未知 test target: {', '.join(unknown)}（可选: {known}）"
        )
    for name in selected:
        bash_step(catalog[name].step, session)


def run_pipeline_stage(stage: PipelineStage, session: OrchestrateSession) -> None:
    log(f"=== Stage: {stage.name} (build → test) ===")
    for build_step in stage.build_steps:
        bash_step(build_step, session)
    run_test_targets(stage.test_target_names, session)
    log(f"=== Stage: {stage.name} 完成 ===")


def run_interleaved_pipeline(
    only: Iterable[str] | None, session: OrchestrateSession
) -> None:
    log("=== 交错 CI 流水线 ===")
    stages_by_name = {stage.name: stage for stage in pipeline_stages()}
    selected = _resolve_stages(only)
    unknown = [name for name in selected if name not in stages_by_name]
    if unknown:
        known = ", ".join(sorted(stages_by_name))
        raise SystemExit(
            f"未知 stage: {', '.join(unknown)}（可选: {known}）"
        )
    for name in selected:
        run_pipeline_stage(stages_by_name[name], session)
    log("=== 交错 CI 流水线 完成 ===")


def run_build_phase(*, session: OrchestrateSession) -> None:
    log("=== Phase Build（全量） ===")
    for stage in pipeline_stages():
        for build_step in stage.build_steps:
            bash_step(build_step, session)
    log("=== Phase Build 完成 ===")


def run_test_phase(
    only: Iterable[str] | None, session: OrchestrateSession
) -> None:
    log("=== Phase Test（全量） ===")
    run_test_targets(only or [], session)
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
    run_interleaved_pipeline(only=getattr(args, "only", None), session=session)
    return 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印步骤，不执行",
    )

    parser = argparse.ArgumentParser(
        description="交错 CI 编排（每阶段 build → test，再进入下一阶段）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印步骤，不执行",
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    build_parser = sub.add_parser(
        "build",
        parents=[common],
        help="全量编译（不跑测试）",
    )
    build_parser.set_defaults(func=cmd_build)

    test_parser = sub.add_parser(
        "test",
        parents=[common],
        help="全量测试（不编译；默认跑全部 test target）",
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
        help="交错执行：每阶段先 build 再 test",
    )
    all_parser.add_argument(
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
        ],
        help="仅运行指定阶段（可重复）；默认运行全部阶段",
    )
    all_parser.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
