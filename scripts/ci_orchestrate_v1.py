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
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ci_orchestrate_lib import OrchestrateSession, bash_step, log
from ci_orchestrate_profiles import (
    KNOWN_TARGETS,
    build_ctest_cmd,
    resolve_profile,
    validate_profiles,
)

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
PROFILES_FILE = REPO_ROOT / ".ci" / "ci_profiles.yaml"
SIM_TEST_DEFINITION = SIM_ROOT / "cmake" / "test_definition.txt"
INFRA_TEST_DEFINITION = INFRA_ROOT / "cmake" / "test_definition.txt"

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


def collect_known_labels() -> set[str]:
    labels: set[str] = set()
    for path in (SIM_TEST_DEFINITION, INFRA_TEST_DEFINITION):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ";" not in line or line.startswith("cd "):
                continue
            for item in line.split(";"):
                item = item.strip()
                if item:
                    labels.add(item)
    return labels


@dataclass(frozen=True)
class ShellStep:
    name: str
    cwd: Path
    body: str


def infra_release_build_step(jobs: int) -> ShellStep:
    prefix = infra_env_block()
    return ShellStep(
        "infra Release",
        INFRA_ROOT,
        (
            f"{prefix}\npython3 {shlex.quote(str(BUILD_INF))} "
            f"-B build --build-type Release --mkl on --cuda on -j {jobs}"
        ),
    )


def infra_asan_build_step(jobs: int) -> ShellStep:
    prefix = infra_env_block()
    return ShellStep(
        "infra ASAN",
        INFRA_ROOT,
        (
            f"{prefix}\npython3 {shlex.quote(str(BUILD_INF))} "
            f"-B build-asan --build-type Debug --mkl on --cuda on -j {jobs}"
        ),
    )


def sim_release_build_step(jobs: int) -> ShellStep:
    return ShellStep(
        "simulation_core Release",
        SIM_ROOT,
        (
            f"{sim_env_block('build')}\n"
            f"python3 {shlex.quote(str(BUILD_SIM))} -B build --build-type Release -j {jobs}"
        ),
    )


def sim_asan_build_step(jobs: int) -> ShellStep:
    return ShellStep(
        "simulation_core ASAN",
        SIM_ROOT,
        (
            f"{sim_env_block('build-asan')}\n"
            f"python3 {shlex.quote(str(BUILD_SIM))} -B build-asan --build-type Debug -j {jobs}"
        ),
    )


def toykits_deploy_step(*, deploy_flags: tuple[str, ...], verify: tuple[str, ...]) -> ShellStep:
    flag_parts = [f"--{item}" for item in deploy_flags]
    deploy_cmd = (
        f"python3 {shlex.quote(str(BUILD_TOYKITS))} {' '.join(flag_parts)}".rstrip()
    )
    verify_cmds: list[str] = []
    checks = {
        "database.bin": "test -f .simulation_toolkits/assets/database.bin",
        "simulation.so": "test -f .simulation_toolkits/simulation.so",
        "simulation_plugins": "test -d .simulation_toolkits/simulation_plugins",
        "fs_compare_html": "test -f .simulation_toolkits/assets/fs_compare/fs_baseline_vs_toykits.html",
    }
    verify_cmds.append("test ! -d .simulation_toolkits/assets/database")
    for key in verify:
        if key in checks:
            verify_cmds.append(checks[key])
    return ShellStep(
        "toykits deploy",
        REPO_ROOT,
        f"{toykits_build_env_block()}\n{deploy_cmd}\n" + "\n".join(verify_cmds),
    )


@dataclass(frozen=True)
class TestTarget:
    name: str
    aliases: tuple[str, ...]
    step: ShellStep


def test_targets(profile_name: str) -> dict[str, TestTarget]:
    profile = resolve_profile(PROFILES_FILE, profile_name)
    targets = [
        TestTarget(
            "infra-release",
            ("infra-release",),
            ShellStep(
                "infra ctest Release",
                INFRA_ROOT,
                (
                    f"{infra_env_block()}\ncd build\n"
                    + build_ctest_cmd(
                        output_flag=profile.ctest_output,
                        include=profile.tests["infra-release"].ctest_include,
                        exclude=profile.tests["infra-release"].ctest_exclude,
                        jobs=profile.ctest_jobs,
                    )
                ),
            ),
        ),
        TestTarget(
            "infra-asan",
            ("infra-asan",),
            ShellStep(
                "infra ctest ASAN",
                INFRA_ROOT,
                (
                    f"{infra_env_block()}\ncd build-asan\n"
                    + build_ctest_cmd(
                        output_flag=profile.ctest_output,
                        include=profile.tests["infra-asan"].ctest_include
                        or ("test_asan",),
                        exclude=profile.tests["infra-asan"].ctest_exclude,
                        jobs=profile.ctest_jobs,
                    )
                ),
            ),
        ),
        TestTarget(
            "sim-release",
            ("sim-release",),
            ShellStep(
                "simulation_core ctest Release",
                SIM_ROOT,
                (
                    f"{sim_env_block('build')}\ncd build\n"
                    + build_ctest_cmd(
                        output_flag=profile.ctest_output,
                        include=profile.tests["sim-release"].ctest_include or ("test",),
                        exclude=profile.tests["sim-release"].ctest_exclude,
                        jobs=profile.ctest_jobs,
                    )
                ),
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
                    + build_ctest_cmd(
                        output_flag=profile.ctest_output,
                        include=profile.tests["sim-asan"].ctest_include or ("test_asan",),
                        exclude=profile.tests["sim-asan"].ctest_exclude,
                        jobs=profile.ctest_jobs,
                    )
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


def pipeline_stages(profile_name: str) -> list[PipelineStage]:
    profile = resolve_profile(PROFILES_FILE, profile_name)
    targets_for_stage = {
        "infra-release": ("infra-release",),
        "infra-asan": ("infra-asan",),
        "sim-release": ("sim-release",),
        "sim-asan": ("sim-asan",),
        "toykits": ("toykits", "tmm")
        if profile.tests["tmm"].enabled
        else ("toykits",),
    }
    build_for_stage = {
        "infra-release": (infra_release_build_step(profile.build_jobs),),
        "infra-asan": (infra_asan_build_step(profile.build_jobs),),
        "sim-release": (sim_release_build_step(profile.build_jobs),),
        "sim-asan": (sim_asan_build_step(profile.build_jobs),),
        "toykits": (
            toykits_deploy_step(
                deploy_flags=profile.deploy.flags,
                verify=profile.deploy.verify,
            ),
        ),
    }
    return [
        PipelineStage(name, build_for_stage[name], targets_for_stage[name])
        for name in profile.stages
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


def _resolve_stages(names: Iterable[str] | None, profile_name: str) -> list[str]:
    if not names:
        return [stage.name for stage in pipeline_stages(profile_name)]
    expanded: list[str] = []
    for name in names:
        if name == "infra":
            expanded.extend(["infra-release", "infra-asan"])
        elif name in ("simulation_core", "sim"):
            expanded.extend(["sim-release", "sim-asan"])
        else:
            expanded.append(name)
    return expanded


def run_test_targets(names: Iterable[str], session: OrchestrateSession, *, profile_name: str) -> None:
    catalog = test_targets(profile_name)
    selected = _resolve_only(names)
    unknown = [name for name in selected if name not in catalog]
    if unknown:
        known = ", ".join(sorted(catalog))
        raise SystemExit(
            f"未知 test target: {', '.join(unknown)}（可选: {known}）"
        )
    for name in selected:
        bash_step(catalog[name].step, session)


def run_pipeline_stage(stage: PipelineStage, session: OrchestrateSession, *, profile_name: str) -> None:
    log(f"=== Stage: {stage.name} (build → test) ===")
    for build_step in stage.build_steps:
        bash_step(build_step, session)
    run_test_targets(stage.test_target_names, session, profile_name=profile_name)
    log(f"=== Stage: {stage.name} 完成 ===")


def run_interleaved_pipeline(
    only: Iterable[str] | None, session: OrchestrateSession, *, profile_name: str
) -> None:
    log("=== 交错 CI 流水线 ===")
    profile = resolve_profile(PROFILES_FILE, profile_name)
    if profile.clean_build_dirs:
        for directory in (INFRA_BUILD, INFRA_BUILD_ASAN, SIM_BUILD, SIM_BUILD_ASAN):
            if directory.is_dir():
                log(f"[clean] rm -rf {directory}")
                shutil.rmtree(directory)
    stages_by_name = {stage.name: stage for stage in pipeline_stages(profile_name)}
    selected = _resolve_stages(only, profile_name)
    unknown = [name for name in selected if name not in stages_by_name]
    if unknown:
        known = ", ".join(sorted(stages_by_name))
        raise SystemExit(
            f"未知 stage: {', '.join(unknown)}（可选: {known}）"
        )
    for name in selected:
        run_pipeline_stage(stages_by_name[name], session, profile_name=profile_name)
    log("=== 交错 CI 流水线 完成 ===")


def run_build_phase(*, session: OrchestrateSession, profile_name: str) -> None:
    log("=== Phase Build（全量） ===")
    for stage in pipeline_stages(profile_name):
        for build_step in stage.build_steps:
            bash_step(build_step, session)
    log("=== Phase Build 完成 ===")


def run_test_phase(
    only: Iterable[str] | None, session: OrchestrateSession, *, profile_name: str
) -> None:
    log("=== Phase Test（全量） ===")
    selected = _resolve_only(only or [])
    profile = resolve_profile(PROFILES_FILE, profile_name)
    selected = [name for name in selected if name in KNOWN_TARGETS and profile.tests[name].enabled]
    run_test_targets(selected, session, profile_name=profile_name)
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
        PROFILES_FILE,
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
    run_build_phase(session=session, profile_name=args.profile)
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    issues = validate_paths()
    if issues:
        for item in issues:
            log(item, err=True)
        return 1
    session = OrchestrateSession.start(dry_run=args.dry_run)
    run_test_phase(only=getattr(args, "only", None), session=session, profile_name=args.profile)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    issues = validate_paths()
    if issues:
        for item in issues:
            log(item, err=True)
        return 1
    session = OrchestrateSession.start(dry_run=args.dry_run)
    run_interleaved_pipeline(
        only=getattr(args, "only", None),
        session=session,
        profile_name=args.profile,
    )
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    if args.profile_cmd == "list":
        for name in ("fast", "full", "nightly"):
            info = resolve_profile(PROFILES_FILE, name)
            log(f"{name}: {info.description}")
        return 0
    if args.profile_cmd == "show":
        info = resolve_profile(PROFILES_FILE, args.name)
        log(f"profile={info.name}")
        log(f"description={info.description}")
        log(f"stages={','.join(info.stages)}")
        log(f"build_jobs={info.build_jobs}, ctest_jobs={info.ctest_jobs}")
        log(f"deploy.flags={','.join(info.deploy.flags)}")
        log(f"deploy.verify={','.join(info.deploy.verify)}")
        for target in sorted(info.tests):
            rule = info.tests[target]
            log(
                f"test.{target}: enabled={rule.enabled} include={list(rule.ctest_include)} exclude={list(rule.ctest_exclude)}"
            )
        return 0
    if args.profile_cmd == "validate":
        issues = validate_profiles(PROFILES_FILE, known_labels=collect_known_labels())
        if issues:
            for issue in issues:
                log(f"[profile][ERROR] {issue}", err=True)
            return 1
        log("profile validate: OK")
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
    build_parser.add_argument(
        "--profile",
        choices=["fast", "full", "nightly"],
        default="full",
        help="构建 profile（默认 full）",
    )
    build_parser.set_defaults(func=cmd_build)

    test_parser = sub.add_parser(
        "test",
        parents=[common],
        help="全量测试（不编译；默认跑全部 test target）",
    )
    test_parser.add_argument(
        "--profile",
        choices=["fast", "full", "nightly"],
        default="full",
        help="测试 profile（默认 full）",
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
        "--profile",
        choices=["fast", "full", "nightly"],
        default="full",
        help="流水线 profile（默认 full）",
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

    profile_parser = sub.add_parser(
        "profile",
        help="查看或校验 profile 配置",
    )
    profile_sub = profile_parser.add_subparsers(dest="profile_cmd", required=True)
    profile_sub.add_parser("list", help="列出 profile")
    show_parser = profile_sub.add_parser("show", help="显示 profile 详情")
    show_parser.add_argument("name", choices=["fast", "full", "nightly"])
    profile_sub.add_parser("validate", help="校验 profile 配置")
    profile_parser.set_defaults(func=cmd_profile)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
