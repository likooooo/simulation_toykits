#!/usr/bin/env python3
"""用 Docker bake 复现 infrastructure / simulation_core 的 GitHub CI（full）。

镜像:
  simulation-ci:deps          共享系统依赖 + 预装 /opt/ci/venv（clang/MKL/CUDA/gfortran/pip）
  simulation-infra-ci:ci      infrastructure bake 结果
  simulation-core-ci:ci       simulation_core bake 结果

用法（仓库根目录 simulation_toykits）:
  python scripts/ci_docker.py build-deps
  python scripts/ci_docker.py [--target infra|sim|all] [--jobs N]

说明:
  build-deps    构建共享 simulation-ci:deps（含 Python venv；不含源码编译）
  默认          docker build bake：源码 COPY 进镜像，构建成功即 CI 通过

禁止对宿主机源码树写盘：唯一 CI 路径为 bake。
venv 预装在 deps 的 /opt/ci/venv；build 在镜像内 /work/build*。
bake 复用该 venv，不再现场 pip install；修改
simulation_core/3rdparty/infrastructure/requirements.txt 后须重新执行
  python scripts/ci_docker.py build-deps

验收后可删 simulation-*-ci:ci 并 docker image prune -f；保留 simulation-ci:deps 作缓存。

环境变量:
  CI_DOCKER_INSTALL_MKL=0|1     deps 是否装 MKL（默认 Dockerfile=1）
  CI_DOCKER_INSTALL_CUDA=0|1    deps 是否装 CUDA（默认 Dockerfile=1）
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_ROOT = REPO_ROOT / "simulation_core"
INFRA_ROOT = SIM_ROOT / "3rdparty" / "infrastructure"

SHARED_DEPS_IMAGE = "simulation-ci:deps"
SHARED_DEPS_DOCKERFILE = REPO_ROOT / ".ci" / "docker" / "Dockerfile"
SHARED_DEPS_CONTEXT = REPO_ROOT / ".ci" / "docker"


@dataclass(frozen=True)
class ImageSpec:
    key: str
    name: str
    context: Path
    dockerfile: Path
    ci_script: Path
    image: str


SPECS: dict[str, ImageSpec] = {
    "infra": ImageSpec(
        key="infra",
        name="infrastructure",
        context=INFRA_ROOT,
        dockerfile=INFRA_ROOT / ".ci" / "docker" / "Dockerfile",
        ci_script=INFRA_ROOT / ".ci" / "docker" / "ci.sh",
        image="simulation-infra-ci",
    ),
    "sim": ImageSpec(
        key="sim",
        name="simulation_core",
        context=SIM_ROOT,
        dockerfile=SIM_ROOT / ".ci" / "docker" / "Dockerfile",
        ci_script=SIM_ROOT / ".ci" / "docker" / "ci.sh",
        image="simulation-core-ci",
    ),
}


def log(msg: str, *, err: bool = False) -> None:
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def require_docker() -> None:
    if not shutil.which("docker"):
        raise SystemExit("未找到 docker 命令")


def resolve_targets(target: str) -> list[ImageSpec]:
    if target == "all":
        return [SPECS["infra"], SPECS["sim"]]
    if target not in SPECS:
        raise SystemExit(f"未知 target: {target}")
    return [SPECS[target]]


def image_exists(tag: str) -> bool:
    probe = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def docker_build_shared_deps() -> None:
    req_src = INFRA_ROOT / "requirements.txt"
    if not req_src.is_file():
        raise SystemExit(
            f"缺少 {req_src}\n"
            "提示: cd simulation_core && git submodule update --init --recursive"
        )
    # 构建上下文仅 .ci/docker；把 requirements 同步进去供 COPY
    req_dst = SHARED_DEPS_CONTEXT / "requirements.txt"
    shutil.copy2(req_src, req_dst)
    log(f">>> docker build shared deps: {SHARED_DEPS_IMAGE}")
    cmd = [
        "docker",
        "build",
        "-f",
        str(SHARED_DEPS_DOCKERFILE),
        "-t",
        SHARED_DEPS_IMAGE,
    ]
    for env_key, arg_name in (
        ("CI_DOCKER_INSTALL_MKL", "INSTALL_MKL"),
        ("CI_DOCKER_INSTALL_CUDA", "INSTALL_CUDA"),
    ):
        val = os.environ.get(env_key)
        if val is not None:
            cmd.extend(["--build-arg", f"{arg_name}={val}"])
            log(f"    build-arg {arg_name}={val}")
    cmd.append(str(SHARED_DEPS_CONTEXT))
    try:
        subprocess.run(cmd, check=True)
    finally:
        req_dst.unlink(missing_ok=True)


def ensure_shared_deps() -> None:
    if image_exists(SHARED_DEPS_IMAGE):
        return
    log(f"缺少镜像 {SHARED_DEPS_IMAGE}，先执行 build-deps", err=True)
    docker_build_shared_deps()


def docker_bake_ci(spec: ImageSpec, *, jobs: int) -> None:
    ensure_shared_deps()
    tag = f"{spec.image}:ci"
    log(f">>> docker bake CI: {tag} (jobs={jobs}, base={SHARED_DEPS_IMAGE})")
    cmd = [
        "docker",
        "build",
        "-f",
        str(spec.dockerfile),
        "--build-arg",
        f"BASE_IMAGE={SHARED_DEPS_IMAGE}",
        "--build-arg",
        f"CI_JOBS={jobs}",
        "-t",
        tag,
        str(spec.context),
    ]
    subprocess.run(cmd, check=True)


def validate_paths(specs: list[ImageSpec] | None = None) -> None:
    issues: list[str] = []
    for path in (SHARED_DEPS_DOCKERFILE, SHARED_DEPS_CONTEXT):
        if not path.exists():
            issues.append(f"缺少路径: {path}")
    infra_req = INFRA_ROOT / "requirements.txt"
    if not infra_req.is_file():
        issues.append(
            f"缺少 simulation_core submodule: {infra_req}\n"
            "提示: cd simulation_core && git submodule update --init --recursive"
        )
    if specs is None:
        specs = list(SPECS.values())
    for spec in specs:
        for path in (spec.context, spec.dockerfile, spec.ci_script):
            if not path.exists():
                issues.append(f"缺少路径: {path}")
        dockerignore = spec.context / ".dockerignore"
        if not dockerignore.is_file():
            issues.append(f"缺少路径: {dockerignore}")
    if issues:
        for item in issues:
            log(item, err=True)
        raise SystemExit(1)


def cmd_build_deps(_args: argparse.Namespace) -> int:
    require_docker()
    validate_paths([])
    docker_build_shared_deps()
    return 0


def cmd_bake(args: argparse.Namespace) -> int:
    require_docker()
    specs = resolve_targets(args.target)
    validate_paths(specs)
    for spec in specs:
        docker_bake_ci(spec, jobs=args.jobs)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Docker bake 复现 infrastructure / simulation_core GitHub CI（full）"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_deps = sub.add_parser(
        "build-deps",
        help="构建共享 simulation-ci:deps（系统依赖 / MKL / CUDA）",
    )
    p_deps.set_defaults(func=cmd_build_deps)

    parser.add_argument(
        "--target",
        choices=("infra", "sim", "all"),
        default="all",
        help="bake 目标（默认 all）",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=5,
        help="并行编译任务数（默认 5；仅 bake）",
    )

    args = parser.parse_args()
    if args.cmd == "build-deps":
        return cmd_build_deps(args)
    return cmd_bake(args)


if __name__ == "__main__":
    sys.exit(main())
