#!/usr/bin/env python3
"""
从本地或 GitHub 构建 Docker 镜像，导出 tar；若指定 --remote 则 SCP 到该主机。

用法:
  python scripts/build_and_deploy.py [path-to-build] [--remote USER@HOST] [可选参数...]

示例（由简到繁）:
  # 仅构建并导出到 docker_artifacts/（已有 docker_artifacts 时）
  python scripts/build_and_deploy.py

  # 先准备 docker_artifacts 再构建并导出
  python scripts/build_and_deploy.py ../simulation/build

  # 构建并传到服务器（默认传到对方 home 目录）
  python scripts/build_and_deploy.py ../simulation/build --remote user@myserver

  # 指定镜像标签
  python scripts/build_and_deploy.py ../simulation/build --remote user@myserver --image-tag myapp:v2

  # 指定服务器上的存放目录
  python scripts/build_and_deploy.py ../simulation/build --remote user@myserver --remote-path /home/user/images

  # 用 Dockerfile（从 GitHub 拉代码）构建并传机
  python scripts/build_and_deploy.py ../simulation/build --remote user@myserver --source github
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run(cmd: list, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    env = {**os.environ, **(env or {})}
    subprocess.run(cmd, cwd=cwd or repo_root(), env=env, check=True)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="构建 Docker 镜像并可选 SCP 到服务器")
    parser.add_argument(
        "path_to_build",
        nargs="?",
        default=os.environ.get("PATH_TO_BUILD", ""),
        help="simulation 构建目录，用于生成 docker_artifacts；若已有可省略",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=os.environ.get("REMOTE", ""),
        help="SCP 目标，如 user@host；指定则导出后传到该主机",
    )
    parser.add_argument(
        "--image-tag",
        type=str,
        default="simulation-toykits:v1",
        help="镜像标签（默认: %(default)s）",
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        default="~",
        help="服务器上存放镜像 tar 的目录，~ 表示对方 home（默认: %(default)s）",
    )
    parser.add_argument(
        "--source",
        choices=["local", "github"],
        default="local",
        help="local 用 Dockerfile.local，github 用 Dockerfile（默认: %(default)s）",
    )
    args = parser.parse_args()

    do_scp = bool(args.remote)

    docker_artifacts = root / "docker_artifacts"
    so_path = docker_artifacts / "simulation.so"

    # 需要 docker_artifacts
    if args.path_to_build:
        path_to_build = Path(args.path_to_build)
        if not path_to_build.is_absolute():
            path_to_build = root / path_to_build
        print(f">>> 准备 docker_artifacts（path_to_build={path_to_build}）...")
        run(
            ["bash", str(root / "scripts" / "prepare_docker.sh"), str(path_to_build)],
            env={**os.environ, "path_to_build": str(path_to_build)},
        )
    elif not so_path.exists():
        print(
            "错误: 未指定 path-to-build 且 docker_artifacts 中无 simulation.so。",
            "请先运行: ./scripts/prepare_docker.sh <simulation_build_dir>",
            file=sys.stderr,
        )
        return 1
    else:
        print(">>> 使用已有 docker_artifacts 构建...")

    dockerfile = "Dockerfile.local" if args.source == "local" else "Dockerfile"
    context = str(root)
    print(f">>> 使用 {dockerfile} 构建镜像: {args.image_tag}")
    run(
        [
            "docker",
            "build",
            "-f", str(root / dockerfile),
            "-t", args.image_tag,
            context,
        ]
    )

    # 导出为 tar
    docker_artifacts.mkdir(parents=True, exist_ok=True)
    tar_name = args.image_tag.replace("/", "-").replace(":", "-") + ".tar"
    image_tar = docker_artifacts / tar_name
    print(f">>> 导出镜像到 {image_tar}")
    run(["docker", "save", args.image_tag, "-o", str(image_tar)])
    size = image_tar.stat().st_size / (1024 * 1024)
    print(f">>> 镜像已保存: {image_tar} ({size:.1f} MiB)")

    if do_scp:
        remote_path = f"{args.remote}:{args.remote_path.rstrip('/')}/{tar_name}"
        print(f">>> SCP 到 {remote_path}")
        run(["scp", str(image_tar), remote_path])
        print(f">>> 完成。在服务器上加载: docker load -i {args.remote_path}/{tar_name}")
    else:
        print(">>> 未指定 --remote，跳过 SCP")

    print("======== 全部完成 ========")
    return 0


if __name__ == "__main__":
    sys.exit(main())
