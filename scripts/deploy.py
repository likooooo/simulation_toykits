#!/usr/bin/env python3
"""Unified deploy CLI: local | docker | hf."""

from __future__ import annotations

import argparse
import shutil
import sys

from _common import (
    DEFAULT_DOCKER_IMAGE_TAG,
    DEFAULT_HF_REPO,
    HF_DEPLOY_DEST,
    deploy_hf,
    default_build_dir,
    prepare_database_bundle,
    prepare_local_runtime,
    repo_root,
    resolve_database_source,
    run,
    simulation_core_root,
    start_local_server,
    verify_docker_image,
)


def cmd_local(_args: argparse.Namespace) -> int:
    code = prepare_local_runtime(default_build_dir())
    if code != 0:
        return code
    return start_local_server()


def cmd_docker(_args: argparse.Namespace) -> int:
    root = repo_root()
    simulation_root = simulation_core_root()

    code = prepare_local_runtime(default_build_dir())
    if code != 0:
        return code

    if not shutil.which("docker"):
        print(
            "错误: 未找到 docker 命令。请先安装 Docker 并确保在 PATH 中；",
            "若使用 WSL 2，请在 Docker Desktop 设置中启用 WSL 集成。",
            file=sys.stderr,
        )
        print("参见: https://docs.docker.com/go/wsl2/", file=sys.stderr)
        return 1

    db_dest = simulation_root / "assets" / "database"
    prepare_database_bundle(db_dest, resolve_database_source())

    image_tag = DEFAULT_DOCKER_IMAGE_TAG
    print(f">>> 使用 Dockerfile 构建镜像: {image_tag}")
    run(
        [
            "docker",
            "build",
            "-f",
            str(root / "Dockerfile"),
            "-t",
            image_tag,
            str(root),
        ]
    )

    print(f">>> 验证镜像内材料库: {image_tag}")
    verify_docker_image(image_tag)
    print("======== 全部完成 ========")
    return 0


def cmd_hf(_args: argparse.Namespace) -> int:
    return deploy_hf(
        hf_repo=DEFAULT_HF_REPO,
        dest_path=HF_DEPLOY_DEST,
        dry_run=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="simulation_toykits 部署工具（默认等同 local）",
    )
    parser.set_defaults(func=cmd_local)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("local", help="编译 + collect 到 .simulation_core/，并启动 Streamlit").set_defaults(
        func=cmd_local
    )
    sub.add_parser("docker", help="编译 + collect + 构建 Docker 镜像").set_defaults(func=cmd_docker)
    sub.add_parser("hf", help="推送 .simulation_core/ 到 Hugging Face Space").set_defaults(func=cmd_hf)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
