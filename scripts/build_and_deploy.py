#!/usr/bin/env python3
"""
从本地或 GitHub 构建 Docker 镜像；默认仅构建，不导出 tar。可选 --export-tar 或 --remote 时再导出/部署。

用法:
  python scripts/build_and_deploy.py [path-to-build] [--remote USER@HOST] [--export-tar] [可选参数...]

示例（由简到繁）:
  # 仅构建镜像（默认，不导出 tar）
  python scripts/build_and_deploy.py

  # 先准备 assets/lib（从 simulation 构建目录）再构建
  python scripts/build_and_deploy.py ../simulation/build

  # 构建并导出 tar 到 dist/
  python scripts/build_and_deploy.py --export-tar

  # 构建并传到服务器（会先导出 tar 再 SCP）
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
import shutil
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
        help="simulation 构建目录，用于生成 assets/lib；若已有可省略",
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
    parser.add_argument(
        "--export-tar",
        action="store_true",
        help="将镜像导出为 tar 到 dist/；未指定时默认不导出",
    )
    args = parser.parse_args()

    do_scp = bool(args.remote)
    do_export_tar = do_scp or args.export_tar

    assets_lib = root / "assets" / "lib"
    so_path = assets_lib / "simulation.so"

    if args.path_to_build:
        path_to_build = Path(args.path_to_build)
        if not path_to_build.is_absolute():
            path_to_build = (root / path_to_build).resolve()
        else:
            path_to_build = path_to_build.resolve()
        print(f">>> 准备 assets/lib（path_to_build={path_to_build}）...")
        print(">>> 在 build 目录下配置并编译 Release ...")
        run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=path_to_build)
        run(["cmake", "--build", ".", "--parallel"], cwd=path_to_build)
        if not (path_to_build / "simulation.so").exists():
            print(f"错误: 编译后未找到 {path_to_build / 'simulation.so'}", file=sys.stderr)
            return 1
        simulation_root = path_to_build.parent
        collect_script = simulation_root / "scripts" / "collect_simulation.py"
        if not collect_script.exists():
            print(f"错误: 未找到 {collect_script}，请确认 path_to_build 位于 simulation 仓库内。", file=sys.stderr)
            return 1
        print(">>> 收集 simulation.so 与依赖到 assets/lib ...")
        run(
            [sys.executable, str(collect_script), "-b", str(path_to_build), "-o", str(assets_lib)],
            cwd=root,
        )
    elif not so_path.exists():
        print(
            "错误: 未指定 path-to-build 且 assets/lib 中无 simulation.so。",
            "请传入 simulation 的 build 目录，例如: python scripts/build_and_deploy.py ../simulation/build",
            file=sys.stderr,
        )
        return 1
    else:
        print(">>> 使用已有 assets/lib 构建...")

    if not shutil.which("docker"):
        print(
            "错误: 未找到 docker 命令。请先安装 Docker 并确保在 PATH 中；",
            "若使用 WSL 2，请在 Docker Desktop 设置中启用 WSL 集成。",
            file=sys.stderr,
        )
        print("参见: https://docs.docker.com/go/wsl2/", file=sys.stderr)
        return 1

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

    if do_export_tar:
        dist_dir = root / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        tar_name = args.image_tag.replace("/", "-").replace(":", "-") + ".tar"
        image_tar = dist_dir / tar_name
        print(f">>> 导出镜像到 {image_tar}")
        run(["docker", "save", args.image_tag, "-o", str(image_tar)])
        size = image_tar.stat().st_size / (1024 * 1024)
        print(f">>> 镜像已保存: {image_tar} ({size:.1f} MiB)")

    if do_scp:
        remote_path = f"{args.remote}:{args.remote_path.rstrip('/')}/{tar_name}"
        print(f">>> SCP 到 {remote_path}")
        run(["scp", str(image_tar), remote_path])
        print(f">>> 完成。在服务器上加载: docker load -i {args.remote_path}/{tar_name}")
    elif not do_export_tar:
        print(">>> 未指定 --export-tar 或 --remote，跳过导出 tar")

    print("======== 全部完成 ========")
    return 0


if __name__ == "__main__":
    sys.exit(main())
