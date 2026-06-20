#!/usr/bin/env python3
"""
从本地或 GitHub 构建 Docker 镜像；默认仅构建，不导出 tar。可选 --export-tar 或 --remote 时再导出/部署。
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


def simulation_core_root() -> Path:
    return repo_root() / "simulation_core"


DEFAULT_BUILD_REL = Path("simulation_core/build")
OGHMA_DB_SUBDIRS = ("materials", "filters", "spectra", "shape", "morphology")


def resolve_database_source(explicit: str = "") -> Path:
    """Locate oghma_local-style root to bundle into simulation_core/assets/database."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.environ.get("SIMULATION_LOCAL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path("/mnt/c/Users/like/oghma_local"),
            Path.home() / ".oghma_local",
        ]
    )
    seen: set[Path] = set()
    for raw in candidates:
        path = raw.resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / "materials").is_dir() and any((path / "materials").iterdir()):
            return path
    raise FileNotFoundError(
        "未找到可用的材料库源目录（需含非空 materials/）。"
        "请设置 SIMULATION_LOCAL_ROOT 或传入 --database-source，"
        "例如: --database-source /mnt/c/Users/like/oghma_local"
    )


def prepare_database_bundle(dest: Path, source: Path) -> None:
    """Copy oghma database subdirs into simulation_core/assets/database for Docker."""
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in OGHMA_DB_SUBDIRS:
        src_dir = source / name
        if not src_dir.is_dir():
            continue
        dst_dir = dest / name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        copied.append(name)
    if "materials" not in copied:
        raise FileNotFoundError(f"材料库源 {source} 缺少 materials/ 目录")
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f">>> 已打包材料库: {source} -> {dest} ({', '.join(copied)}, {size_mb:.1f} MiB)")


def verify_docker_image(image_tag: str) -> None:
    """Smoke-test bundled database inside the built image."""
    script = (
        "import os; os.chdir('/app'); "
        "from core.simulation_database_ui import ensure_simulation_database_initialized; "
        "db = ensure_simulation_database_initialized(); "
        "names = list(db.database_names()); "
        "assert names, 'no databases'; "
        "mats = db.database('materials'); "
        "assert mats.local_path(), 'materials path empty'; "
        "print('verify ok:', db.root_path(), names)"
    )
    run(
        [
            "docker",
            "run",
            "--rm",
            image_tag,
            "python",
            "-c",
            script,
        ]
    )


def run(cmd: list, cwd: Optional[Path] = None, env: Optional[dict] = None, shell: bool = False) -> None:
    env = {**os.environ, **(env or {})}
    subprocess.run(cmd, cwd=cwd or repo_root(), env=env, check=True, shell=shell)


RUNTIME_DIR = ".simulation_core"


def clear_runtime_dir(runtime_dir: Path) -> None:
    if runtime_dir.exists():
        for child in runtime_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    runtime_dir.mkdir(parents=True, exist_ok=True)


def copy_test_diffraction(path_to_build: Path, runtime_dir: Path) -> None:
    src = path_to_build / "test_diffraction"
    if not src.is_file():
        print(f">>> 警告: 未找到 {src}，衍射角计算器将不可用", file=sys.stderr)
        return
    dst = runtime_dir / "test_diffraction"
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | 0o111)
    print(f">>> 已复制 test_diffraction -> {dst}")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="构建 Docker 镜像并可选 SCP 到服务器")
    parser.add_argument(
        "path_to_build",
        nargs="?",
        default=os.environ.get("PATH_TO_BUILD", ""),
        help=f"simulation_core 构建目录（默认: simulation_core/build），用于生成 {RUNTIME_DIR}",
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
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="跳过 cmake 编译，仅 collect artifacts（需 build 目录已有 simulation.so）",
    )
    parser.add_argument(
        "--docker-only",
        action="store_true",
        help=f"仅构建 Docker 镜像，不 collect（需 {RUNTIME_DIR}/ 已有 simulation.so）",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="启用 MKL CPU 后端（SIMULATION_LINK_MKL=ON，不链接 CUDA）",
    )
    parser.add_argument(
        "--full-build",
        action="store_true",
        help="链接 MKL 与 CUDA（SIMULATION_LINK_MKL/CUDA=ON）；缺依赖时 configure 失败",
    )
    parser.add_argument(
        "--database-source",
        type=str,
        default=os.environ.get("SIMULATION_LOCAL_ROOT", ""),
        help="打包进镜像的材料库源目录（simulation_database 根，含 materials/）；默认读 SIMULATION_LOCAL_ROOT",
    )
    parser.add_argument(
        "--skip-database-bundle",
        action="store_true",
        help="跳过复制 simulation_core/assets/database（需目标目录已就绪）",
    )
    parser.add_argument(
        "--skip-docker-verify",
        action="store_true",
        help="镜像构建后跳过容器内材料库冒烟测试",
    )
    args = parser.parse_args()

    if args.cpu_only and args.full_build:
        print("错误: --cpu-only 与 --full-build 不能同时使用", file=sys.stderr)
        return 1

    do_scp = bool(args.remote)
    do_export_tar = do_scp or args.export_tar

    runtime_dir = root / RUNTIME_DIR
    so_path = runtime_dir / "simulation.so"
    path_to_build: Optional[Path] = None
    simulation_root = simulation_core_root()

    should_collect = not args.docker_only and (
        bool(args.path_to_build or os.environ.get("PATH_TO_BUILD")) or not so_path.exists()
    )

    if should_collect:
        path_raw = args.path_to_build or os.environ.get("PATH_TO_BUILD") or str(DEFAULT_BUILD_REL)
        path_to_build = Path(path_raw)
        if not path_to_build.is_absolute():
            path_to_build = (root / path_to_build).resolve()
        else:
            path_to_build = path_to_build.resolve()
        if not simulation_root.is_dir():
            print(f"错误: 未找到 simulation_core 子模块目录: {simulation_root}", file=sys.stderr)
            print("请先执行: git submodule update --init --recursive simulation_core", file=sys.stderr)
            return 1
        init_script = simulation_root / "3rdparty" / "infrastructure" / "scripts" / "init-build-env.sh"
        if init_script.is_file():
            print(f">>> 提示: 若 cmake 失败，请先执行: source {init_script}")

        path_to_build.mkdir(parents=True, exist_ok=True)
        print(f">>> 准备 {RUNTIME_DIR}/（path_to_build={path_to_build}）...")
        clear_runtime_dir(runtime_dir)
        build_sim = simulation_root / "scripts" / "build_simulation.py"
        if not build_sim.is_file():
            print(f"错误: 未找到 {build_sim}", file=sys.stderr)
            return 1
        sim_cmd = [
            sys.executable,
            str(build_sim),
            "-B",
            str(path_to_build),
            "--build-type",
            "Release",
            "--collect",
            str(runtime_dir),
        ]
        if args.full_build:
            sim_cmd.append("--full-build")
        elif args.cpu_only:
            sim_cmd.append("--cpu-only")
        if args.skip_build:
            sim_cmd.append("--skip-build")
        run(sim_cmd, cwd=simulation_root)
        copy_test_diffraction(path_to_build, runtime_dir)
    elif not so_path.exists():
        print(
            f"错误: 未指定 path-to-build 且 {RUNTIME_DIR}/ 中无 simulation.so。",
            "请执行: python scripts/build_and_deploy.py simulation_core/build",
            file=sys.stderr,
        )
        return 1
    else:
        print(f">>> 使用已有 {RUNTIME_DIR}/ 构建...")

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
    if args.source == "local" and not args.skip_database_bundle:
        db_dest = simulation_root / "assets" / "database"
        db_source = resolve_database_source(args.database_source)
        prepare_database_bundle(db_dest, db_source)
    elif args.source == "local":
        db_dest = simulation_root / "assets" / "database"
        if not (db_dest / "materials").is_dir():
            print(
                f"错误: {db_dest}/materials 不存在，请先打包材料库或去掉 --skip-database-bundle",
                file=sys.stderr,
            )
            return 1
        print(f">>> 使用已有材料库: {db_dest}")

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

    if not args.skip_docker_verify:
        print(f">>> 验证镜像内材料库: {args.image_tag}")
        verify_docker_image(args.image_tag)

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
