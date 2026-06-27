#!/usr/bin/env python3
"""simulation_toykits build and deploy CLI."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

RUNTIME_DIR = ".simulation_core"
DEFAULT_BUILD_REL = Path("simulation_core/build")

DEFAULT_HF_REPO = "git@hf.co:spaces/simulation-toykits/v1"
HF_DEPLOY_DEST = "/tmp/simulation-toykits-hf-deploy"
DEFAULT_DOCKER_IMAGE_TAG = "simulation-toykits:v1"
HF_ENDPOINT = "https://huggingface.co"
HF_README_CONFIG_NOTE = (
    "Check out the configuration reference at "
    "https://huggingface.co/docs/hub/spaces-config-reference"
)
HF_README_PREFIX = f"""---
title: V1
emoji: ⚡
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

{HF_README_CONFIG_NOTE}
"""
FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
HF_LFS_PATTERNS = (
    ".simulation_core/*.so",
    ".simulation_core/*.so.*",
    ".simulation_core/test_diffraction",
)

DEFAULT_STREAMLIT_PORT = 8052
DEFAULT_STREAMLIT_ADDRESS = "0.0.0.0"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def simulation_core_root() -> Path:
    return repo_root() / "simulation_core"


def runtime_dir() -> Path:
    return repo_root() / RUNTIME_DIR


def default_build_dir() -> Path:
    return (repo_root() / DEFAULT_BUILD_REL).resolve()


def run(
    cmd: list,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    *,
    check: bool = True,
    capture: bool = False,
    hint: str | None = None,
) -> subprocess.CompletedProcess:
    kwargs: dict = {
        "cwd": cwd or repo_root(),
        "env": {**os.environ, **(env or {})},
        "check": check,
        "text": True,
    }
    if capture:
        kwargs["capture_output"] = True
    try:
        return subprocess.run(cmd, **kwargs)
    except subprocess.CalledProcessError as exc:
        print(f">>> Command failed (exit {exc.returncode}): {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        if capture:
            if exc.stdout:
                print(exc.stdout.rstrip(), file=sys.stderr)
            if exc.stderr:
                print(exc.stderr.rstrip(), file=sys.stderr)
        if hint:
            print(f">>> Hint: {hint}", file=sys.stderr)
        raise


def log(msg: str, *, err: bool = False) -> None:
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def ensure_simulation_core() -> int:
    simulation_root = simulation_core_root()
    if not simulation_root.is_dir():
        print(f"错误: 未找到 simulation_core 子模块目录: {simulation_root}", file=sys.stderr)
        print("请先执行: git submodule update --init --recursive simulation_core", file=sys.stderr)
        return 1
    return 0


def copy_test_diffraction(path_to_build: Path, target: Path) -> None:
    src = path_to_build / "test_diffraction"
    if not src.is_file():
        print(f">>> 警告: 未找到 {src}，衍射角计算器将不可用", file=sys.stderr)
        return
    dst = target / "test_diffraction"
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | 0o111)
    print(f">>> 已复制 test_diffraction -> {dst}")


def run_build_pipeline(path_to_build: Path | None = None) -> int:
    if ensure_simulation_core() != 0:
        return 1

    build_dir = (path_to_build or default_build_dir()).resolve()
    target = runtime_dir()
    build_sim = simulation_core_root() / "scripts" / "build_simulation.py"
    if not build_sim.is_file():
        print(f"错误: 未找到 {build_sim}", file=sys.stderr)
        return 1

    print(f">>> build pipeline: build_dir={build_dir} -> {RUNTIME_DIR}/")
    sim_cmd = [
        sys.executable,
        str(build_sim),
        "-B",
        str(build_dir),
        "--build-type",
        "Release",
        "--collect",
        str(target),
    ]
    try:
        run(sim_cmd, cwd=simulation_core_root())
    except subprocess.CalledProcessError:
        init_sh = repo_root() / "scripts" / "init-toykits-build-env.sh"
        if init_sh.is_file():
            print(f">>> Hint: if build failed, try: source {init_sh}", file=sys.stderr)
        return 1

    copy_test_diffraction(build_dir, target)
    print(f">>> 运行时已收集: {target}")
    return 0


def local_runtime_env() -> dict[str, str]:
    rt = runtime_dir().resolve()
    return {
        "LD_LIBRARY_PATH": f"{rt}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
        "SIMULATION_DATABASE_DIR": str(rt / "assets" / "database"),
        "SIMULATION_ARTIFACTS_DIR": str(rt),
        "PYTHONPATH": f"{repo_root().resolve()}:{rt}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
    }


def _process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def _is_streamlit_pid(pid: int) -> bool:
    return "streamlit" in _process_cmdline(pid).lower()


def _listening_pids_on_port(port: int) -> list[int]:
    pids: list[int] = []
    if shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            for token in result.stdout.strip().split():
                try:
                    pids.append(int(token))
                except ValueError:
                    continue
    if not pids and shutil.which("ss"):
        result = subprocess.run(
            ["ss", "-H", "-tlnp", f"sport = :{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        for match in re.finditer(r"pid=(\d+)", result.stdout):
            pids.append(int(match.group(1)))
    return sorted(set(pids))


def stop_streamlit_on_port(port: int) -> None:
    for pid in _listening_pids_on_port(port):
        if not _is_streamlit_pid(pid):
            cmd = _process_cmdline(pid)
            print(
                f">>> 警告: 端口 {port} 被非 Streamlit 进程占用 (PID {pid})，跳过: {cmd}",
                file=sys.stderr,
            )
            continue
        subprocess.run(["kill", str(pid)], check=False)
        print(f">>> 已停止占用端口 {port} 的 Streamlit 进程 PID {pid}")


def start_local_server() -> int:
    port = DEFAULT_STREAMLIT_PORT
    address = DEFAULT_STREAMLIT_ADDRESS
    root = repo_root()
    app = root / "app.py"
    rt = runtime_dir()
    if not (rt / "simulation.so").is_file():
        print(f"错误: 未找到 {rt / 'simulation.so'}，请先运行 build_toykits 或 build_toykits local", file=sys.stderr)
        return 1
    if not app.is_file():
        print(f"错误: 未找到 {app}", file=sys.stderr)
        return 1
    if not shutil.which("streamlit"):
        print("错误: 未找到 streamlit 命令，请先 pip install -r requirements.txt", file=sys.stderr)
        return 1

    for pid in _listening_pids_on_port(port):
        if not _is_streamlit_pid(pid):
            cmd = _process_cmdline(pid)
            print(
                f"错误: 端口 {port} 已被非 Streamlit 进程占用 (PID {pid}): {cmd}\n"
                f"请先停止该进程，或改用其它端口。",
                file=sys.stderr,
            )
            return 1

    stop_streamlit_on_port(port)

    env = {**os.environ, **local_runtime_env()}
    cmd = [
        "streamlit",
        "run",
        str(app),
        f"--server.port={port}",
        f"--server.address={address}",
    ]
    print(f">>> 启动 Streamlit: http://localhost:{port}/")
    print(f">>> LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
    print(f">>> SIMULATION_DATABASE_DIR={env['SIMULATION_DATABASE_DIR']}")
    return subprocess.run(cmd, cwd=root, env=env).returncode


def resolve_database_source(explicit: str = "") -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    collected = runtime_dir() / "assets" / "database"
    if collected.is_dir():
        candidates.append(collected)
    env_root = os.environ.get("SIMULATION_DATABASE_DIR", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    seen: set[Path] = set()
    for raw in candidates:
        path = raw.resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / "oghma_database" / "materials").is_dir() and any(
            (path / "oghma_database" / "materials").iterdir()
        ):
            return path
    raise FileNotFoundError(
        "未找到可用的材料库源目录（需含非空 oghma_database/materials/）。"
        "请先 build_toykits 或设置 SIMULATION_DATABASE_DIR。"
    )


def prepare_database_bundle(dest: Path, source: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    release_script = source / "database_release.py"
    if not release_script.is_file():
        raise FileNotFoundError(
            f"材料库源 {source} 缺少 database_release.py；"
            "请指定 simulation_core/assets/database 或先完成 export。"
        )
    run(
        [
            sys.executable,
            str(release_script),
            "--dest",
            str(dest),
            "--clean",
        ],
        cwd=source,
    )
    materials = dest / "oghma_database" / "materials"
    if not materials.is_dir() or not any(materials.iterdir()):
        raise FileNotFoundError(f"database_release 未产出 oghma_database/materials: {dest}")
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f">>> 已 release 材料库: {source} -> {dest} ({size_mb:.1f} MiB)")


def verify_docker_image(image_tag: str) -> None:
    script = (
        "import os; os.chdir('/app'); "
        "os.environ.setdefault('SIMULATION_ARTIFACTS_DIR', '/app/.simulation_core'); "
        "os.environ.setdefault('PYTHONPATH', '/app:/app/.simulation_core'); "
        "import simulation; "
        "from simulation_database_parser import get_simulation_database; "
        "db = get_simulation_database(init=True); "
        "q = db.query(); "
        "assert q.keys, 'empty database tree'; "
        "assert db.local_path(), 'database path empty'; "
        "print('verify ok:', db.root_path(), list(q.keys)[:5])"
    )
    run(["docker", "run", "--rm", image_tag, "python", "-c", script])


@dataclass(frozen=True)
class RepoVersion:
    name: str
    path: Path
    commit: str
    date: str
    author: str
    email: str
    subject: str


def ensure_hf_endpoint() -> None:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)


def extract_readme_front_matter(text: str) -> str | None:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return None
    return match.group(0).rstrip() + "\n"


def resolve_hf_readme_prefix(cloned_readme: Path | None) -> str:
    if cloned_readme and cloned_readme.is_file():
        existing = extract_readme_front_matter(cloned_readme.read_text(encoding="utf-8"))
        if existing:
            log(">>> Preserving README YAML front matter from HF Space")
            return f"{existing.rstrip()}\n\n{HF_README_CONFIG_NOTE}\n"
    return HF_README_PREFIX.rstrip() + "\n"


def git_head_info(name: str, path: Path) -> RepoVersion:
    if not (path / ".git").exists():
        raise FileNotFoundError(f"Not a git repo: {path} ({name})")
    result = run(
        ["git", "-C", str(path), "log", "-1", "--format=%H|%ci|%an|%ae|%s"],
        capture=True,
    )
    commit, date, author, email, subject = result.stdout.strip().split("|", 4)
    return RepoVersion(
        name=name,
        path=path,
        commit=commit,
        date=date,
        author=author,
        email=email,
        subject=subject,
    )


def collect_repo_versions(root: Path) -> list[RepoVersion]:
    repos = [
        ("simulation_toykits", root),
        ("simulation", root / "simulation_core"),
        ("infrastructure", root / "simulation_core" / "3rdparty" / "infrastructure"),
    ]
    return [git_head_info(name, path) for name, path in repos]


def render_deploy_readme(root: Path, build_time: str, versions: list[RepoVersion]) -> str:
    lines = [
        "# simulation-toykits Hugging Face Space",
        "",
        f"**BUILD_TIME:** {build_time}",
        "",
        "Application source (`app.py`, `core/`, `pages/`, etc.) is cloned from GitHub "
        "when this Space builds its Docker image. This repository only stores the "
        "pre-built `.simulation_core/` runtime artifacts from CI.",
        "",
        "## Source commits (at deploy time)",
        "",
    ]
    for v in versions:
        try:
            rel_path = v.path.relative_to(root)
        except ValueError:
            rel_path = v.path
        lines.extend(
            [
                f"### {v.name}",
                "",
                f"- **Path:** `{rel_path}`",
                f"- **Commit:** `{v.commit}`",
                f"- **Author:** {v.author}",
                f"- **Email:** {v.email}",
                f"- **Date:** {v.date}",
                f"- **Subject:** {v.subject}",
                "",
            ]
        )
    return "\n".join(lines)


def clear_dest_workspace(dest: Path) -> None:
    for item in dest.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def write_dockerfile(hf_dockerfile: Path, dest: Path, build_time: str) -> None:
    content = hf_dockerfile.read_text(encoding="utf-8")
    new_content = re.sub(
        r"(ARG BUILD_TIME\s*=\s*).*",
        f'\\1"{build_time}"',
        content,
    )
    (dest / "Dockerfile").write_text(new_content, encoding="utf-8")


def setup_lfs_for_artifacts(dest: Path) -> None:
    run(["git", "lfs", "install", "--local"], cwd=dest)
    for pattern in HF_LFS_PATTERNS:
        run(["git", "lfs", "track", pattern], cwd=dest)
    log(">>> .gitattributes for HF LFS:")
    log((dest / ".gitattributes").read_text(encoding="utf-8"))


def ensure_git_xet() -> None:
    if subprocess.run(["git", "xet", "--version"], capture_output=True).returncode != 0:
        log(
            ">>> git-xet not found; install from "
            "https://huggingface.co/docs/hub/xet/using-xet-storage#git",
            err=True,
        )
        raise RuntimeError("git-xet is required for Hugging Face binary uploads")
    run(["git", "xet", "install"])


def ensure_git_identity(dest: Path, root: Path) -> None:
    name = os.environ.get("GIT_AUTHOR_NAME")
    email = os.environ.get("GIT_AUTHOR_EMAIL")
    if not name or not email:
        info = run(
            ["git", "-C", str(root), "log", "-1", "--format=%an|%ae"],
            capture=True,
        )
        name, email = info.stdout.strip().split("|", 1)
    run(["git", "config", "user.name", name], cwd=dest)
    run(["git", "config", "user.email", email], cwd=dest)


def git_commit_with_message(dest: Path, message: str, root: Path) -> None:
    ensure_git_identity(dest, root)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(message)
        msg_path = f.name
    try:
        run(["git", "commit", "-F", msg_path], cwd=dest)
    finally:
        Path(msg_path).unlink(missing_ok=True)


def deploy_hf(*, hf_repo: str, dest_path: str, dry_run: bool) -> int:
    root = repo_root()
    build_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dest = Path(dest_path)
    hf_dockerfile = root / "Dockerfile.hugging_face"
    artifacts = runtime_dir()

    if not hf_dockerfile.is_file():
        log(f">>> Missing {hf_dockerfile}", err=True)
        return 1

    if dest.exists():
        shutil.rmtree(dest)

    log(f">>> Cloning HF Space to {dest}")
    try:
        run(
            ["git", "clone", hf_repo, str(dest)],
            hint=(
                "ensure the public key paired with CI secrets.MY_PRIVATE_KEY_1 "
                "is added at https://huggingface.co/settings/keys"
            ),
        )
    except subprocess.CalledProcessError:
        log(f">>> git clone failed for {hf_repo}", err=True)
        return 1

    simulation_so = artifacts / "simulation.so"
    if not simulation_so.is_file():
        log(f">>> Missing {simulation_so}", err=True)
        return 1

    log(">>> Clearing HF workspace (keep .git)")
    cloned_readme = dest / "README.md"
    readme_prefix = resolve_hf_readme_prefix(cloned_readme)
    clear_dest_workspace(dest)

    log(f">>> Copying {artifacts} -> {dest / '.simulation_core'}")
    shutil.copytree(
        artifacts,
        dest / ".simulation_core",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    versions = collect_repo_versions(root)
    readme_body = render_deploy_readme(root, build_time, versions)
    commit_message = f"{readme_prefix.rstrip()}\n\n{readme_body}"
    (dest / "README.md").write_text(commit_message, encoding="utf-8")
    log(f">>> Wrote README.md (BUILD_TIME={build_time})")
    for v in versions:
        log(f"    {v.name}: {v.commit[:12]} {v.author} <{v.email}> {v.subject}")

    write_dockerfile(hf_dockerfile, dest, build_time)
    log(f">>> Wrote Dockerfile from Dockerfile.hugging_face with BUILD_TIME={build_time}")

    if dry_run:
        log(">>> dry-run: skip git commit/push")
        log(">>> commit message preview:")
        log(commit_message)
        setup_lfs_for_artifacts(dest)
        return 0

    ensure_git_xet()
    setup_lfs_for_artifacts(dest)
    run(["git", "add", ".gitattributes"], cwd=dest)
    run(["git", "add", "-A"], cwd=dest)
    lfs_list = run(["git", "lfs", "ls-files"], cwd=dest, capture=True)
    log(">>> git lfs ls-files:")
    log(lfs_list.stdout.strip() or "(empty)")
    status = run(["git", "status", "--porcelain"], cwd=dest, capture=True)
    if not status.stdout.strip():
        log(">>> No changes to deploy")
        return 0

    git_commit_with_message(dest, commit_message, root)
    ensure_hf_endpoint()
    push = subprocess.run(["git", "push"], cwd=dest, capture_output=True, text=True)
    if push.returncode != 0:
        log(f">>> Command failed (exit {push.returncode}): git push", err=True)
        if push.stdout:
            log(push.stdout.rstrip(), err=True)
        if push.stderr:
            log(push.stderr.rstrip(), err=True)
        combined = (push.stdout or "") + (push.stderr or "")
        if "HF_ENDPOINT" in combined or 'custom adapter "xet"' in combined:
            log(
                f">>> git-xet push failed: set HF_ENDPOINT={HF_ENDPOINT} and use "
                f"remote URL {DEFAULT_HF_REPO!r} (not ssh://…:22/…)",
                err=True,
            )
        elif "Permission denied" in combined or "Could not read from remote" in combined:
            log(
                ">>> git push auth failed: add the public key paired with "
                "CI secrets.MY_PRIVATE_KEY_1 at https://huggingface.co/settings/keys",
                err=True,
            )
        else:
            log(
                ">>> git push failed: ensure git-lfs + git-xet track .simulation_core "
                "binaries (.gitattributes) and Space write access",
                err=True,
            )
        return 1

    log(">>> Successfully pushed to Hugging Face Space")
    return 0


def cmd_build(_args: argparse.Namespace) -> int:
    return run_build_pipeline()


def cmd_local(_args: argparse.Namespace) -> int:
    code = run_build_pipeline()
    if code != 0:
        return code
    return start_local_server()


def cmd_docker(_args: argparse.Namespace) -> int:
    root = repo_root()
    simulation_root = simulation_core_root()

    code = run_build_pipeline()
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


def cmd_hf(args: argparse.Namespace) -> int:
    code = run_build_pipeline()
    if code != 0:
        return code
    return deploy_hf(
        hf_repo=DEFAULT_HF_REPO,
        dest_path=HF_DEPLOY_DEST,
        dry_run=args.dry_run,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="simulation_toykits 编译与部署（默认仅 build/collect）",
    )
    parser.set_defaults(func=cmd_build)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build", help="编译并 collect 到 .simulation_core/").set_defaults(func=cmd_build)
    sub.add_parser("local", help="build + 启动 Streamlit（脚本设置运行 env）").set_defaults(func=cmd_local)
    sub.add_parser("docker", help="build + 构建 Docker 镜像").set_defaults(func=cmd_docker)
    hf_parser = sub.add_parser("hf", help="build + 推送到 Hugging Face Space")
    hf_parser.add_argument("--dry-run", action="store_true", help="build 后仅预览 push，不 git push")
    hf_parser.set_defaults(func=cmd_hf)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
