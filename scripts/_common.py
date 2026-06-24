"""Shared helpers for scripts/deploy.py."""

from __future__ import annotations

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
OGHMA_DB_SUBDIRS = ("materials", "filters", "spectra", "shape", "morphology")

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


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def simulation_core_root() -> Path:
    return repo_root() / "simulation_core"


def runtime_dir() -> Path:
    return repo_root() / RUNTIME_DIR


def plugin_source() -> Path:
    return (
        simulation_core_root()
        / "3rdparty"
        / "infrastructure"
        / "py_core_plugins"
    )


def simulation_plugins_source() -> Path:
    return simulation_core_root() / "simulation_plugins"


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


def sync_dir(src: Path, dst: Path) -> int:
    if not src.is_dir():
        print(f"错误: 源目录不存在: {src}", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    src_names = {p.name for p in src.iterdir() if p.name != "__pycache__"}
    for existing in list(dst.iterdir()):
        if existing.name == "__pycache__":
            continue
        if existing.name not in src_names:
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
            print(f"  删除过时项: {dst / existing.name}")
    copied = 0
    for path in sorted(src.iterdir()):
        if path.name == "__pycache__":
            continue
        target = dst / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(path, target)
        copied += 1
        print(f"  {path.name} -> {target}")
    print(f">>> 已同步 {copied} 项到 {dst}")
    return 0


def sync_py_core_plugins(*, to_build: bool, to_runtime: bool) -> int:
    src = plugin_source()
    code = 0
    if to_runtime:
        code |= sync_dir(src, runtime_dir() / "py_core_plugins")
    if to_build:
        code |= sync_dir(src, simulation_core_root() / "build" / "py_core_plugins")
    return code


def sync_simulation_plugins(*, to_build: bool, to_runtime: bool) -> int:
    src = simulation_plugins_source()
    if not src.is_dir():
        print(f"错误: simulation_plugins 源目录不存在: {src}", file=sys.stderr)
        return 1
    code = 0
    if to_runtime:
        code |= sync_dir(src, runtime_dir() / "simulation_plugins")
    if to_build:
        code |= sync_dir(src, simulation_core_root() / "build" / "simulation_plugins")
    return code


def clear_runtime_dir(target: Path) -> None:
    if target.exists():
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    target.mkdir(parents=True, exist_ok=True)


def copy_test_diffraction(path_to_build: Path, target: Path) -> None:
    src = path_to_build / "test_diffraction"
    if not src.is_file():
        print(f">>> 警告: 未找到 {src}，衍射角计算器将不可用", file=sys.stderr)
        return
    dst = target / "test_diffraction"
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | 0o111)
    print(f">>> 已复制 test_diffraction -> {dst}")


def ensure_simulation_core() -> int:
    simulation_root = simulation_core_root()
    if not simulation_root.is_dir():
        print(f"错误: 未找到 simulation_core 子模块目录: {simulation_root}", file=sys.stderr)
        print("请先执行: git submodule update --init --recursive simulation_core", file=sys.stderr)
        return 1
    init_script = simulation_root / "3rdparty" / "infrastructure" / "scripts" / "init-inf-build-env.sh"
    if init_script.is_file():
        print(f">>> 提示: 若 cmake 失败，请先执行: source {init_script}")
    return 0


def prepare_local_runtime(path_to_build: Path) -> int:
    if ensure_simulation_core() != 0:
        return 1

    path_to_build.mkdir(parents=True, exist_ok=True)
    target = runtime_dir()
    print(f">>> 准备 {RUNTIME_DIR}/（build_dir={path_to_build}）...")

    if sync_py_core_plugins(to_build=True, to_runtime=True) != 0:
        return 1
    if sync_simulation_plugins(to_build=True, to_runtime=True) != 0:
        return 1

    clear_runtime_dir(target)
    build_sim = simulation_core_root() / "scripts" / "build_simulation.py"
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
        str(target),
    ]

    run(sim_cmd, cwd=simulation_core_root())
    copy_test_diffraction(path_to_build, target)
    print(f">>> 本地运行时已部署: {target}")
    env_sh = repo_root() / "scripts" / "init-toykits-build-env.sh"
    if env_sh.is_file():
        print(f">>> 运行前可执行: source {env_sh}")
    return 0


DEFAULT_STREAMLIT_PORT = 8052
DEFAULT_STREAMLIT_ADDRESS = "0.0.0.0"


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

    env_sh = repo_root() / "scripts" / "init-toykits-build-env.sh"
    cmd = [
        "bash",
        "-lc",
        f"source {env_sh} && cd {root} && exec streamlit run {app} "
        f"--server.port={port} --server.address={address}",
    ]
    print(f">>> 启动 Streamlit: http://localhost:{port}/")
    return subprocess.run(cmd, cwd=root).returncode


def resolve_database_source(explicit: str = "") -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.environ.get("SIMULATION_DATABASE_DIR", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
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
        "请设置 SIMULATION_DATABASE_DIR 或传入 --database-source，"
        "例如: --database-source ~/.oghma_local"
    )


def prepare_database_bundle(dest: Path, source: Path) -> None:
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
    script = (
        "import os; os.chdir('/app'); "
        "os.environ.setdefault('SIMULATION_ARTIFACTS_DIR', '/app/.simulation_core'); "
        "os.environ.setdefault('PYTHONPATH', '/app:/app/.simulation_core'); "
        "import simulation; "
        "from simulation_database.database_ui import ensure_simulation_database_initialized; "
        "db = ensure_simulation_database_initialized(); "
        "names = list(db.database_names()); "
        "assert names, 'no databases'; "
        "mats = db.database('materials'); "
        "assert mats.local_path(), 'materials path empty'; "
        "print('verify ok:', db.root_path(), names)"
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
