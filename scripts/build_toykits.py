#!/usr/bin/env python3
"""simulation_toykits build and deploy CLI (orthogonal invocations)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RUNTIME_DIR = ".simulation_toolkits"
DEFAULT_BUILD_REL = Path("simulation_core/build")
ARTIFACTS_REPO = "likooooo/simulation_toolkits_artifacts"
SIMULATION_CORE_REPO = "likooooo/simulation"
ARTIFACT_TAR_NAME = "simulation_toolkits-linux-x86_64.tar.gz"
MANIFEST_NAME = "manifest.json"

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
PRECOMPILED_FILENAME = "database.bin"
FS_COMPARE_HTML_NAME = "fs_baseline_vs_toykits.html"

PY_CORE_PLUGINS_KEEP = frozenset(
    {
        "visualizer.py",
        "viz_io.py",
        "pipe_utils.py",
        "panel_renderer.py",
        "plot_source.py",
    }
)
SIMULATION_PLUGINS_KEEP = frozenset(
    {
        "simulation_database_parser.py",
        "filmstack_visualizer.py",
        "filmstack_optimization_utils.py",
        "layer_visualizer.py",
        "simulation_paths.py",
        "tmm_utils.py",
    }
)

DEFAULT_STREAMLIT_PORT = 8052
DEFAULT_STREAMLIT_ADDRESS = "::"

GITHUB_API = "https://api.github.com"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def simulation_core_root() -> Path:
    return repo_root() / "simulation_core"


def runtime_dir() -> Path:
    return repo_root() / RUNTIME_DIR


def default_build_dir() -> Path:
    return (repo_root() / DEFAULT_BUILD_REL).resolve()


def subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(extra or {})}


def gh_authenticated() -> bool:
    """gh 可用且已配置 token（GitHub Actions 中未设 GH_TOKEN 时 gh 会拒绝执行）。"""
    return bool(
        shutil.which("gh")
        and (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    )


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
        "env": subprocess_env(env),
        "check": check,
        "text": True,
    }
    if capture:
        kwargs["capture_output"] = True
    try:
        return subprocess.run(cmd, **kwargs)
    except subprocess.CalledProcessError as exc:
        print(
            f">>> Command failed (exit {exc.returncode}): {' '.join(str(c) for c in cmd)}",
            file=sys.stderr,
        )
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


def artifact_hint() -> str:
    return (
        f"请先执行 `python scripts/build_toykits.py` 编译，"
        f"或 `python scripts/build_toykits.py --download_toolkits` 下载 latest Release。"
    )


def require_artifact() -> Path:
    rt = runtime_dir()
    so = rt / "simulation.so"
    if not so.is_file():
        raise RuntimeError(f"缺少 {so}；{artifact_hint()}")
    return rt


def env_database_key() -> str | None:
    key = os.environ.get("SIMULATION_DATABASE_KEY", "").strip()
    return key or None


def require_env_database_key() -> str:
    key = env_database_key()
    if not key:
        raise RuntimeError(
            "SIMULATION_DATABASE_KEY is not set；docker 模式须事先 export 该环境变量。"
        )
    return key


def capture_shell_env(bash_snippet: str, *, cwd: Path | None = None) -> dict[str, str]:
    script = f"set -euo pipefail\n{bash_snippet}\nenv -0"
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=cwd or repo_root(),
        env=subprocess_env(),
        capture_output=True,
        check=True,
    )
    env: dict[str, str] = {}
    for entry in result.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        env[key.decode()] = value.decode()
    return env


def capture_simulation_build_env() -> dict[str, str]:
    init = simulation_core_root() / "scripts" / "init-simulation-build-env.sh"
    if not init.is_file():
        raise FileNotFoundError(f"未找到 {init}；请先 init simulation_core 子模块。")
    rt = runtime_dir()
    snippet = f'source "{init}" "{rt}" "{repo_root()}"'
    env = capture_shell_env(snippet)
    for key in ("SIMULATION_DATABASE_KEY", "HOME", "USER"):
        val = os.environ.get(key, "").strip()
        if val:
            env[key] = val
    return env


def ensure_simulation_core() -> None:
    simulation_root = simulation_core_root()
    if not simulation_root.is_dir():
        raise FileNotFoundError(
            f"未找到 simulation_core 子模块: {simulation_root}\n"
            "请先执行: git submodule update --init --recursive simulation_core"
        )


def freesnell_compare_output_dir() -> Path:
    return runtime_dir() / "assets" / "fs_compare"


def copy_test_diffraction(path_to_build: Path, target: Path) -> None:
    src = path_to_build / "test_diffraction"
    if not src.is_file():
        print(f">>> 警告: 未找到 {src}，衍射角计算器将不可用", file=sys.stderr)
        return
    dst = target / "test_diffraction"
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | 0o111)
    print(f">>> 已复制 test_diffraction -> {dst}")


def prune_toykits_plugins(target: Path | None = None) -> None:
    rt = (target or runtime_dir()).resolve()
    removed = 0
    for subdir, keep in (
        ("py_core_plugins", PY_CORE_PLUGINS_KEEP),
        ("simulation_plugins", SIMULATION_PLUGINS_KEEP),
    ):
        plugin_dir = rt / subdir
        if not plugin_dir.is_dir():
            continue
        for item in plugin_dir.iterdir():
            if item.name in keep:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1
    print(f">>> prune_toykits_plugins: 已删除 {removed} 个非 allowlist 项 ({rt})")


def build_toolkits_step(build_dir: Path | None = None) -> None:
    ensure_simulation_core()
    build_dir = (build_dir or default_build_dir()).resolve()
    target = runtime_dir()
    build_sim = simulation_core_root() / "scripts" / "build_simulation.py"
    if not build_sim.is_file():
        raise FileNotFoundError(f"未找到 {build_sim}")

    env = capture_simulation_build_env()
    print(f">>> --toolkits: build_dir={build_dir} -> {RUNTIME_DIR}/")
    run(
        [
            sys.executable,
            str(build_sim),
            "-B",
            str(build_dir),
            "--build-type",
            "Release",
            "--collect",
            str(target),
        ],
        cwd=simulation_core_root(),
        env=env,
        hint="pip install -r requirements-build.txt；确认 simulation_core 子模块已 init",
    )
    copy_test_diffraction(build_dir, target)
    prune_toykits_plugins(target)
    print(f">>> toolkits 已收集: {target}")


def build_bench_step() -> None:
    output = freesnell_compare_output_dir()
    script = repo_root() / "scripts" / "build_freesnell_compare_ui.py"
    if not script.is_file():
        raise FileNotFoundError(f"未找到 {script}")

    scripts_dir = repo_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from build_freesnell_compare_ui import resolve_freesnell_env

    env = {**local_runtime_env(), **resolve_freesnell_env()}
    print(f">>> --bench: FreeSnell 比对 UI -> {output}")
    run(
        [sys.executable, "-B", str(script), "--output", str(output)],
        cwd=repo_root(),
        env=env,
        hint="请将 FreeSnell 放到 $GENERATE_GOLDEN_TOOLS_DIR（默认 ~/repos/simulation_baseline_tools）约定子目录",
    )
    html_path = output / FS_COMPARE_HTML_NAME
    if not html_path.is_file():
        raise RuntimeError(f"未生成 {html_path}")
    print(f">>> bench 完成: {html_path}")


def compile_database_precompiled(db_dir: Path, *, env: dict[str, str] | None = None) -> Path:
    db_dir = db_dir.resolve()
    rt = runtime_dir().resolve()
    if not (rt / "simulation.so").is_file():
        raise FileNotFoundError(f"缺少 {rt / 'simulation.so'}；请先 --toolkits")

    bundle_path = rt / "assets" / PRECOMPILED_FILENAME
    run_env = {**(env or local_runtime_env())}
    run_env["SIMULATION_DATABASE_DIR"] = str(db_dir)
    run_env["SIMULATION_ARTIFACTS_DIR"] = str(rt)
    script = (
        "import simulation\n"
        "from simulation_database_parser import get_simulation_database\n"
        "from simulation_database.database_precompiling import compile_database_index\n"
        "db = get_simulation_database(init=True)\n"
        "print(compile_database_index(db, out_path=r'''" + str(bundle_path) + "'''))\n"
    )
    result = run(
        [sys.executable, "-B", "-c", script],
        cwd=repo_root(),
        env=run_env,
        capture=True,
    )
    line = result.stdout.strip().splitlines()[-1]
    out_path = Path(line)
    if not out_path.is_file():
        raise RuntimeError(f"compile_database_precompiled did not produce bundle: {line!r}")
    size_kb = out_path.stat().st_size / 1024
    print(f">>> 已预编译数据库 bundle: {out_path} ({size_kb:.1f} KiB)")
    if db_dir.is_dir():
        shutil.rmtree(db_dir)
        print(f">>> 已删除明文材料库: {db_dir}")
    return out_path


def remove_runtime_plaintext_database() -> None:
    """Remove plaintext YAML tree under assets/database/ when database.bin is present."""
    rt = runtime_dir().resolve()
    db_dir = rt / "assets" / "database"
    bundle = rt / "assets" / PRECOMPILED_FILENAME
    if not bundle.is_file() or not db_dir.is_dir():
        return
    shutil.rmtree(db_dir)
    print(f">>> 已删除明文材料库: {db_dir}")


def build_database_step() -> None:
    rt = runtime_dir()
    db_dir = rt / "assets" / "database"
    if db_dir.is_dir():
        compile_database_precompiled(db_dir)
    else:
        prepare_database_bundle(resolve_database_source())
    remove_runtime_plaintext_database()


def resolve_database_source(explicit: str = "") -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    collected = runtime_dir() / "assets" / "database"
    if collected.is_dir():
        candidates.append(collected)
    core_db = simulation_core_root() / "assets" / "database"
    if core_db.is_dir():
        candidates.append(core_db)
    env_root = os.environ.get("SIMULATION_DATABASE_DIR", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    seen: set[Path] = set()
    for raw in candidates:
        path = raw.resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / "og" / "materials").is_dir() and any(
            (path / "og" / "materials").iterdir()
        ):
            return path
    raise FileNotFoundError(
        "未找到可用的材料库源目录（需含非空 og/materials/）。"
        "请先 --toolkits（collect 含 database YAML）或设置 SIMULATION_DATABASE_DIR。"
    )


def prepare_database_bundle(source: Path) -> Path:
    release_script = source / "database_release.py"
    if not release_script.is_file():
        raise FileNotFoundError(
            f"材料库源 {source} 缺少 database_release.py；"
            "请指定 simulation_core/assets/database。"
        )
    with tempfile.TemporaryDirectory(prefix="sim-db-release-") as tmp:
        dest = Path(tmp)
        run(
            [
                sys.executable,
                "-B",
                str(release_script),
                "--dest",
                str(dest),
                "--clean",
            ],
            cwd=source,
        )
        materials = dest / "og" / "materials"
        if not materials.is_dir() or not any(materials.iterdir()):
            raise FileNotFoundError(f"database_release 未产出 og/materials: {dest}")
        size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f">>> 已 release 材料库: {source} -> {dest} ({size_mb:.1f} MiB)")
        return compile_database_precompiled(dest)


def resolve_build_steps(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    any_flag = args.toolkits or args.bench or args.database
    if not any_flag:
        return True, True, True
    return args.toolkits, args.bench, args.database


def cmd_build(args: argparse.Namespace) -> int:
    do_toolkits, do_bench, do_database = resolve_build_steps(args)
    labels: list[str] = []
    if do_toolkits:
        labels.append("toolkits")
    if do_database:
        labels.append("database")
    if do_bench:
        labels.append("bench")
    if labels:
        log(f">>> build 步骤: {' → '.join(labels)}")

    try:
        if do_toolkits:
            build_toolkits_step()
        elif do_bench or do_database:
            require_artifact()
        if do_database:
            build_database_step()
        if do_bench:
            build_bench_step()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError:
        print("错误: build 子步骤命令失败（见上方日志）", file=sys.stderr)
        return 1

    if labels:
        log(">>> build 完成")
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def github_request_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_get_json(url: str) -> dict:
    req = Request(url, headers=github_request_headers())
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def github_default_branch(repo: str) -> str:
    if gh_authenticated():
        result = run(
            ["gh", "api", f"repos/{repo}", "--jq", ".default_branch"],
            capture=True,
        )
        branch = result.stdout.strip()
        if branch:
            return branch
    data = github_get_json(f"{GITHUB_API}/repos/{repo}")
    branch = data.get("default_branch")
    if not branch:
        raise RuntimeError(f"无法获取 {repo} 的 default_branch")
    return branch


def github_branch_head(repo: str, branch: str) -> str:
    if gh_authenticated():
        result = run(
            ["gh", "api", f"repos/{repo}/commits/{branch}", "--jq", ".sha"],
            capture=True,
        )
        sha = result.stdout.strip()
        if sha:
            return sha
    data = github_get_json(f"{GITHUB_API}/repos/{repo}/commits/{branch}")
    sha = data.get("sha")
    if not sha:
        raise RuntimeError(f"无法获取 {repo}@{branch} 的 HEAD commit")
    return sha


def github_api_latest_commit(repo: str) -> tuple[str, str]:
    branch = github_default_branch(repo)
    sha = github_branch_head(repo, branch)
    return branch, sha


def git_ssh_remote_head(repo: str) -> tuple[str, str]:
    url = f"git@github.com:{repo}.git"
    result = run(
        ["git", "ls-remote", "--symref", url, "HEAD"],
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git ls-remote 无法访问 {repo}: {result.stderr.strip() or result.stdout.strip()}"
        )
    branch = "main"
    sha = ""
    for line in result.stdout.splitlines():
        if line.startswith("ref: refs/heads/"):
            branch = line.split("refs/heads/", 1)[1].split("\t", 1)[0]
        elif line.endswith("\tHEAD"):
            sha = line.split("\t", 1)[0]
    if not sha:
        raise RuntimeError(f"git ls-remote 未返回 {repo} 的 HEAD commit")
    return branch, sha


def github_latest_simulation_core_commit() -> tuple[str, str]:
    errors: list[str] = []
    for fetch in (
        lambda: github_api_latest_commit(SIMULATION_CORE_REPO),
        lambda: git_ssh_remote_head(SIMULATION_CORE_REPO),
    ):
        try:
            return fetch()
        except (HTTPError, URLError, RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


_BANNER_COMMIT_RE = re.compile(r"\*\s+Git commit : (\S+)")


def runtime_embedded_commit() -> str:
    result = run(
        [
            sys.executable,
            "-c",
            "import simulation; simulation.print_simulation_banner()",
        ],
        env=subprocess_env(),
        capture=True,
    )
    match = _BANNER_COMMIT_RE.search(result.stderr)
    if not match:
        raise RuntimeError("无法从 print_simulation_banner 输出解析 Git commit")
    commit = match.group(1)
    if commit == "unknown":
        raise RuntimeError("runtime 内嵌 Git commit 为 unknown")
    return commit


def cmd_verify_runtime_commit(_args: argparse.Namespace) -> int:
    require_artifact()
    try:
        runtime_commit = runtime_embedded_commit()
        branch, expected_commit = github_latest_simulation_core_commit()
    except (HTTPError, URLError, RuntimeError) as exc:
        print(f"错误: runtime commit 校验失败: {exc}", file=sys.stderr)
        return 1

    if runtime_commit != expected_commit:
        print(
            "错误: runtime 内嵌 simulation_core commit 与 GitHub 最新 commit 不一致\n"
            f"  runtime : {runtime_commit}\n"
            f"  GitHub  : {expected_commit} ({SIMULATION_CORE_REPO}@{branch})",
            file=sys.stderr,
        )
        return 1

    print(f"runtime commit OK: {runtime_commit} ({SIMULATION_CORE_REPO}@{branch})")
    return 0


def latest_release_asset_url(asset_name: str) -> str:
    if gh_authenticated():
        result = run(
            [
                "gh",
                "api",
                f"repos/{ARTIFACTS_REPO}/releases/latest",
                "--jq",
                f'.assets[] | select(.name=="{asset_name}") | .browser_download_url',
            ],
            capture=True,
        )
        url = result.stdout.strip()
        if url:
            return url
    data = github_get_json(f"{GITHUB_API}/repos/{ARTIFACTS_REPO}/releases/latest")
    for asset in data.get("assets", []):
        if asset.get("name") == asset_name:
            return asset["browser_download_url"]
    raise RuntimeError(f"latest Release 中未找到 asset: {asset_name}")


def download_manifest() -> dict | None:
    try:
        url = latest_release_asset_url(MANIFEST_NAME)
    except (HTTPError, URLError, RuntimeError):
        return None
    with urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode())


def cmd_download_toolkits(_args: argparse.Namespace) -> int:
    root = repo_root()
    rt = runtime_dir()

    try:
        with tempfile.TemporaryDirectory(prefix="sim-toolkits-dl-") as tmp:
            tmp_path = Path(tmp)
            tar_path = tmp_path / ARTIFACT_TAR_NAME
            log(f">>> 下载 latest Release: {ARTIFACTS_REPO}/{ARTIFACT_TAR_NAME}")
            if gh_authenticated():
                run(
                    [
                        "gh",
                        "release",
                        "download",
                        "--repo",
                        ARTIFACTS_REPO,
                        "--pattern",
                        ARTIFACT_TAR_NAME,
                        "-D",
                        str(tmp_path),
                    ],
                    hint=f"确认可访问 https://github.com/{ARTIFACTS_REPO}/releases/latest",
                )
            else:
                url = latest_release_asset_url(ARTIFACT_TAR_NAME)
                run(["curl", "-fsSL", "-o", str(tar_path), url])

            if not tar_path.is_file():
                print(f"错误: 未下载到 {tar_path}", file=sys.stderr)
                return 1

            manifest = download_manifest()
            if manifest and manifest.get("sha256"):
                actual = sha256_file(tar_path)
                expected = manifest["sha256"]
                if actual != expected:
                    print(
                        f"错误: sha256 不匹配 (expected {expected}, got {actual})",
                        file=sys.stderr,
                    )
                    return 1
                log(f">>> sha256 校验通过: {actual[:16]}…")

            extract_dir = tmp_path / "extract"
            extract_dir.mkdir()
            run(["tar", "-xzf", str(tar_path), "-C", str(extract_dir)])
            new_rt = extract_dir / RUNTIME_DIR
            if not (new_rt / "simulation.so").is_file():
                print(f"错误: 归档内缺少 {RUNTIME_DIR}/simulation.so", file=sys.stderr)
                return 1

            if rt.exists():
                shutil.rmtree(rt)
            shutil.move(str(new_rt), str(rt))
    except (subprocess.CalledProcessError, HTTPError, URLError, RuntimeError) as exc:
        print(f"错误: 下载失败: {exc}", file=sys.stderr)
        return 1

    log(f">>> artifact 就绪: {rt}")
    return 0


@dataclass(frozen=True)
class RepoVersion:
    name: str
    path: Path
    commit: str
    date: str
    author: str
    email: str
    subject: str


def git_head_info(name: str, path: Path) -> RepoVersion | None:
    if not (path / ".git").exists():
        return None
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
        ("simulation_core", root / "simulation_core"),
        ("infrastructure", root / "simulation_core" / "3rdparty" / "infrastructure"),
    ]
    versions: list[RepoVersion] = []
    for name, path in repos:
        info = git_head_info(name, path)
        if info:
            versions.append(info)
    return versions


def auto_release_tag() -> str:
    root = repo_root()
    short_sha = "unknown"
    try:
        short_sha = run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        pass
    today = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    return f"v{today}+{short_sha}"


def create_artifact_archive(dest: Path) -> Path:
    rt = runtime_dir()
    require_artifact()
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["tar", "-czf", str(dest), "-C", str(repo_root()), RUNTIME_DIR],
        check=True,
    )
    return dest


def cmd_release(_args: argparse.Namespace) -> int:
    require_artifact()
    bundle = runtime_dir() / "assets" / PRECOMPILED_FILENAME
    if not bundle.is_file():
        print(f"错误: 缺少 {bundle}；请先 build --database", file=sys.stderr)
        return 1

    if not shutil.which("gh"):
        print("错误: release 需要 gh CLI", file=sys.stderr)
        return 1

    tag = auto_release_tag()
    build_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    versions = collect_repo_versions(repo_root())

    with tempfile.TemporaryDirectory(prefix="sim-release-") as tmp:
        tmp_path = Path(tmp)
        tar_path = tmp_path / ARTIFACT_TAR_NAME
        create_artifact_archive(tar_path)
        digest = sha256_file(tar_path)
        manifest = {
            "tag": tag,
            "build_time": build_time,
            "artifact": ARTIFACT_TAR_NAME,
            "sha256": digest,
            "repos": {
                v.name: {
                    "commit": v.commit,
                    "date": v.date,
                    "author": v.author,
                    "subject": v.subject,
                }
                for v in versions
            },
        }
        manifest_path = tmp_path / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        notes = f"Build {build_time}\n\nsha256: `{digest}`"
        log(f">>> gh release create {tag} -> {ARTIFACTS_REPO}")
        run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                ARTIFACTS_REPO,
                "--title",
                tag,
                "--notes",
                notes,
                "--latest",
                str(tar_path),
                str(manifest_path),
            ],
            hint=f"确认对 {ARTIFACTS_REPO} 有 release 权限",
        )
    log(f">>> Release 已发布: {tag} (latest)")
    return 0


def local_runtime_env() -> dict[str, str]:
    rt = runtime_dir().resolve()
    env = {
        "LD_LIBRARY_PATH": f"{rt}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
        "SIMULATION_DATABASE_DIR": str(rt / "assets"),
        "SIMULATION_ARTIFACTS_DIR": str(rt),
        "PYTHONPATH": f"{repo_root().resolve()}:{rt}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
    }
    for key in ("SIMULATION_DATABASE_KEY",):
        val = os.environ.get(key, "").strip()
        if val:
            env[key] = val
    return env


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
            env=subprocess_env(),
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
            env=subprocess_env(),
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
        subprocess.run(["kill", str(pid)], env=subprocess_env(), check=False)
        print(f">>> 已停止占用端口 {port} 的 Streamlit 进程 PID {pid}")


def start_local_server() -> int:
    port = DEFAULT_STREAMLIT_PORT
    address = DEFAULT_STREAMLIT_ADDRESS
    root = repo_root()
    app = root / "app.py"
    try:
        require_artifact()
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    if not app.is_file():
        print(f"错误: 未找到 {app}", file=sys.stderr)
        return 1
    if not shutil.which("streamlit"):
        print("错误: 未找到 streamlit；pip install -r requirements.txt", file=sys.stderr)
        return 1

    for pid in _listening_pids_on_port(port):
        if not _is_streamlit_pid(pid):
            cmd = _process_cmdline(pid)
            print(
                f"错误: 端口 {port} 已被非 Streamlit 进程占用 (PID {pid}): {cmd}",
                file=sys.stderr,
            )
            return 1

    stop_streamlit_on_port(port)
    env = subprocess_env(local_runtime_env())
    cmd = [
        "streamlit",
        "run",
        str(app),
        f"--server.port={port}",
        f"--server.address={address}",
    ]
    print(f">>> 启动 Streamlit: http://localhost:{port}/")
    return subprocess.run(cmd, cwd=root, env=env).returncode


def verify_docker_image(image_tag: str) -> None:
    script = (
        "import os; os.chdir('/app'); "
        "os.environ.setdefault('SIMULATION_ARTIFACTS_DIR', '/app/.simulation_toolkits'); "
        "os.environ.setdefault('SIMULATION_DATABASE_DIR', '/app/.simulation_toolkits/assets'); "
        "os.environ.setdefault('PYTHONPATH', '/app:/app/.simulation_toolkits'); "
        "from pathlib import Path; "
        "from simulation_database.database_precompiling import precompiled_bundle_path, load_or_build_database_index; "
        "import simulation; "
        "from simulation_database_parser import get_simulation_database; "
        "assert precompiled_bundle_path().is_file(), 'missing precompiled bundle'; "
        "db = get_simulation_database(init=True); "
        "index = load_or_build_database_index(db); "
        "assert index.leaf_count > 0, 'empty precompiled index'; "
        "print('verify ok:', precompiled_bundle_path(), index.leaf_count)"
    )
    key = require_env_database_key()
    run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            f"SIMULATION_DATABASE_KEY={key}",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            image_tag,
            "python",
            "-B",
            "-c",
            script,
        ]
    )


def cmd_local(_args: argparse.Namespace) -> int:
    return start_local_server()


def cmd_docker(_args: argparse.Namespace) -> int:
    root = repo_root()
    try:
        require_env_database_key()
        require_artifact()
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if not shutil.which("docker"):
        print("错误: 未找到 docker 命令", file=sys.stderr)
        return 1

    image_tag = DEFAULT_DOCKER_IMAGE_TAG
    print(f">>> docker build: {image_tag}")
    try:
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
        print(f">>> 验证镜像: {image_tag}")
        verify_docker_image(image_tag)
    except subprocess.CalledProcessError:
        return 1
    print("======== 全部完成 ========")
    return 0


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


def render_deploy_readme(root: Path, build_time: str, versions: list[RepoVersion]) -> str:
    lines = [
        "# simulation-toykits Hugging Face Space",
        "",
        f"**BUILD_TIME:** {build_time}",
        "",
        "Docker 构建时从 GitHub 克隆应用源码，并从 "
        f"[{ARTIFACTS_REPO}](https://github.com/{ARTIFACTS_REPO}) "
        "latest Release 下载 `.simulation_toolkits/` runtime artifact。",
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


def git_root_commit(dest: Path) -> str:
    result = run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=dest,
        capture=True,
    )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"No root commit found in {dest}")
    return lines[0]


def reset_hf_repo_to_root(dest: Path) -> str:
    root = git_root_commit(dest)
    run(["git", "reset", "--hard", root], cwd=dest)
    log(f">>> Reset HF repo to root commit {root[:12]}")
    return root


def clone_hf_repo(hf_repo: str, dest: Path) -> None:
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--single-branch",
            hf_repo,
            str(dest),
        ],
        env=env,
        hint=(
            "ensure the public key paired with CI secrets.MY_PRIVATE_KEY_1 "
            "is added at https://huggingface.co/settings/keys"
        ),
    )
    reset_hf_repo_to_root(dest)


def verify_hf_commit_count(dest: Path, expected: int = 2) -> None:
    count = int(
        run(["git", "rev-list", "--count", "HEAD"], cwd=dest, capture=True).stdout.strip()
    )
    if count != expected:
        raise RuntimeError(f"Expected {expected} commits after HF deploy, got {count}")
    log(f">>> HF repo has {count} commits (root + deploy snapshot)")


def deploy_hf(*, hf_repo: str, dest_path: str, dry_run: bool) -> int:
    root = repo_root()
    build_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dest = Path(dest_path)
    hf_dockerfile = root / "Dockerfile.hugging_face"

    if not hf_dockerfile.is_file():
        log(f">>> Missing {hf_dockerfile}", err=True)
        return 1

    if dest.exists():
        shutil.rmtree(dest)

    log(f">>> Cloning HF Space to {dest}")
    try:
        clone_hf_repo(hf_repo, dest)
    except subprocess.CalledProcessError:
        return 1

    log(">>> Clearing HF workspace (keep .git)")
    readme_prefix = resolve_hf_readme_prefix(dest / "README.md")
    clear_dest_workspace(dest)

    versions = collect_repo_versions(root)
    readme_body = render_deploy_readme(root, build_time, versions)
    commit_message = f"{readme_prefix.rstrip()}\n\n{readme_body}"
    (dest / "README.md").write_text(commit_message, encoding="utf-8")
    write_dockerfile(hf_dockerfile, dest, build_time)
    log(f">>> Wrote Dockerfile + README (BUILD_TIME={build_time})")

    if dry_run:
        log(">>> dry-run: skip git commit/push")
        log(">>> commit message preview:")
        log(commit_message)
        return 0

    run(["git", "add", "-A"], cwd=dest)
    status = run(["git", "status", "--porcelain"], cwd=dest, capture=True)
    if not status.stdout.strip():
        log(">>> No changes to deploy")
        return 0

    git_commit_with_message(dest, commit_message, root)
    verify_hf_commit_count(dest)
    ensure_hf_endpoint()
    push = subprocess.run(
        ["git", "push", "--force"],
        cwd=dest,
        env=subprocess_env(),
        capture_output=True,
        text=True,
    )
    if push.returncode != 0:
        log(f">>> git push --force failed (exit {push.returncode})", err=True)
        if push.stdout:
            log(push.stdout.rstrip(), err=True)
        if push.stderr:
            log(push.stderr.rstrip(), err=True)
        return 1

    log(">>> Successfully pushed to Hugging Face Space")
    return 0


def cmd_hf(args: argparse.Namespace) -> int:
    return deploy_hf(
        hf_repo=DEFAULT_HF_REPO,
        dest_path=HF_DEPLOY_DEST,
        dry_run=args.dry_run,
    )


CONSUME_COMMANDS = frozenset({"local", "release", "docker", "hf"})
BUILD_COMMAND = "build"


def parse_invocation(args: argparse.Namespace) -> tuple[bool, str | None]:
    """Return (run_build, consume_command_or_none)."""
    step1, step2 = args.step1, args.step2
    if step2 is not None and step1 != BUILD_COMMAND:
        raise SystemExit("双步骤仅支持 'build <local|release|docker|hf>'")
    if step1 is None:
        return True, None
    if step1 == BUILD_COMMAND:
        return True, step2
    if step1 in CONSUME_COMMANDS:
        if step2 is not None:
            raise SystemExit(f"不支持 '{step1} {step2}'；消费子命令不可串联")
        return False, step1
    raise SystemExit(f"未知步骤: {step1!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="simulation_toykits 编译与部署（build 可与 local/docker/hf/release 组合）",
    )
    parser.add_argument(
        "--download_toolkits",
        action="store_true",
        help="下载 GitHub Release latest artifact 到 .simulation_toolkits/",
    )
    parser.add_argument(
        "--verify_runtime_commit",
        action="store_true",
        help="校验 runtime 内嵌 simulation_core commit 与 GitHub 默认分支 HEAD 一致",
    )
    parser.add_argument("--toolkits", action="store_true", help="编译 collect + prune 插件")
    parser.add_argument("--bench", action="store_true", help="FreeSnell 比对 HTML")
    parser.add_argument("--database", action="store_true", help="预编译 database.bin")
    parser.add_argument(
        "step1",
        nargs="?",
        choices=[BUILD_COMMAND, *sorted(CONSUME_COMMANDS)],
        help="build（编译）或消费子命令；默认无参数 = 全量 build",
    )
    parser.add_argument(
        "step2",
        nargs="?",
        choices=sorted(CONSUME_COMMANDS),
        help="与 step1=build 组合：build local / build docker / build hf / build release",
    )
    parser.add_argument("--dry-run", action="store_true", help="hf: 仅预览，不 push")

    args = parser.parse_args()

    if args.download_toolkits:
        if (
            args.step1
            or args.step2
            or args.toolkits
            or args.bench
            or args.database
            or args.verify_runtime_commit
        ):
            parser.error("--download_toolkits 与其它选项互斥")
        return cmd_download_toolkits(args)

    if args.verify_runtime_commit:
        if args.step1 or args.step2 or args.toolkits or args.bench or args.database:
            parser.error("--verify_runtime_commit 与其它选项互斥")
        return cmd_verify_runtime_commit(args)

    do_build, consume = parse_invocation(args)
    handlers = {
        "local": cmd_local,
        "release": cmd_release,
        "docker": cmd_docker,
        "hf": cmd_hf,
    }

    if do_build:
        code = cmd_build(args)
        if code != 0:
            if consume:
                print(f"错误: build 失败，跳过 {consume}", file=sys.stderr)
            return code
    if consume:
        return handlers[consume](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
