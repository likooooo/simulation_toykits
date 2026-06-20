#!/usr/bin/env python3
"""Deploy CI-built .simulation_core to Hugging Face Space (Dockerfile + README).

Artifacts-only push: HF Space git repo holds Dockerfile, .simulation_core/, and an
auto-generated README.md. Application source is cloned during HF Docker build.

One-time Hugging Face setup (must match CI secrets.MY_PRIVATE_KEY_1):
  - Account → Settings → SSH Keys: add the **public** key (for git push to hf.co)
  - Space → Settings → Repository secrets: SSH_PRIVATE_KEY = same **private** key
    (for Dockerfile clone of GitHub + submodules during HF image build)
"""

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


DEFAULT_HF_REPO = "git@hf.co:spaces/simulation-toykits/v1"
# git-xet requires HF_ENDPOINT when remote uses ssh://…:port/…; set anyway for safety.
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


@dataclass(frozen=True)
class RepoVersion:
    name: str
    path: Path
    commit: str
    date: str
    author: str
    email: str
    subject: str


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def log(msg: str, *, err: bool = False) -> None:
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def run(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
    hint: str | None = None,
) -> subprocess.CompletedProcess:
    kwargs: dict = {"cwd": cwd or repo_root(), "check": check, "text": True}
    if capture:
        kwargs["capture_output"] = True
    try:
        return subprocess.run(cmd, **kwargs)
    except subprocess.CalledProcessError as exc:
        log(f">>> Command failed (exit {exc.returncode}): {' '.join(cmd)}", err=True)
        if capture:
            if exc.stdout:
                log(exc.stdout.rstrip(), err=True)
            if exc.stderr:
                log(exc.stderr.rstrip(), err=True)
        if hint:
            log(f">>> Hint: {hint}", err=True)
        raise


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
        r'(ARG BUILD_TIME\s*=\s*).*',
        f'\\1"{build_time}"',
        content,
    )
    (dest / "Dockerfile").write_text(new_content, encoding="utf-8")


def setup_lfs_for_artifacts(dest: Path) -> None:
    """Track all .simulation_core binaries via Git LFS (HF stores pointers, not raw blobs)."""
    run(["git", "lfs", "install", "--local"], cwd=dest)
    for pattern in HF_LFS_PATTERNS:
        run(["git", "lfs", "track", pattern], cwd=dest)
    log(">>> .gitattributes for HF LFS:")
    log((dest / ".gitattributes").read_text(encoding="utf-8"))


def ensure_git_xet() -> None:
    """HF Xet-enabled repos require git-xet for LFS uploads."""
    if subprocess.run(["git", "xet", "--version"], capture_output=True).returncode != 0:
        log(
            ">>> git-xet not found; install from "
            "https://huggingface.co/docs/hub/xet/using-xet-storage#git",
            err=True,
        )
        raise RuntimeError("git-xet is required for Hugging Face binary uploads")
    run(["git", "xet", "install"])


def ensure_git_identity(dest: Path, root: Path) -> None:
    """HF clone repos in CI have no user.name/email; set local identity before commit."""
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
    """Write multiline README content as the HF Space commit message."""
    ensure_git_identity(dest, root)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(message)
        msg_path = f.name
    try:
        run(["git", "commit", "-F", msg_path], cwd=dest)
    finally:
        Path(msg_path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy CI artifacts to Hugging Face Space")
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help="Hugging Face Space git URL (use git@hf.co:spaces/…, not ssh://…:22/…)",
    )
    parser.add_argument(
        "--dest",
        default="/tmp/simulation-toykits-hf-deploy",
        help="Temporary clone directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files but do not commit/push",
    )
    args = parser.parse_args()

    root = repo_root()
    build_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dest = Path(args.dest)
    hf_dockerfile = root / "Dockerfile.hugging_face"
    artifacts_dir = root / ".simulation_core"

    if not hf_dockerfile.is_file():
        log(f">>> Missing {hf_dockerfile}", err=True)
        return 1

    commit_message = f"deploy simulation_toykits @ {build_time}"

    if dest.exists():
        shutil.rmtree(dest)

    log(f">>> Cloning HF Space to {dest}")
    try:
        run(
            ["git", "clone", args.hf_repo, str(dest)],
            check=True,
            hint=(
                "ensure the public key paired with CI secrets.MY_PRIVATE_KEY_1 "
                "is added at https://huggingface.co/settings/keys"
            ),
        )
    except subprocess.CalledProcessError:
        log(f">>> git clone failed for {args.hf_repo}", err=True)
        return 1

    simulation_so = artifacts_dir / "simulation.so"
    if not simulation_so.is_file():
        log(f">>> Missing {simulation_so}", err=True)
        return 1

    log(">>> Clearing HF workspace (keep .git)")
    cloned_readme = dest / "README.md"
    readme_prefix = resolve_hf_readme_prefix(cloned_readme)
    clear_dest_workspace(dest)

    log(f">>> Copying {artifacts_dir} -> {dest / '.simulation_core'}")
    shutil.copytree(
        artifacts_dir,
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

    if args.dry_run:
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
    push = subprocess.run(
        ["git", "push"],
        cwd=dest,
        capture_output=True,
        text=True,
    )
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


if __name__ == "__main__":
    sys.exit(main())
