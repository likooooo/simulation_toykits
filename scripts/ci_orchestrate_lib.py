"""CI 编排脚本共用：控制台仅输出 summary，详细日志写入 .ci/logs。"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


class ShellStep(Protocol):
    name: str
    cwd: Path
    body: str


REPO_ROOT = Path(__file__).resolve().parent.parent
CI_LOG_ROOT = REPO_ROOT / ".ci" / "logs"


def log(msg: str, *, err: bool = False) -> None:
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.lower()).strip("-")
    return slug or "step"


@dataclass
class OrchestrateSession:
    log_dir: Path
    dry_run: bool

    @classmethod
    def start(cls, *, dry_run: bool) -> OrchestrateSession:
        if dry_run:
            return cls(log_dir=CI_LOG_ROOT, dry_run=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = CI_LOG_ROOT / stamp
        log_dir.mkdir(parents=True, exist_ok=True)
        log(f"log dir: {log_dir.relative_to(REPO_ROOT)}")
        return cls(log_dir=log_dir, dry_run=False)


def bash_step(step: ShellStep, session: OrchestrateSession) -> None:
    script = f"set -euo pipefail\n{step.body}"
    if session.dry_run:
        log(f">>> [{step.name}] cwd={step.cwd}")
        for line in step.body.strip().splitlines():
            log(f"    {line}")
        return

    log_path = session.log_dir / f"{_slugify(step.name)}.log"
    log(f"  [..] {step.name}")
    started = time.monotonic()
    header = (
        f"# step: {step.name}\n"
        f"# cwd: {step.cwd}\n"
        f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n"
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(header)
        log_file.flush()
        result = subprocess.run(
            ["bash", "-lc", script],
            cwd=step.cwd,
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    rel_log = log_path.relative_to(REPO_ROOT)
    if result.returncode != 0:
        log(
            f"  [FAIL] {step.name} ({elapsed:.1f}s)  log: {rel_log}",
            err=True,
        )
        raise subprocess.CalledProcessError(result.returncode, ["bash", "-lc", script])
    log(f"  [OK] {step.name} ({elapsed:.1f}s)")
