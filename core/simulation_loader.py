"""按需加载 simulation.so（docker_artifacts 加入 sys.path 后可直接 import simulation）。"""
import os
import sys


def _artifacts_dir():
    env = os.environ.get("ARTIFACTS_DIR", "").strip()
    if env and os.path.isdir(env):
        return env
    if os.path.isdir("/app/artifacts"):
        return "/app/artifacts"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "docker_artifacts")


def ensure_artifacts_on_path():
    artifacts = _artifacts_dir()
    if artifacts not in sys.path:
        sys.path.insert(0, artifacts)


def get_simulation_module():
    # In CI, only skip when simulation.so is not present (e.g. docker_artifacts not cloned)
    if os.environ.get("CI"):
        so_path = os.path.join(_artifacts_dir(), "simulation.so")
        if not os.path.isfile(so_path):
            raise ImportError("simulation skipped in CI (no simulation.so)")
    ensure_artifacts_on_path()
    if "simulation" not in sys.modules:
        so_path = os.path.join(_artifacts_dir(), "simulation.so")
        if not os.path.isfile(so_path):
            raise FileNotFoundError(
                f"未找到 simulation.so，请先执行: ./scripts/prepare_docker.sh <path_to_build>\n"
                f"预期路径: {so_path}"
            )
        import simulation  # noqa: F401
    return sys.modules["simulation"]
