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
    if os.environ.get("CI"):
        return
    artifacts = _artifacts_dir()
    if artifacts not in sys.path:
        sys.path.insert(0, artifacts)


def get_simulation_module():
    if os.environ.get("CI"):
        raise ImportError("simulation skipped in CI")
    ensure_artifacts_on_path()
    if "simulation" not in sys.modules:
        artifacts = _artifacts_dir()
        so_path = os.path.join(artifacts, "simulation.so")
        py_path = os.path.join(artifacts, "simulation.py")
        if not os.path.isfile(so_path) and not os.path.isfile(py_path):
            raise FileNotFoundError(
                f"未找到 simulation.so 或 simulation.py，请先执行: ./scripts/prepare_docker.sh <path_to_build>\n"
                f"预期路径: {so_path} 或 {py_path}"
            )
        import simulation  # noqa: F401
    return sys.modules["simulation"]
