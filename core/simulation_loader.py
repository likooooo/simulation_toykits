"""按需加载 simulation.so：将 docker_artifacts 加入 sys.path 后可直接 import simulation。"""
import os
import sys

def _artifacts_dir():
    """ARTIFACTS_DIR > /app/artifacts > 仓库根/docker_artifacts"""
    env = os.environ.get("ARTIFACTS_DIR", "").strip()
    if env and os.path.isdir(env):
        return env
    if os.path.isdir("/app/artifacts"):
        return "/app/artifacts"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "docker_artifacts")

def ensure_artifacts_on_path():
    """将 artifacts 目录加入 sys.path，之后可直接 import simulation。CI 下不修改 path。"""
    if os.environ.get("CI"):
        return
    artifacts = _artifacts_dir()
    if artifacts not in sys.path:
        sys.path.insert(0, artifacts)

def get_simulation_module():
    """确保 artifacts 在 path 上并加载 simulation，返回该模块；已加载则直接返回。"""
    if os.environ.get("CI"):
        raise ImportError("simulation skipped in CI")
    ensure_artifacts_on_path()
    if "simulation" not in sys.modules:
        so_path = os.path.join(_artifacts_dir(), "simulation.so")
        if not os.path.isfile(so_path):
            raise FileNotFoundError(
                f"未找到 simulation.so，请先执行: ./scripts/prepare_docker.sh <path_to_build>\n"
                f"预期路径: {so_path}"
            )
        import simulation  # noqa: F401 - 从 sys.path 上的目录加载 .so
    return sys.modules["simulation"]
