"""按需加载 simulation.so，注册为 assets.simulation。与 core 解耦。"""
import os
import sys
import importlib.util

def _artifacts_dir():
    """ARTIFACTS_DIR > /app/artifacts > 仓库根/docker_artifacts"""
    env = os.environ.get("ARTIFACTS_DIR", "").strip()
    if env and os.path.isdir(env):
        return env
    if os.path.isdir("/app/artifacts"):
        return "/app/artifacts"
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "docker_artifacts")

def get_simulation_module():
    """加载 simulation.so 并注册为 assets.simulation，返回该模块；已加载则直接返回。"""
    if os.environ.get("CI"):
        raise ImportError("assets.simulation skipped in CI")
    if "assets.simulation" in sys.modules:
        return sys.modules["assets.simulation"]
    artifacts = _artifacts_dir()
    so_path = os.path.join(artifacts, "simulation.so")
    if not os.path.isfile(so_path):
        raise FileNotFoundError(
            f"未找到 simulation.so，请先执行: ./scripts/prepare_docker.sh <path_to_build>\n"
            f"预期路径: {so_path}"
        )
    spec = importlib.util.spec_from_file_location("assets.simulation", so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {so_path} 创建 ModuleSpec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["assets.simulation"] = mod
    spec.loader.exec_module(mod)
    return mod
