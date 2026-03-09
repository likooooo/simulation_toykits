"""按需加载 simulation.so（从 assets/lib 加入 sys.path 后 import simulation）。"""
import os
import sys


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _artifacts_dir():
    """返回包含 simulation.so 的目录（assets/lib）。"""
    return os.path.join(_repo_root(), "assets", "lib")


def ensure_artifacts_on_path():
    artifacts = _artifacts_dir()
    if artifacts not in sys.path:
        sys.path.insert(0, artifacts)


def get_simulation_module():
    # CI 中仅在 simulation.so 不存在时跳过；设 CI=false 可打开与 simulation.so 相关的测试
    if os.environ.get("CI", "").lower() not in ("0", "false", ""):
        so_path = os.path.join(_artifacts_dir(), "simulation.so")
        if not os.path.isfile(so_path):
            raise ImportError("simulation skipped in CI (no simulation.so)")
    ensure_artifacts_on_path()
    if "simulation" not in sys.modules:
        so_path = os.path.join(_artifacts_dir(), "simulation.so")
        if not os.path.isfile(so_path):
            raise FileNotFoundError(
                f"未找到 simulation.so，请确保 assets/lib 下存在 simulation.so\n预期路径: {so_path}"
            )
        import simulation  # noqa: F401
    return sys.modules["simulation"]
