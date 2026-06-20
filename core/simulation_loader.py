"""按需加载 simulation.so（从 .simulation_core 加入 sys.path 后 import simulation）。"""
import os
import sys

RUNTIME_DIR = ".simulation_core"


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _artifacts_dir():
    """返回包含 simulation.so 的目录（.simulation_core）。"""
    return os.path.join(_repo_root(), RUNTIME_DIR)


def ensure_artifacts_on_path():
    artifacts = _artifacts_dir()
    if artifacts not in sys.path:
        sys.path.insert(0, artifacts)
    plugins = os.path.join(artifacts, "py_core_plugins")
    if os.path.isdir(plugins) and plugins not in sys.path:
        sys.path.insert(0, plugins)
    _ensure_minimal_py_core_plugins()


def _ensure_minimal_py_core_plugins():
    """C++ py_plugin loads cwd-relative py_core_plugins/; toykits only needs the DB parser."""
    parser_src = os.path.join(_artifacts_dir(), "py_core_plugins", "simulation_database_parser.py")
    if not os.path.isfile(parser_src):
        return
    plugins_dir = os.path.join(_repo_root(), "py_core_plugins")
    os.makedirs(plugins_dir, exist_ok=True)
    parser_link = os.path.join(plugins_dir, "simulation_database_parser.py")
    src_abs = os.path.abspath(parser_src)
    if os.path.islink(parser_link):
        if os.path.realpath(parser_link) != src_abs:
            os.remove(parser_link)
        else:
            return
    elif os.path.isfile(parser_link):
        return
    os.symlink(src_abs, parser_link)


def get_simulation_module():
    ensure_artifacts_on_path()
    _ensure_minimal_py_core_plugins()
    if "simulation" not in sys.modules:
        so_path = os.path.join(_artifacts_dir(), "simulation.so")
        if not os.path.isfile(so_path):
            raise FileNotFoundError(
                f"未找到 simulation.so，请确保 {RUNTIME_DIR}/ 下存在 simulation.so\n"
                f"预期路径: {so_path}\n"
                "请先编译 simulation_core 并 collect："
                "python simulation_core/scripts/build_simulation.py -B simulation_core/build "
                "--build-type Release --collect .simulation_core"
            )
        import simulation  # noqa: F401
    return sys.modules["simulation"]
