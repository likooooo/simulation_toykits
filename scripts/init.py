#!/usr/bin/env python3
"""
初始化 simulation_toykits 环境：
1. 创建 assets、docker_artifacts 软链接指向 simulation_golden_data 对应目录
2. 创建 venv 并安装 requirements.txt，提示激活环境
"""

import os
import subprocess
import sys


def main() -> int:
    # 通过 __file__ 获取当前文件所在目录，再取父目录得到仓库根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # simulation_golden_data 与 simulation_toykits 同级（均在 repos 下）
    parent_of_repo = os.path.dirname(repo_root)
    golden_base = os.path.join(parent_of_repo, "simulation_golden_data", "simlation_toykits")

    # 需要创建的软链接：(链接名, 目标真实路径)
    symlinks = [
        (os.path.join(repo_root, "assets"), os.path.join(golden_base, "assets")),
        (os.path.join(repo_root, "docker_artifacts"), os.path.join(golden_base, "docker_artifacts")),
    ]

    for link_path, target_path in symlinks:
        if not os.path.exists(target_path):
            print(f"跳过软链接（目标不存在）: {target_path}", file=sys.stderr)
            continue
        if os.path.lexists(link_path):
            if os.path.islink(link_path) and os.path.realpath(link_path) == os.path.realpath(target_path):
                print(f"已存在且指向正确: {link_path} -> {target_path}")
            else:
                print(f"已存在且指向不同，请手动检查: {link_path}", file=sys.stderr)
            continue
        try:
            os.symlink(target_path, link_path)
            print(f"已创建软链接: {link_path} -> {target_path}")
        except OSError as e:
            print(f"创建软链接失败 {link_path}: {e}", file=sys.stderr)
            return 1

    # venv 路径与 requirements.txt
    venv_dir = os.path.join(repo_root, ".venv")
    requirements_path = os.path.join(repo_root, "requirements.txt")

    if not os.path.exists(requirements_path):
        print(f"未找到 requirements.txt: {requirements_path}", file=sys.stderr)
        return 1

    # 创建 venv（若不存在）
    if not os.path.isdir(venv_dir):
        print(f"创建虚拟环境: {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

    pip = os.path.join(venv_dir, "bin", "pip")
    if not os.path.isfile(pip):
        pip = os.path.join(venv_dir, "Scripts", "pip.exe")
    if not os.path.isfile(pip):
        print(f"未找到 venv 中的 pip: {venv_dir}", file=sys.stderr)
        return 1

    print("安装依赖: pip install -r requirements.txt")
    subprocess.run([pip, "install", "-r", requirements_path], check=True, cwd=repo_root)

    activate_script = os.path.join(venv_dir, "bin", "activate")
    if not os.path.isfile(activate_script):
        activate_script = os.path.join(venv_dir, "Scripts", "activate.bat")
    print("\n激活环境请执行:")
    print(f"  source {activate_script}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
