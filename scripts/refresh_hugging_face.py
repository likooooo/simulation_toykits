#!/usr/bin/env python3
import sys
import subprocess
import os
from datetime import datetime

# 获取当前时间，格式可以根据需要调整 (例如: 2026-03-10 09:00:00)
build_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

hugging_face_git_base = '8bc1e6d6e1841895b5522c3c20b3d4ab5e0565ed'
git_repo = 'ssh://git@hf.co:22/spaces/simulation-toykits/v1'
dest = '/tmp/v1'

# 1. 清理并克隆仓库
subprocess.run(f"rm -rf {dest} && git clone {git_repo} {dest}", shell=True, check=True)

# 2. 切换到指定 commit
os.chdir(dest)
subprocess.run(f"git reset --hard {hugging_face_git_base}", shell=True, check=True)

# 3. 读取原始 Dockerfile 并修改 BUILD_TIME
dockerfile_path = os.path.join(HOME, 'Dockerfile')
target_path = os.path.join(dest, 'Dockerfile')

with open(dockerfile_path, 'r') as f:
    content = f.read()

# 替换 ARG BUILD_TIME="..." 部分
# 无论原本是空字符串还是旧时间，都会被更新
import re
new_content = re.sub(r'(ARG BUILD_TIME\s*=\s*).*', f'\\1"{build_time}"', content)

with open(target_path, 'w') as f:
    f.write(new_content)

# 4. 提交并推送
commands = [
    "git add .",
    "git commit -m 'release 1.0'",
    "git push -f"
]
subprocess.run(" && ".join(commands), shell=True, check=True)

print(f"Successfully updated Dockerfile with BUILD_TIME: {build_time}")