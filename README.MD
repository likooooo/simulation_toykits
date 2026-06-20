# Simulation-toykits

### 获取代码

```bash
# 克隆主仓库
git clone https://github.com/likooooo/simulation_toykits.git
cd simulation_toykits
# 初始化子模块（simulation_core）
git submodule update --init --recursive simulation_core
```

### 本地运行

```bash
# 建议使用虚拟环境
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --no-cache-dir -r requirements.txt

# 编译 simulation_core 并 collect 运行时产物到 .simulation_core/（gitignore，首次或更新 core 后需执行）
cd simulation_core
source 3rdparty/infrastructure/scripts/init-build-env.sh
python3 scripts/build_simulation.py -B build --build-type Release --collect ../.simulation_core
cp build/test_diffraction ../.simulation_core/
cd ..

# 启动 Streamlit
export LD_LIBRARY_PATH=$PWD/.simulation_core
streamlit run app.py --server.port=8052 --server.address=0.0.0.0
# 浏览器访问 http://localhost:8052/
```

### Docker 部署

```bash
python scripts/build_and_deploy.py

# 启动容器（前台运行，终端会输出日志）
docker run -p 8052:8052 simulation-toykits:v1
```

### Hugging Face Space（CI 自动部署）

CI 在测试通过后会调用 `scripts/refresh_hugging_face.py`，向 [simulation-toykits/v1](https://huggingface.co/spaces/simulation-toykits/v1) 推送 `Dockerfile`、CI 编译的 `.simulation_core/` 及自动生成的 `README.md`（含 BUILD_TIME、三仓库 commit / author / email）。`git commit` 消息与 `README.md` 内容一致。应用源码在 HF 构建镜像时从 GitHub 克隆。

本地预览（需已有 `.simulation_core/`）：

```bash
python scripts/refresh_hugging_face.py --dry-run
# 查看 /tmp/simulation-toykits-hf-deploy
```

**一次性 HF 密钥配置**（须与 GitHub Actions `secrets.MY_PRIVATE_KEY_1` 配对）：

| 用途 | 配置 |
|------|------|
| CI `git push` 到 `hf.co` | HF Account → Settings → **SSH Keys** → 添加对应**公钥** |
| HF Docker 构建时 clone GitHub 子模块 | Space → Settings → **Repository secrets** → `SSH_PRIVATE_KEY` = 同一**私钥** |

详见 [`Dockerfile.hugging_face`](Dockerfile.hugging_face) 与 [`scripts/refresh_hugging_face.py`](scripts/refresh_hugging_face.py) 文件头注释。

---

## 脚本使用说明

应用依赖 `simulation.so`（及部分 .so 依赖），统一放在 **`.simulation_core/`** 目录下（gitignore，由编译 collect 生成），由 `core/simulation_loader.py` 按需加载。`simulation` 库以 **simulation_core** 子模块形式 vendored 进本仓库；`build_and_deploy.py` 默认调用 `simulation_core/scripts/build_simulation.py` 编译并 collect 到 `.simulation_core/`，再构建镜像。

材料数据库默认位于 **simulation_core/assets/database**（由 `simulation_database` 自动解析）。

**衍射角计算器（Fresnel → Diffraction Angle）** 还会调用与 `simulation.so` 同目录下的可执行文件 **`test_diffraction`**（由 `simulation_core/tests/test_diffraction.cpp` 构建）。`build_and_deploy.py` 会在 collect 时自动复制；也可手动复制：

```bash
cp simulation_core/build/test_diffraction .simulation_core/
```

### build_and_deploy.py — 构建与部署 Docker 镜像

构建 Docker 镜像（本地或从 GitHub 拉代码），并可选择将镜像导出为 tar 到 **dist/** 或 SCP 到远程主机。默认在 `simulation_core/build` 以 **minimal** 模式（不链接 MKL/CUDA）执行 `cmake .. -DCMAKE_BUILD_TYPE=Release` 并编译，再调用 `simulation_core/scripts/collect_simulation.py` 将 `simulation.so`、白名单 .so 及 `py_core_plugins` 收集到 `.simulation_core/`。

```bash
# 仅构建镜像（不导出 tar；需 .simulation_core/ 已有 simulation.so）
python scripts/build_and_deploy.py

# 先编译 simulation_core 并准备 .simulation_core/（默认 minimal 构建）
python scripts/build_and_deploy.py simulation_core/build

# 启用 MKL CPU 后端（需已安装 oneMKL）
python scripts/build_and_deploy.py simulation_core/build --cpu-only

# 启用 MKL + CUDA 双后端
python scripts/build_and_deploy.py simulation_core/build --full-build

# 构建并把镜像导出为 tar 到 dist/
python scripts/build_and_deploy.py --export-tar

# 构建并传到远程服务器（会先导出 tar 再 scp）
python scripts/build_and_deploy.py simulation_core/build --remote user@myserver

# 指定镜像标签与远程存放目录
python scripts/build_and_deploy.py simulation_core/build --remote user@myserver --image-tag myapp:v2 --remote-path /home/user/images

# 使用 GitHub 上的 Dockerfile 构建（在镜像内克隆仓库与子模块）
python scripts/build_and_deploy.py simulation_core/build --remote user@myserver --source github
```

常用参数：


| 参数                      | 说明                                                            |
| ----------------------- | ------------------------------------------------------------- |
| `path_to_build`         | 可选。simulation_core 的 build 目录（默认 `simulation_core/build`）；若提供则先编译并 collect 到 `.simulation_core/`。 |
| `--cpu-only`            | MKL CPU 构建（`SIMULATION_LINK_MKL=ON`，不链接 CUDA）。 |
| `--full-build`          | MKL + CUDA 双后端（缺依赖时 configure 失败）。 |
| `--export-tar`          | 将镜像导出为 tar 到 `dist/`。                                         |
| `--remote USER@HOST`    | 导出后通过 SCP 传到该主机。                                              |
| `--image-tag`           | 镜像标签，默认 `simulation-toykits:v1`。                              |
| `--remote-path`         | 远程主机上存放 tar 的目录，默认 `~`。                                       |
| `--source local|github` | 使用 `Dockerfile.local`（本地代码）或 `Dockerfile`（从 GitHub 克隆）。       |


### collect_simulation.py（simulation_core 子模块）

`simulation_core/scripts/collect_simulation.py` 会拷贝 `simulation.so`、白名单 .so（如 libuca、libmekil 等）以及 `py_core_plugins` 到指定输出目录；`build_and_deploy.py` 在编译完成后自动调用。若需单独使用：

```bash
cd simulation_core
python scripts/collect_simulation.py -b ./build -o ../.simulation_core
cp build/test_diffraction ../.simulation_core/
# 可选 --tar 将输出目录打包为 .tar.gz
```

---

## 项目结构


| 目录 / 文件      | 说明                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| **core/**    | 计算与业务逻辑：公式解析、膜系可视化、材料与折射率等；无全局状态依赖，便于复用与测试。                                                             |
| **pages/**   | Streamlit 页面与 UI，负责 session 状态与用户交互，计算统一调用 `core`。                                                      |
| **docs/**    | 首页简介（`welcome.md`）与 Fresnel 使用说明（`fresnel_calculator_usage.md`）。 |
| **scripts/** | `build_and_deploy.py`：构建/部署 Docker；默认编译 `simulation_core/build` 并 collect 到 `.simulation_core/`。 |
| **test/**    | `core` 的单元测试。运行：`pip install pytest` 后执行 `LD_LIBRARY_PATH=$PWD/.simulation_core python -m pytest test/ -v`（需先 collect）。                          |
| **simulation_core/** | 子模块：[simulation](https://github.com/likooooo/simulation) 核心库；含 `assets/database` 材料库。 |
| **.simulation_core/** | 运行时产物（gitignore）：`simulation.so`、依赖 .so、`py_core_plugins/` 等，由 collect 生成。 |


---

## 可选配置

**WSL2 局域网访问**：若希望在同一局域网内用其他设备访问：

1. 在 `.wslconfig` 中开启镜像网络模式，使 WSL 与 Windows 共享 IP：
  ```ini
   [wsl2]
   networkingMode=mirrored
  ```
2. 在 Windows 上放行 8052 端口（若被防火墙拦截）：
  ```powershell
   New-NetFirewallRule -DisplayName "Streamlit-Access" -Direction Inbound -LocalPort 8052 -Protocol TCP -Action Allow
  ```

---

## 许可证

本项目采用 MIT 许可证。材料数据遵循 refractiveindex.info 的 CC0 1.0 协议。

## 路线图（规划中）

- zernike 波前拟合
- filmstack optimization
- 绘制解空间地形图, 平坦表示鲁棒性好
- p/q grid dependency PDE
- 一些常见的微分算子的计算器
- rayleigh length 计算器
- green's function