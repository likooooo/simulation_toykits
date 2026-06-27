# Simulation-toykits

### 获取代码

```bash
git clone https://github.com/likooooo/simulation_toykits.git
cd simulation_toykits
git submodule update --init --recursive simulation_core
```

### 开发者环境配置

链路：**系统包 → Python venv → 编译 → 测试**。

#### 1. 系统依赖

与 [simulation_core CI](simulation_core/.github/workflows/build-ci.yml) 一致：

```bash
sudo apt install -y \
  build-essential \
  clang-20 \
  llvm-20-dev \
  pkg-config \
  python3-venv \
  python3-dev \
  libboost-all-dev \
  libeigen3-dev \
  libfftw3-dev \
  libopenmpi-dev
```

#### 2. Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --no-cache-dir -r requirements.txt
```

仅开发 simulation_core 子模块时见 [simulation_core/README.md](simulation_core/README.md)。

#### 3. 环境变量

`scripts/init-toykits-build-env.sh` 设置 `SIMULATION_*`、`PYTHONPATH`、`LD_LIBRARY_PATH`（内部 source `simulation_core/scripts/init-simulation-build-env.sh`）。

```bash
source scripts/init-toykits-build-env.sh
```

#### 4. 编译与启动

```bash
source scripts/init-toykits-build-env.sh
python scripts/build_toykits.py              # 编译 + collect 到 .simulation_core/
python scripts/build_toykits.py local        # 上一步 + 启动 Streamlit（8052）
```

#### 5. 测试

须先 collect（见 §4）：

```bash
source scripts/init-toykits-build-env.sh
python -m pytest test/ -v
```

C++ 全量测试见 [simulation_core/README.md](simulation_core/README.md)。

### 本地运行

完成 §1–2 后：

```bash
source .venv/bin/activate
source scripts/init-toykits-build-env.sh
python scripts/build_toykits.py local
```

编译、collect、环境变量细节见上文 §3–4 及 [scripts/simulation_toykits_deploy.md](scripts/simulation_toykits_deploy.md)。

### Docker 部署

```bash
python scripts/build_toykits.py docker
docker run -p 8052:8052 simulation-toykits:v1
```

### Hugging Face Space（CI 自动部署）

CI 测试通过后执行 `scripts/build_toykits.py hf`，向 [simulation-toykits/v1](https://huggingface.co/spaces/simulation-toykits/v1) 推送 `Dockerfile`、`.simulation_core/` 及自动生成的 `README.md`（BUILD_TIME、三仓库 commit / author / email）。HF 构建时从 GitHub 克隆应用源码。

**一次性 HF 密钥**（配对 GitHub Actions `secrets.MY_PRIVATE_KEY_1`）：

| 用途 | 配置 |
|------|------|
| CI `git push` 到 `hf.co` | HF Account → Settings → **SSH Keys** → 对应**公钥** |
| HF Docker 构建 clone 子模块 | Space → Settings → **Repository secrets** → `SSH_PRIVATE_KEY` = 同一**私钥** |

详见 [Dockerfile.hugging_face](Dockerfile.hugging_face) 与 [scripts/build_toykits.py](scripts/build_toykits.py)。

---

## 脚本与运行时

架构与部署 SOP：[scripts/simulation_toykits_deploy.md](scripts/simulation_toykits_deploy.md)。

运行时产物位于 `.simulation_core/`（gitignore）：`simulation.so`、依赖 `.so`、`py_core_plugins/`、`simulation_plugins/`、`assets/database/` 等。pytest 与 Streamlit 运行前须 `source scripts/init-toykits-build-env.sh`（collect 之后）。功能概览见 [docs/welcome.md](docs/welcome.md)。

Diffraction angle 页另需同目录下的 `test_diffraction` 可执行文件；`build_toykits` collect 时自动复制。

### build_toykits.py

| 子命令 | 作用 |
|--------|------|
| **（默认）** / `build` | 编译 + collect 到 `.simulation_core/` |
| `local` | build + 启动 Streamlit（8052，脚本设置运行 env） |
| `docker` | build + 材料库打包 + `docker build` |
| `hf` | build + 推送 `.simulation_core/` 到 Hugging Face Space |

collect 由 `build_simulation.py --collect` 内联实现。

---

## 项目结构

| 目录 / 文件 | 说明 |
|-------------|------|
| **filmstack_simulation/** | 多层膜 Streamlit 模块（可独立拷贝） |
| **simulation_database/** | 仿真数据库 Streamlit 模块（可独立拷贝） |
| **pages/** | Streamlit 多页入口；薄 wrapper |
| **ui/** | 宿主设计 token（`design_tokens.css`） |
| **core/** | beams、衍射角、PDE 等宿主计算逻辑 |
| **docs/** | `welcome.md`、`filmstack_formula_usage.md` |
| **scripts/** | `build_toykits.py`、[simulation_toykits_deploy.md](scripts/simulation_toykits_deploy.md) |
| **test/** | 单元测试；须先 collect 并 source init 脚本 |
| **simulation_core/** | 子模块；含 `assets/database` |
| **.simulation_core/** | collect 产物（gitignore） |

---

## 可选配置

**WSL2 局域网访问**

1. `.wslconfig` 开启镜像网络：
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
2. Windows 防火墙放行 8052：
   ```powershell
   New-NetFirewallRule -DisplayName "Streamlit-Access" -Direction Inbound -LocalPort 8052 -Protocol TCP -Action Allow
   ```

---

## 许可证

MIT 许可证。材料数据遵循 refractiveindex.info CC0 1.0。

## 路线图（规划中）

- Zernike 波前拟合
- Filmstack 全局/多目标优化与解空间地形图（Freehand 局部厚度优化已提供）
- p/q grid dependency PDE
- 常见微分算子计算器
- Rayleigh length 计算器
- Green's function
