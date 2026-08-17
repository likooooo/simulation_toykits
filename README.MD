# Simulation-toykits

### 获取代码

```bash
git clone https://github.com/likooooo/simulation_toykits.git
cd simulation_toykits
```

CI 与纯消费场景**无需**子模块；本地编译 C++ 时再 init：

```bash
git submodule update --init --recursive simulation_core
```

### 开发者环境配置

#### 消费路径（CI / 仅跑 Streamlit / pytest）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
python scripts/build_toykits.py --download_toolkits
source scripts/init-toykits-build-env.sh
python scripts/build_toykits.py local
```

#### 编译路径（改 C++ / 发布 artifact）

系统包与 [simulation_core CI](simulation_core/.github/workflows/build-ci.yml) 一致；Python 用 `requirements-build.txt`：

```bash
sudo apt install -y \
  build-essential clang-20 llvm-20-dev pkg-config python3-venv python3-dev \
  libboost-all-dev libeigen3-dev libfftw3-dev
pip install -r requirements-build.txt
python scripts/build_toykits.py              # 全量：toolkits + database + bench
python scripts/build_toykits.py release      # 发布到 simulation_toolkits_artifacts
```

编译环境由 `build_toykits.py` 内部 subshell 捕获，**无需**事先 `source init-toykits-build-env.sh`。

#### 运行时环境变量

`scripts/init-toykits-build-env.sh` 为**纯 runtime**，pytest / Streamlit 前 source。导出：

- `SIMULATION_ARTIFACTS_DIR`（默认 `{repo}/.simulation_toolkits`）
- `SIMULATION_DATABASE_DIR`（默认 `{artifacts}/assets`）
- `GENERATE_GOLDEN_TOOLS_DIR`
- `PYTHONPATH`、`LD_LIBRARY_PATH`

```bash
source scripts/init-toykits-build-env.sh
```

#### 测试

与 CI（[`.github/workflows/ci.yml`](.github/workflows/ci.yml)）对齐：

```bash
source scripts/init-toykits-build-env.sh
python scripts/build_toykits.py --verify_runtime_commit
python scripts/check_import_contract.py
python -m pytest test/ -v
```

本地对齐全仓 CI（先 build 后 test）：

```bash
python scripts/ci_orchestrate.py build
python scripts/ci_orchestrate.py test
# 或: python scripts/ci_orchestrate.py all
```

TMM 须在 `simulation_core` Release `build/` 下跑（**不是** `.simulation_toolkits`）：

```bash
source simulation_core/scripts/init-simulation-build-env.sh build
bash simulation_core/assets/ipynb/simulation/TMM/run_tmm.sh test
```

C++ 全量测试见 [simulation_core/README.md](simulation_core/README.md)。

**granfilm**（对照测试，需 toykits artifact）：`source scripts/init-toykits-build-env.sh` 后对 `granfilm/` 跑 pytest；对比报告入口为 `granfilm/build_granfilm_compare.py`。

### Docker 部署

须已有 `.simulation_toolkits/` 且事先 `export SIMULATION_DATABASE_KEY`：

```bash
export SIMULATION_DATABASE_KEY=...
python scripts/build_toykits.py docker
docker run -e SIMULATION_DATABASE_KEY=... -p 8052:8052 simulation-toykits:v1
```

### Hugging Face Space（CI 自动部署）

CI：`--download_toolkits` → pytest → `python scripts/build_toykits.py hf`，向 [simulation-toykits/v1](https://huggingface.co/spaces/simulation-toykits/v1) 推送 `Dockerfile` + `README.md`。HF 构建时 HTTPS 克隆应用层，并从 [simulation_toolkits_artifacts](https://github.com/likooooo/simulation_toolkits_artifacts) **latest Release** 下载 runtime。

**HF SSH 密钥**（配对 GitHub Actions `secrets.MY_PRIVATE_KEY_1`）：HF Account → Settings → **SSH Keys** → 公钥。

详见 [Dockerfile.hugging_face](Dockerfile.hugging_face) 与 [scripts/build_toykits.py](scripts/build_toykits.py)。

---

## 脚本与运行时

架构与部署 SOP：[scripts/simulation_toykits_deploy.md](scripts/simulation_toykits_deploy.md)。

运行时产物位于 `.simulation_toolkits/`（gitignore）。功能概览见 [docs/welcome.md](docs/welcome.md)。

### build_toykits.py

| 调用 | 作用 |
|------|------|
| **（默认）** / `build` | 全量编译 |
| `--toolkits` / `--bench` / `--database` | 正向 flag：只跑列出的 build 步骤 |
| `build local` / `build docker` / `build hf` / `build release` | 编译后接消费步骤 |
| `local` / `docker` / `hf` / `release` | 仅消费（不隐式编译） |
| `--download_toolkits` | 下载 GitHub Release latest |

---

## 项目结构

| 目录 / 文件 | 说明 |
|-------------|------|
| **filmstack_simulation/** | 多层膜 Streamlit 模块（可独立拷贝；含嵌套 `filmstack_optimization/`） |
| **simulation_database/** | 仿真数据库 Streamlit 模块（可独立拷贝；`render_page` 须传 material/spectrum keys） |
| **open_filters/** | OpenFilters ↔ TMM 对照与 LM 细化 |
| **granfilm/** | GranFilm 对照测试与报告脚本 |
| **pages/** | Streamlit 多页入口 |
| **core/** | 宿主计算逻辑 |
| **scripts/** | `build_toykits.py`、`ci_orchestrate.py`、[simulation_toykits_deploy.md](scripts/simulation_toykits_deploy.md) |
| **test/** | 单元测试 |
| **simulation_core/** | 子模块（仅编译时需要） |
| **.simulation_toolkits/** | runtime artifact（gitignore） |

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
