# Simulation-toykits

### 获取代码

```bash
# 克隆主仓库
git clone https://github.com/likooooo/simulation_toykits.git
cd simulation_toykits
# 初始化资源子模块（材料库、示例等）
git submodule update --init --recursive
```

### 本地运行

```bash
# 建议使用虚拟环境
python3 -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
pip install --no-cache-dir -r requirements.txt

# 启动 Streamlit
streamlit run app.py --server.port=8052 --server.address=0.0.0.0
# 浏览器访问 http://localhost:8052/
```

### Docker 部署

```bash
python scripts/build_and_deploy.py

# 启动容器（前台运行，终端会输出日志）
docker run -p 8052:8052 simulation-toykits:v1
```

---

## 脚本使用说明

应用依赖 `simulation.so`（及部分 .so 依赖），统一放在 **assets/lib** 目录下，由 `core/simulation_loader.py` 按需加载。传入 simulation 的 build 目录时，`build_and_deploy.py` 会先在该目录下编译并调用 [simulation](https://github.com/likooooo/simulation) 的 `collect_simulation.py` 填充 `assets/lib`，再构建镜像。

### build_and_deploy.py — 构建与部署 Docker 镜像

构建 Docker 镜像（本地或从 GitHub 拉代码），并可选择将镜像导出为 tar 到 **dist/** 或 SCP 到远程主机。若传入 `path_to_build`，会先在该目录执行 `cmake .. -DCMAKE_BUILD_TYPE=Release` 并编译，再调用 simulation 的 `scripts/collect_simulation.py` 将 `simulation.so`、白名单 .so 及 `plugin`/`core_plugins` 收集到 `assets/lib`。

```bash
# 仅构建镜像（不导出 tar）
python scripts/build_and_deploy.py

# 先准备 assets/lib 再构建（若 assets/lib 尚无 simulation.so）
python scripts/build_and_deploy.py ../simulation/build

# 构建并把镜像导出为 tar 到 dist/
python scripts/build_and_deploy.py --export-tar

# 构建并传到远程服务器（会先导出 tar 再 scp）
python scripts/build_and_deploy.py ../simulation/build --remote user@myserver

# 指定镜像标签与远程存放目录
python scripts/build_and_deploy.py ../simulation/build --remote user@myserver --image-tag myapp:v2 --remote-path /home/user/images

# 使用 GitHub 上的 Dockerfile 构建（在镜像内克隆仓库与 assets 子模块）
python scripts/build_and_deploy.py ../simulation/build --remote user@myserver --source github
```

常用参数：


| 参数                      | 说明                                                            |
| ----------------------- | ------------------------------------------------------------- |
| `path_to_build`         | 可选。simulation 的 build 目录；若提供则先在该目录编译并调用 simulation 的 `collect_simulation.py` 填充 assets/lib。 |
| `--export-tar`          | 将镜像导出为 tar 到 `dist/`。                                         |
| `--remote USER@HOST`    | 导出后通过 SCP 传到该主机。                                              |
| `--image-tag`           | 镜像标签，默认 `simulation-toykits:v1`。                              |
| `--remote-path`         | 远程主机上存放 tar 的目录，默认 `~`。                                       |
| `--source local|github` | 使用 `Dockerfile.local`（本地代码）或 `Dockerfile`（从 GitHub 克隆）。       |


### collect_simulation.py（simulation 仓库）

[simulation](https://github.com/likooooo/simulation) 的 `scripts/collect_simulation.py` 会拷贝 `simulation.so`、白名单 .so（如 libuca、libmekil 等）以及 `plugin`/`core_plugins` 到指定输出目录；传入 `path_to_build` 时由 `build_and_deploy.py` 在编译完成后自动调用。若需单独使用（例如指定其他输出路径或打包 tar.gz），可在 simulation 仓库下执行：

```bash
cd /path/to/simulation
python scripts/collect_simulation.py -b ./build -o /path/to/simulation_toykits/assets/lib
# 可选 --tar 将输出目录打包为 .tar.gz
```

---

## 项目结构


| 目录 / 文件      | 说明                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| **core/**    | 计算与业务逻辑：公式解析、膜系可视化、材料与折射率等；无全局状态依赖，便于复用与测试。                                                             |
| **pages/**   | Streamlit 页面与 UI，负责 session 状态与用户交互，计算统一调用 `core`。                                                      |
| **docs/**    | 使用说明与文档（如 Fresnel 计算器、多层膜构建指令等）。                                                                        |
| **scripts/** | `build_and_deploy.py`：构建/部署 Docker；传入 simulation 的 build 目录时会先编译并调用 simulation 的 `collect_simulation.py` 填充 assets/lib。 |
| **test/**    | `core` 的单元测试。运行：`pip install pytest` 后执行 `python -m pytest test/ -v`（需在项目根目录）。                          |
| **assets/**  | 子模块：材料数据库、示例 Notebook 等；**assets/lib** 下存放 simulation.so 及其 .so 依赖，供应用加载。                               |


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

## TODO List

- zernike 波前拟合
- filmstack optimization
- 绘制解空间地形图, 平坦表示鲁棒性好
- p/q grid dependency PDE 
- 一些 SL 的特例，比如带时间项的波动方程
- 一些常见的微分算子的计算器
- diffraction angle caculator
- rayleigh length 计算器  
- green's function