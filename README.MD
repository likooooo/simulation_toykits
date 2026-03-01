# Simulation-toykits

基于 [MIT 协议](https://opensource.org/licenses/MIT) 的开源光学仿真工具（玩具??）集，以 Web 应用形式提供多层膜、高斯光束与衍射等计算与可视化。

---

## 功能概览

- **Fresnel 计算器**：反射率 / 透射率、Fresnel 系数、膜系结构图；支持 [refractiveindex.info](https://refractiveindex.info/) 材料库与自定义 n/k。
- **高斯光束**：Hermite-Gaussian、Laguerre-Gaussian、Flat-top 等（开发中）。  
Laguerre-Gaussian 仿真示例：[下载 MP4 视频](https://github.com/likooooo/simulation_toykits_assets/raw/main/ipynb/resources/laguerre_gaussain_beam.mp4)。
- **衍射**：Fraunhofer、Fresnel、Rayleigh-Sommerfeld 等模块（开发中）。

详细使用说明见应用内「Fresnel 计算器」页的 **📖 使用说明** 及 [docs/fresnel_calculator_usage.md](https://github.com/likooooo/simulation_toykits/blob/main/docs/fresnel_calculator_usage.md)。

---

## 材料数据

材料光学数据来源于 [refractiveindex.info](https://refractiveindex.info/)。该数据库采用 [CC0 1.0 公共领域贡献](https://creativecommons.org/publicdomain/zero/1.0/) 协议，可自由使用、修改与分发，包括商业用途，无需额外许可。

---

## 快速开始

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
# 1）准备 docker_artifacts
./scripts/prepare_docker.sh ../simulation/build

# 2）构建镜像
docker build -t simulation-toykits:v1 .

# 3）启动容器（前台运行，终端会输出日志）
docker run -p 8052:8052 simulation-toykits:v1
```

---

## 项目结构


| 目录 / 文件     | 说明                                                                             |
| ----------- | ------------------------------------------------------------------------------ |
| **core/**   | 计算与业务逻辑：公式解析、膜系可视化、材料与折射率等；无全局状态依赖，便于复用与测试。                                    |
| **pages/**  | Streamlit 页面与 UI，负责 session 状态与用户交互，计算统一调用 `core`。                             |
| **docs/**   | 使用说明与文档（如 Fresnel 计算器、多层膜构建指令等）。                                               |
| **test/**   | `core` 的单元测试。运行：`pip install pytest` 后执行 `python -m pytest test/ -v`（需在项目根目录）。 |
| **assets/** | 子模块：材料数据库、示例 Notebook 等。                                                       |


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