# Simulation-toykits

基于 [MIT 协议](https://opensource.org/licenses/MIT) 的开源光学仿真 Web 工具集。克隆、编译与部署见 [README.MD](https://github.com/likooooo/simulation_toykits/blob/main/README.MD)。

---

## 功能概览

**Filmstack 工具集**（`pages/filmstack_toolkits/`）

- **Simulation Database**：材料/光谱浏览与工作区
- **Filmstack Simulation（多层膜仿真）**：多层膜 R/T/Ψ/Δ 二维图与切片
- **Freehand Optimization（Freehand 局部优化）**：手绘 R/T 目标 + 厚度局部搜索
- **Diffraction Angle**：衍射角计算
- 支持公式内 inline n/k

**高斯光学工具集**（`pages/gaussian_optics_toolkits/`）

- Plane / Quadratic / Spherical 波；Flat-top、Hermite-Gaussian、Laguerre-Gaussian 光束
- Laguerre-Gaussian 示例：[MP4 视频](https://github.com/likooooo/simulation_toykits_assets/raw/main/ipynb/resources/laguerre_gaussain_beam.mp4)（上游 assets 文件名拼写为 `gaussain`）

**PDE 工具集**（`pages/simulation_toykits/`）

- Sturm-Liouville 及带时间项方程（热传导 / 波动）

Filmstack 公式语法见 **Filmstack Simulation** 页帮助链接及 [多层膜构建指令](/filmstack-formula-usage)。
