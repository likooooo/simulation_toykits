# Simulation-toykits

基于 [MIT 协议](https://opensource.org/licenses/MIT) 的开源光学仿真 Web 工具集。克隆、编译与部署见 [README.MD](https://github.com/likooooo/simulation_toykits/blob/main/README.MD)。

---

## 功能概览

**Filmstack 工具集**（`pages/filmstack_toolkits/`）

- **Simulation Database**：材料/光谱浏览与工作区
- **Filmstack Simulation**：多层膜 R/T/Ψ/Δ 二维图与切片
- **Freehand Optimization**：手绘 R/T 目标 + 厚度局部搜索
- **Diffraction angle**：衍射角计算
- 支持 [refractiveindex.info](https://refractiveindex.info/) 材料库与公式内 inline n/k

**高斯光学工具集**（`pages/gaussian_optics_toolkits/`）

- Plane / Quadratic / Spherical 波；Flat-top、Hermite-Gaussian、Laguerre-Gaussian 光束
- Laguerre-Gaussian 示例：[MP4 视频](https://github.com/likooooo/simulation_toykits_assets/raw/main/ipynb/resources/laguerre_gaussain_beam.mp4)

**PDE 工具集**（`pages/simulation_toykits/`）

- Sturm-Liouville 及带时间项方程（热传导 / 波动）

Filmstack 公式语法见 **Filmstack Simulation** 页帮助链接及 [filmstack_formula_usage.md](https://github.com/likooooo/simulation_toykits/blob/main/docs/filmstack_formula_usage.md)。

---

## 材料数据

光学材料数据来自 [refractiveindex.info](https://refractiveindex.info/)，遵循 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)，可自由使用、修改与分发（含商业用途）。
