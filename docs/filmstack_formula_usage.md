# 多层膜构建指令说明

在 **Filmstack Simulation** 页面，通过「多层膜构建指令」或左侧「预设膜系」下拉框定义从入射侧到基底侧的膜层序列。

依赖工作区材料的预设（如减反膜、OLED 栈）须先在 **Simulation Database** 工作区加载对应材料；inline n/k 预设（如 SPR、椭偏基准）可在公式中直接指定 n、k，无需工作区条目。

---

## 多层膜构建指令

### 基本单元：材料 + 厚度

- 每层：**材料名** + **厚度（μm）**，空格分隔。
- 示例：`SiO2 0.1` — 0.1 μm SiO2；`air 0` — 0 厚度空气（常用作入射/出射介质）。

### 可选：覆盖 n、k

- 格式：`材料 厚度 n k`
- 示例：`Ta2O5 0.05 2.1 0.001` — 0.05 μm Ta2O5，n=2.1，k=0.001。

### 括号周期重复

- `( … )^数字` 表示序列重复次数。
- 示例：`(SiO2 0.1 Ta2O5 0.01)^5` — 5 个周期，共 10 层。

### 完整示例

- 单层：`air 0 SiO2 0.1 air 0`
- 周期：`air 0 (SiO2 0.1 Ta2O5 0.01)^5 air 0`
- inline n/k：`air 0 (SiO2 0.1 1.45 0 Ta2O5 0.01 2.1 0.001)^3 air 0`
- 工作区已有材料时可省略 n/k：`air 0 (SiO2 0.1 Ta2O5 0.01)^3 Si 0`

### 解析、TMM 与结构图

唯一入口：`filmstack_visualizer.layers_from_formula`（先 `import simulation`，再 `import filmstack_visualizer`）。流程：空白规范化 → token 解析 → 材料 resolve → **bookend 扩充**。

输出 `(materials, thicknesses_um)` 为 TMM 对齐栈，**恒满足** `thicknesses_um[0] == thicknesses_um[-1] == 0`。

首/末层厚度非 0 时，解析器在同材料外侧补 `depth=0` bookend，例如 `air 0.1 … Si 0.1` → `air(0) | air(0.1) | … | Si(0.1) | Si(0)`。已写 `air 0 … Si 0` 时不重复扩充。

- **TMM**：`layers_from_formula` → `build_tmm_layers`（幂等）→ R/T
- **结构图**：同上 → `plot_filmstack(layers)`；输出由 `SAVE_TO_FILE` 控制（`viz_io.save_or_show_fig`）
- **优化**：`stack_from_formula(formula, materials_db)` 复用上述 bookend；`materials_db` 为运行时 `material_s` 字典，或配置 query path 经 `materials_db_from_token_paths` → `read_at_query_path` 加载

---

## 预设膜系

下拉框含 7 个 TMM/Oghma 对齐预设（另含「自定义」）：

| id | 名称 | 材料策略 |
|----|------|----------|
| `ar_qw_si` | 减反膜系 | 工作区材料 + QW 厚度联动 |
| `bragg_mirror` | 布拉格反射镜 | inline nk（`H`/`air`） |
| `optical_filter` | 光学滤光片 | inline nk（`H`/`L`/`Exit`）；末层 Exit + Si bookend |
| `fabry_perot` | FP 共振腔 | inline nk（`Mirror`）；基底 `Si` |
| `oled_ito_al` | OLED 栈 | 工作区材料（ito / NPD / Alq3 / …） |
| `spr_bk7_cr_au` | SPR 金属膜 | 全 inline nk |
| `paper_sio2_si` | 椭偏基准 | 全 inline nk |

**工作区材料**：Simulation Database 页 standalone 模式下工作区初始为空；Filmstack 预设所需材料须手动加入工作区，或宿主在 `ensure_workspace_initialized(..., material_path_keys=..., spectrum_path_keys=...)` 中传入 [`DEFAULT_MATERIAL_PATH_KEYS`](../toykits_config.py) 与 [`DEFAULT_SPECTRUM_PATH`](../toykits_config.py)（simulation_toykits 的 `app.py` 启动时自动预加载）。推荐路径列表与 preset 说明见 [`filmstack_simulation/README.md`](../filmstack_simulation/README.md)。

---

## 计算结果正确性参考

R/T 角度扫描验证案例：[case_R_T_sweep_angle.ipynb](https://github.com/likooooo/simulation_toykits_assets/blob/main/ipynb/case_R_T_sweep_angle.ipynb)
