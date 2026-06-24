# 多层膜设计/分析专家智能体

`agent/fresnel_caculator` 下的 CLI 专家系统：通过 **Ollama** 驱动，调用 **filmstack_simulation**、**simulation_database** 与 **filmstack_visualizer** 插件，实现与 [pages/filmstack_toolkits](../../pages/filmstack_toolkits) 等价的材料查询、膜系构建、R/T 计算与结果导出。

## 能力

- **分析**：单点 R/T、角度/波长扫描
- **设计**：生成多组膜系，批量计算后按指标（如 R>99%）筛选并导出
- **迭代**：多轮生成膜系 → 分析 → 调整，直至达标或轮数用尽

## 环境

1. Python 3.10+（与主仓库一致）
2. Ollama + 支持工具调用的模型（默认 `qwen2.5:7b`）；Docker 部署见 [docker/README.md](docker/README.md)
3. `.simulation_core/simulation.so`（`python scripts/deploy.py local`）；运行前 `source scripts/init-toykits-build-env.sh`
4. 材料库：`simulation_core/assets/database`（`simulation_database` 自动初始化）

## 使用

须在仓库根目录运行：

```bash
cd /path/to/simulation_toykits

# 单点 R/T（532 nm，0°）
python -m agent.fresnel_caculator.run_agent "请计算 air 0 SiO2 0.1 air 0 在 532nm、0 度下的 R 和 T"

# 指定输出目录与模型
python -m agent.fresnel_caculator.run_agent "设计一个 532nm 高反膜，R>99%" -o ./out -m qwen2.5:7b

# 迭代设计
python -m agent.fresnel_caculator.run_agent "迭代设计 532nm 高反膜，R>99%，最多 10 轮" -o ./out

# stdin
echo "导出 SiO2 nk 到 out/sio2_nk.csv" | python -m agent.fresnel_caculator.run_agent -o out
```

| 参数 | 说明 |
|------|------|
| `-o` / `--output-dir` | 输出目录，默认 `./fresnel_agent_output` |
| `-m` / `--model` | Ollama 模型，默认 `qwen2.5:7b` |
| `--base-url` | Ollama 地址，默认 `http://localhost:11434` |
| `--max-turns` | 最大工具轮数，默认 20 |
| `-v` / `--verbose` | 打印每轮模型与工具结果 |

## 工作流程

1. 用户 prompt + 系统提示（角色与工具列表）→ Ollama
2. 模型输出工具 JSON：`{"tool": "…", "arguments": {…}}` → 本地执行 `tools.py`
3. 工具结果回传模型，循环至 `{"answer": "…"}` 或 `{"text": "…"}`

设计/迭代请求会组合 `compute_filmstack`、`compute_filmstack_batch`、`compute_wavelength_vs_rt`、`compute_angle_vs_rt` 等工具，结果经 `save_results_csv` / `export_nk_to_csv` 落盘。

## 模块

| 文件 | 职责 |
|------|------|
| `tools.py` | filmstack / database 工具封装 |
| `ollama_brain.py` | Ollama 对话与 JSON 协议解析 |
| `run_agent.py` | CLI 与 agent 循环 |

Web 端见 [pages/filmstack_toolkits](../../pages/filmstack_toolkits)；本 agent 面向命令行与脚本集成。
