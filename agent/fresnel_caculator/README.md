# 多层膜设计/分析专家智能体

在 `agent/fresnel_caculator` 下实现的专家系统，能够使用 **core** 中的工具代码，完成与 [pages/fresnel_caculator](../../pages/fresnel_caculator) 网页等价的能力：查询/下载材料 nk、构建多层膜系、计算膜系 R/T 等参数，并将结果输出成文件。使用 **Ollama** 部署的大模型作为“大脑”，由脚本驱动大脑分析用户输入并自动调用 core 中的各种代码。

## 能力概览

- **分析多层膜性能**：直接调用 core 工具，完成单波长/单角度计算、角度扫描、波长扫描、波长-角度二维图等。
- **设计多层膜**：根据用户 prompt 生成多组膜系配置，调用工具得到每组性能，再按用户需求（如 R>99%、T 最大等）挑选最符合要求的膜系并输出结果文件。
- **膜系迭代设计**：按用户指标生成初始膜系后，多轮迭代——每轮根据用户要求选用分析工具（单点 R/T、光谱、角度扫描等）得到当前膜系性能，再根据分析结果与指标生成新一轮膜系公式并重新计算，直到满足要求或达到合理轮数。

## 环境要求

1. **Python**：与仓库主项目一致（建议 3.10+）。
2. **Ollama**：本地已安装并运行，且已拉取支持工具调用/指令遵循的模型（推荐 `qwen2.5:7b` 或 `llama3.2`）。
  **从无到有部署**：可用本目录下的 Docker 一键起 Ollama，见 [docker/README.md](docker/README.md)。
3. **simulation.so**：与网页端相同，需确保 `assets/lib` 下存在 `simulation.so`（可执行 `python scripts/build_and_deploy.py ../simulation/build` 从 simulation 构建目录填充）。
4. **材料数据库**：与网页端相同，使用 `assets/refractiveindex.info-database`（若不存在，refractiveindex 模块会尝试自动下载）。

## 使用方式

**必须从仓库根目录运行**，以便 `core.simulation_loader` 和 `core` 能正确加载：

```bash
cd /path/to/simulation_toykits

# 单次分析：计算给定膜系在 532nm、0° 下的 R/T
python -m agent.fresnel_caculator.run_agent "请计算 Vacuum 0 SiO2 0.1 Vacuum 0 在 532nm、0 度下的 R 和 T"

# 指定输出目录与模型
python -m agent.fresnel_caculator.run_agent "设计一个 532nm 高反膜，R>99%" --output-dir ./out --model qwen2.5:7b

# 迭代设计：按指标多轮优化膜系直到满足
python -m agent.fresnel_caculator.run_agent "迭代设计 532nm 高反膜，R>99%，最多 10 轮，每轮根据 R 结果调整膜系" -o ./out

# 从 stdin 读 prompt
echo "列出材料库里的 SiO2 和 Ta2O5，并导出 SiO2 的 nk 到 out/sio2_nk.csv" | python -m agent.fresnel_caculator.run_agent -o out
```

常用参数：

- `--output-dir` / `-o`：工具生成文件（CSV、图等）的落盘目录，默认 `./fresnel_agent_output`。
- `--model` / `-m`：Ollama 模型名，默认 `qwen2.5:7b`。
- `--base-url`：Ollama 服务地址，默认 `http://localhost:11434`。
- `--max-turns`：最大工具调用轮数，默认 20。
- `--verbose` / `-v`：打印每轮模型输出与工具结果，便于调试。

## 工作流程简述

1. 用户输入一条自然语言请求（prompt）。
2. 脚本将 prompt 与系统提示（专家角色 + 可用工具说明）发给 Ollama。
3. 模型若需使用工具，则按约定输出 JSON：`{"tool": "工具名", "arguments": {...}}`。
4. 脚本在本地执行对应工具（调用 core/refractiveindex 等），将结果再交给模型。
5. 重复 3–4 直到模型输出 `{"answer": "..."}` 或 `{"text": "..."}`，脚本打印最终回答并退出。

设计类请求（如“设计高反膜”）时，模型会先生成多组膜系公式，再通过 `compute_filmstack_batch` 等工具批量计算，根据指标筛选后用 `save_results_csv` 或 `export_nk_to_csv` 等输出文件。

**迭代设计**：当用户要求“迭代设计”或“逐轮优化”时，模型会 (1) 根据需求生成初始膜系；(2) 按需求选用分析工具（如 `compute_filmstack`、`compute_wavelength_vs_rt`、`compute_angle_vs_rt`）得到当前性能；(3) 根据结果与用户指标判断是否达标，未达标则生成新膜系公式并再次调用分析工具；(4) 重复直至满足或轮数用尽，最后总结并输出结果文件。

## 目录与模块

- `tools.py`：封装 core 与 refractiveindex 的各类工具，供大脑通过名称+参数调用。
- `ollama_brain.py`：Ollama 对话与“工具调用/最终回答”的解析协议。
- `run_agent.py`：CLI 入口与 agent 循环（解析参数、维护消息列表、执行工具、输出最终回答）。

## 与网页端的关系

- 功能对齐 [pages/fresnel_caculator](../../pages/fresnel_caculator)：
  - 材料数据库、膜系公式解析、单次 Fresnel 计算、光谱曲线、角度-光谱图等，均由同一套 core 逻辑完成。
- 本 agent 通过 Ollama + 文本/JSON 与工具交互，适合命令行、脚本或后续接入其他前端。

