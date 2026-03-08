# Cursor Cloud Agent + 本地 Fresnel 工具

用 **Cursor Cloud Agents** 当“大脑”，在你的**本地环境**执行 Fresnel 工具（材料库、膜系计算、R/T、导出等），实现与 [fresnel_caculator](../fresnel_caculator) 本地智能体等价的能力，但推理由 Cursor 云端大模型完成。

## 原理

1. **Runner** 根据你的 prompt 创建一条 Cursor Cloud Agent 任务（附带完整工具说明与约定）。
2. Cloud Agent 在 Cursor 云端运行，可阅读你指定的 GitHub 仓库代码。
3. 当 Agent 需要调用工具时，按约定在回复中输出一行：`TOOL_CALL: {"tool": "工具名", "arguments": {...}}`。
4. **本地 Runner** 轮询该 Agent 的对话；发现 `TOOL_CALL` 后，在本地调用 [fresnel_caculator/tools](../fresnel_caculator/tools.py) 执行对应工具，把结果通过 Cursor API 的 **Follow-up** 以 `TOOL_RESULT: <JSON>` 发回。
5. Cloud Agent 根据 `TOOL_RESULT` 继续推理，再输出下一个 `TOOL_CALL` 或最终 `ANSWER: ...`。
6. Runner 收到 `ANSWER` 后打印并退出。

因此：**大脑在 Cursor 云端，工具执行在你本机**（需要 simulation.so、材料库等），与本地 agent 使用同一套工具。

## 环境要求

1. **Python**：与仓库一致（建议 3.10+），从**仓库根目录**运行。
2. **Python 依赖（requirements.txt）**：本 Agent 调用的工具依赖仓库中的 **core** 模块（材料库、膜系计算、R/T 等）。**若要调用 core 里的工具，必须先按仓库根目录的 `requirements.txt` 初始化运行环境**，例如：`pip install -r requirements.txt`。否则 Runner 在执行 TOOL_CALL 时会因缺少依赖或 core 加载失败而报错。
3. **Cursor API Key**：在 [Cursor Dashboard → Integrations](https://cursor.com/dashboard?tab=integrations) 创建；设置环境变量 `CURSOR_API_KEY`。
4. **GitHub 仓库**：Cloud Agents **只能使用「已在 Cursor 中关联」的仓库**（不是任意公开仓库即可）。请用 Cursor 打开该仓库、或在 [cursor.com](https://cursor.com) 设置中连接 GitHub 并授权该仓库；否则会报 `Failed to verify existence of branch 'main'`。创建 Agent 时传 `--repository`（该仓库的 clone URL）。可先运行 `--list-repos` 查看当前可用的仓库列表。
5. **simulation.so、材料库**：与本地 [fresnel_caculator](../fresnel_caculator) 相同，需在运行 Runner 的机器上准备好（如 `docker_artifacts/simulation.so`、`assets/refractiveindex.info-database`）。

## 使用方式

**必须从仓库根目录运行**（以便加载 `core` 与 simulation.so）：

```bash
cd /path/to/simulation_toykits

# 设置 Cursor API Key（网页 Dashboard → Integrations 创建）
export CURSOR_API_KEY=key_xxx

# 指定你的 GitHub 仓库 URL（必须可访问）
python -m agent.cursor_cloud_fresnel.runner \
  "设计一个 532nm 高反膜，R>99%" \
  --repository https://github.com/你的用户名/simulation_toykits \
  --output-dir ./cloud_out

# 从 stdin 读 prompt
echo "请计算 Vacuum 0 SiO2 0.1 Vacuum 0 在 532nm、0 度下的 R 和 T" | \
  python -m agent.cursor_cloud_fresnel.runner -r https://github.com/你的用户名/simulation_toykits -o ./out
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--repository` / `-r` | **必填**。GitHub 仓库 URL。 |
| `--output-dir` / `-o` | 工具生成文件落盘目录，默认 `./fresnel_cloud_output`。 |
| `--ref` | 仓库**默认分支**，必须与 GitHub 上一致；默认 `main`，若仓库是 `master` 等请显式写 `--ref master`。 |
| `--list-repos` | 列出当前 Cursor 可访问的 GitHub 仓库后退出；用于确认目标仓库是否已关联。 |
| `--model` / `-m` | 可选，如 `claude-4-sonnet`；不传则用 Cursor 默认模型。 |
| `--max-rounds` | 最大工具调用轮数，默认 20。 |
| `--poll-interval` | 轮询对话间隔（秒），默认 8。 |
| `--verbose` / `-v` | 打印 Runner 的创建/工具执行等日志。 |

## 目录与模块

| 文件 | 说明 |
|------|------|
| `cursor_api.py` | Cursor Cloud Agents API：创建 Agent、拉取对话、发送 Follow-up。 |
| `prompts.py` | Cloud Agent 的系统说明与 TOOL_CALL/TOOL_RESULT/ANSWER 约定。 |
| `runner.py` | 入口：创建 Agent、轮询、解析 TOOL_CALL、本地执行工具、回传 TOOL_RESULT、解析 ANSWER。 |

工具实现复用 [fresnel_caculator/tools.py](../fresnel_caculator/tools.py)，不做重复开发。

## 与本地 agent 的对比

| 能力 | [fresnel_caculator](../fresnel_caculator)（本地） | cursor_cloud_fresnel（本方案） |
|------|---------------------------------------------------|--------------------------------|
| 大脑 | Ollama / OpenAI（本地或 API） | Cursor Cloud Agent（云端） |
| 工具执行 | 本机 | 本机（Runner 调同一套 tools） |
| 依赖 | Ollama 或 OPENAI_API_KEY | CURSOR_API_KEY + GitHub 仓库 |
| 适用场景 | 内网、无 Cursor 账号 | 希望用 Cursor 付费模型、且能接受轮询延迟 |

## 注意

- **仓库须在 Cursor 中关联**：若报错 `Failed to verify existence of branch 'main'` 且你确认 GitHub 上分支正确，多半是该仓库尚未被 Cursor 授权。请用 **Cursor 打开该仓库**（File → Open Folder 指向该 repo），或到 cursor.com 设置里连接 GitHub 并授权；再运行 `python -m agent.cursor_cloud_fresnel.runner --list-repos` 确认列表中出现该仓库后再创建 Agent。
- **分支**：若默认分支不是 `main`，请用 `--ref` 指定（如 `--ref master`）。
- Cloud Agent 执行与轮询有延迟，单轮工具调用通常需数十秒。
- 需保证仓库在 GitHub 上可访问（或 Cursor 已关联的仓库）。
- 若 Cursor 侧限流或失败，Runner 会抛出相应错误；可重试或检查 API Key / 配额。
