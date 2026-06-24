# Agent 示例命令

默认模型：`qwen2.5:7b`（见 [../README.md](../README.md)）。

```bash
python -m agent.fresnel_caculator.run_agent '列出 SiO2 和 Ta2O5 材料索引'
python -m agent.fresnel_caculator.run_agent '下载 Malitson 的 SiO2 nk 数据'
python -m agent.fresnel_caculator.run_agent '解析多层膜公式：air 0 SiO2 0.1 Ta2O5 0.02 fused_silica 0'
```

CSV 默认写入 `fresnel_agent_output/`。
