#!/usr/bin/env python3
"""
Fresnel 多层膜专家智能体入口脚本。
使用 Ollama 作为大脑，根据用户 prompt 调用 core 工具完成：查询/下载材料 nk、构建膜系、计算 R/T、导出结果。
设计模式：可根据用户需求生成多组膜系配置，批量计算后挑选最符合要求的膜系并输出。

用法示例：
  # 从仓库根目录运行（以便加载 simulation.so）
  python -m agent.fresnel_caculator.run_agent "请计算 Vacuum 0 SiO2 0.1 Vacuum 0 在 532nm、0 度下的 R 和 T"
  python -m agent.fresnel_caculator.run_agent "设计一个 532nm 高反膜，R>99%" --output-dir ./out --model qwen2.5:7b
"""

import argparse
import json
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_AGENT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.fresnel_caculator.ollama_brain import SYSTEM_PROMPT, chat_turn
from agent.fresnel_caculator.tools import run_tool, TOOLS


def main():
    parser = argparse.ArgumentParser(
        description="Fresnel 多层膜设计/分析专家智能体（Ollama + core 工具）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="用户请求（若未提供则从 stdin 读取一行）",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./fresnel_agent_output",
        help="输出目录，工具生成的文件将落在此目录（默认: ./fresnel_agent_output）",
    )
    parser.add_argument(
        "--model", "-m",
        default="qwen2.5:7b",
        help="Ollama 模型名（默认: qwen2.5:7b）",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama 服务地址",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="最大工具调用轮数（默认: 20）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印每轮模型输出与工具结果",
    )
    args = parser.parse_args()

    prompt = args.prompt.strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("请提供 prompt 或从 stdin 输入")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir_abs = os.path.abspath(args.output_dir)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"输出目录为：{output_dir_abs}\n\n用户请求：{prompt}",
        },
    ]

    turn = 0
    while turn < args.max_turns:
        turn += 1
        content, parsed = chat_turn(
            args.model,
            messages,
            base_url=args.base_url,
            timeout=120.0,
        )
        if args.verbose:
            print(f"[Turn {turn}] Model output:\n{content}\n", file=sys.stderr)

        if parsed.get("parse_error"):
            if "answer" not in parsed and "text" not in parsed:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "请用 JSON 格式回复：要么 {\"tool\": \"工具名\", \"arguments\": {...}} 要么 {\"answer\": \"你的最终回答\"}。不要输出其他文字。",
                })
                continue

        if "tool" in parsed and "arguments" in parsed:
            tool_name = parsed["tool"]
            tool_args = parsed["arguments"] or {}
            for key in ("out_path", "out_figure_path", "out_figure_rt_path", "out_figure_nk_path"):
                val = tool_args.get(key)
                if val is None or (isinstance(val, str) and not val.strip()):
                    continue
                if not os.path.isabs(str(val)):
                    tool_args[key] = os.path.join(output_dir_abs, os.path.basename(str(val)))

            result = run_tool(tool_name, tool_args)
            result_str = json.dumps(result, ensure_ascii=False, indent=2)
            if args.verbose:
                print(f"[Tool] {tool_name}({json.dumps(tool_args, ensure_ascii=False)}) => {result_str[:500]}...\n", file=sys.stderr)

            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"工具调用结果：\n{result_str}\n\n请根据结果继续：若还需其他工具则再输出一个 JSON 工具调用；否则输出 JSON 包含 \"answer\" 或 \"text\" 作为对用户的最终回答。",
            })
            continue

        if "answer" in parsed or "text" in parsed:
            answer = parsed.get("answer") or parsed.get("text") or content
            print(answer)
            return 0

        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": "请直接给出对用户的最终回答，用 JSON：{\"answer\": \"你的回答内容\"}",
        })

    print("[Agent] 达到最大轮数，未得到最终回答。", file=sys.stderr)
    if messages and messages[-1].get("role") == "assistant":
        print(messages[-1].get("content", ""), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
