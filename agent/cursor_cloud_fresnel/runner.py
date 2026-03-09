"""
Cloud Agent + 本地工具 Runner：创建 Cursor Cloud Agent，轮询对话；
当云端输出 TOOL_CALL 时在本地执行工具并回传 TOOL_RESULT，直到云端输出 ANSWER 或达到最大轮数。
需从仓库根目录运行，以便加载 simulation.so 与材料库。
"""

import json
import os
import sys
import time
from typing import Optional, Tuple

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_AGENT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.cursor_cloud_fresnel.cursor_api import (
    create_agent,
    followup,
    get_conversation,
    list_repositories,
    wait_until_idle,
)
from agent.cursor_cloud_fresnel.prompts import (
    CLOUD_AGENT_INSTRUCTIONS,
    build_user_prompt,
)
from agent.fresnel_caculator.tools import run_tool

TOOL_CALL_PREFIX = "TOOL_CALL:"
TOOL_RESULT_PREFIX = "TOOL_RESULT:"
ANSWER_PREFIX = "ANSWER:"

OUTPUT_KEYS = (
    "out_path",
    "out_figure_path",
    "out_figure_rt_path",
    "out_figure_nk_path",
)


def _extract_tool_call(text: str) -> Optional[dict]:
    """从一段文本中找出一行 TOOL_CALL: {...}，解析为 dict。找不到或解析失败返回 None。"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(TOOL_CALL_PREFIX):
            raw = line[len(TOOL_CALL_PREFIX) :].strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
    return None


def _extract_answer(text: str) -> Optional[str]:
    """从文本中找一行 ANSWER: xxx，返回 xxx。若没有则返回 None。"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(ANSWER_PREFIX):
            return line[len(ANSWER_PREFIX) :].strip()
    return None


def _last_assistant_message(conv: dict) -> Tuple[Optional[str], Optional[str]]:
    """返回 (message_id, text)。若没有 assistant 消息则 (None, None)。"""
    messages = conv.get("messages") or []
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("type") == "assistant_message":
            return m.get("id"), (m.get("text") or "")
    return None, None


def run(
    user_prompt: str,
    repository: str,
    output_dir: str = "./fresnel_cloud_output",
    ref: str = "main",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tool_rounds: int = 20,
    poll_interval: float = 8.0,
    verbose: bool = False,
) -> int:
    """
    创建 Cloud Agent，循环：等待其产出 TOOL_CALL → 本地执行工具 → 发送 TOOL_RESULT；
    若产出 ANSWER 则打印并返回 0；超轮数或失败返回 1。
    """
    os.makedirs(output_dir, exist_ok=True)
    output_dir_abs = os.path.abspath(output_dir)

    full_prompt = (
        CLOUD_AGENT_INSTRUCTIONS
        + "\n\n---\n\n"
        + build_user_prompt(output_dir_abs, user_prompt)
    )

    if verbose:
        print("[Runner] 正在创建 Cursor Cloud Agent...", file=sys.stderr)
    agent = create_agent(
        prompt_text=full_prompt,
        repository=repository,
        ref=ref,
        model=model,
        api_key=api_key,
    )
    agent_id = agent.get("id")
    if not agent_id:
        print("[Runner] 创建 Agent 失败：未返回 id", file=sys.stderr)
        return 1
    if verbose:
        print(f"[Runner] Agent 已创建: {agent_id}", file=sys.stderr)
        print(f"[Runner] 可在 Cursor 查看: https://cursor.com/agents?id={agent_id}", file=sys.stderr)

    last_processed_msg_id = None
    rounds = 0

    while rounds < max_tool_rounds:
        rounds += 1
        # 等待一段时间让云端跑完当前步
        time.sleep(poll_interval)
        conv = get_conversation(agent_id, api_key=api_key)
        msg_id, text = _last_assistant_message(conv)
        if not text:
            continue

        # 避免对同一条消息重复处理
        if msg_id == last_processed_msg_id:
            continue

        answer = _extract_answer(text)
        if answer is not None:
            print(answer)
            return 0

        tool_spec = _extract_tool_call(text)
        if not tool_spec or "tool" not in tool_spec:
            last_processed_msg_id = msg_id
            continue

        tool_name = tool_spec.get("tool")
        tool_args = dict(tool_spec.get("arguments") or {})
        for key in OUTPUT_KEYS:
            val = tool_args.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            if not os.path.isabs(str(val)):
                tool_args[key] = os.path.join(output_dir_abs, os.path.basename(str(val)))

        if verbose:
            print(f"[Runner] 执行工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]}...)", file=sys.stderr)
        result = run_tool(tool_name, tool_args)
        result_str = json.dumps(result, ensure_ascii=False, indent=2)
        followup_text = f"{TOOL_RESULT_PREFIX} {result_str}"

        followup(agent_id, followup_text, api_key=api_key)
        last_processed_msg_id = msg_id

        # 再次等待并检查是否已经产出 ANSWER（有时 followup 后很快就有最终回答）
        time.sleep(poll_interval)
        conv2 = get_conversation(agent_id, api_key=api_key)
        _, text2 = _last_assistant_message(conv2)
        if text2:
            a = _extract_answer(text2)
            if a is not None:
                print(a)
                return 0

    print("[Runner] 达到最大工具轮数，未收到 ANSWER。", file=sys.stderr)
    conv = get_conversation(agent_id, api_key=api_key)
    _, text = _last_assistant_message(conv)
    if text:
        print("[Runner] 最后一条助理消息:", file=sys.stderr)
        print(text[:2000], file=sys.stderr)
    return 1


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="用 Cursor Cloud Agent + 本地 Fresnel 工具完成多层膜设计/分析（需 CURSOR_API_KEY 与 GitHub 仓库）",
    )
    parser.add_argument("prompt", nargs="?", default="", help="用户请求")
    parser.add_argument(
        "--repository", "-r",
        help="GitHub 仓库 URL，如 https://github.com/your-org/simulation_toykits（创建 Agent 时必填；与 --list-repos 同用时可选）",
    )
    parser.add_argument(
        "--list-repos",
        action="store_true",
        help="列出当前 Cursor 可访问的 GitHub 仓库后退出；若创建 Agent 报 branch/repo 错误，可先运行此命令确认仓库是否已关联",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./fresnel_cloud_output",
        help="工具输出目录",
    )
    parser.add_argument("--ref", default="main", help="仓库分支/ref")
    parser.add_argument("--model", "-m", default="", help="可选，如 claude-4-sonnet")
    parser.add_argument("--max-rounds", type=int, default=20, help="最大工具调用轮数")
    parser.add_argument("--poll-interval", type=float, default=8.0, help="轮询间隔（秒）")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.list_repos:
        try:
            repos = list_repositories(api_key=os.environ.get("CURSOR_API_KEY"))
        except Exception as e:
            print(f"获取仓库列表失败: {e}", file=sys.stderr)
            return 1
        if not repos:
            print("当前 Cursor 未关联任何 GitHub 仓库，请在 Cursor 设置中连接 GitHub 并授权仓库。", file=sys.stderr)
            return 0
        print("Cursor 当前可访问的 GitHub 仓库（创建 Agent 时 --repository 须在此列表中）：", file=sys.stderr)
        for r in repos:
            url = r.get("repository") or f"https://github.com/{r.get('owner','')}/{r.get('name','')}"
            print(url)
        return 0

    if not args.repository:
        parser.error("创建 Agent 需要 --repository；若仅查看可访问仓库请使用 --list-repos")
        return 1

    prompt = (args.prompt or "").strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("请提供 prompt 或从 stdin 输入")
        return 1

    # 避免使用 README 占位符导致 400
    repo = args.repository.strip()
    placeholders = ("你的用户名", "your-username", "your_username", "your-org")
    if any(p in repo for p in placeholders):
        print(
            "错误：--repository 仍是占位符，请换成你的真实 GitHub 仓库地址。\n"
            "例如：https://github.com/<你的GitHub用户名>/simulation_toykits\n"
            "若默认分支不是 main，请用 --ref 指定，如：--ref master",
            file=sys.stderr,
        )
        return 1

    return run(
        user_prompt=prompt,
        repository=repo,
        output_dir=args.output_dir,
        ref=args.ref,
        model=args.model or None,
        api_key=os.environ.get("CURSOR_API_KEY"),
        max_tool_rounds=args.max_rounds,
        poll_interval=args.poll_interval,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
