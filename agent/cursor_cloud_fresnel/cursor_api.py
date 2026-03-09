"""
Cursor Cloud Agents API 客户端：创建 Agent、拉取对话、发送 Follow-up。
认证：Basic Auth，API Key 从环境变量 CURSOR_API_KEY 读取（或在调用时传入）。
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

BASE_URL = "https://api.cursor.com"


def _auth_header(api_key: str) -> str:
    # Cursor 文档：Use your API key as the username in basic authentication (leave password empty)
    raw = f"{api_key.strip()}:"
    return "Basic " + base64.b64encode(raw.encode()).decode()


def _request(
    method: str,
    path: str,
    api_key: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}{path}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": _auth_header(api_key),
    }
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = ""
        msg = f"Cursor API {method} {path}: {e.code} {e.reason}. {err_body}"
        if e.code == 400 and ("branch" in err_body.lower() or "repository" in err_body.lower()):
            msg += (
                " 若仓库与分支在 GitHub 上均正确，通常是「该仓库尚未被 Cursor 关联」：Cloud Agents 只能使用 Cursor 已授权的仓库。"
                " 请用 Cursor 打开该仓库（或到 cursor.com 设置中连接 GitHub 并授权此仓库），或运行本脚本 --list-repos 查看当前可用的仓库列表。"
            )
        raise RuntimeError(msg) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cursor API 请求失败: {e}") from e


def create_agent(
    prompt_text: str,
    repository: str,
    ref: str = "main",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    创建一个 Cloud Agent。返回包含 id, status 等的对象。
    :param prompt_text: 任务描述（会作为第一条用户消息）
    :param repository: GitHub 仓库 URL，如 https://github.com/your-org/your-repo
    :param ref: 分支/ref，默认 main
    :param model: 可选，如 claude-4-sonnet；不传则用 Cursor 默认
    :param api_key: 不传则用环境变量 CURSOR_API_KEY
    """
    key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("请设置环境变量 CURSOR_API_KEY（Cursor 网页 Dashboard → Integrations 创建）")
    payload = {
        "prompt": {"text": prompt_text},
        "source": {"repository": repository, "ref": ref},
        # 不传 model 时 Cursor 会用账户默认（如 gpt-5-high），可能对 Cloud Agents 无效；用 "default" 让 API 自动选可用模型
        "model": model if model else "default",
    }
    return _request("POST", "/v0/agents", api_key=key, body=payload, timeout=timeout)


def get_agent(agent_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """获取 Agent 状态。"""
    key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("请设置环境变量 CURSOR_API_KEY")
    return _request("GET", f"/v0/agents/{agent_id}", api_key=key)


def list_repositories(
    api_key: Optional[str] = None, timeout: float = 60.0
) -> List[Dict[str, Any]]:
    """
    获取当前 Cursor API 可访问的 GitHub 仓库列表。
    Cloud Agents 创建 Agent 时只能使用此列表中的仓库；若你的仓库不在此列表中，需先在 Cursor 中连接 GitHub 并授权该仓库。
    返回 [ {"owner", "name", "repository": "https://github.com/owner/name"}, ... ]
    """
    key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("请设置环境变量 CURSOR_API_KEY")
    data = _request("GET", "/v0/repositories", api_key=key, timeout=timeout)
    return data.get("repositories") or []


def get_conversation(
    agent_id: str, api_key: Optional[str] = None, timeout: float = 30.0
) -> Dict[str, Any]:
    """获取 Agent 对话历史。返回 { id, messages: [ { id, type, text }, ... ] }。"""
    key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("请设置环境变量 CURSOR_API_KEY")
    return _request(
        "GET", f"/v0/agents/{agent_id}/conversation", api_key=key, timeout=timeout
    )


def followup(
    agent_id: str,
    prompt_text: str,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """向已存在的 Agent 发送一条 follow-up 消息（例如 TOOL_RESULT）。"""
    key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("请设置环境变量 CURSOR_API_KEY")
    return _request(
        "POST",
        f"/v0/agents/{agent_id}/followup",
        api_key=key,
        body={"prompt": {"text": prompt_text}},
        timeout=timeout,
    )


def wait_until_idle(
    agent_id: str,
    api_key: Optional[str] = None,
    poll_interval: float = 5.0,
    max_wait: float = 600.0,
) -> str:
    """
    轮询 Agent 状态直到 RUNNING 结束（FINISHED 或 FAILED 等）。返回最终 status。
    """
    key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        data = get_agent(agent_id, api_key=key)
        status = (data.get("status") or "").upper()
        if status not in ("RUNNING", "CREATING", "PENDING"):
            return status
        time.sleep(poll_interval)
    return "TIMEOUT"
