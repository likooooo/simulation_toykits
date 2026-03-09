"""
Ollama 大脑：调用本地 Ollama 模型，按约定输出“工具调用”或“最终回答”，驱动 Fresnel 专家工作流。
工具调用格式：模型输出一个 JSON 对象，包含 "tool" 与 "arguments"，或 "answer"/"text" 表示结束。
"""

import json
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

TOOL_SCHEMAS = [
    {
        "name": "list_material_index",
        "description": "按材料名在索引中查询，返回 shelf_id、一组 page_id。",
        "arguments": {
            "material_name": "材料名称，必填，如 SiO2、TiO2，区分大小写查询",
            "csv_path": "可选，材料索引 CSV 路径，不传则用默认路径",
        },
    },
    {
        "name": "get_material_nk",
        "description": "获取指定材料的 n/k 数据（波长、折射率、消光系数）。可选 ratio 控制采样精度。",
        "arguments": {
            "shelf_id": "Shelf ID",
            "book_id": "Book ID（材料名）",
            "page_id": "Page ID",
            "ratio": "可选，采样精度 0 < ratio <= 1，小模型推荐0.1",
        },
    },
    {
        "name": "export_nk_to_csv",
        "description": "将材料 nk 数据导出为 CSV 文件。",
        "arguments": {"shelf_id": "Shelf ID", "book_id": "Book ID", "page_id": "Page ID", "out_path": "输出文件路径"},
    },
    {
        "name": "parse_film_formula",
        "description": "解析多层膜公式，得到层列表。支持 (Material thickness)^N 和 Material thickness [n k]。",
        "arguments": {"formula": "公式字符串，如 Vacuum 0 (SiO2 0.1 Ta2O5 0.05)^10 Vacuum 0"},
    },
    {
        "name": "compute_filmstack",
        "description": "计算单组膜系在指定波长和角度下的 R/T 及膜系图。可保存膜系结构图。",
        "arguments": {
            "formula": "膜系公式",
            "angle_deg": "入射角（度）",
            "wl_um": "波长（微米）",
            "materials_db": "可选，材料名到 {Shelf ID, Book ID, Page ID} 的映射，用于从数据库取 nk",
            "out_figure_path": "可选，膜系图保存路径",
        },
    },
    {
        "name": "compute_filmstack_batch",
        "description": "批量计算多组膜系配置的 R/T，用于设计时生成多组候选并比较。",
        "arguments": {
            "formulas": "膜系公式列表",
            "angle_deg": "入射角（度）",
            "wl_um": "波长（微米）",
            "materials_db": "可选，材料库映射",
        },
    },
    {
        "name": "compute_angle_vs_rt",
        "description": "固定波长，计算 R/T 随角度的变化，可保存一张 R/T-角度 曲线图。",
        "arguments": {
            "formula": "膜系公式",
            "wl_um": "波长（微米）",
            "materials_db": "可选，材料名到 {Shelf ID, Book ID, Page ID} 的映射",
            "out_figure_path": "可选，曲线图保存路径（与 out_figure_rt_path 二选一）",
            "out_figure_rt_path": "可选，曲线图保存路径（同上）",
            "out_figure_nk_path": "可选，同上，本工具只产出一张图",
        },
    },
    {
        "name": "compute_wavelength_vs_rt",
        "description": "固定角度，计算 R/T 随波长的变化，可保存 R/T 与 n-k 曲线图。",
        "arguments": {
            "formula": "膜系公式",
            "angle_deg": "入射角（度）",
            "wl_min_um": "起始波长（微米）",
            "wl_max_um": "截止波长（微米）",
            "num_points": "可选，默认 100",
            "materials_db": "可选",
            "out_figure_rt_path": "可选，R/T 图路径",
            "out_figure_nk_path": "可选，n-k 图路径",
        },
    },
    {
        "name": "save_results_csv",
        "description": "将结果列表（如多组膜系及其 R/T）保存为 CSV 文件。",
        "arguments": {"rows": "字典列表，每行一条", "out_path": "输出 CSV 路径"},
    },
]

SYSTEM_PROMPT = """你由Simulation-toykits部署, 一个专门用于设计、分析多层光学薄膜的专家系统。你可以使用以下工具完成用户请求。

可用工具（每次只能调用一个）：
- list_material_index: 列出材料索引，查 shelf/book/page
- get_material_nk: 获取某材料的 n/k 数据
- export_nk_to_csv: 将材料 nk 导出为 CSV
- parse_film_formula: 解析膜系公式得到层列表
- compute_filmstack: 单组膜系在单波长单角度下计算 R/T，可保存膜系图
- compute_filmstack_batch: 批量计算多组膜系的 R/T，用于设计时比较
- compute_angle_vs_rt: 固定波长，R/T 随角度变化曲线
- compute_wavelength_vs_rt: 固定角度，R/T 随波长变化曲线
- save_results_csv: 将结果保存为 CSV

🚫 **严禁伪造执行结果！**
在你没有调用工具（如 export_nk_to_csv、save_results_csv 等）并收到系统返回的「执行成功」或工具结果前，**绝对不许**在回答中说文件已导出或任务已完成。
若需要保存文件，你必须先调用相应工具，等待系统返回结果后，再根据结果回复用户。

回复规则：
1. 每次回复可先写思考过程（以「思考：」开头），再输出 JSON。
2. 若需要调用工具：输出一个 JSON，包含 "tool" 和 "arguments"。例如：
   思考：用户要导出 SiO2 数据，需先查材料索引得到 shelf/book/page，再调用 export_nk_to_csv。
   {"tool": "export_nk_to_csv", "arguments": {"shelf_id": "...", "book_id": "SiO2", "page_id": "...", "out_path": "/path/to/out.csv"}}
3. 若不再需要调用工具、要直接回答用户：输出包含 "answer" 或 "text" 的 JSON。**只有在工具已返回成功后再声称文件已导出。** 例如：
   {"answer": "TiO2 的数据已成功为您导出至 2.csv。"}

Few-shot 示例（请模仿）：
User: 导出 TiO2 的数据到 2.csv
Assistant: {"tool": "export_nk_to_csv", "arguments": {"shelf_id": "main", "book_id": "TiO2", "page_id": "Malitson", "out_path": "2.csv"}}
User: 工具执行结果：{"success": true, "path": "/abs/2.csv", "rows": 100}
Assistant: {"answer": "TiO2 的数据已成功为您导出至 2.csv。"}

4. 设计多层膜时：先 list_material_index / get_material_nk 了解材料，再 compute_filmstack_batch 得到 R/T，根据指标挑选结果，最后用 save_results_csv 或 export 类工具输出文件。
5. **迭代设计**（用户要求「迭代设计」或「逐轮优化」时）：(a) 生成初始膜系公式；(b) 调用分析工具（compute_filmstack / compute_wavelength_vs_rt / compute_angle_vs_rt）；(c) 根据结果与指标判断是否满足，若不满足则生成新公式回到 (b)；满足则 answer 总结并用工具输出。
6. 波长单位均为微米(μm)，角度为度(deg)。材料名与材料库 Book ID 一致（如 SiO2, TiO2）。Vacuum 表示空气/真空，厚度写 0。
"""


def _to_ollama_tools(schemas: List[Dict]) -> List[Dict]:
    """将 TOOL_SCHEMAS 转为 Ollama /api/chat 所需的 tools 数组（OpenAI 兼容格式）。"""
    tools = []
    for s in schemas:
        name = s["name"]
        desc = s["description"]
        args = s.get("arguments") or {}
        properties = {}
        required = []
        for k, v in args.items():
            properties[k] = {"type": "string", "description": str(v)}
            if "可选" not in str(v):
                required.append(k)
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools


OLLAMA_TOOLS = _to_ollama_tools(TOOL_SCHEMAS)


def _ollama_chat(
    model: str,
    messages: List[Dict[str, Any]],
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
    tools: Optional[List[Dict]] = None,
    think: bool = False,
) -> tuple[str, Dict[str, Any]]:
    """调用 Ollama /api/chat。若传 tools 则使用原生 Tool Calling；think=True 时返回 message 中的思考过程。返回 (content, message)。"""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    if think:
        payload["think"] = True
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama 请求失败（请确认 Ollama 已启动且模型已拉取）: {e}") from e
    msg = data.get("message") or {}
    content = (msg.get("content") or "").strip()
    return content, msg


def _parse_tool_or_answer(content: str) -> Dict[str, Any]:
    """从模型输出中解析出 JSON：要么 tool+arguments，要么 answer/text。"""
    content = content.strip()
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    if json_match:
        raw = json_match.group(1)
    else:
        raw = content
    start = raw.rfind("{")
    if start == -1:
        return {"parse_error": content}
    depth = 0
    end = -1
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return {"parse_error": raw}
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        return {"parse_error": raw[start:end], "json_error": str(e)}
    return obj


def chat_turn(
    model: str,
    messages: List[Dict[str, Any]],
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
    use_native_tools: bool = True,
    think: bool = False,
) -> tuple[str, Dict[str, Any], Dict[str, Any], str]:
    """
    执行一轮对话，返回 (assistant_content, parsed, assistant_message, thinking)。
    parsed 若含 "tool" 和 "arguments" 则为工具调用；若含 "answer" 或 "text" 则为最终回答。
    assistant_message 用于追加到 messages（原生 tool 时含 tool_calls，否则为 {role, content}）。
    thinking 为 Ollama think 模式下的思考过程（think=False 时为空字符串）。
    当 use_native_tools=True 时使用 Ollama 原生 tools API，提高工具调用成功率。
    """
    content, msg = _ollama_chat(
        model,
        messages,
        base_url=base_url,
        timeout=timeout,
        tools=OLLAMA_TOOLS if use_native_tools else None,
        think=think,
    )
    thinking = (msg.get("thinking") or "").strip()
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args if isinstance(raw_args, dict) else {}
        if name:
            # 返回完整 assistant 消息（含 tool_calls），供 run_agent 追加后发 tool 结果
            out_msg = {"role": "assistant", "content": content or ""}
            if tool_calls:
                out_msg["tool_calls"] = tool_calls
            return content, {"tool": name, "arguments": args}, out_msg, thinking
    parsed = _parse_tool_or_answer(content)
    return content, parsed, {"role": "assistant", "content": content}, thinking
