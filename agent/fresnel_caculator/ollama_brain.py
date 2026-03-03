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
        "description": "列出材料数据库索引（shelf/book/page），用于查找材料。",
        "arguments": {"csv_path": "可选，材料索引 CSV 路径，不传则用默认路径"},
    },
    {
        "name": "get_material_nk",
        "description": "获取指定材料的 n/k 数据（波长、折射率、消光系数）。",
        "arguments": {"shelf_id": "Shelf ID", "book_id": "Book ID（材料名）", "page_id": "Page ID"},
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
        "description": "固定波长，计算 R/T 随角度的变化，可保存曲线图。",
        "arguments": {
            "formula": "膜系公式",
            "wl_um": "波长（微米）",
            "materials_db": "可选",
            "out_figure_path": "可选，曲线图保存路径",
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

SYSTEM_PROMPT = """你是一个专门用于设计、分析多层光学薄膜的专家系统。你可以使用以下工具完成用户请求。

可用工具（每次只能调用一个，用 JSON 格式回复）：
- list_material_index: 列出材料索引，查 shelf/book/page
- get_material_nk: 获取某材料的 n/k 数据
- export_nk_to_csv: 将材料 nk 导出为 CSV
- parse_film_formula: 解析膜系公式得到层列表
- compute_filmstack: 单组膜系在单波长单角度下计算 R/T，可保存膜系图
- compute_filmstack_batch: 批量计算多组膜系的 R/T，用于设计时比较
- compute_angle_vs_rt: 固定波长，R/T 随角度变化曲线
- compute_wavelength_vs_rt: 固定角度，R/T 随波长变化曲线
- save_results_csv: 将结果保存为 CSV

回复规则：
1. 若需要调用工具，必须只输出一个 JSON 对象，且包含 "tool" 和 "arguments" 两个字段，不要输出其他解释。例如：
   {"tool": "compute_filmstack", "arguments": {"formula": "Vacuum 0 SiO2 0.1 Vacuum 0", "angle_deg": 0, "wl_um": 0.532}}
2. 若不再需要调用工具、要直接回答用户，则输出包含 "answer" 或 "text" 的 JSON，例如：
   {"answer": "根据计算，该膜系在 532nm 正入射下 R_s=0.95，满足高反要求。建议将结果保存为 CSV。"}
3. 设计多层膜时：先根据用户需求生成多组候选膜系公式（可先用 list_material_index / get_material_nk 了解材料），再调用 compute_filmstack_batch 得到每组 R/T，根据用户指标（如 R>99%、T 最大等）挑选最佳结果，最后用 save_results_csv 或 export 类工具输出文件。
4. **迭代设计**（用户明确要求“迭代设计”或“逐轮优化”时）：(a) 根据用户需求生成**初始膜系公式**；(b) 根据用户指标选择合适的**分析工具**并调用——单波长单角度用 compute_filmstack，宽带/光谱用 compute_wavelength_vs_rt，角度特性用 compute_angle_vs_rt；(c) 根据工具返回的**分析结果**与用户指标判断是否已满足；若未满足，则根据当前结果与用户要求**生成新的膜系公式**（调整材料、厚度或周期数等），再回到 (b)；若已满足或轮数足够，则用 answer 总结最终膜系与结果，并用 save_results_csv 等输出。每轮只调用一个分析工具，根据结果再决定下一轮的公式。
5. 所有波长单位均为微米(μm)，角度为度(deg)。材料名需与材料库中 Book ID 一致（如 SiO2 (Silicon dioxide, Silica, Quartz)）。Vacuum 表示空气/真空，厚度写 0。
"""


def _ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> str:
    """调用 Ollama /api/chat，返回 assistant 的 content 文本（非流式）。"""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama 请求失败（请确认 Ollama 已启动且模型已拉取）: {e}") from e
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


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
) -> tuple[str, Dict[str, Any]]:
    """
    执行一轮对话，返回 (assistant_content, parsed)。
    parsed 若含 "tool" 和 "arguments" 则为工具调用；若含 "answer" 或 "text" 则为最终回答。
    """
    content = _ollama_chat(model, messages, base_url=base_url, timeout=timeout)
    parsed = _parse_tool_or_answer(content)
    return content, parsed
