"""
PDE 计算器共用框架：data 结构统一、刷新坐标轴逻辑、表格式坐标轴 UI。
sl_web_factory 统一 SL / TDSL 的 UI 交互，页面仅绑定公式/计算/结果回调。
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from common import render_table_editor, PDE_CACHE_KEY_TO_LABEL



def get_ref_shape_from_data(data: dict[str, Any] | None) -> tuple[int, ...] | None:
    """从 data（load_mat_v7 或 cache_pool/beams 选出的 dict）中取参考形状（任一数组）。
    返回的 shape 与坐标轴顺序一致：axes_config[i].n_points == shape[i]，最后一维变化最快（C 序）。
    """
    if not data:
        return None
    for key, val in data.items():
        if hasattr(val, "shape"):
            return tuple(int(x) for x in val.shape)
    return None


BC_OPTIONS = ["periodic", "dirichlet", "neumann"]


def default_axis(dim_index: int) -> dict:
    """统一坐标轴默认配置。"""
    return {
        "n_points": 32,
        "length": 1.0,
        "bc_from": "dirichlet",
        "bc_to": "dirichlet",
        "p": 1.0,
        "q": 0.0,
    }


# ---------------------------------------------------------------------------
# cache_pool (key -> array), sl_web_factory with callbacks
# ---------------------------------------------------------------------------

CACHE_POOL_KEY = "pde_cache_pool"


def get_cache_pool(session_state) -> dict:
    """Return the cache_pool dict (key -> numpy array)."""
    if CACHE_POOL_KEY not in session_state:
        session_state[CACHE_POOL_KEY] = {}
    return session_state[CACHE_POOL_KEY]


def get_cache_pool_keys(session_state) -> list:
    """Return sorted list of keys in cache_pool for dropdown options."""
    pool = get_cache_pool(session_state)
    return sorted(pool.keys())


def get_cache_options(session_state) -> list[dict]:
    """
    Return dropdown options: cache_pool (key -> label=key) + beams_caculator results.
    Each item is {"key": str, "label": str}. Key is used in session state; label is shown in dropdown.
    """
    import numpy as np
    out = []
    pool = get_cache_pool(session_state)
    for k in sorted(pool.keys()):
        out.append({"key": k, "label": k})
    beams = session_state.get("beams_result_cache") or {}
    for page_key, entry in beams.items():
        if not isinstance(entry, dict) or "field" not in entry:
            continue
        arr = entry["field"]
        if not hasattr(arr, "shape") or arr.ndim < 1 or arr.ndim > 2:
            continue
        label = PDE_CACHE_KEY_TO_LABEL.get(page_key, f"{page_key} (场)")
        out.append({"key": f"beams__{page_key}__field", "label": f"Beams / {label}"})
    return out


def get_array_by_key(session_state, key: str):
    """Resolve option key to numpy array. Key is from cache_pool or beams__page_key__field."""
    import numpy as np
    pool = get_cache_pool(session_state)
    if key in pool:
        return np.asarray(pool[key])
    if key.startswith("beams__") and key.endswith("__field"):
        # key is "beams__flat top beam__field"
        mid = key[:-7]  # strip "__field"
        if mid.startswith("beams__"):
            page_key = mid[7:]  # "flat top beam"
            beams = session_state.get("beams_result_cache") or {}
            entry = beams.get(page_key)
            if isinstance(entry, dict) and "field" in entry:
                return np.asarray(entry["field"])
    return None


def merge_mat_into_cache_pool(session_state, data: dict[str, Any]) -> None:
    """Merge loaded .mat data (key -> array) into cache_pool."""
    import numpy as np
    pool = get_cache_pool(session_state)
    for key, val in data.items():
        arr = np.asarray(val)
        if hasattr(arr, "shape") and arr.ndim >= 1 and arr.size > 0:
            pool[f"Uploaded/{key}"] = arr.copy()


def put_result_into_cache_pool(session_state, result_array: Any, label_name: str) -> None:
    """Put SL result array into cache_pool with given key."""
    import numpy as np
    pool = get_cache_pool(session_state)
    pool[label_name] = np.asarray(result_array).copy()


class sl_web_factory:
    """
    PDE 计算器统一框架：cache_pool 存储 key -> array；
    通过回调注册 SL / TDSL 不同的逻辑（选择输入、公式、计算与结果）。
    """

    def __init__(
        self,
        key_prefix: str,
        get_effective_data_cb: Callable[[str], dict[str, Any] | None],
    ):
        self.key_prefix = key_prefix
        self._get_effective_data_cb = get_effective_data_cb
        self._axes_key = f"{key_prefix}_axes"
        self._axes_ver_key = f"{key_prefix}_axes_ver"
        self._result_key = f"{key_prefix}_result"
        self._pending_del_key = f"{key_prefix}_pending_delete"
        self._upload_key = f"{key_prefix}_upload"

    def _ensure_session_state(self) -> None:
        if self._axes_key not in st.session_state:
            st.session_state[self._axes_key] = []
        if self._result_key not in st.session_state:
            st.session_state[self._result_key] = None
        if self._pending_del_key not in st.session_state:
            st.session_state[self._pending_del_key] = None
        get_cache_pool(st.session_state)
        for k in (
            f"{self.key_prefix}_input_key",
            f"{self.key_prefix}_cache_role",
            f"{self.key_prefix}_analytical_key",
            f"{self.key_prefix}_u0_key",
            f"{self.key_prefix}_ut0_key",
        ):
            if k not in st.session_state:
                # cache_role 存 index（0=f, 1=u），与 UI 文案解耦
                st.session_state[k] = 0 if k.endswith("_cache_role") else ""

    def run_start(self) -> None:
        self._ensure_session_state()
        idx_del = st.session_state.get(self._pending_del_key)
        if idx_del is not None:
            axes_list = st.session_state.get(self._axes_key)
            if isinstance(axes_list, list) and 0 <= idx_del < len(axes_list):
                axes_list.pop(idx_del)
            st.session_state[self._pending_del_key] = None
            st.rerun()

    def init_matlab_upload(self, load_mat_fn: Callable[[bytes], dict]) -> None:
        """上传 MATLAB 数据：读取 .mat 后将每个 key 与 numpy array 存入 cache_pool。"""
        st.subheader("上传 MATLAB 数据")
        uploaded = st.file_uploader("选择 .mat 文件", type=["mat"], key=self._upload_key)
        if uploaded is not None:
            try:
                data = load_mat_fn(uploaded.getvalue())
                merge_mat_into_cache_pool(st.session_state, data)
                ref_shape = get_ref_shape_from_data(data)
                st.success(f"已加载数据，维度: {ref_shape}, 变量名: {', '.join(sorted(data.keys()))}。" if ref_shape else "已加载数据。")
            except Exception as e:
                st.error(f"文件读取失败: {e}")

    def init_cache_section(
        self,
        render_cb: Callable[[str, list, list, list, list], None],
    ) -> None:
        """选择输入：由 render_cb(key_prefix, options, keys, labels, choices) 渲染 UI 并同步 session state。"""
        st.subheader("设置输入场")
        options = get_cache_options(st.session_state)
        keys = [o["key"] for o in options]
        labels = [o["label"] for o in options]
        choices = labels if options else ["（暂无缓存，请先上传 .mat 或使用 Beams 计算器）"]
        render_cb(self.key_prefix, options, keys, labels, choices)

    def get_effective_data(self) -> dict[str, Any] | None:
        """当前有效 data：由 get_effective_data_cb(key_prefix) 按选择从 session state 拼出 load_mat_v7 同结构。"""
        return self._get_effective_data_cb(self.key_prefix)

    def _get_ref_shape_for_refresh(self) -> tuple[int, ...] | None:
        data = self.get_effective_data()
        return get_ref_shape_from_data(data)

    def refresh_axes_if_needed(self) -> bool:
        """若当前 axes 维数与有效数据维数不一致，则按数据维度刷新坐标轴并 rerun。返回 True 表示已 rerun，调用方应 return。"""
        data = self.get_effective_data()
        ref_shape = get_ref_shape_from_data(data)
        axes = st.session_state.get(self._axes_key) or []
        if ref_shape is not None and len(axes) != len(ref_shape):
            new_axes = [dict(default_axis(i)) for i in range(len(ref_shape))]
            for i, ax in enumerate(new_axes):
                ax["n_points"] = int(ref_shape[i])
            st.session_state[self._axes_key] = new_axes
            st.session_state[self._axes_ver_key] = st.session_state.get(self._axes_ver_key, 0) + 1
            st.rerun()
            return True
        return False

    def init_axes_config(self) -> None:
        """坐标轴配置：刷新坐标轴 / 添加坐标轴 / 清空。刷新时按当前 input 或 u0 维度先清空再创建。"""
        st.subheader("设置算子 L")
        axes = st.session_state[self._axes_key]
        ver = st.session_state.get(self._axes_ver_key, 0)
        p = self.key_prefix
        v = ver

        def update_bc(idx: int, side: str) -> None:
            ax_list = st.session_state[self._axes_key]
            kf = f"{p}_bcf_{v}_{idx}"
            kt = f"{p}_bct_{v}_{idx}"
            vf = st.session_state.get(kf)
            vt = st.session_state.get(kt)
            if side == "from" and vf == "periodic":
                st.session_state[kt] = "periodic"
                ax_list[idx]["bc_from"] = ax_list[idx]["bc_to"] = "periodic"
            elif side == "to" and vt == "periodic":
                st.session_state[kf] = "periodic"
                ax_list[idx]["bc_from"] = ax_list[idx]["bc_to"] = "periodic"
            else:
                ax_list[idx]["bc_from"] = vf
                ax_list[idx]["bc_to"] = vt

        def set_pending_del(index: int) -> None:
            st.session_state[self._pending_del_key] = index
            st.rerun()

        def render_row(i: int, item: dict, cols: list) -> None:
            with cols[0]:
                item["n_points"] = int(st.number_input("网格点数", min_value=2, value=item["n_points"], key=f"{p}_n_{v}_{i}", label_visibility="collapsed"))
            with cols[1]:
                item["length"] = float(st.number_input("域长", value=float(item["length"]), format="%.4f", key=f"{p}_L_{v}_{i}", label_visibility="collapsed"))
            with cols[2]:
                st.selectbox("起点边界", BC_OPTIONS, index=BC_OPTIONS.index(item["bc_from"]), key=f"{p}_bcf_{v}_{i}", on_change=update_bc, args=(i, "from"), label_visibility="collapsed")
            with cols[3]:
                st.selectbox("终点边界", BC_OPTIONS, index=BC_OPTIONS.index(item["bc_to"]), key=f"{p}_bct_{v}_{i}", on_change=update_bc, args=(i, "to"), label_visibility="collapsed")
            with cols[4]:
                item["p"] = float(st.number_input("p", value=float(item["p"]), format="%.4f", key=f"{p}_p_{v}_{i}", label_visibility="collapsed"))
            with cols[5]:
                item["q"] = float(st.number_input("q", value=float(item["q"]), format="%.4f", key=f"{p}_q_{v}_{i}", label_visibility="collapsed"))

        def on_refresh_axes() -> None:
            ref_shape = self._get_ref_shape_for_refresh()
            if ref_shape is not None:
                new_axes = [default_axis(i).copy() for i in range(len(ref_shape))]
                for i, ax in enumerate(new_axes):
                    ax["n_points"] = int(ref_shape[i])
                st.session_state[self._axes_key] = new_axes
                st.session_state[self._axes_ver_key] = st.session_state.get(self._axes_ver_key, 0) + 1
                st.rerun()

        def on_add() -> None:
            axes_list = st.session_state[self._axes_key]
            axes_list.append(default_axis(len(axes_list)))
            st.rerun()

        def on_clear() -> None:
            st.session_state[self._axes_key] = []
            st.rerun()

        def on_delete(index: int) -> None:
            set_pending_del(index)

        render_table_editor(
            key_prefix=f"{p}_axes_{v}",
            columns=[
                {"label": "网格点数", "width": 1},
                {"label": "域长", "width": 1},
                {"label": "起点边界", "width": 1.2},
                {"label": "终点边界", "width": 1.2},
                {"label": "p", "width": 0.8},
                {"label": "q", "width": 0.8},
            ],
            items=axes,
            render_row=render_row,
            on_add=on_add,
            on_clear=on_clear,
            on_delete=on_delete,
            add_label="➕ 添加维度",
            clear_label="🗑️ 清空",
            delete_label="删除",
            left_buttons=[{"label": "🔄 刷新", "key": f"{p}_refresh_axes", "on_click": on_refresh_axes}],
        )

    def init_formula_preview(self, formula_md_fn: Callable[[list, dict | None], str]) -> None:
        """公式预览：坐标轴配置改变时随 rerun 自动刷新。"""
        axes = st.session_state.get(self._axes_key) or []
        data = self.get_effective_data()
        st.subheader("公式预览")
        st.markdown(formula_md_fn(axes, data))

    def init_compute_and_result(
        self,
        compute_button_label: str,
        compute_key: str,
        on_compute: Callable[[], None],
        render_result_section: Callable[[Any], None],
        empty_caption: str,
    ) -> None:
        """计算按钮与结果展示。"""
        if st.button(compute_button_label, type="primary", width="stretch", key=compute_key):
            on_compute()
        res = st.session_state.get(self._result_key)
        if res:
            render_result_section(res)
        else:
            st.caption(empty_caption)
