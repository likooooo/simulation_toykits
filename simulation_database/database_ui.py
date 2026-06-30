"""Simulation database helpers for Streamlit browser."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np

import simulation  # noqa: F401 — must load before simulation_database_parser
import simulation_database_parser as sdp

__all__ = [
    "material_nk_arrays",
    "search_material_paths",
]


def object_unique_name(obj: Any) -> str:
    return str(obj.unique_name())


def _wl_range_from_array(wl: np.ndarray) -> tuple[float, float] | None:
    if len(wl) == 0:
        return None
    return float(np.min(wl)), float(np.max(wl))


def _entry_wl_range_um(kind: Literal["material", "spectrum"], obj: Any) -> tuple[float, float] | None:
    try:
        if kind == "material":
            wl, _, _ = material_nk_arrays(obj)
        else:
            wl, _ = spectrum_arrays(obj)
    except Exception:
        return None
    return _wl_range_from_array(wl)


def material_wl_range_um(mat: Any) -> tuple[float, float] | None:
    return _entry_wl_range_um("material", mat)


def spectrum_wl_range_um(spec: Any) -> tuple[float, float] | None:
    return _entry_wl_range_um("spectrum", spec)


def intersect_wl_ranges(ranges: list[tuple[float, float] | None]) -> tuple[float, float] | None:
    valid = [r for r in ranges if r is not None]
    if not valid:
        return None
    lo = max(r[0] for r in valid)
    hi = min(r[1] for r in valid)
    return lo, hi


def material_nk_arrays(mat: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wl, n, k = mat.get_tabulated_values()
    return np.asarray(wl, dtype=float), np.asarray(n, dtype=float), np.asarray(k, dtype=float)


def _iter_query_nodes(
    query: Any,
    prefix: list[str],
) -> Iterable[tuple[Any, list[str], bool]]:
    """Depth-first yield (query_node, path_keys, is_leaf) for every tree node."""
    if query.is_leaf:
        yield query, prefix, True
        return
    for key in query.keys:
        try:
            is_leaf, child = query.descend(key)
        except Exception:
            continue
        child_path = [*prefix, key]
        yield child, child_path, is_leaf
        if not is_leaf:
            yield from _iter_query_nodes(child, child_path)


def _iter_matching_leaf_paths(
    sim_db: Any,
    query: str,
    *,
    max_results: int | None = None,
    match_blob: Callable[[list[str]], str] | None = None,
    path_filter: Callable[[list[str]], bool] | None = None,
) -> list[list[str]]:
    key = (query or "").strip()
    if not key:
        return []
    pattern = re.compile(re.escape(key), re.IGNORECASE)
    matches: list[list[str]] = []

    for _node, path_keys, is_leaf in _iter_query_nodes(sim_db.query(), []):
        if max_results is not None and len(matches) >= max_results:
            break
        if not is_leaf:
            continue
        if path_filter is not None and not path_filter(path_keys):
            continue
        if match_blob is not None:
            blob = match_blob(path_keys)
        else:
            blob = breadcrumb_for(path_keys)
        if pattern.search(blob) or pattern.search(path_keys[-1] if path_keys else ""):
            matches.append(list(path_keys))
    return matches


def _search_result_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "path_keys": list(entry["path_keys"]),
        "path_id": entry["path_id"],
        "leaf_type": entry["leaf_type"],
        "label": entry["label"],
    }


def _search_precompiled_entries(
    sim_db: Any,
    query: str,
    *,
    max_results: int | None = None,
    material_only: bool = False,
) -> list[dict[str, Any]] | None:
    from simulation_database.database_precompiling import candidate_entry_indices

    precompiled = _active_precompiled_index(sim_db)
    if precompiled is None:
        return None
    key = (query or "").strip()
    if not key:
        return []
    q_lower = key.lower()
    candidate_indices = candidate_entry_indices(
        precompiled.inverted_index,
        key,
        entry_count=len(precompiled.entries),
    )
    if candidate_indices is None:
        candidate_indices = list(range(len(precompiled.entries)))
    matches: list[dict[str, Any]] = []
    for index in candidate_indices:
        entry = precompiled.entries[index]
        if material_only and not entry.get("is_material_path"):
            continue
        blob = entry.get("search_blob_lower", "")
        if q_lower not in blob:
            continue
        matches.append(_search_result_from_entry(entry))
        if max_results is not None and len(matches) >= max_results:
            break
    return matches


def search_material_paths(sim_db: Any, material_name: str) -> list[list[str]]:
    """Search material leaves; ``materials`` in path_keys is a navigation scope filter, not YAML leaf typing."""
    cached = _search_precompiled_entries(sim_db, material_name, material_only=True)
    return [list(entry["path_keys"]) for entry in (cached or [])]


def path_id(path_keys: list[str]) -> str:
    return "/".join(path_keys)


def breadcrumb_for(path_keys: list[str]) -> str:
    return " > ".join(path_keys)


def _active_precompiled_index(sim_db: Any) -> Any | None:
    from simulation_database.database_precompiling import PrecompiledIndex, get_active_index

    index = get_active_index()
    if index is None or not isinstance(index, PrecompiledIndex):
        return None
    return index


def leaf_type_for_path(
    sim_db: Any,
    path_keys: list[str],
    *,
    kind_cache: dict[str, str] | None = None,
) -> str | None:
    """Classify a leaf from the precompiled index (build-time may read YAML)."""
    cache = kind_cache if kind_cache is not None else {}
    pid = path_id(path_keys)
    if pid in cache:
        return cache[pid]
    precompiled = _active_precompiled_index(sim_db)
    if precompiled is not None:
        kind = precompiled.kind_by_path_id.get(pid)
        if kind is not None:
            cache[pid] = kind
            return kind
    try:
        leaf = sim_db.walk_query_path(path_keys)
        kind = sdp.infer_yml_leaf_kind(leaf.storage_path())
    except Exception:
        return None
    if kind not in ("material", "spectrum"):
        return None
    cache[pid] = kind
    return kind


def infer_leaf_kind(obj: Any) -> Literal["material", "spectrum"]:
    if obj.is_material():
        return "material"
    if obj.is_spectrum():
        return "spectrum"
    raise ValueError("object is neither material nor spectrum")


def get_tree_children(
    sim_db: Any,
    path_keys: list[str],
    cache: dict[str, list[dict[str, Any]]],
    *,
    kind_cache: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    pid = path_id(path_keys)
    if pid in cache:
        return cache[pid]
    precompiled = _active_precompiled_index(sim_db)
    if precompiled is not None:
        if pid in precompiled.children_by_path_id:
            children = [dict(child) for child in precompiled.children_by_path_id[pid]]
            cache[pid] = children
            return children
        return []
    if not path_keys:
        query = sim_db.query()
    else:
        query = sim_db.walk_query_path(path_keys, require_leaf=False)
    children: list[dict[str, Any]] = []
    for key in query.keys:
        try:
            is_leaf, child = query.descend(key)
        except Exception:
            continue
        child_path = [*path_keys, key]
        children.append(
            {
                "key": key,
                "path_id": path_id(child_path),
                "path_keys": child_path,
                "is_leaf": is_leaf,
                "leaf_type": leaf_type_for_path(sim_db, child_path, kind_cache=kind_cache)
                if is_leaf
                else None,
                "child_count": len(child.keys) if not is_leaf else 0,
            }
        )
    cache[pid] = children
    return children


def build_tree_nodes_for_panel(
    sim_db: Any,
    expanded_paths: set[str],
    children_cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    kind_cache: dict[str, str] = {}
    return _build_subtree(sim_db, [], expanded_paths, children_cache, kind_cache)


def _build_subtree(
    sim_db: Any,
    path_keys: list[str],
    expanded_paths: set[str],
    children_cache: dict[str, list[dict[str, Any]]],
    kind_cache: dict[str, str],
) -> list[dict[str, Any]]:
    children_meta = get_tree_children(sim_db, path_keys, children_cache, kind_cache=kind_cache)
    result: list[dict[str, Any]] = []
    for meta in children_meta:
        node = dict(meta)
        node["children"] = []
        if not meta["is_leaf"] and meta["path_id"] in expanded_paths:
            node["children"] = _build_subtree(
                sim_db, meta["path_keys"], expanded_paths, children_cache, kind_cache
            )
        result.append(node)
    return result


def search_db_paths(sim_db: Any, query: str, max_results: int = 80) -> list[dict[str, Any]]:
    cached = _search_precompiled_entries(sim_db, query, max_results=max_results)
    return cached or []


def spectrum_arrays(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    wl_um, value = spec.get_tabulated_values()
    return np.asarray(wl_um, dtype=float), np.asarray(value, dtype=float)


def dump_object_as_csv(obj: Any) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as tmp:
        obj.dump(tmp)
        csv_files = sorted(Path(tmp).glob("*.csv"))
        if not csv_files:
            raise ValueError("dump produced no csv file")
        csv_path = csv_files[0]
        safe = re.sub(r"[^\w.\-]+", "_", object_unique_name(obj)).strip("_") or "export"
        filename = f"{safe}.csv"
        return csv_path.read_bytes(), filename
