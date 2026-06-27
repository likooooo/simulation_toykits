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


def object_catalog_name(obj: Any) -> str:
    return str(obj.catalog_name())


def object_unique_name(obj: Any) -> str:
    return str(obj.unique_name())


def material_variant_label(obj: Any) -> str:
    catalog = object_catalog_name(obj)
    unique = object_unique_name(obj)
    prefix = catalog + "("
    if unique.startswith(prefix) and unique.endswith(")"):
        return unique[len(prefix) : -1]
    return unique


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


def search_material_paths(sim_db: Any, material_name: str) -> list[list[str]]:
    """Search material leaves; ``materials`` in path_keys is a navigation scope filter, not YAML leaf typing."""

    def _is_material_path(path_keys: list[str]) -> bool:
        return "materials" in path_keys

    return _iter_matching_leaf_paths(
        sim_db,
        material_name,
        match_blob=breadcrumb_for,
        path_filter=_is_material_path,
    )


def path_id(path_keys: list[str]) -> str:
    return "/".join(path_keys)


def breadcrumb_for(path_keys: list[str]) -> str:
    return " > ".join(path_keys)


def leaf_type_for_path(
    sim_db: Any,
    path_keys: list[str],
    *,
    kind_cache: dict[str, str] | None = None,
) -> str | None:
    """Classify a yml leaf from on-disk YAML (no C++ material/spectrum object load)."""
    cache = kind_cache if kind_cache is not None else {}
    pid = path_id(path_keys)
    if pid in cache:
        return cache[pid]
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
    matches: list[dict[str, Any]] = []
    kind_cache: dict[str, str] = {}
    for path_keys in _iter_matching_leaf_paths(sim_db, query, max_results=max_results):
        leaf_type = leaf_type_for_path(sim_db, path_keys, kind_cache=kind_cache)
        if leaf_type is None:
            continue
        if leaf_type == "spectrum":
            label = path_keys[-1] if path_keys else "spectrum"
        else:
            label = simulation.material_unique_name_from_path_keys(path_keys)
        matches.append(
            {
                "path_keys": path_keys,
                "leaf_type": leaf_type,
                "label": label,
                "path_id": path_id(path_keys),
            }
        )
    return matches


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
