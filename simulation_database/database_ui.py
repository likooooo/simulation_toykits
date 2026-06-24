"""Simulation database helpers for Streamlit browser."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

DEFAULT_SPECTRUM_PATH = ["AM1.5G"]

__all__ = [
    "ensure_simulation_database_initialized",
    "prepare_simulation_database",
    "material_nk_arrays",
    "search_material_paths",
]


def ensure_simulation_database_initialized(sim_db=None):
    if sim_db is not None:
        sim_db.init()
        return sim_db
    import simulation_database_parser as sdp

    return sdp.get_simulation_database(init=True)


def prepare_simulation_database(
    sim_db=None,
    db_name: str = "materials",
) -> tuple[Any, list[str]]:
    """Ensure database is installed via simulation_database_parser.get_simulation_database."""
    lines: list[str] = []
    try:
        sim_db = ensure_simulation_database_initialized(sim_db)
        lines.append(f"database root: {sim_db.root_path()}")
    except Exception as exc:
        lines.append(f"prepare failed: {exc}")
        return sim_db, lines

    if db_name not in list(sim_db.database_names()):
        lines.append(f"[{db_name}] unknown database")
        return sim_db, lines

    oghma = sim_db.database(db_name)
    lines.append(f"[{db_name}] local path: {oghma.local_path()}")
    lines.append(f"[{db_name}] ready")
    return sim_db, lines


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


def material_wl_range_um(mat: Any) -> tuple[float, float] | None:
    try:
        wl, _, _ = material_nk_arrays(mat)
    except Exception:
        return None
    return _wl_range_from_array(wl)


def spectrum_wl_range_um(spec: Any) -> tuple[float, float] | None:
    try:
        wl, _ = spectrum_arrays(spec)
    except Exception:
        return None
    return _wl_range_from_array(wl)


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
    db_names: tuple[str, ...],
    *,
    max_results: int | None = None,
    match_blob: Callable[[str, list[str]], str] | None = None,
) -> list[tuple[str, list[str]]]:
    key = (query or "").strip()
    if not key:
        return []
    pattern = re.compile(re.escape(key), re.IGNORECASE)
    matches: list[tuple[str, list[str]]] = []

    for db_name in db_names:
        if db_name not in sim_db.database_names():
            continue
        oghma = sim_db.database(db_name)

        for _node, path_keys, is_leaf in _iter_query_nodes(oghma.query(), []):
            if max_results is not None and len(matches) >= max_results:
                break
            if not is_leaf:
                continue
            if match_blob is not None:
                blob = match_blob(db_name, path_keys)
            else:
                blob = breadcrumb_for(db_name, path_keys)
            if pattern.search(blob) or pattern.search(path_keys[-1] if path_keys else ""):
                matches.append((db_name, list(path_keys)))
    return matches


def search_material_paths(sim_db: Any, material_name: str, db_name: str = "materials") -> list[list[str]]:
    def _blob(_db: str, path_keys: list[str]) -> str:
        return " > ".join(path_keys)

    return [
        path_keys
        for _db, path_keys in _iter_matching_leaf_paths(
            sim_db, material_name, (db_name,), match_blob=_blob
        )
    ]


VISIBLE_DATABASE_NAMES = ("materials", "spectra")


def path_id(db_name: str, path_keys: list[str]) -> str:
    if not path_keys:
        return db_name
    return f"{db_name}:{'/'.join(path_keys)}"


def browser_node_path(db_name: str, path_keys: list[str]) -> str:
    if path_keys:
        return f"{db_name}/{'/'.join(path_keys)}"
    return db_name


def breadcrumb_for(db_name: str, path_keys: list[str]) -> str:
    return " > ".join([db_name, *path_keys])


_DATA_KIND_SEGMENTS = frozenset({"nk", "n", "k", "n2"})


def material_unique_name_from_path(path_keys: list[str]) -> str:
    """Mirror oghma_database.cpp material_names_from_path (unique name only)."""
    if not path_keys:
        return "leaf"
    last = path_keys[-1]
    lower = last.lower()
    if lower.endswith(".yml") or lower.endswith(".yaml"):
        page = Path(last).stem
        for i, seg in enumerate(path_keys[:-1]):
            if seg in _DATA_KIND_SEGMENTS and i > 0:
                book = path_keys[i - 1]
                return f"{book}({page})"
        return page
    return last


def leaf_display_label(db_name: str, path_keys: list[str]) -> str:
    if db_name == "spectra":
        return path_keys[-1] if path_keys else db_name
    return material_unique_name_from_path(path_keys)


def leaf_type_for_db(db_name: str) -> str:
    return "spectrum" if db_name == "spectra" else "material"


def read_leaf_object(oghma: Any, leaf_query: Any) -> Any:
    obj = oghma.read(leaf_query)
    if obj is None:
        raise ValueError("read requires a leaf query_object")
    return obj


def query_at_path(oghma: Any, path_keys: list[str]) -> Any:
    query = oghma.query()
    for key in path_keys:
        if query.is_leaf:
            raise ValueError(f"unexpected leaf before key: {key}")
        keys = list(query.keys)
        if key not in keys:
            raise ValueError(f"key not found: {key}")
        _, query = query.descend(key)
    return query


def get_tree_children(
    sim_db: Any,
    db_name: str,
    path_keys: list[str],
    cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    pid = path_id(db_name, path_keys)
    if pid in cache:
        return cache[pid]
    oghma = sim_db.database(db_name)
    if not path_keys:
        query = oghma.query()
    else:
        query = query_at_path(oghma, path_keys)
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
                "path_id": path_id(db_name, child_path),
                "db_name": db_name,
                "path_keys": child_path,
                "is_leaf": is_leaf,
                "leaf_type": leaf_type_for_db(db_name) if is_leaf else None,
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
    nodes: list[dict[str, Any]] = []
    for db_name in VISIBLE_DATABASE_NAMES:
        if db_name not in sim_db.database_names():
            continue
        root_id = path_id(db_name, [])
        root_node: dict[str, Any] = {
            "key": db_name,
            "path_id": root_id,
            "db_name": db_name,
            "path_keys": [],
            "is_leaf": False,
            "leaf_type": None,
            "child_count": len(get_tree_children(sim_db, db_name, [], children_cache)),
            "children": [],
        }
        if root_id in expanded_paths:
            root_node["children"] = _build_subtree(sim_db, db_name, [], expanded_paths, children_cache)
        nodes.append(root_node)
    return nodes


def _build_subtree(
    sim_db: Any,
    db_name: str,
    path_keys: list[str],
    expanded_paths: set[str],
    children_cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    children_meta = get_tree_children(sim_db, db_name, path_keys, children_cache)
    result: list[dict[str, Any]] = []
    for meta in children_meta:
        node = dict(meta)
        node["children"] = []
        if not meta["is_leaf"] and meta["path_id"] in expanded_paths:
            node["children"] = _build_subtree(
                sim_db, db_name, meta["path_keys"], expanded_paths, children_cache
            )
        result.append(node)
    return result


def search_db_paths(sim_db: Any, query: str, max_results: int = 80) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for db_name, path_keys in _iter_matching_leaf_paths(
        sim_db, query, VISIBLE_DATABASE_NAMES, max_results=max_results
    ):
        matches.append(
            {
                "db_name": db_name,
                "path_keys": path_keys,
                "leaf_type": leaf_type_for_db(db_name),
                "label": leaf_display_label(db_name, path_keys),
                "path_id": path_id(db_name, path_keys),
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


def read_leaf_at_path(sim_db: Any, db_name: str, path_keys: list[str]) -> Any:
    oghma = sim_db.database(db_name)
    leaf = oghma.walk_query_path(path_keys)
    return read_leaf_object(oghma, leaf)
