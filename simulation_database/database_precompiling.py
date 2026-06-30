"""Build-time and runtime precompiled bundle for simulation database browsing/search."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import secrets
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import simulation  # noqa: F401 — must load before simulation_database_parser
import simulation_database_parser as sdp

SCHEMA_VERSION = 3
PRECOMPILED_FILENAME = "database.bin"
_BUNDLE_MAGIC = b"SIMDBENC"
_BUNDLE_FORMAT_VERSION = 1
_BUNDLE_SALT_LEN = 16
_BUNDLE_NONCE_LEN = 12
_BUNDLE_TAG_LEN = 16
_BUNDLE_HEADER_LEN = (
    len(_BUNDLE_MAGIC) + 1 + _BUNDLE_SALT_LEN + _BUNDLE_NONCE_LEN + _BUNDLE_TAG_LEN
)
_BUNDLE_HKDF_INFO = b"simulation_db_precompiled_v1"
_MIN_INVERTED_TOKEN_LEN = 2

_logger = logging.getLogger(__name__)
_active_index: PrecompiledIndex | None = None

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True)
class PrecompiledIndex:
    fingerprint: str
    leaf_count: int
    entries: tuple[dict[str, Any], ...]
    children_by_path_id: dict[str, list[dict[str, Any]]]
    kind_by_path_id: dict[str, str]
    inverted_index: dict[str, tuple[int, ...]]
    leaf_objects_by_path_id: dict[str, Any]


def _artifacts_assets_dir() -> Path:
    env = os.environ.get("SIMULATION_ARTIFACTS_DIR", "").strip()
    if not env:
        raise RuntimeError(
            "SIMULATION_ARTIFACTS_DIR is not set; set it to the simulation runtime artifacts directory"
        )
    return Path(env).resolve() / "assets"


def precompiled_bundle_path() -> Path:
    return _artifacts_assets_dir() / PRECOMPILED_FILENAME


def _resolve_key_material() -> bytes:
    env = os.environ.get("SIMULATION_DATABASE_KEY", "").strip()
    if env:
        return env.encode("utf-8")
    raise RuntimeError("SIMULATION_DATABASE_KEY is not set")


def _derive_aes_key(key_material: bytes, file_salt: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=file_salt,
        info=_BUNDLE_HKDF_INFO,
    ).derive(key_material)


def _encrypt_bundle_payload(payload: bytes, *, key_material: bytes | None = None) -> bytes:
    if key_material is None:
        key_material = _resolve_key_material()
    file_salt = secrets.token_bytes(_BUNDLE_SALT_LEN)
    nonce = secrets.token_bytes(_BUNDLE_NONCE_LEN)
    aes_key = _derive_aes_key(key_material, file_salt)
    encrypted = AESGCM(aes_key).encrypt(nonce, payload, None)
    ciphertext = encrypted[:-_BUNDLE_TAG_LEN]
    tag = encrypted[-_BUNDLE_TAG_LEN:]
    return (
        _BUNDLE_MAGIC
        + bytes([_BUNDLE_FORMAT_VERSION])
        + file_salt
        + nonce
        + tag
        + struct.pack(">I", len(ciphertext))
        + ciphertext
    )


def _decrypt_bundle_payload(data: bytes, *, key_material: bytes | None = None) -> bytes:
    min_len = _BUNDLE_HEADER_LEN + 4
    if len(data) < min_len or not data.startswith(_BUNDLE_MAGIC):
        raise ValueError("invalid database bundle")
    offset = len(_BUNDLE_MAGIC)
    version = data[offset]
    offset += 1
    if version != _BUNDLE_FORMAT_VERSION:
        raise ValueError("invalid database bundle")
    file_salt = data[offset : offset + _BUNDLE_SALT_LEN]
    offset += _BUNDLE_SALT_LEN
    nonce = data[offset : offset + _BUNDLE_NONCE_LEN]
    offset += _BUNDLE_NONCE_LEN
    tag = data[offset : offset + _BUNDLE_TAG_LEN]
    offset += _BUNDLE_TAG_LEN
    (ciphertext_len,) = struct.unpack(">I", data[offset : offset + 4])
    offset += 4
    ciphertext = data[offset : offset + ciphertext_len]
    if offset + ciphertext_len != len(data):
        raise ValueError("invalid database bundle")
    if key_material is None:
        key_material = _resolve_key_material()
    aes_key = _derive_aes_key(key_material, file_salt)
    try:
        return AESGCM(aes_key).decrypt(nonce, ciphertext + tag, None)
    except InvalidTag as exc:
        raise ValueError("invalid database bundle") from exc


def _is_navigable_yml(path: Path) -> bool:
    if path.suffix.lower() not in (".yml", ".yaml"):
        return False
    if path.name in ("about.yml", "about.yaml"):
        return False
    return path.is_file()


def compute_database_fingerprint(db_root: Path) -> str:
    root = Path(db_root).resolve()
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not _is_navigable_yml(path):
            continue
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        rows.append(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return f"sha256:{digest}"


def search_blob_lower(entry: dict[str, Any]) -> str:
    label = str(entry.get("label", ""))
    breadcrumb = str(entry.get("breadcrumb", ""))
    last_key = str(entry.get("last_key", ""))
    path_id = str(entry.get("path_id", ""))
    return f"{label}\n{breadcrumb}\n{last_key}\n{path_id}".lower()


def tokenize_search_text(text: str) -> list[str]:
    lower = (text or "").lower()
    tokens: set[str] = set()
    for part in _TOKEN_SPLIT_RE.split(lower):
        if len(part) >= _MIN_INVERTED_TOKEN_LEN:
            tokens.add(part)
    for chunk in lower.replace("_", " ").replace(".", " ").split():
        if len(chunk) >= _MIN_INVERTED_TOKEN_LEN:
            tokens.add(chunk)
    return sorted(tokens)


def build_inverted_index(entries: list[dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    buckets: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        blob = entry.get("search_blob_lower") or search_blob_lower(entry)
        for token in tokenize_search_text(blob):
            buckets.setdefault(token, []).append(index)
    return {token: tuple(indices) for token, indices in buckets.items()}


def candidate_entry_indices(
    inverted_index: dict[str, tuple[int, ...]],
    query: str,
    *,
    entry_count: int,
) -> list[int] | None:
    """Return candidate entry indices; None means scan all entries."""
    key = (query or "").strip()
    if not key:
        return []
    q_lower = key.lower()
    tokens = tokenize_search_text(q_lower)
    if not tokens and len(q_lower) >= _MIN_INVERTED_TOKEN_LEN:
        tokens = [q_lower]
    if not tokens:
        return None

    candidate_set: set[int] | None = None
    for token in tokens:
        hits = inverted_index.get(token)
        if not hits:
            return []
        token_set = set(hits)
        if candidate_set is None:
            candidate_set = token_set
        else:
            candidate_set &= token_set
        if not candidate_set:
            return []
    if candidate_set is None:
        return None
    return sorted(candidate_set)


def _entry_from_leaf(
    path_keys: list[str],
    leaf_type: str,
    *,
    path_id_fn,
    breadcrumb_fn,
) -> dict[str, Any]:
    pid = path_id_fn(path_keys)
    last_key = path_keys[-1] if path_keys else ""
    if leaf_type == "spectrum":
        label = last_key or "spectrum"
    else:
        label = simulation.material_unique_name_from_path_keys(path_keys)
    entry = {
        "path_keys": list(path_keys),
        "path_id": pid,
        "leaf_type": leaf_type,
        "label": label,
        "breadcrumb": breadcrumb_fn(path_keys),
        "last_key": last_key,
        "is_material_path": "materials" in path_keys,
    }
    entry["search_blob_lower"] = search_blob_lower(entry)
    return entry


def _object_from_leaf_payload(sim_db: Any, payload: dict[str, Any]) -> Any:
    kind = payload["kind"]
    source_path = str(payload["source_path"])
    if kind == "material":
        return sim_db.material_from_payload(source_path, payload)
    if kind == "spectrum":
        return sim_db.spectrum_from_payload(source_path, payload)
    raise ValueError(f"unsupported leaf kind: {kind!r}")


def compile_database_index(sim_db: Any, *, out_path: Path | None = None) -> Path:
    """Walk YAML tree once; write precompiled bundle with leaf payloads."""
    from simulation_database.database_ui import (
        _iter_query_nodes,
        breadcrumb_for,
        get_tree_children,
        leaf_type_for_path,
        path_id,
    )

    db_root = Path(sim_db.local_path()).resolve()
    fingerprint = compute_database_fingerprint(db_root)
    entries: list[dict[str, Any]] = []
    children_by_path_id: dict[str, list[dict[str, Any]]] = {}
    kind_by_path_id: dict[str, str] = {}
    leaf_payloads_by_path_id: dict[str, dict[str, Any]] = {}
    kind_cache: dict[str, str] = {}
    children_cache: dict[str, list[dict[str, Any]]] = {}
    seen_dirs: set[str] = set()

    children_by_path_id[""] = get_tree_children(
        sim_db, [], children_cache, kind_cache=kind_cache
    )
    seen_dirs.add("")

    for node, path_keys, is_leaf in _iter_query_nodes(sim_db.query(), []):
        pid = path_id(path_keys)
        if is_leaf:
            leaf_type = leaf_type_for_path(sim_db, path_keys, kind_cache=kind_cache)
            if leaf_type is None:
                continue
            kind_by_path_id[pid] = leaf_type
            entries.append(
                _entry_from_leaf(
                    path_keys,
                    leaf_type,
                    path_id_fn=path_id,
                    breadcrumb_fn=breadcrumb_for,
                )
            )
            leaf_payloads_by_path_id[pid] = sdp.parse_yml_leaf(node.storage_path())
            continue
        if pid in seen_dirs:
            continue
        seen_dirs.add(pid)
        children_by_path_id[pid] = get_tree_children(
            sim_db, path_keys, children_cache, kind_cache=kind_cache
        )

    inverted_index = build_inverted_index(entries)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "leaf_count": len(entries),
        "entries": entries,
        "children_by_path_id": children_by_path_id,
        "kind_by_path_id": kind_by_path_id,
        "inverted_index": inverted_index,
        "leaf_payloads_by_path_id": leaf_payloads_by_path_id,
    }
    destination = Path(out_path) if out_path is not None else precompiled_bundle_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    destination.write_bytes(_encrypt_bundle_payload(payload))
    return destination


def _hydrate_leaf_objects(sim_db: Any, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for pid, payload in payloads.items():
        objects[pid] = _object_from_leaf_payload(sim_db, payload)
    return objects


def _bundle_to_index(sim_db: Any, bundle: dict[str, Any]) -> PrecompiledIndex:
    schema = bundle.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported precompiled schema: {schema!r} (expected {SCHEMA_VERSION})")
    entries = tuple(dict(entry) for entry in bundle["entries"])
    inverted_index = {
        str(token): tuple(int(i) for i in indices)
        for token, indices in bundle["inverted_index"].items()
    }
    payloads = dict(bundle["leaf_payloads_by_path_id"])
    leaf_objects = _hydrate_leaf_objects(sim_db, payloads)
    if len(leaf_objects) != int(bundle["leaf_count"]):
        raise ValueError(
            f"leaf object count mismatch: hydrated={len(leaf_objects)} "
            f"leaf_count={bundle['leaf_count']}"
        )
    return PrecompiledIndex(
        fingerprint=str(bundle["fingerprint"]),
        leaf_count=int(bundle["leaf_count"]),
        entries=entries,
        children_by_path_id=dict(bundle["children_by_path_id"]),
        kind_by_path_id=dict(bundle["kind_by_path_id"]),
        inverted_index=inverted_index,
        leaf_objects_by_path_id=leaf_objects,
    )


def load_database_index(sim_db: Any, path: Path | None = None) -> PrecompiledIndex:
    pkl_path = Path(path) if path is not None else precompiled_bundle_path()
    if not pkl_path.is_file():
        raise FileNotFoundError(f"precompiled database bundle not found: {pkl_path}")
    payload = _decrypt_bundle_payload(pkl_path.read_bytes())
    bundle = pickle.loads(payload)
    return _bundle_to_index(sim_db, bundle)


def set_active_index(index: PrecompiledIndex | None) -> None:
    global _active_index
    _active_index = index


def get_active_index() -> PrecompiledIndex | None:
    return _active_index


def load_or_build_database_index(sim_db: Any) -> PrecompiledIndex:
    """Load bundle, hydrate all leaf objects, and activate index."""
    if _active_index is not None:
        return _active_index
    index = load_database_index(sim_db)
    set_active_index(index)
    _logger.debug(
        "loaded precompiled database: %s leaves, fingerprint=%s",
        index.leaf_count,
        index.fingerprint,
    )
    return index


def get_precompiled_leaf_object(path_keys: list[str]) -> Any:
    from simulation_database.database_ui import path_id as _path_id

    index = get_active_index()
    if index is None:
        raise RuntimeError("precompiled database index is not loaded")
    pid = _path_id(path_keys)
    try:
        return index.leaf_objects_by_path_id[pid]
    except KeyError as exc:
        raise KeyError(f"precompiled leaf not found: {pid!r}") from exc


def panel_search_catalog(sim_db: Any) -> dict[str, Any]:
    """Compact search catalog for browser-side filtering."""
    index = get_active_index()
    if index is None:
        return {"entries": [], "inverted": {}, "fingerprint": ""}
    return {
        "fingerprint": index.fingerprint,
        "entries": [
            [entry["path_keys"], entry["path_id"], entry["leaf_type"], entry["label"]]
            for entry in index.entries
        ],
        "inverted": {token: list(indices) for token, indices in index.inverted_index.items()},
    }
