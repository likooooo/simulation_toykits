"""Tests for encrypted database bundle read/write."""

from __future__ import annotations

import pickle

import pytest

from simulation_database.database_precompiling import (
    _decrypt_bundle_payload,
    _encrypt_bundle_payload,
    _resolve_key_material,
)


@pytest.fixture
def key_material() -> bytes:
    return b"test-simulation-database-key"


def test_encrypt_decrypt_round_trip(key_material: bytes) -> None:
    payload = pickle.dumps({"schema_version": 3, "leaf_count": 0}, protocol=pickle.HIGHEST_PROTOCOL)
    encrypted = _encrypt_bundle_payload(payload, key_material=key_material)
    assert encrypted.startswith(b"SIMDBENC")
    assert _decrypt_bundle_payload(encrypted, key_material=key_material) == payload


def test_wrong_key_fails(key_material: bytes) -> None:
    payload = b"payload-bytes"
    encrypted = _encrypt_bundle_payload(payload, key_material=key_material)
    with pytest.raises(ValueError, match="invalid database bundle"):
        _decrypt_bundle_payload(encrypted, key_material=b"other-key")


def test_corrupted_tag_fails(key_material: bytes) -> None:
    payload = b"payload-bytes"
    encrypted = bytearray(_encrypt_bundle_payload(payload, key_material=key_material))
    encrypted[-5] ^= 0xFF
    with pytest.raises(ValueError, match="invalid database bundle"):
        _decrypt_bundle_payload(bytes(encrypted), key_material=key_material)


def test_invalid_magic_fails(key_material: bytes) -> None:
    payload = b"payload-bytes"
    encrypted = bytearray(_encrypt_bundle_payload(payload, key_material=key_material))
    encrypted[0] ^= 0xFF
    with pytest.raises(ValueError, match="invalid database bundle"):
        _decrypt_bundle_payload(bytes(encrypted), key_material=key_material)


def test_resolve_key_material_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMULATION_DATABASE_KEY", raising=False)
    monkeypatch.setenv("SIMULATION_DATABASE_KEY", "env-key-value")
    assert _resolve_key_material() == b"env-key-value"


def test_resolve_key_material_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMULATION_DATABASE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SIMULATION_DATABASE_KEY is not set"):
        _resolve_key_material()
