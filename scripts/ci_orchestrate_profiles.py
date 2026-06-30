#!/usr/bin/env python3
"""Profile loader for CI orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("缺少 PyYAML，请先安装 requirements.txt") from exc


KNOWN_STAGES = {"infra-release", "infra-asan", "sim-release", "sim-asan", "toykits"}
KNOWN_TARGETS = {"infra-release", "infra-asan", "sim-release", "sim-asan", "toykits", "tmm"}
PLACEHOLDER_RE = re.compile(r"^\$\{label_sets\.([a-zA-Z0-9_\-]+)\}$")


@dataclass(frozen=True)
class ProfileTestRule:
    ctest_include: tuple[str, ...] = ()
    ctest_exclude: tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class DeployRule:
    flags: tuple[str, ...]
    verify: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedProfile:
    name: str
    description: str
    stages: tuple[str, ...]
    build_jobs: int
    ctest_jobs: int
    ctest_output: str
    clean_build_dirs: bool
    tests: dict[str, ProfileTestRule]
    deploy: DeployRule


def load_profiles(config_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"profiles 文件格式错误: {config_path}")
    return data


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_placeholders(value: Any, label_sets: dict[str, list[str]]) -> Any:
    if isinstance(value, str):
        match = PLACEHOLDER_RE.match(value)
        if match:
            key = match.group(1)
            if key not in label_sets:
                raise ValueError(f"未知 label_sets 引用: {value}")
            return list(label_sets[key])
        return value
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            resolved = _resolve_placeholders(item, label_sets)
            if isinstance(resolved, list):
                out.extend(resolved)
            else:
                out.append(resolved)
        return out
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, label_sets) for k, v in value.items()}
    return value


def _resolve_profile_raw(
    name: str,
    profiles: dict[str, Any],
    stack: set[str],
) -> dict[str, Any]:
    if name not in profiles:
        raise ValueError(f"未知 profile: {name}")
    if name in stack:
        chain = " -> ".join([*stack, name])
        raise ValueError(f"profile extends 循环依赖: {chain}")
    current = dict(profiles[name])
    parent_name = current.pop("extends", None)
    if parent_name:
        parent = _resolve_profile_raw(parent_name, profiles, stack | {name})
        return _merge_dict(parent, current)
    return current


def resolve_profile(config_path: Path, profile_name: str) -> ResolvedProfile:
    data = load_profiles(config_path)
    defaults = data.get("defaults", {})
    profiles = data.get("profiles", {})
    label_sets = data.get("label_sets", {})
    if not isinstance(profiles, dict):
        raise ValueError("profiles 必须是对象")
    raw = _resolve_profile_raw(profile_name, profiles, set())
    raw = _merge_dict(defaults, raw)
    raw = _resolve_placeholders(raw, label_sets)

    stages = tuple(raw.get("stages", []))
    tests_raw = raw.get("tests", {})
    tests: dict[str, ProfileTestRule] = {}
    for target in KNOWN_TARGETS:
        item = tests_raw.get(target, {})
        tests[target] = ProfileTestRule(
            ctest_include=tuple(item.get("ctest_include", [])),
            ctest_exclude=tuple(item.get("ctest_exclude", [])),
            enabled=bool(item.get("enabled", True)),
        )
    deploy_raw = raw.get("deploy", {})
    return ResolvedProfile(
        name=profile_name,
        description=str(raw.get("description", "")),
        stages=stages,
        build_jobs=int(raw.get("build_jobs", 5)),
        ctest_jobs=int(raw.get("ctest_jobs", 0)),
        ctest_output=str(raw.get("ctest_output", "--output-on-failure")),
        clean_build_dirs=bool(raw.get("clean_build_dirs", False)),
        tests=tests,
        deploy=DeployRule(
            flags=tuple(deploy_raw.get("flags", [])),
            verify=tuple(deploy_raw.get("verify", [])),
        ),
    )


def build_ctest_cmd(
    *,
    output_flag: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    jobs: int,
) -> str:
    cmd = ["ctest", output_flag]
    if include:
        cmd.extend(["-L", ",".join(include)])
    if exclude:
        cmd.extend(["-LE", ",".join(exclude)])
    if jobs > 0:
        cmd.extend(["-j", str(jobs)])
    return " ".join(cmd)


def validate_profiles(
    config_path: Path,
    *,
    known_labels: set[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    data = load_profiles(config_path)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        return ["profiles 为空或格式错误"]

    for name in profiles:
        try:
            profile = resolve_profile(config_path, name)
        except Exception as exc:  # pylint: disable=broad-except
            issues.append(f"{name}: {exc}")
            continue
        unknown_stages = [stage for stage in profile.stages if stage not in KNOWN_STAGES]
        if unknown_stages:
            issues.append(f"{name}: 未知 stages: {unknown_stages}")
        for target, rule in profile.tests.items():
            if target not in KNOWN_TARGETS:
                issues.append(f"{name}: 未知 test target: {target}")
                continue
            if known_labels:
                labels = set(rule.ctest_include) | set(rule.ctest_exclude)
                unknown_labels = sorted(label for label in labels if label not in known_labels)
                if unknown_labels:
                    issues.append(f"{name}/{target}: 未知 ctest label: {unknown_labels}")
    return issues

