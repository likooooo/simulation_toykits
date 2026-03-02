#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATH_TO_BUILD="${path_to_build:-${1:-}}"

if [[ -z "$PATH_TO_BUILD" ]]; then
  echo "用法: $0 <path_to_build>  或先设置环境变量  path_to_build" >&2
  echo "示例: $0 ../simulation/build" >&2
  exit 1
fi

# 支持相对路径（相对于本仓库根目录）
if [[ "$PATH_TO_BUILD" != /* ]]; then
  PATH_TO_BUILD="$REPO_ROOT/$PATH_TO_BUILD"
fi

SO_PATH="$PATH_TO_BUILD/simulation.so"
ARTIFACTS_DIR="$REPO_ROOT/docker_artifacts"
LIBS_DIR="$ARTIFACTS_DIR/libs"

if [[ ! -f "$SO_PATH" ]]; then
  echo "错误: 未找到 $SO_PATH，请确认 path_to_build 指向已构建的 simulation 目录。" >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$LIBS_DIR"
chmod +x scripts/collect_so_deps.sh

echo "收集 simulation.so 的非系统 .so 依赖到 $LIBS_DIR ..."
./scripts/collect_so_deps.sh "$SO_PATH" "$LIBS_DIR"

echo "复制 simulation.so、core_plugins、plugin 到 $ARTIFACTS_DIR ..."
cp "$SO_PATH" "$ARTIFACTS_DIR/"
[[ -d "$PATH_TO_BUILD/core_plugins" ]] && cp -r "$PATH_TO_BUILD/core_plugins" "$ARTIFACTS_DIR/"
[[ -d "$PATH_TO_BUILD/plugin" ]]       && cp -r "$PATH_TO_BUILD/plugin" "$ARTIFACTS_DIR/"

echo "docker_artifacts 已就绪。在仓库根目录执行: docker build -t simulation-toykits:v1 ."
