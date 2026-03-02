#!/usr/bin/env bash
set -euo pipefail

SO_PATH="${1:?Usage: $0 <path_to_simulation.so> <output_libs_dir>}"
OUT_DIR="${2:?Usage: $0 <path_to_simulation.so> <output_libs_dir>}"

# 仅排除基础系统库（镜像中已有），其余全部打包
is_system_lib() {
    local path="$1"
    local name
    name=$(basename "$path")
    [[ -z "$path" ]] && return 0
    # 内核/动态链接器
    [[ "$name" == linux-vdso* ]] && return 0
    [[ "$name" == ld-linux* ]] && return 0
    # glibc / libstdc++ / libgcc / 线程与运行时
    [[ "$name" == libc.so* ]] && return 0
    [[ "$name" == libm.so* ]] && return 0
    [[ "$name" == libdl.so* ]] && return 0
    [[ "$name" == libpthread.so* ]] && return 0
    [[ "$name" == librt.so* ]] && return 0
    [[ "$name" == libgcc_s.so* ]] && return 0
    [[ "$name" == libstdc++.so* ]] && return 0
    # 使用镜像内 Python，不打包宿主机 libpython
    [[ "$name" == libpython* ]] && return 0
    return 1
}

resolve_path() {
    local path="$1"
    if [[ -L "$path" ]]; then
        readlink -f "$path" 2>/dev/null || realpath "$path" 2>/dev/null || echo "$path"
    else
        echo "$path"
    fi
}

mkdir -p "$OUT_DIR"
declare -A seen
to_visit=("$SO_PATH")

while [[ ${#to_visit[@]} -gt 0 ]]; do
    current="${to_visit[0]}"
    to_visit=("${to_visit[@]:1}")
    [[ -z "$current" || ! -f "$current" ]] && continue

    while IFS= read -r line; do
        path=""
        if [[ "$line" =~ =\>\ ([[:space:]]*)(/[^[:space:]]+) ]]; then
            path="${BASH_REMATCH[2]}"
        elif [[ "$line" =~ ^[[:space:]]*(/[^[:space:]]+\.so[^[:space:]]*)[[:space:]]*\( ]]; then
            path="${BASH_REMATCH[1]}"
        fi
        [[ -z "$path" ]] && continue
        is_system_lib "$path" && continue

        real_path=$(resolve_path "$path")
        [[ ! -f "$real_path" ]] && continue
        [[ -n "${seen[$real_path]:-}" ]] && continue
        seen[$real_path]=1

        name=$(basename "$real_path")
        cp -L "$real_path" "$OUT_DIR/$name" 2>/dev/null || cp "$real_path" "$OUT_DIR/$name"
        to_visit+=("$real_path")
    done < <(ldd "$current" 2>/dev/null || true)
done

echo "Collected ${#seen[@]} non-system library/ies into $OUT_DIR"
