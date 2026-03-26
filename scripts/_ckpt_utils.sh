#!/usr/bin/env bash
# 通用 checkpoint 工具：找最新、生成下一个 ckpt 名、维护 latest.pt
set -euo pipefail

latest_ckpt() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    return 1
  fi
  # shellcheck disable=SC2012
  ls -1t "$dir"/*.pt 2>/dev/null | head -n 1
}

next_ckpt_path() {
  local dir="$1"
  local prefix="${2:-checkpoint}"
  mkdir -p "$dir"
  local n
  n=$(ls -1 "$dir"/"${prefix}"_*.pt 2>/dev/null | wc -l | tr -d ' ')
  printf "%s/%s_%04d.pt" "$dir" "$prefix" "$((n + 1))"
}

update_latest_link() {
  local ckpt_path="$1"
  local dir
  dir="$(dirname "$ckpt_path")"
  (cd "$dir" && ln -sf "$(basename "$ckpt_path")" latest.pt)
}

