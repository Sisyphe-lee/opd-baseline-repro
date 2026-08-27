#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 3 )); then
  echo "Usage: $0 {tcod|vanilla} [GPU_IDS] [SESSION]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="$1"
GPU_IDS="${2:-0,1,2,3}"
SESSION="${3:-tcod_baseline_train_${MODE}}"
if [[ ! "${SESSION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid tmux session name: ${SESSION}" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
tmux new-session -d -s "${SESSION}" "cd '${ROOT}' && bash scripts/_run_train.sh '${MODE}' '${GPU_IDS}'"
echo "Started training in tmux session ${SESSION}."

