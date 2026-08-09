#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 3 )); then
  echo "Usage: $0 {tcod|vanilla} [GPU_IDS] [SESSION]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="$1"
GPU_IDS="${2:-0,1,2,3}"
SESSION="${3:-tcod_baseline_eval_${MODE}}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
tmux new-session -d -s "${SESSION}" "cd '${ROOT}' && bash scripts/_run_eval.sh '${MODE}' '${GPU_IDS}'"
echo "Started full evaluation in tmux session ${SESSION}."

