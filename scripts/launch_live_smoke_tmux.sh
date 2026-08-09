#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
SESSION="${2:-tcod_baseline_live_smoke}"
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Session exists: ${SESSION}" >&2; exit 2; }
tmux new-session -d -s "${SESSION}" "cd '${ROOT}' && bash scripts/_run_live_smoke.sh '${GPU_ID}' > validation/live/live_smoke.log 2>&1"
echo "Started live smoke validation in tmux session ${SESSION}."

