#!/usr/bin/env bash
set -euo pipefail

if (( $# > 2 )); then
  echo "Usage: $0 [GPU_IDS] [SESSION]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
SESSION="${2:-two_stage_zero_diagnostic}"
[[ "${SESSION}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid tmux session name" >&2; exit 2; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
mkdir -p "${ROOT}/runs/experiments/two_stage_distillation/logs"
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && bash scripts/_run_two_stage_zero_diagnostic.sh '${GPU_IDS}' 2>&1 | tee runs/experiments/two_stage_distillation/logs/zero_diagnostic_driver.log"
echo "Started zero diagnostic in tmux session ${SESSION}."
