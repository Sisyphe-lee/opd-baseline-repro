#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "Usage: $0 GPU_IDS_LANE_A GPU_IDS_LANE_B SESSION" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS_LANE_A="$1"
GPU_IDS_LANE_B="$2"
SESSION="$3"
QUEUE_ROOT="${ROOT}/runs/experiments/all_checkpoint_full274_seed42"

if [[ ! "${SESSION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid tmux session name: ${SESSION}" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${QUEUE_ROOT}/logs"
tmux new-session -d -s "${SESSION}"
tmux set-window-option -t "${SESSION}" remain-on-exit on
tmux respawn-pane -k -t "${SESSION}" \
  "bash -lc \"set -o pipefail; cd '${ROOT}' && bash scripts/_run_all_checkpoint_eval_queue.sh '${GPU_IDS_LANE_A}' '${GPU_IDS_LANE_B}' 2>&1 | tee '${QUEUE_ROOT}/logs/queue.log'; status=\\\${PIPESTATUS[0]}; printf '%s\\\\n' \\\"\\\${status}\\\" > '${QUEUE_ROOT}/logs/queue_exit_status'; exit \\\"\\\${status}\\\"\""
echo "Started all-checkpoint full274 queue in tmux session ${SESSION}."
