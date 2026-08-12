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
QUEUE_ROOT="${ROOT}/runs/experiments/warmup_boundary_full274"

for gpu_ids in "${GPU_IDS_LANE_A}" "${GPU_IDS_LANE_B}"; do
  if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
    echo "Each lane requires exactly four comma-separated GPU IDs: ${gpu_ids}" >&2
    exit 2
  fi
done
if (( $(printf '%s\n' "${GPU_IDS_LANE_A},${GPU_IDS_LANE_B}" | tr ',' '\n' | sort -u | wc -l) != 8 )); then
  echo "The two evaluation lanes must contain eight distinct GPU IDs." >&2
  exit 2
fi
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
  "set -o pipefail; cd '${ROOT}' && bash scripts/_run_warmup_boundary_eval_queue.sh '${GPU_IDS_LANE_A}' '${GPU_IDS_LANE_B}' 2>&1 | tee '${QUEUE_ROOT}/logs/queue.log'; status=\${PIPESTATUS[0]}; printf '%s\\n' \"\${status}\" > '${QUEUE_ROOT}/logs/queue_exit_status'; exit \"\${status}\""
echo "Started warm-up boundary evaluation queue in tmux session ${SESSION}."
