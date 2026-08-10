#!/usr/bin/env bash
set -euo pipefail

if (( $# != 6 )); then
  echo "Usage: $0 CONFIG GPU_IDS RUN_TAG RAY_PORT RUN_ROOT SESSION" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
GPU_IDS="$2"
RUN_TAG="$3"
RAY_PORT="$4"
RUN_ROOT="$5"
SESSION="$6"

if [[ ! "${SESSION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid tmux session name: ${SESSION}" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

RUN_ROOT_ABS="$(realpath -m "${RUN_ROOT}")"
mkdir -p "${RUN_ROOT_ABS}/logs"
tmux new-session -d -s "${SESSION}"
tmux set-window-option -t "${SESSION}" remain-on-exit on
tmux respawn-pane -k -t "${SESSION}" \
  "set -o pipefail; cd '${ROOT}' && bash scripts/_run_experiment_eval.sh '${CONFIG}' '${GPU_IDS}' '${RUN_TAG}' '${RAY_PORT}' '${RUN_ROOT}' 2>&1 | tee '${RUN_ROOT_ABS}/logs/tmux.log'; status=\${PIPESTATUS[0]}; printf '%s\\n' \"\${status}\" > '${RUN_ROOT_ABS}/logs/exit_status'; exit \"\${status}\""
echo "Started experiment evaluation in tmux session ${SESSION}."
