#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "Warm-up boundary evaluation queue must run inside tmux." >&2
  exit 2
fi
if (( $# != 1 )); then
  echo "Usage: $0 GPU_IDS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="$1"
QUEUE_ROOT="${ROOT}/runs/experiments/warmup_boundary_full274"
STATUS_PATH="${QUEUE_ROOT}/queue_status.tsv"

if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  echo "Exactly four comma-separated GPU IDs are required: ${GPU_IDS}" >&2
  exit 2
fi

mkdir -p "${QUEUE_ROOT}/logs"
printf 'index\tconfig\trun_root\texit_status\n' > "${STATUS_PATH}"

configs=(
  warmup_boundary_tcod_f2b_step60_full274_seed42
  warmup_boundary_tcod_f2b_step80_full274_seed42
  warmup_boundary_tcod_f2b_step100_full274_seed42
  warmup_boundary_vanilla_opd_step60_full274_seed42
  warmup_boundary_vanilla_opd_step80_full274_seed42
  warmup_boundary_vanilla_opd_step100_full274_seed42
  warmup_boundary_entropy_adaptive_v1_t0100_step60_full274_seed42
  warmup_boundary_entropy_adaptive_v1_t0100_step80_full274_seed42
  warmup_boundary_entropy_adaptive_v1_t0100_step100_full274_seed42
)
methods=(
  tcod_f2b tcod_f2b tcod_f2b
  vanilla_opd vanilla_opd vanilla_opd
  entropy_adaptive_v1_t0100 entropy_adaptive_v1_t0100 entropy_adaptive_v1_t0100
)
steps=(60 80 100 60 80 100 60 80 100)

failed=0
for index in "${!configs[@]}"; do
  ordinal="$((index + 1))"
  config="configs/experiments/${configs[index]}.yaml"
  run_root="runs/experiments/warmup_boundary_full274/${methods[index]}/step_${steps[index]}_seed42"
  run_tag="warmup-boundary-${methods[index]}-step${steps[index]}-seed42"
  ray_port="$((6480 + index))"
  run_root_abs="${ROOT}/${run_root}"

  mkdir -p "${run_root_abs}/logs"
  echo "[${ordinal}/9] Starting ${config}"
  set +e
  bash "${ROOT}/scripts/_run_experiment_eval.sh" \
    "${ROOT}/${config}" \
    "${GPU_IDS}" \
    "${run_tag}" \
    "${ray_port}" \
    "${run_root_abs}" \
    2>&1 | tee "${run_root_abs}/logs/tmux.log"
  status="${PIPESTATUS[0]}"
  set -e

  printf '%s\n' "${status}" > "${run_root_abs}/logs/exit_status"
  printf '%s\t%s\t%s\t%s\n' \
    "${ordinal}" "${config}" "${run_root}" "${status}" >> "${STATUS_PATH}"
  if (( status != 0 )); then
    echo "[${ordinal}/9] FAILED with status ${status}: ${config}" >&2
    failed=1
  else
    echo "[${ordinal}/9] Completed ${config}"
  fi
done

if (( failed != 0 )); then
  echo "Queue completed with one or more failed evaluations. See ${STATUS_PATH}." >&2
  exit 1
fi
echo "All nine warm-up boundary full274 evaluations completed."
