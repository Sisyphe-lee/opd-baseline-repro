#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "Warm-up boundary evaluation queue must run inside tmux." >&2
  exit 2
fi
if (( $# != 2 )); then
  echo "Usage: $0 GPU_IDS_LANE_A GPU_IDS_LANE_B" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS_LANE_A="$1"
GPU_IDS_LANE_B="$2"
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

mkdir -p "${QUEUE_ROOT}/logs"

run_lane() {
  lane="$1"
  gpu_ids="$2"
  ray_port="$3"
  shift 3
  status_path="${QUEUE_ROOT}/queue_status_lane_${lane}.tsv"
  printf 'lane\tconfig\trun_root\texit_status\n' > "${status_path}"
  failed=0

  for item in "$@"; do
    IFS=: read -r method step <<<"${item}"
    config="configs/experiments/warmup_boundary_${method}_step${step}_full274_seed42.yaml"
    run_root="runs/experiments/warmup_boundary_full274/${method}/step_${step}_seed42"
    run_tag="warmup-boundary-${method}-step${step}-seed42-lane${lane}"
    run_root_abs="${ROOT}/${run_root}"

    mkdir -p "${run_root_abs}/logs"
    echo "[lane ${lane}] Starting ${config} on GPUs ${gpu_ids}"
    set +e
    ALLOW_OCCUPIED_EVAL_GPUS=1 bash "${ROOT}/scripts/_run_experiment_eval.sh" \
      "${ROOT}/${config}" \
      "${gpu_ids}" \
      "${run_tag}" \
      "${ray_port}" \
      "${run_root_abs}" \
      2>&1 | tee "${run_root_abs}/logs/tmux.log"
    status="${PIPESTATUS[0]}"
    set -e

    printf '%s\n' "${status}" > "${run_root_abs}/logs/exit_status"
    printf '%s\t%s\t%s\t%s\n' \
      "${lane}" "${config}" "${run_root}" "${status}" >> "${status_path}"
    if (( status != 0 )); then
      echo "[lane ${lane}] FAILED with status ${status}: ${config}" >&2
      failed=1
    else
      echo "[lane ${lane}] Completed ${config}"
    fi
  done
  return "${failed}"
}

run_lane a "${GPU_IDS_LANE_A}" 6480 \
  tcod_f2b:60 vanilla_opd:60 entropy_adaptive_v1_t0100:60 \
  tcod_f2b:100 entropy_adaptive_v1_t0100:100 &
lane_a_pid="$!"
run_lane b "${GPU_IDS_LANE_B}" 6481 \
  tcod_f2b:80 vanilla_opd:80 entropy_adaptive_v1_t0100:80 \
  vanilla_opd:100 &
lane_b_pid="$!"

set +e
wait "${lane_a_pid}"
lane_a_status="$?"
wait "${lane_b_pid}"
lane_b_status="$?"
set -e

if (( lane_a_status != 0 || lane_b_status != 0 )); then
  echo "Parallel queue completed with one or more failed evaluations." >&2
  exit 1
fi
echo "All nine warm-up boundary full274 evaluations completed across both lanes."
