#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "All-checkpoint evaluation queue must run inside tmux." >&2
  exit 2
fi
if (( $# != 2 )); then
  echo "Usage: $0 GPU_IDS_LANE_A GPU_IDS_LANE_B" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS_LANE_A="$1"
GPU_IDS_LANE_B="$2"
QUEUE_ROOT="${ROOT}/runs/experiments/all_checkpoint_full274_seed42"
EXPORT_ROOT="${ROOT}/runs/exports/all_checkpoint_curve"
TCOD_SOURCE="${ROOT}/../opd-alfworld-sync-repro/runs/2026-08-07_tcod-f2b-qwen25-3b-7b-paper-promptfix-4gpu-1s1t2train-v0/checkpoints/ALFWORLD_TCOD_REPRO/f2b_qwen25_3b_7b_eta2_promptfix_4gpu_1s1t2train_v0_20260807131207"
VANILLA_SOURCE="${ROOT}/../opd-alfworld-sync-repro/runs/2026-08-08_vanilla-opd-qwen25-3b-7b-paper-promptfix-4gpu-restart/checkpoints/ALFWORLD_OPD_REPRO/vanilla_opd_qwen25_3b_7b_promptfix_4gpu_1s1t2train_restart_20260808"
ADAPTIVE_SOURCE="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4_seed42"

for gpu_ids in "${GPU_IDS_LANE_A}" "${GPU_IDS_LANE_B}"; do
  if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
    echo "Each lane requires exactly four comma-separated GPU IDs: ${gpu_ids}" >&2
    exit 2
  fi
done
if (( $(printf '%s\n' "${GPU_IDS_LANE_A},${GPU_IDS_LANE_B}" | tr ',' '\n' | sort -u | wc -l) != 8 )); then
  echo "The two lanes must contain eight distinct GPU IDs." >&2
  exit 2
fi

mkdir -p "${QUEUE_ROOT}/logs" "${EXPORT_ROOT}"
"${ROOT}/.venv_tcod/bin/python" "${ROOT}/scripts/prepare_all_checkpoint_full274.py"

source_root() {
  case "$1" in
    tcod_f2b) printf '%s\n' "${TCOD_SOURCE}" ;;
    vanilla_opd) printf '%s\n' "${VANILLA_SOURCE}" ;;
    entropy_adaptive_v1_t0100) printf '%s\n' "${ADAPTIVE_SOURCE}" ;;
    *) echo "Unknown method: $1" >&2; return 2 ;;
  esac
}

ensure_export() {
  method="$1"
  step="$2"
  destination="${EXPORT_ROOT}/${method}/step_${step}"
  if [[ -s "${destination}/model.safetensors.index.json" ]] &&
     compgen -G "${destination}/model-*.safetensors" >/dev/null; then
    echo "[export] Reusing ${destination}"
    return 0
  fi

  source="$(source_root "${method}")/global_step_${step}/actor"
  if [[ ! -s "${source}/model_world_size_2_rank_0.pt" ]] ||
     [[ ! -s "${source}/model_world_size_2_rank_1.pt" ]]; then
    echo "Incomplete FSDP source: ${source}" >&2
    return 1
  fi
  mkdir -p "$(dirname "${destination}")" "${QUEUE_ROOT}/logs/exports"
  partial="$(dirname "${destination}")/.step_${step}.partial.$(date -u +%Y%m%dT%H%M%SZ).$$"
  echo "[export] Merging ${source} -> ${destination}"
  set +e
  "${ROOT}/.venv_tcod/bin/python" -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "${source}" \
    --target_dir "${partial}" \
    2>&1 | tee "${QUEUE_ROOT}/logs/exports/${method}_step${step}.log"
  status="${PIPESTATUS[0]}"
  set -e
  if (( status != 0 )); then
    echo "Export failed with status ${status}: ${method} step ${step}" >&2
    return "${status}"
  fi
  if [[ ! -s "${partial}/model.safetensors.index.json" ]] ||
     ! compgen -G "${partial}/model-*.safetensors" >/dev/null; then
    echo "Export did not produce complete safetensors: ${partial}" >&2
    return 1
  fi
  mv "${partial}" "${destination}"
}

run_lane() {
  lane="$1"
  gpu_ids="$2"
  base_ray_port="$3"
  shift 3
  status_path="${QUEUE_ROOT}/queue_status_lane_${lane}.tsv"
  printf 'lane\tmethod\tstep\trun_root\texit_status\n' > "${status_path}"
  failed=0
  port_offset=0

  for item in "$@"; do
    IFS=: read -r method step <<<"${item}"
    config="${ROOT}/configs/experiments/allckpt_${method}_step${step}_full274_seed42.yaml"
    run_root="${QUEUE_ROOT}/${method}/step_${step}_seed42"
    run_tag="allckpt-${method}-step${step}-seed42-lane${lane}"
    ray_port="$((base_ray_port + port_offset * 8))"
    port_offset="$((port_offset + 1))"

    if [[ -s "${run_root}/summary.json" ]] &&
       [[ "$(cat "${run_root}/logs/exit_status" 2>/dev/null)" == "0" ]] &&
       [[ "$(wc -l < "${run_root}/task_results.jsonl" 2>/dev/null)" == "274" ]]; then
      echo "[lane ${lane}] Skipping completed ${method} step ${step}"
      printf '%s\t%s\t%s\t%s\t0\n' "${lane}" "${method}" "${step}" "${run_root}" >> "${status_path}"
      continue
    fi

    if [[ -e "${run_root}" ]]; then
      archive="${run_root}_failed_$(date -u +%Y%m%dT%H%M%SZ)"
      echo "[lane ${lane}] Archiving partial output to ${archive}"
      mv "${run_root}" "${archive}"
    fi
    mkdir -p "${run_root}/logs"

    set +e
    ensure_export "${method}" "${step}"
    export_status="$?"
    set -e
    if (( export_status != 0 )); then
      printf '%s\n' "${export_status}" > "${run_root}/logs/exit_status"
      printf '%s\t%s\t%s\t%s\t%s\n' "${lane}" "${method}" "${step}" "${run_root}" "${export_status}" >> "${status_path}"
      failed=1
      continue
    fi

    echo "[lane ${lane}] Starting ${method} step ${step} on GPUs ${gpu_ids}, Ray port ${ray_port}"
    set +e
    bash "${ROOT}/scripts/_run_experiment_eval.sh" \
      "${config}" "${gpu_ids}" "${run_tag}" "${ray_port}" "${run_root}" \
      2>&1 | tee "${run_root}/logs/tmux.log"
    status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "${status}" > "${run_root}/logs/exit_status"
    printf '%s\t%s\t%s\t%s\t%s\n' "${lane}" "${method}" "${step}" "${run_root}" "${status}" >> "${status_path}"
    if (( status != 0 )); then
      echo "[lane ${lane}] FAILED: ${method} step ${step}" >&2
      failed=1
    else
      echo "[lane ${lane}] Completed ${method} step ${step}"
    fi
    sleep 10
  done
  return "${failed}"
}

# Interleave methods and steps so partial progress already supports comparisons.
run_lane a "${GPU_IDS_LANE_A}" 16800 \
  tcod_f2b:20 vanilla_opd:20 entropy_adaptive_v1_t0100:20 \
  tcod_f2b:120 vanilla_opd:120 entropy_adaptive_v1_t0100:120 \
  tcod_f2b:160 vanilla_opd:160 entropy_adaptive_v1_t0100:160 \
  tcod_f2b:200 vanilla_opd:200 entropy_adaptive_v1_t0100:200 \
  tcod_f2b:240 vanilla_opd:240 &
lane_a_pid="$!"

run_lane b "${GPU_IDS_LANE_B}" 16801 \
  tcod_f2b:40 vanilla_opd:40 entropy_adaptive_v1_t0100:40 \
  tcod_f2b:140 vanilla_opd:140 entropy_adaptive_v1_t0100:140 \
  tcod_f2b:180 vanilla_opd:180 entropy_adaptive_v1_t0100:180 \
  tcod_f2b:220 vanilla_opd:220 entropy_adaptive_v1_t0100:220 \
  entropy_adaptive_v1_t0100:240 &
lane_b_pid="$!"

set +e
wait "${lane_a_pid}"; lane_a_status="$?"
wait "${lane_b_pid}"; lane_b_status="$?"
set -e
if (( lane_a_status != 0 || lane_b_status != 0 )); then
  echo "All-checkpoint queue finished with one or more failures." >&2
  exit 1
fi
echo "All 27 missing full274 evaluations completed."
