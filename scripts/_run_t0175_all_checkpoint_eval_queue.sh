#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "tau=0.175 checkpoint evaluation queue must run inside tmux." >&2
  exit 2
fi
if (( $# != 1 )); then
  echo "Usage: $0 GPU_IDS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="$1"
QUEUE_ROOT="${ROOT}/runs/experiments/t0175_all_checkpoint_full274_seed42"
EXPORT_ROOT="${ROOT}/runs/exports/t0175_all_checkpoint_curve"
SOURCE_ROOT="${ROOT}/runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_qwen25_step10_8gpu_s2t4_r16_seed42_20260809164432"
STEPS=(20 40 60 80 100 120 140 160 180 200 220 240)

if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+){4}$ ]]; then
  echo "Exactly five comma-separated GPU IDs are required: ${GPU_IDS}" >&2
  exit 2
fi

mkdir -p "${QUEUE_ROOT}/logs/exports" "${EXPORT_ROOT}"
"${ROOT}/.venv_tcod/bin/python" "${ROOT}/scripts/prepare_t0175_all_checkpoint_full274.py"

ensure_export() {
  local step="$1"
  local destination="${EXPORT_ROOT}/step_${step}"
  if [[ -s "${destination}/model.safetensors.index.json" ]] &&
     compgen -G "${destination}/model-*.safetensors" >/dev/null; then
    echo "[export] Reusing ${destination}"
    return 0
  fi

  local source="${SOURCE_ROOT}/global_step_${step}/actor"
  if [[ ! -s "${source}/model_world_size_2_rank_0.pt" ]] ||
     [[ ! -s "${source}/model_world_size_2_rank_1.pt" ]]; then
    echo "Incomplete FSDP source: ${source}" >&2
    return 1
  fi
  local partial="${EXPORT_ROOT}/.step_${step}.partial.$(date -u +%Y%m%dT%H%M%SZ).$$"
  echo "[export] Merging ${source} -> ${destination}"
  set +e
  "${ROOT}/.venv_tcod/bin/python" -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "${source}" \
    --target_dir "${partial}" \
    2>&1 | tee "${QUEUE_ROOT}/logs/exports/step${step}.log"
  local status="${PIPESTATUS[0]}"
  set -e
  if (( status != 0 )); then
    return "${status}"
  fi
  if [[ ! -s "${partial}/model.safetensors.index.json" ]] ||
     ! compgen -G "${partial}/model-*.safetensors" >/dev/null; then
    echo "Export did not produce complete safetensors: ${partial}" >&2
    return 1
  fi
  mv "${partial}" "${destination}"
}

STATUS_PATH="${QUEUE_ROOT}/queue_status.tsv"
printf 'step\trun_root\texit_status\n' > "${STATUS_PATH}"
failed=0
offset=0
for step in "${STEPS[@]}"; do
  run_root="${QUEUE_ROOT}/step_${step}_seed42"
  config="${ROOT}/configs/experiments/allckpt_entropy_adaptive_v1_t0175_step${step}_full274_seed42.yaml"
  if [[ -s "${run_root}/summary.json" ]] &&
     [[ "$(cat "${run_root}/logs/exit_status" 2>/dev/null)" == "0" ]] &&
     [[ "$(wc -l < "${run_root}/task_results.jsonl" 2>/dev/null)" == "274" ]]; then
    echo "[queue] Skipping completed step ${step}"
    printf '%s\t%s\t0\n' "${step}" "${run_root}" >> "${STATUS_PATH}"
    continue
  fi
  if [[ -e "${run_root}" ]]; then
    archive="${run_root}_failed_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "[queue] Archiving partial output to ${archive}"
    mv "${run_root}" "${archive}"
  fi
  mkdir -p "${run_root}/logs"

  set +e
  ensure_export "${step}"
  export_status="$?"
  set -e
  if (( export_status != 0 )); then
    printf '%s\n' "${export_status}" > "${run_root}/logs/exit_status"
    printf '%s\t%s\t%s\n' "${step}" "${run_root}" "${export_status}" >> "${STATUS_PATH}"
    failed=1
    continue
  fi

  # Keep Ray system ports (43000 + (RAY_PORT % 1000) * 10) outside the
  # slot-3 worker range 48000-50999 used by run_alfworld_eval.sh.
  ray_port="$((18003 + offset * 8))"
  offset="$((offset + 1))"
  echo "[queue] Starting tau=0.175 step ${step} on GPUs ${GPU_IDS}, Ray port ${ray_port}"
  set +e
  ALLOW_OCCUPIED_EVAL_GPUS=1 bash "${ROOT}/scripts/_run_experiment_eval.sh" \
    "${config}" "${GPU_IDS}" "t0175-allckpt-step${step}-seed42" "${ray_port}" "${run_root}" \
    2>&1 | tee "${run_root}/logs/tmux.log"
  status="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "${status}" > "${run_root}/logs/exit_status"
  printf '%s\t%s\t%s\n' "${step}" "${run_root}" "${status}" >> "${STATUS_PATH}"
  if (( status != 0 )); then
    echo "[queue] FAILED: step ${step}" >&2
    failed=1
  else
    echo "[queue] Completed step ${step}"
  fi
  sleep 10
done

if (( failed != 0 )); then
  echo "tau=0.175 checkpoint queue finished with one or more failures." >&2
  exit 1
fi
echo "All 12 missing tau=0.175 full274 evaluations completed."
