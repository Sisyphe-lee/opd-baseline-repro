#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "The recent-three post-step-80 evaluation queue must run inside tmux." >&2
  exit 2
fi
if (( $# != 1 )); then
  echo "Usage: $0 GPU_IDS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="$1"
QUEUE_ROOT="${ROOT}/runs/experiments/recent_three_post80_full274_seed42"
EXPORT_ROOT="${ROOT}/runs/exports/recent_three_post80_curve"
STEPS=(100 120 140 160 180 200 220 240 250)
BRANCHES=(linear_to_full cosine_to_t0200 cosine_to_t0175)

declare -A SOURCE_ROOTS=(
  [linear_to_full]="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_step80_linear_anneal_to_full_step160_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_step80_linear_anneal_to_full_step160_250step_4gpu_s1t1_r4_seed42"
  [cosine_to_t0200]="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_step80_cosine_to_t0200_step160_hold_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_step80_cosine_to_t0200_step160_hold_250step_4gpu_s1t1_r4_seed42"
  [cosine_to_t0175]="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_step80_cosine_to_t0175_step160_hold_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_step80_cosine_to_t0175_step160_hold_250step_4gpu_s1t1_r4_seed42"
)

declare -A REUSED_SUMMARIES=(
  [linear_to_full]="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_step80_linear_anneal_to_full_step160_250step_4gpu_s1t1_r4/evaluation/step250_full274/summary.json"
  [cosine_to_t0200]="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_step80_cosine_to_t0200_step160_hold_250step_4gpu_s1t1_r4/evaluation/step250_full274/summary.json"
)

if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  echo "Exactly four comma-separated GPU IDs are required: ${GPU_IDS}" >&2
  exit 2
fi

mkdir -p "${QUEUE_ROOT}/logs/exports" "${EXPORT_ROOT}"
"${ROOT}/.venv_tcod/bin/python" "${ROOT}/scripts/prepare_recent_three_post80_full274.py"

is_complete_summary() {
  local summary="$1"
  [[ -s "${summary}" ]] || return 1
  "${ROOT}/.venv_tcod/bin/python" - "${summary}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if data.get("task_count") != 274:
    raise SystemExit(1)
PY
}

wait_for_checkpoint() {
  local branch="$1"
  local step="$2"
  local actor="${SOURCE_ROOTS[${branch}]}/global_step_${step}/actor"
  while [[ ! -s "${actor}/model_world_size_2_rank_0.pt" ]] ||
        [[ ! -s "${actor}/model_world_size_2_rank_1.pt" ]]; do
    echo "[wait] $(date -u +%FT%TZ) waiting for ${branch} step ${step} checkpoint"
    sleep 60
  done
}

wait_for_gpus() {
  local busy gpu_id gpu_pid used_memory
  while true; do
    busy=0
    IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
    for gpu_id in "${gpu_array[@]}"; do
      used_memory="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
      if [[ "${used_memory}" =~ ^[0-9]+$ ]] && (( used_memory > 2048 )); then
        busy=1
        echo "[wait] $(date -u +%FT%TZ) GPU ${gpu_id} still uses ${used_memory} MiB"
      fi
      while IFS= read -r gpu_pid; do
        [[ -n "${gpu_pid}" ]] || continue
        if kill -0 "${gpu_pid}" 2>/dev/null; then
          busy=1
          echo "[wait] $(date -u +%FT%TZ) GPU ${gpu_id} occupied by live PID ${gpu_pid}"
        fi
      done < <(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
    done
    (( busy == 0 )) && return 0
    sleep 60
  done
}

ensure_export() {
  local branch="$1"
  local step="$2"
  local destination="${EXPORT_ROOT}/${branch}/step_${step}"
  if [[ -s "${destination}/model.safetensors.index.json" ]] &&
     compgen -G "${destination}/model-*.safetensors" >/dev/null; then
    echo "[export] Reusing ${destination}"
    return 0
  fi

  wait_for_checkpoint "${branch}" "${step}"
  local source="${SOURCE_ROOTS[${branch}]}/global_step_${step}/actor"
  local partial="${EXPORT_ROOT}/${branch}/.step_${step}.partial.$(date -u +%Y%m%dT%H%M%SZ).$$"
  mkdir -p "${EXPORT_ROOT}/${branch}"
  echo "[export] Merging ${source} -> ${destination}"
  set +e
  "${ROOT}/.venv_tcod/bin/python" -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "${source}" \
    --target_dir "${partial}" \
    2>&1 | tee "${QUEUE_ROOT}/logs/exports/${branch}_step${step}.log"
  local merge_status="${PIPESTATUS[0]}"
  set -e
  (( merge_status == 0 )) || return "${merge_status}"
  if [[ ! -s "${partial}/model.safetensors.index.json" ]] ||
     ! compgen -G "${partial}/model-*.safetensors" >/dev/null; then
    echo "Export did not produce complete safetensors: ${partial}" >&2
    return 1
  fi
  mv "${partial}" "${destination}"
}

STATUS_PATH="${QUEUE_ROOT}/queue_status.tsv"
printf 'branch\tstep\trun_root\texit_status\tnote\n' > "${STATUS_PATH}"
failed=0
offset=0
for branch in "${BRANCHES[@]}"; do
  for step in "${STEPS[@]}"; do
    run_root="${QUEUE_ROOT}/${branch}/step_${step}_seed42"
    config="${ROOT}/configs/experiments/post80_${branch}_step${step}_full274_seed42.yaml"

    reused="${REUSED_SUMMARIES[${branch}]:-}"
    if [[ "${step}" == "250" ]] && [[ -n "${reused}" ]] && is_complete_summary "${reused}"; then
      echo "[queue] Reusing completed ${branch} step 250: ${reused}"
      printf '%s\t%s\t%s\t0\treused-existing-step250\n' "${branch}" "${step}" "${reused}" >> "${STATUS_PATH}"
      continue
    fi

    if [[ -s "${run_root}/summary.json" ]] &&
       [[ "$(cat "${run_root}/logs/exit_status" 2>/dev/null)" == "0" ]] &&
       [[ "$(wc -l < "${run_root}/task_results.jsonl" 2>/dev/null)" == "274" ]] &&
       is_complete_summary "${run_root}/summary.json"; then
      echo "[queue] Skipping completed ${branch} step ${step}"
      printf '%s\t%s\t%s\t0\talready-complete\n' "${branch}" "${step}" "${run_root}" >> "${STATUS_PATH}"
      continue
    fi
    if [[ -e "${run_root}" ]]; then
      archive="${run_root}_failed_$(date -u +%Y%m%dT%H%M%SZ)"
      echo "[queue] Archiving partial output to ${archive}"
      mv "${run_root}" "${archive}"
    fi
    mkdir -p "${run_root}/logs"

    set +e
    ensure_export "${branch}" "${step}"
    export_status="$?"
    set -e
    if (( export_status != 0 )); then
      printf '%s\n' "${export_status}" > "${run_root}/logs/exit_status"
      printf '%s\t%s\t%s\t%s\texport-failed\n' "${branch}" "${step}" "${run_root}" "${export_status}" >> "${STATUS_PATH}"
      failed=1
      continue
    fi

    wait_for_gpus
    ray_port="$((18101 + offset * 8))"
    offset="$((offset + 1))"
    echo "[queue] Starting ${branch} step ${step} on GPUs ${GPU_IDS}, Ray port ${ray_port}"
    set +e
    bash "${ROOT}/scripts/_run_experiment_eval.sh" \
      "${config}" "${GPU_IDS}" "post80-${branch}-step${step}-seed42" "${ray_port}" "${run_root}" \
      2>&1 | tee "${run_root}/logs/tmux.log"
    eval_status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "${eval_status}" > "${run_root}/logs/exit_status"
    if (( eval_status == 0 )) && is_complete_summary "${run_root}/summary.json" &&
       [[ "$(wc -l < "${run_root}/task_results.jsonl")" == "274" ]]; then
      printf '%s\t%s\t%s\t0\tcompleted\n' "${branch}" "${step}" "${run_root}" >> "${STATUS_PATH}"
      echo "[queue] Completed ${branch} step ${step}"
    else
      printf '%s\t%s\t%s\t%s\teval-or-validation-failed\n' "${branch}" "${step}" "${run_root}" "${eval_status}" >> "${STATUS_PATH}"
      echo "[queue] FAILED: ${branch} step ${step}" >&2
      failed=1
    fi
    sleep 10
  done
done

if (( failed != 0 )); then
  echo "The recent-three post-step-80 queue finished with one or more failures." >&2
  exit 1
fi
echo "All 25 missing recent-three post-step-80 full274 evaluations completed."
"${ROOT}/.venv_tcod/bin/python" "${ROOT}/analysis/plot_recent_three_post80_curve.py"
echo "Joined the three branches to the shared original step-80 result and generated the static curve."
