#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "The fresh-retraining all-checkpoint queue must run inside tmux." >&2
  exit 2
fi
if (( $# != 2 )); then
  echo "Usage: $0 GPU_IDS_FIXED GPU_IDS_COSINE" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS_FIXED="$1"
GPU_IDS_COSINE="$2"
QUEUE_ROOT="${ROOT}/runs/experiments/fresh_retrain_all_checkpoint_full274_seed42"
EXPORT_ROOT="${ROOT}/runs/exports/fresh_retrain_all_checkpoint_curve"
STEPS=(20 40 60 80 100 120 140 160 180 200 220 240 250)

declare -A SOURCE_ROOTS=(
  [fresh_fixed_t0100]="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_fresh_repro_seed42_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_fresh_repro_seed42_250step_4gpu_s1t1_r4"
  [fresh_cosine_t0200]="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_fresh_cosine_to_t0200_step80_160_seed42_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_fresh_cosine_to_t0200_step80_160_seed42_250step_4gpu_s1t1_r4"
)

for gpu_ids in "${GPU_IDS_FIXED}" "${GPU_IDS_COSINE}"; do
  if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
    echo "Each lane requires exactly four comma-separated GPU IDs: ${gpu_ids}" >&2
    exit 2
  fi
done
if (( $(printf '%s\n' "${GPU_IDS_FIXED},${GPU_IDS_COSINE}" | tr ',' '\n' | sort -u | wc -l) != 8 )); then
  echo "The two lanes must contain eight distinct GPU IDs." >&2
  exit 2
fi

mkdir -p "${QUEUE_ROOT}/logs/exports" "${EXPORT_ROOT}"
"${ROOT}/.venv_tcod/bin/python" "${ROOT}/scripts/prepare_fresh_retrain_all_checkpoint_full274.py"

is_complete_run() {
  local run_root="$1"
  [[ -s "${run_root}/summary.json" ]] || return 1
  [[ -s "${run_root}/task_results.jsonl" ]] || return 1
  [[ "$(wc -l < "${run_root}/task_results.jsonl")" == "274" ]] || return 1
  "${ROOT}/.venv_tcod/bin/python" - "${run_root}" <<'PY_CHECK'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in (root / "task_results.jsonl").read_text(encoding="utf-8").splitlines()]
if summary.get("task_count") != 274 or len(rows) != 274:
    raise SystemExit(1)
if len({row["game_file"] for row in rows}) != 274:
    raise SystemExit(1)
if sum(row["split"] == "seen" for row in rows) != 140:
    raise SystemExit(1)
if sum(row["split"] == "unseen" for row in rows) != 134:
    raise SystemExit(1)
PY_CHECK
}

ensure_model() {
  local method="$1"
  local step="$2"
  local source="${SOURCE_ROOTS[${method}]}/global_step_${step}/actor"
  local native_hf="${source}/huggingface"
  local destination="${EXPORT_ROOT}/${method}/step_${step}"

  if [[ -s "${native_hf}/model.safetensors.index.json" ]] &&
     compgen -G "${native_hf}/model-*.safetensors" >/dev/null; then
    echo "[model] Reusing native HF export ${native_hf}"
    return 0
  fi
  if [[ -s "${destination}/model.safetensors.index.json" ]] &&
     compgen -G "${destination}/model-*.safetensors" >/dev/null; then
    echo "[model] Reusing merged export ${destination}"
    return 0
  fi
  if [[ ! -s "${source}/model_world_size_2_rank_0.pt" ]] ||
     [[ ! -s "${source}/model_world_size_2_rank_1.pt" ]]; then
    echo "Incomplete FSDP checkpoint: ${source}" >&2
    return 1
  fi
  if [[ -e "${destination}" ]]; then
    local archive="${destination}_partial_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "[model] Archiving incomplete export to ${archive}"
    mv "${destination}" "${archive}"
  fi
  mkdir -p "$(dirname "${destination}")"
  local partial="$(dirname "${destination}")/.step_${step}.partial.$(date -u +%Y%m%dT%H%M%SZ).$$"
  echo "[model] Merging ${source} -> ${destination}"
  set +e
  "${ROOT}/.venv_tcod/bin/python" -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "${source}" \
    --target_dir "${partial}" \
    2>&1 | tee "${QUEUE_ROOT}/logs/exports/${method}_step${step}.log"
  local merge_status="${PIPESTATUS[0]}"
  set -e
  (( merge_status == 0 )) || return "${merge_status}"
  if [[ ! -s "${partial}/model.safetensors.index.json" ]] ||
     ! compgen -G "${partial}/model-*.safetensors" >/dev/null; then
    echo "Merge did not produce complete safetensors: ${partial}" >&2
    return 1
  fi
  mv "${partial}" "${destination}"
}

export_tensorboard() {
  local method="$1"
  "${ROOT}/.venv_tcod/bin/python" \
    "${ROOT}/analysis/export_fresh_retrain_all_checkpoint_full274_tensorboard.py" \
    --method "${method}"
}

run_lane() {
  local method="$1"
  local gpu_ids="$2"
  local ray_port="$3"
  local status_path="${QUEUE_ROOT}/queue_status_${method}.tsv"
  printf 'method\tstep\trun_root\texit_status\tnote\n' > "${status_path}"
  local failed=0

  for step in "${STEPS[@]}"; do
    local run_root="${QUEUE_ROOT}/${method}/step_${step}_seed42"
    local config="${ROOT}/configs/experiments/freshallckpt_${method}_step${step}_full274_seed42.yaml"
    local run_tag="freshallckpt-${method}-step${step}-seed42"

    if is_complete_run "${run_root}" && [[ "$(cat "${run_root}/logs/exit_status" 2>/dev/null)" == "0" ]]; then
      echo "[${method}] Skipping completed step ${step}"
      export_tensorboard "${method}"
      printf '%s\t%s\t%s\t0\talready-complete\n' "${method}" "${step}" "${run_root}" >> "${status_path}"
      continue
    fi
    if [[ -e "${run_root}" ]]; then
      local archive="${run_root}_failed_$(date -u +%Y%m%dT%H%M%SZ)"
      echo "[${method}] Archiving partial output to ${archive}"
      mv "${run_root}" "${archive}"
    fi
    mkdir -p "${run_root}/logs"

    set +e
    ensure_model "${method}" "${step}"
    local model_status="$?"
    set -e
    if (( model_status != 0 )); then
      printf '%s\n' "${model_status}" > "${run_root}/logs/exit_status"
      printf '%s\t%s\t%s\t%s\tmodel-export-failed\n' "${method}" "${step}" "${run_root}" "${model_status}" >> "${status_path}"
      failed=1
      continue
    fi

    echo "[${method}] Starting step ${step} on GPUs ${gpu_ids}, Ray port ${ray_port}"
    set +e
    bash "${ROOT}/scripts/_run_experiment_eval.sh" \
      "${config}" "${gpu_ids}" "${run_tag}" "${ray_port}" "${run_root}" \
      2>&1 | tee "${run_root}/logs/tmux.log"
    local eval_status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "${eval_status}" > "${run_root}/logs/exit_status"
    if (( eval_status == 0 )) && is_complete_run "${run_root}"; then
      set +e
      export_tensorboard "${method}"
      local tensorboard_status="$?"
      set -e
      if (( tensorboard_status == 0 )); then
        printf '%s\t%s\t%s\t0\tcompleted\n' "${method}" "${step}" "${run_root}" >> "${status_path}"
        echo "[${method}] Completed step ${step}"
      else
        printf '%s\t%s\t%s\t%s\ttensorboard-export-failed\n' "${method}" "${step}" "${run_root}" "${tensorboard_status}" >> "${status_path}"
        echo "[${method}] TensorBoard export failed at step ${step}" >&2
        failed=1
      fi
    else
      printf '%s\t%s\t%s\t%s\teval-or-validation-failed\n' "${method}" "${step}" "${run_root}" "${eval_status}" >> "${status_path}"
      echo "[${method}] FAILED step ${step}" >&2
      failed=1
    fi
    sleep 10
  done
  return "${failed}"
}

run_lane fresh_fixed_t0100 "${GPU_IDS_FIXED}" 17500 &
fixed_pid="$!"
run_lane fresh_cosine_t0200 "${GPU_IDS_COSINE}" 17501 &
cosine_pid="$!"

set +e
wait "${fixed_pid}"; fixed_status="$?"
wait "${cosine_pid}"; cosine_status="$?"
set -e
if (( fixed_status != 0 || cosine_status != 0 )); then
  echo "Fresh-retraining all-checkpoint queue finished with one or more failures." >&2
  exit 1
fi
echo "All 26 fresh-retraining full274 evaluations completed."
