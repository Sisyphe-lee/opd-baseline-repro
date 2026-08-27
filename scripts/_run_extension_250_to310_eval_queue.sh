#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "The extension full274 queue must run inside tmux." >&2
  exit 2
fi
if (( $# != 2 )); then
  echo "Usage: $0 GPU_IDS_ADAPTIVE GPU_IDS_TCOD" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS_ADAPTIVE="$1"
GPU_IDS_TCOD="$2"
QUEUE_ROOT="${ROOT}/runs/experiments/extension_250_to310_full274_seed42"
EXPORT_ROOT="${ROOT}/runs/exports/extension_250_to310"
STEPS=(270 290 310)

declare -A SOURCE_ROOTS=(
  [adaptive_t0100]="${ROOT}/runs/experiments/entropy_adaptive_v1_t0100_extend_step250_to310_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_extend_step250_to310_4gpu_s1t1_r4_seed42"
  [tcod_f2b]="${ROOT}/runs/experiments/tcod_f2b_extend_step250_to310_4gpu_s1t1_r4/checkpoints/ALFWORLD_TCOD_REPRO/tcod_f2b_extend_step250_to310_4gpu_s1t1_r4_seed42"
)

for gpu_ids in "${GPU_IDS_ADAPTIVE}" "${GPU_IDS_TCOD}"; do
  if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
    echo "Each lane requires exactly four comma-separated GPU IDs: ${gpu_ids}" >&2
    exit 2
  fi
done
if (( $(printf '%s\n' "${GPU_IDS_ADAPTIVE},${GPU_IDS_TCOD}" | tr ',' '\n' | sort -u | wc -l) != 8 )); then
  echo "The two lanes must contain eight distinct GPU IDs." >&2
  exit 2
fi

mkdir -p "${QUEUE_ROOT}/logs/exports" "${EXPORT_ROOT}"
"${ROOT}/.venv_tcod/bin/python" "${ROOT}/scripts/prepare_extension_250_to310_full274.py"

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

run_lane() {
  local method="$1"
  local gpu_ids="$2"
  local ray_port="$3"
  local status_path="${QUEUE_ROOT}/queue_status_${method}.tsv"
  printf 'method\tstep\trun_root\texit_status\tnote\n' > "${status_path}"
  local failed=0

  for step in "${STEPS[@]}"; do
    local run_root="${QUEUE_ROOT}/${method}/step_${step}_seed42"
    local config="${ROOT}/configs/experiments/extend310_${method}_step${step}_full274_seed42.yaml"
    local run_tag="extend310-${method}-step${step}-seed42"

    if is_complete_run "${run_root}" && [[ "$(cat "${run_root}/logs/exit_status" 2>/dev/null)" == "0" ]]; then
      echo "[${method}] Skipping completed step ${step}"
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
      printf '%s\t%s\t%s\t0\tcompleted\n' "${method}" "${step}" "${run_root}" >> "${status_path}"
      echo "[${method}] Completed step ${step}"
    else
      printf '%s\t%s\t%s\t%s\teval-or-validation-failed\n' "${method}" "${step}" "${run_root}" "${eval_status}" >> "${status_path}"
      echo "[${method}] FAILED step ${step}" >&2
      failed=1
    fi
    sleep 10
  done
  return "${failed}"
}

run_lane adaptive_t0100 "${GPU_IDS_ADAPTIVE}" 22006 &
adaptive_pid="$!"
run_lane tcod_f2b "${GPU_IDS_TCOD}" 22007 &
tcod_pid="$!"

set +e
wait "${adaptive_pid}"; adaptive_status="$?"
wait "${tcod_pid}"; tcod_status="$?"
set -e
if (( adaptive_status != 0 || tcod_status != 0 )); then
  echo "Extension full274 queue finished with one or more failures." >&2
  exit 1
fi
echo "All six extension full274 evaluations completed."
