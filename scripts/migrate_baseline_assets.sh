#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${ROOT}/.." && pwd)"
OPD_SOURCE="${WORKSPACE}/opd-alfworld-sync-repro"
TCOD_SOURCE="${WORKSPACE}/tcod-f2b-repro"

copy_tree() {
  local source="$1"
  local destination="$2"
  if [[ ! -d "${source}" ]]; then
    echo "Missing source directory: ${source}" >&2
    exit 2
  fi
  mkdir -p "${destination}"
  cp -a "${source}/." "${destination}/"
}

copy_file() {
  local source="$1"
  local destination="$2"
  if [[ ! -f "${source}" ]]; then
    echo "Missing source file: ${source}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${destination}")"
  cp -a "${source}" "${destination}"
}

echo "[1/6] Models"
copy_tree "${OPD_SOURCE}/models/Qwen2.5-3B-Instruct" "${ROOT}/models/Qwen2.5-3B-Instruct"
copy_tree "${OPD_SOURCE}/models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld" "${ROOT}/models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld"

echo "[2/6] ALFWorld data"
copy_tree "${OPD_SOURCE}/data/alfworld_runtime" "${ROOT}/data/alfworld_runtime"
copy_tree "${OPD_SOURCE}/data/alfworld" "${ROOT}/data/alfworld"
copy_tree "${OPD_SOURCE}/data/tcod_official_alfworld" "${ROOT}/data/tcod_official_alfworld"

echo "[3/6] Python environment"
copy_tree "${TCOD_SOURCE}/.venv_tcod" "${ROOT}/.venv_tcod"

echo "[4/6] Complete step-250 checkpoints"
TCOD_CKPT_SOURCE="${OPD_SOURCE}/runs/2026-08-07_tcod-f2b-qwen25-3b-7b-paper-promptfix-4gpu-1s1t2train-v0/checkpoints/ALFWORLD_TCOD_REPRO/f2b_qwen25_3b_7b_eta2_promptfix_4gpu_1s1t2train_v0_20260807131207"
VANILLA_CKPT_SOURCE="${OPD_SOURCE}/runs/2026-08-08_vanilla-opd-qwen25-3b-7b-paper-promptfix-4gpu-restart/checkpoints/ALFWORLD_OPD_REPRO/vanilla_opd_qwen25_3b_7b_promptfix_4gpu_1s1t2train_restart_20260808"
copy_tree "${TCOD_CKPT_SOURCE}/global_step_250" "${ROOT}/checkpoints/tcod_f2b_step250/global_step_250"
copy_tree "${TCOD_CKPT_SOURCE}/buffer" "${ROOT}/checkpoints/tcod_f2b_step250/buffer"
copy_file "${TCOD_CKPT_SOURCE}/explorer_meta.json" "${ROOT}/checkpoints/tcod_f2b_step250/explorer_meta.json"
copy_file "${TCOD_CKPT_SOURCE}/trainer_meta.json" "${ROOT}/checkpoints/tcod_f2b_step250/trainer_meta.json"
copy_file "${TCOD_CKPT_SOURCE}/latest_checkpointed_iteration.txt" "${ROOT}/checkpoints/tcod_f2b_step250/latest_checkpointed_iteration.txt"
copy_file "${TCOD_CKPT_SOURCE}/latest_state_dict_iteration.txt" "${ROOT}/checkpoints/tcod_f2b_step250/latest_state_dict_iteration.txt"
copy_tree "${VANILLA_CKPT_SOURCE}/global_step_250" "${ROOT}/checkpoints/vanilla_opd_step250/global_step_250"
copy_tree "${VANILLA_CKPT_SOURCE}/buffer" "${ROOT}/checkpoints/vanilla_opd_step250/buffer"
copy_file "${VANILLA_CKPT_SOURCE}/explorer_meta.json" "${ROOT}/checkpoints/vanilla_opd_step250/explorer_meta.json"
copy_file "${VANILLA_CKPT_SOURCE}/trainer_meta.json" "${ROOT}/checkpoints/vanilla_opd_step250/trainer_meta.json"
copy_file "${VANILLA_CKPT_SOURCE}/latest_checkpointed_iteration.txt" "${ROOT}/checkpoints/vanilla_opd_step250/latest_checkpointed_iteration.txt"
copy_file "${VANILLA_CKPT_SOURCE}/latest_state_dict_iteration.txt" "${ROOT}/checkpoints/vanilla_opd_step250/latest_state_dict_iteration.txt"

echo "[5/6] Training logs and frozen evaluation evidence"
copy_tree "${OPD_SOURCE}/runs/2026-08-07_tcod-f2b-qwen25-3b-7b-paper-promptfix-4gpu-1s1t2train-v0/launcher_logs" "${ROOT}/results/training/tcod_f2b_step250/launcher_logs"
copy_tree "${OPD_SOURCE}/runs/2026-08-08_vanilla-opd-qwen25-3b-7b-paper-promptfix-4gpu-restart/launcher_logs" "${ROOT}/results/training/vanilla_opd_step250/launcher_logs"

eval_runs=(
  2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory-strict
  2026-08-09_vanilla-opd-qwen25-3b-step250-full274-h30-accmemory-strict
  2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory
  2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory-strict-r4096
  2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-paper
  2026-08-07_gigpo-qwen2p5-7b-native-h50-historyfix
  2026-08-07_gigpo-qwen2p5-7b-native-h50
  2026-08-07_gigpo-qwen2p5-7b-rl-paper-eval
  2026-08-07_gigpo-official-native-val
  2026-08-07_qwen2p5-3b-instruct-init-full274-h50-historyfix
)
for run_name in "${eval_runs[@]}"; do
  copy_tree "${OPD_SOURCE}/runs/${run_name}" "${ROOT}/results/evaluations/${run_name}"
done

echo "[6/6] Completion marker"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "${ROOT}/.asset_copy_completed"
echo "Asset copy complete: ${ROOT}"
