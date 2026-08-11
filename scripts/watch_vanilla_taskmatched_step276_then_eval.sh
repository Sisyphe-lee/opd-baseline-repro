#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ROOT="${ROOT}/runs/experiments/vanilla_opd_taskmatched_step276_8gpu"
TRAIN_STATUS="${TRAIN_ROOT}/logs/exit_status"
RESUME_ROOT="${TRAIN_ROOT}/resume276_to284"
RESUME_STATUS="${RESUME_ROOT}/logs/exit_status"
JOB_DIR="${TRAIN_ROOT}/checkpoints/ALFWORLD_OPD_TASKMATCH/vanilla_opd_taskmatched_step276_8gpu_s2t4_seed42"
STEP276_CHECKPOINT="${JOB_DIR}/global_step_276"
FINAL_CHECKPOINT="${JOB_DIR}/global_step_284"
EVAL_ROOT="${TRAIN_ROOT}/evaluation/full274_seed42"
CHAIN_LOG="${TRAIN_ROOT}/logs/eval_chain.log"

while [[ ! -f "${TRAIN_STATUS}" ]]; do
  sleep 30
done

train_exit="$(<"${TRAIN_STATUS}")"
if [[ "${train_exit}" != "0" ]]; then
  printf 'Training exited with status %s; full274 was not started.\n' "${train_exit}" >>"${CHAIN_LOG}"
  exit 1
fi

for required in \
  "${STEP276_CHECKPOINT}/.full_checkpoint" \
  "${STEP276_CHECKPOINT}/actor/huggingface/config.json" \
  "${STEP276_CHECKPOINT}/actor/huggingface/model.safetensors.index.json"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing required step-276 artifact: %s\n' "${required}" >>"${CHAIN_LOG}"
    exit 2
  fi
done

printf 'Step 276 validated; launching final continuation to step 284.\n' >>"${CHAIN_LOG}"
bash "${ROOT}/scripts/launch_entropy_experiment_tmux.sh" \
  "${ROOT}/configs/experiments/vanilla_opd_taskmatched_resume276_to284_8gpu.yaml" \
  "0,1,2,3,4,5,6,7" \
  "6395" \
  "${RESUME_ROOT}" \
  "vanilla_taskmatched_resume276_to284_8gpu"

while [[ ! -f "${RESUME_STATUS}" ]]; do
  sleep 30
done

resume_exit="$(<"${RESUME_STATUS}")"
if [[ "${resume_exit}" != "0" ]]; then
  printf 'Step 276-to-284 continuation exited with status %s; full274 was not started.\n' "${resume_exit}" >>"${CHAIN_LOG}"
  exit 3
fi

for required in \
  "${FINAL_CHECKPOINT}/.full_checkpoint" \
  "${FINAL_CHECKPOINT}/actor/huggingface/config.json" \
  "${FINAL_CHECKPOINT}/actor/huggingface/model.safetensors.index.json" \
  "${JOB_DIR}/explorer_meta.json" \
  "${JOB_DIR}/trainer_meta.json"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing required final artifact: %s\n' "${required}" >>"${CHAIN_LOG}"
    exit 4
  fi
done

printf 'Step 284 and checkpoint validation passed; launching frozen full274.\n' >>"${CHAIN_LOG}"
bash "${ROOT}/scripts/launch_experiment_eval_tmux.sh" \
  "${ROOT}/configs/experiments/vanilla_opd_taskmatched_step284_full274.yaml" \
  "0,1,2,3" \
  "vanilla_taskmatched_step284_full274" \
  "6494" \
  "${EVAL_ROOT}" \
  "eval_vanilla_taskmatched_step284_full274"
