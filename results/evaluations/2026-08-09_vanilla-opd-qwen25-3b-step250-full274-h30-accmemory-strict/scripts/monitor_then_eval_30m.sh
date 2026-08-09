#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/opd-alfworld-sync-repro
TRAIN_ROOT=${REPO_ROOT}/runs/2026-08-08_vanilla-opd-qwen25-3b-7b-paper-promptfix-4gpu-restart/checkpoints/ALFWORLD_OPD_REPRO/vanilla_opd_qwen25_3b_7b_promptfix_4gpu_1s1t2train_restart_20260808
TRAIN_LOG=${TRAIN_ROOT}/log/trainer.log
TRAIN_CONFIG=${REPO_ROOT}/configs/vanilla_opd_qwen25_3b_7b_paper_promptfix_4gpu_resume120_20260809.yaml
CHECKPOINT=${TRAIN_ROOT}/global_step_250
EVAL_LAUNCHER=${RUN_ROOT}/scripts/eval_full274_h30_accmemory_strict_4gpu.sh
EVAL_SESSION=vanilla_full274_frozen_20260809
MONITOR_LOG=${RUN_ROOT}/logs/monitor_30m.log
INTERVAL_SECONDS=${INTERVAL_SECONDS:-1800}

latest_step() {
  if [[ ! -f ${TRAIN_LOG} ]]; then
    echo 0
    return
  fi
  sed -nE 's/.*Training at step ([0-9]+) finished.*/\1/p' "${TRAIN_LOG}" | sort -n | tail -1
}

training_running() {
  ps -eo args= | grep -F "trinity run --config ${TRAIN_CONFIG}" | grep -qv grep
}

gpus_free() {
  local gpu_id gpu_pids
  for gpu_id in 0 1 2 3; do
    gpu_pids=$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
    if [[ -n ${gpu_pids} ]]; then
      return 1
    fi
  done
  return 0
}

while true; do
  timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  step=$(latest_step)
  checkpoint_ready=false
  process=stopped
  gpu_state=busy
  eval_state=not_started

  if [[ -f ${CHECKPOINT}/.full_checkpoint && -d ${CHECKPOINT}/actor/huggingface ]]; then
    checkpoint_ready=true
  fi
  if training_running; then
    process=running
  fi
  if gpus_free; then
    gpu_state=free
  fi
  if tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; then
    eval_state=running
  elif [[ -f ${RUN_ROOT}/EVAL_COMPLETED ]]; then
    eval_state=completed
  elif [[ -f ${RUN_ROOT}/EVAL_FAILED ]]; then
    eval_state=failed
  elif [[ -f ${RUN_ROOT}/EVAL_STARTED ]]; then
    eval_state=stopped_without_completion
  fi

  echo "${timestamp} trainer_step=${step:-0} process=${process} checkpoint250=${checkpoint_ready} gpus0_3=${gpu_state} eval=${eval_state}" | tee -a "${MONITOR_LOG}"

  if [[ ${eval_state} == completed ]]; then
    echo "${timestamp} MONITOR_COMPLETED" | tee -a "${MONITOR_LOG}"
    exit 0
  fi
  if [[ ${eval_state} == failed || ${eval_state} == stopped_without_completion ]]; then
    echo "${timestamp} MONITOR_NEEDS_ATTENTION eval=${eval_state}" | tee -a "${MONITOR_LOG}"
    exit 1
  fi
  if [[ ${checkpoint_ready} == true && ${process} == stopped && ${gpu_state} == free && ${eval_state} == not_started ]]; then
    tmux new-session -d -s "${EVAL_SESSION}" \
      "cd '${REPO_ROOT}' && exec bash '${EVAL_LAUNCHER}' 0,1,2,3 6706"
    echo "${timestamp} EVAL_LAUNCHED tmux=${EVAL_SESSION}" | tee -a "${MONITOR_LOG}"
  fi

  sleep "${INTERVAL_SECONDS}"
done
