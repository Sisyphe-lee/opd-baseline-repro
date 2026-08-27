#!/usr/bin/env bash
# All-checkpoint full274 evaluation queue for the 2026-08-16 B200 reruns.
# Usage: b200_allckpt_eval_queue.sh RUN_KEY GPU_IDS RAY_PORT_BASE
#   RUN_KEY in {tcod158, vanilla158, adaptive42, adaptive43}
# Evaluates global_step_{20..240 by 20} (step 250 already evaluated separately).
# Resumable: a step with a complete summary.json (274 tasks) is skipped.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_KEY="$1"; GPU_IDS="$2"; PORT_BASE="$3"

case "${RUN_KEY}" in
  tcod158)
    ACTOR_ROOT="${ROOT}/runs/ckpt_store/training/tcod_f2b/checkpoints/ALFWORLD_TCOD_REPRO/f2b_qwen25_3b_7b_eta2_promptfix_4gpu_1s1t2train_v0"
    TEMPLATE="${ROOT}/configs/experiments/b158_repro_tcod_f2b_step250_full274.yaml" ;;
  vanilla158)
    ACTOR_ROOT="${ROOT}/runs/ckpt_store/training/vanilla_opd/checkpoints/ALFWORLD_OPD_REPRO/vanilla_opd_qwen25_3b_7b_promptfix_4gpu_1s1t2train_restart_20260808"
    TEMPLATE="${ROOT}/configs/experiments/b158_repro_vanilla_opd_step250_full274.yaml" ;;
  adaptive42)
    ACTOR_ROOT="${ROOT}/runs/ckpt_store/experiments/entropy_adaptive_v1_t0100_b203_repro_seed42_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_b203_repro_seed42_250step_4gpu_s1t1_r4"
    TEMPLATE="${ROOT}/configs/experiments/entropy_adaptive_v1_t0100_b203_seed42_step250_full274.yaml" ;;
  adaptive43)
    ACTOR_ROOT="${ROOT}/runs/ckpt_store/experiments/entropy_adaptive_v1_t0100_b203_repro_seed43_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_b203_repro_seed43_250step_4gpu_s1t1_r4"
    TEMPLATE="${ROOT}/configs/experiments/entropy_adaptive_v1_t0100_b203_seed43_step250_full274.yaml" ;;
  *) echo "Unknown RUN_KEY: ${RUN_KEY}" >&2; exit 2 ;;
esac

EXPORT_ROOT="${ROOT}/runs/ckpt_store/exports_allckpt/${RUN_KEY}"
EVAL_ROOT="${ROOT}/runs/ckpt_store/evaluation/allckpt/${RUN_KEY}"
CFG_DIR="${ROOT}/configs/experiments/allckpt_b200/${RUN_KEY}"
STATUS_CSV="${EVAL_ROOT}/queue_status.csv"
mkdir -p "${EXPORT_ROOT}" "${EVAL_ROOT}/logs" "${CFG_DIR}"
[[ -f "${STATUS_CSV}" ]] || echo "run,step,success_count,task_count,note" > "${STATUS_CSV}"

PY="${ROOT}/.venv_tcod/bin/python"
idx=0
overall=0
for STEP in 20 40 60 80 100 120 140 160 180 200 220 240; do
  idx=$((idx + 1))
  step_root="${EVAL_ROOT}/step_${STEP}"
  summary="${step_root}/summary.json"
  if [[ -s "${summary}" ]] && "${PY}" -c "import json,sys; sys.exit(0 if json.load(open('${summary}'))['task_count']==274 else 1)" 2>/dev/null; then
    echo "[${RUN_KEY}] step ${STEP} already complete, skipping"
    continue
  fi

  export_dir="${EXPORT_ROOT}/step_${STEP}"
  if ! { [[ -s "${export_dir}/model.safetensors.index.json" ]] && compgen -G "${export_dir}/model-*.safetensors" >/dev/null; }; then
    src="${ACTOR_ROOT}/global_step_${STEP}/actor"
    if [[ ! -s "${src}/model_world_size_2_rank_0.pt" ]]; then
      echo "[${RUN_KEY}] missing checkpoint for step ${STEP}: ${src}" >&2
      echo "${RUN_KEY},${STEP},,,missing_checkpoint" >> "${STATUS_CSV}"; overall=1; continue
    fi
    echo "[${RUN_KEY}] exporting step ${STEP}"
    "${PY}" -m verl.model_merger merge --backend fsdp --use_cpu_initialization \
      --local_dir "${src}" --target_dir "${export_dir}.partial" \
      > "${EVAL_ROOT}/logs/export_step${STEP}.log" 2>&1
    if [[ ! -s "${export_dir}.partial/model.safetensors.index.json" ]]; then
      echo "${RUN_KEY},${STEP},,,export_failed" >> "${STATUS_CSV}"; overall=1; continue
    fi
    # exports lack tokenizer files; copy them from the run's final HF export
    final_hf="${ACTOR_ROOT}/global_step_250/actor/huggingface"
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json generation_config.json chat_template.jinja; do
      [[ -e "${export_dir}.partial/${f}" ]] || cp -n "${final_hf}/${f}" "${export_dir}.partial/" 2>/dev/null
    done
    mv "${export_dir}.partial" "${export_dir}"
  fi

  mkdir -p "${step_root}/task_records"
  cfg="${CFG_DIR}/step_${STEP}.yaml"
  "${PY}" "${ROOT}/scripts/make_allckpt_eval_config.py" --template "${TEMPLATE}" \
    --output "${cfg}" --model-path "${export_dir}" --eval-root "${step_root}" --step "${STEP}"

  port=$((PORT_BASE + idx))
  echo "[${RUN_KEY}] evaluating step ${STEP} on GPUs ${GPU_IDS} port ${port}"
  bash "${ROOT}/scripts/run_alfworld_eval.sh" "${cfg}" "${GPU_IDS}" "allckpt-${RUN_KEY}-step${STEP}" "${port}" \
    > "${EVAL_ROOT}/logs/eval_step${STEP}.log" 2>&1
  eval_status=$?
  if (( eval_status != 0 )); then
    echo "${RUN_KEY},${STEP},,,eval_failed_${eval_status}" >> "${STATUS_CSV}"; overall=1; continue
  fi
  "${PY}" "${ROOT}/scripts/collect_alfworld_eval_results.py" \
    --manifest "${ROOT}/data/eval_manifests/full_valid_seen.jsonl" \
    --manifest "${ROOT}/data/eval_manifests/full_valid_unseen.jsonl" \
    --record-dir "${step_root}/task_records" \
    --output-jsonl "${step_root}/task_results.jsonl" \
    --summary-json "${summary}" --expected-count 274 \
    > "${EVAL_ROOT}/logs/collect_step${STEP}.log" 2>&1
  if [[ -s "${summary}" ]]; then
    "${PY}" -c "
import json; s=json.load(open('${summary}'))
print('${RUN_KEY},${STEP},%d,%d,ok' % (s['success_count'], s['task_count']))" >> "${STATUS_CSV}"
    echo "[${RUN_KEY}] step ${STEP} done: $(tail -1 "${STATUS_CSV}")"
  else
    echo "${RUN_KEY},${STEP},,,collect_failed" >> "${STATUS_CSV}"; overall=1
  fi
done
echo "[${RUN_KEY}] queue finished (overall=${overall})"
exit "${overall}"
