#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${ROOT}/.venv_tcod/bin/python" \
  "${ROOT}/scripts/plot_baseline_reproduction.py" \
  --tcod "${ROOT}/results/evaluations/2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory-strict/evaluation/full274_h30/task_results.jsonl" \
  --vanilla "${ROOT}/results/evaluations/2026-08-09_vanilla-opd-qwen25-3b-step250-full274-h30-accmemory-strict/evaluation/full274_h30/task_results.jsonl" \
  --tcod-train-log "${ROOT}/results/training/tcod_f2b_step250/launcher_logs/f2b_resume80_20260808T0615Z.log" \
  --vanilla-train-log "${ROOT}/results/training/vanilla_opd_step250/launcher_logs/vanilla_restart_20260808T0615Z.log" \
  --vanilla-train-log "${ROOT}/results/training/vanilla_opd_step250/launcher_logs/vanilla_resume100_20260808.log" \
  --vanilla-train-log "${ROOT}/results/training/vanilla_opd_step250/launcher_logs/vanilla_resume120_20260809.log" \
  --output-dir "${ROOT}/analysis/frozen_full274_reproduction"
