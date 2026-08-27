#!/usr/bin/env bash
set -euo pipefail

LEGACY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_ROOT="$(cd "${LEGACY_ROOT}/../.." && pwd)"
exec "${BASELINE_ROOT}/.venv_tcod/bin/python" \
  "${LEGACY_ROOT}/scripts/plot_vanilla_opd_diagnostics.py" "$@"
