#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_REPO="/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/tcod-f2b-repro"
OLD_VENV="${OLD_REPO}/.venv_tcod"
OLD_OPD="/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/opd-alfworld-sync-repro"
OLD_LCY="/vepfs-mlp2/mlp-public/252302025/lcy/tcod-f2b-repro"

[[ -f "${ROOT}/.asset_copy_completed" ]] || { echo "Asset copy has not completed." >&2; exit 2; }

while IFS= read -r file; do
  perl -pi -e "s|\Q${OLD_VENV}\E|${ROOT}/.venv_tcod|g; s|\Q${OLD_REPO}\E|${ROOT}|g" "${file}"
done < <(grep -Ilr --exclude='*.pyc' "${OLD_REPO}" "${ROOT}/.venv_tcod/bin" "${ROOT}/.venv_tcod/lib/python3.10/site-packages" 2>/dev/null || true)

perl -pi -e "s|\Q${OLD_OPD}/data/alfworld_runtime\E|${ROOT}/data/alfworld_runtime|g" \
  "${ROOT}/data/tcod_official_alfworld/"*.jsonl
perl -pi -e "s|\Q${OLD_LCY}/data/alfworld_runtime\E|${ROOT}/data/alfworld_runtime|g" \
  "${ROOT}/data/eval_manifests/"*.jsonl

date -u '+%Y-%m-%dT%H:%M:%SZ' > "${ROOT}/.relocation_completed"
echo "Relocation complete."

