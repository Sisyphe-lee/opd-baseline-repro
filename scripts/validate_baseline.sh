#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

[[ -f .asset_copy_completed ]] || fail "asset copy marker missing"
[[ -f .relocation_completed ]] || fail "relocation marker missing"
pass "copy and relocation markers"

[[ $(wc -l < data/tcod_official_alfworld/train_expert.jsonl) -eq 3553 ]] || fail "train count"
[[ $(wc -l < data/eval_manifests/full_valid_seen.jsonl) -eq 140 ]] || fail "seen count"
[[ $(wc -l < data/eval_manifests/full_valid_unseen.jsonl) -eq 134 ]] || fail "unseen count"
pass "dataset counts 3553/140/134"

for checkpoint in checkpoints/tcod_f2b_step250/global_step_250 checkpoints/vanilla_opd_step250/global_step_250; do
  [[ -f "${checkpoint}/.full_checkpoint" ]] || fail "missing full marker: ${checkpoint}"
  [[ -f "${checkpoint}/actor/huggingface/model-00001-of-00002.safetensors" ]] || fail "missing HF shard: ${checkpoint}"
  [[ -f "${checkpoint}/actor/optim_world_size_2_rank_0.pt" ]] || fail "missing optimizer shard: ${checkpoint}"
done
pass "complete TCOD and Vanilla step-250 checkpoints"

python_bin=.venv_tcod/bin/python
[[ -x "${python_bin}" ]] || fail "relocated Python missing"
"${python_bin}" - <<'PY'
import json
from pathlib import Path
import yaml

root = Path.cwd()
for manifest in (
    root / "data/tcod_official_alfworld/train_expert.jsonl",
    root / "data/eval_manifests/full_valid_seen.jsonl",
    root / "data/eval_manifests/full_valid_unseen.jsonl",
):
    for line in manifest.read_text(encoding="utf-8").splitlines():
        game = Path(json.loads(line)["game_file"])
        if not game.is_file():
            raise SystemExit(f"missing game file: {game}")

for name in ("tcod_f2b", "vanilla_opd"):
    cfg = yaml.safe_load((root / f"configs/train/{name}.yaml").read_text())
    assert cfg["buffer"]["total_steps"] == 250
    assert cfg["buffer"]["train_batch_size"] == 64
    assert cfg["algorithm"]["loss_agg_mode"] == "seq-mean-token-mean"
    assert cfg["model"]["max_response_tokens"] == 512

for name in ("tcod_f2b_step250_full274", "vanilla_opd_step250_full274"):
    cfg = yaml.safe_load((root / f"configs/eval/{name}.yaml").read_text())
    assert cfg["model"]["max_response_tokens"] == 512
    assert cfg["cluster"]["gpu_per_node"] == 4
    for task in cfg["buffer"]["explorer_input"]["eval_tasksets"]:
        args = task["workflow_args"]
        assert args["max_env_steps"] == 30
        assert args["accumulate_memory"] is True
        assert args["strict_action_parser"] is True

from trinity.common.workflows import WORKFLOWS
for workflow_name in (
    "TCOD_f2b_alfworld_workflow",
    "OPD_promptfix_alfworld_workflow",
    "TCOD_eval_alfworld_workflow",
):
    assert WORKFLOWS.get(workflow_name) is not None
print("Python/YAML/workflow validation passed")
PY
pass "paths, configs, and workflow registration"

if rg -n '/lcy/|opd-alfworld-sync-repro|tcod-f2b-repro' \
  configs/train/tcod_f2b.yaml configs/train/vanilla_opd.yaml \
  configs/eval/tcod_f2b_step250_full274.yaml configs/eval/vanilla_opd_step250_full274.yaml \
  data/tcod_official_alfworld data/eval_manifests \
  .venv_tcod/lib/python3.10/site-packages/__editable___trinity_rft_0_5_0_dev0_finder.py; then
  fail "active runtime still references a source directory"
fi
pass "active runtime is independent of source directories"

"${python_bin}" - <<'PY'
import json
from pathlib import Path
expected = {
    "results/evaluations/2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory-strict/evaluation/full274_h30/summary.json": 232,
    "results/evaluations/2026-08-09_vanilla-opd-qwen25-3b-step250-full274-h30-accmemory-strict/evaluation/full274_h30/summary.json": 218,
}
for path, successes in expected.items():
    data = json.loads(Path(path).read_text())
    assert data["task_count"] == 274
    assert data["success_count"] == successes
print("Frozen result validation passed")
PY
pass "frozen headline results 232/274 and 218/274"

[[ -f validation/live/LIVE_SMOKE_COMPLETED ]] || fail "live smoke marker missing"
"${python_bin}" - <<'PY'
import json
from pathlib import Path
for mode, label in (("tcod", "tcod_f2b_step250"), ("vanilla", "vanilla_opd_step250")):
    summary = json.loads(Path(f"validation/live/{mode}/summary.json").read_text())
    assert summary["task_count"] == 2
    assert summary["checkpoint_label"] == label
    assert summary["splits"]["seen"]["task_count"] == 1
    assert summary["splits"]["unseen"]["task_count"] == 1
print("Live smoke validation passed")
PY
pass "live TCOD and Vanilla inference on one Seen plus one Unseen task"

[[ $(wc -l < validation/MODEL_CHECKPOINT_SHA256SUMS) -eq 122 ]] || fail "artifact checksum count"
rg -q '^SHA256_PASSED$' validation/bytecheck.log || fail "bytecheck completion marker"
pass "122-file model/checkpoint SHA256 manifest and source bytecheck"

echo "BASELINE VALIDATION PASSED"
