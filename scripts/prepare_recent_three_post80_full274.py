#!/usr/bin/env python3
"""Generate frozen seed-42 full274 configs for the three post-step-80 branches."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
RUN_ROOT = ROOT / "runs" / "experiments" / "recent_three_post80_full274_seed42"
EXPORT_ROOT = ROOT / "runs" / "exports" / "recent_three_post80_curve"
MANIFEST_ROOT = ROOT / "data" / "eval_manifests"

BRANCHES = {
    "linear_to_full": "linear-to-full",
    "cosine_to_t0200": "cosine-to-t0200",
    "cosine_to_t0175": "cosine-to-t0175",
}
STEPS = (100, 120, 140, 160, 180, 200, 220, 240, 250)


def render(branch: str, step: int) -> str:
    label = f"post80_{branch}_step{step}"
    run_root = RUN_ROOT / branch / f"step_{step}_seed42"
    model_path = EXPORT_ROOT / branch / f"step_{step}"
    seen_manifest = MANIFEST_ROOT / "full_valid_seen.jsonl"
    unseen_manifest = MANIFEST_ROOT / "full_valid_unseen.jsonl"
    return f'''project: "TCOD-POST80-BRANCH-FULL274"
group: "recent-three-post80-full274-seed42"
name: "{label}-full274-seed42-h30-4gpu"
checkpoint_root_dir: {run_root / "trinity_output"}
continue_from_checkpoint: false
mode: bench

algorithm:
  algorithm_type: grpo
  repeat_times: 1
  optimizer:
    lr: 1.0e-6

model:
  model_path: {model_path}
  max_prompt_tokens: 10240
  max_response_tokens: 512

cluster:
  node_num: 1
  gpu_per_node: 4

buffer:
  batch_size: 4
  explorer_input:
    taskset:
      name: unused_train_set
      storage_type: file
      path: {seen_manifest}
      split: train
      format:
        prompt_key: game_file
    eval_tasksets:
      - name: {label}_full_seen
        storage_type: file
        path: {seen_manifest}
        split: test
        format:
          prompt_key: game_file
        rollout_args: &sampling
          temperature: 0.4
          top_p: 1.0
          top_k: -1
          max_tokens: 512
        workflow_args:
          max_env_steps: 30
          accumulate_memory: true
          strict_action_parser: true
          result_dir: {run_root / "task_records"}
          evaluation_id: {label}_full274_seed42
          checkpoint_label: {label}
      - name: {label}_full_unseen
        storage_type: file
        path: {unseen_manifest}
        split: test
        format:
          prompt_key: game_file
        rollout_args: *sampling
        workflow_args:
          max_env_steps: 30
          accumulate_memory: true
          strict_action_parser: true
          result_dir: {run_root / "task_records"}
          evaluation_id: {label}_full274_seed42
          checkpoint_label: {label}
    default_workflow_type: TCOD_eval_alfworld_workflow

explorer:
  eval_on_startup: true
  runner_per_model: 16
  max_timeout: 7200
  rollout_model:
    engine_num: 4
    tensor_parallel_size: 1
    use_v1: false
    enforce_eager: false
    enable_prefix_caching: false
    enable_chunked_prefill: true
    gpu_memory_utilization: 0.45
    dtype: bfloat16
    seed: 42
    enable_thinking: false
  auxiliary_models: []
  bench_on_latest_checkpoint: false

monitor:
  monitor_type: tensorboard

synchronizer:
  sync_method: checkpoint
  sync_style: fixed
  sync_interval: 1
  sync_timeout: 7200
'''


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for branch in BRANCHES:
        for step in STEPS:
            path = CONFIG_DIR / f"post80_{branch}_step{step}_full274_seed42.yaml"
            path.write_text(render(branch, step), encoding="utf-8")
            count += 1
    print(f"Generated {count} frozen full274 configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
