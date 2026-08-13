#!/usr/bin/env python3
"""Generate frozen full274 configs for missing checkpoints on three main runs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
RUN_ROOT = ROOT / "runs" / "experiments" / "all_checkpoint_full274_seed42"
EXPORT_ROOT = ROOT / "runs" / "exports" / "all_checkpoint_curve"
MANIFEST_ROOT = ROOT / "data" / "eval_manifests"

MISSING_STEPS = (20, 40, 120, 140, 160, 180, 200, 220, 240)
METHODS = (
    "tcod_f2b",
    "vanilla_opd",
    "entropy_adaptive_v1_t0100",
)


def render(method: str, step: int) -> str:
    label = f"allckpt_{method}_step{step}"
    gpu_count = 4 if (method, step) in {("tcod_f2b", 20), ("vanilla_opd", 20)} else 5
    run_root = RUN_ROOT / method / f"step_{step}_seed42"
    model_path = EXPORT_ROOT / method / f"step_{step}"
    seen_manifest = MANIFEST_ROOT / "full_valid_seen.jsonl"
    unseen_manifest = MANIFEST_ROOT / "full_valid_unseen.jsonl"
    return f'''project: "TCOD-ALL-CHECKPOINT-FULL274"
group: "all-checkpoint-full274-seed42"
name: "{label}-full274-seed42-h30-{gpu_count}gpu"
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
  gpu_per_node: {gpu_count}

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
    engine_num: {gpu_count}
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
    for method in METHODS:
        for step in MISSING_STEPS:
            path = CONFIG_DIR / f"allckpt_{method}_step{step}_full274_seed42.yaml"
            path.write_text(render(method, step), encoding="utf-8")
            count += 1
    print(f"Generated {count} frozen full274 configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
