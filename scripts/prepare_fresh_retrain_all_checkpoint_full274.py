#!/usr/bin/env python3
"""Generate frozen full274 configs for the two fresh seed-42 retraining runs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
QUEUE_ROOT = ROOT / "runs" / "experiments" / "fresh_retrain_all_checkpoint_full274_seed42"
EXPORT_ROOT = ROOT / "runs" / "exports" / "fresh_retrain_all_checkpoint_curve"
MANIFEST_ROOT = ROOT / "data" / "eval_manifests"
STEPS = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 250)
METHODS = {
    "fresh_fixed_t0100": {
        "source_root": ROOT
        / "runs/experiments/entropy_adaptive_v1_t0100_fresh_repro_seed42_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_fresh_repro_seed42_250step_4gpu_s1t1_r4",
        "project": "TCOD-FRESH-FIXED-T0100-ALLCKPT-FULL274",
        "checkpoint_prefix": "fresh_fixed_t0100",
    },
    "fresh_cosine_t0200": {
        "source_root": ROOT
        / "runs/experiments/entropy_adaptive_v1_t0100_fresh_cosine_to_t0200_step80_160_seed42_250step_4gpu_s1t1_r4/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_t0100_fresh_cosine_to_t0200_step80_160_seed42_250step_4gpu_s1t1_r4",
        "project": "TCOD-FRESH-COSINE-T0200-ALLCKPT-FULL274",
        "checkpoint_prefix": "fresh_cosine_t0200",
    },
}


def model_path(method: str, step: int) -> Path:
    if step == 250:
        return METHODS[method]["source_root"] / "global_step_250/actor/huggingface"
    return EXPORT_ROOT / method / f"step_{step}"


def render(method: str, step: int) -> str:
    info = METHODS[method]
    label = f"freshallckpt_{method}_step{step}"
    checkpoint_label = f"{info['checkpoint_prefix']}_step{step}"
    run_root = QUEUE_ROOT / method / f"step_{step}_seed42"
    seen_manifest = MANIFEST_ROOT / "full_valid_seen.jsonl"
    unseen_manifest = MANIFEST_ROOT / "full_valid_unseen.jsonl"
    return f'''project: "{info['project']}"
group: "fresh-retrain-all-checkpoint-full274-seed42"
name: "{label}-full274-seed42-h30-4gpu"
checkpoint_root_dir: {run_root / 'trinity_output'}
continue_from_checkpoint: false
mode: bench

algorithm:
  algorithm_type: grpo
  repeat_times: 1
  optimizer:
    lr: 1.0e-6

model:
  model_path: {model_path(method, step)}
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
          result_dir: {run_root / 'task_records'}
          evaluation_id: {label}_full274_seed42
          checkpoint_label: {checkpoint_label}
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
          result_dir: {run_root / 'task_records'}
          evaluation_id: {label}_full274_seed42
          checkpoint_label: {checkpoint_label}
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
    for method in METHODS:
        for step in STEPS:
            path = CONFIG_DIR / f"freshallckpt_{method}_step{step}_full274_seed42.yaml"
            path.write_text(render(method, step), encoding="utf-8")
    print(f"Generated {len(METHODS) * len(STEPS)} frozen full274 configs")


if __name__ == "__main__":
    main()
