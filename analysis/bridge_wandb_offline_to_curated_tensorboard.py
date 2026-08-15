#!/usr/bin/env python3
"""Incrementally add a small whitelist of live W&B metrics to the curated board."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


TAG_MAP = {
    "actor/final_loss": "02 Training/Actor loss (live)",
    "critic/score/mean": "02 Training/Batch success (%)",
    "rollout/task_success/mean": "02 Training/Rollout success (%)",
    "rollout/kl_divergence/mean": "04 Entropy/Trajectory KL (live)",
    "rollout/entropy_frontier_retained_turns/mean": "03 Curriculum/Retained turns (live)",
    "rollout/if_teacher/mean": "03 Curriculum/Teacher-use fraction (live)",
    "perf/throughput": "05 System/Throughput (tokens/s)",
    "perf/max_memory_allocated_gb": "05 System/GPU memory allocated (GiB)",
}


def canonical_run(display_name: str) -> str:
    if display_name.startswith("entropy_adaptive_v1_t0100_"):
        return "Adaptive_tau_0.100"
    if display_name.startswith("tcod_f2b_"):
        return "TCOD_F2B"
    return "Live_other"


def numeric_json(text: str) -> float | None:
    try:
        number = float(json.loads(text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return number if math.isfinite(number) else None


def item_key(item: Any) -> str:
    return "/".join(item.nested_key) if item.nested_key else item.key


def scan(path: Path) -> tuple[dict[str, str], list[tuple[int, dict[str, float]]]]:
    store = DataStore()
    store.open_for_scan(str(path))
    identity: dict[str, str] = {}
    histories: list[tuple[int, dict[str, float]]] = []
    try:
        while True:
            raw = store.scan_data()
            if raw is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            if record.HasField("run"):
                identity = {"run_id": record.run.run_id, "display_name": record.run.display_name}
            if not record.HasField("history"):
                continue
            step = None
            values: dict[str, float] = {}
            for item in record.history.item:
                key = item_key(item)
                value = numeric_json(item.value_json)
                if value is None:
                    continue
                if key == "_step":
                    step = int(value)
                elif key in TAG_MAP:
                    if key in {"critic/score/mean", "rollout/task_success/mean"}:
                        value *= 100.0
                    values[TAG_MAP[key]] = value
            if step is None and record.history.HasField("step"):
                step = int(record.history.step.num)
            if step is not None and values:
                histories.append((step, values))
    except Exception:
        pass  # The final record of a live file may be incomplete.
    return identity, histories


def load_state(path: Path) -> dict[str, int]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict[str, int]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bridge_once(wandb_root: Path, output: Path, run_glob: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "live_bridge_state.json"
    state = {key: int(value) for key, value in load_state(state_path).items()}
    result: dict[str, Any] = {"runs": 0, "rows_added": 0, "latest_steps": {}}
    for source in sorted(wandb_root.glob(run_glob)):
        identity, histories = scan(source)
        if not identity:
            continue
        result["runs"] += 1
        run_id = identity["run_id"]
        previous = state.get(run_id, -1)
        fresh = [(step, values) for step, values in histories if step > previous]
        if fresh:
            writer = SummaryWriter(str(output / canonical_run(identity["display_name"])))
            for step, values in fresh:
                for tag, value in values.items():
                    writer.add_scalar(tag, value, step)
                result["rows_added"] += 1
            writer.flush()
            writer.close()
            state[run_id] = max(step for step, _ in fresh)
        result["latest_steps"][identity["display_name"]] = state.get(run_id, previous)
    save_state(state_path, state)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-root", type=Path, default=Path("wandb"))
    parser.add_argument("--output", type=Path, default=Path("analysis/tensorboard_curated"))
    parser.add_argument("--run-glob", default="offline-run-20260814_0614*/run-*.wandb")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    while True:
        result = bridge_once(args.wandb_root.resolve(), args.output.resolve(), args.run_glob)
        print(json.dumps(result, sort_keys=True), flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
