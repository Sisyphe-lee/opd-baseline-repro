#!/usr/bin/env python3
"""Incrementally mirror selected offline W&B scalar histories to TensorBoard."""

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


def safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._=+-" else "_" for char in value).strip("_") or "unnamed"


def numeric_json(text: str) -> float | None:
    try:
        value = json.loads(text)
        number = float(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return number if math.isfinite(number) else None


def item_key(item: Any) -> str:
    if item.nested_key:
        return "/".join(item.nested_key)
    return item.key


def scan_run(path: Path) -> tuple[dict[str, str], list[tuple[int, dict[str, float]]]]:
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
                identity = {
                    "run_id": record.run.run_id,
                    "project": record.run.project,
                    "display_name": record.run.display_name or record.run.run_id,
                    "run_group": record.run.run_group,
                }
            if not record.HasField("history"):
                continue
            values: dict[str, float] = {}
            logical_step: int | None = None
            for item in record.history.item:
                key = item_key(item)
                number = numeric_json(item.value_json)
                if number is None:
                    continue
                if key == "_step":
                    logical_step = int(number)
                elif not key.startswith("_"):
                    values[key] = number
            if logical_step is None and record.history.HasField("step"):
                logical_step = int(record.history.step.num)
            if logical_step is not None and values:
                histories.append((logical_step, values))
    except Exception:
        # A live file may end in one partial record; earlier records are valid.
        pass
    return identity, histories


def load_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return {key: int(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


def save_state(path: Path, state: dict[str, int]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def bridge_once(wandb_root: Path, output: Path, name_contains: str, run_glob: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "bridge_state.json"
    state = load_state(state_path)
    result: dict[str, Any] = {"runs_seen": 0, "runs_matched": 0, "rows_added": 0, "latest_steps": {}}
    for path in sorted(wandb_root.glob(run_glob)):
        result["runs_seen"] += 1
        identity, histories = scan_run(path)
        display_name = identity.get("display_name", "")
        if not identity or name_contains not in display_name:
            continue
        result["runs_matched"] += 1
        run_id = identity["run_id"]
        previous_step = state.get(run_id, -1)
        fresh = [(step, values) for step, values in histories if step > previous_step]
        if fresh:
            writer = SummaryWriter(str(output / safe_component(display_name)))
            for step, values in fresh:
                for tag, value in values.items():
                    writer.add_scalar(tag, value, step)
                result["rows_added"] += 1
            writer.add_text("_provenance/wandb_source", f"`{path}`", 0)
            writer.flush()
            writer.close()
            state[run_id] = max(step for step, _ in fresh)
        result["latest_steps"][display_name] = state.get(run_id, previous_step)
    save_state(state_path, state)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-root", type=Path, default=Path("wandb"))
    parser.add_argument("--output", type=Path, default=Path("analysis/tensorboard_live_training"))
    parser.add_argument("--name-contains", default="extend_step250_to310")
    parser.add_argument("--run-glob", default="offline-run-*/run-*.wandb")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    while True:
        result = bridge_once(args.wandb_root.resolve(), args.output.resolve(), args.name_contains, args.run_glob)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
