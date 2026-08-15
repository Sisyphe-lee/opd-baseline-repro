#!/usr/bin/env python3
"""Append completed fresh-retraining full274 percentages to the curated TensorBoard.

The three tag names intentionally match the historical dashboard so old and
new evaluation curves share the same TensorBoard cards. Raw success counts
remain in the CSV and source ``summary.json`` files but are not emitted as
TensorBoard scalars.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = ROOT / "runs" / "experiments" / "fresh_retrain_all_checkpoint_full274_seed42"
REPORT_ROOT = ROOT / "analysis" / "fresh_retrain_all_checkpoint_full274_seed42"
TENSORBOARD_ROOT = ROOT / "analysis" / "tensorboard_curated"
STEPS = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 250)
METHODS = {
    "fresh_fixed_t0100": "Fresh_repro_tau_0.100_full274",
    "fresh_cosine_t0200": "Fresh_cosine_0.100_to_0.200_full274",
}
TENSORBOARD_SCHEMA_VERSION = 2
EVALUATION_TAGS = {
    "overall": "01 Evaluation/Overall success (%)",
    "seen": "01 Evaluation/Seen success (%)",
    "unseen": "01 Evaluation/Unseen success (%)",
}


def load_rows(method: str) -> list[dict]:
    rows = []
    for step in STEPS:
        path = QUEUE_ROOT / method / f"step_{step}_seed42/summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("task_count") != 274:
            raise ValueError(f"Expected 274 tasks: {path}")
        seen = summary["splits"]["seen"]
        unseen = summary["splits"]["unseen"]
        if seen.get("task_count") != 140 or unseen.get("task_count") != 134:
            raise ValueError(f"Invalid split counts: {path}")
        rows.append(
            {
                "method": method,
                "step": step,
                "success_count": summary["success_count"],
                "success_rate_percent": 100.0 * summary["success_rate"],
                "seen_success_count": seen["success_count"],
                "seen_success_rate_percent": 100.0 * seen["success_rate"],
                "unseen_success_count": unseen["success_count"],
                "unseen_success_rate_percent": 100.0 * unseen["success_rate"],
                "summary": str(path.relative_to(ROOT)),
            }
        )
    return rows


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def export(method: str) -> dict:
    rows = load_rows(method)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_ROOT / f"{method}.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    if rows:
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary_csv.replace(csv_path)

    run_dir = TENSORBOARD_ROOT / METHODS[method]
    state_path = run_dir / "full274_export_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    archived = None
    if run_dir.exists() and state.get("schema_version") != TENSORBOARD_SCHEMA_VERSION:
        archive_root = REPORT_ROOT / "tensorboard_archives"
        archive_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = archive_root / f"{METHODS[method]}_before_schema_v{TENSORBOARD_SCHEMA_VERSION}_{timestamp}"
        shutil.move(str(run_dir), str(archived))
        state = {}
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "full274_export_state.json"
    completed = {int(step) for step in state.get("steps", [])}
    fresh = [row for row in rows if row["step"] not in completed]
    if fresh:
        writer = SummaryWriter(str(run_dir), flush_secs=1)
        for row in fresh:
            step = row["step"]
            writer.add_scalar(EVALUATION_TAGS["overall"], row["success_rate_percent"], step)
            writer.add_scalar(EVALUATION_TAGS["seen"], row["seen_success_rate_percent"], step)
            writer.add_scalar(EVALUATION_TAGS["unseen"], row["unseen_success_rate_percent"], step)
        writer.flush()
        writer.close()
        completed.update(row["step"] for row in fresh)
        atomic_json(
            state_path,
            {
                "method": method,
                "schema_version": TENSORBOARD_SCHEMA_VERSION,
                "steps": sorted(completed),
                "scalar_tags": list(EVALUATION_TAGS.values()),
            },
        )
    return {
        "method": method,
        "rows": len(rows),
        "new_tensorboard_points": len(fresh),
        "archived_previous_run": str(archived) if archived else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.method), sort_keys=True))


if __name__ == "__main__":
    main()
