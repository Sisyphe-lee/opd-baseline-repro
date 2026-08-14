#!/usr/bin/env python3
"""Counterfactually replay entropy-frontier thresholds on recorded trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trinity.common.workflows.envs.TCOD.alfworld.OPD_entropy_mask_workflow import (
    first_entropy_frontier_turn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories: dict[str, list[dict]] = defaultdict(list)
    with args.diagnostics.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trajectories[row["trajectory_id"]].append(row)

    bins = [(0, 79), (80, 119), (120, 159), (160, 199), (200, 250), (80, 250)]
    results = []
    for lower, upper in bins:
        selected = []
        for rows in trajectories.values():
            rows.sort(key=lambda row: row["turn"])
            version = int(rows[0]["student_model_version"])
            if lower <= version <= upper:
                selected.append(rows)
        for threshold in args.thresholds:
            triggered = 0
            imposed = []
            realized = []
            for rows in selected:
                entropies = [row.get("teacher_entropy_topk") for row in rows]
                frontier = first_entropy_frontier_turn(entropies, threshold, 3, 3)
                retained = len(rows) if frontier is None else max(3, frontier)
                triggered += frontier is not None
                imposed.append(30 if frontier is None else retained)
                realized.append(
                    sum(
                        1
                        for row in rows[:retained]
                        if row.get("truncate_status") != "prompt_truncated"
                    )
                )
            results.append(
                {
                    "version_bin": f"{lower}-{upper}",
                    "threshold": threshold,
                    "trajectory_count": len(selected),
                    "trigger_rate": triggered / len(selected),
                    "mean_imposed_horizon": sum(imposed) / len(imposed),
                    "mean_realized_turns": sum(realized) / len(realized),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
