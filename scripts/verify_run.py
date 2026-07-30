#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def newest_smoke(project_root: Path) -> Path:
    smoke_root = project_root / "artifacts" / "response_only_smoke"
    candidates = sorted(
        (path for path in smoke_root.glob("*") if (path / "train.log").is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No smoke runs found under {smoke_root}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the three-step OPD smoke run.")
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve() if args.run_dir else newest_smoke(project_root)
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log: {log_path}")

    log = log_path.read_text(errors="replace")
    # Ray may emit a DataLoader-worker traceback while tearing down a successful run.
    # Treat concrete training failures as fatal and require positive step/checkpoint evidence.
    fatal_markers = ("CUDA out of memory", "TypeError:")
    for marker in fatal_markers:
        if marker in log:
            print(f"FAIL: fatal marker in log: {marker}", file=sys.stderr)
            return 1

    success_markers = ("Training completed successfully.", "Smoke completed successfully.")
    if not any(marker in log for marker in success_markers):
        print("FAIL: success marker is absent", file=sys.stderr)
        return 1

    observed_steps = {int(step) for step in re.findall(r"training/global_step['\"]?:\s*([0-9]+)", log)}
    missing_steps = {1, 2, 3} - observed_steps
    if missing_steps:
        print(f"FAIL: missing optimizer steps: {sorted(missing_steps)}", file=sys.stderr)
        return 1

    actor_dir = run_dir / "checkpoint" / "global_step_3" / "actor"
    model_shards = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    optimizer_shards = sorted(actor_dir.glob("optim_world_size_*_rank_*.pt"))
    if not model_shards or len(model_shards) != len(optimizer_shards):
        print(
            f"FAIL: incomplete checkpoint: model_shards={len(model_shards)}, "
            f"optimizer_shards={len(optimizer_shards)}",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: validated steps 1, 2, 3 and {len(model_shards)} checkpoint ranks")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
