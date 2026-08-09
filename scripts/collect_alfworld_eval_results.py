#!/usr/bin/env python3
"""Validate and merge ALFWorld evaluation task shards in manifest order."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Iterable


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def load_task_shards(record_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(record_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        game_file = record.get("game_file")
        if not game_file:
            raise ValueError(f"Task shard has no game_file: {path}")
        if game_file in records:
            raise ValueError(f"Duplicate task shard for {game_file}")
        records[game_file] = record
    return records


def validate_and_order(
    manifest_rows: Iterable[dict], records: dict[str, dict]
) -> list[dict]:
    expected_rows = list(manifest_rows)
    expected_games = [row["game_file"] for row in expected_rows]
    if len(expected_games) != len(set(expected_games)):
        raise ValueError("Evaluation manifests contain duplicate game_file entries")

    expected_set = set(expected_games)
    missing = [game for game in expected_games if game not in records]
    unexpected = sorted(set(records) - expected_set)
    if missing or unexpected:
        raise ValueError(
            f"Task coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}; "
            f"first_missing={missing[:3]}, first_unexpected={unexpected[:3]}"
        )

    ordered = []
    for manifest_row in expected_rows:
        record = records[manifest_row["game_file"]]
        for key in ("split", "task_type"):
            expected_value = manifest_row.get(key)
            actual_value = record.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f"{key} mismatch for {manifest_row['game_file']}: "
                    f"expected={expected_value!r}, actual={actual_value!r}"
                )
        ordered.append(record)
    return ordered


def _mean(records: list[dict], key: str) -> float:
    return fmean(float(record[key]) for record in records)


def summarize(records: list[dict]) -> dict:
    split_summaries = {}
    for split in ("seen", "unseen"):
        subset = [record for record in records if record["split"] == split]
        success_count = sum(bool(record["task_success"]) for record in subset)
        split_summaries[split] = {
            "task_count": len(subset),
            "success_count": success_count,
            "success_rate": success_count / len(subset),
            "average_env_rounds": _mean(subset, "env_rounds"),
            "timeout_rate": _mean(subset, "env_timeout"),
            "action_parse_valid_rate": _mean(subset, "action_parse_valid_rate"),
            "action_admissible_rate": _mean(subset, "action_admissible_rate"),
            "observation_unchanged_rate": _mean(subset, "observation_unchanged_rate"),
            "repeated_action_rate": _mean(subset, "repeated_action_rate"),
        }

    total_success = sum(bool(record["task_success"]) for record in records)
    return {
        "schema_version": 1,
        "evaluation_id": records[0].get("evaluation_id"),
        "checkpoint_label": records[0].get("checkpoint_label"),
        "task_count": len(records),
        "success_count": total_success,
        "success_rate": total_success / len(records),
        "macro_success_rate": fmean(
            split_summaries[split]["success_rate"] for split in ("seen", "unseen")
        ),
        "splits": split_summaries,
    }


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = [row for path in args.manifest for row in load_jsonl(path)]
    if len(manifest_rows) != args.expected_count:
        raise ValueError(
            f"Manifest count is {len(manifest_rows)}, expected {args.expected_count}"
        )
    records = validate_and_order(manifest_rows, load_task_shards(args.record_dir))
    summary = summarize(records)
    if summary["task_count"] != args.expected_count:
        raise ValueError(
            f"Result count is {summary['task_count']}, expected {args.expected_count}"
        )
    atomic_write_jsonl(args.output_jsonl, records)
    atomic_write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
