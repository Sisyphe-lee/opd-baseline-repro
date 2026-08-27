#!/usr/bin/env python3
"""Create aggregate and paired comparisons for matched ALFWorld evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

from collect_alfworld_eval_results import atomic_write_json, load_jsonl, summarize


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_counts(left: dict[str, dict], right: dict[str, dict], split: str | None) -> dict:
    keys = sorted(left)
    if split is not None:
        keys = [key for key in keys if left[key]["split"] == split]
    both_success = left_only = right_only = both_failure = 0
    for key in keys:
        left_success = bool(left[key]["task_success"])
        right_success = bool(right[key]["task_success"])
        if left_success and right_success:
            both_success += 1
        elif left_success:
            left_only += 1
        elif right_success:
            right_only += 1
        else:
            both_failure += 1
    return {
        "task_count": len(keys),
        "both_success": both_success,
        "left_only_success": left_only,
        "right_only_success": right_only,
        "both_failure": both_failure,
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(left_only, right_only),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_game_key(record: dict) -> str:
    """Return a workspace-independent ALFWorld game identity.

    Historical frozen evaluations were produced from a sibling workspace, so
    their absolute ``game_file`` prefixes differ even when the manifest entry
    is identical. Everything below the versioned ALFWorld data root is the
    stable identity used by the frozen manifests.
    """
    game_file = record["game_file"]
    marker = "/json_2.1.1/"
    if marker not in game_file:
        raise ValueError(f"Cannot canonicalize ALFWorld game_file: {game_file}")
    return game_file.split(marker, 1)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="LABEL=PATH_TO_TASK_RESULTS_JSONL",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets: dict[str, list[dict]] = {}
    for value in args.input:
        label, separator, path_value = value.partition("=")
        if not separator or not label or not path_value:
            raise ValueError(f"Invalid --input value: {value}")
        if label in datasets:
            raise ValueError(f"Duplicate label: {label}")
        datasets[label] = load_jsonl(Path(path_value))

    keyed = {
        label: {canonical_game_key(record): record for record in records}
        for label, records in datasets.items()
    }
    reference_label = next(iter(keyed))
    reference_keys = set(keyed[reference_label])
    for label, records in keyed.items():
        if set(records) != reference_keys:
            raise ValueError(
                f"Task identity mismatch: {reference_label}={len(reference_keys)}, "
                f"{label}={len(records)}"
            )

    aggregate_rows = []
    aggregate_json = {}
    for label, records in datasets.items():
        summary = summarize(records)
        aggregate_json[label] = summary
        aggregate_rows.append(
            {
                "method": label,
                "success_count": summary["success_count"],
                "task_count": summary["task_count"],
                "success_rate": summary["success_rate"],
                "seen_success_count": summary["splits"]["seen"]["success_count"],
                "seen_task_count": summary["splits"]["seen"]["task_count"],
                "seen_success_rate": summary["splits"]["seen"]["success_rate"],
                "unseen_success_count": summary["splits"]["unseen"]["success_count"],
                "unseen_task_count": summary["splits"]["unseen"]["task_count"],
                "unseen_success_rate": summary["splits"]["unseen"]["success_rate"],
                "macro_success_rate": summary["macro_success_rate"],
            }
        )

    pairwise_rows = []
    pairwise_json = []
    for left_label, right_label in combinations(datasets, 2):
        for split in (None, "seen", "unseen"):
            counts = paired_counts(keyed[left_label], keyed[right_label], split)
            row = {
                "left": left_label,
                "right": right_label,
                "split": split or "all",
                **counts,
            }
            pairwise_rows.append(row)
            pairwise_json.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", pairwise_rows)
    atomic_write_json(
        args.output_dir / "comparison.json",
        {"aggregates": aggregate_json, "paired_comparisons": pairwise_json},
    )
    print(json.dumps(aggregate_rows, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
