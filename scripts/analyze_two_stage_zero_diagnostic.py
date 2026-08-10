#!/usr/bin/env python3
"""Create auditable warm-start and final comparison tables/figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGES = (
    ("Student init", "student_init_seed42"),
    ("Offline 30", "offline30_seed42"),
    ("Offline 30 + online 220", "offline30_online220_seed42"),
)
RATE_METRICS = (
    "action_parse_valid_rate",
    "action_admissible_rate",
    "observation_unchanged_rate",
    "repeated_action_rate",
    "env_timeout",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar(left: list[dict], right: list[dict]) -> dict:
    left_by_game = {row["game_file"]: bool(row["task_success"]) for row in left}
    right_by_game = {row["game_file"]: bool(row["task_success"]) for row in right}
    if left_by_game.keys() != right_by_game.keys():
        raise ValueError("Paired evaluations do not cover identical game_file sets")
    left_only = sum(left_by_game[g] and not right_by_game[g] for g in left_by_game)
    right_only = sum(right_by_game[g] and not left_by_game[g] for g in left_by_game)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_only_success": left_only,
        "right_only_success": right_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available = []
    for label, directory in STAGES:
        root = args.evaluation_root / directory
        summary_path = root / "summary.json"
        results_path = root / "task_results.jsonl"
        if summary_path.exists() != results_path.exists():
            raise ValueError(f"Incomplete evaluation output under {root}")
        if summary_path.exists():
            summary = load_json(summary_path)
            rows = load_jsonl(results_path)
            if summary["task_count"] != 274 or len(rows) != 274:
                raise ValueError(f"Expected 274 records for {label}")
            available.append((label, directory, summary, rows))
    if len(available) < 2:
        raise ValueError("Need at least student-init and offline-30 full274 results")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = []
    for label, directory, summary, rows in available:
        stage_rows.append(
            {
                "stage": label,
                "directory": directory,
                "success_count": summary["success_count"],
                "success_rate": summary["success_rate"],
                "seen_success_rate": summary["splits"]["seen"]["success_rate"],
                "unseen_success_rate": summary["splits"]["unseen"]["success_rate"],
                "average_env_rounds": mean(rows, "env_rounds"),
                **{metric: mean(rows, metric) for metric in RATE_METRICS},
            }
        )
    with (args.output_dir / "stage_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)

    comparisons = []
    for left_idx, right_idx in ((0, 1), (1, 2), (0, 2)):
        if right_idx >= len(available):
            continue
        left = available[left_idx]
        right = available[right_idx]
        paired = exact_mcnemar(left[3], right[3])
        comparisons.append(
            {
                "left": left[0],
                "right": right[0],
                "success_rate_delta": right[2]["success_rate"] - left[2]["success_rate"],
                "success_count_delta": right[2]["success_count"] - left[2]["success_count"],
                **paired,
            }
        )
    with (args.output_dir / "paired_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    labels = [row["stage"] for row in stage_rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    width = 0.24
    for offset, (key, name) in zip(
        (-width, 0, width),
        (("success_rate", "Overall"), ("seen_success_rate", "Seen"), ("unseen_success_rate", "Unseen")),
    ):
        values = [100 * row[key] for row in stage_rows]
        bars = ax.bar([idx + offset for idx in x], values, width, label=name)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Two-stage zero diagnostic: frozen full274")
    ax.set_ylim(0, max(20, max(100 * row["success_rate"] for row in stage_rows) * 1.25))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "success_rates.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0))
    behavior_metrics = (
        ("action_parse_valid_rate", "Parse valid", False),
        ("action_admissible_rate", "Admissible", False),
        ("observation_unchanged_rate", "Unchanged obs", True),
        ("repeated_action_rate", "Repeated action", True),
        ("env_timeout", "Timeout", True),
        ("average_env_rounds", "Environment rounds", True),
    )
    colors = ("#6b7280", "#2563eb", "#059669")
    for ax, (key, title, lower_is_better) in zip(axes.flat, behavior_metrics):
        values = [row[key] * (100 if key != "average_env_rounds" else 1) for row in stage_rows]
        bars = ax.bar(labels, values, color=colors[: len(labels)])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_title(f"{title} ({'lower' if lower_is_better else 'higher'} is better)")
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        if key != "average_env_rounds":
            ax.set_ylabel("Rate (%)")
    fig.suptitle("On-policy behavior after offline warm-start", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output_dir / "behavior_metrics.png", dpi=180)
    plt.close(fig)

    output = {
        "schema_version": 1,
        "protocol": "full274_h30_temperature0.4_seed42_response512_promptfix_accmemory_strict",
        "stages": stage_rows,
        "paired_comparisons": comparisons,
        "warm_diagnostic": {
            "definition": (
                "Offline 30 is empirically warmer only to the extent that its student-on-policy "
                "full274 success and behavior metrics improve over the identically evaluated init."
            ),
            "offline_minus_init_success_rate": stage_rows[1]["success_rate"]
            - stage_rows[0]["success_rate"],
            "offline_vs_init_mcnemar_p": comparisons[0]["exact_two_sided_p"],
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
