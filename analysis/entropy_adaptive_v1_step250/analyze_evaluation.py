#!/usr/bin/env python3
"""Compare entropy-adaptive v1 Step 250 with pilot and frozen baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def identity(game_file: str) -> str:
    normalized = game_file.replace("\\", "/")
    for marker in ("/valid_seen/", "/valid_unseen/"):
        if marker in normalized:
            return marker.strip("/") + "/" + normalized.split(marker, 1)[1]
    return "/".join(normalized.split("/")[-4:])


def read_results(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records.append(
                {
                    "identity": identity(row["game_file"]),
                    "split": row["split"],
                    "task_type": row["task_type"],
                    "success": bool(row["task_success"]),
                    "env_rounds": int(row["env_rounds"]),
                    "parse_valid_rate": float(row["action_parse_valid_rate"]),
                    "admissible_rate": float(row["action_admissible_rate"]),
                    "repeat_rate": float(row["repeated_action_rate"]),
                    "timeout": bool(row["env_timeout"]),
                }
            )
    frame = pd.DataFrame(records).sort_values("identity").reset_index(drop=True)
    if len(frame) != 274 or frame["identity"].duplicated().any():
        raise RuntimeError(f"Invalid full274 result at {path}: rows={len(frame)}")
    return frame


def paired(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    merged = left.merge(right, on="identity", suffixes=("_left", "_right"), validate="one_to_one")
    left_only = int((merged["success_left"] & ~merged["success_right"]).sum())
    right_only = int((~merged["success_left"] & merged["success_right"]).sum())
    discordant = left_only + right_only
    p_value = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "left_success": int(merged["success_left"].sum()),
        "right_success": int(merged["success_right"].sum()),
        "left_only": left_only,
        "right_only": right_only,
        "both_success": int((merged["success_left"] & merged["success_right"]).sum()),
        "both_failure": int((~merged["success_left"] & ~merged["success_right"]).sum()),
        "mcnemar_exact_p": p_value,
    }


def load_seed_family(pattern: str) -> dict[int, pd.DataFrame]:
    result = {}
    for seed in (42, 43, 44):
        suffix = "" if seed == 42 else f"_seed{seed}"
        path = Path(pattern.format(seed=seed, suffix=suffix))
        if not path.exists():
            raise FileNotFoundError(path)
        result[seed] = read_results(path)
    return result


def save_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def plot_seed_comparison(families: dict[str, dict[int, pd.DataFrame]], output: Path) -> None:
    labels = list(families)
    seeds = (42, 43, 44)
    x = np.arange(len(seeds))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    for index, label in enumerate(labels):
        values = [families[label][seed]["success"].mean() for seed in seeds]
        offset = (index - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=label, alpha=0.86)
        ax.bar_label(bars, labels=[f"{round(value * 274):.0f}" for value in values], padding=2, fontsize=9)
    ax.set_xticks(x, [f"Seed {seed}" for seed in seeds])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Full274 success rate")
    ax.set_title("Entropy-adaptive v1 evaluation across three paired seeds", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=len(labels), loc="upper left")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_pairs(adaptive: pd.DataFrame, baselines: dict[str, pd.DataFrame], output: Path) -> None:
    names = ["Adaptive v1", *baselines.keys()]
    frames = [adaptive, *baselines.values()]
    seen = [frame.loc[frame["split"] == "seen", "success"].mean() for frame in frames]
    unseen = [frame.loc[frame["split"] == "unseen", "success"].mean() for frame in frames]
    overall = [frame["success"].mean() for frame in frames]
    x = np.arange(len(names))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    for offset, values, label in ((-width, overall, "Overall"), (0, seen, "Seen"), (width, unseen, "Unseen")):
        bars = ax.bar(x + offset, values, width, label=label, alpha=0.86)
        ax.bar_label(bars, labels=[f"{value * 100:.1f}%" for value in values], padding=2, fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_title("Seed 42 frozen-protocol comparison", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_task_type_delta(adaptive: pd.DataFrame, baselines: dict[str, pd.DataFrame], output: Path) -> None:
    task_types = sorted(adaptive["task_type"].unique())
    rows = []
    for task_type in task_types:
        row = {"task_type": task_type, "adaptive": adaptive.loc[adaptive["task_type"] == task_type, "success"].mean()}
        for name, frame in baselines.items():
            row[name] = frame.loc[frame["task_type"] == task_type, "success"].mean()
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("adaptive")
    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    y = np.arange(len(table))
    ax.axvline(0, color="black", lw=1)
    for index, (name, color) in enumerate(zip(baselines, ("C1", "C2"))):
        delta = table["adaptive"] - table[name]
        ax.scatter(delta, y + (index - 0.5) * 0.14, s=52, label=f"Adaptive − {name}", color=color)
    ax.set_yticks(y, table["task_type"])
    ax.set_xlabel("Success-rate difference")
    ax.set_title("Seed 42 task-type deltas", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step250-pattern", required=True)
    parser.add_argument("--step10-adaptive-pattern", required=True)
    parser.add_argument("--step10-full-pattern", required=True)
    parser.add_argument("--tcod", type=Path, required=True)
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    families = {
        "Adaptive step 10": load_seed_family(args.step10_adaptive_pattern),
        "Full-loss step 10": load_seed_family(args.step10_full_pattern),
        "Adaptive step 250": load_seed_family(args.step250_pattern),
    }
    tcod = read_results(args.tcod)
    vanilla = read_results(args.vanilla)

    aggregate_rows = []
    paired_rows = []
    for method, seed_frames in families.items():
        for seed, frame in seed_frames.items():
            aggregate_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "success_count": int(frame["success"].sum()),
                    "success_rate": float(frame["success"].mean()),
                    "seen_success": int(frame.loc[frame["split"] == "seen", "success"].sum()),
                    "unseen_success": int(frame.loc[frame["split"] == "unseen", "success"].sum()),
                    "parse_valid_rate": float(frame["parse_valid_rate"].mean()),
                    "admissible_rate": float(frame["admissible_rate"].mean()),
                    "timeout_rate": float(frame["timeout"].mean()),
                }
            )
    for seed in (42, 43, 44):
        comparison = paired(families["Adaptive step 250"][seed], families["Full-loss step 10"][seed])
        paired_rows.append({"comparison": "Adaptive250 vs Full10", "seed": seed, **comparison})
        comparison = paired(families["Adaptive step 250"][seed], families["Adaptive step 10"][seed])
        paired_rows.append({"comparison": "Adaptive250 vs Adaptive10", "seed": seed, **comparison})
    for name, frame in (("TCOD", tcod), ("Vanilla", vanilla)):
        comparison = paired(families["Adaptive step 250"][42], frame)
        paired_rows.append({"comparison": f"Adaptive250 vs {name}", "seed": 42, **comparison})

    save_csv(aggregate_rows, args.output_dir / "evaluation_by_seed.csv")
    save_csv(paired_rows, args.output_dir / "paired_comparisons.csv")
    plot_seed_comparison(families, args.output_dir / "evaluation_three_seed_comparison.png")
    plot_baseline_pairs(
        families["Adaptive step 250"][42],
        {"TCOD F2B": tcod, "Vanilla OPD": vanilla},
        args.output_dir / "evaluation_seed42_frozen_baselines.png",
    )
    plot_task_type_delta(
        families["Adaptive step 250"][42],
        {"TCOD": tcod, "Vanilla": vanilla},
        args.output_dir / "evaluation_task_type_deltas.png",
    )

    step250_counts = [int(families["Adaptive step 250"][seed]["success"].sum()) for seed in (42, 43, 44)]
    summary = {
        "adaptive_step250_seed_success_counts": dict(zip((42, 43, 44), step250_counts)),
        "adaptive_step250_mean_success_rate": float(np.mean(step250_counts) / 274),
        "adaptive_step250_seed_sd_success_rate": float(np.std(np.asarray(step250_counts) / 274, ddof=1)),
        "adaptive_step250_descriptive_total": int(sum(step250_counts)),
        "adaptive_step250_descriptive_denominator": 822,
        "tcod_seed42_success_count": int(tcod["success"].sum()),
        "vanilla_seed42_success_count": int(vanilla["success"].sum()),
        "paired_comparisons": paired_rows,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
