#!/usr/bin/env python3
"""Plot Vanilla, TCOD, and Adaptive rollout outcomes by Explorer batch."""

from __future__ import annotations

import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_batch_rollout_accuracy as BASE


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SMOOTH_WINDOW = 9


def load_baseline(method: str, relative_path: str) -> pd.DataFrame:
    groups: dict[tuple[int, str, int], dict[str, object]] = defaultdict(
        lambda: {"rewards": set(), "model_versions": set(), "seen_step0": False}
    )
    connection = sqlite3.connect(ROOT / relative_path)
    try:
        for blob, model_version in connection.execute(
            "SELECT experience_bytes, model_version FROM pipeline_input ORDER BY id"
        ):
            experience = pickle.loads(blob)
            key = (
                BASE.numeric_prefix(experience.eid.batch),
                str(experience.eid.task),
                int(experience.eid.run),
            )
            if int(experience.eid.step) == 0 and groups[key]["seen_step0"]:
                groups[key] = {"rewards": set(), "model_versions": set(), "seen_step0": False}
            groups[key]["seen_step0"] = True
            if experience.reward is not None:
                groups[key]["rewards"].add(float(experience.reward))
            groups[key]["model_versions"].add(int(model_version))
    finally:
        connection.close()

    records = []
    for (batch, task, run), values in groups.items():
        rewards = values["rewards"]
        versions = values["model_versions"]
        if len(rewards) != 1:
            raise ValueError(
                f"{method} trajectory {batch}\/{task}\/{run}: rewards={rewards}"
            )
        records.append(
            {
                "method": method,
                "explorer_batch": batch,
                "task_in_batch": task,
                "run_id": run,
                "model_version": float(np.mean(list(versions))),
                "matched_done_reward": float(next(iter(rewards)) > 0),
                "audited_task_success": np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def draw(methods: list[tuple[str, pd.DataFrame, str]], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.4), sharex=True, constrained_layout=True)
    for name, frame, color in methods:
        axes[0].plot(
            frame["explorer_batch"], frame["matched_done_reward"],
            color=color, alpha=0.16, lw=0.8,
        )
        axes[0].plot(
            frame["explorer_batch"], frame["matched_done_reward_smooth"],
            color=color, lw=2.35, label=f"{name}: {SMOOTH_WINDOW}-batch mean",
        )
        axes[1].plot(
            frame["explorer_batch"], frame["matched_done_reward_cumulative"],
            color=color, lw=2.35, label=name,
        )

    adaptive = next(frame for name, frame, _ in methods if name == "Adaptive v1")
    axes[0].plot(
        adaptive["explorer_batch"], adaptive["audited_task_success_smooth"],
        color="#E45756", lw=1.7, ls="--", label="Adaptive: audited task success",
    )
    axes[1].plot(
        adaptive["explorer_batch"], adaptive["audited_task_success_cumulative"],
        color="#E45756", lw=1.7, ls="--", label="Adaptive: audited task success",
    )

    axes[0].set_title(
        "A. Per-Explorer-batch rollout outcome rate (16 tasks per batch)",
        loc="left", weight="bold",
    )
    axes[0].set_ylabel("Batch outcome rate")
    axes[1].set_title("B. Cumulative rollout outcome rate", loc="left", weight="bold")
    axes[1].set_xlabel("Explorer batch (same sequential task order)")
    axes[1].set_ylabel("Cumulative outcome rate")
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2)
    fig.suptitle(
        "Training-rollout outcomes: Vanilla vs TCOD vs Adaptive v1",
        fontsize=15, weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vanilla_raw = load_baseline(
        "Vanilla OPD", "checkpoints/vanilla_opd_step250/buffer/explorer_output.db"
    )
    tcod_raw = load_baseline(
        "TCOD F2B", "checkpoints/tcod_f2b_step250/buffer/explorer_output.db"
    )
    adaptive_raw = BASE.load_adaptive()
    for name, frame in (
        ("Vanilla", vanilla_raw), ("TCOD", tcod_raw), ("Adaptive", adaptive_raw)
    ):
        BASE.validate_batches(name, frame)

    methods = [
        ("Vanilla OPD", BASE.aggregate(vanilla_raw), "#4C78A8"),
        ("TCOD F2B", BASE.aggregate(tcod_raw), "#F58518"),
        ("Adaptive v1", BASE.aggregate(adaptive_raw), "#54A24B"),
    ]
    for name, frame, _ in methods:
        frame.insert(0, "method", name)
    pd.concat([frame for _, frame, _ in methods], ignore_index=True).to_csv(
        OUT / "three_method_batch_rollout_accuracy.csv", index=False
    )
    draw(methods, OUT / "three_method_batch_rollout_accuracy.png")

    common_last_batch = min(int(frame["explorer_batch"].max()) for _, frame, _ in methods)
    records = []
    for name, frame, _ in methods:
        common = frame.loc[frame["explorer_batch"] <= common_last_batch]
        common_rate = float(
            (common["matched_done_reward"] * common["trajectory_count"]).sum()
            / common["trajectory_count"].sum()
        )
        records.append(
            {
                "method": name,
                "total_batches": len(frame),
                "total_trajectories": int(frame["trajectory_count"].sum()),
                "common_prefix_batches": common_last_batch,
                "common_prefix_trajectories": int(common["trajectory_count"].sum()),
                "common_prefix_done_reward_rate": common_rate,
                "full_run_done_reward_rate": float(frame["matched_done_reward_cumulative"].iloc[-1]),
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary.to_csv(OUT / "three_method_batch_rollout_accuracy_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
