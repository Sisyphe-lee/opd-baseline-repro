#!/usr/bin/env python3
"""Plot chronological per-Explorer-batch rollout outcome rates."""

from __future__ import annotations

import pickle
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SMOOTH_WINDOW = 9


def numeric_prefix(value: object) -> int:
    match = re.match(r"^(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse Explorer batch from {value!r}")
    return int(match.group(1))


def load_vanilla() -> pd.DataFrame:
    database = ROOT / "checkpoints/vanilla_opd_step250/buffer/explorer_output.db"
    groups: dict[tuple[int, str, int], dict[str, object]] = defaultdict(
        lambda: {"rewards": set(), "model_versions": set()}
    )
    connection = sqlite3.connect(database)
    try:
        for blob, model_version in connection.execute(
            "SELECT experience_bytes, model_version FROM pipeline_input ORDER BY id"
        ):
            experience = pickle.loads(blob)
            key = (
                numeric_prefix(experience.eid.batch),
                str(experience.eid.task),
                int(experience.eid.run),
            )
            if experience.reward is not None:
                groups[key]["rewards"].add(float(experience.reward))
            groups[key]["model_versions"].add(int(model_version))
    finally:
        connection.close()

    records = []
    for (batch, task, run), values in groups.items():
        rewards = values["rewards"]
        if len(rewards) != 1:
            raise ValueError(f"Vanilla trajectory {batch}/{task}/{run} has rewards {rewards}")
        model_versions = values["model_versions"]
        if len(model_versions) != 1:
            raise ValueError(
                f"Vanilla trajectory {batch}/{task}/{run} has model versions {model_versions}"
            )
        records.append(
            {
                "method": "Vanilla OPD",
                "explorer_batch": batch,
                "task_in_batch": task,
                "run_id": run,
                "model_version": next(iter(model_versions)),
                "matched_done_reward": float(next(iter(rewards)) > 0),
                "audited_task_success": np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def load_adaptive() -> pd.DataFrame:
    diagnostics = (
        ROOT
        / "runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16/diagnostics/trajectory_metrics.jsonl"
    )
    rows = pd.read_json(diagnostics, lines=True)
    rows = rows.drop_duplicates(["trajectory_id", "turn"], keep="last")
    frame = (
        rows.groupby("trajectory_id", sort=False)
        .agg(
            explorer_batch=("training_step", "first"),
            model_version=("student_model_version", "first"),
            matched_done_reward=("env_done", "first"),
            audited_task_success=("task_success", "first"),
        )
        .reset_index()
    )
    frame["method"] = "Adaptive v1"
    frame["task_in_batch"] = frame["trajectory_id"]
    frame["run_id"] = 0
    frame["matched_done_reward"] = frame["matched_done_reward"].astype(float)
    frame["audited_task_success"] = frame["audited_task_success"].astype(float)
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby("explorer_batch")
        .agg(
            trajectory_count=("matched_done_reward", "size"),
            model_version_mean=("model_version", "mean"),
            matched_done_reward=("matched_done_reward", "mean"),
            audited_task_success=("audited_task_success", "mean"),
        )
        .reset_index()
        .sort_values("explorer_batch")
    )
    for column in ("matched_done_reward", "audited_task_success"):
        result[f"{column}_smooth"] = result[column].rolling(
            SMOOTH_WINDOW, center=True, min_periods=3
        ).mean()
        result[f"{column}_cumulative"] = (
            (result[column] * result["trajectory_count"]).cumsum()
            / result["trajectory_count"].cumsum()
        )
    return result


def validate_batches(name: str, frame: pd.DataFrame) -> None:
    counts = frame.groupby("explorer_batch").size()
    bad = counts[counts != 16]
    if len(bad):
        raise ValueError(f"{name} has non-16 trajectory batches: {bad.to_dict()}")


def draw(vanilla: pd.DataFrame, adaptive: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.4), sharex=True, constrained_layout=True)
    colors = {"Vanilla OPD": "#4C78A8", "Adaptive v1": "#54A24B"}
    for name, frame in (("Vanilla OPD", vanilla), ("Adaptive v1", adaptive)):
        color = colors[name]
        axes[0].plot(
            frame["explorer_batch"], frame["matched_done_reward"],
            color=color, alpha=0.20, lw=0.8,
        )
        axes[0].plot(
            frame["explorer_batch"], frame["matched_done_reward_smooth"],
            color=color, lw=2.4, label=f"{name}: {SMOOTH_WINDOW}-batch mean",
        )
        axes[1].plot(
            frame["explorer_batch"], frame["matched_done_reward_cumulative"],
            color=color, lw=2.4, label=name,
        )

    if adaptive["audited_task_success"].notna().any():
        axes[0].plot(
            adaptive["explorer_batch"], adaptive["audited_task_success_smooth"],
            color="#E45756", lw=1.8, ls="--",
            label="Adaptive v1: audited task success",
        )
        axes[1].plot(
            adaptive["explorer_batch"], adaptive["audited_task_success_cumulative"],
            color="#E45756", lw=1.8, ls="--",
            label="Adaptive v1: audited task success",
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
        "Training-rollout outcomes across the shared sequential task prefix",
        fontsize=15, weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vanilla_trajectories = load_vanilla()
    adaptive_trajectories = load_adaptive()
    validate_batches("Vanilla", vanilla_trajectories)
    validate_batches("Adaptive", adaptive_trajectories)
    vanilla = aggregate(vanilla_trajectories)
    adaptive = aggregate(adaptive_trajectories)
    vanilla.insert(0, "method", "Vanilla OPD")
    adaptive.insert(0, "method", "Adaptive v1")
    pd.concat([vanilla, adaptive], ignore_index=True).to_csv(
        OUT / "batch_rollout_accuracy.csv", index=False
    )
    draw(vanilla, adaptive, OUT / "vanilla_adaptive_batch_rollout_accuracy.png")
    print(
        pd.DataFrame(
            [
                {
                    "method": "Vanilla OPD",
                    "batch_count": len(vanilla),
                    "trajectory_count": int(vanilla["trajectory_count"].sum()),
                    "final_cumulative_done_reward": vanilla["matched_done_reward_cumulative"].iloc[-1],
                },
                {
                    "method": "Adaptive v1",
                    "batch_count": len(adaptive),
                    "trajectory_count": int(adaptive["trajectory_count"].sum()),
                    "final_cumulative_done_reward": adaptive["matched_done_reward_cumulative"].iloc[-1],
                    "final_cumulative_audited_success": adaptive["audited_task_success_cumulative"].iloc[-1],
                },
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
