#!/usr/bin/env python3
"""Plot effective trainable turns in chronological exploration order."""

from __future__ import annotations

import json
import pickle
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAX_TURN = 30


def numeric_prefix(value: object) -> int:
    match = re.match(r"^(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse numeric batch from {value!r}")
    return int(match.group(1))


def load_baseline_buffer(method: str, relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    groups: dict[tuple[int, str, int], dict[str, object]] = defaultdict(
        lambda: {"trainable_steps": set(), "recorded_steps": set(), "prompt_truncated": False}
    )
    connection = sqlite3.connect(path)
    try:
        cursor = connection.execute(
            "SELECT experience_bytes FROM pipeline_input ORDER BY id"
        )
        for (blob,) in cursor:
            experience = pickle.loads(blob)
            batch = numeric_prefix(experience.eid.batch)
            task = str(experience.eid.task)
            run = int(experience.eid.run)
            step = int(experience.eid.step)
            group = groups[(batch, task, run)]
            group["recorded_steps"].add(step)
            if experience.truncate_status == "prompt_truncated":
                group["prompt_truncated"] = True
            if experience.action_mask is not None and bool(experience.action_mask.any()):
                group["trainable_steps"].add(step)
    finally:
        connection.close()

    records = []
    for (batch, task, run), values in groups.items():
        trainable_steps = values["trainable_steps"]
        records.append(
            {
                "method": method,
                "explorer_batch": batch,
                "task_in_batch": task,
                "run_id": run,
                "trajectory_id": f"{batch}/{task}/{run}",
                "effective_loss_turns": len(trainable_steps),
                "recorded_turns": len(values["recorded_steps"]),
                "prompt_truncated": bool(values["prompt_truncated"]),
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["task_sort"] = pd.to_numeric(frame["task_in_batch"], errors="coerce")
    frame = frame.sort_values(
        ["explorer_batch", "task_sort", "task_in_batch", "run_id"]
    ).reset_index(drop=True)
    frame["training_order"] = np.arange(1, len(frame) + 1)
    return frame.drop(columns="task_sort")


def load_adaptive() -> pd.DataFrame:
    diagnostics = (
        ROOT
        / "runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16/diagnostics/trajectory_metrics.jsonl"
    )
    rows = pd.read_json(diagnostics, lines=True)
    rows = rows.drop_duplicates(["trajectory_id", "turn"], keep="last")
    rows["effective"] = (
        rows["loss_retained"].astype(bool)
        & ~rows["truncate_status"].eq("prompt_truncated")
    )
    frame = (
        rows.groupby("trajectory_id", sort=False)
        .agg(
            explorer_batch=("training_step", "first"),
            effective_loss_turns=("effective", "sum"),
            recorded_turns=("turn", "size"),
            prompt_truncated=(
                "truncate_status",
                lambda values: bool(values.eq("prompt_truncated").any()),
            ),
        )
        .reset_index()
    )
    frame["method"] = "Adaptive v1"
    frame["task_in_batch"] = frame["trajectory_id"]
    frame["run_id"] = 0
    frame = frame.sort_values(["explorer_batch", "trajectory_id"]).reset_index(drop=True)
    frame["training_order"] = np.arange(1, len(frame) + 1)
    frame["effective_loss_turns"] = frame["effective_loss_turns"].astype(int)
    return frame


def validate(frame: pd.DataFrame) -> None:
    if len(frame) == 0:
        raise ValueError("No trajectories found")
    if not frame["effective_loss_turns"].between(0, MAX_TURN).all():
        raise ValueError("Effective loss turns fall outside [0, 30]")


def raster(frame: pd.DataFrame) -> np.ndarray:
    turns = np.arange(1, MAX_TURN + 1)
    lengths = frame["effective_loss_turns"].to_numpy(dtype=int)
    return (turns[None, :] <= lengths[:, None]).astype(float)


def stats(frame: pd.DataFrame) -> dict[str, float | int | str]:
    values = frame["effective_loss_turns"]
    return {
        "method": str(frame["method"].iloc[0]),
        "trajectory_count": int(len(frame)),
        "mean_effective_loss_turns": float(values.mean()),
        "median_effective_loss_turns": float(values.median()),
        "p25_effective_loss_turns": float(values.quantile(0.25)),
        "p75_effective_loss_turns": float(values.quantile(0.75)),
        "full_30_count": int(values.eq(30).sum()),
        "full_30_fraction": float(values.eq(30).mean()),
        "prompt_truncated_count": int(frame["prompt_truncated"].sum()),
    }


def rolling_median(frame: pd.DataFrame) -> np.ndarray:
    window = min(128, max(16, len(frame) // 12))
    return (
        frame["effective_loss_turns"]
        .rolling(window, center=True, min_periods=max(8, window // 4))
        .median()
        .to_numpy(dtype=float)
    )


def draw_one(frame: pd.DataFrame, color: str, output: Path) -> None:
    method = str(frame["method"].iloc[0])
    summary = stats(frame)
    matrix = raster(frame)
    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0.5, 30.5, len(frame) + 0.5, 0.5),
        cmap=ListedColormap(["white", color]),
        vmin=0,
        vmax=1,
    )
    ax.plot(
        rolling_median(frame) + 0.5,
        frame["training_order"],
        color="black",
        lw=1.4,
        label="Rolling median effective horizon",
    )
    ax.set_title(f"{method}: effective loss turns in training order", loc="left", weight="bold")
    ax.text(
        0.01,
        1.015,
        (
            f"n={len(frame):,} · mean={summary['mean_effective_loss_turns']:.2f} · "
            f"median={summary['median_effective_loss_turns']:.0f} · "
            f"K=30: {summary['full_30_fraction']:.1%}"
        ),
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.set_xlabel("Environment turn with non-zero loss mask")
    ax.set_ylabel("Trajectory in chronological exploration order")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(len(frame) + 0.5, 0.5)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_combined(methods: list[tuple[pd.DataFrame, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), constrained_layout=True)
    for ax, (frame, color) in zip(axes, methods):
        method = str(frame["method"].iloc[0])
        summary = stats(frame)
        ax.imshow(
            raster(frame),
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(0.5, 30.5, 100.0, 0.0),
            cmap=ListedColormap(["white", color]),
            vmin=0,
            vmax=1,
        )
        rank = (frame["training_order"].to_numpy() - 0.5) / len(frame) * 100
        ax.plot(rolling_median(frame) + 0.5, rank, color="black", lw=1.2)
        ax.set_title(
            f"{method}\nn={len(frame):,} · mean={summary['mean_effective_loss_turns']:.2f} · "
            f"K=30: {summary['full_30_fraction']:.1%}",
            loc="left",
            weight="bold",
            fontsize=11,
            pad=8,
        )
        ax.set_xlabel("Environment turn with non-zero loss mask")
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_xlim(0.5, 30.5)
        ax.set_ylim(100, 0)
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Training progress percentile (earliest at top)")
    fig.suptitle(
        "Realized trainable turns in chronological exploration order",
        fontsize=15,
        weight="bold",
        y=1.04,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vanilla = load_baseline_buffer(
        "Vanilla OPD", "checkpoints/vanilla_opd_step250/buffer/explorer_output.db"
    )
    tcod = load_baseline_buffer(
        "TCOD F2B", "checkpoints/tcod_f2b_step250/buffer/explorer_output.db"
    )
    adaptive = load_adaptive()
    methods = [(vanilla, "#4C78A8"), (tcod, "#F58518"), (adaptive, "#54A24B")]
    for frame, _ in methods:
        validate(frame)
    draw_one(vanilla, methods[0][1], OUT / "vanilla_realized_loss_turns.png")
    draw_one(tcod, methods[1][1], OUT / "tcod_realized_loss_turns.png")
    draw_one(adaptive, methods[2][1], OUT / "adaptive_v1_realized_loss_turns.png")
    draw_combined(methods, OUT / "three_method_realized_loss_turns.png")
    pd.DataFrame([stats(frame) for frame, _ in methods]).to_csv(
        OUT / "realized_loss_turns_summary.csv", index=False
    )
    pd.concat([frame for frame, _ in methods], ignore_index=True).to_csv(
        OUT / "realized_loss_turns_by_trajectory.csv", index=False
    )
    print(pd.DataFrame([stats(frame) for frame, _ in methods]).to_string(index=False))


if __name__ == "__main__":
    main()
