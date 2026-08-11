#!/usr/bin/env python3
"""Plot per-task curriculum loss horizons for Vanilla, TCOD, and Adaptive v1."""

from __future__ import annotations

import json
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


def vanilla_horizons() -> np.ndarray:
    meta = json.loads(
        (ROOT / "checkpoints/vanilla_opd_step250/explorer_meta.json").read_text()
    )
    task_count = int(meta["taskset_states"][0]["current_index"])
    return np.full(task_count, MAX_TURN, dtype=int)


def tcod_horizons() -> np.ndarray:
    meta = json.loads(
        (ROOT / "checkpoints/tcod_f2b_step250/explorer_meta.json").read_text()
    )
    task_count = int(meta["taskset_states"][0]["current_index"])
    if task_count % 16:
        raise ValueError(f"TCOD task count {task_count} is not divisible by batch size 16")
    explorer_batches = task_count // 16
    values = []
    for batch_id in range(1, explorer_batches + 1):
        horizon = min(1 + batch_id // 2, MAX_TURN)
        values.extend([horizon] * 16)
    return np.asarray(values, dtype=int)


def adaptive_horizons() -> np.ndarray:
    trajectories = pd.read_csv(
        ROOT / "analysis/entropy_adaptive_v1_step250/trajectory_summary.csv"
    )
    triggered = trajectories["frontier_triggered"].astype(bool).to_numpy()
    retained = trajectories["retained_turns"].to_numpy(dtype=int)
    # No detected frontier means the selector permits the full 30-turn horizon,
    # irrespective of whether the environment happened to terminate earlier.
    return np.where(triggered, retained, MAX_TURN).astype(int)


def summary(name: str, values: np.ndarray) -> dict[str, float | int | str]:
    return {
        "method": name,
        "task_count": int(len(values)),
        "mean_loss_horizon": float(values.mean()),
        "median_loss_horizon": float(np.median(values)),
        "p25_loss_horizon": float(np.quantile(values, 0.25)),
        "p75_loss_horizon": float(np.quantile(values, 0.75)),
        "full_30_count": int((values == MAX_TURN).sum()),
        "full_30_fraction": float((values == MAX_TURN).mean()),
    }


def horizon_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    turns = np.arange(1, MAX_TURN + 1)
    return (turns[None, :] <= ordered[:, None]).astype(float), ordered


def draw_one(name: str, values: np.ndarray, color: str, output: Path) -> None:
    matrix, ordered = horizon_matrix(values)
    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    cmap = ListedColormap(["white", color])
    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0.5, MAX_TURN + 0.5, len(values) + 0.5, 0.5),
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    y = np.arange(1, len(ordered) + 1)
    ax.plot(ordered + 0.5, y, color="black", lw=0.9, label="Loss cutoff boundary")
    stats = summary(name, values)
    ax.set_title(
        f"{name}: per-task curriculum loss horizon",
        loc="left",
        weight="bold",
    )
    ax.text(
        0.01,
        1.015,
        (
            f"n={stats['task_count']:,} · mean={stats['mean_loss_horizon']:.2f} · "
            f"median={stats['median_loss_horizon']:.0f} · "
            f"K=30: {stats['full_30_fraction']:.1%}"
        ),
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.set_xlabel("Environment turn included in loss")
    ax.set_ylabel("Task rank (shortest curriculum at top)")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(len(values) + 0.5, 0.5)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_combined(methods: list[tuple[str, np.ndarray, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), constrained_layout=True)
    for ax, (name, values, color) in zip(axes, methods):
        matrix, ordered = horizon_matrix(values)
        cmap = ListedColormap(["white", color])
        ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(0.5, MAX_TURN + 0.5, 100.0, 0.0),
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        rank = (np.arange(len(ordered)) + 0.5) / len(ordered) * 100
        ax.plot(ordered + 0.5, rank, color="black", lw=1.0)
        stats = summary(name, values)
        ax.set_title(
            f"{name}\nn={len(values):,} · mean={values.mean():.2f} · "
            f"K=30: {stats['full_30_fraction']:.1%}",
            loc="left",
            weight="bold",
            fontsize=11,
            pad=8,
        )
        ax.set_xlabel("Environment turn included in loss")
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_xlim(0.5, 30.5)
        ax.set_ylim(100, 0)
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Task percentile (shortest curriculum at top)")
    fig.suptitle(
        "Curriculum-imposed loss horizons (environment termination excluded)",
        fontsize=15,
        weight="bold",
        y=1.04,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    methods = [
        ("Vanilla OPD", vanilla_horizons(), "#4C78A8"),
        ("TCOD F2B", tcod_horizons(), "#F58518"),
        ("Adaptive v1", adaptive_horizons(), "#54A24B"),
    ]
    filenames = ["vanilla_loss_horizons.png", "tcod_loss_horizons.png", "adaptive_v1_loss_horizons.png"]
    for (name, values, color), filename in zip(methods, filenames):
        draw_one(name, values, color, OUT / filename)
    draw_combined(methods, OUT / "three_method_loss_horizons.png")

    pd.DataFrame([summary(name, values) for name, values, _ in methods]).to_csv(
        OUT / "loss_horizon_summary.csv", index=False
    )
    histogram = pd.DataFrame({"turn": np.arange(1, MAX_TURN + 1)})
    for name, values, _ in methods:
        key = name.lower().replace(" ", "_")
        counts = pd.Series(values).value_counts().reindex(range(1, MAX_TURN + 1), fill_value=0)
        histogram[f"{key}_count"] = counts.to_numpy()
    histogram.to_csv(OUT / "loss_horizon_histogram.csv", index=False)
    print(pd.DataFrame([summary(name, values) for name, values, _ in methods]).to_string(index=False))


if __name__ == "__main__":
    main()
