#!/usr/bin/env python3
"""Replot adaptive-v1 diagnostics after excluding prompt-truncation placeholders."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "adaptive_v1_original_analysis", HERE / "analyze_training.py"
)
ORIGINAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ORIGINAL)


def real_row_aggregates(rows: pd.DataFrame) -> pd.DataFrame:
    real = rows.loc[~rows["truncate_status"].eq("prompt_truncated")].copy()
    for column in (
        "teacher_entropy_topk",
        "student_entropy_topk",
        "sampled_reverse_kl",
    ):
        real[column] = pd.to_numeric(real[column], errors="coerce")
    return (
        real.groupby("training_step")
        .agg(
            teacher_entropy=("teacher_entropy_topk", "mean"),
            student_entropy=("student_entropy_topk", "mean"),
            sampled_reverse_kl=("sampled_reverse_kl", "mean"),
            valid_action_rate=("action_valid", "mean"),
            real_response_turns=("turn", "size"),
        )
        .reset_index()
    )


def corrected_by_step(rows: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    result = ORIGINAL.aggregate_step(rows, trajectories)
    corrected = real_row_aggregates(rows)
    replace = {
        column: f"{column}_real"
        for column in (
            "teacher_entropy",
            "student_entropy",
            "sampled_reverse_kl",
            "valid_action_rate",
            "real_response_turns",
        )
    }
    corrected = corrected.rename(columns=replace)
    result = result.merge(corrected, on="training_step", how="left")
    for column in (
        "teacher_entropy",
        "student_entropy",
        "sampled_reverse_kl",
        "valid_action_rate",
    ):
        result[column] = result.pop(f"{column}_real")
        result[f"{column}_smooth"] = (
            result[column].rolling(7, center=True, min_periods=3).mean()
        )
    result["real_response_turns"] = result.pop("real_response_turns_real")
    return result


def corrected_by_turn(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["prompt_truncated"] = data["truncate_status"].eq("prompt_truncated")
    data["teacher_entropy_topk"] = pd.to_numeric(
        data["teacher_entropy_topk"], errors="coerce"
    )
    records = []
    for (turn, success), group in data.groupby(["turn", "task_success"]):
        real = group.loc[~group["prompt_truncated"]]
        records.append(
            {
                "turn": int(turn),
                "task_success": bool(success),
                "teacher_entropy_real": ORIGINAL.safe_mean(
                    real["teacher_entropy_topk"]
                ),
                "real_trajectory_count": int(real["trajectory_id"].nunique()),
                "truncated_trajectory_count": int(
                    group.loc[group["prompt_truncated"], "trajectory_id"].nunique()
                ),
                "total_trajectory_count": int(group["trajectory_id"].nunique()),
                "real_valid_action_rate": float(real["action_valid"].mean())
                if len(real)
                else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records)


def corrected_aligned(rows: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    lookup = trajectories.set_index("trajectory_id")
    records = []
    real = rows.loc[~rows["truncate_status"].eq("prompt_truncated")]
    for trajectory_id, group in real.groupby("trajectory_id", sort=False):
        trajectory = lookup.loc[trajectory_id]
        if not bool(trajectory["frontier_triggered"]):
            continue
        frontier = int(trajectory["frontier_turn"])
        baseline = float(trajectory["baseline_teacher_entropy"])
        for _, row in group.iterrows():
            entropy = row.get("teacher_entropy_topk")
            if entropy is None or not np.isfinite(float(entropy)):
                continue
            relative = int(row["turn"]) - frontier
            if -8 <= relative <= 15:
                records.append(
                    {
                        "relative_turn": relative,
                        "delta_teacher_entropy": float(entropy) - baseline,
                        "task_success": bool(trajectory["task_success"]),
                        "model_version": int(trajectory["model_version"]),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_turn_outcome(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 7.8), sharex=True, constrained_layout=True
    )
    for success, label, color in ((False, "Failure", "C3"), (True, "Success", "C2")):
        part = frame.loc[frame["task_success"] == success]
        axes[0].plot(
            part["turn"] + 1,
            part["teacher_entropy_real"],
            label=label,
            color=color,
            lw=2,
        )
        axes[1].plot(
            part["turn"] + 1,
            part["real_trajectory_count"],
            label=f"{label}: real response",
            color=color,
            lw=2,
        )
        axes[1].plot(
            part["turn"] + 1,
            part["truncated_trajectory_count"],
            label=f"{label}: prompt placeholder",
            color=color,
            lw=1.5,
            ls="--",
        )
    ORIGINAL.style_axis(
        axes[0],
        "Teacher entropy by outcome — real generated responses only",
        "Environment turn",
        "Teacher top-16 partial entropy",
    )
    axes[0].legend(frameon=False)
    ORIGINAL.style_axis(
        axes[1],
        "Denominator and prompt-truncation contamination",
        "Environment turn",
        "Trajectories",
    )
    axes[1].legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(frame: pd.DataFrame, trajectories: pd.DataFrame, output: Path) -> None:
    summary = (
        frame.groupby("relative_turn")["delta_teacher_entropy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem95"] = 1.96 * summary["std"] / np.sqrt(
        summary["count"].clip(lower=1)
    )
    triggered = trajectories.loc[trajectories["frontier_triggered"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    x = summary["relative_turn"].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem95"].to_numpy(dtype=float)
    axes[0].plot(x, mean, lw=2.2, color="C0")
    axes[0].fill_between(x, mean - sem, mean + sem, color="C0", alpha=0.18)
    axes[0].axvline(0, color="black", ls="--", lw=1, label="Detected frontier")
    axes[0].axhline(0.175, color="C3", ls=":", lw=1.5, label="Threshold")
    ORIGINAL.style_axis(
        axes[0],
        "A. Frontier alignment — placeholders excluded",
        "Turn relative to frontier",
        "Teacher entropy − early baseline",
    )
    axes[0].legend(frameon=False)
    values = triggered["frontier_turn"].dropna().astype(int) + 1
    bins = np.arange(3.5, 31.5, 1)
    axes[1].hist(values, bins=bins, color="C1", alpha=0.82, edgecolor="white")
    ORIGINAL.style_axis(
        axes[1],
        "B. Detected frontier position",
        "Environment turn",
        "Triggered trajectories",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows: pd.DataFrame, trajectories: pd.DataFrame, output: Path) -> dict:
    max_version = int(trajectories["model_version"].max())
    panel = trajectories.loc[trajectories["model_version"] >= max_version - 30].copy()
    panel["sort_frontier"] = panel["frontier_turn"].fillna(99)
    panel = panel.sort_values(
        ["task_success", "frontier_triggered", "sort_frontier", "model_version"],
        ascending=[True, False, True, True],
    )
    ids = panel["trajectory_id"].tolist()
    matrix = np.full((len(ids), 30), np.nan)
    frontier_points = []
    truncation_points = []
    success_boundary = int((~panel["task_success"]).sum())
    grouped = {
        key: group.sort_values("turn")
        for key, group in rows.loc[rows["trajectory_id"].isin(ids)].groupby(
            "trajectory_id"
        )
    }
    for row_index, trajectory_id in enumerate(ids):
        group = grouped[trajectory_id]
        entropy = pd.to_numeric(group["teacher_entropy_topk"], errors="coerce")
        baseline = (
            float(entropy.iloc[:3].mean())
            if len(entropy) >= 3 and np.isfinite(entropy.iloc[:3]).all()
            else float("nan")
        )
        real = group.loc[~group["truncate_status"].eq("prompt_truncated")]
        for _, item in real.iterrows():
            turn = int(item["turn"])
            value = item["teacher_entropy_topk"]
            if turn < 30 and value is not None and np.isfinite(float(value)):
                matrix[row_index, turn] = float(value) - baseline
        truncated = group.loc[group["truncate_status"].eq("prompt_truncated"), "turn"]
        if len(truncated):
            truncation_points.append((float(truncated.min()), row_index))
        frontier = panel.iloc[row_index]["frontier_turn"]
        if pd.notna(frontier):
            frontier_points.append((float(frontier), row_index))
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    finite = matrix[np.isfinite(matrix)]
    limit = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    if frontier_points:
        ax.scatter(
            [x for x, _ in frontier_points],
            [y for _, y in frontier_points],
            s=7,
            c="black",
            alpha=0.7,
            label="Detected frontier",
        )
    if truncation_points:
        ax.scatter(
            [x for x, _ in truncation_points],
            [y for _, y in truncation_points],
            s=12,
            marker="x",
            c="dimgray",
            linewidths=0.7,
            label="First prompt truncation",
        )
    if 0 < success_boundary < len(panel):
        ax.axhline(success_boundary - 0.5, color="black", lw=1.5)
        ax.text(29.4, success_boundary - 2, "failures ↑", ha="right", va="bottom")
        ax.text(29.4, success_boundary + 2, "successes ↓", ha="right", va="top")
    ax.set_title(
        f"Latest-policy heterogeneity — real responses only (versions {max_version - 30}–{max_version})",
        loc="left",
        weight="bold",
    )
    ax.set_xlabel("Environment turn")
    ax.set_ylabel("Trajectory (sorted by outcome and frontier)")
    ax.set_xticks(np.arange(0, 30, 5), labels=np.arange(1, 31, 5))
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("Teacher entropy − trajectory early baseline")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "latest_panel_min_model_version": max_version - 30,
        "latest_panel_max_model_version": max_version,
        "latest_panel_trajectory_count": len(panel),
        "latest_panel_failure_count": success_boundary,
        "latest_panel_success_count": len(panel) - success_boundary,
        "latest_panel_prompt_truncated_count": len(truncation_points),
    }


def corrected_summary(
    rows: pd.DataFrame,
    trajectories: pd.DataFrame,
    aligned: pd.DataFrame,
    heatmap: dict,
) -> dict:
    truncated = rows["truncate_status"].eq("prompt_truncated")
    triggered = trajectories.loc[trajectories["frontier_triggered"]]
    lookup = trajectories.set_index("trajectory_id")
    trajectory_records = []
    for trajectory_id, group in rows.groupby("trajectory_id"):
        trajectory = lookup.loc[trajectory_id]
        if not bool(trajectory["frontier_triggered"]):
            continue
        frontier = int(trajectory["frontier_turn"])
        baseline = float(trajectory["baseline_teacher_entropy"])
        crossing = float(trajectory["crossing_teacher_entropy"])
        post = group.loc[
            (group["turn"] >= frontier)
            & ~group["truncate_status"].eq("prompt_truncated")
        ].sort_values("turn")
        entropy = pd.to_numeric(post["teacher_entropy_topk"], errors="coerce").dropna()
        trajectory_records.append(
            {
                "suffix": float(entropy.mean()) if len(entropy) else float("nan"),
                "last3": float(entropy.iloc[-3:].mean()) if len(entropy) else float("nan"),
                "baseline": baseline,
                "crossing": crossing,
                "success": bool(trajectory["task_success"]),
            }
        )
    frame = pd.DataFrame.from_records(trajectory_records)
    valid_suffix = frame.loc[frame["suffix"].notna()]
    failures = valid_suffix.loc[~valid_suffix["success"]]
    return {
        **heatmap,
        "trajectory_count": len(trajectories),
        "turn_row_count": len(rows),
        "real_generated_turn_count": int((~truncated).sum()),
        "prompt_placeholder_turn_count": int(truncated.sum()),
        "failed_trajectory_count": int((~trajectories["task_success"]).sum()),
        "failed_prompt_truncated_trajectory_count": int(
            ((~trajectories["task_success"]) & trajectories["prompt_truncated"]).sum()
        ),
        "triggered_trajectory_count": len(triggered),
        "triggered_with_real_suffix_count": len(valid_suffix),
        "real_suffix_below_crossing_fraction": float(
            (valid_suffix["suffix"] < valid_suffix["crossing"]).mean()
        ),
        "real_suffix_below_baseline_fraction": float(
            (valid_suffix["suffix"] < valid_suffix["baseline"]).mean()
        ),
        "real_last3_below_baseline_fraction": float(
            (valid_suffix["last3"] < valid_suffix["baseline"]).mean()
        ),
        "real_suffix_mean_entropy": float(valid_suffix["suffix"].mean()),
        "real_last3_mean_entropy": float(valid_suffix["last3"].mean()),
        "triggered_baseline_mean_entropy": float(valid_suffix["baseline"].mean()),
        "triggered_crossing_mean_entropy": float(valid_suffix["crossing"].mean()),
        "failure_real_suffix_below_baseline_fraction": float(
            (failures["suffix"] < failures["baseline"]).mean()
        ),
        "aligned_real_row_count": len(aligned),
    }


def archive_contaminated(output_dir: Path, names: list[str]) -> None:
    for name in names:
        source = output_dir / name
        target = source.with_name(source.stem + "_contaminated" + source.suffix)
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = ORIGINAL.load_rows(args.diagnostics)
    rows = rows.drop_duplicates(["trajectory_id", "turn"], keep="last").copy()
    trajectories, _ = ORIGINAL.trajectory_table(rows)
    by_step = corrected_by_step(rows, trajectories)
    by_turn = corrected_by_turn(rows)
    aligned = corrected_aligned(rows, trajectories)

    affected = [
        "training_overview.png",
        "teacher_entropy_by_turn_outcome.png",
        "frontier_mechanism.png",
        "teacher_entropy_frontier_heatmap_latest.png",
    ]
    archive_contaminated(args.output_dir, affected)

    ORIGINAL.save_csv(
        by_step, args.output_dir / "diagnostics_by_explorer_step_nontruncated.csv"
    )
    ORIGINAL.save_csv(
        by_turn, args.output_dir / "diagnostics_by_trajectory_turn_nontruncated.csv"
    )
    ORIGINAL.save_csv(
        aligned, args.output_dir / "frontier_aligned_rows_nontruncated.csv"
    )
    ORIGINAL.plot_training_overview(by_step, args.output_dir / "training_overview.png")
    plot_turn_outcome(by_turn, args.output_dir / "teacher_entropy_by_turn_outcome.png")
    plot_frontier(aligned, trajectories, args.output_dir / "frontier_mechanism.png")
    heatmap = plot_heatmap(
        rows, trajectories, args.output_dir / "teacher_entropy_frontier_heatmap_latest.png"
    )
    summary = corrected_summary(rows, trajectories, aligned, heatmap)
    (args.output_dir / "summary_nontruncated.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
