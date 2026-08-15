#!/usr/bin/env python3
"""Analyze the completed entropy-adaptive v1 training diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCALAR_FIELDS = (
    "diagnostics_schema_version",
    "diagnostics_kind",
    "training_step",
    "student_model_version",
    "trajectory_id",
    "game_id",
    "game_file",
    "split",
    "task_type",
    "turn",
    "action",
    "action_valid",
    "prior_action_count",
    "admissible_action_count",
    "truncate_status",
    "prior_observation_count",
    "observation_chars",
    "prompt_tokens",
    "step_reward",
    "won",
    "lost",
    "env_terminated",
    "response_tokens",
    "student_surprisal",
    "teacher_surprisal",
    "sampled_reverse_kl",
    "student_entropy_topk",
    "teacher_entropy_topk",
    "student_topk_mass",
    "teacher_topk_mass",
    "student_top1_top2_margin",
    "teacher_top1_top2_margin",
    "task_success",
    "env_done",
    "env_lost",
    "env_rounds",
    "frontier_strategy",
    "entropy_frontier_threshold",
    "entropy_frontier_baseline_turns",
    "entropy_frontier_sustain_turns",
    "entropy_frontier_turn",
    "loss_retained",
    "retained_turns",
)


def finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def safe_mean(values) -> float:
    values = finite(values)
    return float(values.mean()) if values.size else float("nan")


def proportion_ci(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% confidence interval."""
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def load_rows(path: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    malformed = 0
    schema_counts = Counter()
    kind_counts = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            schema_counts[record.get("diagnostics_schema_version")] += 1
            kind_counts[record.get("diagnostics_kind")] += 1
            rows.append({key: record.get(key) for key in SCALAR_FIELDS})
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise RuntimeError(f"No diagnostics rows found in {path}")
    duplicate_count = int(
        frame.duplicated(["training_step", "game_id", "run_id", "turn"]).sum()
        if "run_id" in frame.columns
        else frame.duplicated(["training_step", "game_id", "turn"]).sum()
    )
    metadata = {
        "raw_row_count": len(frame),
        "malformed_line_count": malformed,
        "schema_counts": {str(key): value for key, value in schema_counts.items()},
        "kind_counts": dict(kind_counts),
        "duplicate_turn_row_count": duplicate_count,
    }
    return frame, metadata


def trajectory_table(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    aligned_records = []
    for trajectory_id, group in rows.groupby("trajectory_id", sort=False):
        group = group.sort_values("turn")
        turns = group["turn"].astype(int).tolist()
        if turns != list(range(len(turns))):
            raise RuntimeError(f"Non-contiguous turns for {trajectory_id}: {turns[:8]}...")
        first = group.iloc[0]
        teacher = pd.to_numeric(group["teacher_entropy_topk"], errors="coerce")
        student = pd.to_numeric(group["student_entropy_topk"], errors="coerce")
        reverse_kl = pd.to_numeric(group["sampled_reverse_kl"], errors="coerce")
        response_tokens = pd.to_numeric(group["response_tokens"], errors="coerce").fillna(0)
        frontier = first.get("entropy_frontier_turn")
        frontier = int(frontier) if pd.notna(frontier) else None
        retained = int(first.get("retained_turns") or int(group["loss_retained"].fillna(False).sum()))
        baseline_turns = int(first.get("entropy_frontier_baseline_turns") or 3)
        baseline = safe_mean(teacher.iloc[:baseline_turns])
        suffix = teacher.iloc[retained:]
        finite_suffix = finite(suffix)
        crossing_entropy = (
            float(teacher.iloc[frontier])
            if frontier is not None and frontier < len(teacher) and pd.notna(teacher.iloc[frontier])
            else float("nan")
        )
        post_frontier = teacher.iloc[frontier:] if frontier is not None else pd.Series(dtype=float)
        last_three = safe_mean(post_frontier.iloc[-3:]) if len(post_frontier) else float("nan")
        kl_weighted = safe_mean(reverse_kl)
        if response_tokens.sum() > 0:
            kl_weighted = float((reverse_kl.fillna(0) * response_tokens).sum() / response_tokens.sum())
        retained_mask = group["loss_retained"].fillna(False).astype(bool)
        truncated_mask = group["truncate_status"].eq("prompt_truncated")
        records.append(
            {
                "trajectory_id": trajectory_id,
                "training_step": int(first["training_step"]),
                "model_version": int(first["student_model_version"]),
                "game_id": first["game_id"],
                "task_type": first["task_type"],
                "task_success": bool(first["task_success"]),
                "env_done": bool(first["env_done"]),
                "env_lost": bool(first["env_lost"]),
                "env_rounds": int(first["env_rounds"]),
                "full_turns": len(group),
                "retained_turns": retained,
                "retained_fraction": retained / len(group),
                "frontier_triggered": frontier is not None,
                "frontier_turn": frontier,
                "baseline_teacher_entropy": baseline,
                "crossing_teacher_entropy": crossing_entropy,
                "suffix_teacher_entropy": float(finite_suffix.mean()) if finite_suffix.size else float("nan"),
                "last3_post_frontier_entropy": last_three,
                "teacher_entropy": safe_mean(teacher),
                "student_entropy": safe_mean(student),
                "sampled_reverse_kl": kl_weighted,
                "valid_action_rate": float(group["action_valid"].fillna(False).mean()),
                "retained_valid_action_rate": float(group.loc[retained_mask, "action_valid"].fillna(False).mean()),
                "dropped_valid_action_rate": float(group.loc[~retained_mask, "action_valid"].fillna(False).mean())
                if (~retained_mask).any()
                else float("nan"),
                "prompt_truncated_turns": int(truncated_mask.sum()),
                "prompt_truncated": bool(truncated_mask.any()),
                "prompt_truncated_retained_turns": int((truncated_mask & retained_mask).sum()),
                "response_tokens": int(response_tokens.sum()),
            }
        )
        if frontier is not None and np.isfinite(baseline):
            for _, row in group.iterrows():
                entropy = row.get("teacher_entropy_topk")
                if entropy is None or not np.isfinite(float(entropy)):
                    continue
                relative_turn = int(row["turn"]) - frontier
                if -8 <= relative_turn <= 15:
                    aligned_records.append(
                        {
                            "relative_turn": relative_turn,
                            "delta_teacher_entropy": float(entropy) - baseline,
                            "task_success": bool(first["task_success"]),
                            "model_version": int(first["student_model_version"]),
                        }
                    )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(aligned_records)


def aggregate_step(rows: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    row_metrics = (
        rows.assign(
            student_entropy_topk=pd.to_numeric(rows["student_entropy_topk"], errors="coerce"),
            teacher_entropy_topk=pd.to_numeric(rows["teacher_entropy_topk"], errors="coerce"),
            sampled_reverse_kl=pd.to_numeric(rows["sampled_reverse_kl"], errors="coerce"),
            prompt_truncated=rows["truncate_status"].eq("prompt_truncated"),
        )
        .groupby("training_step")
        .agg(
            teacher_entropy=("teacher_entropy_topk", "mean"),
            student_entropy=("student_entropy_topk", "mean"),
            sampled_reverse_kl=("sampled_reverse_kl", "mean"),
            valid_action_rate=("action_valid", "mean"),
            prompt_truncated_turn_rate=("prompt_truncated", "mean"),
            recorded_turns=("turn", "size"),
        )
        .reset_index()
    )
    trajectory_metrics = (
        trajectories.groupby("training_step")
        .agg(
            model_version=("model_version", "mean"),
            trajectories=("trajectory_id", "size"),
            success_rate=("task_success", "mean"),
            frontier_trigger_rate=("frontier_triggered", "mean"),
            retained_fraction=("retained_fraction", "mean"),
            mean_env_rounds=("env_rounds", "mean"),
            mean_full_turns=("full_turns", "mean"),
            mean_retained_turns=("retained_turns", "mean"),
        )
        .reset_index()
    )
    result = trajectory_metrics.merge(row_metrics, on="training_step", how="left")
    for column in (
        "success_rate",
        "frontier_trigger_rate",
        "retained_fraction",
        "teacher_entropy",
        "student_entropy",
        "sampled_reverse_kl",
        "valid_action_rate",
        "prompt_truncated_turn_rate",
        "mean_env_rounds",
    ):
        result[f"{column}_smooth"] = result[column].rolling(7, center=True, min_periods=3).mean()
    return result


def aggregate_turn(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["teacher_entropy_topk"] = pd.to_numeric(data["teacher_entropy_topk"], errors="coerce")
    data["student_entropy_topk"] = pd.to_numeric(data["student_entropy_topk"], errors="coerce")
    return (
        data.groupby(["turn", "task_success"])
        .agg(
            teacher_entropy=("teacher_entropy_topk", "mean"),
            student_entropy=("student_entropy_topk", "mean"),
            valid_action_rate=("action_valid", "mean"),
            trajectory_count=("trajectory_id", "nunique"),
        )
        .reset_index()
    )


def parse_runtime(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    explorer = []
    trainer = []
    first_train = None
    last_train = None
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ansi.sub("", raw)
            stamp_match = re.search(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d)", line)
            stamp = None
            if stamp_match:
                stamp = datetime.strptime("2026-" + stamp_match.group(1), "%Y-%m-%d %H:%M:%S")
            train_event = re.search(r"Training at step (\d+) (started|finished)", line)
            if train_event and stamp:
                if first_train is None:
                    first_train = stamp
                if train_event.group(2) == "finished":
                    last_train = stamp
            if "rollout/model_version" in line and "time/wait_explore_step" in line:
                step_match = re.search(r"Step (\d+):", line)
                values = {}
                for key in (
                    "rollout/model_version",
                    "time/wait_explore_step",
                    "experience_pipeline/experience_count",
                    "rollout/time/run_execution/mean",
                ):
                    match = re.search(rf"'{re.escape(key)}': ([0-9.eE+-]+)", line)
                    values[key] = float(match.group(1)) if match else float("nan")
                if step_match:
                    explorer.append({"explorer_step": int(step_match.group(1)), **values})
            if "time/train_step" in line and "perf/throughput" in line:
                step_match = re.search(r"Step (\d+):", line)
                values = {}
                for key in (
                    "time/read_experience",
                    "time/train_step",
                    "sample/model_version/mean",
                    "actor/final_loss",
                    "actor/grad_norm",
                    "perf/throughput",
                    "prompt_length/clip_ratio",
                ):
                    match = re.search(rf"'{re.escape(key)}': ([0-9.eE+-]+)", line)
                    values[key] = float(match.group(1)) if match else float("nan")
                if step_match:
                    trainer.append({"trainer_step": int(step_match.group(1)), **values})
    wall_seconds = (last_train - first_train).total_seconds() if first_train and last_train else float("nan")
    runtime = {
        "first_training_timestamp": first_train.isoformat() if first_train else None,
        "last_training_timestamp": last_train.isoformat() if last_train else None,
        "training_wall_seconds_step11_to250": wall_seconds,
        "training_wall_hours_step11_to250": wall_seconds / 3600 if np.isfinite(wall_seconds) else None,
        "optimizer_updates_per_hour": 240 / (wall_seconds / 3600) if wall_seconds > 0 else None,
    }
    return pd.DataFrame(explorer), pd.DataFrame(trainer), runtime


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_training_overview(by_step: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    x = by_step["model_version"]
    axes[0, 0].plot(x, by_step["success_rate_smooth"], label="Train rollout success", lw=2)
    axes[0, 0].plot(x, by_step["valid_action_rate_smooth"], label="Valid action rate", lw=2)
    style_axis(axes[0, 0], "A. On-policy rollout quality", "Student model version", "Rate")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(x, by_step["teacher_entropy_smooth"], label="Teacher", lw=2)
    axes[0, 1].plot(x, by_step["student_entropy_smooth"], label="Student", lw=2)
    style_axis(axes[0, 1], "B. Response top-16 partial entropy", "Student model version", "Entropy")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(x, by_step["frontier_trigger_rate_smooth"], label="Frontier triggered", lw=2)
    axes[1, 0].plot(x, by_step["retained_fraction_smooth"], label="Turns retained", lw=2)
    style_axis(axes[1, 0], "C. Adaptive intervention", "Student model version", "Fraction")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(x, by_step["prompt_truncated_turn_rate_smooth"], label="Prompt-truncated turns", lw=2)
    ax2 = axes[1, 1].twinx()
    ax2.plot(x, by_step["sampled_reverse_kl_smooth"], color="C3", label="Sampled reverse KL", lw=2)
    style_axis(axes[1, 1], "D. Context pressure and distillation gap", "Student model version", "Truncated-turn rate")
    ax2.set_ylabel("Sampled reverse KL")
    lines = axes[1, 1].lines + ax2.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], frameon=False)
    fig.suptitle("Entropy-adaptive v1 — training diagnostics", fontsize=16, weight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_turn_outcome(by_turn: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.8), sharex=True, constrained_layout=True)
    for success, label, color in ((False, "Failure", "C3"), (True, "Success", "C2")):
        part = by_turn[by_turn["task_success"] == success]
        axes[0].plot(part["turn"] + 1, part["teacher_entropy"], label=label, color=color, lw=2)
        axes[1].plot(part["turn"] + 1, part["trajectory_count"], label=label, color=color, lw=2)
    style_axis(axes[0], "Teacher entropy by final training-rollout outcome", "Environment turn", "Teacher entropy")
    axes[0].legend(frameon=False)
    style_axis(axes[1], "Late-turn denominator (survivorship)", "Environment turn", "Trajectories contributing")
    axes[1].legend(frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(aligned: pd.DataFrame, trajectories: pd.DataFrame, output: Path) -> None:
    summary = (
        aligned.groupby("relative_turn")["delta_teacher_entropy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem95"] = 1.96 * summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    triggered = trajectories[trajectories["frontier_triggered"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    x = summary["relative_turn"].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem95"].to_numpy(dtype=float)
    axes[0].plot(x, mean, lw=2.2, color="C0")
    axes[0].fill_between(x, mean - sem, mean + sem, color="C0", alpha=0.18)
    axes[0].axvline(0, color="black", ls="--", lw=1, label="Detected frontier")
    axes[0].axhline(0.175, color="C3", ls=":", lw=1.5, label="Threshold")
    style_axis(axes[0], "A. Entropy aligned to first frontier", "Turn relative to frontier", "Teacher entropy − early baseline")
    axes[0].legend(frameon=False)

    values = triggered["frontier_turn"].dropna().astype(int) + 1
    bins = np.arange(3.5, 31.5, 1)
    axes[1].hist(values, bins=bins, color="C1", alpha=0.82, edgecolor="white")
    style_axis(axes[1], "B. Frontier position distribution", "Environment turn", "Triggered trajectories")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows: pd.DataFrame, trajectories: pd.DataFrame, output: Path) -> dict:
    max_version = int(trajectories["model_version"].max())
    panel = trajectories[trajectories["model_version"] >= max_version - 30].copy()
    panel["sort_frontier"] = panel["frontier_turn"].fillna(99)
    panel = panel.sort_values(["task_success", "frontier_triggered", "sort_frontier", "model_version"], ascending=[True, False, True, True])
    ids = panel["trajectory_id"].tolist()
    matrix = np.full((len(ids), 30), np.nan)
    frontier_points = []
    success_boundary = int((~panel["task_success"]).sum())
    grouped = {key: group.sort_values("turn") for key, group in rows[rows["trajectory_id"].isin(ids)].groupby("trajectory_id")}
    for row_index, trajectory_id in enumerate(ids):
        group = grouped[trajectory_id]
        entropy = pd.to_numeric(group["teacher_entropy_topk"], errors="coerce").to_numpy(dtype=float)
        baseline = np.nanmean(entropy[:3]) if np.isfinite(entropy[:3]).all() else np.nan
        width = min(30, len(entropy))
        matrix[row_index, :width] = entropy[:width] - baseline
        frontier = panel.iloc[row_index]["frontier_turn"]
        if pd.notna(frontier):
            frontier_points.append((float(frontier), row_index))
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    finite_values = matrix[np.isfinite(matrix)]
    limit = float(np.quantile(np.abs(finite_values), 0.98)) if finite_values.size else 1.0
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-limit, vmax=limit)
    if frontier_points:
        ax.scatter([x for x, _ in frontier_points], [y for _, y in frontier_points], s=5, c="black", alpha=0.65, label="Detected frontier")
    if 0 < success_boundary < len(panel):
        ax.axhline(success_boundary - 0.5, color="black", lw=1.5)
        ax.text(29.4, success_boundary - 2, "failures ↑", ha="right", va="bottom", fontsize=9)
        ax.text(29.4, success_boundary + 2, "successes ↓", ha="right", va="top", fontsize=9)
    ax.set_title(f"Latest-policy trajectory heterogeneity (model versions {max_version - 30}–{max_version})", loc="left", weight="bold")
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
    }


def plot_runtime(explorer: pd.DataFrame, trainer: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    if not explorer.empty:
        axes[0].plot(explorer["rollout/model_version"], explorer["time/wait_explore_step"], color="C0", alpha=0.42)
        smooth = explorer["time/wait_explore_step"].rolling(9, center=True, min_periods=3).mean()
        axes[0].plot(explorer["rollout/model_version"], smooth, color="C0", lw=2.3)
    style_axis(axes[0], "A. Explorer batch wall time", "Student model version", "Seconds per 16 trajectories")
    if not trainer.empty:
        axes[1].plot(trainer["trainer_step"], trainer["time/train_step"], color="C1", alpha=0.42)
        smooth = trainer["time/train_step"].rolling(11, center=True, min_periods=3).median()
        axes[1].plot(trainer["trainer_step"], smooth, color="C1", lw=2.3)
    style_axis(axes[1], "B. Optimizer update time", "Trainer step", "Seconds")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary(rows: pd.DataFrame, trajectories: pd.DataFrame, metadata: dict, runtime: dict, heatmap: dict) -> dict:
    triggered = trajectories[trajectories["frontier_triggered"]]
    total_turns = int(trajectories["full_turns"].sum())
    retained_turns = int(trajectories["retained_turns"].sum())
    successes = int(trajectories["task_success"].sum())
    success_ci = proportion_ci(successes, len(trajectories))
    early = trajectories[trajectories["model_version"] <= 30]
    late = trajectories[trajectories["model_version"] >= trajectories["model_version"].max() - 30]
    return {
        **metadata,
        **runtime,
        **heatmap,
        "training_step_min": int(rows["training_step"].min()),
        "training_step_max": int(rows["training_step"].max()),
        "model_version_min": int(rows["student_model_version"].min()),
        "model_version_max": int(rows["student_model_version"].max()),
        "trajectory_count": len(trajectories),
        "turn_row_count": len(rows),
        "task_success_count": successes,
        "training_rollout_success_rate": successes / len(trajectories),
        "training_rollout_success_wilson95": list(success_ci),
        "early_model_version_le30_success_rate": float(early["task_success"].mean()),
        "late_last30_versions_success_rate": float(late["task_success"].mean()),
        "frontier_triggered_count": len(triggered),
        "frontier_trigger_rate": len(triggered) / len(trajectories),
        "full_turn_count": total_turns,
        "retained_turn_count": retained_turns,
        "dropped_turn_count": total_turns - retained_turns,
        "retained_turn_fraction": retained_turns / total_turns,
        "prompt_truncated_trajectory_count": int(trajectories["prompt_truncated"].sum()),
        "prompt_truncated_turn_count": int(trajectories["prompt_truncated_turns"].sum()),
        "prompt_truncated_retained_turn_count": int(trajectories["prompt_truncated_retained_turns"].sum()),
        "mean_full_turns": float(trajectories["full_turns"].mean()),
        "mean_retained_turns": float(trajectories["retained_turns"].mean()),
        "mean_teacher_entropy": safe_mean(trajectories["teacher_entropy"]),
        "mean_student_entropy": safe_mean(trajectories["student_entropy"]),
        "mean_sampled_reverse_kl": safe_mean(trajectories["sampled_reverse_kl"]),
        "triggered_baseline_teacher_entropy": safe_mean(triggered["baseline_teacher_entropy"]),
        "triggered_crossing_teacher_entropy": safe_mean(triggered["crossing_teacher_entropy"]),
        "triggered_suffix_teacher_entropy": safe_mean(triggered["suffix_teacher_entropy"]),
        "triggered_last3_post_frontier_entropy": safe_mean(triggered["last3_post_frontier_entropy"]),
        "retained_valid_action_rate": safe_mean(trajectories["retained_valid_action_rate"]),
        "dropped_valid_action_rate": safe_mean(triggered["dropped_valid_action_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, metadata = load_rows(args.diagnostics)
    rows = rows.drop_duplicates(["trajectory_id", "turn"], keep="last").copy()
    metadata["selected_row_count_after_resume_overlap"] = len(rows)
    trajectories, aligned = trajectory_table(rows)
    by_step = aggregate_step(rows, trajectories)
    by_turn = aggregate_turn(rows)
    explorer, trainer, runtime = parse_runtime(args.log)

    save_csv(trajectories, args.output_dir / "trajectory_summary.csv")
    save_csv(by_step, args.output_dir / "diagnostics_by_explorer_step.csv")
    save_csv(by_turn, args.output_dir / "diagnostics_by_trajectory_turn.csv")
    save_csv(aligned, args.output_dir / "frontier_aligned_rows.csv")
    save_csv(explorer, args.output_dir / "explorer_timing.csv")
    save_csv(trainer, args.output_dir / "trainer_timing.csv")

    plot_training_overview(by_step, args.output_dir / "training_overview.png")
    plot_turn_outcome(by_turn, args.output_dir / "teacher_entropy_by_turn_outcome.png")
    plot_frontier(aligned, trajectories, args.output_dir / "frontier_mechanism.png")
    heatmap = plot_heatmap(rows, trajectories, args.output_dir / "teacher_entropy_frontier_heatmap_latest.png")
    plot_runtime(explorer, trainer, args.output_dir / "pipeline_timing.png")

    summary = build_summary(rows, trajectories, metadata, runtime, heatmap)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
