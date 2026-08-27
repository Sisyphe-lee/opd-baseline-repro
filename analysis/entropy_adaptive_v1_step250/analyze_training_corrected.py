#!/usr/bin/env python3
"""Generate prompt-truncation-corrected adaptive-v1 plots and audit figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import analyze_training_nontruncated as CLEAN


ORIGINAL = CLEAN.ORIGINAL


def plot_training_overview(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(frame["training_step"], frame["success_rate_smooth"], color="C2")
    ORIGINAL.style_axis(
        axes[0], "A. Training rollout success", "Explorer step", "Success rate"
    )
    axes[0].set_ylim(0, 1)

    axes[1].plot(
        frame["training_step"], frame["teacher_entropy_smooth"],
        color="C0", label="Teacher",
    )
    axes[1].plot(
        frame["training_step"], frame["student_entropy_smooth"],
        color="C1", label="Student",
    )
    ORIGINAL.style_axis(
        axes[1], "B. Response top-16 entropy — real responses only",
        "Explorer step", "Partial entropy",
    )
    axes[1].legend(frameon=False)

    axes[2].plot(
        frame["training_step"], frame["frontier_trigger_rate_smooth"],
        color="C3", label="Frontier trigger rate",
    )
    axes[2].plot(
        frame["training_step"], frame["retained_fraction_smooth"],
        color="C4", label="Retained fraction",
    )
    ORIGINAL.style_axis(
        axes[2], "C. Stateless per-rollout selection", "Explorer step", "Fraction"
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(frameon=False)

    axes[3].plot(
        frame["training_step"], frame["sampled_reverse_kl_smooth"],
        color="C5", label="Sampled reverse KL (real responses)",
    )
    axes[3].plot(
        frame["training_step"], frame["prompt_truncated_turn_rate_smooth"],
        color="C6", label="Prompt placeholders (raw rows)",
    )
    ORIGINAL.style_axis(
        axes[3], "D. Divergence and truncation contamination",
        "Explorer step", "Mean / fraction",
    )
    axes[3].legend(frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_contamination_audit(
    rows: pd.DataFrame,
    aligned_contaminated: pd.DataFrame,
    aligned_real: pd.DataFrame,
    threshold: float,
    output: Path,
) -> None:
    failures = rows.loc[~rows["task_success"]].copy()
    failures["prompt_placeholder"] = failures["truncate_status"].eq("prompt_truncated")
    failures["teacher_entropy_topk"] = pd.to_numeric(
        failures["teacher_entropy_topk"], errors="coerce"
    )
    all_turn = failures.groupby("turn")["teacher_entropy_topk"].mean()
    real_turn = failures.loc[~failures["prompt_placeholder"]].groupby("turn")[
        "teacher_entropy_topk"
    ].mean()
    truncation_rate = failures.groupby("turn")["prompt_placeholder"].mean()
    contaminated = aligned_contaminated.groupby("relative_turn")[
        "delta_teacher_entropy"
    ].mean()
    corrected = aligned_real.groupby("relative_turn")["delta_teacher_entropy"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(all_turn.index + 1, all_turn.values, color="C3", lw=2,
                 label="All recorded rows")
    axes[0].plot(real_turn.index + 1, real_turn.values, color="C0", lw=2,
                 label="Real generated responses")
    twin = axes[0].twinx()
    twin.plot(truncation_rate.index + 1, truncation_rate.values, color="C7",
              lw=1.6, ls="--", label="Prompt-placeholder fraction")
    ORIGINAL.style_axis(
        axes[0], "A. Failures: apparent late entropy drop",
        "Environment turn", "Teacher top-16 partial entropy",
    )
    twin.set_ylabel("Prompt-placeholder fraction")
    twin.set_ylim(0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[0].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8)

    axes[1].plot(contaminated.index, contaminated.values, color="C3", lw=2,
                 label="Including prompt placeholders")
    axes[1].plot(corrected.index, corrected.values, color="C0", lw=2,
                 label="Real generated responses only")
    axes[1].axvline(0, color="black", ls="--", lw=1)
    axes[1].axhline(threshold, color="C4", ls=":", lw=1.5)
    ORIGINAL.style_axis(
        axes[1], "B. Frontier-aligned entropy drift",
        "Turn relative to detected frontier", "Teacher entropy − early baseline",
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = ORIGINAL.load_rows(args.diagnostics)
    threshold = CLEAN.unique_frontier_threshold(rows)
    rows = rows.drop_duplicates(["trajectory_id", "turn"], keep="last").copy()
    trajectories, aligned_contaminated = ORIGINAL.trajectory_table(rows)
    by_step = CLEAN.corrected_by_step(rows, trajectories)
    by_turn = CLEAN.corrected_by_turn(rows)
    aligned_real = CLEAN.corrected_aligned(rows, trajectories)

    CLEAN.archive_contaminated(
        args.output_dir,
        ["training_overview.png", "teacher_entropy_by_turn_outcome.png",
         "frontier_mechanism.png", "teacher_entropy_frontier_heatmap_latest.png"],
    )
    ORIGINAL.save_csv(
        by_step, args.output_dir / "diagnostics_by_explorer_step_nontruncated.csv"
    )
    ORIGINAL.save_csv(
        by_turn, args.output_dir / "diagnostics_by_trajectory_turn_nontruncated.csv"
    )
    ORIGINAL.save_csv(
        aligned_real, args.output_dir / "frontier_aligned_rows_nontruncated.csv"
    )
    plot_training_overview(by_step, args.output_dir / "training_overview.png")
    CLEAN.plot_turn_outcome(
        by_turn, args.output_dir / "teacher_entropy_by_turn_outcome.png"
    )
    CLEAN.plot_frontier(
        aligned_real,
        trajectories,
        threshold,
        args.output_dir / "frontier_mechanism.png",
    )
    heatmap = CLEAN.plot_heatmap(
        rows, trajectories, args.output_dir / "teacher_entropy_frontier_heatmap_latest.png"
    )
    plot_contamination_audit(
        rows, aligned_contaminated, aligned_real, threshold,
        args.output_dir / "prompt_truncation_contamination_audit.png",
    )
    summary = CLEAN.corrected_summary(rows, trajectories, aligned_real, heatmap)
    (args.output_dir / "summary_nontruncated.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
