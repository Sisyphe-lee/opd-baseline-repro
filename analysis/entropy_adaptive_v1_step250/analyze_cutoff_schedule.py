#!/usr/bin/env python3
"""Analyze how adaptive-v1 cutoff decisions change across model versions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "trajectory_summary.csv"
CSV_OUT = HERE / "cutoff_schedule_by_model_version.csv"
FIG_OUT = HERE / "cutoff_schedule_over_training.png"

VERSION_BINS = [-1, 30, 60, 90, 120, 150, 180, 210, 250]
VERSION_LABELS = [
    "0–30", "31–60", "61–90", "91–120",
    "121–150", "151–180", "181–210", "211–250",
]


def main() -> None:
    trajectories = pd.read_csv(SOURCE)
    trajectories["version_bin"] = pd.cut(
        trajectories["model_version"], bins=VERSION_BINS, labels=VERSION_LABELS
    )
    # This is the curriculum decision, not the realized trajectory length:
    # no frontier means the selector permits the full horizon of 30.
    trajectories["implicit_policy_limit"] = np.where(
        trajectories["frontier_triggered"],
        trajectories["frontier_turn"],
        30,
    )

    records = []
    for version_bin, group in trajectories.groupby("version_bin", observed=True):
        triggered = group.loc[group["frontier_triggered"]]
        cutoff = triggered["frontier_turn"] + 1
        records.append(
            {
                "model_version_bin": str(version_bin),
                "trajectory_count": len(group),
                "success_rate": group["task_success"].mean(),
                "frontier_trigger_rate": group["frontier_triggered"].mean(),
                "triggered_count": len(triggered),
                "triggered_crossing_mean": cutoff.mean(),
                "triggered_crossing_p25": cutoff.quantile(0.25),
                "triggered_crossing_median": cutoff.median(),
                "triggered_crossing_p75": cutoff.quantile(0.75),
                "implicit_policy_limit_mean": group["implicit_policy_limit"].mean(),
                "implicit_policy_limit_median": group["implicit_policy_limit"].median(),
                "realized_retained_turns_mean": group["retained_turns"].mean(),
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary.to_csv(CSV_OUT, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    x = np.arange(len(summary))
    axes[0].plot(x, summary["frontier_trigger_rate"], marker="o", lw=2, label="Frontier triggered")
    axes[0].plot(x, summary["success_rate"], marker="o", lw=2, label="Training rollout success")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction")
    axes[0].set_title("A. Triggering becomes rarer", loc="left", weight="bold")
    axes[0].legend(frameon=False)

    median = summary["triggered_crossing_median"].to_numpy(dtype=float)
    lower = median - summary["triggered_crossing_p25"].to_numpy(dtype=float)
    upper = summary["triggered_crossing_p75"].to_numpy(dtype=float) - median
    axes[1].errorbar(x, median, yerr=[lower, upper], fmt="o-", lw=2, capsize=4)
    axes[1].set_ylim(0, 31)
    axes[1].set_ylabel("Crossing turn (1-based)")
    axes[1].set_title("B. Crossing position when triggered", loc="left", weight="bold")

    axes[2].plot(
        x, summary["implicit_policy_limit_mean"], marker="o", lw=2,
        label="Implicit policy limit",
    )
    axes[2].plot(
        x, summary["realized_retained_turns_mean"], marker="o", lw=2,
        label="Realized retained turns",
    )
    axes[2].axhline(30, color="black", ls=":", lw=1)
    axes[2].set_ylim(0, 31)
    axes[2].set_ylabel("Turns")
    axes[2].set_title("C. Effective curriculum opens", loc="left", weight="bold")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.set_xticks(x, summary["model_version_bin"], rotation=35, ha="right")
        axis.set_xlabel("Student model version")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
