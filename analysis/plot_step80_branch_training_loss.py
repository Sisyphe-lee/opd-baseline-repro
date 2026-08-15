#!/usr/bin/env python3
"""Plot final PPO training loss for the three tau=0.1 step-80 branches."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    REPO_ROOT
    / "analysis/entropy_adaptive_v1_all_experiments/supplements"
    / "step80_anneal_vs_fixed"
)
LOGS = {
    "Fixed tau=0.1": REPO_ROOT
    / "runs/experiments/entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4"
    / "checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1"
    / "entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4_seed42/log/trainer.log",
    "Linear anneal": REPO_ROOT
    / "runs/experiments/entropy_adaptive_v1_t0100_step80_linear_anneal_to_full_step160_250step_4gpu_s1t1_r4"
    / "checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1"
    / "entropy_adaptive_v1_t0100_step80_linear_anneal_to_full_step160_250step_4gpu_s1t1_r4_seed42/log/trainer.log",
    "Immediate Vanilla": REPO_ROOT
    / "runs/experiments/entropy_adaptive_v1_t0100_step80_immediate_full_250step_4gpu_s1t1_r4"
    / "checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1"
    / "entropy_adaptive_v1_t0100_step80_immediate_full_250step_4gpu_s1t1_r4_seed42/log/trainer.log",
}
COLORS = {
    "Fixed tau=0.1": "#7A5195",
    "Linear anneal": "#EF5675",
    "Immediate Vanilla": "#D45087",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_trainer_log(path: Path, method: str) -> pd.DataFrame:
    records = []
    pattern = re.compile(r"Step (\d+): (\{.*'actor/final_loss'.*\})")
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ansi.sub("", raw)
            match = pattern.search(line)
            if match is None:
                continue
            step = int(match.group(1))
            payload = match.group(2)
            values = {}
            for key in (
                "actor/pg_loss",
                "actor/final_loss",
                "actor/ppo_kl",
                "actor/grad_norm",
                "actor/pg_clipfrac",
                "actor/lr",
                "perf/total_num_tokens",
                "sample/model_version/mean",
            ):
                value_match = re.search(
                    rf"'{re.escape(key)}': ([0-9.eE+-]+)", payload
                )
                values[key] = (
                    float(value_match.group(1)) if value_match else float("nan")
                )
            records.append({"method": method, "trainer_step": step, **values})
    frame = pd.DataFrame.from_records(records).drop_duplicates(
        subset=["trainer_step"], keep="last"
    )
    if frame.empty:
        raise ValueError(f"No trainer metrics found in {path}")
    return frame.sort_values("trainer_step").reset_index(drop=True)


def splice_profiles(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fixed = raw["Fixed tau=0.1"]
    frames = [fixed]
    for method in ("Linear anneal", "Immediate Vanilla"):
        prefix = fixed.loc[fixed["trainer_step"] <= 80].copy()
        suffix = raw[method].loc[raw[method]["trainer_step"] >= 81].copy()
        profile = pd.concat([prefix, suffix], ignore_index=True)
        profile["method"] = method
        frames.append(profile)
    result = pd.concat(frames, ignore_index=True)
    result["final_loss_rolling_mean_11"] = result.groupby("method", sort=False)[
        "actor/final_loss"
    ].transform(lambda series: series.rolling(11, center=True, min_periods=3).mean())
    return result


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE9", alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_loss(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.5), constrained_layout=True)
    methods = ("Fixed tau=0.1", "Linear anneal", "Immediate Vanilla")

    ax = axes[0]
    for method in methods:
        part = data.loc[data["method"].eq(method)]
        ax.plot(
            part["trainer_step"],
            part["actor/final_loss"],
            color=COLORS[method],
            alpha=0.20,
            linewidth=0.9,
        )
        ax.plot(
            part["trainer_step"],
            part["final_loss_rolling_mean_11"],
            color=COLORS[method],
            linewidth=2.3,
            label=f"{method} (11-step rolling mean)",
        )
    ax.axvspan(1, 80, color="#4C566A", alpha=0.035)
    ax.axvline(80, color="#4C566A", linestyle="--", linewidth=1.3)
    ax.text(40, 0.128, "shared tau=0.1 prefix", ha="center", fontsize=9)
    ax.text(83, 0.128, "branch", fontsize=9)
    ax.set_xlim(0, 251)
    ax.set_ylim(0, 0.14)
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Actor final loss")
    ax.set_title("A. Full training course", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=1)
    style_axis(ax)

    ax = axes[1]
    for method in methods:
        part = data.loc[
            data["method"].eq(method) & data["trainer_step"].ge(75)
        ]
        ax.plot(
            part["trainer_step"],
            part["actor/final_loss"],
            color=COLORS[method],
            alpha=0.17,
            linewidth=0.8,
        )
        ax.plot(
            part["trainer_step"],
            part["final_loss_rolling_mean_11"],
            color=COLORS[method],
            linewidth=2.3,
            label=method,
        )
    ax.axvspan(75, 80, color="#4C566A", alpha=0.035)
    ax.axvline(80, color="#4C566A", linestyle="--", linewidth=1.3)
    ax.set_xlim(75, 251)
    ax.set_ylim(0, 0.072)
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Actor final loss")
    ax.set_title("B. Step-80 continuation detail", loc="left", weight="bold")
    ax.legend(frameon=False)
    style_axis(ax)

    fig.suptitle(
        "PPO training loss: fixed tau=0.1 vs step-80 continuations",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {method: parse_trainer_log(path, method) for method, path in LOGS.items()}
    if len(raw["Fixed tau=0.1"]) != 250:
        raise ValueError("Fixed tau=0.1 must contain exactly 250 trainer steps")
    for method in ("Linear anneal", "Immediate Vanilla"):
        steps = raw[method]["trainer_step"]
        if len(raw[method]) != 170 or (int(steps.min()), int(steps.max())) != (81, 250):
            raise ValueError(f"{method} does not contain the expected step81--250 suffix")

    data = splice_profiles(raw)
    csv_path = OUTPUT_DIR / "step80_branch_training_loss.csv"
    data.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    plot_path = OUTPUT_DIR / "step80_branch_training_loss.png"
    plot_loss(data, plot_path)

    summary = {}
    for method, part in data.groupby("method", sort=False):
        post = part.loc[part["trainer_step"] >= 81, "actor/final_loss"]
        summary[method] = {
            "step_count": int(len(part)),
            "post_step80_mean": float(post.mean()),
            "post_step80_median": float(post.median()),
            "last50_mean": float(part.tail(50)["actor/final_loss"].mean()),
            "step250": float(
                part.loc[part["trainer_step"].eq(250), "actor/final_loss"].iloc[0]
            ),
        }
    summary_path = OUTPUT_DIR / "step80_branch_training_loss_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "loss_metric": "actor/final_loss",
        "smoothing": "centered 11-step rolling arithmetic mean; raw step values retained",
        "splice": "Linear anneal and Immediate Vanilla use fixed tau=0.1 steps 1-80, then their own steps 81-250",
        "inputs": {
            method: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for method, path in LOGS.items()
        },
        "outputs": [plot_path.name, csv_path.name, summary_path.name],
    }
    (OUTPUT_DIR / "step80_branch_training_loss_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
