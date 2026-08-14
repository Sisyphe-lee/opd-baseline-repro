#!/usr/bin/env python3
"""Diagnose the step-80 linear anneal against fixed tau=0.1 and immediate full."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_REPO_ROOT))

from trinity.common.experience import Experience


REPO_ROOT = MODULE_REPO_ROOT
DEFAULT_FIXED = (
    REPO_ROOT
    / "runs/experiments/entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4"
    / "diagnostics/trajectory_metrics.jsonl"
)
DEFAULT_ANNEAL = (
    REPO_ROOT
    / "runs/experiments/entropy_adaptive_v1_t0100_step80_linear_anneal_to_full_step160_250step_4gpu_s1t1_r4"
    / "diagnostics/trajectory_metrics.jsonl"
)
DEFAULT_IMMEDIATE = (
    REPO_ROOT
    / "runs/experiments/entropy_adaptive_v1_t0100_step80_immediate_full_250step_4gpu_s1t1_r4"
    / "diagnostics/trajectory_metrics.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "analysis/entropy_adaptive_v1_all_experiments/supplements"
    / "step80_anneal_vs_fixed"
)
DEFAULT_TCOD_BUFFER = REPO_ROOT / "checkpoints/tcod_f2b_step250/buffer/explorer_output.db"
DEFAULT_TCOD_LOG = (
    REPO_ROOT
    / "results/training/tcod_f2b_step250/launcher_logs/f2b_resume80_20260808T0615Z.log"
)
CURRICULUM_SCRIPT = REPO_ROOT / "analysis/plot_adaptive_v1_curriculum.py"
COLORS = {
    "TCOD F2B": "#F58518",
    "Fixed tau=0.1": "#7A5195",
    "Linear anneal": "#EF5675",
    "Immediate Vanilla": "#D45087",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-diagnostics", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--anneal-diagnostics", type=Path, default=DEFAULT_ANNEAL)
    parser.add_argument("--immediate-diagnostics", type=Path, default=DEFAULT_IMMEDIATE)
    parser.add_argument("--tcod-buffer", type=Path, default=DEFAULT_TCOD_BUFFER)
    parser.add_argument("--tcod-log", type=Path, default=DEFAULT_TCOD_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_curriculum_module():
    spec = importlib.util.spec_from_file_location("adaptive_curriculum_plot", CURRICULUM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_effective_fields(path: Path) -> pd.DataFrame:
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            trajectory_id = row["trajectory_id"]
            if trajectory_id in records:
                continue
            records[trajectory_id] = {
                "trajectory_id": trajectory_id,
                "effective_threshold": row.get(
                    "entropy_frontier_effective_threshold",
                    row.get("entropy_frontier_threshold"),
                ),
                "effective_strategy": row.get(
                    "frontier_strategy_effective", row.get("frontier_strategy")
                ),
                "schedule": row.get("entropy_frontier_schedule", "constant"),
                "schedule_start": row.get(
                    "entropy_frontier_schedule_start_model_version"
                ),
                "schedule_end": row.get(
                    "entropy_frontier_schedule_end_model_version"
                ),
            }
    return pd.DataFrame.from_records(list(records.values())).set_index("trajectory_id")


def load_profile(module, path: Path, label: str) -> pd.DataFrame:
    dataset = module.load_adaptive(
        path,
        expected_threshold=0.1,
        label=label,
        max_env_steps=30,
    )
    profile = dataset["profile"].copy()
    rows = dataset["rows"]
    real_turn = ~rows["truncate_status"].eq("prompt_truncated")
    available = real_turn.groupby(rows["trajectory_id"]).sum().astype(int)
    profile["available_turns"] = profile["trajectory_id"].map(available)
    profile["masked_turns"] = (
        profile["available_turns"] - profile["realized_trainable_turns"]
    )
    effective = load_effective_fields(path)
    profile = profile.join(effective, on="trajectory_id", validate="one_to_one")
    profile["method"] = label
    return profile.sort_values(
        ["training_step", "first_source_row_index", "trajectory_id"],
        kind="stable",
    ).reset_index(drop=True)


def load_tcod_profile(path: Path, log_path: Path, label: str = "TCOD F2B") -> pd.DataFrame:
    """Reconstruct post-step80 TCOD trajectories from the frozen replay buffer."""
    branch_versions = {
        int(batch): int(version)
        for batch, version in re.findall(
            r"Step (\d+): \{[^\n]*?'rollout/model_version': ([0-9]+)",
            log_path.read_text(encoding="utf-8", errors="replace"),
        )
    }
    if not branch_versions:
        raise ValueError(f"{label}: no batch-to-model-version mapping in {log_path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    trajectories: dict[tuple[int, str, int], dict] = {}
    try:
        cursor = connection.execute(
            "SELECT id, model_version, experience_bytes FROM pipeline_input ORDER BY id"
        )
        for row_id, model_version, blob in cursor:
            experience = Experience.deserialize(blob)
            match = re.match(r"^(\d+)", str(experience.eid.batch))
            if match is None:
                raise ValueError(f"{label}: cannot parse batch from {experience.eid.batch!r}")
            batch = int(match.group(1))
            if not (
                int(model_version) < 80
                or branch_versions.get(batch) == int(model_version)
            ):
                continue
            task = str(experience.eid.task)
            run = int(experience.eid.run)
            step = int(experience.eid.step)
            key = (batch, task, run)
            record = trajectories.setdefault(
                key,
                {
                    "first_row_id": int(row_id),
                    "model_versions": set(),
                    "available_steps": {},
                    "valid_steps": {},
                    "task_success": False,
                    "prompt_truncated": False,
                },
            )
            record["model_versions"].add(int(model_version))
            nontruncated = experience.truncate_status != "prompt_truncated"
            record["available_steps"][step] = (
                record["available_steps"].get(step, False) or nontruncated
            )
            action_mask = experience.action_mask
            valid = bool(
                nontruncated
                and action_mask is not None
                and action_mask.numel() > 0
                and action_mask.any().item()
            )
            record["valid_steps"][step] = record["valid_steps"].get(step, False) or valid
            record["task_success"] = record["task_success"] or bool(
                experience.metrics.get("env_done", 0.0)
            )
            record["prompt_truncated"] = (
                record["prompt_truncated"] or not nontruncated
            )
    finally:
        connection.close()

    records = []
    for (batch, task, run), record in sorted(
        trajectories.items(), key=lambda item: item[1]["first_row_id"]
    ):
        versions = record["model_versions"]
        if len(versions) != 1:
            raise ValueError(
                f"{label}: trajectory {(batch, task, run)} spans model versions {sorted(versions)}"
            )
        model_version = next(iter(versions))
        horizon = min(1 + batch // 2, 30)
        horizon = int(horizon)
        available_turns = int(sum(record["available_steps"].values()))
        realized_turns = int(sum(record["valid_steps"].values()))
        records.append(
            {
                "method": label,
                "trajectory_id": f"tcod:{batch}:{task}:{run}",
                "training_step": batch,
                "model_version": model_version,
                "first_source_row_index": record["first_row_id"],
                "effective_threshold": np.nan,
                "effective_strategy": "full" if horizon == 30 else "f2b",
                "imposed_horizon": horizon,
                "frontier_triggered": horizon < 30,
                "realized_trainable_turns": realized_turns,
                # Before K=30 the frozen TCOD buffer retains only trainable turns,
                # so the counterfactual available/removed turn counts are not observable.
                "available_turns": available_turns if horizon == 30 else np.nan,
                "masked_turns": 0.0 if horizon == 30 else np.nan,
                "task_success": record["task_success"],
                "prompt_truncated": record["prompt_truncated"],
            }
        )
    profile = pd.DataFrame.from_records(records)
    expected_suffix = len(branch_versions) * 16
    suffix_count = int(profile["model_version"].ge(80).sum())
    if suffix_count != expected_suffix:
        raise ValueError(
            f"{label}: expected {expected_suffix} resume-branch trajectories, "
            f"got {suffix_count}"
        )
    return profile


def splice_step80_prefix(
    original: pd.DataFrame,
    continuation: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Use the original tau=0.1 stream before v80 and branch data from v80 onward."""
    prefix = original.loc[original["model_version"] < 80].copy()
    suffix = continuation.loc[continuation["model_version"] >= 80].copy()
    combined = pd.concat([prefix, suffix], ignore_index=True)
    combined["method"] = label
    return combined.sort_values(
        ["model_version", "training_step", "first_source_row_index", "trajectory_id"],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_version_bins(profile: pd.DataFrame) -> pd.DataFrame:
    post = profile.copy()
    # Keep the terminal v250 observations in the final 245--250 bin rather
    # than plotting a tiny one-version bin whose mean is visually unstable.
    post["version_bin_start"] = ((post["model_version"] // 5) * 5).clip(upper=245)
    grouped = post.groupby("version_bin_start", sort=True)
    result = grouped.agg(
        trajectory_count=("trajectory_id", "size"),
        version_min=("model_version", "min"),
        version_max=("model_version", "max"),
        effective_threshold=("effective_threshold", "mean"),
        full_strategy_fraction=("effective_strategy", lambda x: x.eq("full").mean()),
        frontier_trigger_rate=("frontier_triggered", "mean"),
        imposed_horizon_mean=("imposed_horizon", "mean"),
        full_horizon_fraction=("imposed_horizon", lambda x: x.eq(30).mean()),
        realized_turns_mean=("realized_trainable_turns", "mean"),
        realized_turns_median=("realized_trainable_turns", "median"),
        available_turns_mean=("available_turns", "mean"),
        masked_turns_mean=("masked_turns", "mean"),
        task_success_rate=("task_success", "mean"),
        prompt_truncated_rate=("prompt_truncated", "mean"),
    ).reset_index()
    result.insert(0, "method", profile["method"].iloc[0])
    return result


def aggregate_chunks(profile: pd.DataFrame, chunk_size: int = 16) -> pd.DataFrame:
    post = profile.copy().reset_index(drop=True)
    post["chunk_index"] = np.arange(len(post)) // chunk_size
    grouped = post.groupby("chunk_index", sort=True)
    result = grouped.agg(
        trajectory_count=("trajectory_id", "size"),
        trajectory_start=("trajectory_id", lambda x: int(x.index.min())),
        version_min=("model_version", "min"),
        version_max=("model_version", "max"),
        effective_threshold=("effective_threshold", "mean"),
        frontier_trigger_rate=("frontier_triggered", "mean"),
        realized_turns_mean=("realized_trainable_turns", "mean"),
        realized_turns_median=("realized_trainable_turns", "median"),
        available_turns_mean=("available_turns", "mean"),
        masked_turns_mean=("masked_turns", "mean"),
        task_success_rate=("task_success", "mean"),
        prompt_truncated_rate=("prompt_truncated", "mean"),
    ).reset_index()
    result["trajectory_start"] = result["chunk_index"] * chunk_size
    result.insert(0, "method", profile["method"].iloc[0])
    return result


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE9", alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_detail(
    profiles: dict[str, pd.DataFrame],
    bins: pd.DataFrame,
    chunks: pd.DataFrame,
    output: Path,
) -> None:
    del chunks
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    methods = ("TCOD F2B", "Fixed tau=0.1", "Linear anneal", "Immediate Vanilla")
    line_styles = {
        "TCOD F2B": "--",
        "Fixed tau=0.1": "-",
        "Linear anneal": "-",
        "Immediate Vanilla": ":",
    }
    markers = {
        "TCOD F2B": "s",
        "Fixed tau=0.1": "o",
        "Linear anneal": "o",
        "Immediate Vanilla": "x",
    }

    def mark_step80(axis: plt.Axes) -> None:
        axis.axvspan(0, 80, color="#4C566A", alpha=0.035, zorder=0)
        axis.axvline(80, color="#4C566A", linestyle=":", linewidth=1.25, zorder=1)

    ax = axes[0, 0]
    fixed_versions = np.arange(0, 251)
    ax.plot(
        fixed_versions,
        np.full_like(fixed_versions, 0.1, dtype=float),
        label="Fixed tau=0.1",
        color=COLORS["Fixed tau=0.1"],
        linewidth=2.2,
    )
    anneal = profiles["Linear anneal"]
    schedule = (
        anneal.loc[anneal["model_version"].between(0, 159)]
        .groupby("model_version")["effective_threshold"]
        .mean()
    )
    ax.plot(
        schedule.index,
        schedule.values,
        label="Linear anneal",
        color=COLORS["Linear anneal"],
        linewidth=2.2,
    )
    ax.plot(
        [80, 250],
        [1.0, 1.0],
        label="Immediate Vanilla: explicit full from v80",
        color=COLORS["Immediate Vanilla"],
        linewidth=2.0,
        linestyle=":",
    )
    ax.plot(
        [160, 250],
        [1.0, 1.0],
        label="Linear anneal: explicit full from v160",
        color=COLORS["Linear anneal"],
        linewidth=2.2,
        linestyle="--",
    )
    ax.axvspan(160, 250, color=COLORS["Linear anneal"], alpha=0.055)
    mark_step80(ax)
    ax.axvline(160, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(40, 0.93, "shared original tau=0.1 prefix", ha="center", fontsize=8.5)
    ax.text(205, 0.965, "linear anneal: full strategy", ha="center", va="top", fontsize=8.5)
    for version, threshold in (
        (80, 0.1),
        (90, 0.2125),
        (100, 0.325),
        (120, 0.55),
        (140, 0.775),
        (159, 0.98875),
    ):
        ax.scatter([version], [threshold], color=COLORS["Linear anneal"], s=24, zorder=3)
        ax.annotate(
            f"{threshold:.3g}",
            (version, threshold),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlim(-2, 253)
    ax.set_ylim(0.06, 1.04)
    ax.set_xlabel("Student model version")
    ax.set_ylabel("Adaptive effective threshold (full shown at top)")
    ax.set_title("A. Adaptive thresholds and TCOD F2B horizon", loc="left", weight="bold")
    style_axis(ax)

    tcod_ax = ax.twinx()
    tcod = profiles["TCOD F2B"]
    tcod_schedule = tcod.groupby("model_version")["imposed_horizon"].mean()
    tcod_ax.step(
        tcod_schedule.index,
        tcod_schedule.values,
        where="mid",
        label="TCOD F2B: imposed K",
        color=COLORS["TCOD F2B"],
        linewidth=2.2,
        linestyle="--",
    )
    tcod_full_version = int(tcod.loc[tcod["imposed_horizon"].eq(30), "model_version"].min())
    tcod_ax.set_ylim(0, 33)
    tcod_ax.set_yticks([0, 10, 20, 30])
    tcod_ax.set_ylabel("TCOD imposed horizon K", color=COLORS["TCOD F2B"])
    tcod_ax.tick_params(axis="y", colors=COLORS["TCOD F2B"])
    tcod_ax.spines["top"].set_visible(False)
    tcod_ax.annotate(
        f"TCOD reaches K=30 at v{tcod_full_version}",
        xy=(tcod_full_version, 30),
        xytext=(8, -22),
        textcoords="offset points",
        color=COLORS["TCOD F2B"],
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": COLORS["TCOD F2B"], "lw": 1.0},
    )
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = tcod_ax.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False, loc="center right")

    ax = axes[0, 1]
    for method in methods:
        part = bins.loc[bins["method"].eq(method)]
        ax.plot(
            part["version_bin_start"] + 2.5,
            100 * part["frontier_trigger_rate"],
            marker=markers[method],
            markersize=4,
            linewidth=2,
            linestyle=line_styles[method],
            label=method,
            color=COLORS[method],
        )
    mark_step80(ax)
    ax.set_xlim(-2, 253)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Student model version (5-version bins)")
    ax.set_ylabel("Curriculum-limited trajectories (%)")
    ax.set_title("B. Fraction still limited by the curriculum", loc="left", weight="bold")
    ax.text(
        3,
        99,
        "Adaptive: frontier triggered; TCOD: K < 30",
        fontsize=8.5,
        color="#4C566A",
        va="top",
    )
    ax.legend(frameon=False)
    style_axis(ax)

    ax = axes[1, 0]
    for method in methods:
        part = bins.loc[bins["method"].eq(method)]
        ax.plot(
            part["version_bin_start"] + 2.5,
            part["realized_turns_mean"],
            marker=markers[method],
            markersize=3.5,
            linewidth=1.9,
            linestyle=line_styles[method],
            label=method,
            color=COLORS[method],
        )
    mark_step80(ax)
    realized_max = float(bins["realized_turns_mean"].max())
    ax.set_xlim(-2, 253)
    ax.set_ylim(0, max(15.5, realized_max * 1.08))
    ax.set_xlabel("Student model version (5-version bins)")
    ax.set_ylabel("Mean realized trainable turns")
    ax.set_title(
        "C. Realized trainable turns across the full training course",
        loc="left",
        weight="bold",
    )
    ax.legend(frameon=False)
    style_axis(ax)

    ax = axes[1, 1]
    for method in methods:
        part = bins.loc[bins["method"].eq(method)]
        ax.plot(
            part["version_bin_start"] + 2.5,
            part["masked_turns_mean"],
            marker=markers[method],
            markersize=4,
            linewidth=2,
            linestyle=line_styles[method],
            label=method,
            color=COLORS[method],
        )
    mark_step80(ax)
    masked_max = float(bins["masked_turns_mean"].max())
    ax.set_xlim(-2, 253)
    ax.set_ylim(-0.2, max(4.6, masked_max * 1.08))
    ax.set_xlabel("Student model version (5-version bins)")
    ax.set_ylabel("Mean available turns removed by masking")
    ax.set_title(
        "D. Turns removed by masking across the full training course",
        loc="left",
        weight="bold",
    )
    ax.text(
        0.02,
        0.94,
        "TCOD before K=30: excluded-turn count is not retained in the buffer",
        transform=ax.transAxes,
        color=COLORS["TCOD F2B"],
        fontsize=8.5,
        va="top",
    )
    ax.legend(frameon=False)
    style_axis(ax)

    fig.suptitle(
        "Full-course diagnostic (v0–250): Adaptive continuations vs TCOD F2B",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

def subset_summary(profile: pd.DataFrame, start: int, end: int) -> dict:
    part = profile.loc[profile["model_version"].between(start, end)]
    return {
        "trajectory_count": int(len(part)),
        "frontier_trigger_rate": float(part["frontier_triggered"].mean()),
        "full_horizon_fraction": float(part["imposed_horizon"].eq(30).mean()),
        "realized_turns_mean": float(part["realized_trainable_turns"].mean()),
        "available_turns_mean": float(part["available_turns"].mean()),
        "masked_turns_mean": float(part["masked_turns"].mean()),
        "task_success_rate": float(part["task_success"].mean()),
        "prompt_truncated_rate": float(part["prompt_truncated"].mean()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_curriculum_module()
    paths = {
        "Fixed tau=0.1": args.fixed_diagnostics,
        "Linear anneal": args.anneal_diagnostics,
        "Immediate Vanilla": args.immediate_diagnostics,
    }
    fixed = load_profile(module, paths["Fixed tau=0.1"], "Fixed tau=0.1")
    anneal_suffix = load_profile(module, paths["Linear anneal"], "Linear anneal")
    immediate_suffix = load_profile(
        module, paths["Immediate Vanilla"], "Immediate Vanilla"
    )
    profiles = {
        "Fixed tau=0.1": fixed,
        "Linear anneal": splice_step80_prefix(fixed, anneal_suffix, "Linear anneal"),
        "Immediate Vanilla": splice_step80_prefix(
            fixed, immediate_suffix, "Immediate Vanilla"
        ),
        "TCOD F2B": load_tcod_profile(args.tcod_buffer, args.tcod_log),
    }
    bins = pd.concat(
        [aggregate_version_bins(profile) for profile in profiles.values()],
        ignore_index=True,
    )
    chunks = pd.concat(
        [aggregate_chunks(profile) for profile in profiles.values()],
        ignore_index=True,
    )
    bins.to_csv(args.output_dir / "version_bin_metrics.csv", index=False)
    chunks.to_csv(args.output_dir / "trajectory_chunk_metrics.csv", index=False)
    plot_detail(
        profiles,
        bins,
        chunks,
        args.output_dir / "step80_anneal_vs_fixed_detail.png",
    )

    anneal_post = profiles["Linear anneal"].loc[
        profiles["Linear anneal"]["model_version"] >= 80
    ]
    anneal_triggered = anneal_post.loc[anneal_post["frontier_triggered"]]
    first_full = anneal_post.loc[anneal_post["effective_strategy"].eq("full")]
    fixed_90_99 = subset_summary(profiles["Fixed tau=0.1"], 90, 99)
    anneal_90_99 = subset_summary(profiles["Linear anneal"], 90, 99)
    realized_delta = (
        anneal_90_99["realized_turns_mean"] - fixed_90_99["realized_turns_mean"]
    )
    available_delta = (
        anneal_90_99["available_turns_mean"] - fixed_90_99["available_turns_mean"]
    )
    masking_delta = fixed_90_99["masked_turns_mean"] - anneal_90_99["masked_turns_mean"]
    summary = {
        "schedule": {
            "formula": "tau(v)=0.1+0.9*(v-80)/80 for 80<=v<160; full for v>=160",
            "points": {
                "80": 0.1,
                "90": 0.2125,
                "100": 0.325,
                "120": 0.55,
                "140": 0.775,
                "159": 0.98875,
                "160": None,
            },
            "first_full_model_version": int(first_full["model_version"].min()),
            "first_full_training_step": int(first_full["training_step"].min()),
            "trajectories_before_explicit_full": int(
                (anneal_post["effective_strategy"] != "full").sum()
            ),
        },
        "full_course": {
            label: subset_summary(profile, 0, 250)
            for label, profile in profiles.items()
        },
        "post_step80": {
            label: subset_summary(profile, 80, 250)
            for label, profile in profiles.items()
        },
        "metric_semantics": {
            "curriculum_limited": "Adaptive: frontier_triggered; TCOD: imposed K<30",
            "tcod_model_version": "resume-log batch mapping matched against pipeline_input.model_version",
            "tcod_realized_turns": "distinct eid.step with non-empty action_mask containing True",
            "tcod_available_turns": (
                "observable only after K=30; pre-K30 buffer retains trainable turns "
                "but not the excluded counterfactual suffix"
            ),
        },
        "anneal_frontier": {
            "triggered_count": int(len(anneal_triggered)),
            "trajectory_count": int(len(anneal_post)),
            "triggered_fraction": float(len(anneal_triggered) / len(anneal_post)),
            "last_trigger_model_version": int(anneal_triggered["model_version"].max()),
            "last_trigger_effective_threshold": float(
                anneal_triggered.loc[
                    anneal_triggered["model_version"].idxmax(), "effective_threshold"
                ]
            ),
        },
        "window_90_99_jump_decomposition": {
            "fixed": fixed_90_99,
            "anneal": anneal_90_99,
            "realized_turn_delta_anneal_minus_fixed": float(realized_delta),
            "available_turn_delta_task_and_rollout_component": float(available_delta),
            "reduced_masking_component": float(masking_delta),
            "decomposition_residual": float(realized_delta - available_delta - masking_delta),
        },
        "interpretation": {
            "parameter_assessment": (
                "The raw-threshold ramp is functionally too aggressive: masking is "
                "nearly absent by model version 100, long before explicit full at 160."
            ),
            "boundary_assessment": (
                "The first 80-89 versions do not show the largest realized-turn jump. "
                "The visible spike is centered around versions 90-99 and is partly "
                "task/rollout-length variation, not annealing alone."
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            label: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for label, path in {**paths, "TCOD F2B buffer": args.tcod_buffer, "TCOD F2B resume log": args.tcod_log}.items()
        },
        "outputs": [
            "step80_anneal_vs_fixed_detail.png",
            "version_bin_metrics.csv",
            "trajectory_chunk_metrics.csv",
            "summary.json",
            "provenance.json",
        ],
    }
    write_json(args.output_dir / "provenance.json", provenance)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
