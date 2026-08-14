#!/usr/bin/env python3
"""Compare adaptive-v1 curriculum profiles and plot the tau=0.125 run.

The first figure reconstructs curriculum-imposed horizons, deliberately ignoring
whether the environment terminated early.  The second figure compares realized
non-zero loss-mask turns for Vanilla, TCOD F2B, the Adaptive reference, and the
configured Adaptive target.  The third figure reuses the prompt-truncation-
corrected latest-policy heatmap for the target run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import pickle
import re
import sqlite3
import sys
import tempfile
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



HERE = Path(__file__).resolve().parent
REPO_ROOT = MODULE_REPO_ROOT
CLEAN_PATH = (
    REPO_ROOT
    / "analysis"
    / "entropy_adaptive_v1_step250"
    / "analyze_training_nontruncated.py"
)
SPEC = importlib.util.spec_from_file_location("adaptive_v1_clean_analysis", CLEAN_PATH)
CLEAN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CLEAN)
ORIGINAL = CLEAN.ORIGINAL

DEFAULT_T0175 = (
    REPO_ROOT
    / "runs"
    / "experiments"
    / "entropy_adaptive_v1_step10_8gpu_s2t4_r16"
    / "diagnostics"
    / "trajectory_metrics.jsonl"
)
DEFAULT_T0125 = (
    REPO_ROOT
    / "runs"
    / "experiments"
    / "entropy_adaptive_v1_t0125_250step_8gpu_s2t4_r16"
    / "diagnostics"
    / "trajectory_metrics.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "analysis" / "entropy_adaptive_v1_t0125_step250"
DEFAULT_VANILLA_BUFFER = (
    REPO_ROOT / "checkpoints/vanilla_opd_step250/buffer/explorer_output.db"
)
DEFAULT_TCOD_BUFFER = (
    REPO_ROOT / "checkpoints/tcod_f2b_step250/buffer/explorer_output.db"
)

PANEL_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

REFERENCE_REALIZED_BASELINES = {
    "Vanilla OPD": {"count": 2832, "mean": 12.432203389830509, "at_max_fraction": 0.0},
    "TCOD F2B": {"count": 3088, "mean": 9.178432642487047, "at_max_fraction": 0.0},
}

REFERENCE_T0175 = {
    "trajectory_count": 2736,
    "imposed_horizon_mean": 26.970394736842106,
    "imposed_horizon_at_30_fraction": 0.846125730994152,
    "realized_trainable_turns_mean": 11.517543859649123,
    "realized_trainable_turns_at_30_fraction": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-diagnostics", "--adaptive-0175-diagnostics", dest="reference_diagnostics", type=Path, default=DEFAULT_T0175)
    parser.add_argument("--target-diagnostics", "--adaptive-0125-diagnostics", dest="target_diagnostics", type=Path, default=DEFAULT_T0125)
    parser.add_argument("--target-prefix-diagnostics", type=Path, default=None)
    parser.add_argument("--target-prefix-before-model-version", type=int, default=None)
    parser.add_argument("--target-threshold", type=float, default=0.125)
    parser.add_argument("--target-label", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vanilla-buffer", type=Path, default=DEFAULT_VANILLA_BUFFER)
    parser.add_argument("--tcod-buffer", type=Path, default=DEFAULT_TCOD_BUFFER)
    parser.add_argument("--max-env-steps", type=int, default=30)
    parser.add_argument("--tcod-batches", type=int, default=193)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_adaptive(
    path: Path,
    *,
    expected_threshold: float,
    label: str,
    max_env_steps: int,
) -> dict:
    rows, metadata = ORIGINAL.load_rows(path)
    rows["_source_row_index"] = np.arange(len(rows), dtype=np.int64)
    threshold = CLEAN.unique_frontier_threshold(rows)
    if not np.isclose(threshold, expected_threshold, rtol=0.0, atol=1e-12):
        raise AssertionError(
            f"{label}: expected threshold {expected_threshold}, found {threshold}"
        )

    rows = rows.drop_duplicates(["trajectory_id", "turn"], keep="last").copy()
    metadata["selected_row_count_after_resume_overlap"] = int(len(rows))
    trajectories, _ = ORIGINAL.trajectory_table(rows)
    if trajectories.empty:
        raise RuntimeError(f"{label}: no trajectories after de-duplication")

    first_source_row = rows.groupby("trajectory_id", sort=False)[
        "_source_row_index"
    ].min()
    grouped = {
        trajectory_id: group.sort_values("turn")
        for trajectory_id, group in rows.groupby("trajectory_id", sort=False)
    }

    records = []
    for trajectory in trajectories.itertuples(index=False):
        group = grouped[trajectory.trajectory_id]
        retained = group["loss_retained"].fillna(False).astype(bool)
        real = ~group["truncate_status"].eq("prompt_truncated")
        frontier = trajectory.frontier_turn
        imposed_horizon = (
            int(frontier) if pd.notna(frontier) else int(max_env_steps)
        )
        if not 1 <= imposed_horizon <= max_env_steps:
            raise AssertionError(
                f"{label}: imposed horizon out of range for "
                f"{trajectory.trajectory_id}: {imposed_horizon}"
            )
        records.append(
            {
                "method": label,
                "threshold": threshold,
                "trajectory_id": trajectory.trajectory_id,
                "training_step": int(trajectory.training_step),
                "model_version": int(trajectory.model_version),
                "first_source_row_index": int(
                    first_source_row.loc[trajectory.trajectory_id]
                ),
                "task_success": bool(trajectory.task_success),
                "frontier_triggered": bool(trajectory.frontier_triggered),
                "frontier_turn": int(frontier) if pd.notna(frontier) else np.nan,
                "imposed_horizon": imposed_horizon,
                "realized_trainable_turns": int((retained & real).sum()),
                "prompt_truncated": bool((~real).any()),
            }
        )

    profile = pd.DataFrame.from_records(records).sort_values(
        ["training_step", "first_source_row_index", "trajectory_id"],
        kind="stable",
    )
    profile = profile.reset_index(drop=True)
    profile["chronological_rank"] = np.arange(1, len(profile) + 1)
    profile["training_progress_percent"] = (
        np.linspace(0.0, 100.0, len(profile)) if len(profile) > 1 else 0.0
    )

    versions = profile["model_version"].to_numpy(dtype=int)
    return {
        "label": label,
        "threshold": threshold,
        "path": path.resolve(),
        "rows": rows,
        "trajectories": trajectories,
        "profile": profile,
        "metadata": metadata,
        "model_version_min": int(versions.min()),
        "model_version_max": int(versions.max()),
    }


def combine_adaptive_histories(
    prefix: dict,
    suffix: dict,
    *,
    branch_model_version: int,
    label: str,
) -> dict:
    """Join the original pre-branch history with the resumed branch history."""
    if branch_model_version <= 0:
        raise ValueError("branch_model_version must be positive")
    if not np.isclose(prefix["threshold"], suffix["threshold"], rtol=0.0, atol=1e-12):
        raise AssertionError("prefix and suffix thresholds do not match")

    prefix_profile = prefix["profile"].loc[
        prefix["profile"]["model_version"] < branch_model_version
    ].copy()
    suffix_profile = suffix["profile"].loc[
        suffix["profile"]["model_version"] >= branch_model_version
    ].copy()
    if prefix_profile.empty or suffix_profile.empty:
        raise RuntimeError("combined history requires non-empty prefix and suffix")
    overlap = set(prefix_profile["trajectory_id"]).intersection(
        suffix_profile["trajectory_id"]
    )
    prefix_profile["trajectory_id"] = (
        "prefix:" + prefix_profile["trajectory_id"].astype(str)
    )
    suffix_profile["trajectory_id"] = (
        "suffix:" + suffix_profile["trajectory_id"].astype(str)
    )

    prefix_profile["_history_segment"] = 0
    suffix_profile["_history_segment"] = 1
    profile = pd.concat([prefix_profile, suffix_profile], ignore_index=True)
    profile = profile.sort_values(
        ["_history_segment", "training_step", "first_source_row_index", "trajectory_id"],
        kind="stable",
    ).reset_index(drop=True)
    profile = profile.drop(columns="_history_segment")
    profile["chronological_rank"] = np.arange(1, len(profile) + 1)
    profile["training_progress_percent"] = (
        np.linspace(0.0, 100.0, len(profile)) if len(profile) > 1 else 0.0
    )

    prefix_rows = prefix["rows"].loc[
        prefix["rows"]["student_model_version"] < branch_model_version
    ].copy()
    suffix_rows = suffix["rows"].loc[
        suffix["rows"]["student_model_version"] >= branch_model_version
    ].copy()
    prefix_rows["trajectory_id"] = "prefix:" + prefix_rows["trajectory_id"].astype(str)
    suffix_rows["trajectory_id"] = "suffix:" + suffix_rows["trajectory_id"].astype(str)
    rows = pd.concat([prefix_rows, suffix_rows], ignore_index=True, sort=False)
    if rows.duplicated(["trajectory_id", "turn"]).any():
        raise AssertionError("combined history contains duplicate trajectory turns")
    trajectories, _ = ORIGINAL.trajectory_table(rows)
    if len(trajectories) != len(profile):
        raise AssertionError(
            f"combined trajectory mismatch: table={len(trajectories)}, profile={len(profile)}"
        )

    schema_counts = {
        str(key): int(value)
        for key, value in rows["diagnostics_schema_version"].value_counts().items()
    }
    kind_counts = {
        str(key): int(value)
        for key, value in rows["diagnostics_kind"].value_counts().items()
    }
    versions = profile["model_version"].to_numpy(dtype=int)
    return {
        "label": label,
        "threshold": suffix["threshold"],
        "path": suffix["path"],
        "rows": rows,
        "trajectories": trajectories,
        "profile": profile,
        "metadata": {
            "raw_row_count": int(len(rows)),
            "selected_row_count_after_resume_overlap": int(len(rows)),
            "malformed_line_count": 0,
            "duplicate_turn_row_count": 0,
            "schema_counts": schema_counts,
            "kind_counts": kind_counts,
        },
        "model_version_min": int(versions.min()),
        "model_version_max": int(versions.max()),
        "history_combination": {
            "branch_model_version": branch_model_version,
            "prefix_rule": f"student_model_version < {branch_model_version}",
            "suffix_rule": f"student_model_version >= {branch_model_version}",
            "prefix_trajectory_count": int(len(prefix_profile)),
            "suffix_trajectory_count": int(len(suffix_profile)),
            "combined_trajectory_count": int(len(profile)),
            "source_trajectory_id_overlap_count": int(len(overlap)),
            "prefix_row_count": int(len(prefix_rows)),
            "suffix_row_count": int(len(suffix_rows)),
            "combined_row_count": int(len(rows)),
        },
    }


def numeric_task_key(task: str) -> tuple[int, str]:
    try:
        return int(task), task
    except ValueError:
        match = re.match(r"^(\d+)", task)
        return (int(match.group(1)) if match else 2**31 - 1), task


def load_buffer_realized(path: Path, *, label: str) -> dict:
    """Reconstruct per-trajectory non-zero action-mask turns from a frozen buffer."""
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    trajectories: dict[tuple[int, str, int], dict] = {}
    row_count = 0
    try:
        cursor = connection.execute(
            "SELECT id, experience_bytes FROM pipeline_input ORDER BY id"
        )
        for row_id, blob in cursor:
            row_count += 1
            experience = Experience.deserialize(blob)
            match = re.match(r"^(\d+)", str(experience.eid.batch))
            if match is None:
                raise ValueError(f"{label}: cannot parse batch from {experience.eid.batch!r}")
            batch = int(match.group(1))
            task = str(experience.eid.task)
            run = int(experience.eid.run)
            step = int(experience.eid.step)
            key = (batch, task, run)
            record = trajectories.setdefault(
                key, {"first_row_id": int(row_id), "valid_steps": {}}
            )
            action_mask = experience.action_mask
            valid = bool(
                action_mask is not None
                and action_mask.numel() > 0
                and action_mask.any().item()
            )
            record["valid_steps"][step] = record["valid_steps"].get(step, False) or valid
    finally:
        connection.close()

    ordered = sorted(
        trajectories.items(),
        key=lambda item: (
            item[0][0],
            *numeric_task_key(item[0][1]),
            item[0][2],
        ),
    )
    records = []
    for rank, ((batch, task, run), record) in enumerate(ordered, 1):
        records.append(
            {
                "method": label,
                "threshold": np.nan,
                "trajectory_id": f"buffer:{batch}:{task}:{run}",
                "training_step": batch,
                "model_version": np.nan,
                "first_source_row_index": record["first_row_id"],
                "task_success": np.nan,
                "frontier_triggered": np.nan,
                "frontier_turn": np.nan,
                "imposed_horizon": np.nan,
                "realized_trainable_turns": int(sum(record["valid_steps"].values())),
                "prompt_truncated": np.nan,
                "chronological_rank": rank,
            }
        )
    profile = pd.DataFrame.from_records(records)
    profile["training_progress_percent"] = (
        np.linspace(0.0, 100.0, len(profile)) if len(profile) > 1 else 0.0
    )
    return {
        "label": label,
        "path": path.resolve(),
        "row_count": row_count,
        "profile": profile,
    }


def _buffer_extract_worker(path: Path, label: str, output: Path) -> None:
    dataset = load_buffer_realized(path, label=label)
    with output.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_buffer_realized_isolated(path: Path, *, label: str) -> dict:
    """Decode one large buffer in a short-lived process to release Torch memory."""
    with tempfile.NamedTemporaryFile(
        prefix="adaptive-profile-", suffix=".pkl", dir="/tmp", delete=False
    ) as handle:
        payload_path = Path(handle.name)
    try:
        process = mp.get_context("spawn").Process(
            target=_buffer_extract_worker, args=(path, label, payload_path)
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"{label}: isolated buffer extraction exited with {process.exitcode}"
            )
        with payload_path.open("rb") as handle:
            return pickle.load(handle)
    finally:
        payload_path.unlink(missing_ok=True)


def metric_summary(values: pd.Series, max_env_steps: int) -> dict:
    numeric = values.to_numpy(dtype=float)
    return {
        "count": int(numeric.size),
        "mean": float(numeric.mean()),
        "median": float(np.median(numeric)),
        "min": int(numeric.min()),
        "max": int(numeric.max()),
        "at_max_count": int((numeric == max_env_steps).sum()),
        "at_max_fraction": float((numeric == max_env_steps).mean()),
    }


def assert_t0175_reference(profile: pd.DataFrame, max_env_steps: int) -> dict:
    imposed = metric_summary(profile["imposed_horizon"], max_env_steps)
    realized = metric_summary(profile["realized_trainable_turns"], max_env_steps)
    observed = {
        "trajectory_count": len(profile),
        "imposed_horizon_mean": imposed["mean"],
        "imposed_horizon_at_30_fraction": imposed["at_max_fraction"],
        "realized_trainable_turns_mean": realized["mean"],
        "realized_trainable_turns_at_30_fraction": realized["at_max_fraction"],
    }
    for key, expected in REFERENCE_T0175.items():
        actual = observed[key]
        if isinstance(expected, int):
            if actual != expected:
                raise AssertionError(f"tau=0.175 replay mismatch: {key}={actual}, expected {expected}")
        elif not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
            raise AssertionError(f"tau=0.175 replay mismatch: {key}={actual}, expected {expected}")
    return observed


def assert_realized_baseline_reference(
    dataset: dict, max_env_steps: int
) -> dict:
    summary = metric_summary(
        dataset["profile"]["realized_trainable_turns"], max_env_steps
    )
    expected = REFERENCE_REALIZED_BASELINES[dataset["label"]]
    for key, value in expected.items():
        actual = summary[key]
        if isinstance(value, int):
            if actual != value:
                raise AssertionError(
                    f"{dataset['label']} replay mismatch: {key}={actual}, expected {value}"
                )
        elif not np.isclose(actual, value, rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"{dataset['label']} replay mismatch: {key}={actual}, expected {value}"
            )
    return summary


def synthetic_imposed_profiles(
    adaptive_0175: pd.DataFrame,
    adaptive_0125: pd.DataFrame,
    *,
    max_env_steps: int,
    tcod_batches: int,
    batch_size: int,
) -> pd.DataFrame:
    vanilla_count = 2832
    vanilla = pd.DataFrame(
        {
            "method": "Vanilla OPD",
            "source_kind": "analytical_full_horizon",
            "unit_id": [f"vanilla:{index + 1}" for index in range(vanilla_count)],
            "chronological_rank": np.arange(1, vanilla_count + 1),
            "imposed_horizon": np.full(vanilla_count, max_env_steps, dtype=int),
        }
    )

    tcod_records = []
    rank = 0
    for batch in range(1, tcod_batches + 1):
        horizon = min(1 + batch // 2, max_env_steps)
        for task_index in range(batch_size):
            rank += 1
            tcod_records.append(
                {
                    "method": "TCOD F2B",
                    "source_kind": "analytical_f2b_schedule",
                    "unit_id": f"tcod:batch{batch}:task{task_index}",
                    "chronological_rank": rank,
                    "imposed_horizon": horizon,
                }
            )
    tcod = pd.DataFrame.from_records(tcod_records)

    adaptive_frames = []
    for profile in (adaptive_0175, adaptive_0125):
        part = profile[
            ["method", "trajectory_id", "chronological_rank", "imposed_horizon"]
        ].copy()
        part = part.rename(columns={"trajectory_id": "unit_id"})
        part["source_kind"] = "recorded_frontier_or_full_horizon"
        adaptive_frames.append(part)

    return pd.concat([vanilla, tcod, *adaptive_frames], ignore_index=True)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE9", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_imposed_horizons(
    profiles: pd.DataFrame,
    output: Path,
    max_env_steps: int,
) -> dict:
    methods = profiles["method"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4.2 * len(methods), 5.4),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    summaries = {}
    for panel_index, (ax, method) in enumerate(zip(axes, methods)):
        values = np.sort(
            profiles.loc[profiles["method"].eq(method), "imposed_horizon"].to_numpy(
                dtype=float
            )
        )
        y = np.linspace(0.0, 100.0, len(values))
        color = PANEL_COLORS[panel_index]
        ax.fill_betweenx(y, 1.0, values, step="post", color=color, alpha=0.92)
        ax.plot(values, y, drawstyle="steps-post", color="#222222", linewidth=1.0)
        summary = metric_summary(pd.Series(values), max_env_steps)
        summaries[method] = summary
        display = method.replace("tau=", "τ=")
        ax.set_title(
            f"{display}\n"
            f"n={summary['count']:,} · mean={summary['mean']:.2f} · "
            f"K={max_env_steps}: {100 * summary['at_max_fraction']:.1f}%",
            loc="left",
            fontsize=10.5,
            weight="bold",
        )
        ax.set_xlabel("Environment turn included in loss")
        ax.set_xlim(0.5, max_env_steps + 0.5)
        ax.set_ylim(100.0, 0.0)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        style_axis(ax)
    axes[0].set_ylabel("Task percentile (shortest curriculum at top)")
    fig.suptitle(
        "Curriculum-imposed loss horizons (environment termination excluded)",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summaries


def centered_median(values: np.ndarray) -> np.ndarray:
    window = max(31, int(round(len(values) * 0.02)))
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def plot_realized_turns(
    profiles: pd.DataFrame,
    output: Path,
    max_env_steps: int,
) -> dict:
    methods = profiles["method"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4.2 * len(methods), 5.8),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    summaries = {}
    for panel_index, (ax, method) in enumerate(zip(axes, methods)):
        part = profiles.loc[profiles["method"].eq(method)].sort_values(
            "chronological_rank", kind="stable"
        )
        values = part["realized_trainable_turns"].to_numpy(dtype=float)
        y = np.linspace(0.0, 100.0, len(values))
        color = PANEL_COLORS[panel_index]
        ax.hlines(y, 1.0, values, color=color, alpha=0.18, linewidth=0.55)
        ax.plot(centered_median(values), y, color="black", linewidth=1.7)
        summary = metric_summary(pd.Series(values), max_env_steps)
        summaries[method] = summary
        display = method.replace("tau=", "τ=")
        ax.set_title(
            f"{display}\n"
            f"n={summary['count']:,} · mean={summary['mean']:.2f} · "
            f"K={max_env_steps}: {100 * summary['at_max_fraction']:.1f}%",
            loc="left",
            fontsize=11,
            weight="bold",
        )
        ax.set_xlabel("Environment turn with non-zero loss mask")
        ax.set_xlim(0.5, max_env_steps + 0.5)
        ax.set_ylim(100.0, 0.0)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        style_axis(ax)
    axes[0].set_ylabel("Training progress percentile (earliest at top)")
    fig.suptitle(
        "Realized trainable turns in chronological exploration order",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Black line: centered rolling median (2% of trajectories).",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summaries


def diagnostics_provenance(dataset: dict) -> dict:
    path = dataset["path"]
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "threshold": dataset["threshold"],
        "raw_row_count": dataset["metadata"]["raw_row_count"],
        "selected_row_count_after_resume_overlap": dataset["metadata"][
            "selected_row_count_after_resume_overlap"
        ],
        "malformed_line_count": dataset["metadata"]["malformed_line_count"],
        "duplicate_turn_row_count": dataset["metadata"]["duplicate_turn_row_count"],
        "schema_counts": dataset["metadata"]["schema_counts"],
        "kind_counts": dataset["metadata"]["kind_counts"],
        "trajectory_count": int(len(dataset["profile"])),
        "model_version_min": dataset["model_version_min"],
        "model_version_max": dataset["model_version_max"],
    }


def buffer_provenance(dataset: dict) -> dict:
    path = dataset["path"]
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "table": "pipeline_input",
        "experience_row_count": dataset["row_count"],
        "trajectory_count": int(len(dataset["profile"])),
        "trajectory_key": "(leading integer of eid.batch, str(eid.task), int(eid.run))",
        "turn_rule": "count distinct eid.step with non-empty action_mask and action_mask.any()",
    }


def main() -> None:
    args = parse_args()
    if args.max_env_steps != 30:
        raise ValueError("This reference comparison is defined for max_env_steps=30")
    prefix_args_set = (
        args.target_prefix_diagnostics is not None,
        args.target_prefix_before_model_version is not None,
    )
    if prefix_args_set[0] != prefix_args_set[1]:
        raise ValueError(
            "--target-prefix-diagnostics and "
            "--target-prefix-before-model-version must be supplied together"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Decode the large frozen buffers before retaining the diagnostics DataFrames;
    # this keeps peak memory bounded during Experience deserialization.
    vanilla = load_buffer_realized_isolated(args.vanilla_buffer, label="Vanilla OPD")
    tcod = load_buffer_realized_isolated(args.tcod_buffer, label="TCOD F2B")
    baseline_realized_replay = {
        "Vanilla OPD": assert_realized_baseline_reference(vanilla, args.max_env_steps),
        "TCOD F2B": assert_realized_baseline_reference(tcod, args.max_env_steps),
    }
    adaptive_0175 = load_adaptive(
        args.reference_diagnostics,
        expected_threshold=0.175,
        label="Adaptive v1 (tau=0.175)",
        max_env_steps=args.max_env_steps,
    )
    reference_replay = assert_t0175_reference(
        adaptive_0175["profile"], args.max_env_steps
    )
    target_label = args.target_label or f"Adaptive v1 (tau={args.target_threshold:g})"
    reference_label = "Adaptive v1 (tau=0.175)"
    if target_label == reference_label:
        raise ValueError(
            "Target label collides with the fixed reference label; pass a unique "
            "--target-label (for example, 'Adaptive v1 target repeat')."
        )
    target_suffix = load_adaptive(
        args.target_diagnostics,
        expected_threshold=args.target_threshold,
        label=target_label,
        max_env_steps=args.max_env_steps,
    )
    target_prefix = None
    adaptive_0125 = target_suffix
    if args.target_prefix_diagnostics is not None:
        target_prefix = load_adaptive(
            args.target_prefix_diagnostics,
            expected_threshold=args.target_threshold,
            label=target_label,
            max_env_steps=args.max_env_steps,
        )
        adaptive_0125 = combine_adaptive_histories(
            target_prefix,
            target_suffix,
            branch_model_version=args.target_prefix_before_model_version,
            label=target_label,
        )

    imposed_profiles = synthetic_imposed_profiles(
        adaptive_0175["profile"],
        adaptive_0125["profile"],
        max_env_steps=args.max_env_steps,
        tcod_batches=args.tcod_batches,
        batch_size=args.batch_size,
    )
    realized_profiles = pd.concat(
        [
            vanilla["profile"],
            tcod["profile"],
            adaptive_0175["profile"],
            adaptive_0125["profile"],
        ],
        ignore_index=True,
    )

    imposed_profiles.to_csv(
        args.output_dir / "curriculum_imposed_loss_horizons.csv", index=False
    )
    realized_profiles.to_csv(
        args.output_dir / "realized_trainable_turns.csv", index=False
    )

    imposed_summary = plot_imposed_horizons(
        imposed_profiles,
        args.output_dir / "curriculum_imposed_loss_horizons.png",
        args.max_env_steps,
    )
    realized_summary = plot_realized_turns(
        realized_profiles,
        args.output_dir / "realized_trainable_turns_chronological.png",
        args.max_env_steps,
    )
    heatmap_summary = CLEAN.plot_heatmap(
        adaptive_0125["rows"],
        adaptive_0125["trajectories"],
        args.output_dir / "teacher_entropy_frontier_heatmap_latest.png",
    )

    plot_summary = {
        "max_env_steps": args.max_env_steps,
        "thresholds": {
            "adaptive_reference": adaptive_0175["threshold"],
            "adaptive_target": adaptive_0125["threshold"],
        },
        "adaptive_run_diagnostics": {
            "adaptive_reference": {
                "raw_row_count": adaptive_0175["metadata"]["raw_row_count"],
                "selected_row_count": len(adaptive_0175["rows"]),
                "trajectory_count": len(adaptive_0175["profile"]),
                "frontier_triggered_count": int(
                    adaptive_0175["profile"]["frontier_triggered"].sum()
                ),
                "prompt_truncated_trajectory_count": int(
                    adaptive_0175["profile"]["prompt_truncated"].sum()
                ),
            },
            "adaptive_target": {
                "raw_row_count": adaptive_0125["metadata"]["raw_row_count"],
                "selected_row_count": len(adaptive_0125["rows"]),
                "trajectory_count": len(adaptive_0125["profile"]),
                "frontier_triggered_count": int(
                    adaptive_0125["profile"]["frontier_triggered"].sum()
                ),
                "prompt_truncated_trajectory_count": int(
                    adaptive_0125["profile"]["prompt_truncated"].sum()
                ),
                "history_combination": adaptive_0125.get("history_combination"),
            },
        },
        "reference_t0175_replay_assertions": {
            "status": "passed",
            "observed": reference_replay,
            "expected": REFERENCE_T0175,
        },
        "reference_realized_baseline_replay_assertions": {
            "status": "passed",
            "observed": baseline_realized_replay,
            "expected": REFERENCE_REALIZED_BASELINES,
        },
        "curriculum_imposed_loss_horizons": imposed_summary,
        "realized_trainable_turns": realized_summary,
        "latest_policy_heatmap_target": heatmap_summary,
        "evidence_scope": {
            "imposed_horizon": (
                "Vanilla is analytically K=30; TCOD uses the recorded F2B schedule "
                "K_b=min(1+floor(b/2),30), b=1..193; the reference and target Adaptive panels use "
                "recorded entropy_frontier_turn or K=30 when no frontier triggered."
            ),
            "realized_trainable_turns": (
                "Vanilla/TCOD are reconstructed from frozen checkpoint Experience "
                "buffers by counting distinct environment steps with a non-empty, "
                "non-zero action_mask. Adaptive panels use loss_retained and exclude "
                "prompt_truncated placeholders."
            ),
        },
    }
    json_dump(args.output_dir / "plot_summary.json", plot_summary)

    if target_prefix is None:
        adaptive_target_provenance = diagnostics_provenance(adaptive_0125)
    else:
        adaptive_target_provenance = {
            "history_combination": adaptive_0125["history_combination"],
            "prefix": diagnostics_provenance(target_prefix),
            "suffix": diagnostics_provenance(target_suffix),
        }

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "adaptive_reference": diagnostics_provenance(adaptive_0175),
        "adaptive_target": adaptive_target_provenance,
        "vanilla_realized_buffer": buffer_provenance(vanilla),
        "tcod_realized_buffer": buffer_provenance(tcod),
        "analytical_reference_inputs": {
            "vanilla_trajectory_count": 2832,
            "vanilla_rule": "K=30 for every trajectory",
            "tcod_batch_count": args.tcod_batches,
            "tcod_batch_size": args.batch_size,
            "tcod_trajectory_count": args.tcod_batches * args.batch_size,
            "tcod_rule": "K_b=min(1+floor(b/2),30), one-based batch b",
            "tcod_workflow": str(
                REPO_ROOT
                / "trinity/common/workflows/envs/TCOD/alfworld/TCOD_f2b_workflow.py"
            ),
            "tcod_config": str(REPO_ROOT / "configs/train/tcod_f2b.yaml"),
        },
        "outputs": [
            "curriculum_imposed_loss_horizons.png",
            "realized_trainable_turns_chronological.png",
            "teacher_entropy_frontier_heatmap_latest.png",
            "curriculum_imposed_loss_horizons.csv",
            "realized_trainable_turns.csv",
            "plot_summary.json",
            "provenance.json",
        ],
    }
    json_dump(args.output_dir / "provenance.json", provenance)
    print(json.dumps(plot_summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
