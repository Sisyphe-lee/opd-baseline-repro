#!/usr/bin/env python3
"""Analyze the paired seed-42 full274 evaluation for adaptive-v1 tau=0.125.

This script is deliberately independent of the three-seed evaluation analysis.  It
compares the new run with the previous tau=0.175 run and the two frozen baselines,
after enforcing the frozen evaluation contract and exact per-task identity pairing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


METHODS = (
    "Adaptive v1 tau=0.125",
    "Adaptive v1 tau=0.175",
    "TCOD F2B (frozen)",
    "Vanilla OPD (frozen)",
)
COLORS = {
    METHODS[0]: "#2ca02c",
    METHODS[1]: "#9467bd",
    METHODS[2]: "#ff7f0e",
    METHODS[3]: "#4c78a8",
}
EXPECTED_SPLIT_COUNTS = {"seen": 140, "unseen": 134}


def canonical_identity(game_file: str) -> str:
    """Return a location-independent ALFWorld task identity."""
    normalized = game_file.replace("\\", "/")
    for split in ("valid_seen", "valid_unseen"):
        marker = f"/{split}/"
        if marker in normalized:
            return f"{split}/{normalized.split(marker, 1)[1]}"
    raise ValueError(f"game_file is not under valid_seen/valid_unseen: {game_file}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_sha256(identities: list[str]) -> str:
    payload = "\n".join(sorted(identities)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_equal(name: str, actual: Any, expected: Any, path: Path) -> None:
    if actual != expected:
        raise ValueError(f"{path}: expected {name}={expected!r}, got {actual!r}")


def read_and_validate(path: Path, method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []
    evaluation_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error

            for field in (
                "game_file",
                "split",
                "task_type",
                "task_success",
                "env_rounds",
                "max_env_steps",
                "action_parser",
                "sampling",
            ):
                if field not in raw:
                    raise ValueError(f"{path}:{line_number}: missing {field!r}")

            split = str(raw["split"])
            require_equal("split", split, "seen" if "/valid_seen/" in raw["game_file"] else "unseen", path)
            require_equal("max_env_steps", int(raw["max_env_steps"]), 30, path)
            require_equal("action_parser", raw["action_parser"], "strict_public_tcod", path)
            require_equal("sampling.temperature", float(raw["sampling"]["temperature"]), 0.4, path)
            require_equal("sampling.top_p", float(raw["sampling"]["top_p"]), 1.0, path)
            require_equal("sampling.top_k", int(raw["sampling"]["top_k"]), -1, path)
            require_equal("sampling.max_tokens", int(raw["sampling"]["max_tokens"]), 512, path)

            evaluation_ids.add(str(raw.get("evaluation_id", "")))
            records.append(
                {
                    "identity": canonical_identity(raw["game_file"]),
                    "split": split,
                    "task_type": str(raw["task_type"]),
                    "success": bool(raw["task_success"]),
                    "env_rounds": int(raw["env_rounds"]),
                    "parse_valid_rate": float(raw.get("action_parse_valid_rate", float("nan"))),
                    "admissible_rate": float(raw.get("action_admissible_rate", float("nan"))),
                    "repeat_rate": float(raw.get("repeated_action_rate", float("nan"))),
                    "timeout": bool(raw.get("env_timeout", False)),
                }
            )

    frame = pd.DataFrame.from_records(records)
    require_equal("row count", len(frame), 274, path)
    duplicates = frame.loc[frame["identity"].duplicated(keep=False), "identity"].tolist()
    if duplicates:
        raise ValueError(f"{path}: duplicate task identities: {duplicates[:5]}")
    split_counts = frame["split"].value_counts().to_dict()
    require_equal("split counts", split_counts, EXPECTED_SPLIT_COUNTS, path)

    frame = frame.sort_values("identity").reset_index(drop=True)
    validation = {
        "method": method,
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "rows": int(len(frame)),
        "unique_identities": int(frame["identity"].nunique()),
        "identity_sha256": identity_sha256(frame["identity"].tolist()),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "evaluation_ids": sorted(evaluation_ids),
        "frozen_protocol_validated": True,
    }
    return frame, validation


def validate_pairing(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    reference_name = METHODS[0]
    reference = frames[reference_name].set_index("identity")
    reference_ids = set(reference.index)
    for method, frame in frames.items():
        indexed = frame.set_index("identity")
        actual_ids = set(indexed.index)
        if actual_ids != reference_ids:
            missing = sorted(reference_ids - actual_ids)
            extra = sorted(actual_ids - reference_ids)
            raise ValueError(f"{method}: identity mismatch; missing={missing[:5]}, extra={extra[:5]}")
        aligned = indexed.loc[reference.index]
        for column in ("split", "task_type"):
            mismatch = aligned[column] != reference[column]
            if mismatch.any():
                bad = aligned.index[mismatch].tolist()
                raise ValueError(f"{method}: paired {column} mismatch for {bad[:5]}")
    return {
        "reference_method": reference_name,
        "exact_identity_sets_equal": True,
        "paired_split_and_task_type_equal": True,
        "paired_tasks": int(len(reference)),
        "common_identity_sha256": identity_sha256(reference.index.tolist()),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return float(center - half_width), float(center + half_width)


def summarize_subset(method: str, subset: str, frame: pd.DataFrame) -> dict[str, Any]:
    successes = int(frame["success"].sum())
    total = int(len(frame))
    lower, upper = wilson_interval(successes, total)
    return {
        "method": method,
        "subset": subset,
        "n": total,
        "success_count": successes,
        "success_rate": successes / total,
        "wilson95_low": lower,
        "wilson95_high": upper,
        "mean_env_rounds": float(frame["env_rounds"].mean()),
        "parse_valid_rate": float(frame["parse_valid_rate"].mean()),
        "admissible_rate": float(frame["admissible_rate"].mean()),
        "repeat_rate": float(frame["repeat_rate"].mean()),
        "timeout_rate": float(frame["timeout"].mean()),
    }


def exact_mcnemar(
    left: pd.DataFrame,
    right: pd.DataFrame,
    comparison: str,
    stratum: str,
    stratum_value: str,
) -> dict[str, Any]:
    merged = left[["identity", "success"]].merge(
        right[["identity", "success"]], on="identity", suffixes=("_new", "_reference"), validate="one_to_one"
    )
    new_only = int((merged["success_new"] & ~merged["success_reference"]).sum())
    reference_only = int((~merged["success_new"] & merged["success_reference"]).sum())
    discordant = new_only + reference_only
    p_value = float(binomtest(new_only, discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "comparison": comparison,
        "stratum": stratum,
        "stratum_value": stratum_value,
        "n": int(len(merged)),
        "new_success": int(merged["success_new"].sum()),
        "reference_success": int(merged["success_reference"].sum()),
        "success_rate_delta": float(merged["success_new"].mean() - merged["success_reference"].mean()),
        "both_success": int((merged["success_new"] & merged["success_reference"]).sum()),
        "new_only": new_only,
        "reference_only": reference_only,
        "both_failure": int((~merged["success_new"] & ~merged["success_reference"]).sum()),
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def build_tables(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_rows: list[dict[str, Any]] = []
    task_type_rows: list[dict[str, Any]] = []
    for method, frame in frames.items():
        method_rows.append(summarize_subset(method, "overall", frame))
        for split in ("seen", "unseen"):
            method_rows.append(summarize_subset(method, split, frame.loc[frame["split"] == split]))
        for task_type, group in frame.groupby("task_type", sort=True):
            row = summarize_subset(method, str(task_type), group)
            row["task_type"] = row.pop("subset")
            task_type_rows.append(row)

    new = frames[METHODS[0]]
    paired_rows: list[dict[str, Any]] = []
    for reference_name in METHODS[1:]:
        reference = frames[reference_name]
        comparison = f"{METHODS[0]} vs {reference_name}"
        paired_rows.append(exact_mcnemar(new, reference, comparison, "overall", "all"))
        for split in ("seen", "unseen"):
            paired_rows.append(
                exact_mcnemar(
                    new.loc[new["split"] == split],
                    reference.loc[reference["split"] == split],
                    comparison,
                    "split",
                    split,
                )
            )
        for task_type in sorted(new["task_type"].unique()):
            paired_rows.append(
                exact_mcnemar(
                    new.loc[new["task_type"] == task_type],
                    reference.loc[reference["task_type"] == task_type],
                    comparison,
                    "task_type",
                    str(task_type),
                )
            )

    paired_tasks = new[["identity", "split", "task_type"]].copy()
    for method, frame in frames.items():
        safe_name = (
            method.lower()
            .replace(" ", "_")
            .replace("=", "")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
        )
        outcomes = frame.set_index("identity").loc[paired_tasks["identity"], "success"].astype(int).to_numpy()
        paired_tasks[f"success_{safe_name}"] = outcomes
    for reference_name in METHODS[1:]:
        ref_safe = (
            reference_name.lower()
            .replace(" ", "_")
            .replace("=", "")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
        )
        paired_tasks[f"delta_new_minus_{ref_safe}"] = (
            paired_tasks["success_adaptive_v1_tau0125"] - paired_tasks[f"success_{ref_safe}"]
        )

    return (
        pd.DataFrame(method_rows),
        pd.DataFrame(task_type_rows),
        pd.DataFrame(paired_rows),
        paired_tasks,
    )


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def plot_overall_seen_unseen(method_summary: pd.DataFrame, output: Path) -> None:
    subsets = ("overall", "seen", "unseen")
    x = np.arange(len(METHODS))
    width = 0.23
    fig, ax = plt.subplots(figsize=(12.2, 6.2), constrained_layout=True)
    for index, subset in enumerate(subsets):
        values = [
            float(method_summary.loc[(method_summary["method"] == method) & (method_summary["subset"] == subset), "success_rate"].iloc[0])
            for method in METHODS
        ]
        bars = ax.bar(x + (index - 1) * width, values, width, label=subset.title(), alpha=0.88)
        ax.bar_label(bars, labels=[f"{value * 100:.1f}%" for value in values], padding=3, fontsize=9)
    ax.set_xticks(x, ["Adaptive v1\nτ=0.125", "Adaptive v1\nτ=0.175", "TCOD F2B\n(frozen)", "Vanilla OPD\n(frozen)"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Success rate")
    ax.set_title("Seed-42 full274 evaluation: overall, seen, and unseen", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_task_type_differences(task_summary: pd.DataFrame, output: Path) -> None:
    task_types = sorted(task_summary["task_type"].unique())
    display = [task_type.replace("_", " ") for task_type in task_types]
    y = np.arange(len(task_types))
    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(16, 7.4), sharey=True, constrained_layout=True)

    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for method, offset in zip(METHODS, offsets):
        rates = [
            float(task_summary.loc[(task_summary["method"] == method) & (task_summary["task_type"] == task_type), "success_rate"].iloc[0])
            for task_type in task_types
        ]
        left_ax.scatter(rates, y + offset, s=58, label=method, color=COLORS[method], alpha=0.9)
    left_ax.set_xlim(-0.03, 1.03)
    left_ax.set_xlabel("Success rate")
    left_ax.set_yticks(y, display)
    left_ax.grid(axis="x", alpha=0.22)
    left_ax.set_title("Absolute rate", loc="left", weight="bold")

    new_rates = {
        task_type: float(task_summary.loc[(task_summary["method"] == METHODS[0]) & (task_summary["task_type"] == task_type), "success_rate"].iloc[0])
        for task_type in task_types
    }
    delta_offsets = np.linspace(-0.16, 0.16, len(METHODS) - 1)
    for reference, offset in zip(METHODS[1:], delta_offsets):
        deltas = [
            new_rates[task_type]
            - float(task_summary.loc[(task_summary["method"] == reference) & (task_summary["task_type"] == task_type), "success_rate"].iloc[0])
            for task_type in task_types
        ]
        right_ax.scatter(deltas, y + offset, s=58, label=f"τ=0.125 − {reference}", color=COLORS[reference], alpha=0.9)
    right_ax.axvline(0, color="black", lw=1)
    right_ax.set_xlabel("Paired success-rate difference")
    right_ax.grid(axis="x", alpha=0.22)
    right_ax.set_title("New model minus comparator", loc="left", weight="bold")
    for ax in (left_ax, right_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Seed-42 full274 task-type comparison", x=0.02, ha="left", fontsize=15, weight="bold")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-t0125",
        type=Path,
        default=repo_root / "runs/experiments/entropy_adaptive_v1_t0125_250step_8gpu_s2t4_r16/evaluation/step250_full274/task_results.jsonl",
    )
    parser.add_argument(
        "--old-t0175",
        type=Path,
        default=repo_root / "runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16/evaluation/step250_full274/task_results.jsonl",
    )
    parser.add_argument(
        "--tcod",
        type=Path,
        default=repo_root / "results/evaluations/2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory-strict/evaluation/full274_h30/task_results.jsonl",
    )
    parser.add_argument(
        "--vanilla",
        type=Path,
        default=repo_root / "results/evaluations/2026-08-09_vanilla-opd-qwen25-3b-step250-full274-h30-accmemory-strict/evaluation/full274_h30/task_results.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(zip(METHODS, (args.new_t0125, args.old_t0175, args.tcod, args.vanilla)))

    frames: dict[str, pd.DataFrame] = {}
    input_validations: list[dict[str, Any]] = []
    for method, path in paths.items():
        frames[method], validation = read_and_validate(path, method)
        input_validations.append(validation)
    pairing_validation = validate_pairing(frames)

    method_summary, task_type_summary, paired_comparisons, paired_tasks = build_tables(frames)
    save_csv(method_summary, args.output_dir / "evaluation_method_summary.csv")
    save_csv(task_type_summary, args.output_dir / "evaluation_task_type_summary.csv")
    save_csv(paired_comparisons, args.output_dir / "evaluation_paired_comparisons.csv")
    save_csv(paired_tasks, args.output_dir / "evaluation_paired_task_results.csv")

    plot_overall_seen_unseen(method_summary, args.output_dir / "evaluation_overall_seen_unseen.png")
    plot_task_type_differences(task_type_summary, args.output_dir / "evaluation_task_type_differences.png")

    overall_rows = method_summary.loc[method_summary["subset"] == "overall"].to_dict(orient="records")
    overall_paired = paired_comparisons.loc[paired_comparisons["stratum"] == "overall"].to_dict(orient="records")
    summary = {
        "schema_version": 1,
        "analysis": "seed42_full274_tau_0125_vs_tau_0175_and_frozen_baselines",
        "evaluation_contract": {
            "tasks": 274,
            "seen": 140,
            "unseen": 134,
            "horizon": 30,
            "temperature": 0.4,
            "top_p": 1.0,
            "top_k": -1,
            "response_tokens": 512,
            "action_parser": "strict_public_tcod",
        },
        "input_validation": input_validations,
        "pairing_validation": pairing_validation,
        "overall_results": overall_rows,
        "overall_paired_comparisons": overall_paired,
        "mcnemar_definition": "Exact two-sided binomial test on discordant paired outcomes under p=0.5.",
        "artifacts": {
            "method_summary_csv": "evaluation_method_summary.csv",
            "task_type_summary_csv": "evaluation_task_type_summary.csv",
            "paired_comparisons_csv": "evaluation_paired_comparisons.csv",
            "paired_task_results_csv": "evaluation_paired_task_results.csv",
            "overall_seen_unseen_figure": "evaluation_overall_seen_unseen.png",
            "task_type_differences_figure": "evaluation_task_type_differences.png",
        },
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pairing_validation": pairing_validation, "overall_results": overall_rows, "overall_paired_comparisons": overall_paired}, indent=2))


if __name__ == "__main__":
    main()
