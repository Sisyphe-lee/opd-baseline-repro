#!/usr/bin/env python3
"""Plot the frozen TCOD/Vanilla ALFWorld reproduction results.

This script is deliberately independent of the historical entropy-diagnostics
pipeline.  It consumes the two matched full274 ``task_results.jsonl`` files
from the frozen baseline and, optionally, the final training launcher logs.
Every run writes provenance (input SHA256, protocol signatures, row counts)
beside the figures so the plotted population can be audited later.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


INK = "#1F2937"
GRID = "#E5E7EB"
VANILLA = "#7A8794"
TCOD = "#2F6B9A"
PAPER = "#D9833F"
SUCCESS = "#6B7D3A"
FAILURE = "#B85C5C"

PAPER_TABLE2 = {
    "Vanilla OPD": {
        "seen": {"success_rate": 65.72, "rounds": 14.73},
        "unseen": {"success_rate": 60.45, "rounds": 16.21},
    },
    "TCOD-F2B": {
        "seen": {"success_rate": 81.43, "rounds": 11.76},
        "unseen": {"success_rate": 79.19, "rounds": 12.47},
    },
}

TASK_LABELS = {
    "look_at_obj_in_light": "Look at object",
    "pick_and_place_simple": "Pick & place",
    "pick_clean_then_place_in_recep": "Clean & place",
    "pick_cool_then_place_in_recep": "Cool & place",
    "pick_heat_then_place_in_recep": "Heat & place",
    "pick_two_obj_and_place": "Pick two & place",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcod", type=Path, required=True)
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tcod-train-log", type=Path, action="append", default=[])
    parser.add_argument("--vanilla-train-log", type=Path, action="append", default=[])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_key(record: Mapping[str, Any]) -> str:
    path = str(record["game_file"]).replace("\\", "/")
    for marker in ("/valid_seen/", "/valid_unseen/"):
        if marker in path:
            return marker.strip("/") + "/" + path.split(marker, 1)[1]
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(value)
    return records


def protocol_signature(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    def unique(field: str) -> list[Any]:
        return sorted({record.get(field) for record in records}, key=str)

    sampling_fields = ("max_tokens", "temperature", "top_k", "top_p")
    return {
        "task_count": [len(records)],
        "splits": sorted({record.get("split") for record in records}),
        "max_env_steps": unique("max_env_steps"),
        "action_parser": unique("action_parser"),
        **{
            f"sampling.{field}": sorted(
                {record.get("sampling", {}).get(field) for record in records}, key=str
            )
            for field in sampling_fields
        },
    }


def validate_inputs(
    tcod: Sequence[Mapping[str, Any]], vanilla: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    keyed: list[dict[str, Mapping[str, Any]]] = []
    for label, records in (("TCOD", tcod), ("Vanilla", vanilla)):
        mapping = {task_key(record): record for record in records}
        if len(mapping) != len(records):
            raise ValueError(f"{label}: duplicate task identities")
        if len(records) != 274:
            raise ValueError(f"{label}: expected 274 records, found {len(records)}")
        counts = Counter(str(record.get("split")) for record in records)
        if counts != {"seen": 140, "unseen": 134}:
            raise ValueError(f"{label}: unexpected split counts: {dict(counts)}")
        keyed.append(mapping)
    if set(keyed[0]) != set(keyed[1]):
        raise ValueError("TCOD and Vanilla task identities do not match")
    left_signature = protocol_signature(tcod)
    right_signature = protocol_signature(vanilla)
    for field in (
        "max_env_steps",
        "action_parser",
        "sampling.max_tokens",
        "sampling.temperature",
        "sampling.top_k",
        "sampling.top_p",
    ):
        if left_signature[field] != right_signature[field]:
            raise ValueError(
                f"Protocol mismatch for {field}: "
                f"TCOD={left_signature[field]}, Vanilla={right_signature[field]}"
            )
    expected = {
        "max_env_steps": [30],
        "action_parser": ["strict_public_tcod"],
        "sampling.max_tokens": [512],
        "sampling.temperature": [0.4],
        "sampling.top_k": [-1],
        "sampling.top_p": [1.0],
    }
    for field, value in expected.items():
        if left_signature[field] != value:
            raise ValueError(
                f"Frozen-protocol violation for {field}: "
                f"expected {value}, found {left_signature[field]}"
            )
    return keyed[0], keyed[1]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def summarize(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("seen", "unseen", "all"):
        subset = [r for r in records if split == "all" or r["split"] == split]
        rows.append(
            {
                "split": split,
                "task_count": len(subset),
                "success_count": sum(bool(r["task_success"]) for r in subset),
                "success_rate": mean(bool(r["task_success"]) for r in subset),
                "average_rounds": mean(float(r["env_rounds"]) for r in subset),
                "timeout_rate": mean(bool(r.get("env_timeout")) for r in subset),
                "action_parse_valid_rate": mean(
                    float(r.get("action_parse_valid_rate", 0.0)) for r in subset
                ),
                "action_admissible_rate": mean(
                    float(r.get("action_admissible_rate", 0.0)) for r in subset
                ),
                "repeated_action_rate": mean(
                    float(r.get("repeated_action_rate", 0.0)) for r in subset
                ),
            }
        )
    return rows


def paired_rows(
    tcod: Mapping[str, Mapping[str, Any]],
    vanilla: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(tcod):
        left, right = tcod[key], vanilla[key]
        tcod_success = bool(left["task_success"])
        vanilla_success = bool(right["task_success"])
        if tcod_success and vanilla_success:
            outcome = "both_success"
        elif tcod_success:
            outcome = "tcod_only_success"
        elif vanilla_success:
            outcome = "vanilla_only_success"
        else:
            outcome = "both_failure"
        rows.append(
            {
                "task_key": key,
                "split": left["split"],
                "task_type": left["task_type"],
                "outcome": outcome,
                "tcod_success": int(tcod_success),
                "vanilla_success": int(vanilla_success),
                "tcod_rounds": int(left["env_rounds"]),
                "vanilla_rounds": int(right["env_rounds"]),
                "rounds_delta_tcod_minus_vanilla": int(left["env_rounds"])
                - int(right["env_rounds"]),
                "tcod_admissible_rate": float(left.get("action_admissible_rate", 0.0)),
                "vanilla_admissible_rate": float(
                    right.get("action_admissible_rate", 0.0)
                ),
            }
        )
    return rows


def task_type_rows(paired: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[(str(row["split"]), str(row["task_type"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split, task_type), rows in sorted(groups.items()):
        tcod_rate = mean(float(row["tcod_success"]) for row in rows)
        vanilla_rate = mean(float(row["vanilla_success"]) for row in rows)
        output.append(
            {
                "split": split,
                "task_type": task_type,
                "task_count": len(rows),
                "tcod_success_count": sum(int(row["tcod_success"]) for row in rows),
                "tcod_success_rate": tcod_rate,
                "vanilla_success_count": sum(
                    int(row["vanilla_success"]) for row in rows
                ),
                "vanilla_success_rate": vanilla_rate,
                "delta_tcod_minus_vanilla": tcod_rate - vanilla_rate,
                "tcod_average_rounds": mean(float(row["tcod_rounds"]) for row in rows),
                "vanilla_average_rounds": mean(
                    float(row["vanilla_rounds"]) for row in rows
                ),
            }
        )
    return output


def paired_count_rows(paired: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order = (
        "both_success",
        "tcod_only_success",
        "vanilla_only_success",
        "both_failure",
    )
    output = []
    for split in ("seen", "unseen", "all"):
        subset = [row for row in paired if split == "all" or row["split"] == split]
        counts = Counter(str(row["outcome"]) for row in subset)
        output.append(
            {
                "split": split,
                "task_count": len(subset),
                **{name: counts[name] for name in order},
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK)


def add_bar_labels(axis: plt.Axes, bars: Sequence[Any], suffix: str = "") -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )


def plot_paper_local_comparison(
    summaries: Mapping[str, Sequence[Mapping[str, Any]]], output_dir: Path
) -> Path:
    labels = ["Vanilla\nSeen", "Vanilla\nUnseen", "TCOD-F2B\nSeen", "TCOD-F2B\nUnseen"]
    specs = [
        ("Vanilla OPD", "Vanilla", "seen"),
        ("Vanilla OPD", "Vanilla", "unseen"),
        ("TCOD-F2B", "TCOD", "seen"),
        ("TCOD-F2B", "TCOD", "unseen"),
    ]
    local_by_method = {
        method: {str(row["split"]): row for row in rows}
        for method, rows in summaries.items()
    }
    paper_sr = [PAPER_TABLE2[p][s]["success_rate"] for p, _, s in specs]
    local_sr = [100 * local_by_method[l][s]["success_rate"] for _, l, s in specs]
    paper_rounds = [PAPER_TABLE2[p][s]["rounds"] for p, _, s in specs]
    local_rounds = [local_by_method[l][s]["average_rounds"] for _, l, s in specs]

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for axis in axes:
        style_axis(axis)
    bars_a = axes[0].bar(x - width / 2, paper_sr, width, label="Paper Table 2", color=PAPER)
    bars_b = axes[0].bar(x + width / 2, local_sr, width, label="Local reproduction", color=TCOD)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)
    add_bar_labels(axes[0], bars_a)
    add_bar_labels(axes[0], bars_b)

    bars_c = axes[1].bar(x - width / 2, paper_rounds, width, label="Paper Table 2", color=PAPER)
    bars_d = axes[1].bar(x + width / 2, local_rounds, width, label="Local reproduction", color=TCOD)
    axes[1].set_ylabel("Average environment rounds")
    axes[1].set_ylim(0, max(paper_rounds + local_rounds) * 1.25)
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False)
    add_bar_labels(axes[1], bars_c)
    add_bar_labels(axes[1], bars_d)

    fig.suptitle("Paper vs. frozen local reproduction", fontsize=16, color=INK)
    fig.text(
        0.5,
        0.93,
        "Qwen2.5-3B; local full274 uses h=30, accumulated memory, strict action parser, response=512",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = output_dir / "paper_vs_local_reproduction.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def annotate_heatmap(axis: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                axis.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8)


def plot_task_type_heatmap(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    ordered = sorted(rows, key=lambda row: (0 if row["split"] == "seen" else 1, row["task_type"]))
    labels = [
        f"{str(row['split']).title()} · {TASK_LABELS.get(str(row['task_type']), str(row['task_type']))} (n={row['task_count']})"
        for row in ordered
    ]
    vanilla = np.asarray([[100 * float(row["vanilla_success_rate"])] for row in ordered])
    tcod = np.asarray([[100 * float(row["tcod_success_rate"])] for row in ordered])
    delta = tcod - vanilla
    fig, axes = plt.subplots(1, 3, figsize=(14.5, max(6.8, len(labels) * 0.48)), sharey=True)
    panels = (
        (vanilla, "Vanilla success (%)", "Blues", 0, 100, ".1f"),
        (tcod, "TCOD-F2B success (%)", "Blues", 0, 100, ".1f"),
        (delta, "TCOD − Vanilla (pp)", "RdBu", -25, 25, "+.1f"),
    )
    for axis, (matrix, title, cmap, vmin, vmax, fmt) in zip(axes, panels):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        annotate_heatmap(axis, matrix, fmt)
        axis.set_title(title, fontsize=11, color=INK)
        axis.set_xticks([])
        fig.colorbar(image, ax=axis, fraction=0.08, pad=0.04)
    axes[0].set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    fig.suptitle("Success rate by ALFWorld task type", fontsize=16, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "task_type_success_heatmap.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def plot_paired_outcomes(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    names = (
        "both_success",
        "tcod_only_success",
        "vanilla_only_success",
        "both_failure",
    )
    labels = ("Both success", "TCOD only", "Vanilla only", "Both fail")
    matrix = np.asarray([[float(row[name]) for name in names] for row in rows])
    denominators = np.asarray([float(row["task_count"]) for row in rows])[:, None]
    percentages = 100 * matrix / denominators
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    cmap = LinearSegmentedColormap.from_list("paired", ["#F8FAFC", "#2F6B9A"])
    image = axis.imshow(percentages, aspect="auto", cmap=cmap, vmin=0, vmax=80)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                f"{int(matrix[i, j])}\n({percentages[i, j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if percentages[i, j] > 42 else INK,
            )
    axis.set_xticks(np.arange(len(labels)), labels=labels)
    axis.set_yticks(np.arange(len(rows)), labels=[str(row["split"]).title() for row in rows])
    axis.set_title("Matched-task outcome comparison", fontsize=15, color=INK, pad=15)
    axis.set_xlabel("Outcome on the same ALFWorld task")
    fig.colorbar(image, ax=axis, label="Share of split (%)", pad=0.02)
    fig.tight_layout()
    path = output_dir / "paired_outcome_heatmap.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def plot_task_outcome_matrix(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    outcome_order = {
        "tcod_only_success": 0,
        "vanilla_only_success": 1,
        "both_failure": 2,
        "both_success": 3,
    }
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row["split"] == "seen" else 1,
            str(row["task_type"]),
            outcome_order[str(row["outcome"])],
            str(row["task_key"]),
        ),
    )
    matrix = np.asarray(
        [[int(row["vanilla_success"]), int(row["tcod_success"])] for row in ordered]
    )
    fig, axis = plt.subplots(figsize=(6.2, 12.5))
    cmap = LinearSegmentedColormap.from_list("binary_outcome", [FAILURE, SUCCESS])
    axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    axis.set_xticks([0, 1], labels=["Vanilla", "TCOD-F2B"])
    axis.set_yticks([])
    axis.set_ylabel("274 matched tasks, sorted by split, type, and paired outcome")
    last_group: tuple[str, str] | None = None
    for index, row in enumerate(ordered):
        group = (str(row["split"]), str(row["task_type"]))
        if last_group is not None and group != last_group:
            axis.axhline(index - 0.5, color="white", linewidth=1.2)
        last_group = group
    seen_count = sum(row["split"] == "seen" for row in ordered)
    axis.axhline(seen_count - 0.5, color=INK, linewidth=2.0)
    axis.text(1.55, seen_count / 2, "Seen", rotation=90, va="center", color=INK)
    axis.text(1.55, seen_count + (len(ordered) - seen_count) / 2, "Unseen", rotation=90, va="center", color=INK)
    axis.set_title("Per-task success/failure matrix", fontsize=15, color=INK, pad=15)
    fig.tight_layout()
    path = output_dir / "per_task_outcome_matrix.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def plot_rounds_distribution(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for axis, split in zip(axes, ("seen", "unseen")):
        subset = [row for row in rows if row["split"] == split]
        values = [
            [float(row["vanilla_rounds"]) for row in subset],
            [float(row["tcod_rounds"]) for row in subset],
        ]
        artists = axis.boxplot(values, tick_labels=["Vanilla", "TCOD-F2B"], patch_artist=True, showfliers=False)
        for patch, color in zip(artists["boxes"], (VANILLA, TCOD)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for row in subset:
            axis.plot(
                [1, 2],
                [row["vanilla_rounds"], row["tcod_rounds"]],
                color="#94A3B8",
                linewidth=0.35,
                alpha=0.22,
            )
        style_axis(axis)
        axis.set_title(split.title())
        axis.set_ylabel("Environment rounds" if split == "seen" else "")
        axis.set_ylim(0, 31)
    fig.suptitle("Matched-task round distribution", fontsize=16, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "rounds_distribution.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_DICT_RE = re.compile(r"\bStep\s+(\d+):\s+(\{.*\})")


def parse_training_logs(paths: Sequence[Path]) -> list[dict[str, float]]:
    by_version: dict[int, dict[str, float]] = {}
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = ANSI_RE.sub("", raw_line)
                match = STEP_DICT_RE.search(line)
                if not match or "'rollout/model_version'" not in line:
                    continue
                try:
                    metrics = ast.literal_eval(match.group(2))
                except (SyntaxError, ValueError):
                    continue
                version = int(metrics["rollout/model_version"])
                record = {
                    "model_version": float(version),
                    "completion_rate": float(metrics.get("rollout/env_done/mean", math.nan)),
                    "rounds": float(metrics.get("rollout/env_rounds/mean", math.nan)),
                    "kl": float(metrics.get("rollout/kl_divergence/mean", math.nan)),
                    "task_count": float(metrics.get("rollout/finished_task_count", math.nan)),
                }
                by_version[version] = record
    return [by_version[key] for key in sorted(by_version)]


def rolling(values: Sequence[float], window: int = 10) -> np.ndarray:
    output = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = [value for value in values[start : index + 1] if math.isfinite(value)]
        output.append(mean(chunk))
    return np.asarray(output)


def plot_training_rollouts(
    tcod: Sequence[Mapping[str, float]],
    vanilla: Sequence[Mapping[str, float]],
    output_dir: Path,
) -> Path | None:
    if not tcod or not vanilla:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), sharex=True)
    fields = (
        ("completion_rate", "Rollout completion rate", 100.0),
        ("rounds", "Mean environment rounds", 1.0),
        ("kl", "Logged rollout KL divergence", 1.0),
    )
    for axis, (field, ylabel, scale) in zip(axes, fields):
        for label, records, color in (
            ("Vanilla", vanilla, VANILLA),
            ("TCOD-F2B", tcod, TCOD),
        ):
            x = np.asarray([row["model_version"] for row in records])
            y = np.asarray([row[field] * scale for row in records])
            axis.plot(x, y, color=color, alpha=0.18, linewidth=0.8)
            axis.plot(x, rolling(y), color=color, linewidth=2.0, label=label)
        style_axis(axis)
        axis.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Student model version (10-batch trailing mean)")
    fig.suptitle("Training-rollout diagnostics from the final reproduction logs", fontsize=16, color=INK)
    fig.text(
        0.5,
        0.94,
        "Training batches are stochastic and are not the frozen full274 evaluation; TCOD log coverage begins at resumed model version 80.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / "training_rollout_comparison.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tcod_records = load_jsonl(args.tcod)
    vanilla_records = load_jsonl(args.vanilla)
    tcod_keyed, vanilla_keyed = validate_inputs(tcod_records, vanilla_records)

    summaries = {
        "TCOD": summarize(tcod_records),
        "Vanilla": summarize(vanilla_records),
    }
    paired = paired_rows(tcod_keyed, vanilla_keyed)
    by_type = task_type_rows(paired)
    paired_counts = paired_count_rows(paired)
    summary_rows = [dict(method=method, **row) for method, rows in summaries.items() for row in rows]
    write_csv(args.output_dir / "method_summary.csv", summary_rows)
    write_csv(args.output_dir / "task_type_summary.csv", by_type)
    write_csv(args.output_dir / "paired_outcome_summary.csv", paired_counts)
    write_csv(args.output_dir / "paired_task_results.csv", paired)

    figure_paths: list[Path] = [
        plot_paper_local_comparison(summaries, args.output_dir),
        plot_task_type_heatmap(by_type, args.output_dir),
        plot_paired_outcomes(paired_counts, args.output_dir),
        plot_task_outcome_matrix(paired, args.output_dir),
        plot_rounds_distribution(paired, args.output_dir),
    ]
    training_tcod = parse_training_logs(args.tcod_train_log)
    training_vanilla = parse_training_logs(args.vanilla_train_log)
    training_path = plot_training_rollouts(training_tcod, training_vanilla, args.output_dir)
    if training_path is not None:
        figure_paths.append(training_path)
        write_csv(args.output_dir / "training_tcod_rollouts.csv", training_tcod)
        write_csv(args.output_dir / "training_vanilla_rollouts.csv", training_vanilla)

    provenance = {
        "tcod": {
            "path": str(args.tcod.resolve()),
            "sha256": sha256(args.tcod),
            "protocol": protocol_signature(tcod_records),
        },
        "vanilla": {
            "path": str(args.vanilla.resolve()),
            "sha256": sha256(args.vanilla),
            "protocol": protocol_signature(vanilla_records),
        },
        "paper_source": "arXiv:2604.24005, Table 2",
        "training_logs": {
            "tcod": [str(path.resolve()) for path in args.tcod_train_log],
            "vanilla": [str(path.resolve()) for path in args.vanilla_train_log],
            "tcod_model_version_range": (
                [int(training_tcod[0]["model_version"]), int(training_tcod[-1]["model_version"])]
                if training_tcod
                else None
            ),
            "vanilla_model_version_range": (
                [int(training_vanilla[0]["model_version"]), int(training_vanilla[-1]["model_version"])]
                if training_vanilla
                else None
            ),
        },
        "figures": [path.name for path in figure_paths],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    local = {
        method: {row["split"]: row for row in rows}
        for method, rows in summaries.items()
    }
    counts_all = next(row for row in paired_counts if row["split"] == "all")
    notes = "# 冻结基线复现图表\n\n"
    notes += "本目录由 `scripts/plot_baseline_reproduction.py` 直接读取两份冻结的 full274 逐题结果生成。\n\n"
    notes += "## 核心数值\n\n"
    notes += f"- Vanilla：Seen {100 * local['Vanilla']['seen']['success_rate']:.2f}%，Unseen {100 * local['Vanilla']['unseen']['success_rate']:.2f}%，Overall {100 * local['Vanilla']['all']['success_rate']:.2f}%。\n"
    notes += f"- TCOD-F2B：Seen {100 * local['TCOD']['seen']['success_rate']:.2f}%，Unseen {100 * local['TCOD']['unseen']['success_rate']:.2f}%，Overall {100 * local['TCOD']['all']['success_rate']:.2f}%。\n"
    notes += f"- 逐题配对：共同成功 {counts_all['both_success']}，仅 TCOD 成功 {counts_all['tcod_only_success']}，仅 Vanilla 成功 {counts_all['vanilla_only_success']}，共同失败 {counts_all['both_failure']}。\n\n"
    notes += "## 图表边界\n\n"
    notes += "- `task_type_success_heatmap.png`、`paired_outcome_heatmap.png` 和 `per_task_outcome_matrix.png` 均来自当前严格 512-token full274 评测。\n"
    notes += "- `training_rollout_comparison.png` 来自最终训练 launcher log；它是随机训练 batch 的在线指标，不是正式评测。TCOD 日志仅覆盖恢复后的 model version 80–248。\n"
    notes += "- 旧目录的 teacher-entropy frontier 热力图来自 Qwen3-1.7B、500-step 的另一项 Vanilla OPD 诊断实验，不能标为当前 Qwen2.5-3B TCOD baseline 的复现图。当前最终训练没有保存逐 token teacher/student entropy，因此无法仅靠现有 checkpoint 和日志原样重画该图。\n"
    notes += "\n旧图已作为非 baseline 参考独立保存在 [`../reference_legacy_qwen3_entropy_diagnostics/`](../reference_legacy_qwen3_entropy_diagnostics/)，相应绘图脚本、测试和埋点代码保存在 [`../../research_tools/legacy_entropy_diagnostics/`](../../research_tools/legacy_entropy_diagnostics/)。\n"
    (args.output_dir / "README.md").write_text(notes, encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
