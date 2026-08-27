#!/usr/bin/env python3
"""Build the canonical multi-row Adaptive v1 experiment comparison figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis/entropy_adaptive_v1_all_experiments"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.json"
MAX_ENV_STEPS = 30
PANEL_COLORS = [
    "#4C78A8",
    "#F58518",
    "#72B7B2",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#EDC948",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def metric_summary(values: pd.Series) -> dict:
    numeric = values.to_numpy(dtype=float)
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(np.median(numeric)),
        "min": int(numeric.min()),
        "max": int(numeric.max()),
        "at_max_count": int((numeric == MAX_ENV_STEPS).sum()),
        "at_max_fraction": float((numeric == MAX_ENV_STEPS).mean()),
    }


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


def display_label(label: str) -> str:
    label = label.replace("tau=", "τ=").replace("->", "→")
    if label.startswith("Adaptive v1 ("):
        return label.replace("Adaptive v1 (", "Adaptive v1\n(", 1)
    if label == "Step80 → immediate Vanilla OPD":
        return "Step80 → immediate\nVanilla OPD"
    if label == "Step80 → linear anneal → Vanilla OPD":
        return "Step80 → linear anneal\n→ Vanilla OPD"
    if label.startswith("Step80 → cosine anneal "):
        return label.replace("Step80 → cosine anneal ", "Step80 → cosine anneal\n", 1)
    return label


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE9", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if not isinstance(manifest.get("layout_rows"), int) or manifest["layout_rows"] < 1:
        raise ValueError("layout_rows must be a positive integer")
    panels = manifest.get("panels", [])
    if not panels:
        raise ValueError("Manifest contains no panels")
    labels = [panel["label"] for panel in panels]
    if len(labels) != len(set(labels)):
        raise ValueError("Manifest panel labels must be unique")
    return manifest


def load_panel(panel: dict, order: int) -> dict:
    source_dir = (REPO_ROOT / panel["source_dir"]).resolve()
    imposed_path = source_dir / "curriculum_imposed_loss_horizons.csv"
    realized_path = source_dir / "realized_trainable_turns.csv"
    imposed_all = pd.read_csv(imposed_path)
    realized_all = pd.read_csv(realized_path)
    source_method = panel["source_method"]
    imposed = imposed_all.loc[imposed_all["method"].eq(source_method)].copy()
    realized = realized_all.loc[realized_all["method"].eq(source_method)].copy()
    if imposed.empty or realized.empty:
        raise ValueError(f"Missing method {source_method!r} in {source_dir}")
    if len(imposed) != len(realized):
        raise ValueError(
            f"Trajectory count mismatch for {panel['label']}: "
            f"imposed={len(imposed)}, realized={len(realized)}"
        )

    for frame in (imposed, realized):
        frame["source_method"] = frame["method"]
        frame["method"] = panel["label"]
        frame["panel_order"] = order

    boundary = None
    if panel["kind"] == "step80_branch":
        trajectory_ids = realized["trajectory_id"].astype(str)
        prefix_count = int(trajectory_ids.str.startswith("prefix:").sum())
        suffix_count = int(trajectory_ids.str.startswith("suffix:").sum())
        if prefix_count == 0 or suffix_count == 0:
            raise ValueError(f"Branch panel {panel['label']} is not a spliced history")
        boundary = {
            "prefix_trajectory_count": prefix_count,
            "suffix_trajectory_count": suffix_count,
            "chronological_percent": float(
                100.0 * (prefix_count - 1) / max(1, len(realized) - 1)
            ),
            "rule": "original model_version < 80; branch model_version >= 80",
        }

    source_files = [imposed_path, realized_path]
    for name in ("plot_summary.json", "provenance.json"):
        candidate = source_dir / name
        if candidate.exists():
            source_files.append(candidate)
    return {
        "config": panel,
        "source_dir": source_dir,
        "source_files": source_files,
        "imposed": imposed,
        "realized": realized,
        "boundary": boundary,
    }


def make_grid(panel_count: int, rows: int) -> tuple[plt.Figure, np.ndarray]:
    columns = math.ceil(panel_count / rows)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.55 * columns, 5.15 * rows),
        sharey=True,
        constrained_layout=True,
    )
    return fig, np.asarray(axes).reshape(rows, columns)


def panel_title(panel: dict, summary: dict) -> str:
    label = display_label(panel["config"]["short_label"])
    return (
        f"{label}\n"
        f"n={summary['count']:,} · mean={summary['mean']:.2f} · "
        f"K={MAX_ENV_STEPS}: {100 * summary['at_max_fraction']:.1f}%"
    )


def plot_imposed(panels: list[dict], output: Path, rows: int) -> dict:
    fig, axes = make_grid(len(panels), rows)
    summaries = {}
    for index, (ax, panel) in enumerate(zip(axes.flat, panels)):
        values = np.sort(panel["imposed"]["imposed_horizon"].to_numpy(dtype=float))
        y = np.linspace(0.0, 100.0, len(values))
        color = PANEL_COLORS[index % len(PANEL_COLORS)]
        ax.fill_betweenx(y, 1.0, values, step="post", color=color, alpha=0.90)
        ax.plot(values, y, drawstyle="steps-post", color="#222222", linewidth=1.0)
        summary = metric_summary(pd.Series(values))
        summaries[panel["config"]["label"]] = summary
        ax.set_title(panel_title(panel, summary), loc="left", fontsize=10.2, weight="bold")
        ax.set_xlabel("Environment turn included in loss")
        ax.set_xlim(0.5, MAX_ENV_STEPS + 0.5)
        ax.set_ylim(100.0, 0.0)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        style_axis(ax)
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Task percentile (shortest curriculum at top)")
    fig.suptitle(
        "Curriculum-imposed loss horizons (all matched 4-GPU experiments)",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summaries


def plot_realized(panels: list[dict], output: Path, rows: int) -> dict:
    fig, axes = make_grid(len(panels), rows)
    summaries = {}
    for index, (ax, panel) in enumerate(zip(axes.flat, panels)):
        part = panel["realized"].sort_values("chronological_rank", kind="stable")
        values = part["realized_trainable_turns"].to_numpy(dtype=float)
        y = np.linspace(0.0, 100.0, len(values))
        color = PANEL_COLORS[index % len(PANEL_COLORS)]
        ax.hlines(y, 1.0, values, color=color, alpha=0.18, linewidth=0.55)
        ax.plot(centered_median(values), y, color="black", linewidth=1.7)
        boundary = panel["boundary"]
        if boundary is not None:
            boundary_y = boundary["chronological_percent"]
            ax.axhline(boundary_y, color="#5B2333", linestyle="--", linewidth=1.2)
            ax.text(
                MAX_ENV_STEPS - 0.3,
                boundary_y - 1.2,
                "step 80 splice",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="#5B2333",
            )
        summary = metric_summary(pd.Series(values))
        summaries[panel["config"]["label"]] = summary
        ax.set_title(panel_title(panel, summary), loc="left", fontsize=10.2, weight="bold")
        ax.set_xlabel("Environment turn with non-zero loss mask")
        ax.set_xlim(0.5, MAX_ENV_STEPS + 0.5)
        ax.set_ylim(100.0, 0.0)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        style_axis(ax)
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Training progress percentile (earliest at top)")
    fig.suptitle(
        "Realized trainable turns in chronological exploration order",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Black: centered rolling median (2% of trajectories). "
        "Dashed line: original/step80-branch splice.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summaries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    panels = [load_panel(panel, index) for index, panel in enumerate(manifest["panels"])]
    imposed = pd.concat([panel["imposed"] for panel in panels], ignore_index=True)
    realized = pd.concat([panel["realized"] for panel in panels], ignore_index=True)
    imposed.to_csv(args.output_dir / "curriculum_imposed_loss_horizons.csv", index=False)
    realized.to_csv(args.output_dir / "realized_trainable_turns.csv", index=False)

    imposed_summary = plot_imposed(
        panels,
        args.output_dir / "curriculum_imposed_loss_horizons.png",
        manifest["layout_rows"],
    )
    realized_summary = plot_realized(
        panels,
        args.output_dir / "realized_trainable_turns_chronological.png",
        manifest["layout_rows"],
    )
    boundaries = {
        panel["config"]["label"]: panel["boundary"]
        for panel in panels
        if panel["boundary"] is not None
    }
    plot_summary = {
        "panel_order": [panel["config"]["label"] for panel in panels],
        "layout": {
            "rows": manifest["layout_rows"],
            "columns": math.ceil(len(panels) / manifest["layout_rows"]),
        },
        "curriculum_imposed_loss_horizons": imposed_summary,
        "realized_trainable_turns": realized_summary,
        "step80_splices": boundaries,
    }
    write_json(args.output_dir / "plot_summary.json", plot_summary)

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "sources": [
            {
                "label": panel["config"]["label"],
                "source_dir": str(panel["source_dir"]),
                "files": [
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in panel["source_files"]
                ],
            }
            for panel in panels
        ],
        "update_policy": (
            "This directory is the canonical aggregate analysis. Add future "
            "experiments to manifest.json and rerun this script."
        ),
        "outputs": [
            "curriculum_imposed_loss_horizons.png",
            "realized_trainable_turns_chronological.png",
            "curriculum_imposed_loss_horizons.csv",
            "realized_trainable_turns.csv",
            "plot_summary.json",
            "provenance.json",
        ],
    }
    write_json(args.output_dir / "provenance.json", provenance)
    print(json.dumps(plot_summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
