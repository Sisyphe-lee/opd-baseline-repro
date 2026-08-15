#!/usr/bin/env python3
"""Build a small, decision-oriented TensorBoard dashboard from analysis data."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUTPUT = ANALYSIS / "tensorboard_curated"

EVAL_CSV = ANALYSIS / "recent_three_post80_curve_seed42/overall_success_curve.csv"
LOSS_CSV = ANALYSIS / "entropy_adaptive_v1_all_experiments/supplements/training_loss_baselines/vanilla_tcod_adaptive_t0100_training_loss.csv"
CURRICULUM_CSV = ANALYSIS / "entropy_adaptive_v1_all_experiments/realized_trainable_turns.csv"
ENTROPY_CSV = ANALYSIS / "entropy_adaptive_v1_step250/diagnostics_by_explorer_step_nontruncated.csv"

FIGURES = {
    "06 Reference figures/Checkpoint success curves": ANALYSIS / "recent_three_post80_curve_seed42/overall_success_curve.png",
    "06 Reference figures/Training loss": ANALYSIS / "entropy_adaptive_v1_all_experiments/supplements/training_loss_baselines/vanilla_tcod_adaptive_t0100_training_loss.png",
    "06 Reference figures/Imposed horizons": ANALYSIS / "entropy_adaptive_v1_all_experiments/curriculum_imposed_loss_horizons.png",
    "06 Reference figures/Realized trainable turns": ANALYSIS / "entropy_adaptive_v1_all_experiments/realized_trainable_turns_chronological.png",
    "06 Reference figures/Teacher entropy frontier": ANALYSIS / "entropy_adaptive_v1_step250/teacher_entropy_frontier_heatmap_latest.png",
}


ALIASES = {
    "tcod_f2b": "TCOD_F2B",
    "TCOD F2B": "TCOD_F2B",
    "vanilla_opd": "Vanilla_OPD",
    "Vanilla OPD": "Vanilla_OPD",
    "entropy_adaptive_v1_t0100": "Adaptive_tau_0.100",
    "Adaptive v1 (tau=0.1)": "Adaptive_tau_0.100",
    "Adaptive v1 (tau=0.100, 4 GPU)": "Adaptive_tau_0.100",
    "entropy_adaptive_v1_t0175": "Adaptive_tau_0.175",
    "Adaptive v1 (tau=0.050, 4 GPU)": "Adaptive_tau_0.050",
    "Adaptive v1 (tau=0.075, 4 GPU)": "Adaptive_tau_0.075",
    "Adaptive v1 (tau=0.125, 4 GPU)": "Adaptive_tau_0.125",
    "linear_to_full": "Step80_linear_to_full",
    "Adaptive v1 step80 -> linear anneal -> Vanilla OPD": "Step80_linear_to_full",
    "cosine_to_t0200": "Step80_cosine_to_tau_0.200",
    "Adaptive v1 step80 cosine 0.1->0.2, hold": "Step80_cosine_to_tau_0.200",
    "cosine_to_t0175": "Step80_cosine_to_tau_0.175",
    "Adaptive v1 step80 cosine 0.1->0.175, hold": "Step80_cosine_to_tau_0.175",
    "Adaptive v1 step80 -> immediate Vanilla OPD": "Step80_immediate_full",
}


def run_name(value: Any) -> str:
    text = str(value)
    if text in ALIASES:
        return ALIASES[text]
    return "".join(char if char.isalnum() or char in "._+-" else "_" for char in text).strip("_")


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    staging = OUTPUT.with_name(f".{OUTPUT.name}.partial.{os.getpid()}")
    staging.mkdir(parents=True)
    writers: dict[str, SummaryWriter] = {}
    scalar_events = 0

    def writer(series: str) -> SummaryWriter:
        if series not in writers:
            writers[series] = SummaryWriter(str(staging / series))
        return writers[series]

    def scalar(series: str, tag: str, value: Any, step: Any) -> None:
        nonlocal scalar_events
        number = finite(value)
        logical_step = finite(step)
        if number is None or logical_step is None:
            return
        writer(series).add_scalar(tag, number, int(logical_step))
        scalar_events += 1

    evaluation = pd.read_csv(EVAL_CSV)
    for row in evaluation.itertuples(index=False):
        series = run_name(row.branch)
        scalar(series, "01 Evaluation/Overall success (%)", row.success_rate_percent, row.step)
        scalar(series, "01 Evaluation/Seen success (%)", 100.0 * row.seen_success_count / 140.0, row.step)
        scalar(series, "01 Evaluation/Unseen success (%)", 100.0 * row.unseen_success_count / 134.0, row.step)

    losses = pd.read_csv(LOSS_CSV)
    for row in losses.itertuples(index=False):
        scalar(run_name(row.method), "02 Training/Actor loss (rolling mean)", row.final_loss_rolling_mean_11, row.trainer_step)

    curriculum = pd.read_csv(CURRICULUM_CSV)
    curriculum["prompt_truncated"] = pd.to_numeric(curriculum["prompt_truncated"], errors="coerce")
    clean = curriculum[curriculum["prompt_truncated"].fillna(0) != 1].copy()
    for (method, step), group in clean.groupby(["method", "training_step"], sort=True):
        series = run_name(method)
        scalar(series, "03 Curriculum/Mean trainable turns", group["realized_trainable_turns"].mean(), step)
        scalar(series, "03 Curriculum/Mean imposed horizon", group["imposed_horizon"].mean(), step)
        scalar(series, "03 Curriculum/Frontier trigger rate (%)", 100.0 * group["frontier_triggered"].mean(), step)
        scalar(series, "03 Curriculum/Training task success (%)", 100.0 * group["task_success"].mean(), step)

    entropy = pd.read_csv(ENTROPY_CSV)
    for row in entropy.itertuples(index=False):
        scalar("Adaptive_tau_0.175", "04 Entropy/Teacher entropy", row.teacher_entropy_smooth, row.training_step)
        scalar("Adaptive_tau_0.175", "04 Entropy/Student entropy", row.student_entropy_smooth, row.training_step)
        scalar("Adaptive_tau_0.175", "04 Entropy/Reverse KL", row.sampled_reverse_kl_smooth, row.training_step)

    figure_writer = writer("Reference_figures")
    for tag, path in FIGURES.items():
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            figure_writer.add_image(tag, np.asarray(image), 0, dataformats="HWC")

    readme = (
        "This dashboard intentionally contains only headline evaluation, core training, "
        "curriculum, entropy, and five canonical figures. Full raw TensorBoard exports remain "
        "in `analysis/tensorboard_all_analysis` but are not mounted by the default server. "
        "Curriculum aggregates exclude rows marked `prompt_truncated=1`; baseline rows without "
        "that instrumentation are retained."
    )
    writer("Dashboard_notes").add_text("Dashboard scope", readme, 0)
    for item in writers.values():
        item.flush()
        item.close()

    sources = [EVAL_CSV, LOSS_CSV, CURRICULUM_CSV, ENTROPY_CSV, *FIGURES.values()]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scalar_events": scalar_events,
        "scalar_cards": 11,
        "image_cards": len(FIGURES),
        "series": sorted(writers),
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sources
        ],
        "full_archive_preserved": str(ANALYSIS / "tensorboard_all_analysis"),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    archived = None
    if OUTPUT.exists():
        archived = OUTPUT.with_name(f"{OUTPUT.name}.previous.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        OUTPUT.rename(archived)
    staging.rename(OUTPUT)
    print(json.dumps({**manifest, "output": str(OUTPUT), "archived_previous": str(archived) if archived else None}, indent=2))


if __name__ == "__main__":
    main()
