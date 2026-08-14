#!/usr/bin/env python3
"""Plot canonical Vanilla OPD, TCOD F2B, and Adaptive tau=0.1 PPO losses."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


OUTPUT_DIR = (
    REPO_ROOT
    / "analysis/entropy_adaptive_v1_all_experiments/supplements"
    / "training_loss_baselines"
)
ADAPTIVE_LOG = (
    REPO_ROOT
    / "runs/experiments/entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4"
    / "checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1"
    / "entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4_seed42/log/trainer.log"
)
TCOD_PREFIX_WANDB = (
    REPO_ROOT
    / "runs/experiments/legacy_imports/2026-08-11_pre_cleanup/opd-alfworld-sync-repro"
    / "wandb/offline-run-20260807_131419-ptegqin8/run-ptegqin8.wandb"
)
TCOD_SUFFIX_LOG = (
    REPO_ROOT
    / "results/training/tcod_f2b_step250/launcher_logs"
    / "f2b_resume80_20260808T0615Z.log"
)
VANILLA_LOGS = [
    REPO_ROOT
    / "results/training/vanilla_opd_step250/launcher_logs"
    / "vanilla_restart_20260808T0615Z.log",
    REPO_ROOT
    / "results/training/vanilla_opd_step250/launcher_logs"
    / "vanilla_resume100_20260808.log",
    REPO_ROOT
    / "results/training/vanilla_opd_step250/launcher_logs"
    / "vanilla_resume120_20260809.log",
]
COLORS = {
    "Vanilla OPD": "#4C78A8",
    "TCOD F2B": "#F58518",
    "Adaptive v1 (tau=0.1)": "#7A5195",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> pd.DataFrame:
    pattern = re.compile(r"Step (\d+): \{[^\n]*?'actor/final_loss': ([^,}]+)")
    records = [
        {"trainer_step": int(step), "actor/final_loss": float(loss)}
        for step, loss in pattern.findall(path.read_text(encoding="utf-8", errors="replace"))
    ]
    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates("trainer_step", keep="last")
        .sort_values("trainer_step")
        .reset_index(drop=True)
    )


def parse_wandb_history(path: Path) -> pd.DataFrame:
    store = DataStore()
    store.open_for_scan(str(path))
    records = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        values = {}
        for item in record.history.item:
            key = ".".join(item.nested_key) if item.nested_key else item.key
            if key in {"_step", "actor/final_loss"}:
                values[key] = json.loads(item.value_json)
        if "_step" in values and "actor/final_loss" in values:
            records.append(
                {
                    "trainer_step": int(values["_step"]),
                    "actor/final_loss": float(values["actor/final_loss"]),
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates("trainer_step", keep="last")
        .sort_values("trainer_step")
        .reset_index(drop=True)
    )


def assemble() -> pd.DataFrame:
    adaptive = parse_log(ADAPTIVE_LOG)
    adaptive["method"] = "Adaptive v1 (tau=0.1)"

    tcod_prefix = parse_wandb_history(TCOD_PREFIX_WANDB)
    tcod_suffix = parse_log(TCOD_SUFFIX_LOG)
    tcod = pd.concat(
        [
            tcod_prefix.loc[tcod_prefix["trainer_step"] <= 80],
            tcod_suffix.loc[tcod_suffix["trainer_step"] >= 81],
        ],
        ignore_index=True,
    )
    tcod["method"] = "TCOD F2B"

    vanilla_parts = [parse_log(path) for path in VANILLA_LOGS]
    vanilla = pd.concat(
        [
            vanilla_parts[0].loc[vanilla_parts[0]["trainer_step"] <= 100],
            vanilla_parts[1].loc[vanilla_parts[1]["trainer_step"].between(101, 120)],
            vanilla_parts[2].loc[vanilla_parts[2]["trainer_step"] >= 121],
        ],
        ignore_index=True,
    )
    vanilla["method"] = "Vanilla OPD"

    result = pd.concat([vanilla, tcod, adaptive], ignore_index=True)
    for method, part in result.groupby("method"):
        steps = part["trainer_step"].astype(int).tolist()
        if steps != list(range(1, 251)):
            raise ValueError(f"{method}: expected exactly trainer steps 1--250")
    result["final_loss_rolling_mean_11"] = result.groupby("method", sort=False)[
        "actor/final_loss"
    ].transform(lambda values: values.rolling(11, center=True, min_periods=3).mean())
    return result


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE9", alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot(data: pd.DataFrame, output: Path) -> None:
    methods = ("Vanilla OPD", "TCOD F2B", "Adaptive v1 (tau=0.1)")
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.5), constrained_layout=True)

    ax = axes[0]
    for method in methods:
        part = data.loc[data["method"].eq(method)]
        ax.plot(
            part["trainer_step"], part["actor/final_loss"],
            color=COLORS[method], alpha=0.17, linewidth=0.85,
        )
        ax.plot(
            part["trainer_step"], part["final_loss_rolling_mean_11"],
            color=COLORS[method], linewidth=2.3,
            label=f"{method} (11-step rolling mean)",
        )
    for step, label in ((80, "TCOD resume"), (100, "Vanilla resume"), (120, "Vanilla resume")):
        ax.axvline(step, color="#4C566A", linestyle=":", linewidth=1.0, alpha=0.75)
        ax.text(step + 2, 0.72, label, rotation=90, va="top", fontsize=8, color="#4C566A")
    ax.set_xlim(0, 251)
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Actor final loss")
    ax.set_title("A. Full range (including TCOD cold-start scale)", loc="left", weight="bold")
    ax.legend(frameon=False)
    style_axis(ax)

    ax = axes[1]
    for method in methods:
        part = data.loc[data["method"].eq(method) & data["trainer_step"].ge(20)]
        ax.plot(
            part["trainer_step"], part["actor/final_loss"],
            color=COLORS[method], alpha=0.15, linewidth=0.8,
        )
        ax.plot(
            part["trainer_step"], part["final_loss_rolling_mean_11"],
            color=COLORS[method], linewidth=2.3, label=method,
        )
    for step in (80, 100, 120):
        ax.axvline(step, color="#4C566A", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.set_xlim(20, 251)
    ax.set_ylim(0, 0.17)
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Actor final loss")
    ax.set_title("B. Detail after the earliest TCOD warm-up", loc="left", weight="bold")
    ax.legend(frameon=False)
    style_axis(ax)

    fig.suptitle(
        "PPO training loss: Vanilla OPD vs TCOD F2B vs Adaptive tau=0.1",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = assemble()
    plot_path = OUTPUT_DIR / "vanilla_tcod_adaptive_t0100_training_loss.png"
    csv_path = OUTPUT_DIR / "vanilla_tcod_adaptive_t0100_training_loss.csv"
    summary_path = OUTPUT_DIR / "summary.json"
    provenance_path = OUTPUT_DIR / "provenance.json"
    plot(data, plot_path)
    data.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {}
    for method, part in data.groupby("method", sort=False):
        summary[method] = {
            "step_count": int(len(part)),
            "full_mean": float(part["actor/final_loss"].mean()),
            "steps81_250_mean": float(
                part.loc[part["trainer_step"] >= 81, "actor/final_loss"].mean()
            ),
            "last50_mean": float(part.tail(50)["actor/final_loss"].mean()),
            "step250": float(part.iloc[-1]["actor/final_loss"]),
        }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    inputs = {
        "Adaptive trainer log": ADAPTIVE_LOG,
        "TCOD prefix offline W&B": TCOD_PREFIX_WANDB,
        "TCOD resume80 log": TCOD_SUFFIX_LOG,
        **{f"Vanilla segment {index + 1}": path for index, path in enumerate(VANILLA_LOGS)},
    }
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "metric": "actor/final_loss",
        "smoothing": "centered 11-step rolling arithmetic mean; raw values retained",
        "assembly": {
            "TCOD F2B": "offline W&B steps1-80 + resume log steps81-250",
            "Vanilla OPD": "restart steps1-100 + resume100 steps101-120 + resume120 steps121-250",
            "Adaptive v1 (tau=0.1)": "single trainer log steps1-250",
        },
        "inputs": {
            label: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for label, path in inputs.items()
        },
        "outputs": [plot_path.name, csv_path.name, summary_path.name],
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
