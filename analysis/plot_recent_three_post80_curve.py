#!/usr/bin/env python3
"""Plot every compatible full274 checkpoint success curve on one axis."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "recent_three_post80_curve_seed42"
QUEUE_ROOT = ROOT / "runs" / "experiments" / "recent_three_post80_full274_seed42"
STEPS = (80, 100, 120, 140, 160, 180, 200, 220, 240, 250)
ALL_STEPS = (10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 250)
ALL_CHECKPOINT_CSV = ROOT / "analysis" / "all_checkpoint_overall_success_curve_seed42" / "overall_success_curve.csv"
BRANCHES = {
    "linear_to_full": "Linear anneal to full by step 160",
    "cosine_to_t0200": r"Cosine anneal to $\tau=0.200$ by step 160",
    "cosine_to_t0175": r"Cosine anneal to $\tau=0.175$ by step 160",
}
SERIES = {
    "tcod_f2b": "TCOD F2B",
    "vanilla_opd": "Vanilla OPD",
    "entropy_adaptive_v1_t0100": "Adaptive v1 (τ=0.100)",
    "entropy_adaptive_v1_t0175": "Adaptive v1 (τ=0.175)",
    **BRANCHES,
}
COLORS = {
    "tcod_f2b": "#F28E2B",
    "vanilla_opd": "#4E79A7",
    "entropy_adaptive_v1_t0100": "#B07AA1",
    "entropy_adaptive_v1_t0175": "#59A14F",
    "linear_to_full": "#E15759",
    "cosine_to_t0200": "#76B7B2",
    "cosine_to_t0175": "#EDC948",
}
LINESTYLES = {key: ("--" if key in BRANCHES else "-") for key in SERIES}


def source_path(branch: str, step: int) -> tuple[Path, int, str]:
    if step == 80:
        return (
            ROOT / "runs/experiments/warmup_boundary_full274_lyg_seed42"
            / "entropy_adaptive_v1_t0100/step_80_seed42/summary.json",
            4,
            "shared-original-step80",
        )
    if step == 250 and branch == "linear_to_full":
        return (
            ROOT / "runs/experiments/entropy_adaptive_v1_t0100_step80_linear_anneal_to_full_step160_250step_4gpu_s1t1_r4"
            / "evaluation/step250_full274/summary.json",
            4,
            "reused-existing-step250",
        )
    if step == 250 and branch == "cosine_to_t0200":
        return (
            ROOT / "runs/experiments/entropy_adaptive_v1_t0100_step80_cosine_to_t0200_step160_hold_250step_4gpu_s1t1_r4"
            / "evaluation/step250_full274/summary.json",
            4,
            "reused-existing-step250",
        )
    return (
        QUEUE_ROOT / branch / f"step_{step}_seed42/summary.json",
         4,
        "new-post80-evaluation",
    )


def engine_num_from_config(summary_path: Path, fallback: int) -> int:
    config_path = summary_path.parent / "eval_config.yaml"
    if not config_path.is_file():
        return fallback
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return int(config["explorer"]["rollout_model"]["engine_num"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows() -> list[dict]:
    rows = []
    for branch, label in BRANCHES.items():
        for step in STEPS:
            path, engine_num, provenance = source_path(branch, step)
            engine_num = engine_num_from_config(path, engine_num)
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("task_count") != 274:
                raise ValueError(f"Expected 274 tasks: {path}")
            rows.append(
                {
                    "branch": branch,
                    "branch_label": label.replace("$", ""),
                    "step": step,
                    "success_count": summary["success_count"],
                    "task_count": summary["task_count"],
                    "success_rate": summary["success_rate"],
                    "success_rate_percent": 100.0 * summary["success_rate"],
                    "seen_success_count": summary["splits"]["seen"]["success_count"],
                    "unseen_success_count": summary["splits"]["unseen"]["success_count"],
                    "engine_num": engine_num,
                    "provenance": provenance,
                    "source": str(path.relative_to(ROOT)),
                    "source_sha256": sha256(path),
                }
            )
    with ALL_CHECKPOINT_CSV.open(encoding="utf-8", newline="") as handle:
        for source_row in csv.DictReader(handle):
            rows.append(
                {
                    "branch": source_row["method"],
                    "branch_label": source_row["method_label"],
                    "step": int(source_row["step"]),
                    "success_count": int(source_row["success_count"]),
                    "task_count": int(source_row["task_count"]),
                    "success_rate": float(source_row["success_rate"]),
                    "success_rate_percent": float(source_row["success_rate_percent"]),
                    "seen_success_count": int(source_row["seen_success_count"]),
                    "unseen_success_count": int(source_row["unseen_success_count"]),
                    "engine_num": int(source_row["engine_num"]),
                    "provenance": "existing-all-checkpoint-curve",
                    "source": source_row["source"],
                    "source_sha256": source_row["source_sha256"],
                }
            )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    csv_path = OUTPUT / "overall_success_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    for branch, label in SERIES.items():
        values = [row for row in rows if row["branch"] == branch]
        color = COLORS[branch]
        ax.plot(
            [row["step"] for row in values],
            [row["success_rate_percent"] for row in values],
            color=color,
            linewidth=2.5,
            linestyle=LINESTYLES[branch],
            marker="o",
            markersize=6.5,
            label=label,
            zorder=2,
        )

    ax.set_title("ALFWorld full274: All Training Checkpoint Success Curves (seed 42)", fontsize=16)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Overall success rate (%)")
    ax.set_xticks(ALL_STEPS)
    ax.set_ylim(0, 95)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    fig.savefig(OUTPUT / "overall_success_curve.png", dpi=180)
    fig.savefig(OUTPUT / "overall_success_curve.pdf")
    plt.close(fig)

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "shared_step80": True,
        "combined_series": list(SERIES),
        "input_csv_sha256": sha256(ALL_CHECKPOINT_CSV),
        "evaluation": {
            "tasks": 274,
            "seed": 42,
            "horizon": 30,
            "temperature": 0.4,
            "top_p": 1.0,
            "top_k": -1,
            "response_limit": 512,
            "prompt_limit": 10240,
            "accumulate_memory": True,
            "strict_action_parser": True,
        },
        "topology_note": "The combined source contains both four- and five-engine evaluations; engine_num is retained per point in the CSV. Lines compare checkpoint trends, while exact paired claims require matching topology.",
        "rows": len(rows),
        "outputs": ["overall_success_curve.csv", "overall_success_curve.png", "overall_success_curve.pdf"],
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
