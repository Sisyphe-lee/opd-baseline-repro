#!/usr/bin/env python3
"""Plot full274 overall success across checkpoints for four main runs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "all_checkpoint_overall_success_curve_seed42"
STEPS = (10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 250)
METHODS = {
    "tcod_f2b": "TCOD F2B",
    "vanilla_opd": "Vanilla OPD",
    "entropy_adaptive_v1_t0100": r"Adaptive v1 ($\tau=0.100$)",
    "entropy_adaptive_v1_t0175": r"Adaptive v1 ($\tau=0.175$)",
}
COLORS = {
    "tcod_f2b": "#F58518",
    "vanilla_opd": "#4C78A8",
    "entropy_adaptive_v1_t0100": "#B279A2",
    "entropy_adaptive_v1_t0175": "#54A24B",
}


def source_path(method: str, step: int) -> tuple[Path, int]:
    if method == "entropy_adaptive_v1_t0175":
        if step in {10, 250}:
            evaluation = "full274" if step == 10 else "step250_full274"
            return (
                ROOT / "runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16"
                / f"evaluation/{evaluation}/summary.json",
                4,
            )
        return (
            ROOT / "runs/experiments/t0175_all_checkpoint_full274_seed42"
            / f"step_{step}_seed42/summary.json",
            5,
        )
    if step in {20, 40, 120, 140, 160, 180, 200, 220, 240}:
        path = (
            ROOT
            / "runs/experiments/all_checkpoint_full274_seed42"
            / method
            / f"step_{step}_seed42/summary.json"
        )
        engine_num = 4 if (method, step) in {
            ("tcod_f2b", 20),
            ("vanilla_opd", 20),
        } else 5
        return path, engine_num
    if step in {60, 80, 100}:
        return (
            ROOT
            / "runs/experiments/warmup_boundary_full274_lyg_seed42"
            / method
            / f"step_{step}_seed42/summary.json",
            4,
        )
    if method == "tcod_f2b":
        return (
            ROOT
            / "results/evaluations/2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory-strict/evaluation/full274_h30/summary.json",
            4,
        )
    if method == "vanilla_opd":
        return (
            ROOT
            / "results/evaluations/2026-08-09_vanilla-opd-qwen25-3b-step250-full274-h30-accmemory-strict/evaluation/full274_h30/summary.json",
            4,
        )
    return (
        ROOT
        / "runs/experiments/entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4/evaluation/step250_full274/summary.json",
        4,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows() -> list[dict]:
    rows = []
    for method in METHODS:
        for step in STEPS:
            if step == 10 and method != "entropy_adaptive_v1_t0175":
                continue
            path, engine_num = source_path(method, step)
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary["task_count"] != 274:
                raise ValueError(f"Expected 274 tasks: {path}")
            rows.append(
                {
                    "method": method,
                    "method_label": METHODS[method].replace("$", ""),
                    "step": step,
                    "success_count": summary["success_count"],
                    "task_count": summary["task_count"],
                    "success_rate": summary["success_rate"],
                    "success_rate_percent": 100.0 * summary["success_rate"],
                    "seen_success_count": summary["splits"]["seen"]["success_count"],
                    "unseen_success_count": summary["splits"]["unseen"]["success_count"],
                    "engine_num": engine_num,
                    "source": str(path.relative_to(ROOT)),
                    "source_sha256": sha256(path),
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
    for method, label in METHODS.items():
        values = [row for row in rows if row["method"] == method]
        x = [row["step"] for row in values]
        y = [row["success_rate_percent"] for row in values]
        ax.plot(x, y, color=COLORS[method], linewidth=2.4, label=label, zorder=2)
        for engine_num, marker, fill in ((4, "o", "none"), (5, "o", COLORS[method])):
            subset = [row for row in values if row["engine_num"] == engine_num]
            ax.scatter(
                [row["step"] for row in subset],
                [row["success_rate_percent"] for row in subset],
                s=58,
                marker=marker,
                facecolors=fill,
                edgecolors=COLORS[method],
                linewidths=1.8,
                zorder=3,
            )

    ax.set_title("ALFWorld full274 Overall Success Across Training Checkpoints (seed 42)", fontsize=16)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Overall success rate (%)")
    ax.set_xticks(STEPS)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="both", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    method_legend = ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.add_artist(method_legend)
    topology_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#555555", markeredgewidth=1.6, label="4 inference engines"),
        plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="#777777", markeredgecolor="#777777", label="5 inference engines"),
    ]
    ax.legend(handles=topology_handles, loc="upper left", frameon=False, fontsize=9)
    fig.savefig(OUTPUT / "overall_success_curve.png", dpi=180)
    fig.savefig(OUTPUT / "overall_success_curve.pdf")
    plt.close(fig)

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "evaluation": {
            "tasks": 274,
            "seed": 42,
            "horizon": 30,
            "temperature": 0.4,
            "top_p": 1.0,
            "top_k": -1,
            "response_limit": 512,
            "prompt_limit": 10240,
        },
        "topology_note": "Hollow markers are 4-engine evaluations; filled markers are 5-engine evaluations.",
        "rows": len(rows),
        "outputs": [
            "overall_success_curve.csv",
            "overall_success_curve.png",
            "overall_success_curve.pdf",
        ],
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
