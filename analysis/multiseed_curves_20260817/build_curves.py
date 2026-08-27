#!/usr/bin/env python3
"""Assemble step-vs-success curves for all 2026-08-16/17 B200 runs plus the
collaborators' frozen A100-machine curves, write a merged CSV, and plot.

Series identity = method (fixed hue); run/seed = linestyle within the hue.
Palette validated (dataviz six checks): blue #2a78d6 adaptive, orange #eb6834
TCOD, aqua #1baf7a Vanilla (aqua carries direct labels to satisfy the contrast
WARN relief obligation).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path("/localhome/local-tianhej/opd-baseline-repro")
OUT = ROOT / "analysis/multiseed_curves_20260817"
MIRROR = Path("/raid/data0/local-tianhej/opd-assets/runs/mirror_203_results")

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

SERIES = {  # (method, run_label) -> curve rows
}

def add_queue_csv(method, run_label, path, endpoint=None):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["note"].startswith("ok") and r["success_count"]:
                rows[int(r["step"])] = int(r["success_count"])
    if endpoint is not None:
        rows[250] = endpoint
    SERIES[(method, run_label)] = dict(sorted(rows.items()))

def endpoint_from(path):
    return json.load(open(path))["success_count"]

# --- B200 curves (this work) ---
add_queue_csv("tcod", "B200 seed42", ROOT / "runs/ckpt_store/evaluation/allckpt/tcod158/queue_status.csv",
              endpoint_from(ROOT / "runs/ckpt_store/evaluation/b158_repro_tcod_f2b_step250_full274/summary.json"))
add_queue_csv("vanilla", "B200 seed42", ROOT / "runs/ckpt_store/evaluation/allckpt/vanilla158/queue_status.csv",
              endpoint_from(ROOT / "runs/ckpt_store/evaluation/b158_repro_vanilla_opd_step250_full274/summary.json"))
add_queue_csv("adaptive", "B200 seed42", MIRROR / "runs/ckpt_store/evaluation/allckpt/adaptive42/queue_status.csv", 230)
add_queue_csv("adaptive", "B200 seed43", MIRROR / "runs/ckpt_store/evaluation/allckpt/adaptive43/queue_status.csv", 234)

# --- collaborators' frozen A100-machine curves ---
collab_map = {"tcod_f2b": "tcod", "vanilla_opd": "vanilla", "entropy_adaptive_v1_t0100": "adaptive"}
collab = {}
with open(ROOT / "analysis/all_checkpoint_overall_success_curve_seed42/overall_success_curve.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        m = collab_map.get(r["method"])
        if m:
            collab.setdefault(m, {})[int(r["step"])] = int(r["success_count"])
for m, rows in collab.items():
    SERIES[(m, "A100 seed42 (合作者)")] = dict(sorted(rows.items()))

# --- merged CSV ---
with open(OUT / "curves_merged.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "run", "step", "success_count", "success_rate_percent"])
    for (m, run), rows in sorted(SERIES.items()):
        for step, sc in rows.items():
            w.writerow([m, run, step, sc, round(100 * sc / 274, 2)])

# --- plot ---
HUE = {"adaptive": "#2a78d6", "tcod": "#eb6834", "vanilla": "#1baf7a"}
NAME = {"adaptive": "Adaptive τ=0.100（我们的方法）", "tcod": "TCOD-F2B", "vanilla": "Vanilla OPD"}
STYLE = {"B200 seed42": ("-", "o"), "B200 seed43": ("--", "s"), "A100 seed42 (合作者)": (":", "^")}

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150, facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
for (m, run), rows in sorted(SERIES.items()):
    ls, mk = STYLE[run]
    steps, vals = list(rows), [100 * v / 274 for v in rows.values()]
    ax.plot(steps, vals, ls, marker=mk, color=HUE[m], linewidth=2, markersize=5,
            label=f"{NAME[m]} · {run}", alpha=0.95 if "B200" in run else 0.55)

# direct labels at line ends (relief for aqua contrast WARN; ≤4 series direct-labeled).
# Dodge vertically: endpoints 223/224/230 are within 2.6pp of each other.
ends = []
for m in ["adaptive", "tcod", "vanilla"]:
    rows = SERIES[(m, "B200 seed42")]
    step = max(rows)
    ends.append([m, step, 100 * rows[step] / 274, rows[step]])
ends.sort(key=lambda e: e[2])
MIN_GAP = 3.2
for i in range(1, len(ends)):
    if ends[i][2] - ends[i - 1][2] < MIN_GAP:
        ends[i][2] = ends[i - 1][2] + MIN_GAP
for m, step, y_label, count in ends:
    ax.annotate(f"{NAME[m].split('（')[0]} {count}/274", (step, y_label), xytext=(8, 0),
                textcoords="offset points", color="#0b0b0b", fontsize=9, va="center")

ax.set_xlabel("训练步数 (trainer step)", color="#0b0b0b")
ax.set_ylabel("full274 成功率 (%)", color="#0b0b0b")
ax.set_title("三方法逐 checkpoint 成功率曲线（冻结评测协议，274 题，eval seed 42）", color="#0b0b0b")
ax.grid(True, color="#e6e5e2", linewidth=0.8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color("#c8c7c3")
ax.tick_params(colors="#52514e")
ax.set_xlim(10, 285)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "multiseed_curves.png", bbox_inches="tight")
print("saved", OUT / "multiseed_curves.png")
