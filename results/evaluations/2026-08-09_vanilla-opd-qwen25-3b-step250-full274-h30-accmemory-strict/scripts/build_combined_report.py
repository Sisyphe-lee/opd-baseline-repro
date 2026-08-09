#!/usr/bin/env python3
"""Build the final paper/TCOD/Vanilla comparison from audited summaries."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[1]
VANILLA_SUMMARY = RUN_ROOT / "evaluation/full274_h30/summary.json"
VANILLA_CONFIG = RUN_ROOT / "configs/eval_full274_h30_accmemory_strict_4gpu.yaml"
TCOD_RUN = REPO_ROOT / "runs/2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory-strict"
TCOD_SUMMARY = TCOD_RUN / "evaluation/full274_h30/summary.json"
TCOD_CONFIG = TCOD_RUN / "configs/eval_full274_h30_accmemory_strict_4gpu.yaml"
TCOD_R4096_SUMMARY = (
    REPO_ROOT
    / "runs/2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory-strict-r4096/evaluation/full274_h30/summary.json"
)
TCOD_TOLERANT_SUMMARY = (
    REPO_ROOT
    / "runs/2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-accmemory/evaluation/full274_h30/summary.json"
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_summary(label: str, summary: dict) -> None:
    assert summary["task_count"] == 274, (label, summary["task_count"])
    assert summary["splits"]["seen"]["task_count"] == 140, label
    assert summary["splits"]["unseen"]["task_count"] == 134, label
    assert summary["success_count"] == (
        summary["splits"]["seen"]["success_count"]
        + summary["splits"]["unseen"]["success_count"]
    ), label


def protocol_signature(config: dict) -> dict:
    tasks = config["buffer"]["explorer_input"]["eval_tasksets"]
    first = tasks[0]
    sampling = first["rollout_args"]
    workflow = first["workflow_args"]
    for task in tasks:
        assert task["rollout_args"] == sampling
        for key in ("max_env_steps", "accumulate_memory", "strict_action_parser"):
            assert task["workflow_args"][key] == workflow[key]
    return {
        "max_prompt_tokens": config["model"]["max_prompt_tokens"],
        "max_response_tokens": config["model"]["max_response_tokens"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "max_tokens": sampling["max_tokens"],
        "max_env_steps": workflow["max_env_steps"],
        "accumulate_memory": workflow["accumulate_memory"],
        "strict_action_parser": workflow["strict_action_parser"],
        "runner_per_model": config["explorer"]["runner_per_model"],
        "engine_num": config["explorer"]["rollout_model"]["engine_num"],
        "tensor_parallel_size": config["explorer"]["rollout_model"]["tensor_parallel_size"],
        "seed": config["explorer"]["rollout_model"]["seed"],
    }


def local_row(method: str, summary: dict, note: str) -> dict:
    return {
        "method": method,
        "seen_success_count": summary["splits"]["seen"]["success_count"],
        "seen_task_count": 140,
        "seen_rate": summary["splits"]["seen"]["success_rate"],
        "unseen_success_count": summary["splits"]["unseen"]["success_count"],
        "unseen_task_count": 134,
        "unseen_rate": summary["splits"]["unseen"]["success_rate"],
        "success_count": summary["success_count"],
        "task_count": 274,
        "overall_rate": summary["success_rate"],
        "note": note,
    }


def paper_row(method: str, seen_percent: float, unseen_percent: float, note: str) -> dict:
    weighted = (seen_percent * 140 + unseen_percent * 134) / 274
    return {
        "method": method,
        "seen_rate": seen_percent / 100,
        "unseen_rate": unseen_percent / 100,
        "overall_rate": weighted / 100,
        "note": note,
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def count_pct(row: dict, split: str) -> str:
    count_key = f"{split}_success_count"
    total_key = f"{split}_task_count"
    rate_key = f"{split}_rate"
    if count_key not in row:
        return pct(row[rate_key])
    return f"{row[count_key]}/{row[total_key]} ({pct(row[rate_key])})"


def overall(row: dict) -> str:
    if "success_count" not in row:
        return f"≈{pct(row['overall_rate'])}"
    return f"{row['success_count']}/{row['task_count']} ({pct(row['overall_rate'])})"


def main() -> None:
    vanilla = load_json(VANILLA_SUMMARY)
    tcod = load_json(TCOD_SUMMARY)
    tcod_r4096 = load_json(TCOD_R4096_SUMMARY)
    tcod_tolerant = load_json(TCOD_TOLERANT_SUMMARY)
    for label, summary in (
        ("Vanilla strict-512", vanilla),
        ("TCOD strict-512", tcod),
        ("TCOD strict-4096", tcod_r4096),
        ("TCOD tolerant-512", tcod_tolerant),
    ):
        validate_summary(label, summary)

    vanilla_protocol = protocol_signature(load_yaml(VANILLA_CONFIG))
    tcod_protocol = protocol_signature(load_yaml(TCOD_CONFIG))
    assert vanilla_protocol == tcod_protocol, (vanilla_protocol, tcod_protocol)
    assert vanilla_protocol == {
        "max_prompt_tokens": 10240,
        "max_response_tokens": 512,
        "temperature": 0.4,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 512,
        "max_env_steps": 30,
        "accumulate_memory": True,
        "strict_action_parser": True,
        "runner_per_model": 16,
        "engine_num": 4,
        "tensor_parallel_size": 1,
        "seed": 42,
    }

    rows = [
        paper_row(
            "Qwen2.5-7B RL teacher (paper)",
            85.71,
            76.87,
            "Paper reference; weighted overall is derived from rounded split rates.",
        ),
        paper_row(
            "Vanilla OPD Qwen2.5-3B (paper)",
            65.72,
            60.45,
            "Paper Table 2; weighted overall is derived from rounded split rates.",
        ),
        paper_row(
            "TCOD-F2B eta=2 Qwen2.5-3B (paper)",
            81.43,
            79.19,
            "Paper Table 2; weighted overall is derived from rounded split rates.",
        ),
        local_row(
            "TCOD-F2B eta=2 Qwen2.5-3B (local)",
            tcod,
            "Frozen strict-parser 512-token protocol.",
        ),
        local_row(
            "Vanilla OPD Qwen2.5-3B (local)",
            vanilla,
            "Frozen strict-parser 512-token protocol.",
        ),
    ]
    diagnostics = [
        local_row(
            "TCOD-F2B local, tolerant parser",
            tcod_tolerant,
            "512-token diagnostic; parser differs from frozen protocol.",
        ),
        local_row(
            "TCOD-F2B local, strict parser r4096",
            tcod_r4096,
            "4096-token diagnostic; distributed sampling realization differs.",
        ),
    ]
    local_tcod = rows[-2]
    local_vanilla = rows[-1]
    comparison = {
        "tcod_minus_vanilla_seen_pp": 100
        * (local_tcod["seen_rate"] - local_vanilla["seen_rate"]),
        "tcod_minus_vanilla_unseen_pp": 100
        * (local_tcod["unseen_rate"] - local_vanilla["unseen_rate"]),
        "tcod_minus_vanilla_overall_pp": 100
        * (local_tcod["overall_rate"] - local_vanilla["overall_rate"]),
    }
    payload = {
        "schema_version": 1,
        "protocol": vanilla_protocol,
        "rows": rows,
        "diagnostic_rows": diagnostics,
        "local_comparison": comparison,
        "sources": {
            "paper_reference_record": str(REPO_ROOT / "VANILLA_OPD_4GPU.md"),
            "tcod_summary": str(TCOD_SUMMARY),
            "vanilla_summary": str(VANILLA_SUMMARY),
            "tcod_r4096_summary": str(TCOD_R4096_SUMMARY),
            "tcod_tolerant_summary": str(TCOD_TOLERANT_SUMMARY),
        },
    }
    (RUN_ROOT / "combined_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Vanilla OPD and TCOD full-274 comparison",
        "",
        "## Main results",
        "",
        "| Method | Seen | Unseen | Overall | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {count_pct(row, 'seen')} | "
            f"{count_pct(row, 'unseen')} | {overall(row)} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "Paper overall values marked with ≈ are weighted derivations from the rounded Seen/Unseen rates; the paper does not report those aggregate counts.",
            "",
            "## Matched local comparison",
            "",
            f"- TCOD minus Vanilla Seen: {comparison['tcod_minus_vanilla_seen_pp']:+.2f} percentage points.",
            f"- TCOD minus Vanilla Unseen: {comparison['tcod_minus_vanilla_unseen_pp']:+.2f} percentage points.",
            f"- TCOD minus Vanilla Overall: {comparison['tcod_minus_vanilla_overall_pp']:+.2f} percentage points.",
            "",
            "## Earlier diagnostic runs",
            "",
            "| Method | Seen | Unseen | Overall | Notes |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in diagnostics:
        lines.append(
            f"| {row['method']} | {count_pct(row, 'seen')} | "
            f"{count_pct(row, 'unseen')} | {overall(row)} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen local protocol",
            "",
            "- full 140 Seen + 134 Unseen tasks;",
            "- h=30; accumulated memory; exact public TCOD action parser;",
            "- temperature 0.4, top-p 1.0, top-k -1, response cap 512;",
            "- seed 42; four TP=1 inference engines on GPUs 0-3.",
            "",
        ]
    )
    (RUN_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(RUN_ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
