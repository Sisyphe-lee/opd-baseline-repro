#!/usr/bin/env python3
"""Aggregate and plot Vanilla OPD trajectory diagnostics in one command.

The script is intentionally dependency-light: JSONL and log parsing use the
standard library, and figures use Matplotlib.  It accepts one or more JSONL
files in chronological order.  If files contain the same
``(diagnostics_source, training_step)``, the last file wins for that source-step;
this makes a resumed run replace an earlier partial step without allowing a
fixed-panel file to overwrite training diagnostics at the same step.

Example:

    .venv_tcod/bin/python scripts/plot_vanilla_opd_diagnostics.py \
      --diagnostics runs/.../diagnostics/trajectory_metrics.jsonl \
                   runs/.../diagnostics/trajectory_metrics_resume.jsonl \
      --explorer-log reproduction_outputs/.../log/explorer.log \
      --trainer-log reproduction_outputs/.../log/trainer.log \
      --checkpoint-job-dir reproduction_outputs/.../qwen3-... \
      --output-dir runs/.../analysis

The output directory is safe to regenerate: CSV, PNG, JSON, and Markdown
outputs are overwritten on every invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(r"\bStep\s+(\d+):\s+\{")
SCALAR_RE = re.compile(r"'([^']+)':\s*(" + NUMBER + r")")

STUDENT = "#2F6B9A"
TEACHER = "#D9833F"
NEUTRAL = "#4B5563"
NEUTRAL_LIGHT = "#9CA3AF"
SUCCESS = "#6B7D3A"
GRID = "#E5E7EB"
INK = "#1F2937"

DIAGNOSTIC_FIELDS = [
    "diagnostics_step",
    "model_version",
    "source",
    "diagnostics_top_k",
    "row_count",
    "trajectory_count",
    "complete_step",
    "success_count",
    "success_denominator",
    "success_rate",
    "timeout_rate",
    "lost_rate",
    "mean_env_rounds",
    "valid_action_rate",
    "response_tokens_total",
    "mean_response_tokens",
    "student_entropy_topk_mean",
    "teacher_entropy_topk_mean",
    "entropy_teacher_minus_student",
    "student_surprisal_mean",
    "teacher_surprisal_mean",
    "surprisal_teacher_minus_student",
    "sampled_forward_kl_mean",
    "sampled_forward_kl_token_weighted",
    "sampled_forward_kl_sum_mean_per_trajectory",
    "sampled_reverse_kl_mean",
    "sampled_reverse_kl_token_weighted",
    "student_topk_mass_mean",
    "teacher_topk_mass_mean",
    "topk_mass_student_minus_teacher",
    "rollout_success_rate_from_log",
    "rollout_timeout_rate_from_log",
    "rollout_rounds_mean_from_log",
    "rollout_wait_explore_step_sec",
    "rollout_run_execution_mean_sec",
    "rollout_task_execution_mean_sec",
    "experience_pipeline_total_sec",
]

TRAINER_FIELDS = [
    "trainer_step",
    "time_read_experience_sec",
    "time_train_step_sec",
    "time_step_sec",
    "time_trainer_sync_interval_sec",
    "time_sync_weight_sec",
    "time_save_checkpoint_sec",
    "sample_model_version",
    "sample_task_count",
    "critic_score_mean",
    "perf_throughput_tokens_sec",
    "actor_pg_loss",
    "actor_final_loss",
    "actor_ppo_kl",
    "actor_grad_norm",
    "actor_lr",
]

TURN_FIELDS = [
    "turn",
    "trajectory_count",
    "success_trajectory_count",
    "failure_trajectory_count",
    "student_entropy_topk_mean",
    "teacher_entropy_topk_mean",
    "entropy_teacher_minus_student",
    "student_surprisal_mean",
    "teacher_surprisal_mean",
    "sampled_reverse_kl_mean",
    "sampled_reverse_kl_token_weighted",
    "student_topk_mass_mean",
    "teacher_topk_mass_mean",
    "valid_action_rate",
    "mean_response_tokens",
]

PROGRESS_FIELDS = [
    "outcome",
    "progress_bin",
    "progress_fraction",
    "trajectory_count",
    "student_entropy_topk_mean",
    "teacher_entropy_topk_mean",
    "entropy_teacher_minus_student",
]

TRAJECTORY_FIELDS = [
    "diagnostics_step",
    "diagnostics_source",
    "task_id",
    "game_id",
    "task_type",
    "student_model_version",
    "run_id",
    "task_success",
    "turn_count",
    "first_turn",
    "last_turn",
    "student_entropy_mean",
    "teacher_entropy_mean",
    "student_entropy_slope_per_turn",
    "teacher_entropy_slope_per_turn",
    "teacher_entropy_initial",
    "teacher_entropy_final",
    "teacher_entropy_min",
    "teacher_entropy_max",
    "teacher_entropy_peak_turn",
    "teacher_entropy_last5_minus_first5",
    "teacher_entropy_turns5to9_minus_turns0to4",
    "teacher_entropy_max_positive_jump",
    "teacher_entropy_cumulative",
]


def finite(value: Any) -> Optional[float]:
    """Return a finite float, or None for missing/non-numeric values."""

    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "won"}
    return bool(value)


def mean(values: Iterable[Any]) -> Optional[float]:
    numbers = [number for value in values if (number := finite(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def total(values: Iterable[Any]) -> float:
    return sum(number for value in values if (number := finite(value)) is not None)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_jsonl(path: Path, source_index: int) -> Tuple[List[Tuple[int, Dict[str, Any]]], int]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A live writer can leave a partial final line.  It is ignored
                # and reported in the summary instead of breaking plotting.
                malformed += 1
                continue
            if not isinstance(row, dict) or row.get("training_step") is None:
                malformed += 1
                continue
            row = dict(row)
            # The response-level schema v3 emitted by the TCOD workflow uses
            # ``game_id`` as its stable task identity and stores response-mean
            # reverse KL directly. Preserve the raw JSONL while exposing the
            # legacy aliases expected by the plotting code.
            if row.get("diagnostics_kind") == "response_topk_head_entropy":
                if row.get("task_id") is None and row.get("game_id") is not None:
                    row["task_id"] = row["game_id"]
                if (
                    row.get("sampled_reverse_kl_mean") is None
                    and row.get("sampled_reverse_kl") is not None
                ):
                    row["sampled_reverse_kl_mean"] = row["sampled_reverse_kl"]
                if (
                    row.get("sampled_reverse_kl_sum") is None
                    and (kl_mean := finite(row.get("sampled_reverse_kl_mean"))) is not None
                    and (response_tokens := finite(row.get("response_tokens"))) is not None
                ):
                    row["sampled_reverse_kl_sum"] = kl_mean * response_tokens
                if row.get("env_timeout") is None and row.get("env_done") is not None:
                    row["env_timeout"] = not truthy(row["env_done"])
            row["_source"] = path.name
            row["_source_index"] = source_index
            row["_line_number"] = line_number
            rows.append((source_index, row))
    return rows, malformed


def diagnostics_source(row: Mapping[str, Any]) -> str:
    """Return schema-v2 source, treating historical files as training data."""

    return str(row.get("diagnostics_source") or "train")


def training_step(row: Mapping[str, Any]) -> int:
    value = row.get("training_step")
    if isinstance(value, int):
        return value
    match = re.match(r"^(\d+)", str(value))
    if not match:
        raise ValueError(f"Invalid training_step={value!r}")
    return int(match.group(1))


def select_latest_step_source(
    diagnostics_paths: Sequence[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read JSONL files and let the last file own each overlapping step."""

    all_rows: List[Tuple[int, Dict[str, Any]]] = []
    malformed_by_file: Dict[str, int] = {}
    raw_rows_by_file: Dict[str, int] = {}
    for source_index, path in enumerate(diagnostics_paths):
        rows, malformed = read_jsonl(path, source_index)
        all_rows.extend(rows)
        malformed_by_file[path.name] = malformed
        raw_rows_by_file[path.name] = len(rows)

    source_for_step: Dict[Tuple[str, int], int] = {}
    for source_index, row in all_rows:
        try:
            key = (diagnostics_source(row), training_step(row))
        except (TypeError, ValueError):
            continue
        source_for_step[key] = max(source_for_step.get(key, -1), source_index)

    selected = [
        row
        for source_index, row in all_rows
        if source_index
        == source_for_step.get((diagnostics_source(row), training_step(row)), -1)
    ]
    selected_before_prompt_truncation_filter = len(selected)
    selected = [
        row for row in selected if row.get("truncate_status") != "prompt_truncated"
    ]
    prompt_truncated_rows_excluded = (
        selected_before_prompt_truncation_filter - len(selected)
    )
    selected.sort(
        key=lambda row: (
            diagnostics_source(row),
            training_step(row),
            str(row.get("game_id") or row.get("task_id", "")),
            str(row.get("run_id", "")),
            int(row.get("turn", 0)),
        )
    )
    metadata = {
        "input_files": [str(path) for path in diagnostics_paths],
        "raw_rows_by_file": raw_rows_by_file,
        "malformed_lines_by_file": malformed_by_file,
        "selected_rows": len(selected),
        "selected_rows_before_prompt_truncation_filter": (
            selected_before_prompt_truncation_filter
        ),
        "prompt_truncated_rows_excluded": prompt_truncated_rows_excluded,
        "selected_steps_by_source": {
            source: sorted(step for item_source, step in source_for_step if item_source == source)
            for source in sorted({source for source, _ in source_for_step})
        },
        "step_source": {
            f"{source}:{step}": diagnostics_paths[source_index].name
            for (source, step), source_index in sorted(source_for_step.items())
        },
    }
    return selected, metadata


def extract_scalar(line: str, key: str) -> Optional[float]:
    match = re.search(r"'" + re.escape(key) + r"'\s*:\s*(" + NUMBER + r")", line)
    return float(match.group(1)) if match else None


def parse_explorer_log(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    """Parse the last monitor record for each Explorer collection step."""

    if path is None or not path.exists():
        return {}

    records: Dict[int, Dict[str, Any]] = {}
    scalar_keys = {
        "rollout/model_version": "model_version",
        "time/wait_explore_step": "rollout_wait_explore_step_sec",
        "rollout/time/run_execution/mean": "rollout_run_execution_mean_sec",
        "rollout/time/task_execution/mean": "rollout_task_execution_mean_sec",
        "rollout/task_success/mean": "rollout_success_rate_from_log",
        "rollout/env_timeout/mean": "rollout_timeout_rate_from_log",
        "rollout/env_rounds/mean": "rollout_rounds_mean_from_log",
        "rollout/finished_task_count": "finished_task_count",
        "time/experience_pipeline/total": "experience_pipeline_total_sec",
        "experience_pipeline/experience_count": "experience_count",
    }

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            # Explorer writes separate rollout, eval, and sync monitor lines
            # for one step. Merge them instead of letting a later sparse line
            # erase the complete rollout record.
            record: Dict[str, Any] = dict(records.get(step, {"explorer_step": step}))
            for key, output_key in scalar_keys.items():
                value = extract_scalar(line, key)
                if value is not None:
                    record[output_key] = value

            # Keep evaluation monitor success/time if those metrics are
            # present.  The taskset name is preserved in the column name.
            for key, value_text in SCALAR_RE.findall(line):
                if not key.startswith("eval/"):
                    continue
                value = float(value_text)
                parts = key.split("/")
                if len(parts) < 4:
                    continue
                taskset = safe_name(parts[1])
                if key.endswith("/task_success/mean@1"):
                    record[f"eval_{taskset}_success_rate"] = value
                elif key.endswith("/time/run_execution/mean@1"):
                    record[f"eval_{taskset}_run_time_sec"] = value
                elif key.endswith("/time/task_execution/mean@1"):
                    record[f"eval_{taskset}_task_time_sec"] = value

            # Repeated full rollout records are expected on resume. Scalar
            # values from the latest matching line win, while fields from
            # adjacent eval/sync lines remain attached to the same step.
            records[step] = record
    return records


def parse_trainer_log(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    records: Dict[int, Dict[str, Any]] = {}
    scalar_keys = {
        "time/read_experience": "time_read_experience_sec",
        "time/train_step": "time_train_step_sec",
        "time/step": "time_step_sec",
        "time/trainer_sync_interval": "time_trainer_sync_interval_sec",
        "time/sync_weight": "time_sync_weight_sec",
        "time/save_checkpoint": "time_save_checkpoint_sec",
        "sample/model_version/mean": "sample_model_version",
        "sample/task_count": "sample_task_count",
        "critic/score/mean": "critic_score_mean",
        "perf/throughput": "perf_throughput_tokens_sec",
        "actor/pg_loss": "actor_pg_loss",
        "actor/final_loss": "actor_final_loss",
        "actor/ppo_kl": "actor_ppo_kl",
        "actor/grad_norm": "actor_grad_norm",
        "actor/lr": "actor_lr",
    }
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            record: Dict[str, Any] = {"trainer_step": step}
            for key, output_key in scalar_keys.items():
                value = extract_scalar(line, key)
                if value is not None:
                    record[output_key] = value
            records[step] = record
    return records


def aggregate_diagnostics(
    rows: Sequence[Dict[str, Any]],
    explorer_records: Mapping[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[training_step(row)].append(row)

    output: List[Dict[str, Any]] = []
    for step in sorted(by_step):
        step_rows = by_step[step]
        trajectories: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in step_rows:
            key = (
                str(row.get("game_id") or row.get("task_id", "")),
                str(row.get("run_id", "")),
            )
            trajectories[key].append(row)

        # Keep trajectory-level denominators separate from turn-level means.
        trajectory_success = []
        trajectory_timeout = []
        trajectory_lost = []
        trajectory_rounds = []
        trajectory_kl_sum = []
        for items in trajectories.values():
            trajectory_success.append(
                max(1.0 if truthy(item.get("task_success")) else 0.0 for item in items)
            )
            trajectory_timeout.append(
                max(1.0 if truthy(item.get("env_timeout")) else 0.0 for item in items)
            )
            trajectory_lost.append(
                max(1.0 if truthy(item.get("env_lost")) else 0.0 for item in items)
            )
            trajectory_rounds.append(max((finite(item.get("env_rounds")) or 0.0) for item in items))
            trajectory_kl_sum.append(
                total(
                    item.get("sampled_reverse_kl_sum", item.get("sampled_forward_kl_sum"))
                    for item in items
                )
            )

        response_tokens = total(row.get("response_tokens") for row in step_rows)
        kl_sum = total(
            row.get("sampled_reverse_kl_sum", row.get("sampled_forward_kl_sum"))
            for row in step_rows
        )
        explorer = explorer_records.get(step, {})
        row_model_versions = {
            int(value)
            for row in step_rows
            if (value := finite(row.get("student_model_version"))) is not None
        }
        if len(row_model_versions) > 1:
            raise ValueError(
                f"Diagnostics step {step} mixes Student model versions: "
                f"{sorted(row_model_versions)}"
            )
        model_version = (
            next(iter(row_model_versions))
            if row_model_versions
            else explorer.get("model_version")
        )
        if model_version is not None:
            model_version = int(round(model_version))

        top_k = next((row.get("diagnostics_top_k") for row in step_rows if row.get("diagnostics_top_k") is not None), None)
        record: Dict[str, Any] = {
            "diagnostics_step": step,
            "model_version": model_version if model_version is not None else step,
            "source": diagnostics_source(step_rows[0]),
            "diagnostics_top_k": top_k,
            "row_count": len(step_rows),
            "trajectory_count": len(trajectories),
            "success_count": int(sum(trajectory_success)),
            "success_denominator": len(trajectory_success),
            "success_rate": mean(trajectory_success),
            "timeout_rate": mean(trajectory_timeout),
            "lost_rate": mean(trajectory_lost),
            "mean_env_rounds": mean(trajectory_rounds),
            "valid_action_rate": mean(1.0 if truthy(row.get("action_valid")) else 0.0 for row in step_rows),
            "response_tokens_total": response_tokens,
            "mean_response_tokens": mean(row.get("response_tokens") for row in step_rows),
            "student_entropy_topk_mean": mean(row.get("student_entropy_topk") for row in step_rows),
            "teacher_entropy_topk_mean": mean(row.get("teacher_entropy_topk") for row in step_rows),
            "student_surprisal_mean": mean(row.get("student_surprisal") for row in step_rows),
            "teacher_surprisal_mean": mean(row.get("teacher_surprisal") for row in step_rows),
            "sampled_forward_kl_mean": mean(
                row.get("sampled_reverse_kl_mean", row.get("sampled_forward_kl_mean"))
                for row in step_rows
            ),
            "sampled_forward_kl_token_weighted": kl_sum / response_tokens if response_tokens else None,
            "sampled_forward_kl_sum_mean_per_trajectory": mean(trajectory_kl_sum),
            # The historical JSONL key says "forward", but these values are
            # log p_student - log p_teacher on tokens sampled from the
            # student.  Their expectation is the sampled reverse KL
            # D_KL(student || teacher), so expose a correctly named alias.
            "sampled_reverse_kl_mean": mean(
                row.get("sampled_reverse_kl_mean", row.get("sampled_forward_kl_mean"))
                for row in step_rows
            ),
            "sampled_reverse_kl_token_weighted": kl_sum / response_tokens if response_tokens else None,
            "student_topk_mass_mean": mean(row.get("student_topk_mass") for row in step_rows),
            "teacher_topk_mass_mean": mean(row.get("teacher_topk_mass") for row in step_rows),
        }
        if record["student_entropy_topk_mean"] is not None and record["teacher_entropy_topk_mean"] is not None:
            record["entropy_teacher_minus_student"] = record["teacher_entropy_topk_mean"] - record["student_entropy_topk_mean"]
        if record["student_surprisal_mean"] is not None and record["teacher_surprisal_mean"] is not None:
            record["surprisal_teacher_minus_student"] = record["teacher_surprisal_mean"] - record["student_surprisal_mean"]
        if record["student_topk_mass_mean"] is not None and record["teacher_topk_mass_mean"] is not None:
            record["topk_mass_student_minus_teacher"] = record["student_topk_mass_mean"] - record["teacher_topk_mass_mean"]

        for key in (
            "rollout_success_rate_from_log",
            "rollout_timeout_rate_from_log",
            "rollout_rounds_mean_from_log",
            "rollout_wait_explore_step_sec",
            "rollout_run_execution_mean_sec",
            "rollout_task_execution_mean_sec",
            "experience_pipeline_total_sec",
        ):
            record[key] = explorer.get(key)
        for key, value in explorer.items():
            if key.startswith("eval_"):
                record[key] = value

        output.append(record)
    return output


def group_trajectories(
    rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[int, str, str], List[Dict[str, Any]]]:
    """Validate and group turn rows into unique trajectories.

    Trajectory-level charts are intentionally strict.  Duplicate or missing
    turns, mixed top-k definitions, or inconsistent final outcomes would
    silently change the scientific meaning of the curves, so fail fast rather
    than drawing a plausible-looking but invalid figure.
    """

    if not rows:
        return {}

    for row in rows:
        for field in ("diagnostics_top_k", "task_id", "run_id", "task_success"):
            if field not in row or row.get(field) is None:
                raise ValueError(
                    f"Missing required {field} at {row.get('_source', '?')}:"
                    f"{row.get('_line_number', '?')}."
                )
    top_k_values = {
        int(value)
        for row in rows
        if (value := finite(row.get("diagnostics_top_k"))) is not None
    }
    if len(top_k_values) != 1:
        raise ValueError(
            "Trajectory diagnostics must contain exactly one diagnostics_top_k "
            f"value; found {sorted(top_k_values) or 'none'}."
        )

    kinds = {str(row.get("diagnostics_kind", "")) for row in rows}
    supported_kinds = {"topk_head_entropy", "response_topk_head_entropy"}
    if len(kinds) != 1 or not kinds.issubset(supported_kinds):
        raise ValueError(
            "Trajectory entropy charts require one supported diagnostics_kind "
            f"from {sorted(supported_kinds)}; found {sorted(kinds)}."
        )

    trajectories: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for field in ("turn", "student_entropy_topk", "teacher_entropy_topk"):
            if finite(row.get(field)) is None:
                raise ValueError(
                    f"Missing/non-finite {field} at {row.get('_source', '?')}:"
                    f"{row.get('_line_number', '?')}."
                )
        key = (
            training_step(row),
            str(row.get("game_id") or row.get("task_id", "")),
            str(row.get("run_id", "")),
        )
        trajectories[key].append(row)

    for key, items in trajectories.items():
        items.sort(key=lambda row: int(row["turn"]))
        turns = [int(row["turn"]) for row in items]
        expected_turns = list(range(turns[-1] + 1))
        if turns != expected_turns:
            raise ValueError(
                f"Trajectory {key} has duplicate or missing turns: {turns}; "
                f"expected {expected_turns}."
            )
        outcomes = {truthy(row.get("task_success")) for row in items}
        if len(outcomes) != 1:
            raise ValueError(f"Trajectory {key} has inconsistent task_success values.")
    return dict(trajectories)


def aggregate_by_trajectory_turn(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Aggregate equal-weight row means by within-trajectory turn index."""

    by_turn: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for items in trajectories.values():
        for row in items:
            by_turn[int(row["turn"])].append(row)

    output: List[Dict[str, Any]] = []
    for turn in sorted(by_turn):
        turn_rows = by_turn[turn]
        success_count = sum(truthy(row.get("task_success")) for row in turn_rows)
        response_tokens = total(row.get("response_tokens") for row in turn_rows)
        kl_sum = total(
            row.get("sampled_reverse_kl_sum", row.get("sampled_forward_kl_sum"))
            for row in turn_rows
        )
        record: Dict[str, Any] = {
            "turn": turn,
            "trajectory_count": len(turn_rows),
            "success_trajectory_count": success_count,
            "failure_trajectory_count": len(turn_rows) - success_count,
            "student_entropy_topk_mean": mean(
                row.get("student_entropy_topk") for row in turn_rows
            ),
            "teacher_entropy_topk_mean": mean(
                row.get("teacher_entropy_topk") for row in turn_rows
            ),
            "student_surprisal_mean": mean(
                row.get("student_surprisal") for row in turn_rows
            ),
            "teacher_surprisal_mean": mean(
                row.get("teacher_surprisal") for row in turn_rows
            ),
            "sampled_reverse_kl_mean": mean(
                row.get("sampled_reverse_kl_mean", row.get("sampled_forward_kl_mean"))
                for row in turn_rows
            ),
            "sampled_reverse_kl_token_weighted": (
                kl_sum / response_tokens if response_tokens else None
            ),
            "student_topk_mass_mean": mean(
                row.get("student_topk_mass") for row in turn_rows
            ),
            "teacher_topk_mass_mean": mean(
                row.get("teacher_topk_mass") for row in turn_rows
            ),
            "valid_action_rate": mean(
                1.0 if truthy(row.get("action_valid")) else 0.0 for row in turn_rows
            ),
            "mean_response_tokens": mean(
                row.get("response_tokens") for row in turn_rows
            ),
        }
        record["entropy_teacher_minus_student"] = (
            record["teacher_entropy_topk_mean"] - record["student_entropy_topk_mean"]
        )
        output.append(record)
    return output


def linear_slope(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.asarray(values, dtype=float), 1)[0])


def summarize_trajectories(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for (step, task_id, run_id), items in sorted(trajectories.items()):
        student = [float(row["student_entropy_topk"]) for row in items]
        teacher = [float(row["teacher_entropy_topk"]) for row in items]
        record: Dict[str, Any] = {
            "diagnostics_step": step,
            "diagnostics_source": diagnostics_source(items[0]),
            "task_id": task_id,
            "game_id": items[0].get("game_id", task_id),
            "task_type": items[0].get("task_type", ""),
            "student_model_version": items[0].get("student_model_version", step),
            "run_id": run_id,
            "task_success": truthy(items[0].get("task_success")),
            "turn_count": len(items),
            "first_turn": int(items[0]["turn"]),
            "last_turn": int(items[-1]["turn"]),
            "student_entropy_mean": mean(student),
            "teacher_entropy_mean": mean(teacher),
            "student_entropy_slope_per_turn": linear_slope(student),
            "teacher_entropy_slope_per_turn": linear_slope(teacher),
            "teacher_entropy_initial": teacher[0],
            "teacher_entropy_final": teacher[-1],
            "teacher_entropy_min": min(teacher),
            "teacher_entropy_max": max(teacher),
            "teacher_entropy_peak_turn": int(np.argmax(teacher)),
            "teacher_entropy_last5_minus_first5": (
                mean(teacher[-5:]) - mean(teacher[:5]) if len(teacher) >= 10 else None
            ),
            "teacher_entropy_turns5to9_minus_turns0to4": (
                mean(teacher[5:10]) - mean(teacher[:5]) if len(teacher) >= 10 else None
            ),
            "teacher_entropy_max_positive_jump": (
                max((right - left for left, right in zip(teacher, teacher[1:])), default=0.0)
            ),
            "teacher_entropy_cumulative": sum(teacher),
        }
        output.append(record)
    return output


def aggregate_by_normalized_progress(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Dict[str, Any]]],
    progress_bins: int,
) -> List[Dict[str, Any]]:
    """Interpolate every trajectory onto one progress grid, then average.

    Every trajectory contributes once to every normalized-progress point. This
    avoids the survivorship bias of raw-turn curves, where only long (usually
    failed) episodes remain at late turns.
    """

    if progress_bins < 2:
        raise ValueError("progress_bins must be at least 2.")
    grid = np.linspace(0.0, 1.0, progress_bins)
    interpolated: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for items in trajectories.values():
        outcome = "success" if truthy(items[0].get("task_success")) else "failure"
        source_x = (
            np.linspace(0.0, 1.0, len(items))
            if len(items) > 1
            else np.asarray([0.0])
        )
        student = np.asarray([float(row["student_entropy_topk"]) for row in items])
        teacher = np.asarray([float(row["teacher_entropy_topk"]) for row in items])
        interpolated[outcome].append(
            (
                np.interp(grid, source_x, student),
                np.interp(grid, source_x, teacher),
            )
        )

    output: List[Dict[str, Any]] = []
    for outcome in ("failure", "success"):
        values = interpolated.get(outcome, [])
        if not values:
            continue
        students = np.vstack([student for student, _ in values])
        teachers = np.vstack([teacher for _, teacher in values])
        for index, progress in enumerate(grid):
            student_mean = float(np.mean(students[:, index]))
            teacher_mean = float(np.mean(teachers[:, index]))
            output.append(
                {
                    "outcome": outcome,
                    "progress_bin": index,
                    "progress_fraction": float(progress),
                    "trajectory_count": len(values),
                    "student_entropy_topk_mean": student_mean,
                    "teacher_entropy_topk_mean": teacher_mean,
                    "entropy_teacher_minus_student": teacher_mean - student_mean,
                }
            )
    return output


def trajectory_view_summary(
    turn_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "trajectory_count": len(trajectory_rows),
        "turn_min": min((int(row["turn"]) for row in turn_rows), default=None),
        "turn_max": max((int(row["turn"]) for row in turn_rows), default=None),
    }
    if turn_rows:
        first = turn_rows[0]
        last = turn_rows[-1]
        summary.update(
            {
                "turn0_student_entropy": finite(first.get("student_entropy_topk_mean")),
                "turn0_teacher_entropy": finite(first.get("teacher_entropy_topk_mean")),
                "last_turn_student_entropy": finite(last.get("student_entropy_topk_mean")),
                "last_turn_teacher_entropy": finite(last.get("teacher_entropy_topk_mean")),
                "last_turn_trajectory_count": int(last.get("trajectory_count", 0)),
            }
        )
    for success, name in ((False, "failure"), (True, "success")):
        selected = [
            row
            for row in trajectory_rows
            if truthy(row.get("task_success")) == success
        ]
        slopes = [
            value
            for row in selected
            if (value := finite(row.get("teacher_entropy_slope_per_turn"))) is not None
        ]
        paired_deltas = [
            value
            for row in selected
            if (value := finite(row.get("teacher_entropy_last5_minus_first5"))) is not None
        ]
        summary[name] = {
            "trajectory_count": len(selected),
            "mean_turn_count": mean(row.get("turn_count") for row in selected),
            "teacher_entropy_slope_mean": mean(slopes),
            "teacher_entropy_positive_slope_rate": (
                sum(value > 0 for value in slopes) / len(slopes) if slopes else None
            ),
            "teacher_entropy_last5_minus_first5_mean": mean(paired_deltas),
            "paired_delta_trajectory_count": len(paired_deltas),
        }
    return summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def plot_x(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values: List[float] = []
    for row in rows:
        model_version = finite(row.get("model_version"))
        diagnostic_step = finite(row.get("diagnostics_step"))
        values.append(model_version if model_version is not None else (diagnostic_step or 0.0))
    return np.asarray(values, dtype=float)


def ordered_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: (plot_x([row])[0], int(row.get("diagnostics_step", 0))))


def series(rows: Sequence[Mapping[str, Any]], field: str) -> Tuple[np.ndarray, np.ndarray]:
    selected = [(x, finite(row.get(field))) for x, row in zip(plot_x(rows), rows)]
    selected = [(x, y) for x, y in selected if y is not None]
    if not selected:
        return np.asarray([]), np.asarray([])
    return np.asarray([x for x, _ in selected], dtype=float), np.asarray([y for _, y in selected], dtype=float)


def rolling(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < 2 or window <= 1:
        return values
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9CA3AF")
    axis.spines["bottom"].set_color("#9CA3AF")
    axis.tick_params(colors="#4B5563", labelsize=9)
    axis.xaxis.label.set_color("#4B5563")
    axis.yaxis.label.set_color("#4B5563")


def figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.08, y=0.98, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.08, 0.935, subtitle, ha="left", va="top", fontsize=9, color="#4B5563")


def add_series(axis: plt.Axes, rows: Sequence[Mapping[str, Any]], field: str, label: str, color: str, linestyle: str = "-") -> None:
    x, y = series(rows, field)
    if len(x) == 0:
        return
    axis.plot(x, y, color=color, alpha=0.18, linewidth=0.8, marker=".", markersize=3)
    axis.plot(x, rolling(y), color=color, linewidth=2.0, linestyle=linestyle, label=label)


def save_entropy_by_turn(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    status: str,
    top_k: Any,
) -> Optional[Path]:
    """Write the canonical entropy curve against within-trajectory turn."""

    if not rows:
        return None
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0]},
        constrained_layout=False,
    )
    for axis in axes:
        style_axis(axis)
    x = np.asarray([int(row["turn"]) for row in rows], dtype=float)
    for field, label, color, linestyle in (
        ("student_entropy_topk_mean", "Student", STUDENT, "-"),
        ("teacher_entropy_topk_mean", "Teacher", TEACHER, "--"),
    ):
        y = np.asarray([float(row[field]) for row in rows], dtype=float)
        axes[0].plot(
            x,
            y,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            linestyle=linestyle,
            label=label,
        )
    axes[0].set_ylabel("Top-k head entropy (nats)")
    axes[0].set_title("Entropy at each environment turn", loc="left", fontsize=11, color=INK)
    axes[0].legend(frameon=False, loc="best")

    for field, label, color, linestyle in (
        ("trajectory_count", "All trajectories", NEUTRAL, "-"),
        ("failure_trajectory_count", "Final outcome: failure", TEACHER, "--"),
        ("success_trajectory_count", "Final outcome: success", SUCCESS, ":"),
    ):
        y = np.asarray([int(row[field]) for row in rows], dtype=float)
        axes[1].plot(x, y, color=color, linewidth=1.8, linestyle=linestyle, label=label)
    axes[1].set_xlabel("Trajectory turn (0-based)")
    axes[1].set_ylabel("Contributing trajectories")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Effective denominator at each turn", loc="left", fontsize=11, color=INK)
    axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="best")

    trajectory_count = max(int(row["trajectory_count"]) for row in rows)
    figure_header(
        fig,
        f"Top-{top_k or 16} head entropy by trajectory turn",
        f"{status}; equal-weight turn means from {trajectory_count:,} trajectories in complete Explorer batches; no temporal smoothing; final task_success defines outcome.",
    )
    fig.subplots_adjust(top=0.84, left=0.09, right=0.98, bottom=0.10, hspace=0.32)
    path = output_dir / "entropy_curve.png"
    fig.savefig(path, dpi=160, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_entropy_by_model_version(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    status: str,
    top_k: Any,
) -> Optional[Path]:
    """Preserve the training-evolution view under an unambiguous filename."""

    if not any(finite(row.get("student_entropy_topk_mean")) is not None for row in rows):
        return None
    fig, axis = plt.subplots(figsize=(11, 6.2), constrained_layout=False)
    style_axis(axis)
    add_series(axis, rows, "student_entropy_topk_mean", "Student", STUDENT)
    add_series(axis, rows, "teacher_entropy_topk_mean", "Teacher", TEACHER, "--")
    axis.set_xlabel("Student model version (Explorer log; fallback = diagnostics step)")
    axis.set_ylabel("Top-k head entropy (nats)")
    axis.legend(frameon=False, loc="best")
    figure_header(
        fig,
        f"Top-{top_k or 16} head entropy by Student model version",
        f"{status}; each point collapses every trajectory turn in one complete Explorer batch; faint points are raw and lines are 5-step trailing means.",
    )
    fig.subplots_adjust(top=0.82, left=0.09, right=0.98, bottom=0.13)
    path = output_dir / "entropy_by_model_version.png"
    fig.savefig(path, dpi=160, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_teacher_entropy_by_outcome_progress(
    progress_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    status: str,
    top_k: Any,
) -> Optional[Path]:
    if not progress_rows:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), constrained_layout=False)
    for axis in axes:
        style_axis(axis)

    outcome_style = {
        "failure": ("Final outcome: failure", TEACHER, "-"),
        "success": ("Final outcome: success", SUCCESS, "--"),
    }
    for outcome in ("failure", "success"):
        selected = [row for row in progress_rows if row["outcome"] == outcome]
        if not selected:
            continue
        label, color, linestyle = outcome_style[outcome]
        x = np.asarray([float(row["progress_fraction"]) for row in selected])
        y = np.asarray([float(row["teacher_entropy_topk_mean"]) for row in selected])
        count = int(selected[0]["trajectory_count"])
        axes[0].plot(
            x,
            y,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            linestyle=linestyle,
            label=f"{label} (n={count:,})",
        )
    axes[0].set_xlabel("Normalized trajectory progress (0=start, 1=end)")
    axes[0].set_ylabel("Teacher top-k entropy (nats)")
    axes[0].set_title(
        "Teacher entropy over normalized trajectory progress",
        loc="left",
        fontsize=11,
        color=INK,
    )
    axes[0].legend(frameon=False, loc="best")

    box_values: List[List[float]] = []
    box_labels: List[str] = []
    box_colors: List[str] = []
    for success, label, color in (
        (False, "Failure", TEACHER),
        (True, "Success", SUCCESS),
    ):
        values = [
            float(row["teacher_entropy_last5_minus_first5"])
            for row in trajectory_rows
            if truthy(row.get("task_success")) == success
            and finite(row.get("teacher_entropy_last5_minus_first5")) is not None
        ]
        if values:
            box_values.append(values)
            box_labels.append(f"{label}\n(n={len(values):,})")
            box_colors.append(color)
    if box_values:
        artists = axes[1].boxplot(
            box_values,
            tick_labels=box_labels,
            showfliers=False,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": INK, "linewidth": 1.5},
        )
        for patch, color in zip(artists["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
    axes[1].axhline(0.0, color=NEUTRAL_LIGHT, linewidth=1.0, linestyle=":")
    axes[1].set_ylabel("Last-5 minus first-5 entropy (nats)")
    axes[1].set_title(
        "Paired within-trajectory Teacher entropy change (trajectories ≥10 turns)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    figure_header(
        fig,
        f"Top-{top_k or 16} Teacher entropy by final trajectory outcome",
        f"{status}; every trajectory is linearly interpolated onto the same progress grid and receives equal weight, avoiding late-turn survivorship bias.",
    )
    fig.subplots_adjust(top=0.84, left=0.09, right=0.98, bottom=0.09, hspace=0.38)
    path = output_dir / "teacher_entropy_by_outcome_progress.png"
    fig.savefig(path, dpi=160, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_teacher_entropy_rollout_variability(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Dict[str, Any]]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    status: str,
    top_k: Any,
) -> Optional[Path]:
    """Show trajectory-to-trajectory variation without hiding it in one mean."""

    if not trajectories or not trajectory_rows:
        return None
    summary_by_key = {
        (
            int(row["diagnostics_step"]),
            str(row.get("game_id") or row["task_id"]),
            str(row["run_id"]),
        ): row
        for row in trajectory_rows
    }
    by_outcome: Dict[bool, List[Tuple[np.ndarray, np.ndarray, float]]] = {
        False: [],
        True: [],
    }
    for key, items in trajectories.items():
        summary = summary_by_key[key]
        success = truthy(summary.get("task_success"))
        turns = np.asarray([int(row["turn"]) for row in items], dtype=float)
        entropy = np.asarray(
            [float(row["teacher_entropy_topk"]) for row in items], dtype=float
        )
        slope = float(summary["teacher_entropy_slope_per_turn"] or 0.0)
        by_outcome[success].append((turns, entropy, slope))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.2), constrained_layout=False)
    for axis in axes.flat:
        style_axis(axis)

    for axis, success, label, color in (
        (axes[0, 0], False, "Final outcome: failure", TEACHER),
        (axes[0, 1], True, "Final outcome: success", SUCCESS),
    ):
        items = sorted(by_outcome[success], key=lambda item: item[2])
        if not items:
            continue
        # Draw an evenly spaced sample over slope rank instead of a random
        # sample, so both slow- and fast-changing trajectories remain visible.
        sample_count = min(80, len(items))
        sample_indices = np.unique(
            np.linspace(0, len(items) - 1, sample_count).round().astype(int)
        )
        for index in sample_indices:
            turns, entropy, _ = items[int(index)]
            axis.plot(turns, entropy, color=color, linewidth=0.65, alpha=0.12)

        max_turn = max(int(turns[-1]) for turns, _, _ in items)
        x = np.arange(max_turn + 1, dtype=float)
        q25: List[float] = []
        q50: List[float] = []
        q75: List[float] = []
        counts: List[int] = []
        for turn in range(max_turn + 1):
            values = [entropy[turn] for _, entropy, _ in items if len(entropy) > turn]
            counts.append(len(values))
            q25.append(float(np.quantile(values, 0.25)))
            q50.append(float(np.quantile(values, 0.50)))
            q75.append(float(np.quantile(values, 0.75)))
        axis.fill_between(x, q25, q75, color=color, alpha=0.22, label="IQR")
        axis.plot(x, q50, color=color, linewidth=2.5, label="Median")
        axis.set_xlabel("Trajectory turn (0-based)")
        axis.set_ylabel("Teacher top-k entropy (nats)")
        axis.set_title(
            f"{label} (n={len(items):,})",
            loc="left",
            fontsize=11,
            color=INK,
        )
        axis.legend(frameon=False, fontsize=8, loc="best")
        axis.text(
            0.99,
            0.02,
            f"late-turn n={counts[-1]:,}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#4B5563",
        )

    grouped_slopes: List[List[float]] = []
    grouped_labels: List[str] = []
    grouped_colors: List[str] = []
    rng = np.random.default_rng(0)
    for position, success, label, color in (
        (1, False, "Failure", TEACHER),
        (2, True, "Success", SUCCESS),
    ):
        slopes = [item[2] for item in by_outcome[success]]
        if not slopes:
            continue
        grouped_slopes.append(slopes)
        grouped_labels.append(f"{label}\n(n={len(slopes):,})")
        grouped_colors.append(color)
        jitter = rng.uniform(-0.17, 0.17, len(slopes))
        axes[1, 0].scatter(
            np.full(len(slopes), position) + jitter,
            slopes,
            color=color,
            s=8,
            alpha=0.16,
            linewidths=0,
        )
    if grouped_slopes:
        artists = axes[1, 0].boxplot(
            grouped_slopes,
            tick_labels=grouped_labels,
            showfliers=False,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": INK, "linewidth": 1.5},
        )
        for patch, color in zip(artists["boxes"], grouped_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
    axes[1, 0].axhline(0.0, color=NEUTRAL_LIGHT, linewidth=1.0, linestyle=":")
    axes[1, 0].set_ylabel("Teacher entropy slope (nats/turn)")
    axes[1, 0].set_title(
        "Distribution of per-trajectory linear slopes",
        loc="left",
        fontsize=11,
        color=INK,
    )

    for success, label, color, marker in (
        (False, "Failure", TEACHER, "o"),
        (True, "Success", SUCCESS, "x"),
    ):
        selected = [
            row
            for row in trajectory_rows
            if truthy(row.get("task_success")) == success
        ]
        axes[1, 1].scatter(
            [float(row["teacher_entropy_initial"]) for row in selected],
            [float(row["teacher_entropy_slope_per_turn"] or 0.0) for row in selected],
            color=color,
            marker=marker,
            s=12,
            alpha=0.22,
            linewidths=0.6 if marker == "x" else 0,
            label=f"{label} (n={len(selected):,})",
        )
    axes[1, 1].axhline(0.0, color=NEUTRAL_LIGHT, linewidth=1.0, linestyle=":")
    axes[1, 1].set_xlabel("Teacher entropy at turn 0 (nats)")
    axes[1, 1].set_ylabel("Teacher entropy slope (nats/turn)")
    axes[1, 1].set_title(
        "Initial uncertainty versus subsequent slope",
        loc="left",
        fontsize=11,
        color=INK,
    )
    axes[1, 1].legend(frameon=False, fontsize=8, loc="best")

    figure_header(
        fig,
        f"Top-{top_k or 16} Teacher entropy variability across rollouts",
        f"{status}; thin lines sample trajectories evenly across slope rank; bands show turn-wise IQR and thick lines show medians; late successful turns have fewer survivors.",
    )
    fig.subplots_adjust(
        top=0.86, left=0.075, right=0.98, bottom=0.08, hspace=0.32, wspace=0.22
    )
    path = output_dir / "teacher_entropy_rollout_variability.png"
    fig.savefig(path, dpi=160, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def _teacher_entropy_delta(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.asarray([float(row["teacher_entropy_topk"]) for row in items])
    baseline = float(np.mean(values[: min(3, len(values))]))
    return values - baseline


def _first_crossing(delta: np.ndarray, threshold: float, window: int = 3) -> Optional[int]:
    if len(delta) < window:
        return None
    smoothed = np.convolve(delta, np.ones(window) / window, mode="valid")
    crossings = np.flatnonzero(smoothed >= threshold)
    return int(crossings[0] + window - 1) if len(crossings) else None


def save_teacher_entropy_frontier_heatmap(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Dict[str, Any]]],
    output_dir: Path,
    status: str,
    top_k: Any,
    filename: str = "teacher_entropy_frontier_heatmap.png",
) -> Optional[Path]:
    """Show rollout-specific Teacher-entropy drift, split by task outcome."""

    outcome_groups = [
        (
            "Failure",
            [
                items
                for items in trajectories.values()
                if not truthy(items[0].get("task_success"))
            ],
        ),
        (
            "Success",
            [
                items
                for items in trajectories.values()
                if truthy(items[0].get("task_success"))
            ],
        ),
    ]
    if not any(items for _, items in outcome_groups):
        return None
    threshold = 0.15
    prepared_groups = []
    all_deltas = []
    for label, items_group in outcome_groups:
        prepared = [(_teacher_entropy_delta(items), items) for items in items_group]
        prepared.sort(
            key=lambda pair: (
                _first_crossing(pair[0], threshold)
                if _first_crossing(pair[0], threshold) is not None
                else 10**6,
                -(linear_slope(pair[0].tolist()) or 0.0),
            )
        )
        prepared_groups.append((label, prepared))
        all_deltas.extend(delta for delta, _ in prepared)

    max_turns = max(len(delta) for delta in all_deltas)
    finite_values = np.concatenate(all_deltas)
    limit = max(float(np.quantile(np.abs(finite_values), 0.98)), 0.05)
    cmap = LinearSegmentedColormap.from_list(
        "frontier", ["#2F6B9A", "#F7F4ED", "#D9833F"]
    )
    cmap.set_bad("#FFFFFF")

    ordered = [pair for _, prepared in prepared_groups for pair in prepared]
    matrix = np.full((len(ordered), max_turns), np.nan)
    for row_index, (delta, _) in enumerate(ordered):
        matrix[row_index, : len(delta)] = delta

    fig, axis = plt.subplots(
        figsize=(12, min(15, max(6, len(ordered) * 0.025)))
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
    )
    group_sizes = [len(prepared) for _, prepared in prepared_groups]
    failure_count, success_count = group_sizes
    if failure_count and success_count:
        axis.axhline(
            failure_count - 0.5,
            color=INK,
            linewidth=1.1,
            linestyle="--",
            alpha=0.9,
        )
    tick_positions = []
    tick_labels = []
    offset = 0
    for (label, _), group_size in zip(prepared_groups, group_sizes):
        if group_size:
            tick_positions.append(offset + (group_size - 1) / 2)
            tick_labels.append(f"{label} (n={group_size:,})")
        offset += group_size
    axis.set_yticks(tick_positions, labels=tick_labels)
    axis.set_xlabel("Trajectory turn (0-based)")
    axis.set_ylabel("Rollouts grouped by outcome")
    figure_header(
        fig,
        f"Top-{top_k or 16} Teacher entropy drift by outcome and rollout",
        f"{status}; one row per rollout; failures precede successes; each group is sorted by first sustained ΔH≥{threshold:.2f}; white cells are after termination.",
    )
    colorbar = fig.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Teacher entropy ΔH from first ≤3 turns (nats)")
    fig.subplots_adjust(top=0.88, left=0.13, right=0.94, bottom=0.10)
    path = output_dir / filename
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_teacher_entropy_threshold_crossing(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Dict[str, Any]]],
    output_dir: Path,
    status: str,
    outcome: Optional[bool] = None,
    filename: str = "teacher_entropy_threshold_crossing_all.png",
) -> Optional[Path]:
    """Plot censor-aware survival curves for sustained entropy-threshold crossing."""

    if outcome is None:
        selected = list(trajectories.values())
        outcome_label = "all"
    else:
        selected = [
            items
            for items in trajectories.values()
            if truthy(items[0].get("task_success")) == outcome
        ]
        outcome_label = "successful" if outcome else "failed"
    if not selected:
        return None
    fig, axis = plt.subplots(figsize=(11, 6.3))
    style_axis(axis)
    max_turn = max(len(items) for items in selected) - 1
    for threshold, linestyle, linewidth in (
        (0.10, "-", 2.4),
        (0.15, "--", 2.2),
        (0.20, ":", 2.4),
    ):
        event_times = [
            _first_crossing(_teacher_entropy_delta(items), threshold)
            for items in selected
        ]
        censor_times = [len(items) - 1 for items in selected]
        survival = 1.0
        xs = [0]
        ys = [1.0]
        for turn in range(1, max_turn + 1):
            at_risk = sum(
                censor >= turn and (event is None or event >= turn)
                for event, censor in zip(event_times, censor_times)
            )
            events = sum(event == turn for event in event_times)
            if at_risk:
                survival *= 1.0 - events / at_risk
            xs.append(turn)
            ys.append(survival)
        axis.step(
            xs,
            ys,
            where="post",
            color=TEACHER,
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"ΔH ≥ {threshold:.2f}",
        )
    axis.set_xlabel("Trajectory turn (0-based)")
    axis.set_ylabel("Fraction not yet crossing")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(frameon=False, title="3-turn sustained mean")
    figure_header(
        fig,
        f"Teacher-entropy frontier crossing for {outcome_label} rollouts",
        f"{status}; Kaplan–Meier style estimate over {len(selected):,} {outcome_label} rollouts; termination before crossing is right-censored.",
    )
    fig.subplots_adjust(top=0.82, left=0.10, right=0.98, bottom=0.13)
    path = output_dir / filename
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_token_block_observation_boundary(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Dict[str, Any]]],
    output_dir: Path,
    status: str,
) -> Optional[Path]:
    """Align 4-token entropy blocks around new-observation boundaries."""

    values: Dict[Tuple[bool, int], List[float]] = defaultdict(list)
    jumps: Dict[bool, List[float]] = defaultdict(list)
    for items in trajectories.values():
        success = truthy(items[0].get("task_success"))
        for previous, current in zip(items, items[1:]):
            before = [finite(value) for value in previous.get("teacher_entropy_topk_blocks", [])]
            after = [finite(value) for value in current.get("teacher_entropy_topk_blocks", [])]
            before = [value for value in before if value is not None]
            after = [value for value in after if value is not None]
            if not before or not after:
                continue
            for offset, value in zip(range(-min(2, len(before)), 0), before[-2:]):
                values[(success, offset)].append(value)
            for offset, value in enumerate(after[:4], start=1):
                values[(success, offset)].append(value)
            jumps[success].append(after[0] - before[-1])
    if not values:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), gridspec_kw={"height_ratios": [2.2, 1.0]})
    for axis in axes:
        style_axis(axis)
    for success, label, color, linestyle in (
        (False, "Final failure", TEACHER, "-"),
        (True, "Final success", SUCCESS, "--"),
    ):
        offsets = sorted(offset for outcome, offset in values if outcome == success)
        means = np.asarray([np.mean(values[(success, offset)]) for offset in offsets])
        errors = np.asarray(
            [
                1.96 * np.std(values[(success, offset)]) / math.sqrt(len(values[(success, offset)]))
                if len(values[(success, offset)]) > 1
                else 0.0
                for offset in offsets
            ]
        )
        axes[0].plot(offsets, means, color=color, linestyle=linestyle, linewidth=2.2, marker="o", label=label)
        axes[0].fill_between(offsets, means - errors, means + errors, color=color, alpha=0.16)
    axes[0].axvline(0, color=NEUTRAL_LIGHT, linestyle=":", linewidth=1.3)
    axes[0].set_xticks([-2, -1, 1, 2, 3, 4])
    axes[0].set_xticklabels(["prev −2", "prev last", "new +1", "new +2", "new +3", "new +4"])
    axes[0].set_ylabel("Teacher top-k entropy (nats)")
    axes[0].set_title("Entropy around receipt of a new environment observation", loc="left", fontsize=11, color=INK)
    axes[0].legend(frameon=False)

    box_values = [jumps[outcome] for outcome in (False, True) if jumps[outcome]]
    labels = [f"{'Failure' if not outcome else 'Success'}\n(n={len(jumps[outcome]):,})" for outcome in (False, True) if jumps[outcome]]
    artists = axes[1].boxplot(box_values, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(artists["boxes"], [TEACHER, SUCCESS][: len(box_values)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    axes[1].axhline(0, color=NEUTRAL_LIGHT, linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("First-new minus last-old block (nats)")
    figure_header(
        fig,
        "Teacher entropy at observation boundaries (4-token blocks)",
        f"{status}; means and approximate 95% intervals above; paired boundary jumps below; turn-level entropy_curve.png remains the primary trajectory view.",
    )
    fig.subplots_adjust(top=0.84, left=0.10, right=0.98, bottom=0.09, hspace=0.38)
    path = output_dir / "teacher_entropy_observation_boundary.png"
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


FIXED_PANEL_FIELDS = [
    "diagnostics_step",
    "student_model_version",
    "game_id",
    "task_type",
    "rollout_count",
    "success_rate",
    "teacher_entropy_slope_mean",
    "teacher_entropy_slope_std",
    "teacher_entropy_delta_mean",
    "teacher_entropy_delta_std",
]

SAME_TASK_EFFECT_FIELDS = [
    "term",
    "estimate",
    "cluster_se",
    "cluster_ci_low",
    "cluster_ci_high",
    "cluster_p_value",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "trajectory_count",
    "game_count",
    "model_version_count",
]

SAME_TASK_CONTRAST_FIELDS = [
    "game_id",
    "student_model_version",
    "failure_count",
    "success_count",
    "failure_minus_success_slope",
    "failure_minus_success_delta",
    "failure_minus_success_early_delta",
]

FAILURE_PREDICTION_FIELDS = [
    "cutoff_turns",
    "model",
    "trajectory_count",
    "game_count",
    "failure_count",
    "failure_rate",
    "excluded_early_success_count",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "average_precision",
    "average_precision_ci_low",
    "average_precision_ci_high",
    "brier_score",
]

BOUNDARY_EVENT_FIELDS = [
    "diagnostics_step",
    "student_model_version",
    "game_id",
    "run_id",
    "turn",
    "task_success",
    "action_category",
    "action_verb",
    "observation_words",
    "raw_boundary_jump",
    "same_position_change",
    "position_reset_contrast",
]

ACTION_BOUNDARY_FIELDS = [
    "diagnostics_step",
    "student_model_version",
    "game_id",
    "run_id",
    "turn",
    "task_success",
    "boundary",
    "relative_block",
    "teacher_entropy_topk",
]

FIXED_PANEL_FRONTIER_FIELDS = [
    "student_model_version",
    "outcome",
    "threshold",
    "trajectory_count",
    "crossed_count",
    "crossed_fraction",
    "crossing_turn_q10",
    "crossing_turn_median",
    "crossing_turn_q90",
]


def summarize_fixed_panel_same_task(
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[(int(row["diagnostics_step"]), str(row.get("game_id") or row["task_id"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (step, game_id), rows in sorted(grouped.items()):
        slopes = [float(row["teacher_entropy_slope_per_turn"] or 0.0) for row in rows]
        deltas = [float(row["teacher_entropy_final"]) - float(row["teacher_entropy_initial"]) for row in rows]
        output.append(
            {
                "diagnostics_step": step,
                "student_model_version": rows[0].get("student_model_version", step),
                "game_id": game_id,
                "task_type": rows[0].get("task_type", ""),
                "rollout_count": len(rows),
                "success_rate": mean(truthy(row.get("task_success")) for row in rows),
                "teacher_entropy_slope_mean": mean(slopes),
                "teacher_entropy_slope_std": float(np.std(slopes)),
                "teacher_entropy_delta_mean": mean(deltas),
                "teacher_entropy_delta_std": float(np.std(deltas)),
            }
        )
    return output


def save_fixed_panel_same_task(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, status: str
) -> Optional[Path]:
    if not rows:
        return None
    games = sorted({str(row["game_id"]) for row in rows})
    versions = sorted({int(row["student_model_version"]) for row in rows})
    game_index = {game: index for index, game in enumerate(games)}
    version_index = {version: index for index, version in enumerate(versions)}
    slope_std = np.full((len(games), len(versions)), np.nan)
    success_rate = np.full_like(slope_std, np.nan)
    for row in rows:
        i = game_index[str(row["game_id"])]
        j = version_index[int(row["student_model_version"])]
        slope_std[i, j] = float(row["teacher_entropy_slope_std"])
        success_rate[i, j] = float(row["success_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(14, max(7, len(games) * 0.36)), sharey=True)
    images = [
        axes[0].imshow(slope_std, aspect="auto", interpolation="nearest", cmap="YlOrBr"),
        axes[1].imshow(success_rate, aspect="auto", interpolation="nearest", cmap="YlGn", vmin=0, vmax=1),
    ]
    labels = [Path(game).parts[-3][:42] for game in games]
    for axis, title in zip(axes, ("Within-task SD of entropy slope", "Same-task rollout success rate")):
        axis.set_xticks(range(len(versions)), labels=versions, rotation=45, ha="right")
        axis.set_xlabel("Student model version")
        axis.set_title(title, loc="left", fontsize=11, color=INK)
    axes[0].set_yticks(range(len(games)), labels=labels, fontsize=7)
    axes[0].set_ylabel("Fixed training task")
    for axis, image in zip(axes, images):
        fig.colorbar(image, ax=axis, pad=0.02, shrink=0.8)
    figure_header(
        fig,
        "Fixed-panel stochastic rollout variability within the same task",
        f"{status}; each cell compares repeated stochastic rollouts of one frozen checkpoint on one fixed training task.",
    )
    fig.subplots_adjust(top=0.87, left=0.25, right=0.97, bottom=0.13, wspace=0.18)
    path = output_dir / "fixed_panel_same_task_variability.png"
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def _normalized_progress_model_data(
    trajectories: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]],
    progress_points: int = 11,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """Build an equal-trajectory-weighted progress model matrix.

    Game and Student-version fixed effects condition the outcome interaction on
    the repeated-task panel.  This is a dependency-light fixed-effects analogue
    of the proposed random-intercept mixed model.
    """

    grid = np.linspace(0.0, 1.0, progress_points)
    observations: List[Tuple[float, float, float, str, str]] = []
    trajectory_ids: List[str] = []
    for trajectory_id, items in trajectories:
        if not items:
            continue
        values = np.asarray(
            [float(row["teacher_entropy_topk"]) for row in items], dtype=float
        )
        source_grid = (
            np.linspace(0.0, 1.0, len(values))
            if len(values) > 1
            else np.asarray([0.0])
        )
        interpolated = np.interp(grid, source_grid, values)
        failure = 0.0 if truthy(items[0].get("task_success")) else 1.0
        game = str(items[0].get("game_id") or items[0].get("task_id", ""))
        version = str(items[0].get("student_model_version", training_step(items[0])))
        for progress, entropy in zip(grid, interpolated):
            observations.append((float(entropy), float(progress), failure, game, version))
            trajectory_ids.append(trajectory_id)

    if not observations:
        return np.empty((0, 0)), np.empty(0), [], [], []
    games = sorted({item[3] for item in observations})
    versions = sorted({item[4] for item in observations}, key=lambda value: int(value))
    columns = ["intercept", "progress", "failure", "failure_x_progress"]
    columns.extend(f"game={value}" for value in games[1:])
    columns.extend(f"version={value}" for value in versions[1:])
    matrix: List[List[float]] = []
    outcome: List[float] = []
    cluster: List[str] = []
    for entropy, progress, failure, game, version in observations:
        row = [1.0, progress, failure, failure * progress]
        row.extend(1.0 if game == value else 0.0 for value in games[1:])
        row.extend(1.0 if version == value else 0.0 for value in versions[1:])
        matrix.append(row)
        outcome.append(entropy)
        cluster.append(game)
    return (
        np.asarray(matrix, dtype=float),
        np.asarray(outcome, dtype=float),
        columns,
        cluster,
        trajectory_ids,
    )


def _fit_clustered_linear_model(
    matrix: np.ndarray,
    outcome: np.ndarray,
    clusters: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coefficients = np.linalg.lstsq(matrix, outcome, rcond=None)[0]
    residuals = outcome - matrix @ coefficients
    bread = np.linalg.pinv(matrix.T @ matrix)
    meat = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    unique_clusters = sorted(set(clusters))
    cluster_array = np.asarray(clusters)
    for cluster in unique_clusters:
        selected = cluster_array == cluster
        score = matrix[selected].T @ residuals[selected]
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    n, p = matrix.shape
    g = len(unique_clusters)
    if g > 1 and n > p:
        covariance *= (g / (g - 1)) * ((n - 1) / (n - p))
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    critical = float(stats.t.ppf(0.975, max(g - 1, 1)))
    ci_low = coefficients - critical * standard_errors
    ci_high = coefficients + critical * standard_errors
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.divide(
            coefficients,
            standard_errors,
            out=np.full_like(coefficients, np.nan),
            where=standard_errors > 0,
        )
    p_values = 2.0 * stats.t.sf(np.abs(statistic), max(g - 1, 1))
    return coefficients, standard_errors, ci_low, ci_high, p_values


def analyze_fixed_panel_same_task_effects(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Mapping[str, Any]]],
    bootstrap_replicates: int = 250,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Estimate task/checkpoint-controlled outcome differences in entropy drift."""

    labelled = [("|".join(map(str, key)), items) for key, items in trajectories.items()]
    matrix, outcome, columns, clusters, _ = _normalized_progress_model_data(labelled)
    if matrix.size == 0:
        return [], []
    estimates, standard_errors, ci_low, ci_high, p_values = (
        _fit_clustered_linear_model(matrix, outcome, clusters)
    )

    games = sorted({str(items[0].get("game_id") or items[0].get("task_id", "")) for _, items in labelled})
    versions = sorted({str(items[0].get("student_model_version", training_step(items[0]))) for _, items in labelled})
    by_game: Dict[str, List[Tuple[str, Sequence[Mapping[str, Any]]]]] = defaultdict(list)
    for trajectory_id, items in labelled:
        game = str(items[0].get("game_id") or items[0].get("task_id", ""))
        by_game[game].append((trajectory_id, items))

    bootstrap_values: Dict[str, List[float]] = defaultdict(list)
    if bootstrap_replicates > 0 and len(games) > 1:
        rng = np.random.default_rng(20260806)
        for _ in range(bootstrap_replicates):
            sampled = rng.choice(games, size=len(games), replace=True)
            boot: List[Tuple[str, Sequence[Mapping[str, Any]]]] = []
            for sample_index, game in enumerate(sampled):
                for trajectory_id, items in by_game[str(game)]:
                    copied = [dict(row, game_id=f"bootstrap-{sample_index}") for row in items]
                    boot.append((f"{sample_index}|{trajectory_id}", copied))
            boot_matrix, boot_outcome, boot_columns, _, _ = _normalized_progress_model_data(boot)
            boot_coefficients = np.linalg.lstsq(
                boot_matrix, boot_outcome, rcond=None
            )[0]
            for term in ("progress", "failure", "failure_x_progress"):
                bootstrap_values[term].append(
                    float(boot_coefficients[boot_columns.index(term)])
                )

    selected_terms = ("progress", "failure", "failure_x_progress")
    effects: List[Dict[str, Any]] = []
    for term in selected_terms:
        index = columns.index(term)
        bootstrap = bootstrap_values.get(term, [])
        effects.append(
            {
                "term": term,
                "estimate": float(estimates[index]),
                "cluster_se": float(standard_errors[index]),
                "cluster_ci_low": float(ci_low[index]),
                "cluster_ci_high": float(ci_high[index]),
                "cluster_p_value": float(p_values[index]),
                "bootstrap_ci_low": (
                    float(np.quantile(bootstrap, 0.025)) if bootstrap else None
                ),
                "bootstrap_ci_high": (
                    float(np.quantile(bootstrap, 0.975)) if bootstrap else None
                ),
                "trajectory_count": len(labelled),
                "game_count": len(games),
                "model_version_count": len(versions),
            }
        )

    grouped: Dict[Tuple[str, int], List[Sequence[Mapping[str, Any]]]] = defaultdict(list)
    for _, items in labelled:
        game = str(items[0].get("game_id") or items[0].get("task_id", ""))
        version = int(items[0].get("student_model_version", training_step(items[0])))
        grouped[(game, version)].append(items)
    contrasts: List[Dict[str, Any]] = []
    for (game, version), items_group in sorted(grouped.items()):
        failures = [items for items in items_group if not truthy(items[0].get("task_success"))]
        successes = [items for items in items_group if truthy(items[0].get("task_success"))]
        if not failures or not successes:
            continue

        def metric(items: Sequence[Mapping[str, Any]], kind: str) -> Optional[float]:
            values = [float(row["teacher_entropy_topk"]) for row in items]
            if kind == "slope":
                return linear_slope(values)
            if kind == "delta":
                return values[-1] - values[0]
            if len(values) < 10:
                return None
            return float(np.mean(values[5:10]) - np.mean(values[:5]))

        record: Dict[str, Any] = {
            "game_id": game,
            "student_model_version": version,
            "failure_count": len(failures),
            "success_count": len(successes),
        }
        for kind, field in (
            ("slope", "failure_minus_success_slope"),
            ("delta", "failure_minus_success_delta"),
            ("early", "failure_minus_success_early_delta"),
        ):
            failure_values = [value for items in failures if (value := metric(items, kind)) is not None]
            success_values = [value for items in successes if (value := metric(items, kind)) is not None]
            record[field] = (
                float(np.mean(failure_values) - np.mean(success_values))
                if failure_values and success_values
                else None
            )
        contrasts.append(record)
    return effects, contrasts


def save_fixed_panel_same_task_effects(
    effects: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    output_dir: Path,
    status: str,
) -> Optional[Path]:
    if not effects:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9))
    for axis in axes:
        style_axis(axis)
    term_labels = {
        "progress": "Progress (success)",
        "failure": "Failure level at start",
        "failure_x_progress": "Failure × progress",
    }
    positions = np.arange(len(effects))
    estimates = np.asarray([float(row["estimate"]) for row in effects])
    lows = np.asarray([
        float(row.get("bootstrap_ci_low") if row.get("bootstrap_ci_low") is not None else row["cluster_ci_low"])
        for row in effects
    ])
    highs = np.asarray([
        float(row.get("bootstrap_ci_high") if row.get("bootstrap_ci_high") is not None else row["cluster_ci_high"])
        for row in effects
    ])
    axes[0].errorbar(
        estimates,
        positions,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        color=TEACHER,
        ecolor=TEACHER,
        capsize=4,
        linewidth=1.8,
    )
    axes[0].axvline(0.0, color=NEUTRAL_LIGHT, linestyle=":", linewidth=1.2)
    axes[0].set_yticks(positions, [term_labels[str(row["term"])] for row in effects])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Teacher entropy coefficient (nats)")
    axes[0].set_title("Task/checkpoint-controlled progress model", loc="left", fontsize=11, color=INK)

    slope_differences = [
        float(row["failure_minus_success_slope"])
        for row in contrasts
        if finite(row.get("failure_minus_success_slope")) is not None
    ]
    axes[1].axvline(0.0, color=NEUTRAL_LIGHT, linestyle=":", linewidth=1.2)
    if slope_differences:
        axes[1].hist(
            slope_differences,
            bins=min(14, max(6, int(math.sqrt(len(slope_differences))))),
            color=TEACHER,
            edgecolor="#FFFFFF",
            alpha=0.72,
        )
        axes[1].axvline(np.median(slope_differences), color=INK, linewidth=1.8, label=f"Median {np.median(slope_differences):+.4f}")
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Failure minus success entropy slope (nats/turn)")
    axes[1].set_ylabel("Task × checkpoint cells")
    axes[1].set_title("Within-cell stochastic-rollout contrasts", loc="left", fontsize=11, color=INK)

    first = effects[0]
    figure_header(
        fig,
        "Fixed-panel same-task Teacher-entropy outcome effects",
        f"{status}; game and Student-version fixed effects; intervals are task-cluster bootstrap 95% CIs; n={int(first['trajectory_count']):,} trajectories, {int(first['game_count'])} games; right panel uses cells containing both outcomes.",
    )
    fig.subplots_adjust(top=0.82, left=0.16, right=0.98, bottom=0.13, wspace=0.32)
    path = output_dir / "fixed_panel_same_task_effects.png"
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _ridge_logit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    penalty: float = 0.08,
) -> np.ndarray:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train_x), train_x, medians)
    test = np.where(np.isfinite(test_x), test_x, medians)
    centers = np.mean(train, axis=0)
    scales = np.std(train, axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    train = (train - centers) / scales
    test = (test - centers) / scales
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    coefficients = np.zeros(train.shape[1], dtype=float)
    penalty_mask = np.ones(train.shape[1], dtype=float)
    penalty_mask[0] = 0.0
    for _ in range(50):
        probabilities = _sigmoid(train @ coefficients)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-5)
        gradient = train.T @ (probabilities - train_y) / len(train)
        gradient += penalty * penalty_mask * coefficients
        hessian = (train.T * weights) @ train / len(train)
        hessian += penalty * np.diag(penalty_mask)
        update = np.linalg.solve(hessian + np.eye(len(coefficients)) * 1e-8, gradient)
        coefficients -= update
        if float(np.max(np.abs(update))) < 1e-7:
            break
    return _sigmoid(test @ coefficients)


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    auc: Optional[float] = None
    average_precision: Optional[float] = None
    if positives and negatives:
        ranks = stats.rankdata(scores, method="average")
        auc = float((np.sum(ranks[labels == 1]) - positives * (positives + 1) / 2) / (positives * negatives))
    if positives:
        order = np.argsort(-scores, kind="mergesort")
        ordered = labels[order]
        precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
        average_precision = float(np.sum(precision * ordered) / positives)
    return auc, average_precision, float(np.mean((scores - labels) ** 2))


def _group_bootstrap_metric_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    replicates: int = 500,
) -> Dict[str, Optional[float]]:
    unique_groups = sorted(set(groups))
    group_array = np.asarray(groups)
    rng = np.random.default_rng(20260806)
    aucs: List[float] = []
    aps: List[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(group_array == group) for group in sampled])
        auc, average_precision, _ = _binary_metrics(labels[indices], scores[indices])
        if auc is not None:
            aucs.append(auc)
        if average_precision is not None:
            aps.append(average_precision)
    return {
        "auroc_ci_low": float(np.quantile(aucs, 0.025)) if aucs else None,
        "auroc_ci_high": float(np.quantile(aucs, 0.975)) if aucs else None,
        "average_precision_ci_low": float(np.quantile(aps, 0.025)) if aps else None,
        "average_precision_ci_high": float(np.quantile(aps, 0.975)) if aps else None,
    }


def analyze_fixed_panel_failure_prediction(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Mapping[str, Any]]],
    cutoffs: Sequence[int] = (5, 10, 15),
) -> List[Dict[str, Any]]:
    """Grouped-CV landmark prediction using only history available by each cutoff."""

    feature_sets = {
        "Checkpoint": ["version"],
        "Context/action": ["version", "prompt_latest", "observation_mean", "invalid_rate", "repeat_rate"],
        "Teacher level": ["version", "teacher_latest", "teacher_mean"],
        "Teacher dynamics": ["version", "teacher_latest", "teacher_mean", "teacher_slope", "teacher_delta", "teacher_excess_area"],
        "Reverse KL": ["version", "kl_latest", "kl_mean", "kl_slope"],
        "Combined history": [
            "version", "prompt_latest", "observation_mean", "invalid_rate", "repeat_rate",
            "teacher_latest", "teacher_mean", "teacher_slope", "teacher_delta", "teacher_excess_area",
            "kl_latest", "kl_mean", "kl_slope",
        ],
    }
    output: List[Dict[str, Any]] = []
    all_items = list(trajectories.values())
    for cutoff in cutoffs:
        eligible = [items for items in all_items if len(items) >= cutoff]
        excluded_early_success = sum(
            truthy(items[0].get("task_success")) and len(items) < cutoff
            for items in all_items
        )
        if not eligible:
            continue
        records: List[Dict[str, float]] = []
        labels: List[int] = []
        groups: List[str] = []
        for items in eligible:
            history = items[:cutoff]
            teacher = [float(row["teacher_entropy_topk"]) for row in history]
            kl = [
                finite(row.get("sampled_reverse_kl_mean", row.get("sampled_forward_kl_mean")))
                for row in history
            ]
            kl_values = [value if value is not None else math.nan for value in kl]
            prompt = [finite(row.get("prompt_tokens")) for row in history]
            observation = [finite(row.get("observation_words")) for row in history]
            record = {
                "version": float(history[0].get("student_model_version", training_step(history[0]))),
                "prompt_latest": prompt[-1] if prompt[-1] is not None else math.nan,
                "observation_mean": mean(observation) or 0.0,
                "invalid_rate": float(np.mean([not truthy(row.get("action_valid")) for row in history])),
                "repeat_rate": float(np.mean([(finite(row.get("consecutive_action_repeat_count")) or 0.0) > 0 for row in history])),
                "teacher_latest": teacher[-1],
                "teacher_mean": float(np.mean(teacher)),
                "teacher_slope": linear_slope(teacher) or 0.0,
                "teacher_delta": teacher[-1] - float(np.mean(teacher[: min(3, len(teacher))])),
                "teacher_excess_area": float(np.sum(np.asarray(teacher) - np.mean(teacher[: min(3, len(teacher))]))),
                "kl_latest": kl_values[-1],
                "kl_mean": float(np.nanmean(kl_values)) if np.any(np.isfinite(kl_values)) else math.nan,
                "kl_slope": linear_slope([value for value in kl_values if math.isfinite(value)]) or 0.0,
            }
            records.append(record)
            labels.append(0 if truthy(history[0].get("task_success")) else 1)
            groups.append(str(history[0].get("game_id") or history[0].get("task_id", "")))
        label_array = np.asarray(labels, dtype=int)
        group_array = np.asarray(groups)
        for model_name, fields in feature_sets.items():
            features = np.asarray([[record[field] for field in fields] for record in records], dtype=float)
            predictions = np.full(len(features), np.nan, dtype=float)
            for game in sorted(set(groups)):
                test = group_array == game
                train = ~test
                if len(np.unique(label_array[train])) < 2:
                    predictions[test] = float(np.mean(label_array[train]))
                else:
                    predictions[test] = _ridge_logit_predict(
                        features[train], label_array[train], features[test]
                    )
            auc, average_precision, brier = _binary_metrics(label_array, predictions)
            intervals = _group_bootstrap_metric_intervals(
                label_array, predictions, groups
            )
            output.append(
                {
                    "cutoff_turns": cutoff,
                    "model": model_name,
                    "trajectory_count": len(eligible),
                    "game_count": len(set(groups)),
                    "failure_count": int(np.sum(label_array)),
                    "failure_rate": float(np.mean(label_array)),
                    "excluded_early_success_count": excluded_early_success,
                    "auroc": auc,
                    "average_precision": average_precision,
                    "brier_score": brier,
                    **intervals,
                }
            )
    return output


def save_fixed_panel_failure_prediction(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, status: str
) -> Optional[Path]:
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9), sharex=True)
    for axis in axes:
        style_axis(axis)
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    colors = ["#374151", "#6B7280", "#A16207", "#D9833F", "#2F6B9A", "#163E63"]
    linestyles = [":", "--", "-.", "--", "-.", "-"]
    markers = ["o", "s", "^", "D", "v", "P"]
    for model, color, linestyle, marker in zip(models, colors, linestyles, markers):
        selected = sorted([row for row in rows if row["model"] == model], key=lambda row: int(row["cutoff_turns"]))
        x = np.asarray([int(row["cutoff_turns"]) for row in selected])
        for axis, field, low_field, high_field in (
            (axes[0], "auroc", "auroc_ci_low", "auroc_ci_high"),
            (axes[1], "average_precision", "average_precision_ci_low", "average_precision_ci_high"),
        ):
            y = np.asarray([float(row[field]) for row in selected])
            low = np.asarray([float(row[low_field]) for row in selected])
            high = np.asarray([float(row[high_field]) for row in selected])
            axis.errorbar(
                x, y, yerr=np.vstack([y - low, high - y]), color=color,
                linestyle=linestyle, marker=marker, linewidth=1.8, capsize=3,
                label=model,
            )
    axes[0].axhline(0.5, color=NEUTRAL_LIGHT, linestyle=":", linewidth=1.0)
    prevalence = sorted(
        {(int(row["cutoff_turns"]), float(row["failure_rate"])) for row in rows}
    )
    axes[1].plot(
        [item[0] for item in prevalence], [item[1] for item in prevalence],
        color=NEUTRAL_LIGHT, linestyle=":", linewidth=1.4, label="Failure prevalence",
    )
    for axis, title, ylabel in (
        (axes[0], "Grouped-CV AUROC", "AUROC"),
        (axes[1], "Grouped-CV average precision", "Average precision"),
    ):
        axis.set_xticks(sorted({int(row["cutoff_turns"]) for row in rows}))
        axis.set_xlabel("Observed turns at landmark")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.02)
        axis.set_title(title, loc="left", fontsize=11, color=INK)
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    figure_header(
        fig,
        "Fixed-panel final-failure prediction from online history",
        f"{status}; leave-one-game-out CV; task-cluster bootstrap 95% CIs; risk set = trajectories active at each landmark; predicts outcome, not data value.",
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.13, wspace=0.24)
    path = output_dir / "fixed_panel_failure_prediction.png"
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def build_boundary_event_records(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for items in trajectories.values():
        for previous, current in zip(items, items[1:]):
            previous_blocks = [value for raw in previous.get("teacher_entropy_topk_blocks", []) if (value := finite(raw)) is not None]
            current_blocks = [value for raw in current.get("teacher_entropy_topk_blocks", []) if (value := finite(raw)) is not None]
            if not previous_blocks or not current_blocks:
                continue
            repeated = (finite(previous.get("consecutive_action_repeat_count")) or 0.0) > 0
            if not truthy(previous.get("action_valid")):
                action_category = "Invalid"
            elif repeated:
                action_category = "Repeated"
            else:
                action_category = "Valid non-repeat"
            raw_jump = current_blocks[0] - previous_blocks[-1]
            same_position = current_blocks[0] - previous_blocks[0]
            output.append(
                {
                    "diagnostics_step": training_step(current),
                    "student_model_version": current.get("student_model_version", training_step(current)),
                    "game_id": current.get("game_id") or current.get("task_id", ""),
                    "run_id": current.get("run_id", ""),
                    "turn": int(current["turn"]),
                    "task_success": truthy(current.get("task_success")),
                    "action_category": action_category,
                    "action_verb": previous.get("action_verb", ""),
                    "observation_words": finite(current.get("observation_words")),
                    "raw_boundary_jump": raw_jump,
                    "same_position_change": same_position,
                    "position_reset_contrast": previous_blocks[0] - previous_blocks[-1],
                }
            )
    return output


def _mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    error = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return float(np.mean(array)), error


def _cluster_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    replicates: int = 300,
) -> Tuple[float, float]:
    """Return a mean and task-cluster bootstrap half-width."""

    estimate = float(np.mean([float(row[field]) for row in rows]))
    games = sorted({str(row.get("game_id", "")) for row in rows})
    if len(games) < 2:
        return estimate, 0.0
    by_game = {
        game: [float(row[field]) for row in rows if str(row.get("game_id", "")) == game]
        for game in games
    }
    rng = np.random.default_rng(20260806)
    bootstrap = []
    for _ in range(replicates):
        sampled = rng.choice(games, size=len(games), replace=True)
        values = [value for game in sampled for value in by_game[str(game)]]
        bootstrap.append(float(np.mean(values)))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return estimate, max(estimate - float(low), float(high) - estimate)


def save_fixed_panel_boundary_deconfounded(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, status: str
) -> Optional[Path]:
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.9))
    for axis in axes:
        style_axis(axis)
        axis.axhline(0.0, color=NEUTRAL_LIGHT, linestyle=":", linewidth=1.0)

    positions = np.arange(2)
    width = 0.34
    for offset, success, label, color in (
        (-width / 2, False, "Failure", TEACHER),
        (width / 2, True, "Success", SUCCESS),
    ):
        selected = [row for row in rows if truthy(row.get("task_success")) == success]
        summaries = [
            _cluster_mean_ci(selected, field)
            for field in ("raw_boundary_jump", "same_position_change")
        ]
        axes[0].bar(positions + offset, [item[0] for item in summaries], width=width, color=color, alpha=0.42, edgecolor=color, label=label)
        axes[0].errorbar(positions + offset, [item[0] for item in summaries], yerr=[item[1] for item in summaries], fmt="none", color=color, capsize=3)
    axes[0].set_xticks(positions, ["Raw: new first −\nprevious last", "Position-controlled:\nnew first − previous first"])
    axes[0].set_ylabel("Teacher entropy change (nats)")
    axes[0].set_title("Token-position reset control", loc="left", fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=8)

    categories = ["Valid non-repeat", "Repeated", "Invalid"]
    for success, label, color, marker in (
        (False, "Failure", TEACHER, "o"), (True, "Success", SUCCESS, "s")
    ):
        means = []
        errors = []
        for category in categories:
            selected = [row for row in rows if row["action_category"] == category and truthy(row.get("task_success")) == success]
            value, error = _cluster_mean_ci(selected, "same_position_change") if selected else (math.nan, 0.0)
            means.append(value)
            errors.append(error)
        shift = -0.08 if not success else 0.08
        axes[1].errorbar(np.arange(len(categories)) + shift, means, yerr=errors, fmt=marker, color=color, capsize=3, label=label)
    axes[1].set_xticks(range(len(categories)), categories, rotation=16, ha="right")
    axes[1].set_ylabel("Position-controlled change (nats)")
    axes[1].set_title("Previous action category", loc="left", fontsize=11, color=INK)
    axes[1].legend(frameon=False, fontsize=8)

    observation_values = np.asarray([float(row["observation_words"]) for row in rows if finite(row.get("observation_words")) is not None])
    edges = np.unique(np.quantile(observation_values, [0.0, 0.25, 0.5, 0.75, 1.0])) if len(observation_values) else np.asarray([])
    if len(edges) >= 3:
        centers = []
        labels = []
        for low, high in zip(edges[:-1], edges[1:]):
            centers.append((low + high) / 2)
            labels.append(f"{low:.0f}–{high:.0f}")
        for success, label, color, linestyle in (
            (False, "Failure", TEACHER, "-"), (True, "Success", SUCCESS, "--")
        ):
            means = []
            errors = []
            for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                selected = [row for row in rows if truthy(row.get("task_success")) == success and finite(row.get("observation_words")) is not None and float(row["observation_words"]) >= low and (float(row["observation_words"]) < high or index == len(edges) - 2)]
                value, error = _cluster_mean_ci(selected, "same_position_change") if selected else (math.nan, 0.0)
                means.append(value)
                errors.append(error)
            axes[2].errorbar(range(len(means)), means, yerr=errors, color=color, linestyle=linestyle, marker="o", capsize=3, label=label)
        axes[2].set_xticks(range(len(labels)), labels)
    axes[2].set_xlabel("New observation length (word quartile)")
    axes[2].set_ylabel("Position-controlled change (nats)")
    axes[2].set_title("Observation-length check", loc="left", fontsize=11, color=INK)
    axes[2].legend(frameon=False, fontsize=8)

    figure_header(
        fig,
        "Fixed-panel Teacher entropy around observation boundaries",
        f"{status}; 4-token blocks; controlled change = new first − previous first response block; task-cluster bootstrap 95% CIs.",
    )
    fig.subplots_adjust(top=0.80, left=0.07, right=0.99, bottom=0.18, wspace=0.28)
    path = output_dir / "fixed_panel_teacher_entropy_boundary_deconfounded.png"
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def _find_subsequence(values: Sequence[int], query: Sequence[int]) -> Optional[int]:
    if not query or len(query) > len(values):
        return None
    for index in range(len(values) - len(query) + 1):
        if list(values[index : index + len(query)]) == list(query):
            return index
    return None


def build_action_boundary_records(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Mapping[str, Any]]],
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for items in trajectories.values():
        for row in items:
            action = row.get("action")
            token_ids = row.get("response_token_ids") or []
            entropies = [value for raw in row.get("teacher_entropy_topk_blocks", []) if (value := finite(raw)) is not None]
            sizes = [int(value) for value in row.get("token_block_sizes", [])]
            if not action or not token_ids or not entropies or len(entropies) != len(sizes):
                continue
            marker_ids = tokenizer.encode(f"<action>{action}</action>", add_special_tokens=False)
            start = _find_subsequence(token_ids, marker_ids)
            if start is None:
                continue
            end = start + len(marker_ids)
            cumulative = np.cumsum([0] + sizes)
            start_block = int(np.searchsorted(cumulative[1:], start, side="right"))
            end_block = int(np.searchsorted(cumulative[1:], end - 1, side="right"))
            for boundary, anchor in (("action_start", start_block), ("action_end", end_block)):
                for relative in range(-2, 3):
                    index = anchor + relative
                    if index < 0 or index >= len(entropies):
                        continue
                    output.append(
                        {
                            "diagnostics_step": training_step(row),
                            "student_model_version": row.get("student_model_version", training_step(row)),
                            "game_id": row.get("game_id") or row.get("task_id", ""),
                            "run_id": row.get("run_id", ""),
                            "turn": int(row["turn"]),
                            "task_success": truthy(row.get("task_success")),
                            "boundary": boundary,
                            "relative_block": relative,
                            "teacher_entropy_topk": entropies[index],
                        }
                    )
    return output


def save_fixed_panel_action_boundary(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, status: str
) -> Optional[Path]:
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    for axis in axes:
        style_axis(axis)
        axis.axvline(0, color=NEUTRAL_LIGHT, linestyle=":", linewidth=1.1)
    for axis, boundary, title in zip(
        axes,
        ("action_start", "action_end"),
        ("Reasoning → <action>", "</action> → response end"),
    ):
        for success, label, color, linestyle in (
            (False, "Failure", TEACHER, "-"), (True, "Success", SUCCESS, "--")
        ):
            selected = [row for row in rows if row["boundary"] == boundary and truthy(row.get("task_success")) == success]
            offsets = sorted({int(row["relative_block"]) for row in selected})
            values = []
            errors = []
            for offset in offsets:
                group = [row for row in selected if int(row["relative_block"]) == offset]
                value, error = _cluster_mean_ci(group, "teacher_entropy_topk")
                values.append(value)
                errors.append(error)
            axis.errorbar(offsets, values, yerr=errors, color=color, linestyle=linestyle, marker="o", capsize=3, label=label)
        axis.set_xticks(range(-2, 3))
        axis.set_xlabel("4-token block offset from boundary")
        axis.set_title(title, loc="left", fontsize=11, color=INK)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Teacher top-k entropy (nats)")
    figure_header(
        fig,
        "Fixed-panel Teacher entropy around parsed action boundaries",
        f"{status}; exact token subsequence alignment to <action>…</action>; block 0 contains the boundary token; task-cluster bootstrap 95% intervals.",
    )
    fig.subplots_adjust(top=0.82, left=0.09, right=0.98, bottom=0.14, wspace=0.18)
    path = output_dir / "fixed_panel_teacher_entropy_action_boundary.png"
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def analyze_fixed_panel_frontier(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Sequence[Mapping[str, Any]]]] = defaultdict(list)
    for items in trajectories.values():
        version = int(items[0].get("student_model_version", training_step(items[0])))
        outcome = "success" if truthy(items[0].get("task_success")) else "failure"
        grouped[(version, outcome)].append(items)
    output: List[Dict[str, Any]] = []
    for (version, outcome), items_group in sorted(grouped.items()):
        for threshold in (0.10, 0.15, 0.20):
            crossings = [_first_crossing(_teacher_entropy_delta(items), threshold) for items in items_group]
            observed = [value for value in crossings if value is not None]
            output.append(
                {
                    "student_model_version": version,
                    "outcome": outcome,
                    "threshold": threshold,
                    "trajectory_count": len(items_group),
                    "crossed_count": len(observed),
                    "crossed_fraction": len(observed) / len(items_group),
                    "crossing_turn_q10": float(np.quantile(observed, 0.10)) if observed else None,
                    "crossing_turn_median": float(np.median(observed)) if observed else None,
                    "crossing_turn_q90": float(np.quantile(observed, 0.90)) if observed else None,
                }
            )
    return output


def save_fixed_panel_frontier_by_checkpoint(
    trajectories: Mapping[Tuple[int, str, str], Sequence[Mapping[str, Any]]],
    summary_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    status: str,
) -> Optional[Path]:
    if not trajectories or not summary_rows:
        return None
    versions = sorted({int(row["student_model_version"]) for row in summary_rows})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.0))
    for axis in axes.flat:
        style_axis(axis)
    version_colors = plt.cm.cividis(np.linspace(0.12, 0.88, len(versions)))
    for axis, outcome in zip(axes[0], ("failure", "success")):
        for version, color in zip(versions, version_colors):
            selected = [
                items for items in trajectories.values()
                if int(items[0].get("student_model_version", training_step(items[0]))) == version
                and ("success" if truthy(items[0].get("task_success")) else "failure") == outcome
            ]
            if not selected:
                continue
            crossings = [_first_crossing(_teacher_entropy_delta(items), 0.15) for items in selected]
            max_turn = max(len(items) for items in selected) - 1
            curve = [sum(value is not None and value <= turn for value in crossings) / len(selected) for turn in range(max_turn + 1)]
            axis.step(range(max_turn + 1), curve, where="post", color=color, linewidth=1.6, label=str(version))
        axis.set_xlabel("Trajectory turn (0-based)")
        axis.set_ylabel("Fraction crossed before termination")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"{outcome.title()}: observed ΔH ≥ 0.15", loc="left", fontsize=11, color=INK)
    axes[0, 1].legend(frameon=False, fontsize=7, title="Student version", ncol=2, loc="lower right")

    for axis, outcome in zip(axes[1], ("failure", "success")):
        for threshold, linestyle, marker in ((0.10, "-", "o"), (0.15, "--", "s"), (0.20, ":", "^")):
            selected = sorted([row for row in summary_rows if row["outcome"] == outcome and abs(float(row["threshold"]) - threshold) < 1e-9], key=lambda row: int(row["student_model_version"]))
            axis.plot([int(row["student_model_version"]) for row in selected], [float(row["crossed_fraction"]) for row in selected], color=TEACHER if outcome == "failure" else SUCCESS, linestyle=linestyle, marker=marker, linewidth=1.8, label=f"ΔH ≥ {threshold:.2f}")
        axis.set_xlabel("Student model version")
        axis.set_ylabel("Fraction crossing before termination")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"{outcome.title()}: terminal observed crossing rate", loc="left", fontsize=11, color=INK)
        axis.legend(frameon=False, fontsize=8)
    figure_header(
        fig,
        "Fixed-panel Teacher-entropy frontier across Student checkpoints",
        f"{status}; same 16 tasks × 8 stochastic rollouts at each panel step; curves are observed cumulative incidence, not Kaplan–Meier—termination without crossing remains non-crossing.",
    )
    fig.subplots_adjust(top=0.88, left=0.08, right=0.98, bottom=0.08, hspace=0.30, wspace=0.22)
    path = output_dir / "fixed_panel_frontier_by_checkpoint.png"
    fig.savefig(path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_kl_surprisal(rows: Sequence[Mapping[str, Any]], output_dir: Path, status: str) -> Optional[Path]:
    if not any(finite(row.get("sampled_reverse_kl_token_weighted")) is not None for row in rows):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=False)
    for axis in axes:
        style_axis(axis)
    add_series(axes[0], rows, "sampled_reverse_kl_token_weighted", "Sampled reverse KL / token", NEUTRAL)
    add_series(axes[0], rows, "sampled_reverse_kl_mean", "Unweighted turn mean", NEUTRAL_LIGHT, "--")
    axes[0].set_xlabel("Student model version")
    axes[0].set_ylabel("KL-like log-prob gap (nats/token)")
    axes[0].set_title("Student–teacher sampled reverse KL", loc="left", fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=8)

    add_series(axes[1], rows, "student_surprisal_mean", "Student", STUDENT)
    add_series(axes[1], rows, "teacher_surprisal_mean", "Teacher", TEACHER)
    axes[1].set_xlabel("Student model version")
    axes[1].set_ylabel("Surprisal (nats/token)")
    axes[1].set_title("Surprisal on sampled response tokens", loc="left", fontsize=11, color=INK)
    axes[1].legend(frameon=False, fontsize=8)
    figure_header(fig, "Sampled reverse KL and surprisal during Vanilla OPD", f"{status}; reverse KL is selected-token log p(student) − log p(teacher) on student-sampled tokens, not full-vocabulary KL; trailing mean window = 5.")
    fig.subplots_adjust(top=0.80, left=0.07, right=0.98, bottom=0.15, wspace=0.25)
    path = output_dir / "kl_surprisal_curve.png"
    fig.savefig(path, dpi=160, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_reverse_kl_loss(
    rows: Sequence[Mapping[str, Any]],
    trainer_records: Mapping[int, Dict[str, Any]],
    output_dir: Path,
    status: str,
) -> Optional[Path]:
    """Plot the sampled reverse-KL objective and the trainer loss.

    The workflow's historical ``sampled_forward_kl_*`` JSONL field is
    ``log p_student(token) - log p_teacher(token)`` on student-sampled
    response tokens.  Its expectation is the reverse KL
    D_KL(student || teacher), so the plot uses the correctly named derived
    aliases.  The Trainer's actor loss is shown separately because this
    Vanilla OPD configuration optimizes a PPO surrogate for that objective;
    it is not itself a full-distribution KL value.
    """

    has_reverse_kl = any(
        finite(row.get("sampled_reverse_kl_token_weighted")) is not None for row in rows
    )
    has_trainer_loss = any(
        finite(record.get("actor_final_loss")) is not None
        or finite(record.get("actor_pg_loss")) is not None
        for record in trainer_records.values()
    )
    if not has_reverse_kl and not has_trainer_loss:
        return None

    panel_count = int(has_reverse_kl) + int(has_trainer_loss)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(11, 4.4 * panel_count),
        squeeze=False,
        constrained_layout=False,
    )
    axes_flat = list(axes[:, 0])
    panel_index = 0
    for axis in axes_flat:
        style_axis(axis)

    if has_reverse_kl:
        axis = axes_flat[panel_index]
        panel_index += 1
        add_series(
            axis,
            rows,
            "sampled_reverse_kl_token_weighted",
            "Token-weighted sampled reverse KL",
            NEUTRAL,
        )
        add_series(
            axis,
            rows,
            "sampled_reverse_kl_mean",
            "Unweighted turn mean",
            NEUTRAL_LIGHT,
            "--",
        )
        axis.set_xlabel("Student model version")
        axis.set_ylabel("Reverse KL estimate (nats/token)")
        axis.set_title("Sampled reverse-KL loss", loc="left", fontsize=11, color=INK)
        axis.legend(frameon=False, fontsize=8)

    if has_trainer_loss:
        axis = axes_flat[panel_index]
        trainer_rows = [trainer_records[step] for step in sorted(trainer_records)]
        x = np.asarray([row["trainer_step"] for row in trainer_rows], dtype=float)
        for field, label, color, style in (
            ("actor_final_loss", "actor/final_loss", STUDENT, "-"),
            ("actor_pg_loss", "actor/pg_loss", TEACHER, "--"),
        ):
            y = np.asarray(
                [
                    finite(row.get(field)) if finite(row.get(field)) is not None else np.nan
                    for row in trainer_rows
                ],
                dtype=float,
            )
            valid = np.isfinite(y)
            if valid.any():
                axis.plot(
                    x[valid],
                    y[valid],
                    color=color,
                    alpha=0.18,
                    linewidth=0.8,
                    marker=".",
                    markersize=3,
                )
                axis.plot(
                    x[valid],
                    rolling(y[valid]),
                    color=color,
                    linewidth=2.0,
                    linestyle=style,
                    label=label,
                )
        axis.set_xlabel("Trainer global step")
        axis.set_ylabel("Loss")
        axis.set_title("Trainer actor loss (PPO surrogate)", loc="left", fontsize=11, color=INK)
        axis.legend(frameon=False, fontsize=8)

    figure_header(
        fig,
        "Reverse-KL objective and training loss during Vanilla OPD",
        f"{status}; direct reverse-KL estimate is plotted separately from the Trainer's PPO surrogate; raw points are faint, solid lines are 5-step trailing means.",
    )
    fig.subplots_adjust(top=0.84, left=0.09, right=0.98, bottom=0.10, hspace=0.34)
    path = output_dir / "reverse_kl_loss_curve.png"
    fig.savefig(path, dpi=160, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_mass_success(rows: Sequence[Mapping[str, Any]], output_dir: Path, status: str, top_k: Any) -> Optional[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.0), sharex=True, constrained_layout=False)
    for axis in axes:
        style_axis(axis)
    add_series(axes[0], rows, "student_topk_mass_mean", "Student", STUDENT)
    add_series(axes[0], rows, "teacher_topk_mass_mean", "Teacher", TEACHER)
    mass_values = [finite(row.get(field)) for row in rows for field in ("student_topk_mass_mean", "teacher_topk_mass_mean")]
    mass_values = [value for value in mass_values if value is not None]
    if mass_values:
        lower = max(0.0, min(mass_values) - max(0.0001, (max(mass_values) - min(mass_values)) * 0.25))
        upper = min(1.001, max(mass_values) + max(0.00005, (max(mass_values) - min(mass_values)) * 0.25))
        if upper - lower < 0.0002:
            lower, upper = max(0.0, max(mass_values) - 0.0002), min(1.001, max(mass_values) + 0.00005)
        axes[0].set_ylim(lower, upper)
    axes[0].axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("Mass in returned head")
    axes[0].set_title(f"Top-{top_k or 16} head probability mass", loc="left", fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=8)

    add_series(axes[1], rows, "success_rate", "Training rollout success", SUCCESS)
    eval_fields = sorted({key for row in rows for key in row if key.startswith("eval_") and key.endswith("_success_rate")})
    eval_colors = [NEUTRAL, TEACHER, STUDENT, NEUTRAL_LIGHT]
    for index, field in enumerate(eval_fields):
        add_series(axes[1], rows, field, field.removeprefix("eval_").removesuffix("_success_rate"), eval_colors[index % len(eval_colors)], "--")
    axes[1].set_xlabel("Student model version")
    axes[1].set_ylabel("Task success rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Success rate", loc="left", fontsize=11, color=INK)
    axes[1].legend(frameon=False, fontsize=8)
    figure_header(fig, "Top-k mass and task success during Vanilla OPD", f"{status}; success is trajectory-level and deduplicated by task/run; mass uses a focused y-axis because top-{top_k or 16} coverage is near 1.")
    fig.subplots_adjust(top=0.83, left=0.09, right=0.98, bottom=0.10, hspace=0.28)
    path = output_dir / "mass_success_curve.png"
    fig.savefig(path, dpi=160, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def save_timing(
    rows: Sequence[Mapping[str, Any]],
    explorer_records: Mapping[int, Dict[str, Any]],
    trainer_records: Mapping[int, Dict[str, Any]],
    output_dir: Path,
    status: str,
) -> Optional[Path]:
    plotted_steps = {int(row["diagnostics_step"]) for row in rows}
    explorer_rows = [
        explorer_records[step]
        for step in sorted(explorer_records)
        if not plotted_steps or step in plotted_steps
    ]
    if not explorer_rows and not trainer_records:
        return None
    panel_count = int(bool(explorer_rows)) + int(bool(trainer_records))
    fig, axes = plt.subplots(panel_count, 1, figsize=(11, 4.2 * panel_count), squeeze=False, constrained_layout=False)
    axes_flat = list(axes[:, 0])
    for axis in axes_flat:
        style_axis(axis)
    panel_index = 0
    if explorer_rows:
        axis = axes_flat[panel_index]
        panel_index += 1
        # Explorer records use explorer_step as a fallback when no model
        # version has been logged yet.
        x = np.asarray([finite(row.get("model_version")) if finite(row.get("model_version")) is not None else row["explorer_step"] for row in explorer_rows], dtype=float)
        for field, label, color, style in (
            ("rollout_wait_explore_step_sec", "Explorer step wait", NEUTRAL, "-"),
            ("rollout_run_execution_mean_sec", "Rollout run time / task", STUDENT, "-"),
            ("rollout_task_execution_mean_sec", "Rollout task time", TEACHER, "--"),
        ):
            y = np.asarray([finite(row.get(field)) if finite(row.get(field)) is not None else np.nan for row in explorer_rows], dtype=float)
            valid = np.isfinite(y)
            if valid.any():
                axis.plot(x[valid], y[valid], color=color, alpha=0.18, linewidth=0.8, marker=".", markersize=3)
                axis.plot(x[valid], rolling(y[valid]), color=color, linewidth=2.0, linestyle=style, label=label)
        axis.set_xlabel("Student model version")
        axis.set_ylabel("Seconds")
        axis.set_title("Explorer rollout and step wall time", loc="left", fontsize=11, color=INK)
        axis.legend(frameon=False, fontsize=8)
    if trainer_records:
        axis = axes_flat[panel_index]
        trainer_rows = [trainer_records[step] for step in sorted(trainer_records)]
        x = np.asarray([row["trainer_step"] for row in trainer_rows], dtype=float)
        for field, label, color, style in (
            ("time_read_experience_sec", "Read experience", NEUTRAL, "-"),
            ("time_train_step_sec", "Train step", STUDENT, "-"),
            ("time_step_sec", "Total trainer step", TEACHER, "--"),
        ):
            y = np.asarray([finite(row.get(field)) if finite(row.get(field)) is not None else np.nan for row in trainer_rows], dtype=float)
            valid = np.isfinite(y)
            if valid.any():
                axis.plot(x[valid], y[valid], color=color, alpha=0.18, linewidth=0.8, marker=".", markersize=3)
                axis.plot(x[valid], rolling(y[valid]), color=color, linewidth=2.0, linestyle=style, label=label)
        axis.set_xlabel("Trainer global step")
        axis.set_ylabel("Seconds")
        axis.set_title("Trainer timing", loc="left", fontsize=11, color=INK)
        axis.legend(frameon=False, fontsize=8)
    figure_header(fig, "Rollout and trainer timing during Vanilla OPD", f"{status}; Explorer wait includes the collection-step wall time and may include monitor/evaluation waiting; faint points are raw, lines are trailing means.")
    fig.subplots_adjust(top=0.84, left=0.09, right=0.98, bottom=0.12, hspace=0.32)
    path = output_dir / "timing_curve.png"
    fig.savefig(path, dpi=160, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def completion_status(args: argparse.Namespace, max_step: Optional[int]) -> str:
    if args.status in {"preliminary", "complete"}:
        return args.status
    if args.checkpoint_job_dir and args.final_trainer_step:
        checkpoint = args.checkpoint_job_dir / f"global_step_{args.final_trainer_step}" / "actor" / "huggingface" / "model.safetensors"
        if checkpoint.exists():
            return "complete"
    return "preliminary"


def write_notes(
    output_dir: Path,
    status: str,
    metadata: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    plotted_diagnostics: Sequence[Mapping[str, Any]],
    turn_diagnostics: Sequence[Mapping[str, Any]],
    trajectory_summaries: Sequence[Mapping[str, Any]],
    explorer_records: Mapping[int, Dict[str, Any]],
    trainer_records: Mapping[int, Dict[str, Any]],
) -> None:
    steps = [int(row["diagnostics_step"]) for row in diagnostics]
    plotted_steps = [int(row["diagnostics_step"]) for row in plotted_diagnostics]
    partial_steps = [step for step in steps if step not in set(plotted_steps)]
    model_versions = [int(row["model_version"]) for row in diagnostics if row.get("model_version") is not None]
    top_k = next((row.get("diagnostics_top_k") for row in diagnostics if row.get("diagnostics_top_k") is not None), 16)
    notes = f"""# Vanilla OPD diagnostics snapshot

Status: **{status}**

Diagnostics steps: {min(steps) if steps else 'n/a'}–{max(steps) if steps else 'n/a'}

Student model-version range: {min(model_versions) if model_versions else 'n/a'}–{max(model_versions) if model_versions else 'n/a'}

Top-k head: {top_k}
Plotted complete steps: {len(plotted_diagnostics)}; partial steps excluded from plots: {partial_steps or 'none'}
Trajectory-turn view: {len(trajectory_summaries)} trajectories; turns 0–{max((int(row['turn']) for row in turn_diagnostics), default='n/a')}

## Metric contract

- `student_entropy_topk_mean` and `teacher_entropy_topk_mean` are the mean entropy of the returned top-{top_k} logprob head. They are not full-vocabulary entropy.
- `entropy_curve.png` is the canonical hypothesis-facing chart: its horizontal axis is the **within-trajectory environment turn**, not training step. The denominator panel makes late-turn survivorship visible.
- `entropy_by_model_version.png` is the separate training-evolution chart. Each point first collapses all turns and trajectories in one Explorer batch, then the displayed line applies a 5-step trailing mean.
- `teacher_entropy_by_outcome_progress.png` linearly interpolates each trajectory onto a shared 0–1 progress grid and equal-weights trajectories. This controls the raw-turn chart's length/survivorship effect and splits by final audited `task_success`.
- `teacher_entropy_frontier_heatmap.png` keeps every rollout separate, places failures above successes, subtracts each rollout's own early baseline, and sorts within outcome by first sustained threshold crossing.
- `teacher_entropy_threshold_crossing_{{all,failure,success}}.png` show when pooled and outcome-specific rollouts cross several online-compatible ΔH thresholds; termination before crossing is treated as right censoring.
- `teacher_entropy_observation_boundary.png` is the original raw schema-v2 alignment. Its previous-last/new-first contrast mixes observation effects with response-token position and should not be interpreted alone.
- `fixed_panel_same_task_variability.png` uses repeated stochastic rollouts of the same fixed task and checkpoint; fixed-panel rows are evaluation-only and never enter the training buffer.
- `fixed_panel_same_task_effects.png` fits an equal-trajectory-weighted progress model with game and Student-version fixed effects and task-cluster uncertainty; `failure × progress` tests whether failed rollouts drift faster after conditioning on the repeated task/checkpoint panel.
- `fixed_panel_failure_prediction.png` reports leave-one-game-out turn-5/10/15 landmark prediction. Only trajectories still active at each cutoff enter that risk set; final-failure predictiveness does not imply that the trajectory has no distillation value.
- `fixed_panel_teacher_entropy_boundary_deconfounded.png` compares the raw previous-last/new-first jump with a first-block-to-first-block position control, then stratifies the controlled change by previous action category and new-observation length.
- `fixed_panel_teacher_entropy_action_boundary.png` uses exact token alignment to the parsed `<action>…</action>` span when `--tokenizer-path` is supplied.
- `fixed_panel_frontier_by_checkpoint.png` tracks the same fixed tasks across checkpoints using observed cumulative crossing incidence. Unlike the pooled Kaplan–Meier-style training plots, termination without crossing remains non-crossing.
- `trajectory_summary.csv` contains paired within-trajectory Teacher entropy deltas and slopes. Last-5 minus first-5 is only defined for trajectories with at least 10 turns.
- `sampled_reverse_kl_token_weighted` is the response-token-weighted mean of `log p_student(token) - log p_teacher(token)` on student-sampled tokens. Its expectation is the sampled reverse KL `D_KL(student || teacher)`, but it is not full-distribution KL. Schema-v2 writes the correct reverse-KL name and retains the historical forward-KL alias for compatibility.
- `reverse_kl_loss_curve.png` separates this direct sampled reverse-KL estimate from `actor/final_loss`, which is the Trainer's PPO surrogate loss.
- Surprisal is `-log p(token)` for the sampled response token.
- Top-k mass is the sum of probabilities represented by the returned top-k head. Its near-one value is a coverage diagnostic, not a quality score.
- Success rate is trajectory-level: rows are deduplicated by `(diagnostics_step, task_id, run_id)` and the trajectory is successful if any row reports the audited `task_success` flag.
- `training_step` in the JSONL is the Explorer `batch_id`. When `explorer.log` is available, model-version charts add the corresponding student `model_version`; trajectory-turn charts never use it as their horizontal axis.
- `rollout_wait_explore_step_sec` is Explorer collection-step wall time. It can include waiting for the step and monitor/evaluation work; it is not a pure single-environment execution time.

## Coverage and source handling

The diagnostics JSONL files are read in the order supplied on the command line. If multiple files contain the same `(diagnostics_source, training_step)`, the last file owns that whole source-step. Training and fixed-panel rows at the same step therefore cannot overwrite one another. Malformed final lines are ignored and counted in `summary.json`, which makes the script safe to run while training is still appending JSONL.

Before trajectory charts are drawn, the script fails fast on mixed top-k definitions, unexpected diagnostics kinds, duplicate/missing turns, missing entropy values, or inconsistent final outcomes. The tables contain extra guardrail metrics including valid-action rate, timeout/lost rate, mean environment rounds, response-token volume, entropy/surprisal gaps, and trainer timing when the corresponding logs are supplied.
"""
    (output_dir / "analysis_notes.md").write_text(notes, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostics", nargs="+", type=Path, required=True, help="JSONL files, in chronological/source precedence order.")
    parser.add_argument(
        "--explorer-log",
        nargs="+",
        type=Path,
        default=None,
        help="Optional Explorer logs in chronological/source-precedence order.",
    )
    parser.add_argument(
        "--trainer-log",
        nargs="+",
        type=Path,
        default=None,
        help="Optional Trainer logs in chronological/source-precedence order.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; defaults to <first JSONL parent>/analysis.")
    parser.add_argument("--checkpoint-job-dir", type=Path, default=None, help="Optional checkpoint job directory for automatic completion detection.")
    parser.add_argument("--final-trainer-step", type=int, default=250, help="Expected final trainer step for completion detection (default: 250).")
    parser.add_argument(
        "--expected-trajectories",
        type=int,
        default=16,
        help="Expected trajectories per diagnostics step; incomplete steps stay in CSV but are excluded from plots (default: 16; use 0 to disable).",
    )
    parser.add_argument(
        "--expected-fixed-panel-trajectories",
        type=int,
        default=128,
        help="Expected fixed-panel trajectories per panel step (default: 16 tasks × 8 repeats = 128; use 0 to disable).",
    )
    parser.add_argument(
        "--progress-bins",
        type=int,
        default=10,
        help="Points in the normalized trajectory-progress view (default: 10; minimum: 2).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional local tokenizer used to align <action> token boundaries.",
    )
    parser.add_argument("--status", choices=["auto", "preliminary", "complete"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_bins < 2:
        raise SystemExit("--progress-bins must be at least 2.")
    diagnostics_paths = [path.resolve() for path in args.diagnostics]
    for path in diagnostics_paths:
        if not path.exists():
            raise SystemExit(f"Diagnostics file does not exist: {path}")
    output_dir = (args.output_dir or diagnostics_paths[0].parent / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows, metadata = select_latest_step_source(diagnostics_paths)
    explorer_log_paths = [path.resolve() for path in (args.explorer_log or [])]
    trainer_log_paths = [path.resolve() for path in (args.trainer_log or [])]
    for path in explorer_log_paths + trainer_log_paths:
        if not path.exists():
            raise SystemExit(f"Log file does not exist: {path}")
    explorer_records: Dict[int, Dict[str, Any]] = {}
    for path in explorer_log_paths:
        explorer_records.update(parse_explorer_log(path))
    trainer_records: Dict[int, Dict[str, Any]] = {}
    for path in trainer_log_paths:
        trainer_records.update(parse_trainer_log(path))
    training_rows = [row for row in selected_rows if diagnostics_source(row) == "train"]
    fixed_panel_rows = [
        row for row in selected_rows if diagnostics_source(row) == "fixed_panel"
    ]
    if not training_rows:
        raise SystemExit("No training diagnostics rows were found.")
    diagnostics = aggregate_diagnostics(training_rows, explorer_records)
    diagnostics.sort(key=lambda row: (finite(row.get("model_version")) or 0.0, int(row["diagnostics_step"])))
    for row in diagnostics:
        row["complete_step"] = (
            True
            if args.expected_trajectories <= 0
            else int(row.get("trajectory_count", 0)) >= args.expected_trajectories
        )
    plotted_diagnostics = [row for row in diagnostics if row["complete_step"]]
    complete_step_ids = {
        int(row["diagnostics_step"]) for row in plotted_diagnostics
    }
    plotted_turn_rows = [
        row for row in training_rows if training_step(row) in complete_step_ids
    ]
    try:
        trajectories = group_trajectories(plotted_turn_rows)
    except ValueError as error:
        raise SystemExit(f"Trajectory diagnostics validation failed: {error}") from error
    turn_diagnostics = aggregate_by_trajectory_turn(trajectories)
    trajectory_summaries = summarize_trajectories(trajectories)
    progress_diagnostics = aggregate_by_normalized_progress(
        trajectories, args.progress_bins
    )

    fixed_panel_diagnostics = aggregate_diagnostics(fixed_panel_rows, explorer_records)
    for row in fixed_panel_diagnostics:
        row["complete_step"] = (
            True
            if args.expected_fixed_panel_trajectories <= 0
            else int(row.get("trajectory_count", 0))
            >= args.expected_fixed_panel_trajectories
        )
    complete_panel_steps = {
        int(row["diagnostics_step"])
        for row in fixed_panel_diagnostics
        if row["complete_step"]
    }
    try:
        fixed_panel_trajectories = group_trajectories(
            [row for row in fixed_panel_rows if training_step(row) in complete_panel_steps]
        )
    except ValueError as error:
        raise SystemExit(f"Fixed-panel diagnostics validation failed: {error}") from error
    fixed_panel_trajectory_summaries = summarize_trajectories(fixed_panel_trajectories)
    fixed_panel_same_task = summarize_fixed_panel_same_task(
        fixed_panel_trajectory_summaries
    )
    same_task_effects, same_task_contrasts = analyze_fixed_panel_same_task_effects(
        fixed_panel_trajectories
    )
    failure_prediction = analyze_fixed_panel_failure_prediction(
        fixed_panel_trajectories
    )
    boundary_events = build_boundary_event_records(fixed_panel_trajectories)
    fixed_panel_frontier = analyze_fixed_panel_frontier(fixed_panel_trajectories)
    action_boundary_events: List[Dict[str, Any]] = []
    if args.tokenizer_path and fixed_panel_trajectories:
        if not args.tokenizer_path.exists():
            raise SystemExit(f"Tokenizer path does not exist: {args.tokenizer_path}")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True
        )
        action_boundary_events = build_action_boundary_records(
            fixed_panel_trajectories, tokenizer
        )

    status = completion_status(args, max((int(row["diagnostics_step"]) for row in diagnostics), default=None))
    top_k = next((row.get("diagnostics_top_k") for row in diagnostics if row.get("diagnostics_top_k") is not None), 16)
    write_csv(output_dir / "diagnostics_by_step.csv", diagnostics, DIAGNOSTIC_FIELDS + sorted({key for row in diagnostics for key in row if key.startswith("eval_")}))
    write_csv(
        output_dir / "diagnostics_by_trajectory_turn.csv",
        turn_diagnostics,
        TURN_FIELDS,
    )
    write_csv(
        output_dir / "diagnostics_by_normalized_progress.csv",
        progress_diagnostics,
        PROGRESS_FIELDS,
    )
    write_csv(
        output_dir / "trajectory_summary.csv",
        trajectory_summaries,
        TRAJECTORY_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_by_step.csv",
        fixed_panel_diagnostics,
        DIAGNOSTIC_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_trajectory_summary.csv",
        fixed_panel_trajectory_summaries,
        TRAJECTORY_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_same_task.csv",
        fixed_panel_same_task,
        FIXED_PANEL_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_same_task_effects.csv",
        same_task_effects,
        SAME_TASK_EFFECT_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_same_task_contrasts.csv",
        same_task_contrasts,
        SAME_TASK_CONTRAST_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_failure_prediction.csv",
        failure_prediction,
        FAILURE_PREDICTION_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_boundary_events.csv",
        boundary_events,
        BOUNDARY_EVENT_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_action_boundary.csv",
        action_boundary_events,
        ACTION_BOUNDARY_FIELDS,
    )
    write_csv(
        output_dir / "fixed_panel_frontier_by_checkpoint.csv",
        fixed_panel_frontier,
        FIXED_PANEL_FRONTIER_FIELDS,
    )
    write_csv(output_dir / "explorer_timing_by_step.csv", list(explorer_records.values()), sorted({key for row in explorer_records.values() for key in row}))
    write_csv(output_dir / "trainer_timing_by_step.csv", list(trainer_records.values()), TRAINER_FIELDS)

    generated: List[str] = []
    for path in (
        save_entropy_by_turn(turn_diagnostics, output_dir, status, top_k),
        save_entropy_by_model_version(
            plotted_diagnostics, output_dir, status, top_k
        ),
        save_teacher_entropy_by_outcome_progress(
            progress_diagnostics,
            trajectory_summaries,
            output_dir,
            status,
            top_k,
        ),
        save_teacher_entropy_rollout_variability(
            trajectories,
            trajectory_summaries,
            output_dir,
            status,
            top_k,
        ),
        save_teacher_entropy_frontier_heatmap(
            trajectories, output_dir, status, top_k
        ),
        save_teacher_entropy_threshold_crossing(
            trajectories,
            output_dir,
            status,
            outcome=None,
            filename="teacher_entropy_threshold_crossing_all.png",
        ),
        save_teacher_entropy_threshold_crossing(
            trajectories,
            output_dir,
            status,
            outcome=False,
            filename="teacher_entropy_threshold_crossing_failure.png",
        ),
        save_teacher_entropy_threshold_crossing(
            trajectories,
            output_dir,
            status,
            outcome=True,
            filename="teacher_entropy_threshold_crossing_success.png",
        ),
        save_token_block_observation_boundary(
            trajectories, output_dir, status
        ),
        save_fixed_panel_same_task(
            fixed_panel_same_task, output_dir, status
        ),
        save_fixed_panel_same_task_effects(
            same_task_effects, same_task_contrasts, output_dir, status
        ),
        save_fixed_panel_failure_prediction(
            failure_prediction, output_dir, status
        ),
        save_fixed_panel_boundary_deconfounded(
            boundary_events, output_dir, status
        ),
        save_fixed_panel_action_boundary(
            action_boundary_events, output_dir, status
        ),
        save_fixed_panel_frontier_by_checkpoint(
            fixed_panel_trajectories,
            fixed_panel_frontier,
            output_dir,
            status,
        ),
        save_teacher_entropy_frontier_heatmap(
            fixed_panel_trajectories,
            output_dir,
            status,
            top_k,
            filename="fixed_panel_teacher_entropy_frontier_heatmap.png",
        ),
        save_kl_surprisal(plotted_diagnostics, output_dir, status),
        save_reverse_kl_loss(plotted_diagnostics, trainer_records, output_dir, status),
        save_mass_success(plotted_diagnostics, output_dir, status, top_k),
        save_timing(plotted_diagnostics, explorer_records, trainer_records, output_dir, status),
    ):
        if path is not None:
            generated.append(path.name)

    metadata = dict(metadata)
    metadata.update(
        {
            "status": status,
            "diagnostic_step_count": len(diagnostics),
            "diagnostic_step_min": min((int(row["diagnostics_step"]) for row in diagnostics), default=None),
            "diagnostic_step_max": max((int(row["diagnostics_step"]) for row in diagnostics), default=None),
            "plotted_step_count": len(plotted_diagnostics),
            "partial_steps_excluded_from_plots": [
                int(row["diagnostics_step"]) for row in diagnostics if not row["complete_step"]
            ],
            "expected_trajectories": args.expected_trajectories,
            "expected_fixed_panel_trajectories": args.expected_fixed_panel_trajectories,
            "fixed_panel_step_count": len(fixed_panel_diagnostics),
            "fixed_panel_complete_step_count": len(complete_panel_steps),
            "fixed_panel_trajectory_count": len(fixed_panel_trajectory_summaries),
            "fixed_panel_same_task_mixed_outcome_cell_count": len(same_task_contrasts),
            "fixed_panel_failure_prediction_row_count": len(failure_prediction),
            "fixed_panel_boundary_event_count": len(boundary_events),
            "fixed_panel_action_boundary_row_count": len(action_boundary_events),
            "fixed_panel_action_aligned_turn_count": sum(
                row["boundary"] == "action_start" and row["relative_block"] == 0
                for row in action_boundary_events
            ),
            "fixed_panel_action_alignment_rate": (
                sum(
                    row["boundary"] == "action_start"
                    and row["relative_block"] == 0
                    for row in action_boundary_events
                )
                / len([row for items in fixed_panel_trajectories.values() for row in items])
                if fixed_panel_trajectories
                else None
            ),
            "tokenizer_path": str(args.tokenizer_path.resolve()) if args.tokenizer_path else None,
            "progress_bins": args.progress_bins,
            "trajectory_view": trajectory_view_summary(
                turn_diagnostics, trajectory_summaries
            ),
            "explorer_logs": [str(path) for path in explorer_log_paths],
            "trainer_logs": [str(path) for path in trainer_log_paths],
            "explorer_record_count": len(explorer_records),
            "trainer_record_count": len(trainer_records),
            "generated_files": generated
            + [
                "diagnostics_by_step.csv",
                "diagnostics_by_trajectory_turn.csv",
                "diagnostics_by_normalized_progress.csv",
                "trajectory_summary.csv",
                "fixed_panel_by_step.csv",
                "fixed_panel_trajectory_summary.csv",
                "fixed_panel_same_task.csv",
                "fixed_panel_same_task_effects.csv",
                "fixed_panel_same_task_contrasts.csv",
                "fixed_panel_failure_prediction.csv",
                "fixed_panel_boundary_events.csv",
                "fixed_panel_action_boundary.csv",
                "fixed_panel_frontier_by_checkpoint.csv",
                "explorer_timing_by_step.csv",
                "trainer_timing_by_step.csv",
                "analysis_notes.md",
                "summary.json",
            ],
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_notes(
        output_dir,
        status,
        metadata,
        diagnostics,
        plotted_diagnostics,
        turn_diagnostics,
        trajectory_summaries,
        explorer_records,
        trainer_records,
    )
    print(json.dumps({"status": status, "output_dir": str(output_dir), "steps": len(diagnostics), "generated": metadata["generated_files"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
