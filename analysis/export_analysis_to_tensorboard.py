#!/usr/bin/env python3
"""Export repository analysis CSV/JSON/PNG assets into a TensorBoard log tree.

Each source file is isolated as its own TensorBoard run.  Repeated observations
at the same logical step are summarized with mean/min/max/count instead of
emitting ambiguous duplicate scalar events.  Existing exports are archived,
never overwritten in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


STEP_CANDIDATES = (
    "global_step",
    "training_step",
    "trainer_step",
    "explorer_step",
    "model_version",
    "checkpoint_step",
    "step",
    "trajectory_index",
    "turn",
    "epoch",
)
CATEGORY_CANDIDATES = (
    "method",
    "branch",
    "run",
    "run_name",
    "experiment",
    "series",
    "series_label",
    "split",
    "seed",
    "task_type",
    "threshold",
    "tau",
)
LEGACY_MARKERS = ("legacy", "reference_legacy", "qwen3")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_component(value: Any) -> str:
    text = str(value).strip().replace(os.sep, "_")
    text = re.sub(r"[^A-Za-z0-9._=+\-]+", "_", text)
    return text.strip("_") or "unnamed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_namespace(relative: Path) -> Path:
    lowered = relative.as_posix().lower()
    return Path("legacy_reference" if any(x in lowered for x in LEGACY_MARKERS) else "current")


def source_run_dir(output: Path, kind: str, relative: Path) -> Path:
    stem = relative.with_suffix("")
    parts = [safe_component(part) for part in stem.parts]
    return output / source_namespace(relative) / kind / Path(*parts)


def infer_step(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in STEP_CANDIDATES:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        usable = numeric.dropna()
        if len(usable) < max(1, int(0.8 * len(frame))):
            continue
        integer_fraction = np.isclose(usable, np.rint(usable), atol=1e-8).mean()
        if integer_fraction >= 0.98:
            fallback = pd.Series(np.arange(len(frame)), index=frame.index, dtype=float)
            return numeric.fillna(fallback).round().astype("int64"), column
    return pd.Series(np.arange(len(frame)), index=frame.index, dtype="int64"), "row_index"


def infer_categories(frame: pd.DataFrame, step_name: str) -> list[str]:
    result: list[str] = []
    for column in CATEGORY_CANDIDATES:
        if column == step_name or column not in frame:
            continue
        count = frame[column].nunique(dropna=True)
        if 1 < count <= 24:
            result.append(column)
        if len(result) == 2:
            break
    return result


def group_label(columns: list[str], key: Any) -> str:
    if not columns:
        return "all"
    values = key if isinstance(key, tuple) else (key,)
    return "/".join(f"{safe_component(col)}={safe_component(val)}" for col, val in zip(columns, values))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def add_scalar(writer: SummaryWriter, tag: str, value: Any, step: int) -> bool:
    number = finite_float(value)
    if number is None:
        return False
    writer.add_scalar(tag, number, int(step))
    return True


def export_csv(source: Path, relative: Path, output: Path) -> dict[str, Any]:
    frame = pd.read_csv(source, low_memory=False)
    writer = SummaryWriter(str(source_run_dir(output, "csv", relative)))
    step, step_name = infer_step(frame)
    categories = infer_categories(frame, step_name)
    numeric: dict[str, pd.Series] = {}
    for column in frame.columns:
        if column == step_name or column in categories:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            numeric[column] = values

    work = pd.DataFrame({"_tb_step": step}, index=frame.index)
    for column in categories:
        work[column] = frame[column].fillna("NA").astype(str)
    for column, values in numeric.items():
        work[column] = values

    if categories:
        grouper: str | list[str] = categories[0] if len(categories) == 1 else categories
        groups: Iterable[tuple[Any, pd.DataFrame]] = work.groupby(grouper, dropna=False, sort=False)
    else:
        groups = [("all", work)]

    scalar_events = 0
    duplicate_step_groups = 0
    for key, group in groups:
        label = group_label(categories, key)
        grouped = group.groupby("_tb_step", sort=True)
        has_duplicates = bool(grouped.size().max() > 1)
        duplicate_step_groups += int(has_duplicates)
        for metric in numeric:
            summary = grouped[metric].agg(["mean", "min", "max", "count"])
            summary = summary[summary["count"] > 0]
            for logical_step, row in summary.iterrows():
                if has_duplicates:
                    for stat in ("mean", "min", "max", "count"):
                        scalar_events += int(add_scalar(writer, f"{label}/{safe_component(metric)}/{stat}", row[stat], logical_step))
                else:
                    scalar_events += int(add_scalar(writer, f"{label}/{safe_component(metric)}", row["mean"], logical_step))

    metadata = {
        "source": relative.as_posix(),
        "rows": len(frame),
        "columns": list(frame.columns),
        "step_axis": step_name,
        "categorical_series": categories,
        "numeric_metrics": list(numeric),
        "same_step_groups_aggregated": duplicate_step_groups,
    }
    writer.add_text("_provenance/source", f"```json\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n```", 0)
    writer.flush()
    writer.close()
    return {**metadata, "scalar_events": scalar_events}


def walk_json(value: Any, prefix: str = "root") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_json(child, f"{prefix}/{safe_component(key)}")
    elif isinstance(value, list):
        if len(value) <= 10000 and all(isinstance(item, (int, float, bool)) for item in value):
            for index, item in enumerate(value):
                yield prefix, (index, item)
        else:
            for index, child in enumerate(value[:1000]):
                yield from walk_json(child, f"{prefix}/{index}")
    elif isinstance(value, (int, float, bool)):
        yield prefix, (0, value)


def export_json(source: Path, relative: Path, output: Path) -> dict[str, Any]:
    data = json.loads(source.read_text(encoding="utf-8"))
    writer = SummaryWriter(str(source_run_dir(output, "json", relative)))
    scalar_events = 0
    for tag, (step, value) in walk_json(data):
        scalar_events += int(add_scalar(writer, tag, value, step))
    preview = json.dumps(data, ensure_ascii=False, indent=2)
    if len(preview) > 100000:
        preview = preview[:100000] + "\n… [truncated in TensorBoard text view]"
    writer.add_text("_provenance/source", f"Path: `{relative.as_posix()}`\n\n```json\n{preview}\n```", 0)
    writer.flush()
    writer.close()
    return {"source": relative.as_posix(), "scalar_events": scalar_events}


def export_png(source: Path, relative: Path, output: Path) -> dict[str, Any]:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        original_size = image.size
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        imported_size = image.size
        array = np.asarray(image)
    writer = SummaryWriter(str(source_run_dir(output, "images", relative)))
    writer.add_image("image", array, 0, dataformats="HWC")
    writer.add_text(
        "_provenance/source",
        f"Path: `{relative.as_posix()}`  \nOriginal: {original_size[0]}×{original_size[1]}  \nImported: {imported_size[0]}×{imported_size[1]}",
        0,
    )
    writer.flush()
    writer.close()
    return {
        "source": relative.as_posix(),
        "original_size": original_size,
        "imported_size": imported_size,
    }


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve()
    repo = analysis_root.parent
    output = (args.output or analysis_root / "tensorboard_all_analysis").resolve()
    staging = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)

    inputs = []
    for suffix in ("*.csv", "*.json", "*.png"):
        for path in analysis_root.rglob(suffix):
            resolved = path.resolve()
            relative_parts = resolved.relative_to(analysis_root).parts
            if any(part.startswith("tensorboard_") or part.startswith(".tensorboard_") for part in relative_parts):
                continue
            if resolved == output or output in resolved.parents or staging in resolved.parents:
                continue
            inputs.append(resolved)
    inputs = sorted(set(inputs))

    manifest: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "analysis_root": str(analysis_root),
        "output": str(output),
        "git_head": git_value(repo, "rev-parse", "HEAD"),
        "git_status_short": git_value(repo, "status", "--short"),
        "scientific_namespace_rule": {
            "current": "current experiment analyses",
            "legacy_reference": "paths containing legacy/reference_legacy/qwen3; displayed separately and never merged",
        },
        "files": [],
        "failures": [],
    }

    exporters = {".csv": export_csv, ".json": export_json, ".png": export_png}
    for index, source in enumerate(inputs, start=1):
        relative = source.relative_to(analysis_root)
        record: dict[str, Any] = {
            "source": relative.as_posix(),
            "kind": source.suffix.lstrip("."),
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
        }
        try:
            record.update(exporters[source.suffix](source, relative, staging))
            manifest["files"].append(record)
        except Exception as exc:  # Keep a complete audit instead of hiding one bad asset.
            manifest["failures"].append({**record, "error": f"{type(exc).__name__}: {exc}"})
        if index % 20 == 0 or index == len(inputs):
            print(f"exported {index}/{len(inputs)} files; failures={len(manifest['failures'])}", flush=True)

    counts: dict[str, int] = {}
    scalar_events = 0
    for record in manifest["files"]:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
        scalar_events += int(record.get("scalar_events", 0))
    manifest["counts"] = counts
    manifest["scalar_events"] = scalar_events
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    archived = None
    if output.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = output.with_name(f"{output.name}.previous.{suffix}")
        output.rename(archived)
    staging.rename(output)
    print(json.dumps({"output": str(output), "archived_previous": str(archived) if archived else None, "counts": counts, "scalar_events": scalar_events, "failures": len(manifest["failures"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
