#!/usr/bin/env python3
"""Create deterministic, repository-local ALFWorld full-274 manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_ROOT.parents[1]
GAME_ROOT = REPO_ROOT / "data" / "alfworld_runtime" / "json_2.1.1"
OUTPUT_ROOT = RUN_ROOT / "manifests"


def task_type(game: Path) -> str:
    return game.parent.parent.name.rsplit("-", 4)[0]


def write_split(source: str, filename: str, label: str, expected: int) -> dict:
    games = sorted((GAME_ROOT / source).glob("*/*/game.tw-pddl"))
    if len(games) != expected:
        raise RuntimeError(f"Expected {expected} {source} games, found {len(games)}")

    output = OUTPUT_ROOT / filename
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8") as handle:
        for game in games:
            row = {
                "game_file": str(game.resolve()),
                "target": "",
                "split": label,
                "task_type": task_type(game),
            }
            line = json.dumps(row, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return {"count": len(games), "sha256": digest.hexdigest()}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ordering": "sorted game path",
        "seen": write_split("valid_seen", "full_valid_seen.jsonl", "seen", 140),
        "unseen": write_split(
            "valid_unseen", "full_valid_unseen.jsonl", "unseen", 134
        ),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
