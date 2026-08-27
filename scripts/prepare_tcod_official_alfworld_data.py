# Copyright 2026 OPD ALFWorld contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build local-path ALFWorld JSONL manifests in the TCOD source order."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_ROOT = REPO_ROOT / "data" / "alfworld_runtime" / "json_2.1.1"
OUTPUT_ROOT = REPO_ROOT / "data" / "tcod_official_alfworld"
SEED = 42


def _sample_all(split: str, rng: random.Random) -> list[Path]:
    games = sorted((GAME_ROOT / split).glob("*/*/game.tw-pddl"))
    if not games:
        raise FileNotFoundError(f"No ALFWorld games found for split {split!r} under {GAME_ROOT}")
    # This intentionally mirrors upstream get_alfworld_data.py. random.sample
    # permutes the entire sorted population even when every game is retained.
    return rng.sample(games, len(games))


def _write_jsonl(path: Path, games: list[Path]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for game in games:
            row = {"game_file": str(game.resolve()), "target": ""}
            line = json.dumps(row) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    rng = random.Random(SEED)
    train = _sample_all("train", rng)
    seen = _sample_all("valid_seen", rng)
    unseen = _sample_all("valid_unseen", rng)

    expected_counts = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}
    actual_counts = {"train": len(train), "valid_seen": len(seen), "valid_unseen": len(unseen)}
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected ALFWorld population: expected {expected_counts}, got {actual_counts}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    hashes = {
        "train.jsonl": _write_jsonl(OUTPUT_ROOT / "train.jsonl", train),
        "train_expert.jsonl": _write_jsonl(OUTPUT_ROOT / "train_expert.jsonl", train),
        "test.jsonl": _write_jsonl(OUTPUT_ROOT / "test.jsonl", seen),
        "test_unseen.jsonl": _write_jsonl(OUTPUT_ROOT / "test_unseen.jsonl", unseen),
    }
    metadata = {
        "source_algorithm": "TCOD_examples/alfworld/get_alfworld_data.py",
        "seed": SEED,
        "counts": actual_counts,
        "sha256": hashes,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
