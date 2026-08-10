#!/usr/bin/env python3
"""Convert successful TCOD ALFWorld records into prefix-level hard-label SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from trinity.common.workflows.envs.TCOD.alfworld.eval_utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    _extract_task,
    _format_history,
)


@dataclass(frozen=True)
class TokenLimits:
    max_prompt_tokens: int
    max_response_tokens: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return _sha256_bytes(raw)


def _strict_action(response: str) -> str | None:
    if response.count("<action>") != 1 or response.count("</action>") != 1:
        return None
    action = response.split("<action>", 1)[1].split("</action>", 1)[0].strip()
    return action or None


def _reconstruct_user_content(trajectory: Sequence[dict], index: int) -> str:
    """Compatibility path for old records that predate persisted prompts."""

    turn = trajectory[index]
    admissible = turn.get("admissible_actions")
    if not isinstance(admissible, list):
        raise ValueError(f"turn {index}: admissible_actions is not a list")
    actions = "\n ".join(f"'{item}'" for item in admissible if item != "help")
    if index == 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=turn["observation"], admissible_actions=actions
        )
    history = [
        _format_history(item["observation"], step + 1, item["action"])
        for step, item in enumerate(trajectory[:index])
    ]
    return ALFWORLD_TEMPLATE.format(
        task_description=_extract_task(trajectory[0]["observation"]),
        step_count=index,
        history_length=min(HISTORY_LENGTH, len(history)),
        action_history="\n".join(history[-HISTORY_LENGTH:]),
        current_step=index + 1,
        current_observation=turn["observation"],
        admissible_actions=actions,
    )


def record_to_prefix_samples(
    record: dict, *, require_all_admissible: bool = False
) -> list[dict]:
    """Validate one successful trajectory and emit one training row per turn."""

    if not record.get("task_success"):
        return []
    game_file = record.get("game_file")
    trajectory = record.get("trajectory")
    if not isinstance(game_file, str) or not game_file:
        raise ValueError("successful record is missing game_file")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"successful record has no trajectory: {game_file}")
    for index, turn in enumerate(trajectory):
        if turn.get("turn") != index:
            raise ValueError(f"non-contiguous turn index for {game_file}")
        response, action = turn.get("response_text"), turn.get("action")
        if not isinstance(response, str) or _strict_action(response) != action:
            raise ValueError(f"strict action/response mismatch for {game_file} turn {index}")
        if require_all_admissible and not turn.get("action_admissible"):
            raise ValueError(f"inadmissible teacher action for {game_file} turn {index}")

    trajectory_id = record.get("record_id") or _sha256_bytes(game_file.encode())
    source_sha256 = _canonical_sha256(record)
    messages: list[dict] = []
    samples = []
    for index, turn in enumerate(trajectory):
        prompt = turn.get("user_content")
        if not isinstance(prompt, str) or not prompt:
            prompt = _reconstruct_user_content(trajectory, index)
        messages += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": turn["response_text"]},
        ]
        samples.append(
            {
                "messages": [dict(message) for message in messages],
                "game_file": game_file,
                "trajectory_id": trajectory_id,
                "turn": index,
                "teacher_action_admissible": bool(turn.get("action_admissible")),
                "source_record_sha256": source_sha256,
            }
        )
    return samples


def _iter_records(directories: Sequence[Path], jsonls: Sequence[Path]) -> Iterator[dict]:
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"record directory does not exist: {directory}")
        for path in sorted(directory.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                yield json.load(handle)
    for path in jsonls:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{number}") from exc


def _token_lengths(tokenizer, messages: list[dict]) -> tuple[int, int]:
    prompt = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, return_tensors="pt"
    )[0]
    full = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, return_tensors="pt"
    )[0]
    return len(prompt), len(full) - len(prompt)


def filter_by_token_limits(
    samples: Iterable[dict], tokenizer, limits: TokenLimits
) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for sample in samples:
        prompt_tokens, response_tokens = _token_lengths(tokenizer, sample["messages"])
        sample.update(prompt_tokens=prompt_tokens, response_tokens=response_tokens)
        if (
            prompt_tokens > limits.max_prompt_tokens
            or response_tokens > limits.max_response_tokens
        ):
            dropped += 1
        else:
            kept.append(sample)
    return kept, dropped


def _atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", action="append", default=[], type=Path)
    parser.add_argument("--record-jsonl", action="append", default=[], type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=10240)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--min-samples", type=int, default=1280)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--require-all-admissible", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.record_dir and not args.record_jsonl:
        raise ValueError("at least one record source is required")
    for path in (args.output_jsonl, args.manifest_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    records = list(_iter_records(args.record_dir, args.record_jsonl))
    if args.expected_records is not None and len(records) != args.expected_records:
        raise ValueError(f"record count is {len(records)}, expected {args.expected_records}")
    game_files = [record.get("game_file") for record in records]
    if len(game_files) != len(set(game_files)):
        raise ValueError("duplicate game_file records are not allowed")

    raw_samples, accepted, rejected, examples = [], 0, 0, []
    for record in records:
        if not record.get("task_success"):
            continue
        try:
            rows = record_to_prefix_samples(
                record, require_all_admissible=args.require_all_admissible
            )
        except ValueError as exc:
            rejected += 1
            if len(examples) < 20:
                examples.append(str(exc))
            continue
        accepted += 1
        raw_samples.extend(rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    limits = TokenLimits(args.max_prompt_tokens, args.max_response_tokens)
    samples, over_limit = filter_by_token_limits(raw_samples, tokenizer, limits)
    if len(samples) < args.min_samples:
        raise ValueError(f"only {len(samples)} usable samples; require {args.min_samples}")
    _atomic_write_jsonl(args.output_jsonl, samples)
    manifest = {
        "schema_version": 1,
        "distillation_type": "teacher-success hard-label SFT (SeqKD)",
        "record_count": len(records),
        "record_success_count": sum(bool(row.get("task_success")) for row in records),
        "accepted_successful_trajectories": accepted,
        "rejected_successful_trajectories": rejected,
        "rejection_examples": examples,
        "raw_prefix_samples": len(raw_samples),
        "over_token_limit_samples": over_limit,
        "output_samples": len(samples),
        "inadmissible_output_samples": sum(
            not row["teacher_action_admissible"] for row in samples
        ),
        "sample_unit": "one accumulated prefix and one teacher response per row",
        "require_all_admissible": args.require_all_admissible,
        "tokenizer": str(args.tokenizer.resolve()),
        "max_prompt_tokens": limits.max_prompt_tokens,
        "max_response_tokens": limits.max_response_tokens,
        "source_record_dirs": [str(path.resolve()) for path in args.record_dir],
        "source_record_jsonls": [str(path.resolve()) for path in args.record_jsonl],
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_sha256": _sha256_bytes(args.output_jsonl.read_bytes()),
    }
    _atomic_write_jsonl(args.manifest_json, [manifest])
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
