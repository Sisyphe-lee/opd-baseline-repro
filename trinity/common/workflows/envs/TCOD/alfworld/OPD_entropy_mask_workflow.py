# -*- coding: utf-8 -*-
"""Prompt-fixed OPD with retrospective response-level entropy masking.

The environment and teacher always process the complete trajectory. Only the
prefix before the selected frontier is returned to the replay queue. This
isolates loss selection from online early-stopping and compute savings.
"""

import asyncio
import fcntl
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.alfworld.OPD_promptfix_workflow import (
    PromptFixedOnPolicyDistillAlfworldWorkflow,
)
from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)

_TOPK_INFO_KEYS = (
    "rollout_topk_entropy",
    "rollout_topk_mass",
    "rollout_topk_count",
    "rollout_top1_top2_margin",
)


def _float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(float("nan"))
    return result


def _finite_mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _block_means(values: Sequence[float], block_size: int) -> List[Optional[float]]:
    if block_size <= 0:
        raise ValueError("diagnostics_token_block_size must be positive")
    return [
        _finite_mean(values[start : start + block_size])
        for start in range(0, len(values), block_size)
    ]


def first_entropy_frontier_turn(
    values: Sequence[Optional[float]],
    threshold: float,
    baseline_turns: int,
    sustain_turns: int,
) -> Optional[int]:
    """Return the end turn of the first sustained drift window."""
    if threshold <= 0:
        raise ValueError("entropy_frontier_threshold must be positive")
    if baseline_turns <= 0 or sustain_turns <= 0:
        raise ValueError("frontier turn counts must be positive")
    cleaned = [
        float(value) if value is not None and math.isfinite(float(value)) else None
        for value in values
    ]
    baseline_values = [value for value in cleaned[:baseline_turns] if value is not None]
    if len(baseline_values) != baseline_turns:
        return None
    baseline = sum(baseline_values) / baseline_turns
    for turn in range(max(baseline_turns, sustain_turns - 1), len(cleaned)):
        window = cleaned[turn - sustain_turns + 1 : turn + 1]
        if any(value is None for value in window):
            continue
        drift = sum(float(value) - baseline for value in window) / sustain_turns
        if drift >= threshold:
            return turn
    return None


def _training_step(batch_id: Any) -> int:
    match = re.match(r"^(\d+)", str(batch_id))
    if not match:
        raise ValueError(f"Cannot extract training step from batch_id={batch_id!r}")
    return int(match.group(1))


def _task_identity(game_file: str) -> Dict[str, str]:
    path = Path(game_file)
    parts = path.parts
    if "json_2.1.1" in parts:
        relative = parts[parts.index("json_2.1.1") + 1 :]
        game_id = "/".join(relative)
        split = relative[0] if relative else ""
    else:
        game_id = "/".join(parts[-4:])
        split = ""
    task_dir = path.parent.parent.name
    task_type = task_dir.split("-", 1)[0]
    return {
        "game_id": game_id,
        "game_file": game_file,
        "split": split,
        "task_type": task_type,
    }


@WORKFLOWS.register_module("OPD_entropy_mask_promptfix_alfworld_workflow")
class EntropyMaskPromptFixedOPDWorkflow(PromptFixedOnPolicyDistillAlfworldWorkflow):
    """Full rollout plus retrospective teacher-entropy loss selection."""

    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        args = task.workflow_args
        self.diagnostics_enabled = bool(args.get("diagnostics_enabled", True))
        self.diagnostics_required = bool(args.get("diagnostics_required", True))
        self.diagnostics_top_k = int(args.get("diagnostics_top_k", 16))
        self.diagnostics_teacher_concurrency = int(args.get("diagnostics_teacher_concurrency", 4))
        self.diagnostics_token_block_size = int(args.get("diagnostics_token_block_size", 32))
        self.diagnostics_path = args.get("diagnostics_path")
        self.frontier_strategy = str(args.get("entropy_frontier_strategy", "entropy"))
        self.frontier_threshold = float(args.get("entropy_frontier_threshold", 0.175))
        self.frontier_baseline_turns = int(args.get("entropy_frontier_baseline_turns", 3))
        self.frontier_sustain_turns = int(args.get("entropy_frontier_sustain_turns", 3))
        self.min_retained_turns = int(
            args.get("entropy_frontier_min_retained_turns", self.frontier_baseline_turns)
        )
        self.fixed_retained_turns = int(
            args.get("entropy_frontier_fixed_retained_turns", self.max_env_steps)
        )
        if not self.diagnostics_enabled:
            raise ValueError("This workflow requires diagnostics_enabled=true")
        if self.diagnostics_top_k <= 0:
            raise ValueError("diagnostics_top_k must be positive")
        if self.diagnostics_teacher_concurrency <= 0:
            raise ValueError("diagnostics_teacher_concurrency must be positive")
        if not self.diagnostics_path:
            raise ValueError("diagnostics_path is required")
        if self.frontier_strategy not in {"entropy", "full", "fixed"}:
            raise ValueError("entropy_frontier_strategy must be entropy, full, or fixed")
        if self.frontier_strategy == "entropy":
            first_entropy_frontier_turn(
                [],
                self.frontier_threshold,
                self.frontier_baseline_turns,
                self.frontier_sustain_turns,
            )

    def _append_records(self, records: List[Dict[str, Any]]) -> None:
        path = Path(self.diagnostics_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            if self.diagnostics_required:
                raise

    async def _run_episode(self, env) -> List[Experience]:
        model_version_start = await self.model.model_version_async
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0
        self._task_success = False
        self._env_lost = False

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        responses: List[Experience] = []
        contexts: List[Dict[str, Any]] = []
        actions: List[str] = []
        observation_hashes: List[str] = []

        kwargs = {**self.rollout_args, "n": 1}
        if not kwargs.get("logprobs") or int(kwargs["logprobs"]) <= 0:
            raise ValueError("rollout_args.logprobs must be positive for student entropy")

        for turn in range(self.max_env_steps):
            formatted_observation = format_observation(observation)
            observation_hash = hashlib.sha256(formatted_observation.encode("utf-8")).hexdigest()
            admissible = info.get("admissible_commands", [])
            if admissible and isinstance(admissible[0], list):
                admissible = admissible[0]
            admissible = list(admissible)
            reformatted = "\n ".join(f"'{action}'" for action in admissible if action != "help")

            if not history:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=formatted_observation,
                    admissible_actions=reformatted,
                )
            else:
                history_text = "\n".join(history[-HISTORY_LENGTH:])
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=turn,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=history_text,
                    current_step=turn + 1,
                    current_observation=formatted_observation,
                    admissible_actions=reformatted,
                )

            memory = memory + [{"role": "user", "content": user_content}]
            response = (await self.model.chat_async(memory, **kwargs))[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})
            if response.logprobs is None:
                raise RuntimeError("student response logprobs are missing")
            missing = [key for key in _TOPK_INFO_KEYS if key not in (response.info or {})]
            is_prompt_truncated = response.truncate_status == "prompt_truncated"
            if missing and not is_prompt_truncated:
                raise RuntimeError("student top-k diagnostics are missing: " + ", ".join(missing))
            if is_prompt_truncated and response.action_mask.any():
                raise RuntimeError("prompt-truncated response unexpectedly has trainable tokens")

            action = parse_action(response_text)
            prior_action_count = actions.count(action)
            prior_observation_count = observation_hashes.count(observation_hash)
            history.append(_format_history(formatted_observation, turn + 1, action))
            actions.append(action)
            observation_hashes.append(observation_hash)
            observation, step_reward, done, step_info = env.step(action)
            step_info = step_info or {}
            won = bool(step_info.get("won", False) or step_reward > 0)
            lost = bool(step_info.get("lost", False))
            self._task_success = bool(self._task_success or won)
            self._env_lost = bool(self._env_lost or lost)
            responses.append(response)
            contexts.append(
                {
                    "turn": turn,
                    "action": action,
                    "action_valid": action in admissible,
                    "prior_action_count": prior_action_count,
                    "admissible_action_count": len(admissible),
                    "observation_hash": observation_hash,
                    "truncate_status": response.truncate_status,
                    "prior_observation_count": prior_observation_count,
                    "observation_chars": len(formatted_observation),
                    "prompt_tokens": response.prompt_length,
                    "step_reward": float(step_reward),
                    "won": won,
                    "lost": lost,
                    "env_terminated": bool(done),
                }
            )
            info = step_info
            if done:
                self._env_done = True
                self._env_rounds = turn + 1
                # Preserve the frozen training workflow's done-based reward.
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0

        teacher_semaphore = asyncio.Semaphore(self.diagnostics_teacher_concurrency)

        async def score_with_teacher(response: Experience) -> Tuple[int, Dict[str, Any]]:
            response_start = response.prompt_length - 1
            async with teacher_semaphore:
                teacher_output = await self.teacher_model.logprobs_async(
                    tokens=response.tokens.tolist(),
                    temperature=self.temperature,
                    top_logprobs=self.diagnostics_top_k,
                    return_diagnostics=True,
                    diagnostics_start_index=response_start,
                )
            if not isinstance(teacher_output, dict):
                raise RuntimeError("teacher top-k diagnostics are missing")
            return response_start, teacher_output

        teacher_results = await asyncio.gather(
            *(score_with_teacher(response) for response in responses)
        )
        turn_diagnostics: List[Dict[str, Any]] = []
        for turn, (response, teacher_result) in enumerate(zip(responses, teacher_results)):
            _response_start, teacher_output = teacher_result
            teacher_logprobs = teacher_output["logprobs"]
            student_logprobs = response.logprobs
            if len(teacher_logprobs) != len(student_logprobs):
                raise RuntimeError(
                    f"teacher/student token mismatch: {len(teacher_logprobs)} != "
                    f"{len(student_logprobs)}"
                )
            response.teacher_logprobs = teacher_logprobs
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = turn

            student_values = _float_list(student_logprobs)
            teacher_values = _float_list(teacher_logprobs)
            if response.truncate_status == "prompt_truncated":
                missing_values = [float("nan")] * len(student_values)
                student_entropy = missing_values
                student_mass = missing_values
                student_margin = missing_values
            else:
                student_entropy = _float_list(response.info["rollout_topk_entropy"])
                student_mass = _float_list(response.info["rollout_topk_mass"])
                student_margin = _float_list(response.info["rollout_top1_top2_margin"])
            teacher_entropy = _float_list(teacher_output["topk_entropy"])
            teacher_mass = _float_list(teacher_output["topk_mass"])
            teacher_margin = _float_list(teacher_output["top1_top2_margin"])
            series = {
                "student_surprisal": [-value for value in student_values],
                "teacher_surprisal": [-value for value in teacher_values],
                "sampled_reverse_kl": [
                    student - teacher for student, teacher in zip(student_values, teacher_values)
                ],
                "student_entropy_topk": student_entropy,
                "teacher_entropy_topk": teacher_entropy,
                "student_topk_mass": student_mass,
                "teacher_topk_mass": teacher_mass,
                "student_top1_top2_margin": student_margin,
                "teacher_top1_top2_margin": teacher_margin,
            }
            expected = len(student_values)
            bad_lengths = {
                key: len(value) for key, value in series.items() if len(value) != expected
            }
            if bad_lengths:
                raise RuntimeError(f"token diagnostic length mismatch: {bad_lengths}")
            metrics: Dict[str, Any] = {
                **contexts[turn],
                "response_tokens": expected,
            }
            for name, values in series.items():
                metrics[name] = _finite_mean(values)
                metrics[f"{name}_blocks"] = _block_means(values, self.diagnostics_token_block_size)
            metrics["token_block_sizes"] = [
                min(self.diagnostics_token_block_size, expected - start)
                for start in range(0, expected, self.diagnostics_token_block_size)
            ]
            turn_diagnostics.append(metrics)

        entropies = [item["teacher_entropy_topk"] for item in turn_diagnostics]
        detected_frontier: Optional[int] = None
        if self.frontier_strategy == "entropy":
            detected_frontier = first_entropy_frontier_turn(
                entropies,
                self.frontier_threshold,
                self.frontier_baseline_turns,
                self.frontier_sustain_turns,
            )
            retained_turns = (
                len(responses)
                if detected_frontier is None
                else max(self.min_retained_turns, detected_frontier)
            )
        elif self.frontier_strategy == "fixed":
            retained_turns = self.fixed_retained_turns
        else:
            retained_turns = len(responses)
        retained_turns = max(1, min(int(retained_turns), len(responses)))

        model_version_end = await self.model.model_version_async
        if model_version_start != model_version_end:
            raise RuntimeError(
                f"student model version changed within trajectory: "
                f"{model_version_start} != {model_version_end}"
            )

        identity = _task_identity(str(self.task_desc))
        training_step = _training_step(self.task.batch_id)
        run_id = getattr(self, "run_id_base", 0)
        trajectory_id = f"train:{training_step}:{identity['game_id']}:{run_id}"
        records = []
        for item in turn_diagnostics:
            records.append(
                {
                    "diagnostics_schema_version": 3,
                    "diagnostics_kind": "response_topk_head_entropy",
                    "diagnostics_top_k": self.diagnostics_top_k,
                    "training_step": training_step,
                    "student_model_version": model_version_start,
                    "trajectory_id": trajectory_id,
                    "run_id": run_id,
                    **identity,
                    **item,
                    "task_success": self._task_success,
                    "env_done": self._env_done,
                    "env_lost": self._env_lost,
                    "env_rounds": self._env_rounds,
                    "frontier_strategy": self.frontier_strategy,
                    "entropy_frontier_threshold": self.frontier_threshold,
                    "entropy_frontier_baseline_turns": self.frontier_baseline_turns,
                    "entropy_frontier_sustain_turns": self.frontier_sustain_turns,
                    "entropy_frontier_turn": detected_frontier,
                    "loss_retained": item["turn"] < retained_turns,
                    "retained_turns": retained_turns,
                }
            )
        self._append_records(records)

        retained = responses[:retained_turns]
        last = retained[-1]
        if last.metrics is None:
            last.metrics = {}
        full_kl = sum(
            float(item["sampled_reverse_kl"] or 0.0) * int(item["response_tokens"])
            for item in turn_diagnostics
        )
        retained_kl = sum(
            float(item["sampled_reverse_kl"] or 0.0) * int(item["response_tokens"])
            for item in turn_diagnostics[:retained_turns]
        )
        last.metrics.update(
            {
                "env_rounds": self._env_rounds,
                "env_done": 1.0 if self._env_done else 0.0,
                "task_success": 1.0 if self._task_success else 0.0,
                "kl_divergence": retained_kl,
                "diagnostics/full_kl_divergence": full_kl,
                "entropy_frontier_triggered": 1.0 if detected_frontier is not None else 0.0,
                "entropy_frontier_turn": float(
                    detected_frontier if detected_frontier is not None else -1
                ),
                "entropy_frontier_retained_turns": float(retained_turns),
                "entropy_frontier_full_turns": float(len(responses)),
            }
        )
        for response in responses:
            if response.info:
                for key in _TOPK_INFO_KEYS:
                    response.info.pop(key, None)
        return retained
