# -*- coding: utf-8 -*-
"""Vanilla on-policy distillation workflow for AlfWorld.

The student samples a multi-turn trajectory and the frozen teacher scores the
same student responses.  The workflow stores teacher selected-token logprobs
for the OPD loss and, when enabled, writes one JSONL diagnostic row per turn.
"""

import fcntl
import hashlib
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    _create_alfworld_env,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)


def _as_float_list(value: Any) -> List[float]:
    """Convert a tensor/list-like diagnostic value to plain Python floats."""
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    result: List[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(float("nan"))
    return result


def _finite_mean(values: List[float]) -> Optional[float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _finite_sum(values: List[float]) -> Optional[float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return None
    return sum(finite_values)


def _block_means(values: Sequence[float], block_size: int) -> Tuple[List[Optional[float]], List[int]]:
    """Return finite means and physical sizes for contiguous token blocks."""
    if block_size <= 0:
        raise ValueError("diagnostics_token_block_size must be positive")
    means: List[Optional[float]] = []
    sizes: List[int] = []
    for start in range(0, len(values), block_size):
        block = list(values[start : start + block_size])
        means.append(_finite_mean(block))
        sizes.append(len(block))
    return means, sizes


def _first_entropy_frontier_turn(
    values: Sequence[Optional[float]],
    threshold: float,
    baseline_turns: int,
    sustain_turns: int,
) -> Optional[int]:
    """Return the first causal sustained Teacher-entropy crossing turn."""
    if baseline_turns <= 0 or sustain_turns <= 0:
        raise ValueError("Entropy-frontier turn counts must be positive")
    finite_values = [
        float(value) if value is not None and math.isfinite(float(value)) else None
        for value in values
    ]
    baseline_values = [
        value for value in finite_values[:baseline_turns] if value is not None
    ]
    if not baseline_values or len(finite_values) < sustain_turns:
        return None
    baseline = sum(baseline_values) / len(baseline_values)
    for turn in range(sustain_turns - 1, len(finite_values)):
        window = finite_values[turn - sustain_turns + 1 : turn + 1]
        if any(value is None for value in window):
            continue
        drift = sum(float(value) - baseline for value in window) / sustain_turns
        if drift >= threshold:
            return turn
    return None


def _training_step(batch_id: Any) -> int:
    """Extract the numeric Explorer step from train or eval batch ids."""
    if isinstance(batch_id, int):
        return batch_id
    match = re.match(r"^(\d+)", str(batch_id))
    if not match:
        raise ValueError(f"Cannot extract training step from batch_id={batch_id!r}")
    return int(match.group(1))


_ALFWORLD_TASK_FAMILIES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)


def _task_identity(raw_task: Any, game_file: str) -> Dict[str, str]:
    """Build stable game/split/task-family identifiers from an ALFWorld path."""
    raw = raw_task if isinstance(raw_task, dict) else {}
    path = Path(game_file)
    parts = path.parts
    if "json_2.1.1" in parts:
        root_index = parts.index("json_2.1.1")
        relative_parts = parts[root_index + 1 :]
        game_id = "/".join(relative_parts)
        inferred_split = relative_parts[0] if relative_parts else ""
    else:
        game_id = "/".join(path.parts[-4:])
        inferred_split = ""

    task_directory = path.parent.parent.name
    inferred_family = next(
        (family for family in _ALFWORLD_TASK_FAMILIES if task_directory.startswith(family)),
        task_directory.split("-", 1)[0],
    )
    return {
        "game_id": str(raw.get("game_id") or game_id),
        "game_file": game_file,
        "split": str(raw.get("split") or inferred_split),
        "task_type": str(raw.get("task_type") or inferred_family),
    }


def _action_verb(action: str) -> str:
    return action.strip().split(maxsplit=1)[0].lower() if action.strip() else ""


@WORKFLOWS.register_module("OPD_alfworld_workflow")
class OnPolicyDistillVerlAgentAlfworldWorkflow(Workflow):
    """Vanilla OPD for multi-turn ALFWorld episodes.

    Student and teacher entropy diagnostics are top-k head summaries.  They
    are deliberately not presented as full-vocabulary entropy because vLLM
    returns only the requested logprob head.
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    _ROLLOUT_DIAGNOSTIC_KEYS = (
        "rollout_topk_entropy",
        "rollout_topk_mass",
        "rollout_topk_count",
        "rollout_top1_top2_margin",
    )

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.reset(task)

    def reset(self, task: Task):
        """Reset workflow and read per-task diagnostic settings."""
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

        self.diagnostics_enabled = bool(task.workflow_args.get("diagnostics_enabled", False))
        self.diagnostics_top_k = int(task.workflow_args.get("diagnostics_top_k", 0) or 0)
        self.diagnostics_path = task.workflow_args.get("diagnostics_path")
        self.diagnostics_required = bool(task.workflow_args.get("diagnostics_required", True))
        self.diagnostics_source = str(task.workflow_args.get("diagnostics_source", "train"))
        self.diagnostics_token_block_size = int(
            task.workflow_args.get("diagnostics_token_block_size", 4) or 4
        )
        self.diagnostics_store_token_ids = bool(
            task.workflow_args.get("diagnostics_store_token_ids", True)
        )
        self.diagnostics_store_text = bool(
            task.workflow_args.get("diagnostics_store_text", True)
        )
        self.entropy_frontier_enabled = bool(
            task.workflow_args.get("entropy_frontier_enabled", False)
        )
        self.entropy_frontier_threshold = float(
            task.workflow_args.get("entropy_frontier_threshold", 0.175)
        )
        self.entropy_frontier_baseline_turns = int(
            task.workflow_args.get("entropy_frontier_baseline_turns", 3)
        )
        self.entropy_frontier_sustain_turns = int(
            task.workflow_args.get("entropy_frontier_sustain_turns", 3)
        )
        if self.diagnostics_enabled:
            if self.diagnostics_top_k <= 0:
                raise ValueError("diagnostics_top_k must be positive when diagnostics are enabled")
            if not self.diagnostics_path:
                raise ValueError("diagnostics_path is required when diagnostics are enabled")
            if self.diagnostics_token_block_size <= 0:
                raise ValueError("diagnostics_token_block_size must be positive")
        if self.entropy_frontier_enabled:
            if not self.diagnostics_enabled:
                raise ValueError(
                    "Teacher-entropy frontier masking requires diagnostics_enabled=true"
                )
            if self.entropy_frontier_threshold < 0:
                raise ValueError("entropy_frontier_threshold must be non-negative")
            if self.entropy_frontier_baseline_turns <= 0:
                raise ValueError("entropy_frontier_baseline_turns must be positive")
            if self.entropy_frontier_sustain_turns <= 0:
                raise ValueError("entropy_frontier_sustain_turns must be positive")

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def compute_reward(self, response: Experience) -> float:
        """Return the audited episode success reward for every turn."""
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        """Keep the workflow API compatible with the base workflow."""
        return []

    def _append_diagnostics(self, records: List[Dict[str, Any]]) -> None:
        """Append diagnostic rows with a process-safe file lock."""
        if not self.diagnostics_enabled or not records:
            return
        try:
            path = os.path.abspath(os.fspath(self.diagnostics_path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as output_file:
                fcntl.flock(output_file.fileno(), fcntl.LOCK_EX)
                try:
                    for record in records:
                        output_file.write(
                            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                    output_file.flush()
                finally:
                    fcntl.flock(output_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            if self.diagnostics_required:
                raise
            self.logger.exception("Failed to append Vanilla OPD diagnostics")

    async def run_async(self) -> List[Experience]:
        env = _create_alfworld_env(self.task_desc)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _run_episode(self, env) -> List[Experience]:
        student_model_version_start = (
            await self.model.model_version_async if self.diagnostics_enabled else None
        )
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0
        self._task_success = False
        self._env_lost = False
        self._env_timeout = False
        self._final_reward = 0.0

        task_description = _extract_task(observation)
        history: List[str] = []
        actions: List[str] = []
        observation_hashes: List[str] = []
        turn_responses: List[Experience] = []
        turn_contexts: List[Dict[str, Any]] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            observation_hash = hashlib.sha256(format_obs.encode("utf-8")).hexdigest()
            prior_observation_count = observation_hashes.count(observation_hash)
            admissible_commands = info.get("admissible_commands", [])
            if admissible_commands and isinstance(admissible_commands[0], list):
                admissible_commands = admissible_commands[0]
            admissible_commands = list(admissible_commands)
            reformatted_admissible = "\n ".join(
                f"'{command}'" for command in admissible_commands if command != "help"
            )

            if len(history) < HISTORY_LENGTH:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )

            # The prompt is self-contained: bounded history is embedded above,
            # and is not also accumulated as an unbounded chat-message history.
            response = (
                await self.model.chat_async(
                    [{"role": "user", "content": user_content}], **kwargs
                )
            )[0]
            if response.logprobs is None:
                raise RuntimeError(
                    "Vanilla OPD requires student response logprobs. "
                    "Set rollout_args.logprobs (16 when diagnostics are enabled)."
                )
            if self.diagnostics_enabled and self.diagnostics_required:
                missing = [
                    key for key in self._ROLLOUT_DIAGNOSTIC_KEYS if key not in response.info
                ]
                if missing:
                    raise RuntimeError(
                        "Student top-k diagnostics are missing from the rollout response: "
                        + ", ".join(missing)
                    )
            turn_responses.append(response)

            response_text = response.response_text or ""
            action = parse_action(response_text)
            action_valid = action in admissible_commands
            prior_action_count = actions.count(action)
            consecutive_action_repeat_count = 0
            for previous_action in reversed(actions):
                if previous_action != action:
                    break
                consecutive_action_repeat_count += 1
            history.append(_format_history(format_obs, r + 1, action))
            actions.append(action)
            observation_hashes.append(observation_hash)
            observation, step_reward, env_terminated, info = env.step(action)
            info = info or {}
            won = bool(info.get("won", False) or step_reward > 0)
            lost = bool(info.get("lost", False))
            self._task_success = bool(self._task_success or won)
            self._env_lost = bool(self._env_lost or lost)
            self._env_rounds = r + 1
            turn_contexts.append(
                {
                    "action": action,
                    "action_verb": _action_verb(action),
                    "action_valid": action_valid,
                    "prior_action_count": prior_action_count,
                    "consecutive_action_repeat_count": consecutive_action_repeat_count,
                    "admissible_action_count": len(admissible_commands),
                    "observation_hash": observation_hash,
                    "prior_observation_count": prior_observation_count,
                    "observation_chars": len(format_obs),
                    "observation_words": len(format_obs.split()),
                    "observation_text": format_obs if self.diagnostics_store_text else None,
                    "prompt_tokens": response.prompt_length,
                    "response_text": response_text if self.diagnostics_store_text else None,
                    "response_token_ids": (
                        response.tokens[response.prompt_length :].tolist()
                        if self.diagnostics_store_token_ids
                        else None
                    ),
                    "step_reward": float(step_reward),
                    "won": won,
                    "lost": lost,
                    "env_terminated": bool(env_terminated),
                }
            )
            if env_terminated:
                self._env_done = True
                break
        else:
            self._env_rounds = self.max_env_steps

        self._task_success = bool(self._task_success or info.get("won", False))
        self._env_lost = bool(self._env_lost or info.get("lost", False))
        self._env_timeout = bool(
            not self._task_success
            and not self._env_lost
            and (self._env_done or self._env_rounds >= self.max_env_steps)
        )
        self._final_reward = 1.0 if self._task_success else 0.0

        # Teacher scores each full prefix + response sequence.  The response
        # slice starts at prompt_length - 1 because prompt logprobs are shifted.
        per_turn_kl_sums: List[float] = []
        turn_diagnostics: List[Dict[str, Any]] = []
        for i, response in enumerate(turn_responses):
            teacher_output = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
                top_logprobs=self.diagnostics_top_k if self.diagnostics_enabled else None,
                return_diagnostics=self.diagnostics_enabled,
            )
            if self.diagnostics_enabled:
                if not isinstance(teacher_output, dict):
                    raise RuntimeError("Teacher did not return top-k diagnostics")
                teacher_logprobs = teacher_output["logprobs"]
                teacher_topk_entropy = _as_float_list(teacher_output.get("topk_entropy"))
                teacher_topk_mass = _as_float_list(teacher_output.get("topk_mass"))
                teacher_topk_count = _as_float_list(teacher_output.get("topk_count"))
                teacher_top1_top2_margin = _as_float_list(
                    teacher_output.get("top1_top2_margin")
                )
            else:
                teacher_logprobs = teacher_output
                teacher_topk_entropy = []
                teacher_topk_mass = []
                teacher_topk_count = []
                teacher_top1_top2_margin = []

            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs
            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )
            response.teacher_logprobs = teacher_resp_logprobs

            student_values = _as_float_list(student_resp_logprobs)
            teacher_values = _as_float_list(teacher_resp_logprobs)
            sampled_kl_values = [student - teacher for student, teacher in zip(student_values, teacher_values)]
            kl_sum = _finite_sum(sampled_kl_values) or 0.0
            per_turn_kl_sums.append(kl_sum)

            student_topk_entropy = _as_float_list(
                response.info.get("rollout_topk_entropy") if response.info else None
            )
            student_topk_mass = _as_float_list(
                response.info.get("rollout_topk_mass") if response.info else None
            )
            student_topk_count = _as_float_list(
                response.info.get("rollout_topk_count") if response.info else None
            )
            student_top1_top2_margin = _as_float_list(
                response.info.get("rollout_top1_top2_margin") if response.info else None
            )
            teacher_topk_entropy = teacher_topk_entropy[resp_start:]
            teacher_topk_mass = teacher_topk_mass[resp_start:]
            teacher_topk_count = teacher_topk_count[resp_start:]
            teacher_top1_top2_margin = teacher_top1_top2_margin[resp_start:]

            token_series = {
                "student_logprobs": student_values,
                "teacher_logprobs": teacher_values,
                "sampled_reverse_kl": sampled_kl_values,
                "student_entropy_topk": student_topk_entropy,
                "teacher_entropy_topk": teacher_topk_entropy,
                "student_topk_mass": student_topk_mass,
                "teacher_topk_mass": teacher_topk_mass,
                "student_topk_count": student_topk_count,
                "teacher_topk_count": teacher_topk_count,
                "student_top1_top2_margin": student_top1_top2_margin,
                "teacher_top1_top2_margin": teacher_top1_top2_margin,
            }
            if self.diagnostics_enabled:
                expected_tokens = len(student_values)
                mismatches = {
                    name: len(values)
                    for name, values in token_series.items()
                    if len(values) != expected_tokens
                }
                if mismatches:
                    raise RuntimeError(
                        "Token diagnostic lengths do not match the sampled response: "
                        f"expected={expected_tokens}, observed={mismatches}"
                    )

            student_surprisal_values = [-value for value in student_values]
            teacher_surprisal_values = [-value for value in teacher_values]
            block_names = (
                "student_surprisal",
                "teacher_surprisal",
                "sampled_reverse_kl",
                "student_entropy_topk",
                "teacher_entropy_topk",
                "student_topk_mass",
                "teacher_topk_mass",
                "student_topk_count",
                "teacher_topk_count",
                "student_top1_top2_margin",
                "teacher_top1_top2_margin",
            )
            block_series: Dict[str, List[Optional[float]]] = {
                f"{name}_blocks": [] for name in block_names
            }
            token_block_sizes: List[int] = []
            if self.diagnostics_enabled:
                block_inputs = {
                    "student_surprisal": student_surprisal_values,
                    "teacher_surprisal": teacher_surprisal_values,
                    "sampled_reverse_kl": sampled_kl_values,
                    "student_entropy_topk": student_topk_entropy,
                    "teacher_entropy_topk": teacher_topk_entropy,
                    "student_topk_mass": student_topk_mass,
                    "teacher_topk_mass": teacher_topk_mass,
                    "student_topk_count": student_topk_count,
                    "teacher_topk_count": teacher_topk_count,
                    "student_top1_top2_margin": student_top1_top2_margin,
                    "teacher_top1_top2_margin": teacher_top1_top2_margin,
                }
                for name, values in block_inputs.items():
                    means, sizes = _block_means(values, self.diagnostics_token_block_size)
                    if token_block_sizes and sizes != token_block_sizes:
                        raise RuntimeError(
                            f"Inconsistent token block boundaries for {name}: {sizes}"
                        )
                    token_block_sizes = sizes
                    block_series[f"{name}_blocks"] = means

            context = turn_contexts[i]
            metrics = {
                "student_surprisal": _finite_mean(student_surprisal_values),
                "teacher_surprisal": _finite_mean(teacher_surprisal_values),
                "sampled_reverse_kl_mean": _finite_mean(sampled_kl_values),
                "sampled_reverse_kl_sum": kl_sum,
                # Kept so existing analysis can still consume schema-v1 files.
                "sampled_forward_kl_mean": _finite_mean(sampled_kl_values),
                "sampled_forward_kl_sum": kl_sum,
                "student_entropy_topk": _finite_mean(student_topk_entropy),
                "teacher_entropy_topk": _finite_mean(teacher_topk_entropy),
                "student_topk_mass": _finite_mean(student_topk_mass),
                "teacher_topk_mass": _finite_mean(teacher_topk_mass),
                "student_topk_count": _finite_mean(student_topk_count),
                "teacher_topk_count": _finite_mean(teacher_topk_count),
                "student_top1_top2_margin": _finite_mean(student_top1_top2_margin),
                "teacher_top1_top2_margin": _finite_mean(teacher_top1_top2_margin),
                "response_tokens": len(student_values),
                "token_block_size": self.diagnostics_token_block_size,
                "token_block_sizes": token_block_sizes,
                **block_series,
            }
            turn_diagnostics.append({**context, **metrics, "turn": i})

            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i

        entropy_frontier_turn: Optional[int] = None
        if self.entropy_frontier_enabled:
            entropy_frontier_turn = _first_entropy_frontier_turn(
                [item.get("teacher_entropy_topk") for item in turn_diagnostics],
                threshold=self.entropy_frontier_threshold,
                baseline_turns=self.entropy_frontier_baseline_turns,
                sustain_turns=self.entropy_frontier_sustain_turns,
            )
        retained_response_tokens = 0
        for turn, (response, item) in enumerate(zip(turn_responses, turn_diagnostics)):
            loss_masked = entropy_frontier_turn is not None and turn >= entropy_frontier_turn
            item["entropy_frontier_loss_masked"] = loss_masked
            if loss_masked:
                if response.action_mask is None:
                    raise RuntimeError("OPD response is missing action_mask")
                response.action_mask = response.action_mask.clone()
                response.action_mask.zero_()
            else:
                retained_response_tokens += int(item["response_tokens"])

        trajectory_kl_divergence = sum(per_turn_kl_sums)
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics.update(
                {
                    "task_success": 1.0 if self._task_success else 0.0,
                    "env_terminated": 1.0 if self._env_done else 0.0,
                    "env_timeout": 1.0 if self._env_timeout else 0.0,
                    "env_lost": 1.0 if self._env_lost else 0.0,
                    "env_rounds": self._env_rounds,
                    "env_done": 1.0 if self._env_done else 0.0,
                    "kl_divergence": trajectory_kl_divergence,
                    "kl_divergence_per_token": (
                        trajectory_kl_divergence
                        / max(sum(item["response_tokens"] for item in turn_diagnostics), 1)
                    ),
                    "entropy_frontier_triggered": (
                        1.0 if entropy_frontier_turn is not None else 0.0
                    ),
                    "entropy_frontier_turn": float(
                        entropy_frontier_turn
                        if entropy_frontier_turn is not None
                        else -1
                    ),
                    "entropy_frontier_retained_turns": float(
                        entropy_frontier_turn
                        if entropy_frontier_turn is not None
                        else len(turn_responses)
                    ),
                    "entropy_frontier_masked_turns": float(
                        len(turn_responses) - entropy_frontier_turn
                        if entropy_frontier_turn is not None
                        else 0
                    ),
                    "entropy_frontier_retained_token_fraction": (
                        retained_response_tokens
                        / max(sum(item["response_tokens"] for item in turn_diagnostics), 1)
                    ),
                }
            )
            for item in turn_diagnostics:
                turn_number = item["turn"] + 1
                for key in (
                    "student_surprisal",
                    "teacher_surprisal",
                    "sampled_reverse_kl_mean",
                    "sampled_reverse_kl_sum",
                    "student_entropy_topk",
                    "teacher_entropy_topk",
                    "student_topk_mass",
                    "teacher_topk_mass",
                    "student_topk_count",
                    "teacher_topk_count",
                    "student_top1_top2_margin",
                    "teacher_top1_top2_margin",
                    "response_tokens",
                ):
                    value = item.get(key)
                    if value is not None:
                        last_response.metrics[
                            f"diagnostics/turn_{turn_number:02d}/{key}"
                        ] = float(value)
            for key in (
                "student_surprisal",
                "teacher_surprisal",
                "sampled_reverse_kl_mean",
                "student_entropy_topk",
                "teacher_entropy_topk",
                "student_topk_mass",
                "teacher_topk_mass",
                "student_top1_top2_margin",
                "teacher_top1_top2_margin",
            ):
                value = _finite_mean(
                    [item[key] for item in turn_diagnostics if item.get(key) is not None]
                )
                if value is not None:
                    last_response.metrics[f"diagnostics/trajectory/{key}"] = float(value)

        student_model_version_end = (
            await self.model.model_version_async if self.diagnostics_enabled else None
        )
        if student_model_version_start != student_model_version_end:
            raise RuntimeError(
                "Student model version changed within one trajectory: "
                f"start={student_model_version_start}, end={student_model_version_end}"
            )

        identity = _task_identity(self.raw_task, str(self.task_desc))
        task_id = self.task.task_id if self.task.task_id is not None else self.task_desc
        dataset_index = (
            self.task.index.get("index")
            if isinstance(getattr(self.task, "index", None), dict)
            else None
        )
        run_id = getattr(self, "run_id_base", 0)
        training_step = _training_step(self.task.batch_id)
        trajectory_id = (
            f"{self.diagnostics_source}:{training_step}:{identity['game_id']}:{run_id}"
        )
        records: List[Dict[str, Any]] = []
        for item in turn_diagnostics:
            records.append(
                {
                    "diagnostics_schema_version": 2,
                    "diagnostics_kind": "topk_head_entropy",
                    "diagnostics_source": self.diagnostics_source,
                    "diagnostics_top_k": self.diagnostics_top_k,
                    "diagnostics_batch_id": str(self.task.batch_id),
                    "training_step": training_step,
                    "is_eval": bool(self.is_eval),
                    "task_id": task_id,
                    "dataset_index": dataset_index,
                    **identity,
                    "student_model_version": student_model_version_start,
                    "trajectory_id": trajectory_id,
                    "run_id": run_id,
                    "turn": item["turn"],
                    "action": item["action"],
                    "action_verb": item["action_verb"],
                    "action_valid": item["action_valid"],
                    "prior_action_count": item["prior_action_count"],
                    "consecutive_action_repeat_count": item[
                        "consecutive_action_repeat_count"
                    ],
                    "admissible_action_count": item["admissible_action_count"],
                    "observation_hash": item["observation_hash"],
                    "prior_observation_count": item["prior_observation_count"],
                    "observation_chars": item["observation_chars"],
                    "observation_words": item["observation_words"],
                    "observation_text": item["observation_text"],
                    "prompt_tokens": item["prompt_tokens"],
                    "response_text": item["response_text"],
                    "response_token_ids": item["response_token_ids"],
                    "step_reward": item["step_reward"],
                    "won": item["won"],
                    "lost": item["lost"],
                    "env_terminated": item["env_terminated"],
                    "trajectory_env_terminated": bool(self._env_done),
                    "env_timeout": bool(self._env_timeout),
                    "env_lost": bool(self._env_lost),
                    "env_done": bool(self._env_done),
                    "env_rounds": self._env_rounds,
                    "task_success": bool(self._task_success),
                    "entropy_frontier_enabled": self.entropy_frontier_enabled,
                    "entropy_frontier_threshold": self.entropy_frontier_threshold,
                    "entropy_frontier_baseline_turns": self.entropy_frontier_baseline_turns,
                    "entropy_frontier_sustain_turns": self.entropy_frontier_sustain_turns,
                    "entropy_frontier_turn": entropy_frontier_turn,
                    "entropy_frontier_triggered": entropy_frontier_turn is not None,
                    "entropy_frontier_loss_masked": item[
                        "entropy_frontier_loss_masked"
                    ],
                    "response_tokens": item["response_tokens"],
                    "student_surprisal": item["student_surprisal"],
                    "teacher_surprisal": item["teacher_surprisal"],
                    "sampled_reverse_kl_mean": item["sampled_reverse_kl_mean"],
                    "sampled_reverse_kl_sum": item["sampled_reverse_kl_sum"],
                    "sampled_forward_kl_mean": item["sampled_forward_kl_mean"],
                    "sampled_forward_kl_sum": item["sampled_forward_kl_sum"],
                    "student_entropy_topk": item["student_entropy_topk"],
                    "teacher_entropy_topk": item["teacher_entropy_topk"],
                    "student_topk_mass": item["student_topk_mass"],
                    "teacher_topk_mass": item["teacher_topk_mass"],
                    "student_topk_count": item["student_topk_count"],
                    "teacher_topk_count": item["teacher_topk_count"],
                    "student_top1_top2_margin": item["student_top1_top2_margin"],
                    "teacher_top1_top2_margin": item["teacher_top1_top2_margin"],
                    "token_block_size": item["token_block_size"],
                    "token_block_sizes": item["token_block_sizes"],
                    "student_surprisal_blocks": item["student_surprisal_blocks"],
                    "teacher_surprisal_blocks": item["teacher_surprisal_blocks"],
                    "sampled_reverse_kl_blocks": item["sampled_reverse_kl_blocks"],
                    "student_entropy_topk_blocks": item[
                        "student_entropy_topk_blocks"
                    ],
                    "teacher_entropy_topk_blocks": item[
                        "teacher_entropy_topk_blocks"
                    ],
                    "student_topk_mass_blocks": item["student_topk_mass_blocks"],
                    "teacher_topk_mass_blocks": item["teacher_topk_mass_blocks"],
                    "student_topk_count_blocks": item["student_topk_count_blocks"],
                    "teacher_topk_count_blocks": item["teacher_topk_count_blocks"],
                    "student_top1_top2_margin_blocks": item[
                        "student_top1_top2_margin_blocks"
                    ],
                    "teacher_top1_top2_margin_blocks": item[
                        "teacher_top1_top2_margin_blocks"
                    ],
                }
            )
        self._append_diagnostics(records)

        # Top-k tensors are useful during the workflow but are not training
        # fields.  Remove them before the experiences enter the replay buffer.
        for response in turn_responses:
            if response.info:
                for key in self._ROLLOUT_DIAGNOSTIC_KEYS:
                    response.info.pop(key, None)

        return turn_responses
