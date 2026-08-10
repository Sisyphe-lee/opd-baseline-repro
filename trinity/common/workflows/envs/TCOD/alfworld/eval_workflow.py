# -*- coding: utf-8 -*-
"""Evaluation-only ALFWorld workflow using the exact TCOD prompt contract."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import Task, Workflow
from trinity.common.workflows.envs.TCOD.alfworld.eval_utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    _create_alfworld_env,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)


def _parse_action_strict(response: str) -> str:
    """Match the exact parser shipped in the public TCOD repository."""
    try:
        return response.split("<action>")[1].split("</action>")[0].strip()
    except Exception as exc:
        print(f"Error parsing action: {exc}, response = {response}")
        return ""


class TCODEvalAlfworldWorkflow(Workflow):
    """Run a model with the same prompt, history, and parser used by TCOD F2B.

    Unlike the F2B training workflow this class does not require a teacher and does
    not calculate token-level teacher log probabilities, so it can benchmark the
    teacher and student independently under an identical inference contract.
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset(task)

    def reset(self, task: Task):
        self.task = task
        self.task_desc = task.task_desc or "0"
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.accumulate_memory = task.workflow_args.get("accumulate_memory", False)
        self.strict_action_parser = task.workflow_args.get(
            "strict_action_parser", False
        )
        self.result_dir = task.workflow_args.get("result_dir")
        self.evaluation_id = task.workflow_args.get("evaluation_id")
        self.checkpoint_label = task.workflow_args.get("checkpoint_label")

    def _write_task_record(self, record: dict) -> None:
        """Atomically persist one compact task record when explicitly requested.

        Bench mode intentionally discards evaluation ``Experience`` objects before
        they reach the Explorer.  Writing one uniquely named shard per game avoids
        transferring every turn's token tensors through Ray and avoids concurrent
        JSONL appends from many workflow workers.  A post-processing script merges
        these shards in frozen-manifest order.
        """

        if not self.result_dir:
            return
        result_dir = Path(self.result_dir)
        if not result_dir.is_absolute():
            result_dir = Path.cwd() / result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        record_id = hashlib.sha256(self.task_desc.encode("utf-8")).hexdigest()
        record["record_id"] = record_id
        destination = result_dir / f"{record_id}.json"
        temporary = result_dir / f".{record_id}.{os.getpid()}.{uuid4().hex}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)

    async def run_async(self) -> List[Experience]:
        env = _create_alfworld_env(self.task_desc)
        try:
            observation, info = env.reset()
            task_description = _extract_task(observation)
            history: List[str] = []
            memory = []
            turn_responses: List[Experience] = []
            trajectory = []
            env_terminated = False
            task_success = False
            total_reward = 0.0
            parsed_action_count = 0
            admissible_action_count = 0
            unchanged_observation_count = 0
            repeated_action_count = 0
            previous_action = None

            kwargs = {**asdict(self.task.rollout_args), "n": 1}
            # Evaluation does not consume token log probabilities.
            kwargs.pop("logprobs", None)

            for r in range(self.max_env_steps):
                formatted_observation = format_observation(observation)
                admissible_commands = info.get("admissible_commands", [])
                if admissible_commands and isinstance(admissible_commands[0], list):
                    admissible_commands = admissible_commands[0]
                admissible_actions = "\n ".join(
                    f"'{action}'" for action in admissible_commands if action != "help"
                )

                # The initial TextWorld observation contains the task, but later
                # observations do not. Switch to the history template as soon as
                # one transition exists so step 2 still receives task_description.
                if not history:
                    user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=formatted_observation,
                        admissible_actions=admissible_actions,
                    )
                else:
                    user_content = ALFWORLD_TEMPLATE.format(
                        task_description=task_description,
                        step_count=r,
                        history_length=min(HISTORY_LENGTH, len(history)),
                        action_history="\n".join(history[-HISTORY_LENGTH:]),
                        current_step=r + 1,
                        current_observation=formatted_observation,
                        admissible_actions=admissible_actions,
                    )

                # Keep the existing self-contained behavior by default.  The
                # opt-in mode reproduces the public TCOD workflow's accumulated
                # chat memory while retaining the corrected step-2 prompt.
                messages = memory + [{"role": "user", "content": user_content}]
                response = (await self.model.chat_async(messages, **kwargs))[0]
                if self.accumulate_memory:
                    memory = messages + [
                        {"role": "assistant", "content": response.response_text or ""}
                    ]
                turn_responses.append(response)

                response_text = response.response_text or ""
                action = (
                    _parse_action_strict(response_text)
                    if self.strict_action_parser
                    else parse_action(response_text)
                )
                action_parsed = bool(action)
                action_admissible = bool(action and action in admissible_commands)
                action_repeated = bool(previous_action is not None and action == previous_action)
                parsed_action_count += int(action_parsed)
                admissible_action_count += int(action_admissible)
                repeated_action_count += int(action_repeated)
                history.append(_format_history(formatted_observation, r + 1, action))
                next_observation, step_reward, env_terminated, info = env.step(action)
                observation_unchanged = bool(
                    format_observation(next_observation) == formatted_observation
                )
                unchanged_observation_count += int(observation_unchanged)
                total_reward += float(step_reward)
                trajectory.append(
                    {
                        "turn": r,
                        # Persist the exact prompt rather than relying on a later
                        # reconstruction. This keeps teacher-trajectory SFT
                        # provenance auditable at the sample level.
                        "user_content": user_content,
                        "observation": formatted_observation,
                        "admissible_actions": list(admissible_commands),
                        "response_text": response_text,
                        "action": action,
                        "action_parsed": action_parsed,
                        "action_admissible": action_admissible,
                        "action_repeated": action_repeated,
                        "observation_unchanged": observation_unchanged,
                        "step_reward": float(step_reward),
                        "next_observation": format_observation(next_observation),
                        "env_terminated": bool(env_terminated),
                        "won": bool(info.get("won", False)),
                        "lost": bool(info.get("lost", False)),
                    }
                )
                previous_action = action
                observation = next_observation
                # TextWorld's Gym wrapper has a default 50-step Limit wrapper.
                # At that limit it returns done=True even when the ALFWorld goal
                # was not reached. Only `won` (or the positive task reward) is
                # task success; `done` by itself is merely episode termination.
                task_success = bool(info.get("won", False) or step_reward > 0)
                if env_terminated:
                    break

            reward = 1.0 if task_success else 0.0
            for response in turn_responses:
                response.reward = reward
            if turn_responses:
                timed_out = bool(
                    not task_success
                    and not info.get("lost", False)
                    and (env_terminated or r + 1 >= self.max_env_steps)
                )
                env_rounds = r + 1
                turn_responses[-1].metrics = {
                    "task_success": reward,
                    "env_terminated": 1.0 if env_terminated else 0.0,
                    "env_timeout": 1.0 if timed_out else 0.0,
                    "env_rounds": env_rounds,
                    "action_parse_valid_rate": parsed_action_count / env_rounds,
                    "action_admissible_rate": admissible_action_count / env_rounds,
                    "observation_unchanged_rate": unchanged_observation_count / env_rounds,
                    "repeated_action_rate": repeated_action_count / env_rounds,
                }
                raw_task = self.task.raw_task or {}
                self._write_task_record(
                    {
                        "schema_version": 2,
                        "evaluation_id": self.evaluation_id,
                        "checkpoint_label": self.checkpoint_label,
                        "game_file": self.task_desc,
                        "split": raw_task.get("split"),
                        "task_type": raw_task.get("task_type"),
                        "task_success": task_success,
                        "env_rounds": env_rounds,
                        "env_terminated": bool(env_terminated),
                        "env_timeout": timed_out,
                        "won": bool(info.get("won", False)),
                        "lost": bool(info.get("lost", False)),
                        "total_reward": total_reward,
                        "max_env_steps": self.max_env_steps,
                        "action_parser": (
                            "strict_public_tcod"
                            if self.strict_action_parser
                            else "tolerant_local"
                        ),
                        "sampling": {
                            "temperature": kwargs.get("temperature"),
                            "top_p": kwargs.get("top_p"),
                            "top_k": kwargs.get("top_k"),
                            "max_tokens": kwargs.get("max_tokens"),
                        },
                        "action_parse_valid_rate": parsed_action_count / env_rounds,
                        "action_admissible_rate": admissible_action_count / env_rounds,
                        "observation_unchanged_rate": unchanged_observation_count / env_rounds,
                        "repeated_action_rate": repeated_action_count / env_rounds,
                        "trajectory": trajectory,
                    }
                )
            return turn_responses
        finally:
            env.close()
