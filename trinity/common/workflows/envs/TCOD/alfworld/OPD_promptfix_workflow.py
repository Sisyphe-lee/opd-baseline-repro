# -*- coding: utf-8 -*-
"""Prompt-fixed Vanilla OPD for the controlled TCOD comparison.

The upstream ``OPD_alfworld_workflow`` is intentionally left unchanged. This
side-by-side workflow changes only prompt selection: the no-history template is
used on turn 1, while turn 2 onward receives the available interaction history.
"""

from typing import List

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.alfworld.OPD_workflow import (
    OnPolicyDistillVerlAgentAlfworldWorkflow,
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


@WORKFLOWS.register_module("OPD_promptfix_alfworld_workflow")
class PromptFixedOnPolicyDistillAlfworldWorkflow(
    OnPolicyDistillVerlAgentAlfworldWorkflow
):
    """Vanilla OPD with history exposed from the second environment turn."""

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            admissible_commands = info.get("admissible_commands", [])
            if admissible_commands and isinstance(admissible_commands[0], list):
                admissible_commands = admissible_commands[0]
            reformatted_admissible = "\n ".join(
                f"'{s}'" for s in admissible_commands if s != "help"
            )

            # The sole behavioral difference from upstream Vanilla OPD.
            if not history:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )
            else:
                action_history_str = "\n".join(
                    history[-HISTORY_LENGTH:]
                    if len(history) >= HISTORY_LENGTH
                    else history
                )
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )

            memory = memory + [{"role": "user", "content": user_content}]
            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "PromptFixedOnPolicyDistillAlfworldWorkflow requires "
                    "student logprobs; set rollout_args.logprobs (for example 0)."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0

        per_turn_kl_sums: List[float] = []
        for i, response in enumerate(turn_responses):
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )

            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs

            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, "
                f"prompt_length={response.prompt_length}"
            )

            response.teacher_logprobs = teacher_resp_logprobs
            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        trajectory_kl_divergence = sum(per_turn_kl_sums)
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence

        return turn_responses
