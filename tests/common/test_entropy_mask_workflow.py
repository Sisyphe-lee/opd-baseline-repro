import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from trinity.common.config import FormatConfig, GenerationConfig
from trinity.common.experience import Experience
from trinity.common.models.vllm_model import vLLMRolloutModel, _summarize_logprob_dict
from trinity.common.models.vllm_patch.worker_patch import (
    PROMPT_LOGPROBS_START_ARG,
    _get_prompt_logprobs_start,
    _prompt_logprobs_chunk_overlap,
)
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.alfworld.OPD_entropy_mask_workflow import (
    EntropyMaskPromptFixedOPDWorkflow,
    first_entropy_frontier_turn,
    scheduled_frontier_threshold,
)
from trinity.common.workflows.workflow import Task


def test_frontier_requires_sustained_positive_drift() -> None:
    values = [0.10, 0.10, 0.10, 0.25, 0.25, 0.25, 0.25]
    assert first_entropy_frontier_turn(values, 0.10, 3, 3) == 5
    assert first_entropy_frontier_turn(values, 0.16, 3, 3) is None
    assert first_entropy_frontier_turn([0.1, None, 0.1, 0.3], 0.1, 3, 2) is None
    with pytest.raises(ValueError):
        first_entropy_frontier_turn(values, 0.0, 3, 3)


def test_linear_frontier_schedule_switches_explicitly_to_full() -> None:
    kwargs = {
        "base_threshold": 0.075,
        "schedule": "linear_to_full",
        "start_model_version": 80,
        "end_model_version": 160,
        "end_threshold": 1.0,
    }
    assert scheduled_frontier_threshold(model_version=79, **kwargs) == (0.075, False)
    assert scheduled_frontier_threshold(model_version=80, **kwargs) == (0.075, False)
    threshold, is_full = scheduled_frontier_threshold(model_version=120, **kwargs)
    assert threshold == pytest.approx(0.5375)
    assert is_full is False
    assert scheduled_frontier_threshold(model_version=160, **kwargs) == (None, True)
    assert scheduled_frontier_threshold(model_version=250, **kwargs) == (None, True)


def test_cosine_frontier_schedule_holds_end_threshold() -> None:
    kwargs = {
        "base_threshold": 0.1,
        "schedule": "cosine_hold",
        "start_model_version": 80,
        "end_model_version": 160,
        "end_threshold": 0.175,
    }
    assert scheduled_frontier_threshold(model_version=79, **kwargs) == (0.1, False)
    assert scheduled_frontier_threshold(model_version=80, **kwargs) == (0.1, False)
    threshold, is_full = scheduled_frontier_threshold(model_version=120, **kwargs)
    assert threshold == pytest.approx(0.1375)
    assert is_full is False
    assert scheduled_frontier_threshold(model_version=160, **kwargs) == (0.175, False)
    assert scheduled_frontier_threshold(model_version=250, **kwargs) == (0.175, False)


def test_frontier_schedule_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        scheduled_frontier_threshold(
            base_threshold=0.1,
            model_version=80,
            schedule="linear_to_full",
            start_model_version=80,
            end_model_version=80,
        )
    with pytest.raises(ValueError):
        scheduled_frontier_threshold(
            base_threshold=0.1,
            model_version=80,
            schedule="cosine",
        )


def test_topk_summary_excludes_extra_sampled_token() -> None:
    values = {
        99: SimpleNamespace(logprob=-8.0, rank=99),
        1: SimpleNamespace(logprob=-0.2, rank=1),
        2: SimpleNamespace(logprob=-1.2, rank=2),
    }
    entropy, mass, count, margin = _summarize_logprob_dict(values, top_k=2)
    expected_probs = torch.exp(torch.tensor([-0.2, -1.2], dtype=torch.float64))
    expected_entropy = -(expected_probs * torch.tensor([-0.2, -1.2])).sum().item()
    assert entropy == pytest.approx(expected_entropy)
    assert mass == pytest.approx(expected_probs.sum().item())
    assert count == 2
    assert margin == pytest.approx(1.0)


class FakeStudent:
    def __init__(self, model_version: int = 7):
        self.messages = []
        self._model_version = model_version

    @property
    async def model_version_async(self):
        return self._model_version

    async def chat_async(self, messages, **kwargs):
        assert kwargs["logprobs"] == 16
        self.messages.append([dict(message) for message in messages])
        return [
            Experience(
                tokens=torch.tensor([10, 11, 12, 20, 21]),
                prompt_length=3,
                logprobs=torch.tensor([-0.2, -0.3]),
                response_text="<action>look</action>",
                info={
                    "rollout_topk_entropy": torch.tensor([0.4, 0.5]),
                    "rollout_topk_mass": torch.tensor([0.98, 0.97]),
                    "rollout_topk_count": torch.tensor([16, 16]),
                    "rollout_top1_top2_margin": torch.tensor([0.6, 0.7]),
                },
            )
        ]


class FakeTeacher:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.levels = [0.10, 0.10, 0.10, 0.25, 0.25, 0.25, 0.25]

    def get_openai_async_client(self):
        return object()

    async def logprobs_async(self, **kwargs):
        assert kwargs["top_logprobs"] == 16
        assert kwargs["return_diagnostics"] is True
        assert kwargs["diagnostics_start_index"] == 2
        call = self.calls
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        level = self.levels[call]
        return {
            "logprobs": torch.tensor([-0.4, -0.5]),
            "topk_entropy": torch.tensor([level, level]),
            "topk_mass": torch.tensor([0.95, 0.94]),
            "topk_count": torch.tensor([16, 16]),
            "top1_top2_margin": torch.tensor([0.4, 0.5]),
        }


class SevenTurnEnv:
    def __init__(self):
        self.count = 0

    def reset(self):
        return (
            "Your task is to: inspect the room",
            {"admissible_commands": ["look"]},
        )

    def step(self, action):
        assert action == "look"
        self.count += 1
        return (
            f"observation {self.count}",
            0.0,
            self.count == 7,
            {
                "won": False,
                "lost": False,
                "admissible_commands": ["look"],
            },
        )


class TruncatedStudent(FakeStudent):
    async def chat_async(self, messages, **kwargs):
        self.messages.append([dict(message) for message in messages])
        return [
            Experience(
                tokens=torch.tensor([10, 11, 12, 20]),
                prompt_length=3,
                logprobs=torch.zeros(1),
                response_text="x",
                truncate_status="prompt_truncated",
            )
        ]


class TruncatedTeacher:
    def get_openai_async_client(self):
        return object()

    async def logprobs_async(self, **kwargs):
        assert kwargs["diagnostics_start_index"] == 2
        return {
            "logprobs": torch.tensor([-0.3]),
            "topk_entropy": torch.tensor([0.3]),
            "topk_mass": torch.tensor([0.9]),
            "topk_count": torch.tensor([16]),
            "top1_top2_margin": torch.tensor([0.2]),
        }


class TruncatedEnv(SevenTurnEnv):
    def step(self, action):
        assert action == ""
        self.count += 1
        return (
            "truncated observation",
            0.0,
            True,
            {"won": False, "lost": False, "admissible_commands": ["look"]},
        )


def test_full_rollout_is_recorded_but_suffix_is_not_returned(tmp_path) -> None:
    diagnostics_path = tmp_path / "trajectory_metrics.jsonl"
    task = Task(
        format_args=FormatConfig(prompt_key="game_file"),
        rollout_args=GenerationConfig(logprobs=16),
        workflow_args={
            "max_env_steps": 7,
            "diagnostics_enabled": True,
            "diagnostics_required": True,
            "diagnostics_top_k": 16,
            "diagnostics_teacher_concurrency": 4,
            "diagnostics_path": str(diagnostics_path),
            "entropy_frontier_strategy": "entropy",
            "entropy_frontier_threshold": 0.10,
            "entropy_frontier_baseline_turns": 3,
            "entropy_frontier_sustain_turns": 3,
            "entropy_frontier_min_retained_turns": 3,
        },
        raw_task={
            "game_file": (
                "/repo/json_2.1.1/train/pick_and_place_simple-Mug-None-Table-1/"
                "trial_abc/game.tw-pddl"
            )
        },
        batch_id=4,
        task_id=9,
        is_eval=False,
    )
    student = FakeStudent()
    teacher = FakeTeacher()
    workflow = EntropyMaskPromptFixedOPDWorkflow(
        task=task,
        model=student,
        auxiliary_models=[teacher],
    )

    returned = asyncio.run(workflow._run_episode(SevenTurnEnv()))

    rows = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]

    assert teacher.calls == 7
    assert teacher.max_active == 4
    assert len(student.messages) == 7
    assert len(rows) == 7
    assert len(returned) == 5
    assert all(experience.action_mask.any().item() for experience in returned)
    assert [row["loss_retained"] for row in rows] == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert {row["entropy_frontier_turn"] for row in rows} == {5}
    assert len(student.messages[0]) == 1
    assert len(student.messages[1]) == 3
    assert "inspect the room" in student.messages[1][-1]["content"]
    assert returned[-1].metrics["entropy_frontier_full_turns"] == 7.0
    assert returned[-1].metrics["entropy_frontier_retained_turns"] == 5.0


def test_completed_schedule_uses_explicit_full_strategy(tmp_path) -> None:
    diagnostics_path = tmp_path / "scheduled_full.jsonl"
    task = Task(
        format_args=FormatConfig(prompt_key="game_file"),
        rollout_args=GenerationConfig(logprobs=16),
        workflow_args={
            "max_env_steps": 7,
            "diagnostics_enabled": True,
            "diagnostics_required": True,
            "diagnostics_top_k": 16,
            "diagnostics_teacher_concurrency": 4,
            "diagnostics_path": str(diagnostics_path),
            "entropy_frontier_strategy": "entropy",
            "entropy_frontier_threshold": 0.10,
            "entropy_frontier_schedule": "linear_to_full",
            "entropy_frontier_schedule_start_model_version": 0,
            "entropy_frontier_schedule_end_model_version": 6,
            "entropy_frontier_schedule_end_threshold": 1.0,
            "entropy_frontier_baseline_turns": 3,
            "entropy_frontier_sustain_turns": 3,
            "entropy_frontier_min_retained_turns": 3,
        },
        raw_task={"game_file": "/repo/train/pick_and_place_simple/game.tw-pddl"},
        batch_id=5,
        task_id=10,
        is_eval=False,
    )
    workflow = EntropyMaskPromptFixedOPDWorkflow(
        task=task,
        model=FakeStudent(model_version=7),
        auxiliary_models=[FakeTeacher()],
    )

    returned = asyncio.run(workflow._run_episode(SevenTurnEnv()))
    rows = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]

    assert len(returned) == 7
    assert all(row["loss_retained"] for row in rows)
    assert {row["frontier_strategy"] for row in rows} == {"entropy"}
    assert {row["frontier_strategy_effective"] for row in rows} == {"full"}
    assert {row["entropy_frontier_effective_threshold"] for row in rows} == {None}
    assert {row["entropy_frontier_turn"] for row in rows} == {None}


def test_prompt_truncation_records_null_student_entropy(tmp_path) -> None:
    diagnostics_path = tmp_path / "truncated.jsonl"
    task = Task(
        format_args=FormatConfig(prompt_key="game_file"),
        rollout_args=GenerationConfig(logprobs=16),
        workflow_args={
            "max_env_steps": 1,
            "diagnostics_enabled": True,
            "diagnostics_required": True,
            "diagnostics_top_k": 16,
            "diagnostics_path": str(diagnostics_path),
            "entropy_frontier_strategy": "full",
        },
        raw_task={"game_file": "/repo/train/pick_and_place_simple/game.tw-pddl"},
        batch_id=1,
        task_id=1,
        is_eval=False,
    )
    workflow = EntropyMaskPromptFixedOPDWorkflow(
        task=task,
        model=TruncatedStudent(),
        auxiliary_models=[TruncatedTeacher()],
    )

    returned = asyncio.run(workflow._run_episode(TruncatedEnv()))
    row = json.loads(diagnostics_path.read_text())

    assert len(returned) == 1
    assert not returned[0].action_mask.any().item()
    assert row["truncate_status"] == "prompt_truncated"
    assert row["student_entropy_topk"] is None
    assert row["student_entropy_topk_blocks"] == [None]
    assert row["teacher_entropy_topk"] == pytest.approx(0.3)


def test_workflow_registry_resolves_new_type() -> None:
    assert (
        WORKFLOWS.get("OPD_entropy_mask_promptfix_alfworld_workflow")
        is EntropyMaskPromptFixedOPDWorkflow
    )


def test_prompt_logprobs_start_is_request_local_and_validated() -> None:
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None))
    assert _get_prompt_logprobs_start(request, num_prompt_tokens=8) == 0

    request.sampling_params.extra_args = {PROMPT_LOGPROBS_START_ARG: 5}
    assert _get_prompt_logprobs_start(request, num_prompt_tokens=8) == 5

    for invalid in (-1, 8, 1.5, True):
        request.sampling_params.extra_args = {PROMPT_LOGPROBS_START_ARG: invalid}
        with pytest.raises(ValueError):
            _get_prompt_logprobs_start(request, num_prompt_tokens=8)


@pytest.mark.parametrize(
    ("start_idx", "num_logits", "output_start", "expected"),
    [
        (0, 4, 6, None),
        (4, 4, 6, (2, 2, 0)),
        (6, 3, 6, (0, 3, 0)),
        (8, 2, 6, (0, 2, 2)),
        (0, 3, 0, (0, 3, 0)),
    ],
)
def test_prompt_logprobs_chunk_overlap(
    start_idx: int,
    num_logits: int,
    output_start: int,
    expected,
) -> None:
    assert (
        _prompt_logprobs_chunk_overlap(start_idx, num_logits, output_start) == expected
    )


def _fake_prompt_logprob_row(row: int):
    return {
        100 + row: SimpleNamespace(logprob=-0.25 - row, rank=1),
        200 + row: SimpleNamespace(logprob=-1.25 - row, rank=2),
    }


class FakeVLLMLogprobModel:
    config = SimpleNamespace(temperature=1.0)
    logprobs_no_prefix_cache = False

    def __init__(self, shortened: bool):
        self.shortened = shortened
        self.call_kwargs = None

    async def _generate_internal(self, **kwargs):
        self.call_kwargs = kwargs
        rows = [_fake_prompt_logprob_row(row) for row in range(5)]
        if self.shortened:
            rows = rows[3:]
        return SimpleNamespace(prompt_logprobs=[None, *rows])


@pytest.mark.parametrize("shortened", [True, False])
def test_vllm_logprobs_accepts_suffix_or_legacy_full_output(shortened: bool) -> None:
    model = FakeVLLMLogprobModel(shortened=shortened)
    result = asyncio.run(
        vLLMRolloutModel.logprobs(
            model,
            token_ids=list(range(6)),
            top_logprobs=2,
            return_diagnostics=True,
            diagnostics_start_index=3,
        )
    )

    assert model.call_kwargs["extra_args"] == {PROMPT_LOGPROBS_START_ARG: 3}
    assert model.call_kwargs["detokenize"] is False
    assert result["logprobs"].tolist() == pytest.approx([-3.25, -4.25])
    assert result["topk_count"].tolist() == [2, 2]


def test_vllm_logprobs_default_path_remains_full_prompt() -> None:
    model = FakeVLLMLogprobModel(shortened=False)
    result = asyncio.run(
        vLLMRolloutModel.logprobs(
            model,
            token_ids=list(range(6)),
            return_diagnostics=False,
        )
    )

    assert "extra_args" not in model.call_kwargs
    assert "detokenize" not in model.call_kwargs
    assert result.tolist() == pytest.approx([-0.25, -1.25, -2.25, -3.25, -4.25])


def test_vllm_logprobs_rejects_invalid_suffix_before_request() -> None:
    model = FakeVLLMLogprobModel(shortened=True)
    with pytest.raises(ValueError):
        asyncio.run(
            vLLMRolloutModel.logprobs(
                model,
                token_ids=list(range(6)),
                top_logprobs=2,
                return_diagnostics=True,
                diagnostics_start_index=6,
            )
        )
    assert model.call_kwargs is None
