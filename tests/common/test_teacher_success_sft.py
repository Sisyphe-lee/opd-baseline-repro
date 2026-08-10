import copy

import pytest

from scripts.build_teacher_success_sft import (
    TokenLimits,
    filter_by_token_limits,
    record_to_prefix_samples,
)


def _record(with_prompts: bool = True) -> dict:
    turns = [
        {
            "turn": 0,
            "observation": "Your task is to: inspect the room",
            "admissible_actions": ["look", "help"],
            "response_text": "<think>inspect</think><action>look</action>",
            "action": "look",
            "action_admissible": True,
        },
        {
            "turn": 1,
            "observation": "a table is visible",
            "admissible_actions": ["inventory", "help"],
            "response_text": "<think>check</think><action>inventory</action>",
            "action": "inventory",
            "action_admissible": False,
        },
    ]
    if with_prompts:
        turns[0]["user_content"] = "exact prompt zero"
        turns[1]["user_content"] = "exact prompt one"
    return {
        "schema_version": 2 if with_prompts else 1,
        "record_id": "trajectory-1",
        "game_file": "/tmp/game.tw-pddl",
        "task_success": True,
        "trajectory": turns,
    }


def test_successful_record_becomes_independent_prefix_samples() -> None:
    samples = record_to_prefix_samples(_record())
    assert len(samples) == 2
    assert [len(sample["messages"]) for sample in samples] == [2, 4]
    assert samples[0]["messages"][0]["content"] == "exact prompt zero"
    assert samples[1]["messages"][-2]["content"] == "exact prompt one"
    samples[1]["messages"][0]["content"] = "mutated"
    assert samples[0]["messages"][0]["content"] == "exact prompt zero"
    assert samples[1]["teacher_action_admissible"] is False


def test_schema_v1_prompt_reconstruction_preserves_step2_task_and_history() -> None:
    samples = record_to_prefix_samples(_record(with_prompts=False))
    step2_prompt = samples[1]["messages"][-2]["content"]
    assert "inspect the room" in step2_prompt
    assert "Observation 1" in step2_prompt
    assert "'inventory'" in step2_prompt


def test_failed_or_malformed_records_are_not_silently_used() -> None:
    failed = _record()
    failed["task_success"] = False
    assert record_to_prefix_samples(failed) == []

    malformed = _record()
    malformed["trajectory"][0]["response_text"] = "Action: look"
    with pytest.raises(ValueError, match="strict action/response mismatch"):
        record_to_prefix_samples(malformed)

    with pytest.raises(ValueError, match="inadmissible teacher action"):
        record_to_prefix_samples(_record(), require_all_admissible=True)


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, return_tensors):
        assert return_tensors == "pt"
        length = 10 * len(messages) + (1 if add_generation_prompt else 0)
        return [list(range(length))]


def test_token_limits_filter_and_annotate_samples() -> None:
    samples = copy.deepcopy(record_to_prefix_samples(_record()))
    kept, dropped = filter_by_token_limits(
        samples, _FakeTokenizer(), TokenLimits(max_prompt_tokens=25, max_response_tokens=20)
    )
    assert len(kept) == 1
    assert dropped == 1
    assert kept[0]["prompt_tokens"] == 11
    assert kept[0]["response_tokens"] == 9
