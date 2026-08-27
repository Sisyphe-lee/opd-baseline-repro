from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_vanilla_opd_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("plot_vanilla_opd_diagnostics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def diagnostic_row(
    *,
    task_id: int,
    turn: int,
    success: bool,
    student_entropy: float,
    teacher_entropy: float,
    top_k: int = 16,
) -> dict:
    return {
        "training_step": 10,
        "task_id": task_id,
        "run_id": 0,
        "turn": turn,
        "task_success": success,
        "diagnostics_kind": "topk_head_entropy",
        "diagnostics_top_k": top_k,
        "student_entropy_topk": student_entropy,
        "teacher_entropy_topk": teacher_entropy,
        "student_surprisal": student_entropy + 0.1,
        "teacher_surprisal": teacher_entropy + 0.1,
        "student_topk_mass": 0.999,
        "teacher_topk_mass": 0.998,
        "sampled_forward_kl_mean": 0.2,
        "sampled_forward_kl_sum": 2.0,
        "response_tokens": 10,
        "action_valid": True,
    }


def sample_rows() -> list[dict]:
    return [
        diagnostic_row(
            task_id=1,
            turn=turn,
            success=False,
            student_entropy=0.10,
            teacher_entropy=teacher,
        )
        for turn, teacher in enumerate((0.10, 0.20, 0.30))
    ] + [
        diagnostic_row(
            task_id=2,
            turn=turn,
            success=True,
            student_entropy=0.20,
            teacher_entropy=teacher,
        )
        for turn, teacher in enumerate((0.30, 0.20))
    ]


def test_turn_curve_uses_within_trajectory_turn_and_exposes_denominator() -> None:
    trajectories = plotter.group_trajectories(sample_rows())
    turns = plotter.aggregate_by_trajectory_turn(trajectories)

    assert [row["turn"] for row in turns] == [0, 1, 2]
    assert [row["trajectory_count"] for row in turns] == [2, 2, 1]
    assert [row["failure_trajectory_count"] for row in turns] == [1, 1, 1]
    assert [row["success_trajectory_count"] for row in turns] == [1, 1, 0]
    assert turns[0]["teacher_entropy_topk_mean"] == pytest.approx(0.20)
    assert turns[2]["teacher_entropy_topk_mean"] == pytest.approx(0.30)


def test_normalized_progress_equal_weights_every_trajectory() -> None:
    trajectories = plotter.group_trajectories(sample_rows())
    progress = plotter.aggregate_by_normalized_progress(trajectories, progress_bins=3)

    failure = [row for row in progress if row["outcome"] == "failure"]
    success = [row for row in progress if row["outcome"] == "success"]
    assert [row["trajectory_count"] for row in failure] == [1, 1, 1]
    assert [row["teacher_entropy_topk_mean"] for row in failure] == pytest.approx(
        [0.10, 0.20, 0.30]
    )
    assert [row["teacher_entropy_topk_mean"] for row in success] == pytest.approx(
        [0.30, 0.25, 0.20]
    )


def test_trajectory_summary_preserves_opposite_teacher_slopes() -> None:
    trajectories = plotter.group_trajectories(sample_rows())
    summaries = plotter.summarize_trajectories(trajectories)
    by_task = {row["task_id"]: row for row in summaries}

    assert by_task["1"]["teacher_entropy_slope_per_turn"] == pytest.approx(0.10)
    assert by_task["2"]["teacher_entropy_slope_per_turn"] == pytest.approx(-0.10)
    assert by_task["1"]["teacher_entropy_initial"] == pytest.approx(0.10)
    assert by_task["1"]["teacher_entropy_final"] == pytest.approx(0.30)


def test_validation_rejects_mixed_top_k_and_duplicate_turns() -> None:
    mixed = sample_rows()
    mixed[-1]["diagnostics_top_k"] = 8
    with pytest.raises(ValueError, match="exactly one diagnostics_top_k"):
        plotter.group_trajectories(mixed)

    duplicated = sample_rows()
    duplicated.append(dict(duplicated[0]))
    with pytest.raises(ValueError, match="duplicate or missing turns"):
        plotter.group_trajectories(duplicated)


def test_entropy_plot_filenames_make_the_two_time_axes_explicit(tmp_path: Path) -> None:
    trajectories = plotter.group_trajectories(sample_rows())
    turns = plotter.aggregate_by_trajectory_turn(trajectories)
    model_version_rows = [
        {
            "diagnostics_step": 10,
            "model_version": 9,
            "student_entropy_topk_mean": 0.15,
            "teacher_entropy_topk_mean": 0.20,
        }
    ]

    turn_path = plotter.save_entropy_by_turn(turns, tmp_path, "preliminary", 16)
    version_path = plotter.save_entropy_by_model_version(
        model_version_rows, tmp_path, "preliminary", 16
    )
    summaries = plotter.summarize_trajectories(trajectories)
    variability_path = plotter.save_teacher_entropy_rollout_variability(
        trajectories, summaries, tmp_path, "preliminary", 16
    )

    assert turn_path is not None and turn_path.name == "entropy_curve.png"
    assert version_path is not None
    assert version_path.name == "entropy_by_model_version.png"
    assert variability_path is not None
    assert variability_path.name == "teacher_entropy_rollout_variability.png"
    assert turn_path.exists() and version_path.exists() and variability_path.exists()


def test_source_precedence_is_independent_for_train_and_fixed_panel(tmp_path: Path) -> None:
    train = diagnostic_row(
        task_id=1,
        turn=0,
        success=False,
        student_entropy=0.1,
        teacher_entropy=0.2,
    )
    panel = dict(train, diagnostics_source="fixed_panel", game_id="same-game", run_id=1)
    first = tmp_path / "train.jsonl"
    second = tmp_path / "panel.jsonl"
    first.write_text(json.dumps(train) + "\n", encoding="utf-8")
    second.write_text(json.dumps(panel) + "\n", encoding="utf-8")

    rows, metadata = plotter.select_latest_step_source([first, second])

    assert len(rows) == 2
    assert {plotter.diagnostics_source(row) for row in rows} == {"train", "fixed_panel"}
    assert metadata["selected_steps_by_source"] == {"fixed_panel": [10], "train": [10]}


def test_token_block_boundary_and_frontier_plots(tmp_path: Path) -> None:
    rows = sample_rows()
    for row in rows:
        row["teacher_entropy_topk_blocks"] = [
            row["teacher_entropy_topk"] - 0.01,
            row["teacher_entropy_topk"] + 0.01,
        ]
    trajectories = plotter.group_trajectories(rows)

    boundary = plotter.save_token_block_observation_boundary(
        trajectories, tmp_path, "preliminary"
    )
    heatmap = plotter.save_teacher_entropy_frontier_heatmap(
        trajectories, tmp_path, "preliminary", 16
    )
    crossing_all = plotter.save_teacher_entropy_threshold_crossing(
        trajectories,
        tmp_path,
        "preliminary",
        outcome=None,
        filename="teacher_entropy_threshold_crossing_all.png",
    )
    crossing_failure = plotter.save_teacher_entropy_threshold_crossing(
        trajectories,
        tmp_path,
        "preliminary",
        outcome=False,
        filename="teacher_entropy_threshold_crossing_failure.png",
    )
    crossing_success = plotter.save_teacher_entropy_threshold_crossing(
        trajectories,
        tmp_path,
        "preliminary",
        outcome=True,
        filename="teacher_entropy_threshold_crossing_success.png",
    )

    assert boundary is not None and boundary.exists()
    assert heatmap is not None and heatmap.exists()
    assert crossing_all is not None and crossing_all.exists()
    assert crossing_failure is not None and crossing_failure.exists()
    assert crossing_success is not None and crossing_success.exists()


def test_fixed_panel_summary_keeps_same_task_rollouts_together() -> None:
    rows = []
    for run_id, slope in enumerate((0.1, 0.3)):
        rows.append(
            {
                "diagnostics_step": 10,
                "student_model_version": 9,
                "task_id": "game-a",
                "game_id": "game-a",
                "task_type": "pick_and_place_simple",
                "run_id": str(run_id),
                "task_success": bool(run_id),
                "teacher_entropy_slope_per_turn": slope,
                "teacher_entropy_initial": 0.2,
                "teacher_entropy_final": 0.2 + slope,
            }
        )

    summary = plotter.summarize_fixed_panel_same_task(rows)

    assert len(summary) == 1
    assert summary[0]["rollout_count"] == 2
    assert summary[0]["success_rate"] == pytest.approx(0.5)
    assert summary[0]["teacher_entropy_slope_std"] == pytest.approx(0.1)


def fixed_panel_rows() -> list[dict]:
    rows = []
    for game_index in range(4):
        for run_id, success in enumerate((True, False)):
            for turn in range(15):
                teacher = 0.10 + (0.002 if success else 0.015) * turn
                row = diagnostic_row(
                    task_id=game_index,
                    turn=turn,
                    success=success,
                    student_entropy=0.12,
                    teacher_entropy=teacher,
                )
                row.update(
                    {
                        "diagnostics_source": "fixed_panel",
                        "game_id": f"game-{game_index}",
                        "task_type": "pick_and_place_simple",
                        "student_model_version": 9,
                        "run_id": run_id,
                        "prompt_tokens": 300 + turn,
                        "observation_words": 20 + turn,
                        "consecutive_action_repeat_count": int(not success and turn > 8),
                        "sampled_reverse_kl_mean": 0.05 + (0.01 if success else 0.03) * turn,
                        "teacher_entropy_topk_blocks": [teacher - 0.02, teacher],
                    }
                )
                rows.append(row)
    return rows


def test_same_task_effects_and_landmark_prediction_are_rerunnable() -> None:
    trajectories = plotter.group_trajectories(fixed_panel_rows())
    effects, contrasts = plotter.analyze_fixed_panel_same_task_effects(
        trajectories, bootstrap_replicates=8
    )
    predictions = plotter.analyze_fixed_panel_failure_prediction(
        trajectories, cutoffs=(5, 10, 15)
    )

    interaction = next(row for row in effects if row["term"] == "failure_x_progress")
    assert interaction["estimate"] > 0
    assert len(contrasts) == 4
    assert all(row["failure_minus_success_slope"] > 0 for row in contrasts)
    assert len(predictions) == 18
    combined = [row for row in predictions if row["model"] == "Combined history"]
    assert all(row["auroc"] >= 0.5 for row in combined)


def test_boundary_controls_and_fixed_panel_frontier(tmp_path: Path) -> None:
    trajectories = plotter.group_trajectories(fixed_panel_rows())
    events = plotter.build_boundary_event_records(trajectories)
    frontier = plotter.analyze_fixed_panel_frontier(trajectories)

    assert events
    first = events[0]
    assert first["raw_boundary_jump"] == pytest.approx(
        first["same_position_change"] + first["position_reset_contrast"]
    )
    assert frontier
    assert plotter.save_fixed_panel_boundary_deconfounded(
        events, tmp_path, "preliminary"
    ).exists()
    assert plotter.save_fixed_panel_frontier_by_checkpoint(
        trajectories, frontier, tmp_path, "preliminary"
    ).exists()


def test_action_boundary_exact_token_alignment(tmp_path: Path) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [7, 8, 9]

    rows = fixed_panel_rows()[:15]
    for row in rows:
        row["action"] = "go to table 1"
        row["response_token_ids"] = [1, 2, 3, 4, 7, 8, 9, 10, 11]
        row["token_block_sizes"] = [4, 4, 1]
        row["teacher_entropy_topk_blocks"] = [0.1, 0.2, 0.3]
    trajectories = plotter.group_trajectories(rows)
    aligned = plotter.build_action_boundary_records(trajectories, FakeTokenizer())

    assert aligned
    assert {row["boundary"] for row in aligned} == {"action_start", "action_end"}
    assert plotter.save_fixed_panel_action_boundary(
        aligned, tmp_path, "preliminary"
    ).exists()
