from types import SimpleNamespace

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trinity.utils import monitor as monitor_module


class FakeWandbRun:
    def __init__(self) -> None:
        self.log_calls = []
        self.finish_calls = 0

    def log(self, data, step, commit=False) -> None:
        self.log_calls.append((data, step, commit))

    def finish(self) -> None:
        self.finish_calls += 1


class FakeWandb:
    def __init__(self) -> None:
        self.run = FakeWandbRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def make_config(tmp_path, monitor_args=None):
    return SimpleNamespace(
        monitor=SimpleNamespace(
            cache_dir=str(tmp_path),
            monitor_args={} if monitor_args is None else monitor_args,
        )
    )


def test_wandb_monitor_writes_curated_tensorboard_metrics(monkeypatch, tmp_path) -> None:
    fake_wandb = FakeWandb()
    monkeypatch.setattr(monitor_module, "wandb", fake_wandb)
    monitor = monitor_module.WandbMonitor(
        project="project",
        group="group",
        name="experiment",
        role="trainer",
        config=make_config(tmp_path),
    )

    payload = {
        "actor/final_loss": 0.125,
        "critic/score/mean": 0.75,
        "perf/throughput": 1234.5,
        "not/curated": 999.0,
        "actor/ppo_kl": float("nan"),
    }
    monitor.log(payload, step=17, commit=True)
    monitor.close()

    assert fake_wandb.run.log_calls == [(payload, 17, True)]
    events = EventAccumulator(str(tmp_path / "tensorboard" / "trainer"))
    events.Reload()
    scalar_tags = set(events.Tags()["scalars"])
    assert scalar_tags == {
        "02 Training/Actor loss",
        "02 Training/Batch success (%)",
        "06 System/Throughput (tokens/s)",
    }
    assert events.Scalars("02 Training/Batch success (%)")[0].value == pytest.approx(75.0)
    assert events.Scalars("02 Training/Actor loss")[0].step == 17


def test_wandb_monitor_can_disable_tensorboard_mirror(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(monitor_module, "wandb", FakeWandb())
    monitor = monitor_module.WandbMonitor(
        project="project",
        group="group",
        name="experiment",
        role="explorer",
        config=make_config(tmp_path, {"tensorboard_mirror": False}),
    )
    monitor.log({"rollout/task_success/mean": 1.0}, step=3)
    monitor.close()

    assert not (tmp_path / "tensorboard").exists()


def test_wandb_monitor_defaults_enable_tensorboard() -> None:
    defaults = monitor_module.WandbMonitor.default_args()
    assert defaults["tensorboard_mirror"] is True
    assert defaults["tensorboard_flush_secs"] == 30
