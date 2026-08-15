import asyncio
from types import SimpleNamespace

from trinity.explorer.explorer import Explorer


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _FakeModel:
    def __init__(self, index):
        self.address = f"127.0.0.{index + 1}"
        self.port = 31000 + index
        self.init_calls = []
        self.get_available_address = _RemoteMethod(self._get_available_address)
        self.init_process_group = _RemoteMethod(self._init_process_group)

    async def _get_available_address(self):
        return self.address, self.port

    async def _init_process_group(self, **kwargs):
        self.init_calls.append(kwargs)


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


def test_auxiliary_replicas_use_independent_rank_zero_groups():
    models = [_FakeModel(index) for index in range(4)]
    explorer = object.__new__(Explorer)
    explorer.use_nccl_sync = True
    explorer.models = []
    explorer.auxiliary_models = [models]
    explorer.logger = _Logger()
    explorer.config = SimpleNamespace(
        explorer=SimpleNamespace(
            name="test_explorer",
            rollout_model=SimpleNamespace(tensor_parallel_size=1),
            auxiliary_models=[SimpleNamespace(tensor_parallel_size=1)],
        ),
        synchronizer=SimpleNamespace(sync_timeout=10),
    )

    asyncio.run(
        explorer.setup_weight_sync_group(
            master_address="trainer", master_port=30000, state_dict_meta=[]
        )
    )

    assert all(len(model.init_calls) == 1 for model in models)
    for model in models:
        call = model.init_calls[0]
        assert call["master_address"] == model.address
        assert call["master_port"] == model.port
        assert call["rank_offset"] == 0
        assert call["world_size"] == 1
