#!/usr/bin/env python3

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))

    value = torch.tensor([float(rank + 1)], device="cuda")
    pending = dist.all_reduce(value, async_op=True)

    static_input = torch.ones(1024, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = static_input.square().add_(1)

    pending.wait()
    graph.replay()
    torch.cuda.synchronize()

    expected_sum = world_size * (world_size + 1) / 2
    if value.item() != expected_sum:
        raise RuntimeError(
            f"rank {rank}: all-reduce returned {value.item()}, expected {expected_sum}"
        )
    if not torch.all(static_output == 2):
        raise RuntimeError(f"rank {rank}: CUDA graph replay produced an unexpected value")

    dist.barrier()
    if rank == 0:
        print(f"PASS: NCCL all-reduce and CUDA graph completed on {world_size} GPUs")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
