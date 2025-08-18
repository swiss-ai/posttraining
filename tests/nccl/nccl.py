import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)  # 1 GPU per rank

    x = torch.ones(1, device="cuda")
    dist.all_reduce(x)  # sum across ranks
    print(f"rank {rank} sees {x.item()}")  # should equal world_size

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
