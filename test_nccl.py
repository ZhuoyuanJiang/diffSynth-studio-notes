#!/usr/bin/env python3
"""
Simple NCCL test to verify GPU communication works
"""
import torch
import torch.distributed as dist
import os

def test_nccl():
    # Initialize process group
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')

    print(f"[Rank {rank}/{world_size}] Initialized on {device}")

    # Create a small tensor
    tensor = torch.ones(100, device=device) * rank

    print(f"[Rank {rank}] Before broadcast: {tensor[:5]}")

    # Broadcast from rank 0
    dist.broadcast(tensor, src=0)

    print(f"[Rank {rank}] After broadcast: {tensor[:5]}")

    # All-reduce test
    tensor2 = torch.ones(100, device=device) * rank
    dist.all_reduce(tensor2, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))

    print(f"[Rank {rank}] All-reduce result: {tensor2[0].item()}, expected: {expected}")

    if tensor2[0].item() == expected:
        print(f"[Rank {rank}] ✅ NCCL test PASSED")
    else:
        print(f"[Rank {rank}] ❌ NCCL test FAILED")

    dist.destroy_process_group()

if __name__ == "__main__":
    test_nccl()
