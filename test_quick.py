#!/usr/bin/env python3
"""Quick test of MCCL multi-stream engine optimizations."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import mccl

def test_two_rank_allreduce(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['MCCL_PORT_BASE'] = '29600'
    os.environ['MCCL_LOG_LEVEL'] = 'INFO'
    
    try:
        dist.init_process_group('mccl', rank=rank, world_size=world_size)
        
        # Test basic allreduce
        x = torch.ones(1000, device='mps') * (rank + 1)
        dist.all_reduce(x)
        
        expected = sum(range(1, world_size + 1)) * torch.ones(1000, device='mps')
        correct = torch.allclose(x, expected)
        print(f'Rank {rank}: allreduce result correct = {correct}')
        
        if not correct:
            print(f'  Expected: {expected[:5]}')
            print(f'  Got:      {x[:5]}')
        
        dist.destroy_process_group()
        return correct
    except Exception as e:
        print(f'Rank {rank}: ERROR - {e}')
        return False

if __name__ == '__main__':
    print("Testing MCCL multi-stream engine optimizations...")
    mp.spawn(test_two_rank_allreduce, args=(2,), nprocs=2, join=True)