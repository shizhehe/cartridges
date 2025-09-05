#!/usr/bin/env python3
"""
Minimal script to reproduce NCCL collective operation timeout issues
without running actual training.

Usage:
    # Run with torchrun (e.g., 2 processes)
    torchrun --nproc_per_node=2 reproduce_nccl_timeout.py

This script simulates the distributed operations from train.py that could
cause NCCL timeouts:
- Process group initialization
- DDP model wrapping
- All-reduce operations (simulating loss reduction)
- Barrier synchronization
- All-gather operations (simulating result collection)
"""

import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class DummyModel(nn.Module):
    """Minimal model to simulate the CacheAndModel structure"""
    def __init__(self, hidden_size=768, vocab_size=32000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.linear(x.mean(dim=1))


def simulate_training_loop(model, local_rank, num_steps=10):
    """Simulate the training loop collective operations that can timeout"""
    print(f"[Rank {dist.get_rank()}] Starting training simulation...")
    
    for step in range(num_steps):
        print(f"[Rank {dist.get_rank()}] Step {step}")
        
        # Simulate forward pass
        dummy_input = torch.randint(0, 1000, (4, 128), device=local_rank)
        outputs = model(dummy_input)
        
        # Simulate loss computation
        dummy_labels = torch.randint(0, 1000, (4,), device=local_rank)
        loss = nn.functional.cross_entropy(outputs, dummy_labels)
        
        # Simulate backward pass
        loss.backward()
        
        # Simulate gradient accumulation - this is where NCCL operations happen
        if step % 2 == 1:  # Simulate accumulate_grad_steps
            print(f"[Rank {dist.get_rank()}] Performing optimizer step with NCCL operations")
            
            # Simulate the all_reduce operations from train.py:449-452
            accum_loss = loss.detach()
            accum_tokens = torch.tensor(dummy_input.size(0), device=local_rank)
            
            print(f"[Rank {dist.get_rank()}] Before all_reduce - loss: {accum_loss.item()}")
            
            # This is the operation that was timing out in your logs
            dist.all_reduce(accum_loss, op=dist.ReduceOp.SUM)
            accum_loss /= dist.get_world_size()
            
            dist.all_reduce(accum_tokens, op=dist.ReduceOp.SUM)
            
            print(f"[Rank {dist.get_rank()}] After all_reduce - loss: {accum_loss.item()}")
        
        # Simulate periodic evaluation barriers (from train.py:702, 741)
        if step % 5 == 0:
            print(f"[Rank {dist.get_rank()}] Evaluation barrier")
            dist.barrier()
        
        # Add some processing time to simulate real workload
        time.sleep(0.1)
    
    print(f"[Rank {dist.get_rank()}] Training simulation completed")


def simulate_evaluation(local_rank):
    """Simulate evaluation collective operations"""
    print(f"[Rank {dist.get_rank()}] Starting evaluation simulation...")
    
    # Simulate evaluation metrics collection
    dummy_results = [{"loss": torch.rand(1).item(), "rank": dist.get_rank()}]
    
    # Simulate all_gather_object from train.py:707, 889
    print(f"[Rank {dist.get_rank()}] Performing all_gather_object")
    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_results, dummy_results)
    
    if dist.get_rank() == 0:
        print(f"Gathered results: {gathered_results}")
    
    # Evaluation barrier
    dist.barrier()
    print(f"[Rank {dist.get_rank()}] Evaluation completed")


def main():
    # Check if running with torchrun
    if "LOCAL_RANK" not in os.environ:
        print("ERROR: This script should be run with torchrun")
        print("Example: torchrun --nproc_per_node=2 reproduce_nccl_timeout.py")
        exit(1)
    
    # Get distributed training parameters
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Initializing process {local_rank} of {world_size}")
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group (from train.py:157-159)
    print(f"[Rank {local_rank}] Initializing process group with NCCL backend...")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", 
            device_id=torch.device(local_rank)
        )
    
    print(f"[Rank {dist.get_rank()}] Process group initialized. World size: {dist.get_world_size()}")
    
    # Create and wrap model in DDP
    model = DummyModel().to(local_rank).to(torch.bfloat16)
    
    # Wrap in DDP (from train.py:255)
    print(f"[Rank {dist.get_rank()}] Wrapping model in DDP...")
    model = DDP(model, device_ids=[local_rank])
    
    # Initial barrier (from train.py:256)
    print(f"[Rank {dist.get_rank()}] Initial barrier after DDP setup")
    dist.barrier()
    
    # Simulate the training loop operations that can cause timeouts
    simulate_training_loop(model, local_rank, num_steps=1000)
    
    # Simulate evaluation
    simulate_evaluation(local_rank)
    
    # Final barrier (from train.py:538)
    print(f"[Rank {dist.get_rank()}] Final barrier before cleanup")
    dist.barrier()
    
    print(f"[Rank {dist.get_rank()}] Script completed successfully!")


if __name__ == "__main__":
    main()