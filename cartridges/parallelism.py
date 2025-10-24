"""
Parallelism strategies for training large models.

This module provides pluggable parallelism strategies to support different
distributed training approaches (DDP, FSDP, etc.) without changing the
core training loop.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pydantic import Field
from pydrantic import BaseConfig

from cartridges.utils import get_logger

logger = get_logger(__name__)


class ParallelismStrategy(ABC):
    """Abstract base class for parallelism strategies."""
    
    def __init__(self, config: "ParallelismConfig"):
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self, local_rank: int) -> Tuple[bool, bool, int]:
        """
        Initialize the parallelism strategy.
        
        Returns:
            is_ddp: whether distributed training is enabled
            is_rank_zero: whether this is the main process
            num_devices: total number of devices
        """
        pass
    
    @abstractmethod
    def wrap_model(self, model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
        """Wrap the model with appropriate parallelism."""
        pass
    
    @abstractmethod
    def create_sampler(self, dataset, seed: int):
        """Create appropriate data sampler for the dataset."""
        pass
    
    @abstractmethod
    def sync_gradients(self, loss: torch.Tensor, world_size: int) -> torch.Tensor:
        """Synchronize gradients across processes if needed."""
        pass
    
    @abstractmethod
    def barrier(self):
        """Synchronize all processes."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up distributed resources."""
        pass


class DataParallelStrategy(ParallelismStrategy):
    """Standard Distributed Data Parallel (DDP) strategy."""
    
    def initialize(self, local_rank: int) -> Tuple[bool, bool, int]:
        """Initialize DDP."""
        is_ddp = "LOCAL_RANK" in os.environ
        
        if is_ddp:
            torch.cuda.set_device(local_rank)
            
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    device_id=torch.device(local_rank)
                )
            
            logger.info(f"[Rank {dist.get_rank()}] DDP initialized.")
            is_rank_zero = dist.get_rank() == 0
            num_devices = dist.get_world_size()
        else:
            is_rank_zero = True
            num_devices = 1
            
        self.is_initialized = is_ddp
        return is_ddp, is_rank_zero, num_devices
    
    def wrap_model(self, model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
        """Wrap model with DDP."""
        if self.is_initialized:
            logger.info("Wrapping model in DDP")
            return DDP(model, device_ids=[local_rank])
        return model
    
    def create_sampler(self, dataset, seed: int):
        """Create distributed or random sampler."""
        if self.is_initialized:
            return DistributedSampler(dataset, shuffle=True, seed=seed)
        else:
            return torch.utils.data.RandomSampler(
                dataset,
                replacement=False,
                num_samples=None,
                generator=torch.Generator().manual_seed(seed),
            )
    
    def sync_gradients(self, loss: torch.Tensor, world_size: int) -> torch.Tensor:
        """Synchronize loss across processes."""
        if self.is_initialized:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size
        return loss
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def cleanup(self):
        """Clean up DDP resources."""
        if self.is_initialized:
            dist.barrier()


class FullyShardedDataParallelStrategy(ParallelismStrategy):
    """Fully Sharded Data Parallel (FSDP) strategy."""
    
    def initialize(self, local_rank: int) -> Tuple[bool, bool, int]:
        """Initialize FSDP."""
        is_ddp = "LOCAL_RANK" in os.environ
        
        if is_ddp:
            torch.cuda.set_device(local_rank)
            
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    device_id=torch.device(local_rank)
                )
            
            logger.info(f"[Rank {dist.get_rank()}] FSDP initialized.")
            is_rank_zero = dist.get_rank() == 0
            num_devices = dist.get_world_size()
        else:
            is_rank_zero = True
            num_devices = 1
            
        self.is_initialized = is_ddp
        return is_ddp, is_rank_zero, num_devices
    
    def wrap_model(self, model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
        """Wrap model with FSDP."""
        if self.is_initialized:
            logger.info("Wrapping model in FSDP")
            
            # Create auto wrap policy based on transformer layers
            # This assumes your model has transformer layers - adjust as needed
            auto_wrap_policy = None
            if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
                # For models like LlamaForCausalLM, Qwen3ForCausalLM
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    # Get the layer class for auto wrapping
                    layer_cls = type(model.model.layers[0])
                    auto_wrap_policy = transformer_auto_wrap_policy.partial(
                        transformer_layer_cls={layer_cls}
                    )
            
            return FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=self.config.mixed_precision_policy,
                sharding_strategy=self.config.sharding_strategy,
                device_id=local_rank,
                sync_module_states=True,
            )
        return model
    
    def create_sampler(self, dataset, seed: int):
        """Create distributed or random sampler."""
        if self.is_initialized:
            return DistributedSampler(dataset, shuffle=True, seed=seed)
        else:
            return torch.utils.data.RandomSampler(
                dataset,
                replacement=False,
                num_samples=None,
                generator=torch.Generator().manual_seed(seed),
            )
    
    def sync_gradients(self, loss: torch.Tensor, world_size: int) -> torch.Tensor:
        """FSDP handles gradient synchronization automatically."""
        # FSDP automatically handles gradient synchronization
        # We still need to average the loss for logging
        if self.is_initialized:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size
        return loss
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def cleanup(self):
        """Clean up FSDP resources."""
        if self.is_initialized:
            dist.barrier()


class ParallelismConfig(BaseConfig):
    """Configuration for parallelism strategies."""
    
    strategy: str = "ddp"  # "ddp" or "fsdp"
    backend: str = "nccl"  # "nccl" or "gloo"
    
    # FSDP-specific options
    sharding_strategy: Optional[Any] = None  # FSDP sharding strategy
    mixed_precision_policy: Optional[Any] = None  # FSDP mixed precision policy
    
    def instantiate(self) -> ParallelismStrategy:
        """Create the appropriate parallelism strategy."""
        if self.strategy == "ddp":
            return DataParallelStrategy(self)
        elif self.strategy == "fsdp":
            # Import FSDP-specific configs here to avoid import errors if not available
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
            
            # Set default FSDP configurations if not provided
            if self.sharding_strategy is None:
                self.sharding_strategy = ShardingStrategy.FULL_SHARD
            
            if self.mixed_precision_policy is None:
                self.mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            
            return FullyShardedDataParallelStrategy(self)
        else:
            raise ValueError(f"Unknown parallelism strategy: {self.strategy}")