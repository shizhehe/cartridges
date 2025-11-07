from pathlib import Path
from typing import TYPE_CHECKING, Optional
import os
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import wandb

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.utils import get_logger
from cartridges.models.helpers import ModelHelper

logger = get_logger(__name__)

def _list_cache_files(full_run_path: str) -> list[str]:
    import wandb
    import re
    import os

    api = wandb.Api()
    
    # Get all files from the run
    files = [file.name for file in api.run(full_run_path).files()]

    # Filter for cache-*.pt files using regex
    cache_files = [file for file in files if re.match(r"^cache-.*\.pt$", file)]

    # Extract the epoch or step number from each cache file and create a mapping
    file_to_step = {}
    for file in cache_files:
        # Try to match both epoch and step patterns
        match = re.search(r"cache-(epoch|step)(\d+)\.pt", file)
        if match:
            step_num = int(match.group(2))
            file_to_step[file] = step_num

    # Sort the files by their step/epoch number
    sorted_cache_files = sorted(cache_files, key=lambda x: file_to_step.get(x, 0), reverse=True)
    return sorted_cache_files

class KVFromPretrained(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        # path: Path

        wandb_run_id: str
        filename: Optional[str] = None

    def __init__(self, config: Config):
        self.config = config

    def initialize_kv_cache(
        self,
        tokenizer: Optional[AutoTokenizer]=None,
        model: Optional[torch.nn.Module]=None,
        model_helper: Optional[ModelHelper]=None,
        attn_config: Optional[AttnConfig]=None,
    ) -> TrainableCache:
        is_ddp = "LOCAL_RANK" in os.environ
        print(f"is_ddp: {is_ddp}")
        is_rank_zero = (not is_ddp) or (dist.get_rank() == 0)

        wandb_entity = os.environ.get("CARTRIDGES_WANDB_ENTITY", "shizhehe")
        wandb_project = os.environ.get("CARTRIDGES_WANDB_PROJECT", "dynamic-cartridges")
        
        # Ensure all ranks use the same wandb_run_id by broadcasting from rank 0
        wandb_run_id = self.config.wandb_run_id
        if is_ddp:
            # Broadcast wandb_run_id from rank 0 to ensure consistency
            if is_rank_zero:
                run_id_bytes = wandb_run_id.encode('utf-8')
                run_id_len = len(run_id_bytes)
            else:
                run_id_len = 0
            
            # First broadcast the length
            run_id_len_tensor = torch.tensor(run_id_len, dtype=torch.int64, device='cuda')
            dist.broadcast(run_id_len_tensor, src=0)
            run_id_len = run_id_len_tensor.item()
            
            # Then broadcast the run_id
            if is_rank_zero:
                run_id_tensor = torch.frombuffer(run_id_bytes, dtype=torch.uint8).cuda()
            else:
                run_id_tensor = torch.zeros(run_id_len, dtype=torch.uint8, device='cuda')
            
            dist.broadcast(run_id_tensor, src=0)
            
            if not is_rank_zero:
                wandb_run_id = run_id_tensor.cpu().numpy().tobytes().decode('utf-8')
            
            logger.info(f"[Rank {dist.get_rank()}] Using wandb run ID: {wandb_run_id}")
        else:
            logger.info(f"Using wandb run ID: {wandb_run_id}")

        full_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
        logger.info(f"Restoring cache from wandb run {full_run_path}")

        # Only rank 0 should query WandB API to ensure consistent file selection across ranks
        if is_rank_zero:
            cache_files = _list_cache_files(full_run_path)
            if len(cache_files) == 0:
                raise ValueError(f"No cache checkpoints found for wandb run {full_run_path}")
            
            if self.config.filename is not None:
                assert self.config.filename in cache_files, f"Cache file {self.config.filename} not found in wandb run {wandb_run_id}"
                filename = self.config.filename
            else:
                filename = cache_files[0]
        else:
            filename = None

        # Broadcast the filename from rank 0 to all other ranks
        if is_ddp:
            # Create tensor for filename on all ranks
            if is_rank_zero:
                filename_bytes = filename.encode('utf-8')
                filename_len = len(filename_bytes)
            else:
                filename_len = 0
            
            # First broadcast the length
            filename_len_tensor = torch.tensor(filename_len, dtype=torch.int64, device='cuda')
            dist.broadcast(filename_len_tensor, src=0)
            filename_len = filename_len_tensor.item()
            
            # Then broadcast the filename
            if is_rank_zero:
                filename_tensor = torch.frombuffer(filename_bytes, dtype=torch.uint8).cuda()
            else:
                filename_tensor = torch.zeros(filename_len, dtype=torch.uint8, device='cuda')
            
            dist.broadcast(filename_tensor, src=0)
            
            if not is_rank_zero:
                filename = filename_tensor.cpu().numpy().tobytes().decode('utf-8')
            
            logger.info(f"[Rank {dist.get_rank()}] Using cache file: {filename}")
        else:
            logger.info(f"Using cache file: {filename}")

        cache_dir = Path(os.environ["CARTRIDGES_OUTPUT_DIR"]) / "checkpoints" / f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        path = cache_dir / filename
        if not path.exists():
            logger.info(f"Downloading cache from wandb run {full_run_path} to {cache_dir}")
            if is_rank_zero:
                out = wandb.restore(
                    filename, 
                    run_path=full_run_path, 
                    root=cache_dir,
                )
        
        # Ensure all ranks wait for rank 0 to finish downloading
        if is_ddp:
            dist.barrier()

        logger.info(f"Loading cache from {cache_dir / filename}")
        cache = TrainableCache.from_pretrained(
            str(cache_dir / filename), device='cuda'
        )
                
        return cache
