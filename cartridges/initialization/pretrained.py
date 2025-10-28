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
        full_run_path = f"{wandb_entity}/{wandb_project}/{self.config.wandb_run_id}"

        logger.info(f"Restoring cache from wandb run {full_run_path}")
        filename = ...

        cache_files = _list_cache_files(full_run_path)
        if len(cache_files) == 0:
            raise ValueError(f"No cache checkpoints found for wandb run {full_run_path}")
        
        if self.config.filename is not None:
            assert self.config.filename in cache_files, f"Cache file {self.config.filename} not found in wandb run {self.config.wandb_run_id}"
            filename = self.config.filename
        else:
            filename = cache_files[0]

        cache_dir = Path(os.environ["CARTRIDGES_OUTPUT_DIR"]) / "checkpoints" / f"{wandb_entity}/{wandb_project}/{self.config.wandb_run_id}"
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
        if is_ddp:
            dist.barrier()

        logger.info(f"Loading cache from {cache_dir / filename}")
        cache = TrainableCache.from_pretrained(
            str(cache_dir / filename), device='cuda'
        )
                
        return cache
