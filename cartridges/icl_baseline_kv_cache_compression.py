import math
from capsules.baselines.kv_press import ModelWithCompressedCache, make_press
from capsules.icl_baseline import download_wandb_artifacts
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
import itertools
import time
from typing import Literal, Optional

import os

import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer
import wandb

import pydrantic
import torch.nn as nn

# capsules specific imports
from capsules.context import BaseContextConfig
from capsules.datasets import (
    CapsuleDataset,
    CapsuleGenerateDataset,
)
from capsules.config import ModelConfig

from capsules.generate.generate_training import BaseSectionedContextConfig
from capsules.generate.structs import Context
from capsules.train import (
    EvalDataset,
    EvalDatasetConfig,
    GenerateDataset,
    GenerateDatasetConfig,
    evaluate,
    evaluate_generations,
)
from capsules.utils import WandBConfig, prepare_wandb, seed_everything, get_logger
from capsules.utils.wandb import download_artifacts, figure_to_wandb


logger = get_logger(__name__)


class ICLBaselineWithKVCacheCompressionConfig(pydrantic.RunConfig):
    name: str = "default"  # A name for the run for wandb
    model: ModelConfig
    wandb: Optional[WandBConfig] = None

    # dataset for evaluating perplexity on other generations
    eval_datasets: list[EvalDatasetConfig] = field(default_factory=list)
    eval_log_table: bool = True

    context: BaseContextConfig
    system_prompt_template: str

    tokenizer: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda"

    seed: int = 42
    distributed_backend: Literal["nccl", "gloo"] = "nccl"

    kv_compression: str
    kv_compression_ratio: float

    def run(self):
        return icl_baseline(self)


def icl_baseline(config: ICLBaselineWithKVCacheCompressionConfig):
    with torch.no_grad():
        seed_everything(config.seed)
        is_ddp = "LOCAL_RANK" in os.environ
        if is_ddp:
            assert False
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)

            # SE (03/21): Sometimes, we want to run train multipled times within
            # a single launch of `torchrun`. So, we need to check if the process group
            # if the process group is already initialized, in which case we can reuse it.
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=config.distributed_backend,
                    device_id=torch.device(local_rank),
                )
            logger.info(f"[Rank {dist.get_rank()}] initialized.")
            is_rank_zero = dist.get_rank() == 0
            num_devices = dist.get_world_size()
        else:
            local_rank = config.device
            is_rank_zero = True
            num_devices = 1

        if is_ddp:
            raise NotImplementedError("TODO")

        logger.info(f"ICL will be saved to {config.run_dir}")
        logger.info("Initializing tokenizer and dataset")
        download_wandb_artifacts(config)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        context = (
            config.context.instantiate(tokenizer)
            if isinstance(config.context, BaseSectionedContextConfig)
            else config.context.instantiate()
        )

        system_prompt = config.system_prompt_template.format(
            content=context.text,
        )

        eval_datasets = []

        for dataset_config in config.eval_datasets:
            dataset = EvalDataset(
                dataset=dataset_config.dataset.instantiate(tokenizer=tokenizer),
                batch_size=dataset_config.local_batch_size,
                name=dataset_config.name_for_wandb,
                only_eval_rank_0=dataset_config.only_eval_rank_0,
            )
            eval_datasets.append(dataset)

        model = config.model.instantiate().to(torch.bfloat16).to(local_rank)

        if is_ddp:
            assert False
            logger.info("Wrapping model in DDP")
            logger.info(f"local_rank: {local_rank}")
            model = DDP(model, device_ids=[local_rank])
            dist.barrier()

        # Only set up W&B if rank 0 or running single-process
        if config.wandb is not None and is_rank_zero:
            config.wandb.name = config.name
            prepare_wandb(config.wandb, config.to_dict())

            wandb_log_dict = {
                "num_model_params": sum(p.numel() for p in model.parameters()),
                "num_trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }

            wandb.log(
                wandb_log_dict,
                # SE (03/10): by setting commit=False, we avoid incrementing the step count
                # to 1. Without this, the first evaluation at step 0 will not be logged
                commit=False,
            )

            logger.info(f"Setup wandb with model stats: {wandb_log_dict}")

        logger.info(f"Doing prefill of shared prompt, char_len = {len(system_prompt)}")
        system_prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}]
        )

        press = make_press(press_type=config.kv_compression, ratio=config.kv_compression_ratio)  # type: ignore
        model_with_compressed_cache = ModelWithCompressedCache(
            press=press,
            model=model,
            tokenizer=tokenizer,
            prefix_tokens=system_prompt_tokens,
        )

        for dataset in eval_datasets:
            assert dataset.batch_size == 1, "batch size greater than 1 not supported"
            epoch_loss, epoch_denom = 0.0, 0

            dataloader = DataLoader(
                dataset.dataset,
                batch_size=dataset.batch_size,
                collate_fn=dataset.dataset.collate,
            )

            (
                epoch_num_system_and_user_tokens,
                epoch_num_assistant_tokens,
                epoch_num_elements,
            ) = (0, 0, 0)

            elem_losses = []
            for batch in dataloader:

                epoch_num_system_and_user_tokens += sum(
                    [
                        counts.num_system_and_user_tokens + len(system_prompt_tokens)
                        for counts in batch.token_counts
                    ]
                )
                epoch_num_assistant_tokens += sum(
                    [counts.num_assistant_tokens for counts in batch.token_counts]
                )
                epoch_num_elements += len(batch.token_counts)

                cache = model_with_compressed_cache._copy_cache(model_with_compressed_cache.prefix_cache)
                outputs = model(
                    batch.input_ids.to(local_rank)[:, :-1],
                    past_key_values=cache,
                    use_cache=True,
                )

                labels = batch.labels.to(local_rank)[:, 1:]
                mask = labels != -100

                epoch_loss += nn.functional.cross_entropy(
                    outputs.logits[mask],
                    labels[mask],
                    reduction="sum",
                )

                # compute macro loss
                for elem_logits, elem_labels, elem_mask in zip(
                    outputs.logits, labels, mask, strict=True
                ):
                    elem_loss = nn.functional.cross_entropy(
                        elem_logits[elem_mask],
                        elem_labels[elem_mask],
                        reduction="mean",
                    )
                    elem_losses.append(elem_loss)

                epoch_denom += mask.sum()

            prefix = f"eval_{dataset.name}"
            wandb.log(
                {
                    f"{prefix}/loss": epoch_loss / epoch_denom,
                    f"{prefix}/perplexity": math.exp(epoch_loss / epoch_denom),
                    f"{prefix}/macro_loss": sum(elem_losses) / len(elem_losses),
                    f"{prefix}/macro_perplexity": math.exp(
                        sum(elem_losses) / len(elem_losses)
                    ),
                    f"{prefix}/num_system_and_user_tokens": epoch_num_system_and_user_tokens
                    / epoch_num_elements,
                    f"{prefix}/num_assistant_tokens": epoch_num_assistant_tokens
                    / epoch_num_elements,
                    f"{prefix}/num_elements": epoch_num_elements,
                    "kv_size_bytes": model_with_compressed_cache.kv_size_bytes(),
                },
                step=0,
            )

        # SE (03/21): Careful to synchronize all processes before finishing in case
        # there's another call to `train` after that will reuse the same process group.
        if is_ddp:
            dist.barrier()

        if config.wandb is not None and is_rank_zero:
            wandb.finish()
