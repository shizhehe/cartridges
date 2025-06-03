from dataclasses import dataclass, field
import itertools
import time
from typing import Optional

import os

import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer
import wandb

import pydrantic


# capsules specific imports
from capsules.datasets import (
    CapsuleDataset,
    CapsuleGenerateDataset,
)
from capsules.config import ModelConfig

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


class EvaluateRAGBaselineConfig(pydrantic.RunConfig):
    name: str = "default"  # A name for the run for wandb
    model: ModelConfig
    wandb: Optional[WandBConfig] = None

    # dataset for evaluating perplexity on other generations
    # NOTE: steps here is the number of **optimizer steps**, which we keep track of
    # with the `optimizer_step` variable. This is different than the number of batches
    # processed, which is given by `iter_idx`.
    eval_datasets: list[EvalDatasetConfig] = field(default_factory=list)
    eval_log_table: bool = True

    # dataset for actually producing generations
    # NOTE: steps here is the number of **optimizer steps**, which we keep track of
    # with the `optimizer_step` variable. This is different than the number of batches
    # processed, which is given by `iter_idx`.
    generate_datasets: list[GenerateDatasetConfig] = field(default_factory=list)
    generate_max_new_tokens: int = 128

    context: pydrantic.BaseConfig
    system_prompt_template: str

    # the global batch size is the total batch size across all devices and gradient
    # accumulation steps

    tokenizer: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda"

    seed: int = 42

    def run(self):
        return main(self)


def get_dataset_names(
    config: CapsuleDataset.Config | CapsuleGenerateDataset.Config,
) -> list[str]:
    return (
        [artifact_name for (artifact_name, _) in config.data_sources]
        if config.is_wandb
        else []
    )


# def download_wandb_artifacts(config: EvaluateRAGBaselineConfig):
#     artifact_names = []

#     for eval_or_gen_ds in itertools.chain(
#         config.eval_datasets, config.generate_datasets
#     ):
#         if isinstance(
#             eval_or_gen_ds.dataset,
#             (CapsuleDataset.Config, CapsuleGenerateDataset.Config),
#         ):
#             artifact_names += get_dataset_names(eval_or_gen_ds.dataset)

#     download_artifacts(artifact_names)


def main(config: EvaluateRAGBaselineConfig):
    with torch.no_grad():
        seed_everything(config.seed)
        is_ddp = "LOCAL_RANK" in os.environ
        if is_ddp:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)

            # SE (03/21): Sometimes, we want to run train multipled times within
            # a single launch of `torchrun`. So, we need to check if the process group
            # if the process group is already initialized, in which case we can reuse it.
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl", device_id=torch.device(local_rank)
                )
            logger.info(f"[Rank {dist.get_rank()}] initialized.")
            is_rank_zero = dist.get_rank() == 0
            num_devices = dist.get_world_size()
        else:
            local_rank = config.device
            is_rank_zero = True
            num_devices = 1

        logger.info(f"Train outputs will be saved to {config.run_dir}")
        logger.info("Initializing tokenizer and dataset")
        # download_wandb_artifacts(config)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        context = config.context.instantiate()

        system_prompt = config.system_prompt_template.format(
            title=context.title,
            content=context.to_string(),
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

        generate_datasets = []

        for generate_dataset_config in config.generate_datasets:
            dataset = GenerateDataset(
                name=generate_dataset_config.name_for_wandb,
                dataset=generate_dataset_config.dataset.instantiate(
                    tokenizer=tokenizer
                ),
            )
            generate_datasets.append(dataset)

        model = config.model.instantiate().to(torch.bfloat16).to(local_rank)

        if is_ddp:
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

        for dataset in eval_datasets:
            evaluate(
                config=config,
                model=model,
                cache=None,
                eval_dataset=dataset,
                optimizer_step=0,
                epoch=0,
                local_rank=local_rank,
                cache_tuning=False,
                dont_use_lm_head_optimization=True,
            )

        # for generate_dataset in generate_datasets:
        #     evaluate_generations(
        #         config=config,
        #         model=model,
        #         tokenizer=tokenizer,
        #         dataset=generate_dataset,
        #         optimizer_step=0,
        #         local_rank=local_rank,
        #         step=0,
        #     )

        # SE (03/21): Careful to synchronize all processes before finishing in case
        # there's another call to `train` after that will reuse the same process group.
        if is_ddp:
            dist.barrier()

        if config.wandb is not None and is_rank_zero:
            wandb.finish()
