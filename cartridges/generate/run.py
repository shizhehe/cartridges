from __future__ import annotations
from abc import ABC
import os
from pathlib import Path
import math

from typing import List, Literal, Optional

import concurrent.futures
from pydrantic import RunConfig, BaseConfig
from pydantic import BaseModel
import pandas as pd
import wandb

from capsules.generate.generators.base import ContextConvoGenerator
from capsules.generate.context_convo_generators.base import ContextConvoGenerator as LegacyContextConvoGenerator
from capsules.utils import WandBConfig, prepare_wandb, get_logger
from capsules.generate.structs import Context, ContextConvoDataset
from capsules.context import BaseContextConfig


logger = get_logger(__name__)




class GenerateConfig(RunConfig):
    name: Optional[str] = "generate"

    # SE(03/12): Either pass document_path_or_url or context
    context: Optional[BaseContextConfig] = None
    document_title: Optional[str] = None
    document_path_or_url: Optional[str] = None

    convo_generator: ContextConvoGenerator.Config | LegacyContextConvoGenerator.Config

    # SE(04/01): if num_samples is None, we will determine the number of samples
    # by calling `len(convo_generator)`.
    num_samples: Optional[int]
    batch_size: int

    max_num_batches_in_parallel: int
    parallelism_strategy: Literal["thread", "process"] = "thread"
    handle_exceptions: bool = True

    wandb: Optional[WandBConfig] = None
    save_wandb_preview: bool = True

    run_dir: Optional[Path] = None

    # TODO: overriding run_dir and not output_dir doesn't work
    previous_run_dir: Optional[Path] = None

    def run(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        assert self.run_dir is not None
        self.run_dir = Path(self.run_dir)
        logger.info(f"Generating dataset with run dir: {self.run_dir}")

        if self.wandb is not None:
            self.wandb.name = self.name
            prepare_wandb(self.wandb, self.to_dict())

        if self.context is not None:
            assert self.document_path_or_url is None and self.document_title is None
            context = self.context.instantiate()
        else:
            assert self.context is None
            context = Context.from_path_or_url(
                self.document_path_or_url, title=self.document_title
            )

        logger.info(f"Instantiating convo generator...")
        convo_generator = self.convo_generator.instantiate(context=context)

        if self.num_samples is None:
            self.num_samples = len(convo_generator)
        
        total_batches = math.ceil(self.num_samples / self.batch_size)

        all_rows = []
        if self.max_num_batches_in_parallel > 1:
            if self.parallelism_strategy == "thread":
                executor_cls = concurrent.futures.ThreadPoolExecutor
            elif self.parallelism_strategy == "process":
                executor_cls = concurrent.futures.ProcessPoolExecutor

            with executor_cls(
                max_workers=self.max_num_batches_in_parallel
            ) as executor:
                futures = [
                    executor.submit(
                        _process_batch,
                        batch_idx=batch_idx,
                        convo_generator=convo_generator,
                        total_batches=total_batches,
                        config=self
                    ) for batch_idx in range(total_batches)]
                for future in concurrent.futures.as_completed(futures):
                    batch_rows = future.result()
                    all_rows += batch_rows
        else: # useful for debugging to run serially 
            for batch_idx in range(total_batches):
                all_rows += _process_batch(batch_idx, convo_generator, total_batches, self)

        logger.info("Dataset finished, loading individual components and saving to disk")
        dataset = ContextConvoDataset(context=context, rows=all_rows, config=self)
        path = self.run_dir / "dataset.feather"
        dataset.save(path, to_wandb=self.wandb is not None, wandb_name=self.name, save_wandb_preview=self.save_wandb_preview)
        logger.info(f"Wrote dataset to {os.path.abspath(path)}")


        wandb.finish()



def _process_batch(batch_idx: int, convo_generator: ContextConvoGenerator, total_batches: int, config: GenerateConfig) -> str:
    path = config.run_dir / f"batch_{batch_idx}.feather"
    batch_size = min(
        config.batch_size, config.num_samples - batch_idx * config.batch_size
    )

    if config.previous_run_dir is not None:
        existing_path = config.previous_run_dir / f"batch_{batch_idx}.feather"
        if existing_path.exists():
            print(f"Batch {batch_idx + 1}/{total_batches} exists")
            rows = ContextConvoDataset.load_rows(existing_path)
            # ContextConvoDataset.save_rows(rows, path)
            # assert len(rows) == batch_size
            return rows
    

    logger.info(
        f"Processing batch {batch_idx + 1}/{total_batches} with {batch_size} questions"
    )

    try:
        convos = convo_generator.sample_convos(batch_idx, batch_size)
    except Exception as e:
        if config.handle_exceptions:
            logger.error(f"Error processing batch {batch_idx + 1}/{total_batches}: {e}")
            return []
        else:
            raise e

    ContextConvoDataset.save_rows(convos, path)
    return convos
