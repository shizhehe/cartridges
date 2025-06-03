from __future__ import annotations
from abc import ABC
import os
from pathlib import Path
import math

import pickle
import time
from typing import Literal, Optional

from transformers import AutoTokenizer
import concurrent.futures
from pydrantic import RunConfig
import pandas as pd
import tqdm
import wandb

from cartridges.generate.generators.base import ContextConvoGenerator
from cartridges.utils import WandBConfig, prepare_wandb, get_logger
from cartridges.structs import SectionedContext, TrainingExample
from cartridges.context import BaseContextConfig


logger = get_logger(__name__)

class BaseSectionedContextConfig(BaseContextConfig, ABC):
    """This should be subclassed by different tasks to specify the parameters
    and method for instantiating a Context object.
    For example, see LongHealthContextConfig in tasks/longhealth/__init__.py
    """

    max_tokens_per_section: int
    def instantiate(self, tokenizer) -> SectionedContext:
        raise NotImplementedError("Subclasses must implement this method")


class GenerateTrainingConfig(RunConfig):
    name: Optional[str] = "generate"

    context: BaseContextConfig
    tokenizer: Optional[str] = None

    convo_generator: ContextConvoGenerator.Config

    max_num_batches_in_parallel: int
    num_samples: int
    batch_size: int

    num_samples_stage_1: int
    batch_size_stage_1: int
    max_num_batches_in_parallel_stage_1: int

    parallelism_strategy: Literal["thread", "process"] = "thread"
    handle_exceptions: bool = True

    wandb: Optional[WandBConfig] = None
    save_wandb_preview: bool = True
    save_wandb_artifact: bool = True

    run_dir: Optional[Path] = None

    # TODO: overriding run_dir and not output_dir doesn't work
    previous_run_dir: Optional[Path] = None

    def run(self):
        assert self.name is not None

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        assert self.run_dir is not None
        self.run_dir = Path(self.run_dir)
        logger.info(f"Generating dataset with run dir: {self.run_dir}")

        if self.wandb is not None:
            self.wandb.name = self.name
            prepare_wandb(self.wandb, self.to_dict())

        if isinstance(self.context, BaseSectionedContextConfig):
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
            context = self.context.instantiate(tokenizer)
        else:
            context = self.context.instantiate()

        logger.info(f"Instantiating convo generator...")
        convo_generator = self.convo_generator.instantiate(context=context)
        total_batches = math.ceil(self.num_samples / self.batch_size)

        # Stage 1
        all_rows = []
        total_batches_stage_1 = math.ceil(self.num_samples_stage_1 / self.batch_size_stage_1)
        if self.max_num_batches_in_parallel_stage_1 > 1:

            with (
                concurrent.futures.ThreadPoolExecutor
                if self.parallelism_strategy == "thread"
                else concurrent.futures.ProcessPoolExecutor
            )(max_workers=self.max_num_batches_in_parallel) as executor:
                futures = [
                    executor.submit(
                        _process_batch,
                        batch_idx=batch_idx,
                        total_batches=total_batches_stage_1,
                        convo_generator=convo_generator,
                        stage=1,
                        config=self,
                    )
                    for batch_idx in range(total_batches_stage_1)
                ]

                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Waiting for requests to complete in stage 1",
                ):
                    batch_rows = future.result()
                    all_rows += batch_rows
        else:
            for batch_idx in tqdm.tqdm(range(total_batches_stage_1)):
                all_rows += _process_batch(
                    batch_idx=batch_idx,
                    convo_generator=convo_generator,
                    config=self,
                    stage=1,
                    total_batches=total_batches,
                )

        cleaned_rows = convo_generator.stage_1_postprocess(all_rows)

        # Stage 2
        all_rows = []
        total_batches = math.ceil(self.num_samples / self.batch_size)
        if self.max_num_batches_in_parallel > 1:

            with (
                concurrent.futures.ThreadPoolExecutor
                if self.parallelism_strategy == "thread"
                else concurrent.futures.ProcessPoolExecutor
            )(max_workers=self.max_num_batches_in_parallel) as executor:
                futures = [
                    executor.submit(
                        _process_batch,
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        convo_generator=convo_generator,
                        config=self,
                        stage=2,
                        prior_stage_output=cleaned_rows,
                    )
                    for batch_idx in range(total_batches)
                ]

                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Waiting for requests to complete in stage 2",
                ):
                    batch_rows = future.result()
                    all_rows += batch_rows
        else:
            for batch_idx in tqdm.tqdm(range(total_batches)):
                all_rows += _process_batch(
                    batch_idx=batch_idx,
                    convo_generator=convo_generator,
                    config=self,
                    total_batches=total_batches,
                    stage=2,
                    prior_stage_output=cleaned_rows,
                )


        # saving the dataset
        logger.info("All data done, starting to save artifact")

        if self.save_wandb_preview:
            _save_wandb_preview(all_rows)

        output_dir = Path(f"/data/simran/{self.name}") / "artifact"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        final_output_path = output_dir / "dataset.pkl"
        with open(final_output_path, "wb") as f:
            pickle.dump(
                {
                    "rows": all_rows,
                    "context": context,
                },
                f,
            )
        logger.info(f"Final output saved to {final_output_path}")

        if self.save_wandb_artifact:
            artifact = wandb.Artifact(name=self.name, type="dataset")
            artifact.add_dir(local_path=str(output_dir.absolute()), name="dataset")
            wandb.log_artifact(artifact)

            # important to wait for the artifact to be saved so we get the version
            artifact.wait()
            logger.info(
                f"Saved dataset to wandb as artifact {artifact.name}"
            )

        wandb.finish()


def _process_batch(
    batch_idx: int,
    total_batches: int,
    convo_generator: ContextConvoGenerator,
    config: GenerateTrainingConfig,
    stage: int = 1,
    prior_stage_output = None,
) -> list[TrainingExample]:
    batch_size = min(
        config.batch_size,
        config.num_samples - batch_idx * config.batch_size,
    )
    if stage == 1: 
        convos = convo_generator.sample_convos_stage_1(
            batch_idx, batch_size, total_batches
        )
    else: 
        convos = convo_generator.sample_convos_stage_2(
            batch_idx, batch_size, total_batches, prior_stage_output
        )
    return convos


def _save_wandb_preview(rows: list[TrainingExample]):
    import random
    sampled_rows = random.sample(rows, min(256, len(rows)))
    preview_df = pd.DataFrame(
        [
            {   
                "type": row.type,
                "num_output_tokens": row.num_output_tokens,
                "type": row.type,
                "metadata": row.metadata,
                **{f"message_{i}": row.messages[i].content for i in range(len(row.messages))},
            }
            for row in sampled_rows
        ]
    )
    wandb.log({"preview": preview_df})


# Backwards compatibility
class GenerateConfig(GenerateTrainingConfig):
    pass
