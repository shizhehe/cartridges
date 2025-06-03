from __future__ import annotations
from abc import ABC
import json
import os
from pathlib import Path
import math

import pickle
import threading
import time
from typing import List, Literal, Optional, Union

from transformers import AutoTokenizer
import concurrent.futures
from pydrantic import RunConfig, BaseConfig
from pydantic import BaseModel
import pandas as pd
import tqdm
import wandb

from capsules.generate.generators.base import ContextConvoGenerator
from capsules.generate.context_convo_generators.base import (
    ContextConvoGenerator as LegacyContextConvoGenerator,
)
from capsules.utils import WandBConfig, prepare_wandb, get_logger
from capsules.generate.structs import Context, ContextConvoDataset, SectionedContext, TrainingExample
from capsules.generate.run import BaseContextConfig


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

    num_samples: int
    batch_size: int

    max_num_batches_in_parallel: int
    parallelism_strategy: Literal["thread", "process"] = "thread"
    handle_exceptions: bool = True
    thread_timeout: int = 4 * 60  # only allow two minutes between completed batches

    wandb: Optional[WandBConfig] = None
    save_wandb_preview: bool = True
    save_wandb_artifact: bool = True
    save_metadata: bool = True  # metadata is previewed in wandb always, but can exclude from dataset

    run_dir: Optional[Union[Path, str]] = None

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

        if hasattr(convo_generator, "preprocess"):
            convo_generator.preprocess()

        total_batches = math.ceil(self.num_samples / self.batch_size)

        all_rows: list[TrainingExample] = []
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
                    )
                    for batch_idx in range(total_batches)
                ]

                try:
                    for future in tqdm.tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Waiting for requests to complete",
                    ):
                        batch_rows = future.result()
                        all_rows += batch_rows
                except TimeoutError as e:
                    logger.info(f"Timeout error {e}, skipping remaining batches")
        else:
            for batch_idx in tqdm.tqdm(range(total_batches)):
                all_rows += _process_batch(
                    batch_idx=batch_idx,
                    convo_generator=convo_generator,
                    config=self,
                    total_batches=total_batches,
                )

        t = time.time()
        logger.info(f"Generation done, starting to save {len(all_rows)} rows to artifact")

        if self.save_wandb_preview:
            _save_wandb_preview(all_rows)
        
        if not self.save_metadata:
            for row in all_rows:
                row.metadata = {}

        output_dir = self.run_dir / "artifact"
        output_dir.mkdir()
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
                f"Saved dataset to wandb as artifact {artifact.name}, took {time.time() - t:.1f}s"
            )

        wandb.finish()


def _process_batch(
    batch_idx: int,
    total_batches: int,
    convo_generator: ContextConvoGenerator,
    config: GenerateTrainingConfig,
) -> list[TrainingExample]:
    batch_size = min(
        config.batch_size,
        config.num_samples - batch_idx * config.batch_size,
    )

    try:
        convos = convo_generator.sample_convos(batch_idx, batch_size, total_batches)
    except Exception as e:
        if config.handle_exceptions:
            logger.info(f"Error processing batch {batch_idx + 1}/{total_batches}: {e}")
            return []
        else:
            raise e

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
                # SE (05/01): convert this to a string to avoid any wandb bugs
                "metadata": json.dumps(row.metadata, indent=2),
                **{f"message_{i}": row.messages[i].content for i in range(len(row.messages))},
            }
            for row in sampled_rows
        ]
    )
    wandb.log({"preview": preview_df})


# Backwards compatibility
class GenerateConfig(GenerateTrainingConfig):
    pass
