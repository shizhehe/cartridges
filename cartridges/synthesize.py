from __future__ import annotations
from abc import ABC
import asyncio
import json
import os
from pathlib import Path
import math

import pickle
import time
from typing import Literal, Optional, Union

import concurrent.futures
from pydrantic import RunConfig
import pandas as pd
import tqdm
import wandb

from cartridges.contexts.mcp.base import MCPContext
from cartridges.synthesizers.base import AsyncConvoSynthesizer, ConvoSynthesizer
from cartridges.utils import WandBConfig, prepare_wandb, get_logger
from cartridges.structs import TrainingExample
from cartridges.context import BaseContextConfig


logger = get_logger(__name__)


class SynthesizeConfig(RunConfig):
    name: Optional[str] = "generate"

    context: BaseContextConfig
    tokenizer: Optional[str] = None

    synthesizer: ConvoSynthesizer.Config

    num_samples: int
    batch_size: int

    max_num_batches_in_parallel: int
    parallelism_strategy: Literal["thread", "process", "async"] = "thread"
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

        total_batches = math.ceil(self.num_samples / self.batch_size)

        all_rows: list[TrainingExample] = []
        if self.parallelism_strategy == "async":
            all_rows = asyncio.run(
                self._run_async_batches_with_queue(total_batches=total_batches)
            )
        else:
            context = self.context.instantiate()

            logger.info(f"Instantiating convo generator...")
            synthesizer = self.synthesizer.instantiate(context=context)

            if hasattr(synthesizer, "preprocess"):
                synthesizer.preprocess()
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
                            synthesizer=synthesizer,
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
                        synthesizer=synthesizer,
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
                    "context": self.context.to_dict(),
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

    async def _run_async_batches_with_queue(
        self, 
        total_batches: int
    ) -> list[TrainingExample]:
        """Run batches using a queue for better control."""
        all_rows: list[TrainingExample] = []
        
        # Create queue of batch indices
        batch_queue = asyncio.Queue()
        for batch_idx in range(total_batches):
            batch_queue.put_nowait(batch_idx)
        
        # Results queue
        results_queue = asyncio.Queue()
        
        async def worker(worker_id: int):
            """Worker that processes batches from the queue."""

            context = self.context.instantiate()
            logger.info(f"Instantiating convo generator...")
            synthesizer = self.synthesizer.instantiate(context=context)
            if hasattr(synthesizer, "preprocess"):
                synthesizer.preprocess()
            if isinstance(context, MCPContext):
                await context.connect_to_server()
                
            while True:
                try:
                    # Get batch with timeout
                    batch_idx = await asyncio.wait_for(
                        batch_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No more batches
                    break
                
                try:
                    batch_rows = await _process_batch_async(
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        synthesizer=synthesizer,
                        config=self,
                    )
                    await results_queue.put((batch_idx, batch_rows))
                except Exception as e:
                    if self.handle_exceptions:
                        logger.error(f"Worker {worker_id} error on batch {batch_idx}: {e}")
                        await results_queue.put((batch_idx, []))
                    else:
                        raise
                finally:
                    batch_queue.task_done()
            
            # SE (06/18): This is important for avoiding the "Attempted to exit cancel 
            # scope in a different task than it was entered in" error.
            await context.exit_stack.aclose()
        
        # Start workers
        workers = [
            asyncio.create_task(worker(i))
            for i in range(self.max_num_batches_in_parallel)
        ]
        
        # Collect results with progress tracking
        completed_batches = 0
        with tqdm.tqdm(total=total_batches, desc="Processing batches") as pbar:
            while completed_batches < total_batches:
                try:
                    batch_idx, batch_rows = await asyncio.wait_for(
                        results_queue.get(),
                        timeout=self.thread_timeout
                    )
                    all_rows.extend(batch_rows)
                    completed_batches += 1
                    pbar.update(1)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for batch results")
                    break
        
        # Wait for all workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        
        return all_rows

async def _process_batch_async(
    batch_idx: int,
    total_batches: int,
    synthesizer: AsyncConvoSynthesizer,
    config: SynthesizeConfig,
) -> list[TrainingExample]:
    batch_size = min(
        config.batch_size,
        config.num_samples - batch_idx * config.batch_size,
    )

    try:
        convos = await synthesizer.sample_convos(batch_idx, batch_size, total_batches)
    except Exception as e:
        if config.handle_exceptions:
            logger.info(f"Error processing batch {batch_idx + 1}/{total_batches}: {e}")
            return []
        else:
            raise e

    return convos


def _process_batch(
    batch_idx: int,
    total_batches: int,
    synthesizer: ConvoSynthesizer,
    config: SynthesizeConfig,
) -> list[TrainingExample]:
    batch_size = min(
        config.batch_size,
        config.num_samples - batch_idx * config.batch_size,
    )

    try:
        convos = synthesizer.sample_convos(batch_idx, batch_size, total_batches)
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
