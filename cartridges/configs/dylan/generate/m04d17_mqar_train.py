import os
from pathlib import Path
from capsules.generate.structs import Context, Section
import pydrantic

from capsules.generate.run import GenerateConfig
from capsules.tasks.synthetics.mqar import MQARContextConfig, MQARGenerator
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=MQARGenerator.Config(
        vocab_size=8192,
        num_examples=2048,
        input_seq_len=64,
        num_kv_pairs=16,
        power_a=0.01,
        random_non_queries=False,
        include_slices=True,
    ),
    context=MQARContextConfig(),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=2048,
    batch_size=256,
    max_num_batches_in_parallel=1,
    wandb=WandBConfig(
        project="capsules",
        entity="buffalo-theory",
    ),
)

if __name__ == "__main__":
    pydrantic.main([config])