import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable
from capsules.tasks.finance.questions import GenerateConfig, SGLangGenerator
from capsules.utils import WandBConfig


file_name = Path(__file__).stem
config = GenerateConfig(
    name=FormatStringVariable(f"{file_name}"), 
    generator=SGLangGenerator.Config(
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    ),
    samples_per_chunk=128,
    chunk_sizes=[512, 1024, 2048], 
    temperature=0.4,
    output_dir=os.environ["capsules_OUTPUT_DIR"],
)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
