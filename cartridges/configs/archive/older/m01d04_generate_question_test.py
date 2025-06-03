import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable
from capsules.tasks.finance.questions import GenerateConfig, OpenAIGenerator
from capsules.utils import WandBConfig


file_name = Path(__file__).stem
config = GenerateConfig(
    name=FormatStringVariable(f"{file_name}"), 
    generator=OpenAIGenerator.Config(
        model_name="gpt-4o"
    ),
    samples_per_chunk=16,
    chunk_sizes=[4096], 
    temperature=0.2,
    output_dir=os.environ["capsules_OUTPUT_DIR"],
)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
