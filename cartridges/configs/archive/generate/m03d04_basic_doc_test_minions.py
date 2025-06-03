import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.openai import OpenAIClient
from capsules.generate.generate_single_round import BasicGenerateConfig
from capsules.utils import WandBConfig


client_config = OpenAIClient.Config(model_name="gpt-4o")

file_name = Path(__file__).stem

config = BasicGenerateConfig(
    name=file_name,
    question_client=client_config,
    answer_client=client_config,
    document_path_or_url=str(
        (
            Path(__file__).parent.parent.parent.parent / "data/example_docs/minions.txt"
        ).absolute()
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    top_logprobs=20,
    num_samples=256,
    chunk_size=(500, 10_000),
    question_temperature=0.65,
    max_answer_tokens=512,
    max_question_tokens=128,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
