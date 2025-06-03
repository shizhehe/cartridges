import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.together import TogetherClient
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.make_training_dataset_for_document import GenerateDatasetConfig, GenerateSettings
from capsules.utils import WandBConfig

# Make sure you have a valid API key for Together in your environment (e.g. put it in your ~/.bashrc)
client_config = TogetherClient.Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct-Turbo",
)

file_name = Path(__file__).stem
config = GenerateDatasetConfig(
    question_client=client_config,
    answer_client=client_config,
    document_path_or_url="https://gist.githubusercontent.com/MattIPv4/045239bc27b16b2bcf7a3a9a4648c08a/raw/2411e31293a35f3e565f61e7490a806d4720ea7e/bee%2520movie%2520script",
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    settings=GenerateSettings(
        num_samples=10,
        chunk_size=500,
        question_temperature=0.6,
        max_answer_tokens=1024,
        max_question_tokens=1024,
    ),
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desirinioed.
    pydrantic.main([config])


