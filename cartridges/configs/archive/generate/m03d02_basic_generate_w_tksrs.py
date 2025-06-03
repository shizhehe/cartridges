import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate.generate_single_round import BasicGenerateConfig
from capsules.utils import WandBConfig



client_config = TokasaurusClient.Config(
    url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
    model_name=""
)

file_name = Path(__file__).stem
config = BasicGenerateConfig(
    name=file_name,
    question_client=client_config,
    answer_client=client_config,
    document_path_or_url=os.path.join(
        os.environ.get("CAPSULES_DIR"),
        "data/example_docs/minions.txt"
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
   
    # generate config
    num_samples=10,
    chunk_size=500,
    question_temperature=0.6,
    max_answer_tokens=1024,
    max_question_tokens=1024,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
