import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.together import TogetherClient
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.data.generate_basic import BasicGenerateConfig

client_config = TokasaurusClient.Config(
    port=8001,
    model_name=""
)

# client_config = TogetherClient.Config(
#     model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
# )

file_name = Path(__file__).stem
config = BasicGenerateConfig(
    question_client=client_config,
    answer_client=client_config,
    document_path_or_url="https://gist.githubusercontent.com/MattIPv4/045239bc27b16b2bcf7a3a9a4648c08a/raw/2411e31293a35f3e565f61e7490a806d4720ea7e/bee%2520movie%2520script",
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
   
    # generate config
    num_samples=10,
    chunk_size=500,
    question_temperature=0.6,
    max_answer_tokens=1024,
    max_question_tokens=1024,
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
