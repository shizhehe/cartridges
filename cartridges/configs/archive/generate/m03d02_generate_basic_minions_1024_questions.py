import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.together import TogetherClient
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate.generate_single_round import BasicGenerateConfig

# Make sure you have a valid API key for Together in your environment (e.g. put it in your ~/.bashrc)
together_config = TogetherClient.Config(
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    show_progress_bar=True,
)
tok_client = TokasaurusClient.Config(
    port=8001,
    model_name="3B-but-this-field-is-ignorgned",
)

data_file = (
    Path(__file__).parent.parent.parent.parent / "data" / "example_docs" / "minions.txt"
)
config = BasicGenerateConfig(
    question_client=together_config,
    answer_client=tok_client,
    document_path_or_url=str(data_file.absolute()),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=1024,
    chunk_size=500,
    question_temperature=0.6,
    max_answer_tokens=1024,
    max_question_tokens=1024,
)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
