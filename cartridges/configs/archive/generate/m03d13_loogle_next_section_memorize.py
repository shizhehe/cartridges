import os
from pathlib import Path
from capsules.generate.context_convo_generators.memorization import FairNextSectionPrediction, NextSectionPrediction
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithChunk,
    AnswerSystemPromptWithEntireContext,
    QuestionSystemPromptWithEntireContext,
)
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.clients.together import TogetherClient
from capsules.clients.openai import OpenAIClient
from capsules.generate.run import GenerateConfig
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.utils import WandBConfig


# client_config = TokasaurusClient.Config(
#     url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
#     model_name="not_empty"
# )


client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)


file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=FairNextSectionPrediction.Config(
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=256,
        chunk_size_range=(100, 500),
        answer_system_prompt_generator=AnswerSystemPromptWithEntireContext.Config(),
    ),
    document_title="57th Medical Detachment",
    document_path_or_url=str(
        (
            Path(__file__).parent.parent.parent.parent / "data/loogle/cc4f70a5.txt"
        ).absolute(),
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=4096,
    num_top_logprobs=20,
    batch_size=128,
    max_num_batches_in_parallel=8,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
