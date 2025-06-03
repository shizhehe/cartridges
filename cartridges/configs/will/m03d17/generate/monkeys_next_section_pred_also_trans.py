import sys
import os
from pathlib import Path
from capsules.data.paths import MONKEYS
from capsules.generate.context_convo_generators.memorization import FairNextSectionPrediction, FairNextSectionPredictionTrans, FairNextSectionPredictionAlsoTrans, NextSectionPrediction
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithChunk,
    AnswerSystemPromptWithEntireContext,
    QuestionSystemPromptWithEntireContext,
)
import pydrantic
from pydrantic.variables import FormatStringVariable

sys.path.append(str(Path(__file__).parent.parent.parent))
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.clients.together import TogetherClient
from capsules.clients.openai import OpenAIClient
from capsules.generate.run import GenerateConfig
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.utils import WandBConfig


client_config = TokasaurusClient.Config(
    url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
    model_name="not_empty"
)


# client_config = TokasaurusClient.Config(
#     url="http://localhost:8012/v1", model_name="not_empty"
# )


file_name = Path(__file__).stem

def trans_config(lang: str):
    return GenerateConfig(
        name=file_name,
        convo_generator=FairNextSectionPredictionTrans.Config(
            answer_client=client_config,
            answer_temperature=0.0,
            answer_max_completion_tokens=256,
            chunk_size_range=(100, 500),
            answer_system_prompt_generator=AnswerSystemPromptWithEntireContext.Config(),
            lang = lang,
        ),
        document_title="Large Language Monkeys: Scaling Inference Compute with Repeated Sampling",
        document_path_or_url=str(MONKEYS.absolute()),
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        # generate config
        num_samples=4096,
        batch_size=128,
        max_num_batches_in_parallel=8,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )

def also_trans_config(lang: str):
    return GenerateConfig(
        name=file_name,
        convo_generator=FairNextSectionPredictionAlsoTrans.Config(
            answer_client=client_config,
            answer_temperature=0.0,
            answer_max_completion_tokens=256,
            chunk_size_range=(100, 500),
            answer_system_prompt_generator=AnswerSystemPromptWithEntireContext.Config(),
        ),
        document_title="Large Language Monkeys: Scaling Inference Compute with Repeated Sampling",
        document_path_or_url=str(MONKEYS.absolute()),
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        # generate config
        num_samples=4096,
        batch_size=128,
        max_num_batches_in_parallel=8,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )
