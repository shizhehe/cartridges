import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient



from capsules.data.mtob import KalamangSectionedConfig
from capsules.generate.generators.test_examples import TestExamples
from capsules.generate.generators.training_examples import TrainingExamples
from capsules.generate.run_new import GenerateConfig

from pydrantic.variables import FormatStringVariable


from capsules.generate.subcontext import (
    RandomSubcontextGenerator,
    RandomizedSlidingWindowGenerator,
)

from capsules.tasks.finance import (
    FinanceBenchContextConfig,
    FinanceBenchSectionedContextConfig,
)
from capsules.utils import WandBConfig


file_name = Path(__file__).stem


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(8)],
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)


ANSWER_SYSTEM_PROMPT_TEMPLATE = """Please use the following information from the document to answer the user's question.

<title>
{title}
</title>

Here is the document:
{context}
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""


def config():
    return GenerateConfig(
        
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=TestExamples.Config(
            answer_client=client,
            answer_temperature=0.0,
            answer_max_completion_tokens=384,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT_TEMPLATE,
        ),
        context=KalamangSectionedConfig(
            max_tokens_per_section=100_000, book_size="medium"
        ),
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        num_samples=1,
        batch_size=1,
        max_num_batches_in_parallel=1,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
