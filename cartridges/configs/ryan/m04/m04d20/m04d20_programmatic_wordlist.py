import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient



from capsules.data.mtob import KalamangSectionedConfig
from capsules.generate.generators.kalamang_grammar import KalamangWordlistGenerator
from capsules.generate.generators.training_examples import TrainingExamples
from capsules.generate.generate_training import GenerateConfig

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




def config():
    return GenerateConfig(
        
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=KalamangWordlistGenerator.Config(
            tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        ),
        context=KalamangSectionedConfig(
            max_tokens_per_section=100_000, book_size="medium"
        ),
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
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
