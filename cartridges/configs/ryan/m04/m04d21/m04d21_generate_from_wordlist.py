import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient



from capsules.data.mtob import KalamangSectionedConfig
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generators.kalamang.train_examples_or_wordlist import ParallelSentencesOrWordlist
from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.generate_training import GenerateConfig

from pydrantic.variables import FormatStringVariable

from capsules.utils import WandBConfig

file_name = Path(__file__).stem


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(8)],
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    timeout=20 * 60,
)


def config():
    return GenerateConfig(
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=ParallelSentencesOrWordlist.Config(
            question_client=client,
            question_temperature=0.7,
            answer_client=client,
            answer_temperature=0.0,
            answer_max_completion_tokens=384,
            source="wordlist",
        ),
        context=KalamangSectionedConfig(
            max_tokens_per_section=10_000, book_size="long"
        ),
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        num_samples=4000,
        batch_size=100,
        max_num_batches_in_parallel=10,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
