import os
from pathlib import Path


import pydrantic


from capsules.clients.tokasaurus import TokasaurusClient

from capsules.configs.ryan.m03d27.doc_names import DOC_NAMES
from capsules.generate.context_convo_generators.anthropic_finance_bench import FinanceBenchClaudeSonnet37Amd
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPrompt,
    QuestionSystemPrompt,
)
from capsules.generate.run import GenerateConfig
from capsules.generate.context_convo_generators.basic_question_from_chunk import (
    SimpleQuestionFromChunk,
)
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.tasks.finance import FinanceBenchContextConfig
from capsules.utils import WandBConfig

file_name = Path(__file__).stem


def config(doc_name):
    assert doc_name in DOC_NAMES

    return GenerateConfig(
        name=f"m03d27_amd10k_sonnet37_manual_eval",
        convo_generator=FinanceBenchClaudeSonnet37Amd.Config(),
        context=FinanceBenchContextConfig(doc_names=[doc_name], force_single_doc=True),
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        # generate config
        num_samples=48,
        batch_size=48,
        max_num_batches_in_parallel=1,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main(config(DOC_NAMES[0]))
