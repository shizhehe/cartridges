import os
from pathlib import Path


import pydrantic


from capsules.clients.tokasaurus import TokasaurusClient

from capsules.configs.ryan.m03d27.doc_names import DOC_NAMES
from capsules.generate.context_convo_generators.memorization import FairSectionLocator
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
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.utils import WandBConfig

answer_client_config = TokasaurusClient.Config(
    url="https://hazyresearch--tokasaurus-llama-3b-8xh100-min0-serve.modal.run/v1",
    model_name="llama-3.2-3b-instruct"  # this argument does not matter
)

ANSWER_SYSTEM_PROMPT = """Please use the information in the following document to answer the user's questions.
<title>
{title}
</title>

<document>
{context}
</document>


First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""

file_name = Path(__file__).stem

STATE = "Alabama"

config = GenerateConfig(
        name=f"housing_qa_train_mem_shorter_chunks_data_{STATE.lower()}",
        convo_generator=FairSectionLocator.Config(
            answer_client=answer_client_config,
            answer_temperature=0.0,
            answer_max_completion_tokens=512,
            chunk_size_range=(250, 1000),
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            answer_system_prompt_generator=AnswerSystemPrompt.Config(
                prompt_template=ANSWER_SYSTEM_PROMPT,
            ),
        ),
        context=HousingStatutesContextConfig(states=[STATE]),
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        # generate config
        num_samples=10,  # 8192
        batch_size=2,
        max_num_batches_in_parallel=8,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    doc_name = "AMD_2022_10K"
    pydrantic.main(config)
