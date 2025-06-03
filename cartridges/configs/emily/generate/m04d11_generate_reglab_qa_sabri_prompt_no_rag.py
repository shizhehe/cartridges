import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient

from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.run import GenerateConfig
from capsules.generate.subcontext import RandomizedSlidingWindowGenerator
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.utils import WandBConfig

answer_client_config = TokasaurusClient.Config(
    url="https://hazyresearch--tokasaurus-llama-3b-8xh100-min0-serve.modal.run/v1",
    model_name="llama-3.2-3b-instruct"  # this argument does not matter
)

client = TokasaurusClient.Config(
    url="https://hazyresearch--tokasaurus-llama-3b-8xh100-min0-serve.modal.run/v1",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


STATE = "Alabama"


QUESTION_PROMPT_TEMPLATE = """<context>
{chunk}
</context>

Please generate a single question/instruction related to the context.

Here are some examples of the type of question/instruction you should generate: 
- "Rewrite the statute in simpler language."
- "Summarize the state's policy on data privacy."
- "List the three ways a company can be sued for a breach of contract."
- "Translate the state's regulations of the sale of goods into French." 

Only generate a single chat message, and do not include any other text or formatting."""

QUESTION_SYSTEM_PROMPT = f"""You will be given part of {STATE}'s legal code, and you should generate a single chat message that 
instructs the model to do something related to the context.

Only generate a single chat message, and do not include any other text or formatting.
"""


ANSWER_SYSTEM_PROMPT = f"""You are a helpful assistant that follows instructions, relying on the following information from the {STATE} state statutes to answer the question.

<excerpt>
{{subcontext}}
</excerpt>
Act as if you knew this information off the top of your head (i.e. do not say things like "Based on the retrieved context...").
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""

file_name = Path(__file__).stem

STATE = "Alabama"

config = GenerateConfig(
        name=FormatStringVariable(f"{file_name}_{STATE.lower()}_n{{num_samples}}_nxstatutes{{context.num_extra_statutes}}"),
        convo_generator=SimpleQuestionFromChunk.Config(
            question_client=client,
            question_temperature=0.9,
            question_max_completion_tokens=512,
            answer_client=client,
            answer_temperature=0.0,
            answer_max_completion_tokens=384,
            chunker=SimpleCharacterChunker.Config(
                min_chunk_size_in_chars=500,
                max_chunk_size_in_chars=10_000,
            ),
            question_prompt_template=QUESTION_PROMPT_TEMPLATE,
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            question_system_prompt_template=QUESTION_SYSTEM_PROMPT,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
            subcontext_generator=RandomizedSlidingWindowGenerator.Config(
                num_passes=10,
                min_window_size=2,
                max_window_size=4,
                min_step_size=3,
                max_step_size=5,
                seed=32,
            ),
        ),
        context=HousingStatutesContextConfig(
            states=[STATE],
            split_on="statute",
            num_extra_statutes=1024,
        ),
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        # generate config
        num_samples=8192,
        batch_size=256,
        max_num_batches_in_parallel=16,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
        previous_run_dir=Path(
            "/home/emilyryliu/capsules-outputs/2025-04-12-05-02-03-m04d11_generate_reglab_qa_sabri_prompt_no_rag/4ac7c2f0-e3cc-476d-ac7c-ca4e82b851ac-1-2-3-4-5-6-7"
        )
    )


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config] * 8)
