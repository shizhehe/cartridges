import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate.generators.rag import RAGContextConvoGenerator
from capsules.retrievers import BM25Retriever
from capsules.generate.run import GenerateConfig
from capsules.tasks.finance import FinanceBenchContextConfig
from capsules.utils import WandBConfig


file_name = Path(__file__).stem

DOC_NAME = "AMD_2022_10K"


client = TokasaurusClient.Config(
    url="https://hazyresearch--tokasaurus-llama-3b-8xh100-min0-serve.modal.run/v1",
    model_name="llama3.2-3b-instruct",
)

# client = OpenAIClient.Config(model_name="gpt-4o-mini"),

USER_SYSTEM_PROMPT_TEMPLATE = """You will be given some context from a financial document, and you should generate a single chat message that 
instructs the model to do something related to the context.

Only generate a single chat message, and do not include any other text or formatting.
"""
USER_PROMPT_TEMPLATE = """<context>
{context}
</context>

Please generate a single question/instruction related to the context.

Here are some examples of the type of question/instruction you should generate: 
- "Tell me more about the company's revenue growth in FY2022."
- "Summarize the company's policy on data privacy."
- "List the top 5 customers of the company?"
- "Translate the company's description of the Instagram acquisition into French." 

Only generate a single chat message, and do not include any other text or formatting."""


ASSISTANT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that follows instructions, relying on retrieved context. 
<context>
{context}
</context> 
Act as if you knew this information off the top of your head (i.e. do not say things like "Based on the retrieved context...").
"""

ASSISTANT_PROMPT_TEMPLATE = """{instruction}"""

generator = RAGContextConvoGenerator.Config(
    assistant_client=client,
    user_client=client,
    retriever=BM25Retriever.Config(
        max_tokens_per_chunk=256,
    ),
    retrieve_top_k=4,
    assistant_system_prompt_template=ASSISTANT_SYSTEM_PROMPT_TEMPLATE,
    assistant_prompt_template=ASSISTANT_PROMPT_TEMPLATE,
    user_system_prompt_template=USER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=USER_PROMPT_TEMPLATE,
    assistant_max_completion_tokens=1024,
    user_max_completion_tokens=128,
    num_tokens_in_user_context=1024
)

config = GenerateConfig(
    name=FormatStringVariable(f"{file_name}_{DOC_NAME}_n{{num_samples}}"),
    convo_generator=generator,
    context=FinanceBenchContextConfig(doc_names=[DOC_NAME], split_on="page"),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=128,
    max_num_batches_in_parallel=32,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
    previous_run_dir=Path("/Users/sabrieyuboglu/code/capsules/output/2025-04-05-16-03-20-m04d04_generate_rag/c0b576d5-57e5-4318-a092-3942021c1e44"),
)


if __name__ == "__main__":
    pydrantic.main(config)
