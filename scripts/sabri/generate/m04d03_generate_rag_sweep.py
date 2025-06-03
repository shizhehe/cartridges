import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.generate.context_convo_generators.programmatic import NTPContextConvoGenerator
from cartridges.tasks.finance import FinanceBenchContextConfig
from cartridges.generate.context_convo_generators.rag import RAGContextConvoGenerator, BM25Retriever
from cartridges.generate.run import GenerateConfig
from cartridges.utils import WandBConfig


file_name = Path(__file__).stem

DOC_NAME = "AMD_2022_10K"

from cartridges.clients.openai import OpenAIClient
from cartridges.clients.tokasaurus import TokasaurusClient

client = TokasaurusClient.Config(
    # url="https://hazyresearch--tokasaurus-llama-3b-8xh100-min0-serve.modal.run/v1",
    url="http://localhost:8888/v1",
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
- "Who are the top 5 customers of the company?"
- "Translate the company's description of the Instagram acquisition into French." 

Only generate a single chat message, and do not include any other text or formatting."""

USER_PROMPT_TEMPLATE_NO_EXAMPLES = """<context>
{context}
</context>

Please generate a single question/instruction related to the context.
Only generate a single chat message, and do not include any other text or formatting."""


ASSISTANT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that follows instructions, relying on retrieved context. 
<retrieved_context>
{context}
</retrieved_context> """

ASSISTANT_PROMPT_TEMPLATE = """{instruction}"""


for max_tokens_per_chunk in [512, 1024, 2048]:
    for top_k in [4, 8, 16]:
        config = GenerateConfig(
            name=FormatStringVariable(
                f"{file_name}_{DOC_NAME}_k{top_k}_max{max_tokens_per_chunk}_no_examples"
            ),
            convo_generator=RAGContextConvoGenerator.Config(
                assistant_client=client,
                user_client=client,
                retriever=BM25Retriever.Config(
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    top_k=top_k
                ),
                assistant_system_prompt_template=ASSISTANT_SYSTEM_PROMPT_TEMPLATE,
                assistant_prompt_template=ASSISTANT_PROMPT_TEMPLATE,
                user_system_prompt_template=USER_SYSTEM_PROMPT_TEMPLATE,
                user_prompt_template=USER_PROMPT_TEMPLATE_NO_EXAMPLES,
                assistant_max_completion_tokens=1024,
                user_max_completion_tokens=128,     
            ),
            context=FinanceBenchContextConfig(doc_names=[DOC_NAME], split_on="page"),
            output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
            num_samples=32_768,
            batch_size=64,
            max_num_batches_in_parallel=128,
            wandb=WandBConfig(
                project="cartridges",
                entity="hazy-research",
            ),
        )


if __name__ == "__main__":
    pydrantic.main(config)
