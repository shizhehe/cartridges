import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate.generators.rag import RAGContextConvoGenerator
from capsules.retrievers import BM25Retriever
from capsules.generate.run import GenerateConfig
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.utils import WandBConfig


file_name = Path(__file__).stem

# client = TokasaurusClient.Config(
#     url="https://hazyresearch--tokasaurus-llama-3b-8xh100-min0-serve.modal.run/v1",
#     model_name="llama3.2-3b-instruct",
# )

client = TokasaurusClient.Config(
    url="http://localhost:8889/v1",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

# client = OpenAIClient.Config(model_name="gpt-4o-mini"),

STATE = "Alabama"

USER_SYSTEM_PROMPT_TEMPLATE = f"""You will be given part of {STATE}'s legal code, and you should generate a single chat message that 
instructs the model to do something related to the context.

Only generate a single chat message, and do not include any other text or formatting.
"""
USER_PROMPT_TEMPLATE = """<context>
{{context}}
</context>

Please generate a single question/instruction related to the context.

Here are some examples of the type of question/instruction you should generate: 
- "Rewrite the statute in simpler language."
- "Summarize the state's policy on data privacy."
- "List the three ways a company can be sued for a breach of contract."
- "Translate the state's regulations of the sale of goods into French." 

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
    retrieve_top_k=8,
    assistant_system_prompt_template=ASSISTANT_SYSTEM_PROMPT_TEMPLATE,
    assistant_prompt_template=ASSISTANT_PROMPT_TEMPLATE,
    user_system_prompt_template=USER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=USER_PROMPT_TEMPLATE,
    assistant_max_completion_tokens=384,
    user_max_completion_tokens=128,
    num_tokens_in_user_context=2048
)

config = GenerateConfig(
    name=FormatStringVariable(f"{file_name}_{STATE.lower()}_n{{num_samples}}"),
    convo_generator=generator,
    context=HousingStatutesContextConfig(
        states=[STATE],
        split_on="statute"
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=128,
    max_num_batches_in_parallel=32,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    pydrantic.main(config)
