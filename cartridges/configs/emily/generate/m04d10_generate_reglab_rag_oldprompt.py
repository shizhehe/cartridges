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

client = TokasaurusClient.Config(
    url="https://hazyresearch--tokasaurus-llama-3b-8xh100-min0-serve.modal.run/v1",
    model_name="llama3.2-3b-instruct",
)

# client = TokasaurusClient.Config(
#     url="http://localhost:8889/v1",
#     model_name="meta-llama/Llama-3.2-3B-Instruct",
# )

# client = OpenAIClient.Config(model_name="gpt-4o-mini"),

STATE = "Alabama"

USER_SYSTEM_PROMPT_TEMPLATE = f"""You are working to train a language model on the {STATE} state legal statutes.

You will be given excerpts of {STATE} state statutes and your job is to generate a training question that is grounded in the provided statutes. 
After, we will train a langauge model to answer the question without access to the statute text (i.e. closed book). 
The question should be challenging and can require the model to remember specific legal details."""

USER_PROMPT_TEMPLATE = f"""Please generate a challenging question about the following excerpt from the state legal statutes of {STATE}. 

---

{{context}}

---
The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""


ASSISTANT_SYSTEM_PROMPT_TEMPLATE = f"""You are a legal expert on the state statutes of {STATE}. 
Please use the following information from the {STATE} state statutes to answer the question.

<excerpt>
{{context}}
</excerpt>

First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase."""

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
    num_samples=8192,  # was 32768
    batch_size=128,
    max_num_batches_in_parallel=32,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
    previous_run_dir=Path(
        "/home/emilyryliu/capsules-outputs/2025-04-10-21-56-00-m04d10_generate_reglab_rag_oldprompt/485f1a5b-9dce-4d06-b14f-e365890a0ecc/"
    )
)


if __name__ == "__main__":
    pydrantic.main(config)
