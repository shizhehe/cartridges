import os
from pathlib import Path
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPrompt,
    AnswerSystemPromptWithChunk,
    AnswerSystemPromptWithDoc,
    QuestionSystemPrompt,
    QuestionSystemPromptWithDoc,
    QuestionSystemPromptWithTitle,
)
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.clients.together import TogetherClient
from capsules.clients.openai import OpenAIClient
from capsules.generate.run import GenerateConfig
from capsules.generate.context_convo_generators.basic_question_from_chunk import (
    SimpleQuestionFromChunk,
)
from capsules.generate.chunk import DocumentChunker, SimpleCharacterChunker
from capsules.utils import WandBConfig
from capsules.tasks.reglab.housing_qa import HousingEvalAnswerGenerator, HousingStatutesContextConfig

client_config = TokasaurusClient.Config(
    # url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
    # url="https://hazyresearch--tokasaurus-llama-3b-dp4-serve.modal.run/v1",
    url="https://hazyresearch--tokasaurus-llama-3b-8xh100-serve.modal.run/v1",
    model_name="llama-3.2-3b-instruct"  # this argument does not matter
)

STATE = "Alabama"


QUESTION_PROMPT_TEMPLATE = f"""Please generate a challenging question about the following excerpt from the state legal statutes of {STATE}. 

---

{{chunk}}

---
The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""

QUESTION_SYSTEM_PROMPT = f"""You are working to train a language model on the {STATE} state legal statutes.

You will be given excerpts of {STATE} state statutes and your job is to generate a training question that is grounded in the provided statutes. 
After, we will train a langauge model to answer the question without access to the statute text (i.e. closed book). 
The question should be challenging and can require the model to remember specific legal details. 
"""


ANSWER_SYSTEM_PROMPT = f"""You are a legal expert on the state statutes of {STATE}. 
Please use the following information from the {STATE} state statutes to answer the question.

<excerpt>
{{chunk}}
</excerpt>

First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""


ANSWER_PROMPT_TEMPLATE = """{question}"""


file_name = Path(__file__).stem

config = GenerateConfig(
    name=f"housing_qa_{STATE.lower()}",
    convo_generator=SimpleQuestionFromChunk.Config(
        question_client=client_config,
        question_temperature=0.6,
        question_max_completion_tokens=256,
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=512,
        chunker=SimpleCharacterChunker.Config(
            min_chunk_size_in_chars=500,
            max_chunk_size_in_chars=10_000,
        ),
        question_prompt_template=QUESTION_PROMPT_TEMPLATE,
        answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
        question_system_prompt_generator=QuestionSystemPrompt.Config(
            prompt_template=QUESTION_SYSTEM_PROMPT,
            max_tokens=12_000,
            tokenizer="meta-llama/Llama-3.2-1B-Instruct",
        ),
        answer_system_prompt_generator=AnswerSystemPrompt.Config(
            prompt_template=ANSWER_SYSTEM_PROMPT,
            max_tokens=12_000,
            tokenizer="meta-llama/Llama-3.2-1B-Instruct",
        ),
        num_top_logprobs=1,
    ),

    context=HousingStatutesContextConfig(states=[STATE]),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    num_samples=8192,
    batch_size=64,
    max_num_batches_in_parallel=1,

    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
    save_wandb_preview=True,
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
