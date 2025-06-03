import os
from pathlib import Path
import pydrantic


from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.subcontext import RandomSubcontextGenerator, RandomizedSlidingWindowGenerator
from capsules.tasks.finance import FinanceBenchContextConfig, FinanceBenchSectionedContextConfig
from capsules.utils import WandBConfig



client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


QUESTION_PROMPT_TEMPLATE = """Please generate a challenging question about the following excerpt from a document. 

---

{chunk}

---
The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""

QUESTION_SYSTEM_PROMPT = """You are working to train a language model on the information in a document.

You will be given excerpts of the document and your job is to generate a training question that is grounded in the provided document. 
After we will train a langauge model to answer the question without access to the document (i.e. closed book). 
The question should be challenging and can require the model to remember details like specific numerical figures, dates, and names. 

<title>
{title}
</title>

Here is the content of the document.
{subcontext}
"""


ANSWER_SYSTEM_PROMPT = """You are an expert financial analyst. 
Please use the following information from a financial document to answer the question.

<title>
{title}
</title>

Here is the content of the document.
{subcontext}


First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""

file_name = Path(__file__).stem


doc_name = "AMD_2022_10K"

num_samples = 32678


subcontext_generator = RandomizedSlidingWindowGenerator.Config(
    num_passes=10,
    min_window_size=6,
    max_window_size=8,
    min_step_size=3,
    max_step_size=5,
    seed=32,
    # max_tokens_per_section=2_000,
    # chunk_size=2000,
)

def config(doc_name):
    config = GenerateTrainingConfig(
        name=f"amd_random_windowed_basic_qa",

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
            subcontext_generator=subcontext_generator,
        ), 

        context=FinanceBenchContextConfig(
            doc_names=[doc_name], 
        ),

        # generate config
        num_samples=num_samples,
        batch_size=min(num_samples, 256),
        max_num_batches_in_parallel=min(num_samples, 16),

        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        
    )

    return config


if __name__ == "__main__":
    pydrantic.main(config(doc_name))
