import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.generators.auto import AutoConvoGenerator, SlicePromptSampler
from cartridges.tasks.longhealth import LongHealthContextConfig
from cartridges.tasks.longhealth.generators import PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, QUESTION_PROMPT_TEMPLATE, GeneratedReformatWithToCGenerator
from cartridges.tasks.longhealth.seed_prompts import LongHealthPromptSampler, LongHealthTreePromptSampler
from cartridges.tools.retrieval import BM25RetrieverTool
from cartridges.utils import WandBConfig

from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    # url="http://localhost",
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    # ports=[8880 + i for i in range(8)],
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

if "SLICES" in os.environ:
    SLICES = os.environ["SLICES"].split(",")
else:
    SLICES = [
        "structuring",
        # "summarization",
        # "aggregation",
        # "question",
        # "use_case",
    ]

config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{patients_str}_{"+".join(SLICES)}_n{{num_samples}}_cot{{convo_generator.prob_cot_a}}"),
    run_id=FormatStringVariable("{name}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        prompt_sampler=SlicePromptSampler.Config(
            max_tokens_initial_context=4096,
            slices=SLICES,
        ),
        prob_cot_a=0.5,
        use_tools=False, 
        tools=[
            BM25RetrieverTool.Config(
                max_tokens_per_chunk=256,
            ),
            BM25RetrieverTool.Config(
                max_tokens_per_chunk=1024,
            ),
            # OpenAIRetrieverTool.Config(
            #     max_tokens_per_chunk=256,
            #     embedding_model="text-embedding-3-large",
            # ),
        ]
    ),
    context=LongHealthStructuredContextConfig(
        patient_ids=patient_ids,
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=128,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"longhealth", "generate", f"patients_{patients_str}"],
    ),
)


if __name__ == "__main__": 
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
