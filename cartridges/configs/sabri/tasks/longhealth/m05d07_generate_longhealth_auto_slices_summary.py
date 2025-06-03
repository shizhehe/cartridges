import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.clients.tokasaurus_batch import TokasaurusBatchClient

from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.generators.auto import AutoConvoGenerator, SlicePromptSampler, SlicePromptSamplerWithSummarization
from capsules.tasks.longhealth import LongHealthContextConfig
from capsules.tasks.longhealth.generators import PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, QUESTION_PROMPT_TEMPLATE, GeneratedReformatWithToCGenerator
from capsules.tasks.longhealth.seed_prompts import LongHealthPromptSampler, LongHealthTreePromptSampler
from capsules.tools.retrieval import BM25RetrieverTool
from capsules.utils import WandBConfig

from capsules.tasks.longhealth.context import LongHealthStructuredContextConfig


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
        prompt_sampler=SlicePromptSamplerWithSummarization.Config(
            client=client,
            slices=SLICES,
            
            max_tokens_per_page=4096,
            num_focus_leaves_per_context=1,
            max_tokens_in_context=(1024, 8192),
            
            min_tokens_to_summarize=512,
            max_tokens_to_summarize=8192,
            max_tokens_in_summary=128,
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
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=128,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"longhealth", "generate", f"patients_{patients_str}"],
    ),
)


if __name__ == "__main__": 
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
