import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient

from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.generators.auto import AutoConvoGenerator, SlicePromptSampler
from capsules.tasks.reglab.context import ReglabHousingStructuredContextConfig
from capsules.tasks.reglab.utils import ALL_STATES
from capsules.tools.retrieval import BM25RetrieverTool
from capsules.utils import WandBConfig



client = TokasaurusBatchClient.Config(
    # url="http://localhost",
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    # ports=[8880 + i for i in range(8)],
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


NUM_STATES = 5
states_str = f"{NUM_STATES}states"
states = ALL_STATES[:NUM_STATES]

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
    name=FormatStringVariable(f"{file_name}_{states_str}_{"+".join(SLICES)}_n{{num_samples}}_cot{{convo_generator.prob_cot_a}}"),
    run_id=FormatStringVariable("{name}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        prompt_sampler=SlicePromptSampler.Config(
            max_tokens_initial_context=4096,
            slices=SLICES,
        ),
        prob_cot_a=0.2,
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
    context=ReglabHousingStructuredContextConfig(
        states=states,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32768 * 2,
    batch_size=128,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"reglab-housing", "generate", f"{states_str}"],
    ),
)


if __name__ == "__main__": 
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
