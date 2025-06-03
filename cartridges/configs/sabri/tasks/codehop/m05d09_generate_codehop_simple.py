import os
from pathlib import Path


from capsules.tasks.codehop.code_hop_synth import CodeHopSynthConfig
from capsules.tasks.codehop.context import CodeHopStructuredContextConfig
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
    url="https://hazyresearch--tksrs-entry-capsules-8b-1xh100-min0-max32-serve.modal.run",
    # ports=[8880 + i for i in range(8)],
    ports=None,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

ch_config = CodeHopSynthConfig(
    seed=42,
    num_files=1,
    num_methods_per_file=5,
    method_name_length=4,
    deepest_call_chain=4,
    input_vocab_size=4,
    output_vocab_size=4,
    function_name_vocab_size=50,
)
ch_str = ch_config.hash()


file_name = Path(__file__).stem

if "SLICES" in os.environ:
    SLICES = os.environ["SLICES"].split(",")
else:
    SLICES = [
        # "structuring",
        # "summarization",
        # "aggregation",
        "question",
        # "use_case",
    ]

config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{ch_str}_{"+".join(SLICES)}_files{ch_config.num_files}_n{{num_samples}}_cot{{convo_generator.prob_cot_a}}"),
    run_id=FormatStringVariable("{name}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        max_rounds=1,
        prompt_sampler=SlicePromptSampler.Config(
            slices=SLICES,
            max_tokens_initial_context=4096,
            leaves_only=True,
            include_outline=False,
        ),
        temperature_a=0.8,
        temperature_b=0.8,
        prob_cot_a=0.2,
        use_tools=False, 
        tools=[
            # BM25RetrieverTool.Config(
            #     max_tokens_per_chunk=256,
            # ),
            # BM25RetrieverTool.Config(
            #     max_tokens_per_chunk=1024,
            # ),
            # OpenAIRetrieverTool.Config(
            #     max_tokens_per_chunk=256,
            #     embedding_model="text-embedding-3-large",
            # ),
        ],
    ),
    context=CodeHopStructuredContextConfig(
        code_hop_config=ch_config,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=128,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"codehop", "generate", f"ch_{ch_str}"],
    ),
)


if __name__ == "__main__": 
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
