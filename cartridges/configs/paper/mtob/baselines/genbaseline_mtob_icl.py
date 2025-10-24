

from pathlib import Path
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate_baseline import ICLBaseline
import pydrantic

from capsules.configs.paper.mtob.baselines.genbaseline_mtob_common import get_configs

tokasaurus_client = TokasaurusClient.Config(
    # url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    url="https://hazyresearch--tksrs-batch-main-llama-8b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)


file_name = Path(__file__).stem

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the following content to answer the user's questions.

<content>
{{content}}
</content>
"""

generator = ICLBaseline.Config(
    client=tokasaurus_client,
    system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
    max_context_tokens=110_000,

)

configs = get_configs(
    names=["mtob_icl"],
    generators=[generator],
    max_num_batches_in_parallel=32,
    batch_size=16,
    direction="ke",
    context="medium_and_sentences",
)


if __name__ == "__main__":
    pydrantic.main(configs)

