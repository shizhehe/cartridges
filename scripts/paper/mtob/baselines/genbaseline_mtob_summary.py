

from pathlib import Path
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.generate_baseline import ICLBaseline
import pydrantic

from cartridges.configs.paper.mtob.baselines.genbaseline_mtob_common import get_configs
from cartridges.clients.openai import OpenAIClient
from cartridges.generate_baseline import ICLBaselineSummaryFromLargeModel


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

generators = [
    (f"{file_name}_n{num_tokens}", ICLBaselineSummaryFromLargeModel.Config(
        client=tokasaurus_client,
        summary_client=OpenAIClient.Config(model_name="gpt-4o"),
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        temperature=0.0,
        max_completion_tokens=512,
        summary_tokens=num_tokens,
    ))
    for num_tokens in [128, 256, 512, 1024, 2048, 4096]
]
names, generators = zip(*generators)


configs = get_configs(
    names=names,
    generators=generators,
    max_num_batches_in_parallel=32,
    batch_size=16,
    direction="ke",
    context="medium_and_sentences",
)

if __name__ == "__main__":
    pydrantic.main(configs)

