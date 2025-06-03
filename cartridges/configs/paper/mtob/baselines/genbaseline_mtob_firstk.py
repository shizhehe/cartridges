

from pathlib import Path
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate_baseline import ICLBaselineFirstKTokens
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

generators = [
    (f"mtob_firstk_frac{frac_of_tokens}", ICLBaselineFirstKTokens.Config(
        client=tokasaurus_client,
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        frac_of_tokens=frac_of_tokens,
    ))
    for frac_of_tokens in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
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

