import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.mtob.resources import MTOBResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.ntp import NTPSynthesizer
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.utils import WandBConfig
from cartridges.configs.utils import short_model_name



# client = TokasaurusClient.Config(
#     url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
#     model_name="Qwen/Qwen3-4b",
# )

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-llama-3-1-8b-1xh100-batch-serve.modal.run",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)


# client = TokasaurusClient.Config(
#     url="http://0.0.0.0:10210",
#     model_name="Qwen/Qwen3-4b",
# )


config = SynthesizeConfig(
    
    synthesizer=NTPSynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.0,
        resources=[
            MTOBResource.Config(
                setup="latex_and_sentences",
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536, 
    batch_size=1,    # Smaller batches 
    
    max_num_batches_in_parallel=512,

    name=FormatStringVariable(f"{Path(__file__).stem}_{short_model_name(client.model_name)}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"mtob_synthesis"],
    ),
    save_wandb_artifact=False,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    pydrantic.main([config])