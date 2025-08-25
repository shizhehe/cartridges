import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import CartridgeConfig
from cartridges.data.code.resources import PythonRepositoryResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig


# DATASET_DIR = "/data/sabri/cartridges/2025-08-20-10-12-26-make_codehop/codehop-nf4-nm3-dc2-v4-fn36-0/repo-e2c17c"
# DATASET_DIR = "/data/sabri/cartridges/2025-08-20-10-26-45-make_codehop/codehop-nf14-nm1-dc2-v4-fn36-0/repo-df0b47"
DATASET_DIR = "/data/sabri/cartridges/2025-08-24-22-00-09-make_codehop/codehop-nf12-nm1-dc3-v4-fn36-0/repo-27f65b"


MODEL = os.environ.get("MODEL", "qwen")
if MODEL == "qwen":
    client = TokasaurusClient.Config(
        url="https://hazyresearch--toka-qwen3-4b-1xh100-cartridges-serve.modal.run",
        model_name="Qwen/Qwen3-4b",
    )
elif MODEL == "llama":
    client = TokasaurusClient.Config(
        # url="https://hazyresearch--toka-llama-3-2-3b-1xh100-batch-serve.modal.run",
        url="https://hazyresearch--toka-llama-3-2-3b-1xh100-cartridges-serve.modal.run",
        # url="https://hazyresearch--toka-llama-3-2-3b-1xh100-main-serve.modal.run",
        # url="http://0.0.0.0:10210",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        cartridges=[
            CartridgeConfig(
                # id="hazy-research/cartridges/85axrvk4",
                id="hazy-research/cartridges/4d3h0kl4", #8192
                source="wandb"
            )
        ]
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

SYSTEM_PROMPT_TEMPLATE = """Below is a Python file that is part of a larger repository.
It contains a number of methods. Note, that some of the methods may make calls to other
methods which are defined in other files.

Other files may define methods with the same name, so when discussing the methods in 
this file, make sure you reference them with the module name (e.g. `file_name.method_name`). 
Also, make sure you consider different inputs to the methods and test both the else case in the if statements.

<code>
{subcorpus}
</code>"""

config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.0,
        temperature_a=1.0,
        temperature_b=0.0,
        use_tools_a=False, 
        use_tools_b=False,
        # max_completion_tokens_b=256,
        tools=[],
        resources=[
            PythonRepositoryResource.Config(
                path=DATASET_DIR,
                max_level=1,
            )
        ],
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65768, 
    batch_size=32,    # Smaller batches 
    
    max_num_batches_in_parallel=512,

    name=FormatStringVariable(f"{Path(__file__).stem}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"codehop_synthesize"],
    ),
    upload_to_wandb=True,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    pydrantic.main([config])