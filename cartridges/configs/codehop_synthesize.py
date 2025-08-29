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

LEVEL = int(os.environ.get("LEVEL", "1"))
MODEL = os.environ.get("MODEL", "qwen")
REPO = "repo-244c02"

REPO_TO_DATA = {
    "repo-244c02": {
        "dataset_dir": "/data/sabri/cartridges/2025-08-26-16-23-39-make_codehop/codehop-nf16-nm1-dc3-v5-fn36-0/repo-244c02",
        "cartridges": {
            "qwen": {
                1: [],
            },
            "llama": {
                1: [],
            },
        }
    },
    "repo-b0b268": {
        "dataset_dir": "/data/sabri/cartridges/2025-08-25-11-32-21-make_codehop/codehop-nf10-nm1-dc3-v5-fn36-0/repo-b0b268",
        "level_to_cartridge": {
            1: [],
            2: [
                CartridgeConfig(
                    id="hazy-research/cartridges/k9zti795",
                    source="wandb"
                )
            ],
        }
    }
}

cartridges = REPO_TO_DATA[REPO]["cartridges"][MODEL][LEVEL]
dataset_dir = REPO_TO_DATA[REPO]["dataset_dir"]

if MODEL == "qwen":
    client = TokasaurusClient.Config(
        url="https://hazyresearch--toka-qwen3-4b-1xh100-cartridges-serve.modal.run",
        model_name="Qwen/Qwen3-4b",
        cartridges=cartridges
    )
elif MODEL == "llama":
    client = TokasaurusClient.Config(
        url="https://hazyresearch--toka-llama-3-2-3b-1xh100-cartridges-serve.modal.run",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        cartridges=cartridges
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
                path=dataset_dir,
                max_level=LEVEL,
            )
        ],
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65768, 
    batch_size=32,    # Smaller batches 
    
    max_num_batches_in_parallel=512,

    name=FormatStringVariable(f"{Path(__file__).stem}_{MODEL}_{REPO}_level{LEVEL}_n{{num_samples}}"),
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