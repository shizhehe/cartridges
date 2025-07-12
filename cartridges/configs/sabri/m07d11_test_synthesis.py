import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient

from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.resources import FileResource
from cartridges.utils import WandBConfig



client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
    model_name="Qwen/Qwen3-4b",
)

client = TokasaurusClient.Config(
    url="http://0.0.0.0:10210",
    model_name="Qwen/Qwen3-4b",
)



config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_cot_a=0.3,
        use_tools_a=False, 
        use_tools_b=False,
        max_completion_tokens_b=256,
        tools=[],
        resources=[
            FileResource.Config(
                path=os.path.join(os.environ["CARTRIDGES_DIR"], "README.md"),
                seed_prompts=["generic"]
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=16, 
    batch_size=4,    # Smaller batches 
    
    max_num_batches_in_parallel=1,

    name=FormatStringVariable(f"file_synthesis"),
    run_id="file_synthesis",
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"gmail_synthesis", "tools", "mcp"],
    ),
    save_wandb_artifact=False,
    save_wandb_preview=False,

)


if __name__ == "__main__": 
    pydrantic.main([config])