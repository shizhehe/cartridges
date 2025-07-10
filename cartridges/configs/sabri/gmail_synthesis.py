import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient

from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils import WandBConfig
from cartridges.tools.gmail.tools import GmailToolSet
from cartridges.tools.gmail.resources import GmailResource, LabelConfig



client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run/v1",
    model_name="Qwen/Qwen3-4b",
)


config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_cot_a=0.3,
        use_tools_a=False, 
        use_tools_b=False,
        tools=[
            # GmailToolSet.Config(email="sabri@stanford.edu"  )
        ],
        resources=[
            GmailResource.Config(
                labels=[
                    LabelConfig(name="categories--stanford--primary--stanford-", weight=1.0),
                    LabelConfig(name="categories--stanford--updates--stanford-", weight=1.0),
                ]
            )
        ],
    ),
    output_dir=os.environ.get("CODEMEM_OUTPUT_DIR", "."),
    num_samples=32768, 
    batch_size=64,    # Smaller batches 
    
    max_num_batches_in_parallel=32,

    name=FormatStringVariable(f"gmail_synthesis"),
    run_id="gmail_synthesis",
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