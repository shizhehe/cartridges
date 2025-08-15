import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.resources import DirectoryResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig



client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
    model_name="Qwen/Qwen3-4b",
)

config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.75,
        use_tools_a=False, 
        use_tools_b=False,
        # max_completion_tokens_b=256,
        tools=[],
        resources=[
            DirectoryResource.Config(
                path="/data/sabri/cartridges/2025-08-15-12-20-31-synthesize_task/codehop-nf10-nm10-mc10-dc10-iv36-ov36-fn36-0/repo",
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=1024, 
    batch_size=32,    # Smaller batches 
    
    max_num_batches_in_parallel=256,

    name=FormatStringVariable(f"{Path(__file__).stem}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"longhealth_synthesis"],
    ),
    upload_to_wandb=False,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    pydrantic.main([config])