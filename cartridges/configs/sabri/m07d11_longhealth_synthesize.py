import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.utils import WandBConfig



client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
    model_name="Qwen/Qwen3-4b",
)

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-llama-3-2-3b-instruct-1xh100-min0-serve.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

# client = TokasaurusClient.Config(
#     url="http://0.0.0.0:10210",
#     model_name="Qwen/Qwen3-4b",
# )

NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]



config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_cot_a=0.2,
        use_tools_a=False, 
        use_tools_b=False,
        # max_completion_tokens_b=256,
        tools=[],
        resources=[
            LongHealthResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                patient_ids=patient_ids,
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536, 
    batch_size=32,    # Smaller batches 
    
    max_num_batches_in_parallel=256,

    name=FormatStringVariable(f"{Path(__file__).stem}_{patients_str}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
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