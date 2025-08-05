import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.ntp import NTPSynthesizer
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.utils import WandBConfig
from cartridges.configs.utils import short_model_name



# client = TokasaurusClient.Config(
#     url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
#     model_name="Qwen/Qwen3-4b",
# )

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-llama-3-2-3b-1xh100-batch-serve.modal.run",
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
    
    synthesizer=NTPSynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.0,
        resources=[
            LongHealthResource.Config(
                patient_ids=patient_ids,
                max_chars_per_note=4096
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536, 
    batch_size=1,    # Smaller batches 
    
    max_num_batches_in_parallel=512,

    name=FormatStringVariable(f"{Path(__file__).stem}_{short_model_name(client.model_name)}_{patients_str}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"longhealth_synthesis"],
    ),
    save_wandb_artifact=False,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    pydrantic.main([config])