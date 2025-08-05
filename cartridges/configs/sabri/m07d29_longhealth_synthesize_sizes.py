import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.utils import WandBConfig
from cartridges.configs.utils import short_model_name


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

configs = []
for size in ["8"]:
    size_modal = size.replace(".", "-")
    client = TokasaurusClient.Config(
        url=f"https://hazyresearch--toka-qwen3-{size_modal}b-1xh100-min0-serve.modal.run",
        model_name=f"Qwen/Qwen3-{size}b",
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
                LongHealthResource.Config(
                    seed_prompts=[
                        "structuring",
                        "summarization",
                        "question",
                        "use_case",
                        "creative",
                    ],
                    patient_ids=patient_ids,
                    min_notes_per_prompt=2,
                    max_notes_per_prompt=2,
                    max_chars_per_note=2048,
                )
            ],
        ),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        num_samples=65536 * 2, 
        batch_size=32,    # Smaller batches 
        
        max_num_batches_in_parallel=256,

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
    configs.append(config)

if __name__ == "__main__": 
    pydrantic.main(configs)