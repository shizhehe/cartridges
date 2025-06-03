import os
from capsules.generate.run import GenerateConfig
from capsules.tasks.reglab.housing_qa import HousingEvalAnswerGenerator, HousingStatutesContextConfig
from capsules.clients.openai import OpenAIClient
from capsules.utils.wandb import WandBConfig
import pydrantic
from pathlib import Path

STATE = "Alabama"

config = GenerateConfig(
    name=f"housing_qa_eval_data_{STATE.lower()}",
    convo_generator=HousingEvalAnswerGenerator.Config(
        answer_client=OpenAIClient.Config(model_name="gpt-4o"),
        states=[STATE],
    ),
    context=HousingStatutesContextConfig(states=[STATE]),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # SE(04/01): by setting num_samples to None, we determine the number of samples
    # by calling `len(convo_generator)`.
    num_samples=6853,  # size of the housing_qa dataset
    batch_size=64,
    max_num_batches_in_parallel=8,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
    previous_run_dir=Path(
        "/home/emilyryliu/capsules-outputs/2025-04-02-21-48-29-m03d31_generate_housing_qa_eval_alabama/eff0d20c-575f-4773-9369-6050cb363a28/"
    )
)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])