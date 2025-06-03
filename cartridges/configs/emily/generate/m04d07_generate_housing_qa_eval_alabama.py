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
    # target="capsules.generate.GenerateConfig",
    # convo_generator={
    #     "target": "capsules.tasks.reglab.housing_qa.HousingEvalAnswerGenerator",
    #     "states": [STATE],
    #     "answer_client": {
    #         "target": "capsules.clients.openai.OpenAIClient",
    #         "model_name": "gpt-4o",
    #     }
    # },
    context=HousingStatutesContextConfig(states=[STATE]),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # SE(04/01): by setting num_samples to None, we determine the number of samples
    # by calling `len(convo_generator)`.
    num_samples=147,  # size of the housing_qa dataset: 6853. alabama has 147 questions
    batch_size=64,
    max_num_batches_in_parallel=8,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])