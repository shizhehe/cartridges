import os
import pydrantic
from capsules.capsule_dataset_to_ref_dist import GetRefDistConfig
from capsules.config import HFModelConfig
from capsules.utils.wandb import WandBConfig


config = GetRefDistConfig(
    name="m04d06_housing_AL_train_mem_data_ref_dist",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
    ),
    context_convo_dataset_path="hazy-research/capsules/housing_qa_train_mem_data_alabama:v0",
    dataset_is_wandb=True,
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    system_prompt_template="""Please use the information in the following Alabama state legal statutes to answer the user's questions.
<title>
{title}
</title>

<document>
{content}
</document>
""",
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", None),
    batch_size=1,
)

if __name__ == "__main__":
    pydrantic.main([config])
