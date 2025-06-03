import os
import pydrantic
from cartridges.Cartridge_dataset_to_ref_dist import GetRefDistConfig
from cartridges.models.config import HFModelConfig
from cartridges.utils.wandb import WandBConfig


config = GetRefDistConfig(
    name="m04d02_amd_mem_data_3b_ref_dist",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
    ),
    context_convo_dataset_path="hazy-research/Cartridges/m04d02_amd_gen_data_mem_3b:v1",
    dataset_is_wandb=True,
    wandb=WandBConfig(
        project="cartridges",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    system_prompt_template="""Please use the information in the following financial document to answer the user's questions.
<title>
{title}
</title>

<document>
{content}
</document>
""",
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", None),
    batch_size=1,
)

if __name__ == "__main__":
    pydrantic.main([config])
