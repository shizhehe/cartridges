import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.Cartridge_dataset_to_ref_dist import GetRefDistConfig
from cartridges.models.config import HFModelConfig
from cartridges.utils.wandb import WandBConfig

file_name = Path(__file__).stem

config = GetRefDistConfig(
    name=FormatStringVariable(
        f"{file_name}_3b_{{context_convo_dataset_path}}"
    ),
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
    ),
    context_convo_dataset_path="m04d03_generate_ntp_3b_docAMD_2022_10K_sections4_npreview64_max1024_32000:v0",
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
    batch_size=1
)

if __name__ == "__main__":
    pydrantic.main([config])
