from capsules.capsule_dataset_to_ref_dist_subcontext import GetRefDistConfig
from capsules.config import HFModelConfig
from capsules.utils.wandb import WandBConfig
import os


def refdist_config(input_artifact: str, output_artifact: str):
    return GetRefDistConfig(
        name=output_artifact,
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        ),
        context_convos_dataset_from_subcontexts=input_artifact,
        wandb=WandBConfig(
            project="capsules",
            tags=["cache_tuning", "development"],
            entity="hazy-research",
        ),
        system_prompt_template="""Please use the information in the following document to answer the user's questions.
{instructions}

Title: {title}

Here is the content of the document.
{subcontext}
""",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", None),
        batch_size=1,
    )
