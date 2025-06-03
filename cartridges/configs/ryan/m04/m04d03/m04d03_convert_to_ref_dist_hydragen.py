# import os
# import pydrantic
# from capsules.capsule_dataset_to_ref_dist import GetRefDistConfig
# from capsules.capsule_dataset_to_ref_dist_hydragen import GetRefDistHydragenConfig
# from capsules.config import HFModelConfig
# from capsules.utils.wandb import WandBConfig


# config = GetRefDistHydragenConfig(
#     name="m04d02_amd_gen_data_mem_vary_chunks_3b_hydragen",
#     model=HFModelConfig(
#         pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
#         use_hydragen=True
#     ),
#     context_convo_dataset_path="hazy-research/capsules/m04d02_amd_gen_data_mem_vary_chunks_3b:v0",
#     dataset_is_wandb=True,
#     wandb=WandBConfig(
#         project="capsules",
#         tags=["cache_tuning", "development"],
#         entity="hazy-research",
#     ),
#     system_prompt_template="""Please use the information in the following financial document to answer the user's questions.
# <title>
# {title}
# </title>

# <document>
# {content}
# </document>
# """,
#     output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", None),
#     batch_size=1,
# )

# if __name__ == "__main__":
#     pydrantic.main([config])
