import os


import pydrantic

from capsules.clients.together import TogetherClient
from capsules.train import TrainConfig, TrainableCache
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset
from capsules.utils import WandBConfig


config = TrainConfig(
    name="fine_tune_on_minions_r1",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3b-Instruct"
    ),
    dataset=CapsuleDataset.Config(
        # data_path='/data/sabri/code/capsules/output/minions_1024_questions_for_sabri.feather',
        data_sources=[("hazy-research/capsules/m03d07_basic_qa_train:v2", None)],
        is_wandb=True,
        label_type="logits",
    ),
    # eval_dataset=CapsuleDataset.Config(
    #     # data_path='/data/sabri/code/capsules/output/minions_1024_questions_for_sabri.feather',
    #     data_path="hazy-research/capsules/m03d04_basic_doc_test_minions:v1",
    #     is_wandb=True,
    #     label_type="tokens",
    # ),
    cache=TrainableCache.Config(
        num_tokens=1000,
    ),
    epochs=3,
    cache_init=None,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    batch_size=2,
    summary_client=TogetherClient.Config(
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key="195a6baaffdf631b283af4866ccc1c07e2979b1685bd1de86cfd94cd35086ead",
    ),
)

if __name__ == "__main__":
    pydrantic.main([config])
