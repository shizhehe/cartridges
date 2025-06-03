import os

import pydrantic

from capsules.clients.together import TogetherClient
from capsules.kv_initialization.strategies.summarization import KVCacheInitFromSummary
from capsules.train import TrainConfig, TrainableCache
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset
from capsules.utils import WandBConfig


config = TrainConfig(
    name="fine_tune_on_minions_r1",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    ),
    dataset=CapsuleDataset.Config(
        data_sources=[
            # ("hazy-research/capsules/m03d09_between_sections_memorization:v0", None),
            # ("hazy-research/capsules/m03d09_next_section_memorization:v0", None),
            # ("hazy-research/capsules/m03d09_previous_section_memorization:v0", None),
            ("hazy-research/capsules/m03d07_basic_qa_train:v4", None),
        ],
        is_wandb=True,
        label_type="logits",
    ),
    eval_dataset=CapsuleDataset.Config(
        data_sources=[("hazy-research/capsules/m03d07_basic_qa_train:v2", None)],
        is_wandb=True,
        label_type="tokens",
    ),
    kv_cache_initializer=KVCacheInitFromSummary.Config(
        num_tokens=1000,
        is_wandb=False,
        directory=os.path.join(
            os.environ["CAPSULES_OUTPUT_DIR"],
            "kv_cache_summary/minions_summary_1000_tokens/",
        ),
        client=TogetherClient.Config(
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            api_key="195a6baaffdf631b283af4866ccc1c07e2979b1685bd1de86cfd94cd35086ead",
        ),
    ),
    epochs=3,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    batch_size=2,
)

if __name__ == "__main__":
    pydrantic.main([config])
