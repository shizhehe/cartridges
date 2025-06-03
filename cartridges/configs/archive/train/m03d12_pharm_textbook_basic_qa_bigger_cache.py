import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.kv_initialization.strategies.prompt import PromptInitializer
from capsules.train import TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.tasks.openstax.datasets import OpenStaxMultipleChoiceGenerateDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem
configs = []
config = TrainConfig(
    name=f"{file_name}",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    ),
    dataset=CapsuleDataset.Config(
        data_sources=[
            ("hazy-research/capsules/m03d11_basic_qa_train_pharm_textbook:v0", None),
        ],
        is_wandb=True,
        label_type="tokens",
    ),
    generate_every_n_steps=512,
    generate_dataset=OpenStaxMultipleChoiceGenerateDataset.Config(
        # path="/data/sabri/code/capsules/data/tasks/openstax/business_law_questions.json",
        # path="/data/sabri/code/capsules/data/tasks/openstax/pharm_questions.json",
        path=str(Path(__file__).parent.parent.parent.parent / "data/tasks/openstax/pharm_questions.json")
    ),
    generate_max_new_tokens=10,
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=10_000
    ),
    loss_type="tokens",
    save_every_n_steps=512,
    epochs=1,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    batch_size=2,
)
configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
