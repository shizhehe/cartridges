import os
from pathlib import Path
import numpy as np
import pydrantic
from pydrantic.variables import FormatStringVariable
from capsules.config import HFModelConfig
from capsules.train import TrainConfig, TrainableCache
from capsules.tasks.finance.questions import FinanceQuestionDataset, FinanceQuestionTestDataset
from capsules.utils import WandBConfig


file_name = Path(__file__).stem


# https://wandb.ai/hazy-research/capsules/reports/Untitled-Report--VmlldzoxMDkxMTE2MQ
NUM_TOKENS_TO_PRETRAINED_PATHS = {
    2048: "/data/sabri/capsules/outputs/2025-01-11-22-23-45-m01d11_ntp_cache_size/b8c9cb00-cf84-487d-b35d-5912c3b5077c",
    4096: "/data/sabri/capsules/outputs/2025-01-11-22-23-45-m01d11_ntp_cache_size/844d40f5-e336-4917-be1e-1a165d2e4c59",
    8192: "/data/sabri/capsules/outputs/2025-01-11-22-23-45-m01d11_ntp_cache_size/665e4e7d-2b17-4cae-8b90-3b14e9b1531e",
    16384: "/data/sabri/capsules/outputs/2025-01-11-22-23-45-m01d11_ntp_cache_size/d6ead3f4-0410-429f-a937-9c863b859994",
    32768: "/data/sabri/capsules/outputs/2025-01-11-22-23-45-m01d11_ntp_cache_size/17294458-68c2-4c47-804b-e0b02ac73db3",
    65536: "/data/sabri/capsules/outputs/2025-01-11-22-23-45-m01d11_ntp_cache_size/933439d4-ac92-4f18-9612-78ab3fcd15a8",
}

configs = []
# for lr in [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
for num_tokens in [2048, 4096, 8192, 16384]:
    for lr in [1e-3, 5e-3]:
        config = TrainConfig(
            name=FormatStringVariable(f"{file_name}_{num_tokens}_lr{{lr}}"), #_l{model.pretrained_model_name_or_path}"),
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct"
            ),
            dataset=FinanceQuestionDataset.Config(
                data_path="/data/sabri/capsules/outputs/2025-01-03-17-57-45-m01d03_generate/46977143-a197-48aa-98e3-cd80c3a702e0/generated_questions.feather",
                questions_per_epoch=128
            ),
            test_dataset=FinanceQuestionTestDataset.Config(
                data_path="/data/sabri/capsules/outputs/2025-01-04-16-53-16-m01d04_generate_question_test/80317e27-cb29-48af-a5b1-3e436f5dc4af/generated_questions.feather"
            ),
            cache=TrainableCache.PretrainedConfig(
                path=os.path.join(NUM_TOKENS_TO_PRETRAINED_PATHS[num_tokens], "cache_last.pt")
            ),
            cache_init=None,
            lr=lr,
            epochs=64,
            wandb=WandBConfig(
                project="capsules",
                tags=["cache_tuning", "development"],
            ),
            output_dir=os.environ["capsules_OUTPUT_DIR"],
        )
        configs.append(config)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main(configs)