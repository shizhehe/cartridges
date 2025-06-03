import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable
from capsules.config import HFModelConfig
from capsules.train import TrainConfig, TrainableCache
from capsules.tasks.finance.questions import FinanceQuestionDataset, FinanceQuestionTestDataset
from capsules.utils import WandBConfig


file_name = Path(__file__).stem

config = TrainConfig(
    name=FormatStringVariable(f"{file_name}_lr{{lr}}"), #_l{model.pretrained_model_name_or_path}"),
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct"
    ),
    dataset=FinanceQuestionDataset.Config(
        data_path="/data/sabri/capsules/outputs/2025-01-03-17-57-45-m01d03_generate/46977143-a197-48aa-98e3-cd80c3a702e0/generated_questions.feather",
        questions_per_epoch=1024
    ),
    test_dataset=FinanceQuestionTestDataset.Config(
        data_path="/data/sabri/capsules/outputs/2025-01-04-16-53-16-m01d04_generate_question_test/80317e27-cb29-48af-a5b1-3e436f5dc4af/generated_questions.feather"
    ),
    cache=TrainableCache.PretrainedConfig(
        path="/data/sabri/capsules/outputs/2025-01-03-11-02-22-m12d23_base/ce292335-e18f-48b1-abde-25feac97bb4d/cache_last.pt"
    ),
    cache_init=None,
    lr=1e-3, 
    epochs=64,
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
    ),
    output_dir=os.environ["capsules_OUTPUT_DIR"],
)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])