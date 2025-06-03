import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.optim import CosWithWarmup
from capsules.tasks.novelqa import NovelEvalDataset, NovelMultipleChoiceGenerateDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
bs = 64
configs = []
num_tokens = 8192

BOOK_IDXS = 'Frankenstein_Demo'
prefix = "hazy-research/capsules"
expt_tag = "simpleqa_memorize_v1"
configs.append(
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_book{BOOK_IDXS}_lr{{lr}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        
        lr=5e-3,
        lr_scheduler=CosWithWarmup.Config(
            warmup_steps=128,
            max_steps=2048,
            warmup_min_lr=5e-5,
            alpha_f=0.1,
        ),

        dataset=CapsuleDatasetLatest.Config(
            data_sources=[
                # (f"{prefix}/m04d19_memorize_F_r_a_n_k_e_n_s_t_e_i_n___D_e_m_o_memorize:v0", None),
                (f"/data/simran/capsules/outputs/2025-04-20-22-00-38-m04d19_memorize_v1/41b97e62-09f6-4993-836d-9d857c83bc69/artifact/dataset.pkl", None),
                (f"{prefix}/m04d19_simple_qa_F_r_a_n_k_e_n_s_t_e_i_n___D_e_m_o:v0", None),
            ],
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,

        ),
        
        save_every_n_steps=2048,
        generate_every_n_steps=64,
        generate_max_new_tokens=512,
        generate_datasets=[
            GenerateDatasetConfig(
                dataset=NovelMultipleChoiceGenerateDataset.Config(
                    book_id=BOOK_IDXS, 
                    cot=False,
                ),
                name_for_wandb=f"novelqa_{expt_tag}_{BOOK_IDXS}_mc",
            ),
        ],
        eval_every_n_steps=64,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb=f"novelqa_{expt_tag}_mc",
                local_batch_size=16,
                dataset=NovelEvalDataset.Config(
                    book_id=BOOK_IDXS,
                    max_questions=256,
                    label_type="tokens",
                    data_sources=[]  # ignore this arg
                )
            )
        ],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        epochs=1,

        wandb=WandBConfig(
            project="capsules",
            tags=["train", f"novelqa_{expt_tag}", f"book{BOOK_IDXS}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=4,
    )
)

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)
