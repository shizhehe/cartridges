import os
from pathlib import Path
import pydrantic

from cartridges.initialization.random import KVFromRandomText
from cartridges.train import TrainConfig, LossEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import TrainDataset, LossEvalDataset


DATA_SOURCE = "/data/sabri/cartridges/2025-08-08-17-26-20-arxiv_synthesize/arxiv_synthesize_Qwen/Qwen3-4b_n8192-0/artifact/dataset.pkl"


config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromRandomText.Config(
        max_tokens=2048
    ),
    
    lr=2e-2,
    epochs=1,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[(DATA_SOURCE, None)],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    loss_eval_every_n_steps=128,
    loss_evals=[
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                path="/data/sabri/cartridges/2025-08-11-16-39-24-arxiv_synthesize_eval/arxiv_synthesize_eval_gpt-5-mini-2025-08-07_n32-0/artifact/dataset.pkl",
                packed_seq_length=2048,
                packing_mode="truncate",
            ),
            name_for_wandb="arxiv_synthesize",
        )
    ],

    save_every_n_steps=512,
    name="cartridges-tutorial-train",
)


if __name__ == "__main__":
    pydrantic.main(config)