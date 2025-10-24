import os
import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig
from cartridges.models import HFModelConfig
from cartridges.datasets import DataSource, TrainDataset



config = TrainConfig(
    model=HFModelConfig(
        # pretrained_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
        pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source=os.path.join(os.environ["CARTRIDGES_DIR"], "examples/arxiv/cartridges.tex"),
        max_tokens=2048
    ),
    
    lr=2e-2,
    epochs=1,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[
            # TODO: replace below with your own dataset you just synthesized and 
            # remove our huggingface dataset below
            DataSource(
                path="/Users/sabrieyuboglu/code/cartridges/output/2025-09-14-17-22-03-m09d13_gmail_synthesize/m09d13_gmail_synthesize_meta-llama/Llama-3.1-8B-Instruct_n65536-0/artifact/dataset.parquet",
                type="local"
            ),    
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    loss_eval_every_n_steps=16,
    loss_evals=[],

    generate_eval_every_n_steps=32,
    generate_evals=[],
    distributed_backend="nccl",

    save_every_n_steps=16,
    name="cartridges-tutorial-train",
)


if __name__ == "__main__":
    pydrantic.main(config)