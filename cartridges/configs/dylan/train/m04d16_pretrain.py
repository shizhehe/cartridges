import os
from pathlib import Path

import pydrantic

from capsules.tasks.synthetics.pretrain import PretrainConfig
from capsules.config import HFModelConfig, ScratchModelConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig
from capsules.utils import WandBConfig

config = PretrainConfig(
    # Basic configuration
    name=Path(__file__).stem,  # Name for the run

    model=ScratchModelConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
    ),
    dataset=CapsuleDataset.Config(  # Training dataset configuration
        data_sources=[
            ("hazy-research/capsules/m03d17_enron_basic_qa_train:v1", None),
        ],
        is_wandb=True,
        label_type="tokens",  # or "logits" depending on your data
        top_k_logits=20,
    ),

    # Generation configuration (optional)
    generate_datasets=[
        #GenerateDatasetConfig(
        #    name_for_wandb="generate_dataset_name",
        #    dataset=CapsuleGenerateDataset.Config(
        #        data_sources=[
        #            # Add your generation data sources here
        #            # Example: ("hazy-research/capsules/your_generate_dataset:v0", 16),
        #        ],
        #        is_wandb=True,
        #        label_type="tokens",  # or "logits" depending on your data
        #    ),
        #),
    ],


    # Evaluation configuration
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="generated_questions",
            local_batch_size=16,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m04d03_enron_basic_qa_test:v0", None),
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        )
    ],

    # Training parameters
    epochs=5,  # Number of training epochs
    lr=1e-4,  # Learning rate
    global_batch_size=64,  # Total batch size across all devices
    local_batch_size=4,  # Batch size per device
    
    # Evaluation and generation frequency
    eval_every_n_steps=128,  # How often to run evaluation
    generate_every_n_steps=256,  # How often to run generation
    generate_max_new_tokens=64,  # Maximum tokens to generate
    
    # Checkpointing
    save_every_n_steps=512,  # How often to save checkpoints
    save_after_training=True,  # Whether to save after training completes
    keep_last_n_saved=1,  # Number of checkpoints to keep
    save_to_wandb=True,  # Whether to save to wandb
    
    # Loss configuration
    loss_type="tokens",  # or "logits" depending on your data
    
    # Wandb configuration
    wandb=WandBConfig(
        project="capsules",
        tags=["pretraining", "development"],
        entity="buffalo-theory",
    ),
    
    # Other parameters
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",  # Tokenizer to use
    device="cuda",  # Device to train on
    seed=42,  # Random seed
)

if __name__ == "__main__":
    pydrantic.main([config])
