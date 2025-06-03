import os
from pathlib import Path

from capsules.datasets import CapsuleDatasetLatest
import pydrantic

from capsules.tasks.codehop.code_hop_dataset import CodeHopDataset
from capsules.tasks.codehop.code_hop_synth import CodeHopSynthConfig
from capsules.tasks.codehop.generate_dataset import CodeHopGenerateDataset
from capsules.train import GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.kv_initialization.strategies.random import KVFromRandomText

from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64

data_config = CodeHopSynthConfig(
    seed=34,
    num_files=5,
    num_methods_per_file=5,
    method_name_length=4,
    deepest_call_chain=4,
    input_vocab_size=4,
    output_vocab_size=4,
    function_name_vocab_size=50,
)

data_sources = {
    "question": [
        "hazy-research/capsules/m05d09_generate_codehop_simple_d2128e_question_n32768_cot0.2:v0",
    ],
}

if "SLICES" in os.environ:
    SLICES = os.environ["SLICES"].split(",")
else:
    SLICES = [
        # "structuring",
        "question",
    ]


configs = []
if True:
    num_tokens = 256

    configs.append(
        TrainConfig(
            name=f"{file_name}_nt{num_tokens}_{data_config.hash()}_files{data_config.num_files}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                # model_cls=LlamaForCausalLM,
                # attn_implementation="einsum",
            ),
            dataset=CapsuleDatasetLatest.Config(
                data_sources=[
                    (source, None)
                    for slc in SLICES
                    for source in data_sources[slc]
                ],
                max_sequence_length=1024,
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),
            generate_every_n_steps=4,
            generate_datasets=[
                GenerateDatasetConfig(
                    dataset=CodeHopGenerateDataset.Config(
                        code_hop_config=data_config,
                    ),
                    name_for_wandb="code_hop",
                    batch_size=1,
                ),
            ],
            eval_every_n_steps=64,
            eval_datasets=[
                # *mtob_eval_datasets(
                #     local_batch_size=16,
                # ),
                # EvalDatasetConfig(
                #     name_for_wandb="mmlu",
                #     local_batch_size=16,
                #     dataset=MMLUEvalDataset.Config(num_samples=128),
                # ),
            ],
            generate_max_new_tokens=4,
            kv_cache_initializer=KVFromRandomText.Config(max_tokens=num_tokens),
            loss_type="logits",
            save_every_n_steps=128,
            epochs=64,
            online_model=False,
            lr=0.01,
            # lr_scheduler=CosWithWarmup.Config(
            #     max_steps=100, warmup_steps=5, warmup_min_lr=0.001
            # ),
            wandb=WandBConfig(
                project="capsules",
                tags=["train", "codehop"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=2,
            # distributed_backend="nccl",
        )
    )

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])
