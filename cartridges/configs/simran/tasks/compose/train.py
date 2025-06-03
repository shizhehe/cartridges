import os
from pathlib import Path
import pydrantic


from capsules.optim import CosWithWarmup
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from capsules.config import HFModelConfig
from capsules.datasets import ( CapsuleDatasetLatest )
from capsules.utils import WandBConfig




file_name = Path(__file__).stem


configs = []
bs = 64


# Experiment: Cartridge V1 SimplePromptSampler
dataset_sources = [
   ("/root/simran/capsules/2025-05-07-01-22-04-m05d06_generate_tk/37573bb3-6507-480c-8e95-1b08c1d36c08/artifact/dataset.pkl", None),
]


configs = []


for num_tokens, lr in [(1024, 0.005)]:
   configs.append(


       TrainConfig(
           name=f"{file_name}_nt{num_tokens}_auto_lr{lr}",
           model=HFModelConfig(
               pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
           ),


           lr=lr,
           # lr_scheduler=CosWithWarmup.Config(
           #     warmup_steps=128,
           #     max_steps=2048,
           #     warmup_min_lr=5e-3,
           #     alpha_f=0.1,
           # ),


           dataset=CapsuleDatasetLatest.Config(
               data_sources=dataset_sources,
               is_wandb=True,
               label_type="logits",
               top_k_logits=20,
           ),


           generate_max_new_tokens=512,
           kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
               max_tokens=num_tokens
           ),


           loss_type="logits",
           save_every_n_steps=128,
           epochs=2,
          
           wandb=WandBConfig(
               project="capsules",
               tags=["compose", "auto_train"],
               entity="hazy-research",
           ),
           output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
           global_batch_size=bs,
           local_batch_size=4,


           distributed_backend="gloo",
       )
   )


if __name__ == "__main__":
   for config in configs:
       pydrantic.main([config])


