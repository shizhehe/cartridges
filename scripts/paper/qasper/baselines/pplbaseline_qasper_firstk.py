import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.icl_baseline import FirstKTokensTransform, ICLBaselineConfig
from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.models.llama import LlamaForCausalLM
from cartridges.tasks.qasper.context import QasperStructuredContextConfig
from cartridges.tasks.qasper.dataset import QasperEvalDataset
from cartridges.train import EvalDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


TOPIC = "question"


SYSTEM_PROMPT_TEMPLATE = """{content}"""

configs = [
    ICLBaselineConfig(
        name=FormatStringVariable(f"{file_name}_{TOPIC}_k{k_tokens}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        ),

        context=QasperStructuredContextConfig(topic=TOPIC),

        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb="qasper_rewrite",
                local_batch_size=1,
                dataset=QasperEvalDataset.Config(
                    dataset="sabrieyuboglu/qasper-rewrite-gpt-4.1",
                    topic=TOPIC,
                    label_type="tokens",
                    data_sources=[]  # ignore this arg
                )
            )
        ],

        wandb=WandBConfig(
            project="cartridges",
            tags=["ppl-baseline", "qasper", f"qasper-{TOPIC}"],
            entity="hazy-research",
        ),
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        context_transform=FirstKTokensTransform.Config(
            first_k_tokens=k_tokens,
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        distributed_backend="gloo" if socket.gethostname().startswith("mk-ix-") else "nccl",
    )
    for k_tokens in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
]

if __name__ == "__main__":
    pydrantic.main(configs)
