import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.icl_baseline import ICLBaselineConfig
from cartridges.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
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


if TOPIC == "question":
    data_sources = [
        "hazy-research/Cartridges/generate_qasper_simple_question_s5_n65536:v1",
    ]
else:
    raise ValueError(f"Invalid topic: {TOPIC}")

SYSTEM_PROMPT_TEMPLATE = """{content}"""

configs = [
    ICLBaselineConfig(
        name=FormatStringVariable(f"{file_name}_{TOPIC}"),
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
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        distributed_backend="gloo" if socket.gethostname().startswith("mk-ix-") else "nccl",
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
