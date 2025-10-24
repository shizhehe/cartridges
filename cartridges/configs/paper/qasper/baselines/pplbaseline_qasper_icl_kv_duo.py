import os
from pathlib import Path
import socket

from capsules.icl_baseline_kv_cache_compression import ICLBaselineWithKVCacheCompressionConfig
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.icl_baseline import ICLBaselineConfig
from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.models.llama import LlamaForCausalLM
from capsules.tasks.qasper.context import QasperStructuredContextConfig
from capsules.tasks.qasper.dataset import QasperEvalDataset
from capsules.train import EvalDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


TOPIC = "question"


if TOPIC == "question":
    data_sources = [
        "hazy-research/capsules/generate_qasper_simple_question_s5_n65536:v1",
    ]
else:
    raise ValueError(f"Invalid topic: {TOPIC}")

SYSTEM_PROMPT_TEMPLATE = """{content}"""

configs = [
    ICLBaselineWithKVCacheCompressionConfig(
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
            project="capsules",
            tags=["ppl-baseline", "qasper", f"qasper-{TOPIC}"],
            entity="hazy-research",
        ),
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        distributed_backend="gloo" if socket.gethostname().startswith("mk-ix-") else "nccl",

        kv_compression="expected_attention",
        kv_compression_ratio=ratio,
    )
    for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
]

if __name__ == "__main__":
    pydrantic.main(configs)
