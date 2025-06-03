import os
from pathlib import Path
import pydrantic

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.train import GenerateDatasetConfig
from capsules.utils import WandBConfig

from capsules.tasks.thunderkittens.cuda_context import CUDAContextConfig
from capsules.tasks.thunderkittens.kernelbench_evals import ( KernelBenchGenerateDataset )


file_name = Path(__file__).stem


configs = []
bs = 64

model_path = "meta-llama/Llama-3.2-3B-Instruct"
model_path = "meta-llama/Llama-3.1-8B-Instruct"


tokasaurus_client = TokasaurusClient.Config(
    # url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the information below to answer the questions.

<reference>
{{content}}
</reference>
"""

for num_tokens in [131702]:
    configs.append(

        GenerateBaselineConfig(
            generator=ICLBaseline.Config(
                client=tokasaurus_client,
                system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
                max_completion_tokens=1024,

            ),
            dataset=GenerateDatasetConfig(
                dataset=KernelBenchGenerateDataset.Config(
                    max_questions=5,
                    mode = "cuda",
                ),
                name_for_wandb="kernel-bench-ppl",
            ),
            batch_size = 1,
            context=CUDAContextConfig(
                mode = "icl",
            ),
            wandb=WandBConfig(
                project="capsules",
                tags=["thunderkittens", "icl"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        )
    )

if __name__ == "__main__":
    config_idx = 0
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])


