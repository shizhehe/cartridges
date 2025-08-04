import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.ruler.resources import VariableTrackingResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils import WandBConfig
from cartridges.configs.utils import short_model_name

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-llama-3-2-3b-instruct-1xh100-min0-serve.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

# Use the default variable tracking dataset path
BASE_PATH = "/home/sabri/code/cartridges/cartridges/data/ruler/_data"
# VT_PATH = f"{BASE_PATH}/qwen3_4b-l100000-n1-c64-h2-essay-979310a3.json"
# VT_PATH = f"{BASE_PATH}/llama_3.2_3b_instruct-l100000-n1-c64-h2-essay-7ba69bcb.json"
VT_PATH = f"{BASE_PATH}/llama_3.2_3b_instruct-l10000-n1-c16-h2-essay-words-1d31e1f5.json"
config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.5,
        use_tools_a=False, 
        use_tools_b=False,
        tools=[],
        resources=[
            VariableTrackingResource.Config(
                seed_prompts=[
                    "structuring",
                    # "summarization",
                    "question",
                    "use_case",
                    # "creative",
                ],
                variable_tracking_path=VT_PATH,
                sentences_per_chunk=(1, 1),
                chunks_per_prompt=(1, 8),
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536, 
    batch_size=8,
    
    max_num_batches_in_parallel=256,

    name=FormatStringVariable(f"{Path(__file__).stem}_{short_model_name(client.model_name)}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"variable_tracking_synthesis"],
    ),
    save_wandb_artifact=False,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    pydrantic.main([config])