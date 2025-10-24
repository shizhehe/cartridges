import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.ruler.resources import NIAHResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.utils.wandb import WandBConfig
from cartridges.utils import short_model_name



# client = TokasaurusClient.Config(
#     url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
#     model_name="Qwen/Qwen3-4b",
# )

client = TokasaurusClient.Config(
    url=os.environ["CARTRIDGES_TOKASAURUS_LLAMA_3_2_3B_URL"],
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

NUM_KEYS = (1, 1)
num_keys_str = f"k{NUM_KEYS[0]}_{NUM_KEYS[1]}"

BASE_PATH = os.path.join(
    os.environ["CARTRIDGES_DIR"], "cartridges/data/ruler/_data"
)
NUM_KEYS_TO_PATH = {
    (1, 1): f"{BASE_PATH}/llama_3.2_3b_instruct-l128000-n1-k128-v1_1-essay-key_words-val_numbers-74198fb4.json"
}


niah_path = NUM_KEYS_TO_PATH[NUM_KEYS]

config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        use_tools_a=False, 
        use_tools_b=False,
        # max_completion_tokens_b=256,
        tools=[],
        resources=[
            NIAHResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                niah_path=niah_path,
                sentences_per_chunk=(1, 1),
                chunks_per_prompt=(64, 256),
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536, 
    batch_size=32,    # Smaller batches 
    
    max_num_batches_in_parallel=256,

    name=FormatStringVariable(f"{Path(__file__).stem}_{short_model_name(client.model_name)}_n{{num_samples}}_{num_keys_str}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"longhealth_synthesis"],
    ),
    upload_to_wandb=False,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    
    pydrantic.main([config])

