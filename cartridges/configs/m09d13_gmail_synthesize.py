import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.chunkers import TokenChunker
from cartridges.data.gmail.resources import GmailResource
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.resources import TextFileResource
from cartridges.clients.tokasaurus import TokasaurusClient

# client = TokasaurusClient.Config(
#     url="https://hazyresearch--toka-qwen3-4b-1xh100-cartridges-serve.modal.run",
#     model_name="Qwen/Qwen3-4b",
# )

# client = TokasaurusClient.Config(
#     url="https://hazyresearch--toka-qwen3-4b-instr-2507-1xh100-cartridges-serve.modal.run",
#     model_name="Qwen/Qwen3-4B-Instruct-2507"
# )

client = TokasaurusClient.Config(model_name="meta-llama/Llama-3.1-8B-Instruct")

config = SynthesizeConfig(

    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        tools=[],
        resources=[
            GmailResource.Config(
                labels=["categories--stanford--primary--stanford-"],
                date_start="2025/09/01",
                date_end="2025/09/14",
            )
        ],
    ),

    num_samples=65_536, 
    batch_size=32,  
    max_num_batches_in_parallel=256,

    name=FormatStringVariable(f"{Path(__file__).stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    
    upload_to_wandb=False,
    save_wandb_preview=False,
    
    upload_to_hf=False,
    hf_repo_id="hazyresearch/{wandb_run_id}",
)


if __name__ == "__main__": 
    pydrantic.main([config])