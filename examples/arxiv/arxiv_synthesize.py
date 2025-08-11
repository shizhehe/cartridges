import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.chunkers import TokenChunker
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils import WandBConfig
from cartridges.data.tex.resources import LaTeXResource
from cartridges.clients.tokasaurus import TokasaurusClient

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
    model_name="Qwen/Qwen3-4b",
)

config = SynthesizeConfig(

    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        tools=[],
        resources=[
            LaTeXResource.Config(
                arxiv_id="2506.06266",
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                chunker=TokenChunker.Config(
                    tokenizer=client.model_name,
                    min_tokens_per_chunk=512,
                    max_tokens_per_chunk=1024,
                ),
            )
        ],
    ),

    num_samples=8192, 
    batch_size=32,  
    max_num_batches_in_parallel=256,

    name=FormatStringVariable(f"{Path(__file__).stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"arxiv_synthesis"],
    ),
    save_wandb_artifact=False,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    pydrantic.main([config])