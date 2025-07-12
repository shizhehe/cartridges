import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.data.text import StaticTextResource
from cartridges.data.code import CodeResource
from cartridges.data.slack import SlackResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils import WandBConfig
from cartridges.data.retrieval import RetrievalTool, BM25Retriever, SourceConfig
from cartridges.data.retrieval.tools import AMD_TK_SOURCES


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        prob_cot_a=0.2,
        use_tools_a=True, 
        use_tools_b=True,
        tools=[
            RetrievalTool.Config(
                sources=AMD_TK_SOURCES,
                retriever=BM25Retriever.Config(
                    k1=1.5,
                    b=0.75,
                    epsilon=0.25,
                )
            )
        ],
        resources=[
            # StaticTextResource.Config(
            #     path="codemem/focus/database/video/amd_convo_062025.txt"
            # )
            # CodeResource.Config(
            #     path="/home/sabri/code/code-memory/codemem/focus/database/code"
            # ),
            SlackResource.Config(
                path="cartridges/resources/database/slack"
            )
        ],
    ),
    output_dir=os.environ.get("MEMORY_OUTPUT_DIR", "."),
    num_samples=16,
    batch_size=4,
    
    max_num_batches_in_parallel=1,

    name=FormatStringVariable(f"synthesize"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"synthesize"],
    ),
    save_wandb_artifact=True,

)


if __name__ == "__main__": 
    pydrantic.main([config])
