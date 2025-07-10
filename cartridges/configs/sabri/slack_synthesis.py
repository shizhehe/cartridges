import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.resources.text import StaticTextResource
from cartridges.resources.code import CodeResource
from cartridges.resources.slack import SlackResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils import WandBConfig
from cartridges.tools.retrieval import RetrievalTool, BM25Retriever, SourceConfig
from cartridges.tools.retrieval.tools import AMD_TK_SOURCES
from cartridges.tools.slack.tools import SlackToolSet


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=2,
        prob_cot_a=0.3,
        use_tools_a=False, 
        use_tools_b=True,
        tools=[
            # Keep existing retrieval tool
            # RetrievalTool.Config(
            #     sources=AMD_TK_SOURCES,
            #     retriever=BM25Retriever.Config(
            #         k1=1.5,
            #         b=0.75,
            #         epsilon=0.25,
            #     )
            # ),
            # Add new Slack tool
            SlackToolSet.Config(
                # bot_token and team_id will be read from environment
            )
        ],
        resources=[
            # Include Slack resource for context
            SlackResource.Config(
                path="cartridges/resources/database/slack"
            ),
            # Optionally include code resource for richer synthesis
            # CodeResource.Config(
            #     path="/home/sabri/code/code-memory/codemem/focus/database/code"
            # ),
        ],
    ),
    output_dir=os.environ.get("MEMORY_OUTPUT_DIR", "."),
    num_samples=4,  # Slightly reduced for faster testing
    batch_size=4,    # Smaller batches for more manageable parallel processing
    
    max_num_batches_in_parallel=1,

    name=FormatStringVariable(f"slack_synthesis"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"slack_synthesis", "tools", "mcp"],
    ),
    save_wandb_artifact=True,

)


if __name__ == "__main__": 
    pydrantic.main([config])