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
from cartridges.data.slack.slack import SlackToolSet
from cartridges.data.gmail.gmail import GmailToolSet


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=3,  # More rounds to explore multiple tools
        prob_cot_a=0.4,  # Higher CoT for richer reasoning
        use_tools_a=True, 
        use_tools_b=True,
        tools=[
            # AMD ThunderKittens retrieval
            RetrievalTool.Config(
                sources=AMD_TK_SOURCES,
                retriever=BM25Retriever.Config(
                    k1=1.5,
                    b=0.75,
                    epsilon=0.25,
                )
            ),
            # Slack integration
            SlackToolSet.Config(
                # bot_token and team_id will be read from environment
            ),
            # Gmail integration  
            GmailToolSet.Config(
                email="sabri@stanford.edu"
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
    num_samples=8,   # Reduced for more manageable multi-tool testing
    batch_size=2,    # Smaller batches due to multiple tools
    
    max_num_batches_in_parallel=1,

    name=FormatStringVariable(f"multi_tool_synthesis"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"multi_tool_synthesis", "slack", "gmail", "retrieval", "mcp"],
    ),
    save_wandb_artifact=True,

)


if __name__ == "__main__": 
    pydrantic.main([config])