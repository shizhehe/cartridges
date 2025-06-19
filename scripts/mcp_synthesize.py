import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.clients.sglang import SGLangClient
from cartridges.contexts.mcp.gmail import GmailMCPContext
from cartridges.contexts.mcp.slack import SlackMCPContext
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study_mcp import MCPSelfStudySynthesizer, SlicePromptSamplerWithChunks
from cartridges.utils import WandBConfig

from cartridges.contexts.tex import TexDocument

ARXIV_ID = "2506.06266"
context = SlackMCPContext.Config(
    bot_token=os.environ["SLACK_BOT_TOKEN"],
    team_id=os.environ["SLACK_TEAM_ID"]
)

context = GmailMCPContext.Config(
    email="eyuboglu@stanford.edu"
)

client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


config = SynthesizeConfig(
    context=context,

    synthesizer=MCPSelfStudySynthesizer.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        # prompt_sampler=SlicePromptSamplerWithChunks.Config(
        #     slices=[
        #         "structuring",
        #         "summarization",
        #         "question",
        #         "use_case",
        #         "creative",
        #     ],
        #     min_chunk_size=512,
        #     max_chunk_size=4096,
        #     desc=f"Below is a research paper on Cartridges."
        # ),
        prob_cot_a=0.2,
        use_tools=False, 
        tools=[]
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=16,
    batch_size=4,
    
    max_num_batches_in_parallel=4,
    handle_exceptions=False,
    parallelism_strategy="async",

    name=FormatStringVariable(f"arxiv_{ARXIV_ID}_synthesize"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"arxiv", "generate", ARXIV_ID],
    ),
    save_wandb_artifact=True,

)


if __name__ == "__main__": 
    pydrantic.main([config])
