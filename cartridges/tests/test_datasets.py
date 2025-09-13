import os
import pytest
from transformers import AutoTokenizer

from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.datasets import TrainDataset, DataSource
from cartridges.structs import write_conversations
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.clients.tokasaurus import TokasaurusClient

# EXISTING_DATASOURCE = None
EXISTING_DATASOURCE = DataSource(
    path="codehop_synthesize_qwen-8b_repo-244c02_level1_n65768_e0.5:v1", type="wandb"
)
TOKENIZER = "Qwen/Qwen3-4b"

@pytest.fixture(scope="session")
@pytest.mark.asyncio
async def synthetic_conversations(tmp_path_factory):
    """Fixture for TokasaurusClient config."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    if EXISTING_DATASOURCE is not None:
        return EXISTING_DATASOURCE, tokenizer
    
    

    client = TokasaurusClient.Config(
        url=os.environ.get("CARTRIDGES_TOKASAURUS_QWEN3_4B_URL", "http://localhost:8000"),
        model_name="Qwen/Qwen3-4b",
    )

    client = TokasaurusClient.Config(
        # url="https://hazyresearch--toka-llama-3-2-3b-1xh100-batch-serve.modal.run",
        # url="https://hazyresearch--toka-llama-3-2-3b-1xh100-cartridges-serve.modal.run",
        # url="https://hazyresearch--toka-llama-3-2-3b-1xh100-main-serve.modal.run",
        url="http://0.0.0.0:10210",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        cartridges=[]
    )

    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        tools=[],
        resources=[
            TextFileResource.Config(
                path=os.path.join(os.environ["CARTRIDGES_DIR"], "examples/arxiv/cartridges.tex"),
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                chunker=TokenChunker.Config(
                    tokenizer=client.model_name,
                    min_tokens_per_chunk=2048,
                    max_tokens_per_chunk=4096,
                ),
            )
        ],
    )
    synthesizer = synthesizer.instantiate()
    await synthesizer.setup()  # this is needed to run async steps like starting MCP clients

    convos = await synthesizer.sample_convos(0, 32, 1)

    tmp_dir = tmp_path_factory.mktemp("synthetic_data")
    path = tmp_dir / "test_synthesized_conversations.parquet"
    write_conversations(convos, str(path))
    return DataSource(path=str(path), type="local"), tokenizer


@pytest.mark.asyncio
async def test_no_thinking(synthetic_conversations: DataSource):
    synthetic_conversations, tokenizer  = await synthetic_conversations
    dataset = TrainDataset.Config(
        data_sources=[synthetic_conversations],
        is_wandb=False,
        top_k_logits=20,
        targets="logits",
        prob_drop_thinking=1.0,
        packing_mode="pad",
        packed_seq_length=2048,
    )
    dataset = dataset.instantiate()
    await dataset.setup(tokenizer=tokenizer, seed=42)
    breakpoint() 




