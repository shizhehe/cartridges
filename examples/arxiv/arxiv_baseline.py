import os
from pathlib import Path
import pydrantic

from cartridges.initialization import KVFromText
from cartridges.initialization.pretrained import KVFromPretrained
from cartridges.train import TrainConfig, LossEvalConfig, GenerationEvalConfig
from cartridges.evaluate import LossEvalRunConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource, GenerateEvalDataset, TrainDataset, LossEvalDataset


text = Path(os.path.join(os.environ["CARTRIDGES_DIR"], "examples/arxiv/cartridges.tex")).read_text()

system_prompt = f"You are a helpful assistant that can answer questions about the following text: {text}"


config = LossEvalRunConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    ),


    
    batch_size=16,

    eval=LossEvalConfig(
        dataset=LossEvalDataset.Config(
            data_source=DataSource(
                path="hazyresearch/arxiv_synthesize_eval_gpt-5-mini-2025-08-07_n32-0",
                type="hf",
            ),

            system_prompt=system_prompt, # this system prompt is prepended to every question
            packed_seq_length=32_000,  # this needs to be longer than the system prompt + questions

        ),
        name_for_wandb="arxiv_synthesize",
    ),

    name="cartridges-tutorial-baseline",
)


if __name__ == "__main__":
    pydrantic.main(config)