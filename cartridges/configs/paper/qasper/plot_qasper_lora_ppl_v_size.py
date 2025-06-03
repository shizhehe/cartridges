
import math
from capsules.analysis.figures.quality_v_cache_size_w_groups import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

from capsules.utils.cache_size import MODEL_TO_CACHE_SIZE_FN

def convert_x_to_bytes(row, x):
    model_name = row["model.pretrained_model_name_or_path"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

def convert_x_to_bytes_baseline(row, x):
    model_name = "meta-llama/Llama-3.2-3B-Instruct" #row["generator.client.model_name"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        # "hazy-research/capsules/fmau6q7u",  # 256
        # "hazy-research/capsules/612em9sl",  # 512
        # "hazy-research/capsules/jh3y7bpo",  # 1024
        # "hazy-research/capsules/25piv8uy",  # 2048
        # # "hazy-research/capsules/wcn8bowf", # 4096
        # "hazy-research/capsules/m05mduy8", # 4096
        # # "hazy-research/capsules/kile5m48",  # 8192
        # "hazy-research/capsules/44xxqf78", # 8192


        "hazy-research/capsules/vx150k9s",  # 256
        "hazy-research/capsules/d03tbnhq",  # 512
        "hazy-research/capsules/k9yhwis6",  # 1024
        "hazy-research/capsules/jyosinu3",  # 2048
        "hazy-research/capsules/v4ffr7l4", # 4096
        "hazy-research/capsules/gage720m", # 8192
  
    ],
    hue="cartridge",
    shade=lambda row: - math.log(row["num_trainable_tokens"], 2),
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

lora = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/capsules/osfr5u45", # 51
        "hazy-research/capsules/zsd8zqnl", # 102
        "hazy-research/capsules/lojn9146", # 204
        "hazy-research/capsules/7s84pyxg", # 408
        "hazy-research/capsules/47a5wo5e", # 816
        "hazy-research/capsules/itd8p6rj", # 1632
    ],
    hue="lora_rank",
    modify_x=lambda row, x: convert_x_to_bytes(row, x) + row["num_trainable_params"] * 2 / 1e9,
    filter=lambda row: row["state"] == "finished",
    shade=lambda row: - math.log(row["model.peft.r"], 2),
)


run_sets = [
    cache_tuning,
    lora,
]


mmlu_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    include_datasets=["mmlu"],
    show_baseline_points=True,
    y_label="Accuracy",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
    # y_min=0.0,
    # y_max=2.28,
)

mmlu_ppl_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="loss",
    dataset_type="eval",
    include_datasets=["mmlu"],
    show_baseline_points=True,
    y_label="log (Perplexity) (←)",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
    # y_min=0.0,
    # y_max=2.28,
)

qasper_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="macro_loss",
    dataset_type="eval",
    include_datasets=["qasper_rewrite"],
    show_baseline_points=True,
    y_label="log (Perplexity) (←)",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
    # y_min=0.0,
    # y_max=2.28,
)


if __name__ == "__main__":
    pydrantic.main([
        qasper_plot,
        mmlu_ppl_plot,
        mmlu_plot,
    ])





