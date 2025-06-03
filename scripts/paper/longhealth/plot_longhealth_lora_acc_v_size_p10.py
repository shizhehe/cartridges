import math
from cartridges.analysis.figures.quality_v_cache_size_w_groups import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

from cartridges.utils.cache_size import MODEL_TO_CACHE_SIZE_FN

def convert_x_to_bytes(row, x):
    model_name = row["model.pretrained_model_name_or_path"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

def convert_x_to_bytes_baseline(row, x):
    model_name = row["generator.client.model_name"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/kslaoop1", # 256
        "hazy-research/Cartridges/1jo1jstu", # 512
        "hazy-research/Cartridges/wbg6gk3m", # 1024
        "hazy-research/Cartridges/z8jfj8io", # 2048
        "hazy-research/Cartridges/xoajifmc", # 4096
        "hazy-research/Cartridges/jokb5jx8", # 8192
    ],
    hue="cartridge",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    shade=lambda row: - math.log(row["num_trainable_tokens"], 2),
    filter=lambda row: row["state"] == "finished",
)

lora = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/2ya9c8c4", # 51
        "hazy-research/Cartridges/978f5vpk", # 102
        "hazy-research/Cartridges/3dd9eh2h", # 204
        "hazy-research/Cartridges/yxmsq2kj", # 408
        "hazy-research/Cartridges/g8r701mf", # 816
        "hazy-research/Cartridges/kdcq4w8g", # 1632
    ],
    hue="lora_rank",
    modify_x=lambda row, x: convert_x_to_bytes(row, x) + row["num_trainable_params"] * 2 / 1e9,
    shade=lambda row: - math.log(row["model.peft.r"], 2),
    filter=lambda row: row["state"] == "finished",

)

# icl_baseline = PlotRunSetConfig(
#     wandb_run_ids=["hazy-research/Cartridges/gcc2a7pa"],
#     hue="icl",
#     baseline=True,
#     modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
# )

# kv_cache_compression_expected = PlotRunSetConfig(
#     launch_ids=["2025-05-11-00-06-06-llama_3b"],
#     hue="KV Cache Compression (Expected)",
#     modify_x=lambda row, x: row[f"generate_longhealth_mc/kv_cache_size_bytes"] / 1e9,
# )


run_sets = [
    cache_tuning,
    lora,
]

configs = []

for excluded in ["mmlu", "longhealth_mc"]:
    avg_plot = QualityVsCacheSizePlotter.Config(
        run_sets=run_sets,
        y_metric="score",
        dataset_type="generate",
        excluded_datasets=[excluded],
        show_baseline_points=True,
        x_scale_base=2,
        y_label="Accuracy",
        x_label="Cache Size (GB)",
        hue_label="Method",
        run_id="avg_plot",
        legend_loc="best",
    )
    configs.append(avg_plot)

if __name__ == "__main__":
    pydrantic.main(configs)
