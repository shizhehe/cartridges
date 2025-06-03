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

cache_tuning_mmlu = PlotRunSetConfig(
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
    filter=lambda row: row["state"] == "finished",
)

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        # "hazy-research/Cartridges/pl0rrcgu", # 512
        # "hazy-research/Cartridges/ra8slhvl", # 1024
        # "hazy-research/Cartridges/msad7zj8", # 2048
        # "hazy-research/Cartridges/hltr6yg3", # 4096
        # "hazy-research/Cartridges/c5tyjmep", # 8192
        
        # "hazy-research/Cartridges/4a7ulyib", # 128
        "hazy-research/Cartridges/abheqepq", # 256
        "hazy-research/Cartridges/4rlvsxso", # 512
        "hazy-research/Cartridges/u8s4oe0y", # 1024
        "hazy-research/Cartridges/wauoq23f", # 2048
        "hazy-research/Cartridges/rz5wgj2l", # 4096
        "hazy-research/Cartridges/xd1b7oi7", # 8192
        
    ],
    hue="cartridge",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

no_freeze = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/cx5y33l6", # 256
        "hazy-research/Cartridges/xfo449d3", # 512
        "hazy-research/Cartridges/tysme16o", # 1024
        "hazy-research/Cartridges/j9ln7r8y", # 2048
        "hazy-research/Cartridges/jv73wvdm", # 4096
        "hazy-research/Cartridges/2wvdfy7a", # 8192
    ],
    hue="no_freeze",
    modify_x=lambda row, x: convert_x_to_bytes(row, x) + row["num_trainable_params"] * 2 / 1e9,
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
    no_freeze,
]

configs = []

for excluded in ["mmlu", "longhealth_mc"]:
    avg_plot = QualityVsCacheSizePlotter.Config(
        run_sets=(
            [cache_tuning if excluded == "mmlu" else cache_tuning_mmlu, no_freeze]
        ),
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
