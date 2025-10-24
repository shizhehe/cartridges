from cartridges.analysis.figures.quality_v_cache_size_w_groups import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

from cartridges.utils.cache_size import MODEL_TO_CACHE_SIZE_FN

def convert_x_to_bytes(row, x):
    model_name = row["model.pretrained_model_name_or_path"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name.lower()](x)
    return x_in_bytes / (1e9)

def convert_x_to_bytes_baseline(row, x):
    model_name = row["generator.client.model_name"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name.lower()](x)
    return x_in_bytes / (1e9)

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        # "hazy-research/capsules/pl0rrcgu", # 512
        # "hazy-research/capsules/ra8slhvl", # 1024
        # "hazy-research/capsules/msad7zj8", # 2048
        # "hazy-research/capsules/hltr6yg3", # 4096
        # "hazy-research/capsules/c5tyjmep", # 8192
        
        "hazy-research/cartridges/pc5pc5r3", # 64
        "hazy-research/cartridges/u3c2r525", # 128
        "hazy-research/cartridges/31n23c97", # 256
        "hazy-research/cartridges/k8z0gc3a", # 512
        "hazy-research/cartridges/n4x76fsc", # 1024
        "hazy-research/cartridges/n99aewpc", # 2048
        "hazy-research/cartridges/69kgmdq5", # 4096
        # "hazy-research/cartridges/", # 8192

    ],
    hue="cartridge",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)


icl_baseline = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/y1fds7j6"],
    hue="icl",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)

# kv_cache_compression_expected = PlotRunSetConfig(
#     launch_ids=["2025-05-11-00-06-06-llama_3b"],
#     hue="KV Cache Compression (Expected)",
#     modify_x=lambda row, x: row[f"generate_longhealth_mc/kv_cache_size_bytes"] / 1e9,
# )

kv_cache_compression_first_k_tokens = PlotRunSetConfig(
    launch_ids=[
        "2025-09-24-13-31-23-m09d23_niah_baseline",
        "2025-09-24-12-32-45-m09d23_niah_baseline" # include the full icl baselnie

    ],
    hue="first_k_tokens",
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)

kv_cache_compression_duo = PlotRunSetConfig(
    launch_ids=[
        # "2025-05-11-14-28-25-llama_3b_duo_on_the_fly"
        # "2025-05-12-18-48-25-llama_3b_duo_on_the_fly",
        "2025-09-24-16-16-55-m09d23_niah_baseline",
        "2025-09-24-16-45-24-m09d23_niah_baseline",
    ],
    hue="duo",
    modify_x=lambda row, x: row[f"generate_niah_mc/kv_cache_size_bytes"] / 1e9,
)


prompt_compression_summary = PlotRunSetConfig(
    launch_ids=[
        "2025-09-24-13-08-30-m09d23_niah_baseline"
    ],
    hue="summary",
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)


run_sets = [
    cache_tuning,
    icl_baseline, 
    kv_cache_compression_first_k_tokens,
    kv_cache_compression_duo,
    prompt_compression_summary,
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    excluded_datasets=[],
    show_baseline_points=True,
    y_label="Accuracy",
    x_scale_base=2,
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
