from cartridges.analysis.figures.quality_v_cache_size_per_dataset import (
    QualityVsCacheSizePerDatasetPlotter,
)
from cartridges.analysis.figures.quality_v_cache_size import (
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
        # "hazy-research/Cartridges/pl0rrcgu", # 512
        # "hazy-research/Cartridges/ra8slhvl", # 1024
        # "hazy-research/Cartridges/msad7zj8", # 2048
        # "hazy-research/Cartridges/hltr6yg3", # 4096
        # "hazy-research/Cartridges/c5tyjmep", # 8192
        
        "hazy-research/Cartridges/4a7ulyib", # 128
        "hazy-research/Cartridges/abheqepq", # 256
        "hazy-research/Cartridges/4rlvsxso", # 512
        "hazy-research/Cartridges/u8s4oe0y", # 1024
        "hazy-research/Cartridges/wauoq23f", # 2048
        "hazy-research/Cartridges/rz5wgj2l", # 4096
        "hazy-research/Cartridges/xd1b7oi7", # 8192
        
    ],
    hue="Cache Tuning",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

# openai_runs = PlotRunSetConfig(
#     launch_ids=[
#         "2025-04-23-09-18-08-m04d23_genbaseline_longhealth_rag_px",
#         "2025-04-23-10-24-57-m04d23_genbaseline_longhealth_rag_px",
#         "2025-04-23-10-53-58-m04d23_genbaseline_longhealth_rag_px"
#     ],
#     hue="RAG Baseline",
#     modify_x=lambda row, x: convert_x_to_bytes(row, x),
# )

baseline_runs = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/Cartridges/gcc2a7pa"],
    hue="ICL Baseline",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)

kv_cache_compression_expected = PlotRunSetConfig(
    launch_ids=["2025-05-11-00-06-06-llama_3b"],
    hue="KV Cache Compression (Expected)",
    modify_x=lambda row, x: row[f"generate_longhealth_mc/kv_cache_size_bytes"] / 1e9,
)

run_sets = [
    cache_tuning,
    # cache_tuning_old,
    baseline_runs, 
    # bm_25_runs,
    # openai_runs,
    kv_cache_compression_expected,
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    excluded_datasets=[],
    show_baseline_points=True,
    y_label="Accuracy",
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
