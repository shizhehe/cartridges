from capsules.analysis.figures.quality_v_cache_size_per_dataset import (
    QualityVsCacheSizePerDatasetPlotter,
)
from capsules.analysis.figures.quality_v_cache_size import (
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
        "hazy-research/capsules/9ct54wed", # 512
        "hazy-research/capsules/sc2bhdsx", # 1024
        "hazy-research/capsules/48yw22k9", # 2048
        "hazy-research/capsules/ax4cg2aa", # 4096
        "hazy-research/capsules/tkhajsus", # 8192
    ],
    hue="Cache Tuning",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/capsules/bqjoo55z", # 512
    ],
    hue="ICL Baseline",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)


run_sets = [
    cache_tuning,
    baseline_runs, 
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="perplexity",
    dataset_type="eval",
    excluded_datasets=[],
    show_baseline_points=True,
    y_label="Perplexity (←)",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
)

if __name__ == "__main__":
    pydrantic.main([
        avg_plot
    ])





