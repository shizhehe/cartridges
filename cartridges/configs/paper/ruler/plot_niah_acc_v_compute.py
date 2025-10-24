from cartridges.analysis.figures.quality_v_compute import (
    PlotRunSetConfig,
    QualityVsComputePlotter,
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

        
        # "hazy-research/cartridges/pc5pc5r3", # 64
        # # "hazy-research/cartridges/u3c2r525", # 128
        # "hazy-research/cartridges/31n23c97", # 256
        # # "hazy-research/cartridges/k8z0gc3a", # 512
        # "hazy-research/cartridges/n4x76fsc", # 1024
        # # "hazy-research/cartridges/n99aewpc", # 2048
        # "hazy-research/cartridges/69kgmdq5", # 4096

        "hazy-research/cartridges/iesajyv1", # 64
        "hazy-research/cartridges/9at1w8qx", # 256
        "hazy-research/cartridges/xrx2jxex", # 1024
        "hazy-research/cartridges/2sjvzaas", # 4096
        
    ],
    hue=lambda row: row['num_trainable_tokens'],
)

icl_baseline = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/cartridges/y1fds7j6"],
    hue="icl",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)

prompt_compression_summary = PlotRunSetConfig(
    launch_ids=["2025-09-24-13-08-30-m09d23_niah_baseline"],
    hue="summary",
    baseline=True,
)

run_sets = [
    cache_tuning,
    icl_baseline, 
    prompt_compression_summary
]

avg_plot = QualityVsComputePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    excluded_datasets=[],
    y_label="Accuracy",
    x_label="Train Steps",
    hue_label="Method",
    bin_size=64,
    run_id="avg_plot",
    legend_loc="best",
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
