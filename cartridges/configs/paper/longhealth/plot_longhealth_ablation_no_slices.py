from capsules.analysis.figures.quality_v_compute_ablation import (
    PlotRunSetConfig,
    QualityVsComputePlotter,
)
import pydrantic
from capsules.utils.cache_size import MODEL_TO_CACHE_SIZE_FN


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
        
        # "hazy-research/capsules/4a7ulyib", # 128
        # "hazy-research/capsules/abheqepq", # 256
        # "hazy-research/capsules/4rlvsxso", # 512
        # "hazy-research/capsules/u8s4oe0y", # 1024
        "hazy-research/capsules/wauoq23f", # 2048
        # "hazy-research/capsules/rz5wgj2l", # 4096
        # "hazy-research/capsules/xd1b7oi7", # 8192
        
    ],
    hue=lambda row: f"b{row['num_trainable_tokens']:04d}",
)

cache_tuning_no_logits = PlotRunSetConfig(
    wandb_run_ids=[
        
        # "hazy-research/capsules/", # 128
        # "hazy-research/capsules/", # 256
        # "hazy-research/capsules/", # 512
        # "hazy-research/capsules/ruaqfcsn", # 2048
        "hazy-research/capsules/wi8pf5sy", # 1024
        # "hazy-research/capsules/", # 4096
        # "hazy-research/capsules/", # 8192
        
    ],
    hue=lambda row: f"a{row['num_trainable_tokens']:04d}",
)

icl_baseline = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/gcc2a7pa"],
    hue="icl",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)

run_sets = [
    cache_tuning,
    # icl_baseline, 
    cache_tuning_no_logits
]

avg_plot = QualityVsComputePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    excluded_datasets=[],
    y_label="Accuracy",
    x_label="Train Steps",
    hue_label="Method",
    bin_size=1024,
    run_id="avg_plot",
    legend_loc="best",
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
