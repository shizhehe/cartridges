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
        
        # "hazy-research/capsules/6cy15cyw",  # 128
        # "hazy-research/capsules/3y6svmxn",  # 256
        "hazy-research/capsules/dt6vfe8q",  # 512
        # "hazy-research/capsules/tzhe958g",  # 1024
        "hazy-research/capsules/bcyg44th",  # 2048
        # "hazy-research/capsules/8auiop80", # 4096
        # "hazy-research/capsules/q1sebt5s",  # 8192
        
        
    ],
    hue=lambda row: f"b{row['num_trainable_tokens']:04d}",
)

cache_tuning_no_logits = PlotRunSetConfig(
    wandb_run_ids=[
        
        "hazy-research/capsules/3et4ezd5", # 512
        "hazy-research/capsules/n9tqxd2r", # 2048

        
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
    y_metric="macro_loss",
    dataset_type="eval",
    excluded_datasets=["mmlu"],
    y_label="log (Perplexity) (←)",
    x_label="Train Steps",
    hue_label="Method",
    bin_size=256,
    run_id="avg_plot",
    legend_loc="best",
)


if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
