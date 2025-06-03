
from capsules.analysis.figures.quality_v_compute import (
    PlotRunSetConfig,
    QualityVsComputePlotter,
)
import pydrantic

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/capsules/7gi2ngeh", # 2048
        "hazy-research/capsules/gs11ibhg", # 4096
        "hazy-research/capsules/u08vygew", # 8192
    ],
    hue=lambda row: f"Cache Tuning ({row['num_trainable_tokens']})",
    modify_x=lambda row, x: x + row["num_trainable_tokens"],
)


baseline_runs = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/mhr6jmg3"],
    hue="ICL Baseline",
    baseline=True,
)


run_sets = [
    cache_tuning,
    baseline_runs, 
]

avg_plot = QualityVsComputePlotter.Config(
    run_sets=run_sets,
    y_metric="avg_score",
    dataset_type="generate",
    excluded_datasets=[],
    y_label="Accuracy",
    x_label="Train Steps",
    hue_label="Method",
    bin_size=512,
    run_id="avg_plot",
    legend_loc="best",
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
