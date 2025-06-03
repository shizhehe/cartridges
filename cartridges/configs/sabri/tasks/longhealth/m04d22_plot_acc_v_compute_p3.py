
from capsules.analysis.figures.quality_v_compute import (
    PlotRunSetConfig,
    QualityVsComputePlotter,
)
import pydrantic

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/capsules/tfqjtak0",
        "hazy-research/capsules/nphif7bj",
        "hazy-research/capsules/9zprtqnv"
    ],
    hue=lambda row: f"Cache Tuning ({row['num_trainable_tokens']})",
    modify_x=lambda row, x: x + row["num_trainable_tokens"],
)


baseline_runs = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/jhwglir5"],
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
