from capsules.analysis.figures.quality_v_cache_size_per_dataset import (
    QualityVsCacheSizePerDatasetPlotter,
)
from capsules.analysis.figures.quality_v_cache_size import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/capsules/r3j8l53h",
        "hazy-research/capsules/arb9mwu8",
        "hazy-research/capsules/erdp4xa3"
    ],
    hue=lambda row: f"Cache Tuning ({row['num_trainable_tokens']})",
    modify_x=lambda row, x: x + row["num_trainable_tokens"],
)


baseline_runs = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/5v1ahjjz"],
    hue="ICL Baseline",
    baseline=True,
)


run_sets = [
    cache_tuning,
    baseline_runs, 
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    excluded_datasets=[],
    show_baseline_points=True,
    y_label="Accuracy",
    x_label="Cache Size (tokens)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
