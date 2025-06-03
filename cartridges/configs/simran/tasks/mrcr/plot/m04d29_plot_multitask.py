
from capsules.configs.simran.tasks.mrcr.plot.plot_multitask import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        # "hazy-research/capsules/7pebvlr2",
        # "hazy-research/capsules/dhe4ep3x",
        "hazy-research/capsules/k5zejd45"
    ],
    hue="Cache Tuning",
    modify_x=lambda row, x: x + row["num_trainable_tokens"],
)

baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/capsules/3ivp5xku",
        "hazy-research/capsules/64qw1jew",
        "hazy-research/capsules/yenkylfx"
    ],
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
        avg_plot
    ])


