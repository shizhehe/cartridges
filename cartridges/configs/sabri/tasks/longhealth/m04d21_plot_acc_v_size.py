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
        # 8192
        # "hazy-research/capsules/pqxrilfq"
        "hazy-research/capsules/v581b3r4",  # 16 samples


        # 2048
        "hazy-research/capsules/mw9ub87j", # 16 samples

        "hazy-research/capsules/mx4vijke", # 16 samples

        "hazy-research/capsules/sw5zz3zl", # 16 samples
    ],
    hue="Cache Tuning",
    modify_x=lambda row, x: x + row["num_trainable_tokens"],
)


# cache_tuning = PlotRunSetConfig(
#     launch_ids=[
#         # "2025-04-10-17-50-01-m04d10_train_reglab_housing",
#         "2025-04-10-20-41-56-m04d10_train_reglab_housing"
#     ],
#     hue="Cache Tuning",
#     modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
# )

openai_runs = PlotRunSetConfig(
    launch_ids=[
        "2025-04-21-20-17-39-m04d16_genbaseline_longhealth_rag",
        "2025-04-21-20-25-42-m04d16_genbaseline_longhealth_rag"
    ],
    hue="RAG Baseline",
)    


baseline_runs = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/jhwglir5"],
    hue="ICL Baseline",
    baseline=True,
)


run_sets = [
    cache_tuning,
    # cache_tuning_old,
    baseline_runs, 
    # bm_25_runs,
    openai_runs,
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="avg_score",
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
