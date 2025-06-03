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
        "hazy-research/capsules/7gi2ngeh",  # 2048
        "hazy-research/capsules/gs11ibhg",  # 4096
        "hazy-research/capsules/u08vygew",  # 8192
    ],
    hue="Cache Tuning",
    modify_x=lambda row, x: x + row["num_trainable_tokens"],
)




openai_runs = PlotRunSetConfig(
    launch_ids=[
        "2025-04-23-09-18-08-m04d23_genbaseline_longhealth_rag_px",
        "2025-04-23-10-24-57-m04d23_genbaseline_longhealth_rag_px",
        "2025-04-23-10-53-58-m04d23_genbaseline_longhealth_rag_px"
    ],
    hue="RAG Baseline",
)    


baseline_runs = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/mhr6jmg3"],
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
