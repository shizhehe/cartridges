from capsules.analysis.figures.quality_v_cache_size_per_dataset import (
    QualityVsCacheSizePerDatasetPlotter,
)
from capsules.analysis.figures.quality_v_cache_size import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

cache_tuning_old = PlotRunSetConfig(
    launch_ids=[
        "2025-04-09-16-19-25-m04d09_train_housing_AL_new_eval"
    ],
    hue="Cache Tuning (Legacy)",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)

cache_tuning = PlotRunSetConfig(
    launch_ids=[
        "2025-04-10-17-50-01-m04d10_train_reglab_housing",

    ],
    hue="Cache Tuning",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)

cache_tuning_rel_only = PlotRunSetConfig(
    launch_ids=[
        "2025-04-11-00-03-02-m04d10_train_housing_AL_relevant_only_new_eval",
    ],
    hue="Cache Tuning (Relevant Statutes Only)",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)

cache_tuning_rag_old_prompt = PlotRunSetConfig(
    launch_ids=[
        "2025-04-10-22-11-36-m04d10_train_housing_AL_rag_new_prompt_new_eval",
    ],
    hue="Cache Tuning (RAG, old prompt)",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)

baseline_runs = PlotRunSetConfig(
    launch_ids=["2025-04-10-11-23-21-m04d10_icl_baseline_reglab_housing"],
    hue="ICL Baseline",
    baseline=True,
)

bm_25_runs = PlotRunSetConfig(
    launch_ids=[
        "2025-04-10-14-02-05-m04d10_rag_baseline_reglab_housing",
    ],
    hue=lambda row: "BM25 RAG (chunk size: "
    + str(row["eval_datasets.0.dataset.convo_transforms.0.retriever.max_tokens_per_chunk"])
    + ")",
    filter=lambda row: row[
        "eval_datasets.0.dataset.convo_transforms.0.retriever.max_tokens_per_chunk"
    ] == 256,
)

openai_runs = PlotRunSetConfig(
    launch_ids=[
        "2025-04-10-13-02-02-m04d10_rag_baseline_reglab_housing"
    ],
    hue=lambda row: "OpenAI RAG (chunk size: "
    + str(row["eval_datasets.0.dataset.convo_transforms.0.retriever.max_tokens_per_chunk"])
    + ")",
    filter=lambda row: row[
        "eval_datasets.0.dataset.convo_transforms.0.retriever.max_tokens_per_chunk"
    ] == 256,
)

run_sets = [
    cache_tuning,
    cache_tuning_old,
    cache_tuning_rel_only,
    cache_tuning_rag_old_prompt,
    baseline_runs, 
    bm_25_runs,
    openai_runs,
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="perplexity",
    excluded_datasets=[],
    show_baseline_points=True,
    y_label="Perplexity",
    x_label="Cache Size (tokens)",
    hue_label="Method",
    run_id="avg_plot",
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
