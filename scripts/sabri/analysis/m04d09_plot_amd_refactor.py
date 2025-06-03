from cartridges.analysis.figures.quality_v_cache_size_per_dataset import (
    QualityVsCacheSizePerDatasetPlotter,
)
from cartridges.analysis.figures.quality_v_cache_size import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic



cache_runs_old = PlotRunSetConfig(
    run_ids=[
        "53960c86-c599-46c5-98da-d3289996b4eb",
        "23b79c84-c5dd-43f0-b2ef-21afe4574b1f"
    ],
    hue="Cache Tuning (Old)",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)

cache_runs_refactor = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/boh1oite"
    ],
    hue="Cache Tuning (Refactor)",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)

cache_runs_refactor_rag = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/v4j7h4gy"
    ],
    hue="Cache Tuning (Refactor RAG)",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)


baseline_runs = PlotRunSetConfig(
    launch_ids=["2025-04-01-12-56-11-m03d30_amd_anthropic_icl_baseline"],
    hue="ICL Baseline",
    baseline=True,
)

rag_runs = PlotRunSetConfig(
    launch_ids=["2025-04-01-11-34-11-m03d31_eval_rag_baseline_amd"],
    hue=lambda row: "RAG (chunk size: "
    + str(row["eval_datasets.1.dataset.convo_transforms.0.max_tokens_per_chunk"])
    + ")",
    filter=lambda row: row[
        "eval_datasets.1.dataset.convo_transforms.0.max_tokens_per_chunk"
    ]
    > 128,
)

run_sets = [
    cache_runs_old, 
    cache_runs_refactor,
    cache_runs_refactor_rag,
    baseline_runs, 
    rag_runs
]


dataset_plots = QualityVsCacheSizePerDatasetPlotter.Config(
    run_sets=run_sets,
    y_metric="perplexity",
    excluded_datasets=[
        "eval__anthropic_qa_AMD_2022_10K_tag_memorization",
        "mmlu",
        "eval__anthropic_qa_AMD_2022_10K_tag_coding",
        # "finance-ppl-gt",
    ],
    show_baseline_points=True,
    y_label="Perplexity",
    x_label="Cache Size (tokens)",
    hue_label="Method",
    run_id="dataset_plots",
)

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="perplexity",
    excluded_datasets=[
        "eval__anthropic_qa_AMD_2022_10K_tag_memorization",
        "mmlu",
        "eval__anthropic_qa_AMD_2022_10K_tag_coding",
        "finance-ppl-gt",
        "eval__anthropic_qa_AMD_2022_10K_tag_creative",
    ],
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
